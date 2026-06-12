"""VidTok video tokenizer wrapper for Vision-Time FM.

Wraps Microsoft VidTok (https://github.com/microsoft/VidTok) as a frozen
spatial-temporal video encoder.

Setup
-----
Clone VidTok and add to PYTHONPATH before importing::

    git clone https://github.com/microsoft/VidTok /opt/vidtok
    export PYTHONPATH="/opt/vidtok:$PYTHONPATH"

Or pass ``vidtok_root`` to ``VidTokEncoder`` to set sys.path at runtime.

Input  : V  ∈ ℝ^(B, C, T_v, H, W)   values in [0, 1]  (dataset canonical)
Output : z  ∈ ℝ^(B, T_lat, P, D_v)   continuous latent tokens
           where P = H_lat × W_lat, D_v = latent channels
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class VidTokEncoder(nn.Module):
    """Frozen VidTok encoder that produces spatial-temporal latent tokens.

    Parameters
    ----------
    cfg_path:
        Path to VidTok YAML config (e.g. ``configs/vidtok_kl_causal_488_4chn.yaml``).
    ckpt_path:
        Path to VidTok checkpoint (``.ckpt``).
    vidtok_root:
        Optional path to the VidTok repository root. Added to ``sys.path``
        so ``scripts.inference_evaluate`` can be imported.
    is_causal:
        If True the model is causal (requires T_in == 4k+1 frames, e.g. 17).
        Set False for non-causal variants (T_in == 4k frames, e.g. 16).
    model:
        Pre-instantiated VidTok model. When provided, ``cfg_path``,
        ``ckpt_path``, and ``vidtok_root`` are ignored. Useful for testing.
    """

    def __init__(
        self,
        cfg_path: str = "",
        ckpt_path: str = "",
        vidtok_root: Optional[str] = None,
        is_causal: bool = True,
        model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.is_causal = is_causal

        if model is not None:
            self._model = model
        else:
            if not cfg_path or not ckpt_path:
                raise ValueError(
                    "Provide cfg_path + ckpt_path, or pass a pre-built model= instance."
                )
            self._model = self._load_vidtok(cfg_path, ckpt_path, vidtok_root)

        # Freeze all VidTok parameters — only adapter/summarizer train
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()

        # Infer latent shape from a dummy forward pass
        self._d_v: Optional[int] = None
        self._h_lat: Optional[int] = None
        self._w_lat: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_vidtok(cfg_path: str, ckpt_path: str, vidtok_root: Optional[str]):
        if vidtok_root is not None:
            root = str(Path(vidtok_root).expanduser().resolve())
            if root not in sys.path:
                sys.path.insert(0, root)

        try:
            from scripts.inference_evaluate import load_model_from_config  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Cannot import VidTok. Clone https://github.com/microsoft/VidTok "
                "and either add it to PYTHONPATH or pass vidtok_root= to VidTokEncoder."
            ) from exc

        # Patch VidTok's LPIPS loader to use pre-cached weights from vidtok_root.
        # VidTok calls get_ckpt_path(name, "checkpoints/lpips") with a CWD-relative
        # root, which fails on offline compute nodes. We redirect to the absolute
        # path inside vidtok_root where login_node_setup.sh pre-downloads the file.
        if vidtok_root is not None:
            VidTokEncoder._patch_lpips_cache(root)

        model = load_model_from_config(cfg_path, ckpt_path)
        return model

    @staticmethod
    def _patch_lpips_cache(vidtok_root: str) -> None:
        import importlib
        try:
            lpips_mod = importlib.import_module("vidtok.modules.lpips")
        except ModuleNotFoundError:
            return

        lpips_dir = str(Path(vidtok_root) / "checkpoints" / "lpips")
        _orig = lpips_mod.get_ckpt_path

        def _patched(name: str, root: str, check: bool = False) -> str:
            cached = str(Path(lpips_dir) / (name + ".ckpt"))
            if Path(cached).exists():
                return cached
            return _orig(name, root, check)

        lpips_mod.get_ckpt_path = _patched

    def _probe_latent_shape(
        self, device: torch.device, dtype: torch.dtype, h_in: int = 64, w_in: int = 64
    ) -> None:
        """Run one dummy encode to cache latent spatial dims.

        Parameters
        ----------
        h_in, w_in:
            Spatial size of the probe input. Should match the real inference size so
            that ``spatial_patches`` returns the correct value. Default 64 matches
            the dataset configs (img_size=64). The old hard-coded 256 was wrong.
        """
        if self._d_v is not None:
            return
        n_frames = 17 if self.is_causal else 16
        dummy = torch.zeros(1, 3, n_frames, h_in, w_in, device=device, dtype=dtype)
        with torch.no_grad():
            z = self._encode_raw(dummy)
        _, d_v, _, h_lat, w_lat = z.shape
        self._d_v = d_v
        self._h_lat = h_lat
        self._w_lat = w_lat

    def _encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Call VidTok encode; returns raw latent [B, D_v, T_lat, H_lat, W_lat]."""
        self._model.eval()
        # VidTok's deep conv stack overflows bfloat16 with out-of-distribution
        # (e.g. synthetic random) inputs. Force float32 for the encode pass.
        # Parameters are frozen, so converting back once is cheap and permanent.
        first_param = next(self._model.parameters(), None)
        if first_param is not None and first_param.dtype != torch.float32:
            self._model.float()
        device_type = "cpu" if x.device.type == "mps" else x.device.type
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            z, _ = self._model.encode(x.float(), return_reg_log=True)
        return z

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def d_v(self) -> int:
        if self._d_v is None:
            raise RuntimeError("Call probe_latent_shape() or run a forward pass first.")
        return self._d_v

    @property
    def spatial_patches(self) -> int:
        if self._h_lat is None or self._w_lat is None:
            raise RuntimeError("Call probe_latent_shape() or run a forward pass first.")
        return self._h_lat * self._w_lat

    def probe_latent_shape(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        img_size: int = 64,
    ) -> None:
        """Manually trigger latent shape inference on a dummy input.

        Parameters
        ----------
        img_size:
            Spatial edge length of the probe input. Should match the real
            dataset image size so ``spatial_patches`` is accurate.
            Default 64 matches all current dataset configs.
        """
        self._probe_latent_shape(device, dtype, h_in=img_size, w_in=img_size)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a batch of video clips.

        Parameters
        ----------
        video:
            ``[B, C, T_v, H, W]`` in [0, 1] (dataset canonical schema).
            Rescaled to [-1, 1] internally before passing to VidTok.

        Returns
        -------
        z : ``[B, T_lat, P, D_v]``
            Continuous latent tokens. ``P = H_lat * W_lat``.
        """
        # Rescale [0, 1] → [-1, 1]
        x = video * 2.0 - 1.0

        # Encode: raw z is [B, D_v, T_lat, H_lat, W_lat]
        z = self._encode_raw(x)

        B, D_v, T_lat, H_lat, W_lat = z.shape
        # Always update cached dims from live output — ensures spatial_patches is accurate
        # regardless of what probe_latent_shape was called with.
        self._d_v   = D_v
        self._h_lat = H_lat
        self._w_lat = W_lat

        # Guard against any residual NaN/Inf/outlier latents from VidTok.
        # Synthetic random frames are far outside VidTok's training distribution;
        # unbounded latents can overflow the trainable adapter/summarizer stack.
        z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
        z = z.clamp_(-10.0, 10.0)

        # Reshape → [B, T_lat, P, D_v]  (P = H_lat * W_lat)
        z = z.permute(0, 2, 3, 4, 1)   # [B, T_lat, H_lat, W_lat, D_v]
        z = z.reshape(B, T_lat, H_lat * W_lat, D_v)

        return z
