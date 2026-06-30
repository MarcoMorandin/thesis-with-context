"""VisualEncoder: V-JEPA 2.1 spatiotemporal encoder wrapper.

Replaces the V-JEPA video encoder. Output shape [B, T_lat, P, D_v] is compatible
with LatentSummarizer KV input.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.hub
import torch.utils.model_zoo


_ARCH_D_V = {
    "vit_large": 1024,
    "vit_base": 768,
}
_ARCH_PATCH_SIZE = {
    "vit_large": 16,
    "vit_base": 16,
}
_ARCH_TEMPORAL_STRIDE = {
    "vit_large": 2,
    "vit_base": 2,
}
# torch.hub entry points from facebookresearch/vjepa2 hubconf.py
_HUB_MODEL_NAMES = {
    "vit_large": "vjepa2_1_vit_large_384",
    "vit_base":  "vjepa2_1_vit_base_384",
}


class VisualEncoder(nn.Module):
    """Wraps V-JEPA 2.1 spatiotemporal encoder.

    Loaded via torch.hub from facebookresearch/vjepa2. Weights are cached
    under TORCH_HOME and reused offline on compute nodes.

    Args:
        arch: Architecture variant: "vit_large" (D_v=1024) or "vit_base" (D_v=768).
        freeze: If True, freezes all encoder parameters after loading.
    """

    def __init__(
        self,
        arch: str = "vit_large",
        freeze: bool = True,
    ):
        super().__init__()
        assert arch in _ARCH_D_V, f"arch must be one of {list(_ARCH_D_V)}, got {arch!r}"
        self.arch = arch
        self.freeze = freeze
        self._d_v = _ARCH_D_V[arch]
        self._patch_size = _ARCH_PATCH_SIZE[arch]
        self._temporal_stride = _ARCH_TEMPORAL_STRIDE[arch]
        self._encoder: Optional[nn.Module] = None
        self._load()

    def _load(self) -> None:
        import os, sys, types
        hub_name = _HUB_MODEL_NAMES[self.arch]
        hub_dir = torch.hub.get_dir()
        repo_dir = os.path.join(hub_dir, "facebookresearch_vjepa2_main")

        # Download hub repo if not already cached. The first call will fail because
        # the installed vjepa2 wheel puts a partial src/ in site-packages (missing
        # src/utils/tensors), causing Python to resolve src.utils.tensors from the
        # wheel instead of the hub repo. We allow the failure and fix it below.
        if not os.path.isdir(repo_dir):
            try:
                torch.hub.load("facebookresearch/vjepa2", hub_name, trust_repo=True)
            except Exception:
                pass
        if not os.path.isdir(repo_dir):
            raise RuntimeError(
                f"vjepa2 hub repo not found at {repo_dir}. "
                "Run scripts/precache_login.sh from a login node first."
            )
        # Apply in-memory patch to redirect localhost weights URL to the public CDN.
        # This prevents concurrency write races on shared file systems.
        orig_hub_load = torch.hub.load_state_dict_from_url
        def patched_load(url, *args, **kwargs):
            if "localhost:8300" in url:
                url = url.replace("http://localhost:8300", "https://dl.fbaipublicfiles.com/vjepa2")
            return orig_hub_load(url, *args, **kwargs)
        torch.hub.load_state_dict_from_url = patched_load
        torch.utils.model_zoo.load_url = patched_load

        # Pin sys.modules['src'] to the hub repo's src/ directory so that
        # src.utils.tensors (and all other src.* imports inside the hub) resolve
        # exclusively from the hub repo, not from the installed wheel's partial src/.
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        for key in list(sys.modules.keys()):
            if key in ("src", "app") or key.startswith(("src.", "app.")):
                del sys.modules[key]
        src_stub = types.ModuleType("src")
        src_stub.__path__ = [os.path.join(repo_dir, "src")]  # type: ignore[attr-defined]
        src_stub.__package__ = "src"
        sys.modules["src"] = src_stub

        result = torch.hub.load(repo_dir, hub_name, source="local", trust_repo=True)
        # vjepa2_1 hub entry points return (encoder, predictor); take encoder only.
        encoder = result[0] if isinstance(result, (tuple, list)) else result
        if self.freeze:
            for p in encoder.parameters():
                p.requires_grad_(False)
        self._encoder = encoder

    def partial_unfreeze(self, n_last_layers: int) -> None:
        """Unfreeze last n transformer layers for domain adaptation (Stage 2a)."""
        if self._encoder is None:
            return
        all_blocks = list(self._encoder.blocks)  # type: ignore[attr-defined]
        for block in all_blocks[-n_last_layers:]:
            for p in block.parameters():
                p.requires_grad_(True)

    def set_freeze(self, freeze: bool) -> None:
        """Toggle full freeze/unfreeze (Stage 3)."""
        if self._encoder is None:
            return
        for p in self._encoder.parameters():
            p.requires_grad_(not freeze)

    @property
    def d_v(self) -> int:
        return self._d_v

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video frames to spatial-temporal patch tokens.

        Args:
            video: ``[B, C, T_v, H, W]`` — V-JEPA convention, normalized to [0, 1].
        Returns:
            ``[B, T_lat, P, D_v]`` where T_lat = T_v // temporal_stride,
            P = (H // patch_size) * (W // patch_size).
        """
        if self._encoder is None:
            raise RuntimeError("VisualEncoder not loaded.")
        B, C, T_v, H, W = video.shape
        tokens = self._encoder(video)  # [B, T_lat * P, D_v]
        T_lat = T_v // self._temporal_stride
        P = (H // self._patch_size) * (W // self._patch_size)
        return tokens.reshape(B, T_lat, P, self._d_v)
