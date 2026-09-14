"""A45a — latent-depth control (``visual_latent_keep_newest``).

The cache key is ``{dataset}_{site}_{origin}`` and does not encode the ladder,
so a shallower arm can reuse a deeper cache. These tests pin the two things
that makes safe: the slice takes the NEWEST latents (not the oldest), and the
configured frame ladder is forced to agree with the sliced stack.
"""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


class _Slicer:
    """The slice as it runs in PVRecordDataset.__getitem__, isolated.

    PVRecordDataset.__init__ reads the uk_pv parquet + H5, so the loader itself
    is not constructible in a unit test. The arithmetic is what needs pinning.
    """

    def __init__(self, video_frames: int, keep: int | None):
        self.T_v = video_frames
        self.visual_latent_keep_newest = keep
        if keep is not None:
            if keep <= 0:
                raise ValueError("visual_latent_keep_newest must be positive")
            if self.T_v != 2 * keep:
                raise ValueError(
                    f"visual_latent_keep_newest={keep} needs video_frames="
                    f"{2 * keep} (V-JEPA pools 2 frames per latent), got "
                    f"video_frames={self.T_v}"
                )

    def slice(self, Z: torch.Tensor) -> torch.Tensor:
        if self.visual_latent_keep_newest is None:
            return Z
        k = self.visual_latent_keep_newest
        cached = Z.shape[1]
        if cached < k:
            raise ValueError(
                f"visual_latent_keep_newest={k} but the cache holds only "
                f"{cached} latents"
            )
        return Z[:, -k:]


def _cache(T_lat: int, N: int = 1, P: int = 49, D: int = 8) -> torch.Tensor:
    """[N, T_lat, P, D] whose value encodes the latent index, oldest = 0."""
    Z = torch.zeros(N, T_lat, P, D)
    for t in range(T_lat):
        Z[:, t] = float(t)
    return Z


class TestLatentSliceGeometry:
    def test_none_is_a_passthrough(self):
        """Default must leave every pre-A45 run bit-identical."""
        Z = _cache(4)
        assert torch.equal(_Slicer(8, None).slice(Z), Z)

    def test_keeps_the_newest_not_the_oldest(self):
        """An s2e burst spans the NEWEST 2.25 h; taking [:2] would invert it."""
        out = _Slicer(4, 2).slice(_cache(4))
        assert out.shape[1] == 2
        # latents 2 and 3 of 0..3 — the two most recent.
        assert out[0, 0, 0, 0].item() == 2.0
        assert out[0, 1, 0, 0].item() == 3.0

    def test_a45a_reproduces_one_s2e_anchor(self):
        """s2d cache (T_lat=4) trimmed to one s2e burst's depth (T_lat=2)."""
        out = _Slicer(4, 2).slice(_cache(4))
        assert out.shape[1] == 2, "one s2e anchor carries T_lat=2"
        # 2 latents * 49 cells = 98 EVS candidates, against s2d's 196.
        assert out.shape[1] * out.shape[2] == 98

    def test_frame_ladder_must_match_the_slice(self):
        """video_frames != 2k would mislabel video_delta_t silently."""
        with pytest.raises(ValueError, match="needs video_frames=4"):
            _Slicer(8, 2)

    def test_rejects_non_positive_keep(self):
        with pytest.raises(ValueError, match="must be positive"):
            _Slicer(0, 0)

    def test_rejects_a_cache_shallower_than_the_request(self):
        """Asking a 2-latent cache for 4 must raise, not silently under-fill."""
        with pytest.raises(ValueError, match="holds only 2 latents"):
            _Slicer(8, 4).slice(_cache(2))


class TestA45aConfig:
    """The cross-file invariant video_frames == 2 * keep has no runtime
    reconciler between the yaml and the loader — pin it at the config level."""

    @staticmethod
    def _cfg():
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(config_name="config", overrides=["+ablation=A45a"])

    def test_frames_match_the_latent_keep(self):
        cfg = self._cfg()
        assert cfg.data.video_frames == 2 * cfg.data.visual_latent_keep_newest

    def test_single_anchor(self):
        """A45a varies depth+density only; the anchor count stays at s2d's 1."""
        cfg = self._cfg()
        assert cfg.model.vision_cfg.get("n_visual_context_steps", 1) == 1

    def test_keep_matches_one_s2e_anchor(self):
        cfg = self._cfg()
        assert cfg.model.vision_cfg.visual_evs_keep == 20

    def test_marginal_gain_is_on(self):
        """Without the vision-OFF pass the split is unattributable."""
        assert self._cfg().model.compute_marginal_gain is True
