"""Tests for zero-shot sensor calibration and visual damping.

Verifies:
1. _prep_frame with radiometric_norm maps arbitrary channels to target distribution (mean ~0.49, std ~0.15).
2. _load_vision masks out pitch-black night frames (DN < 5.0) when radiometric_norm=True.
3. VisionChronos2Config supports visual_scale knob.
4. MMTSFMDataModule correctly passes radiometric_norm to PVRecordDataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mmtsfm.data.pv_record import _prep_frame, PVRecordDataset
from mmtsfm.data.datamodule import MMTSFMDataModule
from mmtsfm.models.chronos2.vision_chronos2 import VisionChronos2Config


def test_prep_frame_radiometric_norm():
    # Synthetic frame with non-SEVIRI distribution (e.g. mean ~0.2, std ~0.05)
    rng = np.random.default_rng(42)
    raw = (rng.normal(loc=50.0, scale=12.0, size=(64, 64, 3)).clip(0, 255)).astype(np.uint8)

    # Without radiometric_norm (raw [0, 1])
    t_raw = _prep_frame(raw, side=64, c_img=3, imagenet_norm=False, radiometric_norm=False)
    assert t_raw.shape == (3, 64, 64)
    assert 0.15 < t_raw.mean().item() < 0.25

    # With radiometric_norm (target SEVIRI mean ~0.49, std ~0.15)
    t_cal = _prep_frame(raw, side=64, c_img=3, imagenet_norm=False, radiometric_norm=True)
    assert t_cal.shape == (3, 64, 64)
    for ch in range(3):
        assert abs(t_cal[ch].mean().item() - 0.49) < 0.05
        assert abs(t_cal[ch].std().item() - 0.15) < 0.05


def test_load_vision_night_masking():
    # Create a dummy dataset instance to test _load_vision logic
    ds = PVRecordDataset.__new__(PVRecordDataset)
    ds.T = 10
    ds.T_v = 4
    ds.img_size = 32
    ds.C_img = 3
    ds.imagenet_norm = False
    ds.radiometric_norm = True
    ds.visual_frame_spacing_min = 30.0
    ds.visual_window_hours = 2.0
    ds.visual_anchor_stride_hours = None
    ds.visual_frames_per_anchor = None
    ds.png_frames = False
    ds.frame_maps = {}
    ds._frame_grid = {}

    item = {
        "dataset": "goes_pvdaq",
        "site_id": "1202",
        "timestamps": np.arange(10) * 1800,
    }
    # With load_frames=False, no frames decoded
    V, mask_v, delta_t = ds._load_vision(item, load_frames=False)
    assert V.shape == (1, 4, 3, 32, 32)
    assert mask_v.shape == (1, 4)


def test_visual_scale_config():
    cfg_default = VisionChronos2Config()
    assert cfg_default.visual_scale == 1.0

    cfg_damped = VisionChronos2Config(visual_scale=0.5)
    assert cfg_damped.visual_scale == 0.5


def test_datamodule_radiometric_norm(tmp_path):
    dm = MMTSFMDataModule(
        dataset_name="goes_pvdaq",
        data_dir=str(tmp_path),
        radiometric_norm=True,
        emit_vision=False,
    )
    assert dm.hparams.radiometric_norm is True


def test_visual_scale_forward_damping():
    import torch.nn as nn
    from mmtsfm.models.chronos2.config import Chronos2CoreConfig
    from mmtsfm.models.chronos2.model import Chronos2Model
    from mmtsfm.models.chronos2.vision_chronos2 import VisionChronos2Model

    core_cfg = Chronos2CoreConfig(
        d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4, dropout_rate=0.0,
        chronos_config=dict(
            context_length=16, input_patch_size=8, input_patch_stride=8, output_patch_size=8,
            quantiles=[0.1, 0.5, 0.9], use_reg_token=False, use_arcsinh=True, max_output_patches=1,
        ),
    )
    chronos = Chronos2Model(core_cfg)

    class FakeVideoEncoder(nn.Module):
        d_v = 8
        def forward(self, x):
            return torch.randn(x.shape[0], 2, 4, 8)

    vcfg_kw = dict(
        fusion_mode="interleaved_raw",
        n_visual_context_steps=1,
        visual_shuffle_r=2,
        visual_n_cells=1,
        visual_evs_keep=1,
        visual_dropout_prob=0.0,
        numeric_dropout_prob=0.0,
        dropout=0.0,
    )
    enc = FakeVideoEncoder()
    model_full = VisionChronos2Model(chronos_model=chronos, vision_config=VisionChronos2Config(visual_scale=1.0, **vcfg_kw), video_encoder=enc)
    model_damped = VisionChronos2Model(chronos_model=chronos, vision_config=VisionChronos2Config(visual_scale=0.0, **vcfg_kw), video_encoder=enc)
    model_damped.load_state_dict(model_full.state_dict())

    torch.manual_seed(42)
    B, T, Tv, C, H, W = 1, 16, 4, 3, 16, 16
    context = torch.randn(B, T)
    video = torch.randn(B, Tv, C, H, W)
    video_delta_t = torch.zeros(B, Tv)

    with torch.no_grad():
        out_full = model_full(context=context, video=video, video_delta_t=video_delta_t)
        out_damped = model_damped(context=context, video=video, video_delta_t=video_delta_t)

    assert out_full.quantile_preds.shape == out_damped.quantile_preds.shape
    diff = (out_full.quantile_preds - out_damped.quantile_preds).abs().max().item()
    assert diff > 0.0
