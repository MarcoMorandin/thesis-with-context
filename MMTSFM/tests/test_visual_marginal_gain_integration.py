"""W6 — visual marginal gain: forced vision-off + dual-pass test_step.

CPU-only, tiny dims, fake V-JEPA video encoder (no real weights / network).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _make_chronos2(d_model: int = 32, context_length: int = 32):
    from mmtsfm.models.chronos2 import Chronos2Model, Chronos2CoreConfig

    cfg = Chronos2CoreConfig(
        d_model=d_model,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        use_grassmann=False,
        chronos_config={
            "context_length": context_length,
            "output_patch_size": 8,
            "input_patch_size": 8,
            "input_patch_stride": 8,
            "quantiles": [0.1, 0.5, 0.9],
            "use_reg_token": False,
            "use_arcsinh": False,
            "max_output_patches": 4,
        },
    )
    return Chronos2Model(cfg)


def _make_fake_video_encoder(
    d_v: int = 4, t_lat: int = 5, h_lat: int = 8, w_lat: int = 8
):
    class FakeVideoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_v = d_v
            self.proj = nn.Linear(d_v, d_v)

        def forward(self, video: torch.Tensor) -> torch.Tensor:
            B = video.shape[0]
            z = torch.randn(
                B, t_lat, h_lat * w_lat, d_v, device=video.device, dtype=video.dtype
            )
            return self.proj(z)

    return FakeVideoEncoder()


def _make_vision_model(d_model: int = 32, d_v: int = 4):
    from mmtsfm.models.chronos2 import VisionChronos2Model, VisionChronos2Config

    chronos = _make_chronos2(d_model=d_model)
    vcfg = VisionChronos2Config(
        n_visual_context_steps=4,
        n_soft_tokens=1,
        adapter_type="linear",
        visual_dropout_prob=0.0,
        dropout=0.0,
    )
    return VisionChronos2Model(
        chronos_model=chronos,
        vision_config=vcfg,
        video_encoder=_make_fake_video_encoder(d_v=d_v),
    )


def test_force_vision_off_matches_pure_ts():
    """force_vision_off=True must equal a numeric-only (video=None) forward."""
    torch.manual_seed(0)
    vm = _make_vision_model()
    vm.eval()

    B, L = 2, 32
    context = torch.randn(B, L)
    group_ids = torch.arange(B)
    video = torch.rand(B, 3, 17, 64, 64)

    with torch.no_grad():
        out_off = vm.forward(
            context=context,
            group_ids=group_ids,
            num_output_patches=1,
            video=video,
            force_vision_off=True,
        )
        out_ts = vm.forward(
            context=context,
            group_ids=group_ids,
            num_output_patches=1,
            video=None,
        )

    assert out_off.visual_active is not None
    assert not out_off.visual_active.any(), "force_vision_off must mark visual inactive"
    torch.testing.assert_close(
        out_off.quantile_preds, out_ts.quantile_preds, atol=1e-5, rtol=1e-5
    )


def test_lightning_test_step_runs_both_passes():
    """test_step runs vision-on + vision-off passes and finalize reports Δ."""
    from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule

    model_mock = MagicMock()
    out_on = MagicMock()
    out_on.quantile_preds = torch.zeros(2, 9, 4)
    out_on.loss = torch.tensor(0.5)
    out_off = MagicMock()
    out_off.quantile_preds = torch.full((2, 9, 4), 0.2)
    out_off.loss = torch.tensor(0.6)

    def mock_forward(**kwargs):
        return out_off if kwargs.get("force_vision_off", False) else out_on

    model_mock.forward.side_effect = mock_forward
    model_mock.chronos.num_quantiles = 9

    hparams = {
        "horizon": 4,
        "sp_reference_path": None,
        "compute_marginal_gain": True,
        "results_dir": "results",
        "results_tag": "test_run",
    }

    class StubModule(VisionChronos2LightningModule):
        def __init__(self, model):
            super(VisionChronos2LightningModule, self).__init__()
            self.model = model
            self.save_hyperparameters(hparams)
            self._protocol_eval = None

        @property
        def device(self):
            class _D:
                type = "cpu"

            return _D()

        def _unpack_batch(self, batch):
            return {
                "context": batch["context"],
                "future_target": batch["future_target"],
                "future_target_mask": batch["future_target_mask"],
            }

        def log(self, *args, **kwargs):
            pass

    lm = StubModule(model_mock)
    lm.on_test_start()
    assert lm._protocol_eval.compute_marginal_gain is True

    batch = {
        "context": torch.zeros(2, 32),
        "future_target": torch.zeros(2, 4),
        "future_target_mask": torch.ones(2, 4),
        "daylight_future": torch.ones(2, 4),
        "site_id": ["site1", "site2"],
    }

    lm.test_step(batch, 0)
    assert model_mock.forward.call_count == 2, "expected vision-on + vision-off passes"

    res = lm._protocol_eval.finalize()
    assert "delta_nmae" in res["overall"]
    assert "delta_nrmse" in res["overall"]
    assert abs(res["overall"]["delta_nmae"] - 0.2) < 1e-6
