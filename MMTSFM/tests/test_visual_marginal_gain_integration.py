import pytest
import torch
import torch.nn as nn
import numpy as np

from mmtsfm.models.chronos2 import VisionChronos2Model, VisionChronos2Config
from mmtsfm.models.chronos2.model import Chronos2Model, Chronos2CoreConfig
from eval.protocol_eval import ProtocolEvaluator
from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule

def _make_chronos2(d_model=32, context_length=16):
    cfg = Chronos2CoreConfig(
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        dropout_rate=0.0,
        use_grassmann=False,
        chronos_config={
            "context_length": context_length,
            "output_patch_size": 8,
            "input_patch_size": 8,
            "input_patch_stride": 8,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "use_reg_token": False,
            "use_arcsinh": False,
            "max_output_patches": 4,
        },
    )
    return Chronos2Model(cfg)

def _make_fake_vidtok(d_v: int = 4, t_lat: int = 5, h_lat: int = 8, w_lat: int = 8):
    """Fake VidTok model for tests (no real checkpoint needed)."""
    class FakeVidTok(nn.Module):
        def __init__(self):
            super().__init__()
            self._d_v = d_v
            self._t_lat = t_lat
            self._h_lat = h_lat
            self._w_lat = w_lat

        def encode(self, x: torch.Tensor, return_reg_log: bool = False):
            B = x.shape[0]
            z = torch.randn(B, self._d_v, self._t_lat, self._h_lat, self._w_lat,
                            device=x.device, dtype=x.dtype)
            return z, {}

        def forward(self, x):
            z, _ = self.encode(x)
            return x, x, {}
    return FakeVidTok()

def test_forced_vision_off_matches_pure_ts():
    d_model = 32
    d_v = 16
    context_length = 16
    n_vis_steps = 4
    n_soft_tokens = 4

    chronos = _make_chronos2(d_model=d_model, context_length=context_length)
    vcfg = VisionChronos2Config(
        d_video_latent=d_v,
        n_visual_context_steps=n_vis_steps,
        n_soft_tokens=n_soft_tokens,
        adapter_type="linear",
        visual_dropout_prob=0.0,
        dropout=0.0,
    )
    model = VisionChronos2Model(
        chronos_model=chronos,
        vision_config=vcfg,
        vidtok_model=_make_fake_vidtok(d_v=d_v),
    )

    model.eval()

    B = 2
    context = torch.randn(B, context_length)
    context_mask = torch.ones(B, context_length)
    # Synthetic pre-computed latents: [B, T_lat, P, D_v]
    video_latents = torch.randn(B, n_vis_steps, 4, d_v)

    # 1. Run with force_vision_off=True
    with torch.no_grad():
        out_forced = model(
            context=context,
            context_mask=context_mask,
            video_latents=video_latents,
            force_vision_off=True,
        )

    # 2. Run pure TS forward (no video, no latents)
    with torch.no_grad():
        out_pure_ts = model(
            context=context,
            context_mask=context_mask,
            video_latents=None,
            video=None,
        )

    # They should be mathematically identical
    np.testing.assert_allclose(
        out_forced.quantile_preds.numpy(),
        out_pure_ts.quantile_preds.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    assert out_forced.visual_active is not None
    assert not out_forced.visual_active.any()


def test_lightning_module_dual_pass():
    # Verify the test step performs both passes when compute_marginal_gain=True
    # and accumulates correct structures.
    from unittest.mock import MagicMock

    # Create dummy lightning module
    model_mock = MagicMock()
    # Mock model forward return (9 quantiles, 4 horizon)
    out_on = MagicMock()
    out_on.quantile_preds = torch.zeros(2, 9, 4)
    out_on.loss = torch.tensor(0.5)

    out_off = MagicMock()
    out_off.quantile_preds = torch.zeros(2, 9, 4)
    out_off.loss = torch.tensor(0.6)

    # Make self.model.forward return out_on or out_off depending on force_vision_off
    def mock_forward(**kwargs):
        if kwargs.get("force_vision_off", False):
            return out_off
        return out_on

    model_mock.forward.side_effect = mock_forward
    # Setup Chronos quantiles count to match 9
    model_mock.chronos.num_quantiles = 9

    # Instantiate lightning module
    hparams = {
        "horizon": 4,
        "sp_reference_path": None,
        "compute_marginal_gain": True,
        "results_dir": "results",
        "results_tag": "test_run",
    }
    
    # We subclass to stub unpack_batch and logging
    class StubModule(VisionChronos2LightningModule):
        def __init__(self, model):
            super(VisionChronos2LightningModule, self).__init__()
            self.model = model
            self.save_hyperparameters(hparams)
            self._protocol_eval = None
            self.device_type = "cpu"

        @property
        def device(self):
            class DummyDevice:
                type = "cpu"
            return DummyDevice()

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

    # Prepare dummy batch
    batch = {
        "context": torch.zeros(2, 16),
        "future_target": torch.zeros(2, 4),
        "future_target_mask": torch.ones(2, 4),
        "daylight_future": torch.ones(2, 4),
        "site_id": ["site1", "site2"],
        "Y": torch.zeros(2, 1, 16, 1),
        "Y_future": torch.zeros(2, 1, 4, 1),
        "X_cov": torch.zeros(2, 1, 20, 0),
        "V": torch.zeros(2, 1, 4, 3, 8, 8),
        "mask_target": torch.ones(2, 1, 16, 1),
        "mask_future": torch.ones(2, 1, 4, 1),
        "mask_visual": torch.ones(2, 1, 4),
    }

    lm.test_step(batch, 0)

    # finalize and verify delta is computed
    res = lm._protocol_eval.finalize()
    assert "delta_nmae" in res["overall"]
    assert "delta_nrmse" in res["overall"]
