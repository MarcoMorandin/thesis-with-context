"""Verify per-param-group grad-norm logging in VisionChronos2LightningModule."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule


@pytest.fixture
def lit_module(monkeypatch):
    # Build a tiny module without loading pretrained weights.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    chronos_cfg = dict(
        d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=2,
        dropout_rate=0.0, layer_norm_epsilon=1e-6, initializer_factor=0.05,
        feed_forward_proj="relu", rope_theta=10000.0,
        use_grassmann=False, grassmann_reduced_dim=4,
        grassmann_window_offsets=[1], grassmann_modality_pair_bias=False,
        attn_implementation="eager",
        chronos_config=dict(
            context_length=64, input_patch_size=8, input_patch_stride=8,
            output_patch_size=8, quantiles=[0.1, 0.5, 0.9],
            use_reg_token=False, use_arcsinh=True, max_output_patches=2,
        ),
    )
    vision_cfg = dict(
        visual_encoder_type="vjepa2",
        visual_encoder_ckpt_path="",
        freeze_visual_encoder=True,
        skip_vision_stack=True,  # fast — bypass V-JEPA stack
        fusion_mode="late",
        d_video_latent=32,
        n_visual_context_steps=4,
        n_soft_tokens=1,
        adapter_type="linear",
        adapter_n_layers=1,
        summarizer_n_heads=2,
        visual_dropout_prob=0.0,
        numeric_dropout_prob=0.0,
        dropout=0.0,
        n_entities=0,
    )
    module = VisionChronos2LightningModule(
        chronos_core_cfg=chronos_cfg,
        vision_cfg=vision_cfg,
        freeze_chronos=True,
        pretrained_model_name_or_path=None,
    )
    module.trainer = MagicMock()
    module.trainer.global_step = 0
    module._last_loss = torch.tensor(1.0)
    return module


def test_grad_norm_logged_per_group(lit_module):
    """on_before_optimizer_step must log a grad_norm per declared group."""
    # Seed gradients on every trainable parameter.
    for p in lit_module.model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)

    logs: dict[str, float] = {}
    lit_module.log = lambda name, value, **kw: logs.update({name: float(value)})

    lit_module.on_before_optimizer_step(optimizer=MagicMock())

    # Aggregate group still logged for backwards compatibility.
    assert "train/grad_norm" in logs
    # New per-group logs.
    assert "train/grad_norm/vision_adapter" in logs
    assert "train/grad_norm/latent_summarizer" in logs
    assert "train/grad_norm/output_patch_embedding" in logs
    assert "train/grad_norm/multimodal_embed" in logs
    # All groups finite under the seeded random gradients.
    for k, v in logs.items():
        assert torch.isfinite(torch.tensor(v)), f"{k} not finite: {v}"


def test_nan_grad_zeroes_param_grads(lit_module):
    """If any grad is NaN/Inf, on_before_optimizer_step zeroes all grads
    so AdamW receives a no-op step instead of polluting its moment buffers."""
    trainable = [p for p in lit_module.model.parameters() if p.requires_grad]
    assert trainable, "fixture must produce some trainable params"

    for p in trainable:
        p.grad = torch.zeros_like(p)
    trainable[0].grad = torch.full_like(trainable[0], float("nan"))

    lit_module.log = lambda *a, **kw: None
    lit_module.on_before_optimizer_step(optimizer=MagicMock())

    for p in trainable:
        assert torch.all(p.grad == 0), "NaN grad must trigger global zeroing"
