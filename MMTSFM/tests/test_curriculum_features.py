"""Curriculum-feature tests: future-covariate influence + Grassmann LR warmup.

Run with: uv run pytest tests/test_curriculum_features.py -v
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from tests.test_vision_chronos2 import (
    _make_vision_model,
    _make_chronos2,
    _make_fake_video_encoder,
)


def _make_interleaved_model(d_model=64, d_v=4, n_vis=2, context_length=64):
    from mmtsfm.models.chronos2 import VisionChronos2Model, VisionChronos2Config

    chronos = _make_chronos2(d_model=d_model, context_length=context_length)
    vcfg = VisionChronos2Config(
        n_visual_context_steps=n_vis,
        n_soft_tokens=1,
        fusion_mode="interleaved",
        visual_dropout_prob=0.0,
        dropout=0.0,
    )
    return VisionChronos2Model(
        chronos_model=chronos,
        vision_config=vcfg,
        video_encoder=_make_fake_video_encoder(d_v=d_v),
    )


# ---------------------------------------------------------------------------
# 1. Known future covariates must actually influence the forecast.
#    They enter the encoder as covariate token-rows (token-type=covariate) fused
#    via GroupSelfAttention — NOT via the future_covariates loss arg (whose mask
#    only governs which horizon steps are scored). This asserts the token path
#    is live: perturbing one covariate channel changes the median forecast.
# ---------------------------------------------------------------------------
def test_future_covariates_influence_forecast():
    torch.manual_seed(0)
    vm = _make_vision_model(d_model=64)
    vm.eval()

    B, T_ctx_steps, H = 2, 32, 8
    context = torch.randn(B, T_ctx_steps)
    group_ids = torch.arange(B)
    # Two covariate channels, each [B, H] over the horizon.
    cov = [torch.randn(B, H) for _ in range(2)]

    def run(cov_channels):
        with torch.no_grad():
            out = vm.forward(
                context=context,
                group_ids=group_ids,
                num_output_patches=1,
                covariate_channels=cov_channels,
                video=None,
            )
        q = out.quantile_preds.float()
        return q[:, q.shape[1] // 2, :].clone()  # median [B, H_out]

    base = run(cov)
    perturbed = [cov[0] + 5.0, cov[1]]  # shift channel 0 hard
    changed = run(perturbed)

    delta = (base - changed).abs().max().item()
    # Untrained model with initializer_factor=0.05 → small but strictly non-zero
    # cross-row response. The regression this guards is the *structurally inert*
    # path (exactly 0.0 when covariate values were masked to zero pre-embedding).
    assert delta > 1e-6, (
        f"Future covariates do not influence the forecast (max Δmedian={delta:.2e}); "
        "the covariate token-row path is inert."
    )


def test_future_covariates_influence_interleaved():
    """Same audit for the interleaved fusion path (the final S2b/S3 model), which
    previously omitted covariate rows from the encoder input entirely."""
    torch.manual_seed(0)
    vm = _make_interleaved_model(d_model=64, n_vis=2, context_length=64)
    vm.eval()

    B, T_ctx_steps, H = 2, 64, 8
    context = torch.randn(B, T_ctx_steps)
    group_ids = torch.arange(B)
    video = torch.randn(B, 3, 8, 16, 16)
    cov = [torch.randn(B, H) for _ in range(2)]

    def run(cov_channels):
        with torch.no_grad():
            out = vm.forward(
                context=context,
                group_ids=group_ids,
                num_output_patches=1,
                covariate_channels=cov_channels,
                video=video,
            )
        q = out.quantile_preds.float()
        return q[:, q.shape[1] // 2, :].clone()

    base = run(cov)
    changed = run([cov[0] + 5.0, cov[1]])
    delta = (base - changed).abs().max().item()
    assert delta > 1e-6, (
        f"Covariates inert in interleaved mode (max Δmedian={delta:.2e}); "
        "the interleaved encoder input is dropping covariate rows."
    )


# ---------------------------------------------------------------------------
# 2. Grassmann LR warmup: with grassmann_warmup_steps>0 the Grassmann param
#    group's LR multiplier ramps from ~0 at step 0 up to full, while the main
#    groups follow the standard (shorter) warmup. Guards the Stage-1/2b warmup.
# ---------------------------------------------------------------------------
def test_grassmann_warmup_scales_lr():
    from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule

    core_cfg = {
        "d_model": 64,
        "d_kv": 16,
        "d_ff": 128,
        "num_layers": 2,
        "num_heads": 4,
        "dropout_rate": 0.0,
        "use_grassmann": True,
        "chronos_config": {
            "context_length": 64,
            "input_patch_size": 8,
            "input_patch_stride": 8,
            "output_patch_size": 8,
            "quantiles": [0.1, 0.5, 0.9],
            "use_reg_token": False,
            "use_arcsinh": True,
            "max_output_patches": 4,
        },
    }
    vision_cfg = {"skip_vision_stack": True, "fusion_mode": "late"}
    g_warmup = 100
    module = VisionChronos2LightningModule(
        chronos_core_cfg=core_cfg,
        vision_cfg=vision_cfg,
        pretrained_model_name_or_path=None,  # fresh weights → no download
        warmup_steps=10,
        grassmann_warmup_steps=g_warmup,
    )

    # Stub the trainer so _total_steps resolves without a real fit.
    module._trainer = types.SimpleNamespace(
        estimated_stepping_batches=1000,
        max_epochs=1,
        accumulate_grad_batches=1,
        train_dataloader=None,
    )

    cfg = module.configure_optimizers()
    scheduler = cfg["lr_scheduler"]["scheduler"]
    optimizer = cfg["optimizer"]

    g_idx = [
        i
        for i, g in enumerate(optimizer.param_groups)
        if "grassmann" in g.get("name", "")
    ]
    assert g_idx, "no grassmann param group found — Grassmann params not separated"

    # At step 0 the grassmann group's lambda multiplier must be < the main
    # warmup multiplier (ramping over the longer g_warmup horizon).
    lambdas = scheduler.lr_lambdas
    for i in g_idx:
        assert lambdas[i](1) < lambdas[i](g_warmup), "grassmann LR does not ramp up"
        assert lambdas[i](1) <= 1.0 / g_warmup + 1e-6, (
            "grassmann LR not suppressed at step 1"
        )
