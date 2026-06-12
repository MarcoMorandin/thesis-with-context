"""Tests for Task 3.5 — Training Loop.

Roadmap done-when criteria:
  1. Training run on smoke dataset completes without error
  2. Gradient norms are logged (verified by checking logged metrics)

Run: uv run pytest tests/test_training_loop.py -v
"""
from __future__ import annotations

import math
import pytest
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHRONOS_CFG = dict(
    d_model=64,
    d_kv=16,
    d_ff=128,
    num_layers=2,
    num_heads=4,
    dropout_rate=0.0,
    use_grassmann=False,
    chronos_config=dict(
        context_length=32,
        input_patch_size=8,
        input_patch_stride=8,
        output_patch_size=8,
        quantiles=[0.1, 0.5, 0.9],
        use_reg_token=False,
        use_arcsinh=False,
        max_output_patches=2,
    ),
)

_VISION_CFG = dict(
    d_video_latent=4,
    n_visual_context_steps=4,
    n_soft_tokens=1,
    adapter_type="linear",
    visual_dropout_prob=0.0,
    numeric_dropout_prob=0.0,
    dropout=0.0,
)

HORIZON = 8


def _make_module(**overrides):
    from src.mmtsfm.models.chronos2 import VisionChronos2LightningModule
    from tests.test_vision_chronos2 import _make_fake_vidtok

    cfg = dict(
        chronos_core_cfg=_CHRONOS_CFG,
        vision_cfg=_VISION_CFG,
        lr=1e-3,
        weight_decay=0.0,
        warmup_steps=2,
        min_lr_ratio=0.1,
        horizon=HORIZON,
        freeze_chronos=False,
        vidtok_model=_make_fake_vidtok(d_v=4, t_lat=2, h_lat=4, w_lat=4),
    )
    cfg.update(overrides)

    return VisionChronos2LightningModule(**cfg)


def _make_batch(bs=2, N=2, T=32, H=8, T_v=4, C=3, img=16):
    return {
        "Y":                     torch.randn(bs, N, T, 1),
        "Y_future":              torch.randn(bs, N, H, 1),
        "X_cov":                 torch.randn(bs, N, T + H, 3),
        "V":                     torch.rand(bs, N, T_v, C, img, img),
        "mask_target":           torch.ones(bs, N, T, 1),
        "mask_future":           torch.ones(bs, N, H, 1),
        "mask_visual":           torch.ones(bs, N, T_v),
        "mask_modality_dropout": torch.ones(bs, N, 2),
        "entity_ids":            torch.zeros(bs, N, dtype=torch.long),
        "timestamps":            torch.arange(T + H).unsqueeze(0).expand(bs, -1),
        "timestamps_v":          torch.arange(T_v).unsqueeze(0).expand(bs, -1),
        "adj_matrix":            torch.eye(N).unsqueeze(0).expand(bs, -1, -1),
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestBatchUnpacking:
    def test_context_shape(self):
        module = _make_module()
        batch = _make_batch(bs=2, N=3, T=32)
        inputs = module._unpack_batch(batch)
        assert inputs["context"].shape == (6, 32)   # BS*N

    def test_future_target_shape(self):
        module = _make_module()
        batch = _make_batch(bs=2, N=3, H=8)
        inputs = module._unpack_batch(batch)
        assert inputs["future_target"].shape == (6, 8)

    def test_video_shape(self):
        module = _make_module()
        batch = _make_batch(bs=2, N=2, T_v=4, C=3, img=16)
        inputs = module._unpack_batch(batch)
        assert inputs["video"].shape == (4, 3, 4, 16, 16)  # [BS*N, C, T_v, H, W]

    def test_group_ids(self):
        module = _make_module()
        batch = _make_batch(bs=3, N=2)
        inputs = module._unpack_batch(batch)
        # Group IDs: [0,0, 1,1, 2,2]
        expected = torch.tensor([0, 0, 1, 1, 2, 2])
        assert torch.equal(inputs["group_ids"], expected)

    def test_no_video_when_all_masked(self):
        module = _make_module()
        batch = _make_batch()
        batch["mask_visual"] = torch.zeros_like(batch["mask_visual"])
        inputs = module._unpack_batch(batch)
        assert inputs["video"] is None


class TestTrainingStep:
    def test_training_step_returns_loss(self):
        module = _make_module()
        module.train()
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        assert loss is not None
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_validation_step_no_crash(self):
        module = _make_module()
        module.eval()
        batch = _make_batch()
        module.validation_step(batch, 0)   # returns None, just checks no crash

    def test_training_step_numeric_only(self):
        """With all-zero visual mask, training step still works."""
        module = _make_module()
        module.train()
        batch = _make_batch()
        batch["mask_visual"] = torch.zeros_like(batch["mask_visual"])
        loss = module.training_step(batch, 0)
        assert not torch.isnan(loss)

    def test_gradient_update(self):
        """One training step: vision params must receive non-zero gradients."""
        module = _make_module()
        module.train()

        optim = torch.optim.AdamW(module.model.trainable_parameters(), lr=1e-3)
        optim.zero_grad()

        batch = _make_batch()
        loss = module.training_step(batch, 0)
        loss.backward()

        # Check adapter + summarizer have gradients
        for name, p in module.model.cross_modal_adapter.named_parameters():
            assert p.grad is not None, f"No grad: adapter.{name}"
            assert not torch.isnan(p.grad).any()

        for name, p in module.model.latent_summarizer.named_parameters():
            assert p.grad is not None, f"No grad: summarizer.{name}"


class TestOptimizerScheduler:
    def test_configure_optimizers_returns_valid_structure(self):
        module = _make_module()

        # Attach a minimal mock trainer
        class _MockTrainer:
            max_epochs = 5
            train_dataloader = None
            accumulate_grad_batches = 1
        module._trainer = _MockTrainer()

        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["interval"] == "step"

    def test_warmup_phase(self):
        module = _make_module(warmup_steps=10)

        class _MockTrainer:
            max_epochs = 5
            train_dataloader = None
            accumulate_grad_batches = 1
        module._trainer = _MockTrainer()

        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        # At step 0: lr_lambda(0) = 0/10 = 0
        assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-6)
        scheduler.step()  # step 1
        lr_1 = scheduler.get_last_lr()[0]
        # Should be > 0 after first step
        assert lr_1 > 0.0

    def test_param_groups_wd_and_lr(self):
        """Decay groups have wd>0; no-decay groups have wd=0; backbone LR < new LR."""
        module = _make_module(weight_decay=1e-2)

        class _MockTrainer:
            max_epochs = 5
            train_dataloader = None
            accumulate_grad_batches = 1
        module._trainer = _MockTrainer()

        result = module.configure_optimizers()
        optim = result["optimizer"]

        decay_groups   = [g for g in optim.param_groups if g["weight_decay"] > 0]
        nodecay_groups = [g for g in optim.param_groups if g["weight_decay"] == 0.0]

        assert len(decay_groups) > 0,   "Must have at least one WD>0 group"
        assert len(nodecay_groups) > 0, "Must have at least one WD=0 group"

        backbone_groups = [g for g in optim.param_groups if "backbone" in g.get("name", "")]
        new_groups      = [g for g in optim.param_groups if "new" in g.get("name", "")]
        if backbone_groups and new_groups:
            max_backbone_lr = max(g["lr"] for g in backbone_groups)
            min_new_lr      = min(g["lr"] for g in new_groups)
            assert max_backbone_lr <= min_new_lr, (
                f"Backbone LR ({max_backbone_lr}) must be ≤ new-module LR ({min_new_lr})"
            )

    def test_total_steps_uses_estimated_stepping_batches(self):
        """N7: _total_steps must use estimated_stepping_batches when available."""
        module = _make_module()

        class _MockTrainer:
            max_epochs = 5
            train_dataloader = None  # Lightning 2.x returns None here
            accumulate_grad_batches = 1
            estimated_stepping_batches = 250  # 50 steps/epoch × 5 epochs

        module._trainer = _MockTrainer()
        assert module._total_steps == 250, (
            f"Expected _total_steps=250 from estimated_stepping_batches, got {module._total_steps}"
        )


class TestFreezeChronos:
    def test_freeze_chronos_no_grad_backbone(self):
        module = _make_module(freeze_chronos=True)
        for p in module.model.chronos.parameters():
            assert not p.requires_grad

    def test_freeze_chronos_vision_still_trains(self):
        module = _make_module(freeze_chronos=True)
        module.train()
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        loss.backward()
        for name, p in module.model.cross_modal_adapter.named_parameters():
            assert p.grad is not None, f"No grad with frozen backbone: {name}"


class TestSmokeTraining:
    """End-to-end smoke: 3 steps on synthetic data — roadmap done-when criterion 1."""

    def test_smoke_3_steps(self):
        import lightning.pytorch as pl

        module = _make_module()

        from mmtsfm.data.dataset import MMTSFMDataset
        ds = MMTSFMDataset(
            num_samples=4,
            num_entities=2,
            hist_steps=32,
            horizon=8,
            target_dim=1,
            covariate_dim=2,
            video_frames=4,
            img_size=16,
            dataset_name="synthetic",
        )
        dl_train = DataLoader(ds, batch_size=2, shuffle=False)
        dl_val   = DataLoader(ds, batch_size=2, shuffle=False)

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            gradient_clip_val=1.0,
        )
        trainer.fit(module, train_dataloaders=dl_train, val_dataloaders=dl_val)

        final_loss = trainer.callback_metrics.get("train/loss_epoch")
        assert final_loss is not None, "train/loss_epoch metric not logged"

        # M8 fix: loss must be strictly positive — catches silent collapse to 0 (C1-style).
        assert final_loss > 1e-6, (
            f"train/loss collapsed to ~0 ({final_loss:.2e}). "
            "Check future_covariates_mask is zero (not all-ones) in _unpack_batch."
        )

        # Loss must be finite
        assert not torch.isnan(final_loss), "train/loss is NaN"
        assert not torch.isinf(final_loss), "train/loss is Inf"
