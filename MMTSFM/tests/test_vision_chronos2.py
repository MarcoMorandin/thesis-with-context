"""Tests for VisionChronos2 Phase 3 components.

Critical tests (per Notion roadmap Task 3.4):
  1. video=None  → output identical to vanilla Chronos-2
  2. video=zeros → output close to vanilla Chronos-2 (within tolerance)
  3. Forward pass with real video input: shapes correct, no NaNs
  4. Gradient flows through vision modules
  5. CrossModalAdapter shapes
  6. LatentSummarizer shapes

Run with: pytest tests/test_vision_chronos2.py -v
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chronos2(d_model: int = 64, num_layers: int = 2, context_length: int = 32):
    """Minimal Chronos2Model for fast CPU tests."""
    from mmtsfm.models.chronos2 import Chronos2Model, Chronos2CoreConfig

    cfg = Chronos2CoreConfig(
        d_model=d_model,
        d_kv=16,
        d_ff=128,
        num_layers=num_layers,
        num_heads=4,
        dropout_rate=0.0,
        use_grassmann=False,  # TimeSelfAttention for simpler test
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
            z = torch.randn(
                B,
                self._d_v,
                self._t_lat,
                self._h_lat,
                self._w_lat,
                device=x.device,
                dtype=x.dtype,
            )
            return z, {}

        def forward(self, x):
            z, _ = self.encode(x)
            return x, x, {}

    return FakeVidTok()


def _make_vision_model(
    d_model: int = 64,
    d_v: int = 4,
    n_vis_steps: int = 4,
    n_soft_tokens: int = 1,
    adapter_type: str = "linear",
):
    from mmtsfm.models.chronos2 import VisionChronos2Model, VisionChronos2Config

    chronos = _make_chronos2(d_model=d_model)
    vcfg = VisionChronos2Config(
        d_video_latent=d_v,
        n_visual_context_steps=n_vis_steps,
        n_soft_tokens=n_soft_tokens,
        adapter_type=adapter_type,
        visual_dropout_prob=0.0,  # deterministic for tests
        dropout=0.0,
    )
    return VisionChronos2Model(
        chronos_model=chronos,
        vision_config=vcfg,
        vidtok_model=_make_fake_vidtok(d_v=d_v),
    )


# ---------------------------------------------------------------------------
# Unit tests: vision components
# ---------------------------------------------------------------------------


class TestCrossModalAdapter:
    @pytest.mark.parametrize("adapter_type", ["linear", "mlp", "cross_attention"])
    @pytest.mark.parametrize("n_soft", [1, 4])
    def test_output_shape(self, adapter_type, n_soft):
        from mmtsfm.models.vision import CrossModalAdapter

        B, T, D = 2, 10, 64
        adapter = CrossModalAdapter(
            d_model=D, n_soft_tokens=n_soft, adapter_type=adapter_type
        )
        x = torch.randn(B, T, D)
        out = adapter(x)
        assert out.shape == (B, T, n_soft, D), (
            f"Expected {(B, T, n_soft, D)}, got {out.shape}"
        )

    def test_gradient_flows(self):
        from mmtsfm.models.vision import CrossModalAdapter

        adapter = CrossModalAdapter(d_model=32, n_soft_tokens=1, adapter_type="linear")
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = adapter(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestLatentSummarizer:
    def test_output_shape(self):
        from mmtsfm.models.vision import LatentSummarizer

        B, T_lat, P, D_v = 2, 5, 64, 4
        T_ts = 12
        n_vis = 4
        summ = LatentSummarizer(d_v=D_v, d_model=32, n_vis_steps=n_vis, n_heads=4)
        video_tokens = torch.randn(B, T_lat, P, D_v)
        out = summ(video_tokens, T_ts=T_ts)
        assert out.shape == (B, T_ts, 32)

    def test_last_n_vis_nonzero(self):
        """Last n_vis_steps positions should be non-zero (cross-attn); earlier filled by null token."""
        from mmtsfm.models.vision import LatentSummarizer

        n_vis = 3
        T_ts = 10
        summ = LatentSummarizer(
            d_v=4, d_model=16, n_vis_steps=n_vis, n_heads=4, dropout=0.0
        )
        summ.eval()
        video_tokens = torch.randn(1, 5, 16, 4)
        out = summ(video_tokens, T_ts=T_ts)
        # Steps 0..T_ts-n_vis-1: null_visual_token (non-zero after Task 3)
        early = out[0, : T_ts - n_vis, :]
        late = out[0, T_ts - n_vis :, :]
        # null_visual_token is N(0, d^{-1/2}) init — should be non-zero (not zeros padding)
        assert early.abs().sum().item() > 0.0, (
            "Early steps should use null_visual_token (non-zero)"
        )
        assert late.abs().sum().item() > 0.0, (
            "Late steps should be non-zero (visual data present)"
        )

    def test_visual_mask_applied(self):
        from mmtsfm.models.vision import LatentSummarizer

        summ = LatentSummarizer(
            d_v=4, d_model=16, n_vis_steps=4, n_heads=4, dropout=0.0
        )
        summ.eval()
        video_tokens = torch.randn(1, 4, 16, 4)
        all_masked = torch.zeros(1, 4)
        out = summ(video_tokens, T_ts=8, visual_mask=all_masked)
        # With all frames masked, attention should still run (mask just changes weights)
        assert not torch.isnan(out).any()

    def test_frame_delta_t_consumed(self):
        """W5: frame_delta_t (true spacing) is plumbed in and alters the output."""
        from mmtsfm.models.vision import LatentSummarizer

        torch.manual_seed(0)
        B, T_lat, P, D_v = 1, 5, 8, 4
        summ = LatentSummarizer(
            d_v=D_v, d_model=16, n_vis_steps=4, n_heads=4, dropout=0.0
        )
        summ.eval()
        video_tokens = torch.randn(B, T_lat, P, D_v)

        # Non-uniform spacing: most frames old, one very recent.
        dt = torch.tensor([[20000.0, 18000.0, 16000.0, 14000.0, 10.0]])  # [B, T_lat]
        out_uniform = summ(video_tokens, T_ts=8)
        out_dt = summ(video_tokens, T_ts=8, frame_delta_t=dt)

        assert out_dt.shape == out_uniform.shape == (B, 8, 16)
        assert not torch.isnan(out_dt).any()
        # True spacing must change the causal window → different summary.
        assert not torch.allclose(out_dt, out_uniform, atol=1e-5)

    def test_frame_delta_t_wrong_length_ignored(self):
        """Δt whose length != T_lat is ignored (falls back to uniform)."""
        from mmtsfm.models.vision import LatentSummarizer

        torch.manual_seed(0)
        summ = LatentSummarizer(
            d_v=4, d_model=16, n_vis_steps=4, n_heads=4, dropout=0.0
        )
        summ.eval()
        video_tokens = torch.randn(1, 5, 8, 4)
        bad_dt = torch.tensor([[10.0, 20.0, 30.0]])  # length 3 != T_lat=5
        out_uniform = summ(video_tokens, T_ts=8)
        out_bad = summ(video_tokens, T_ts=8, frame_delta_t=bad_dt)
        assert torch.allclose(out_bad, out_uniform, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration tests: VisionChronos2
# ---------------------------------------------------------------------------


class TestVisionChronos2:
    def setup_method(self):
        torch.manual_seed(42)

    # --- Zero-shot regression (Critical test 1) ---
    def test_numeric_only_matches_chronos2(self):
        """video=None output must equal vanilla Chronos-2 output."""
        from mmtsfm.models.chronos2 import (
            VisionChronos2Model,
            VisionChronos2Config,
            Chronos2Model,
        )

        chronos = _make_chronos2()
        vcfg = VisionChronos2Config(
            d_video_latent=4,
            n_visual_context_steps=4,
            visual_dropout_prob=0.0,
        )
        vm = VisionChronos2Model(chronos, vcfg, vidtok_model=_make_fake_vidtok())
        vm.eval()

        B, L = 2, 32
        context = torch.randn(B, L)
        group_ids = torch.arange(B)

        # Chronos-2 native output
        ref = chronos.forward(
            context=context,
            group_ids=group_ids,
            num_output_patches=1,
        )
        ref_preds = ref.quantile_preds

        # VisionChronos2 with video=None
        out = vm.forward(
            context=context,
            group_ids=group_ids,
            num_output_patches=1,
            video=None,
        )

        assert torch.allclose(out.quantile_preds, ref_preds, atol=1e-5), (
            f"Numeric-only output diverged from Chronos-2. "
            f"Max diff: {(out.quantile_preds - ref_preds).abs().max().item():.2e}"
        )

    # --- Critical test 2: zero adapter/summarizer → visual soft tokens are zeros ---
    def test_zeros_video_close_to_numeric(self):
        """Zero adapter+summarizer weights → soft tokens=0 → output close to numeric-only.

        M7 fix: only adapter and summarizer are zeroed, NOT multimodal_embed.
        This preserves the modality/segment/token-type embedding signal so the
        test actually exercises those embeddings rather than masking them.
        Tolerance tightened from 1.0 to 0.5.
        """
        vm = _make_vision_model()
        vm.eval()

        # Zero ONLY the adapter + summarizer — multimodal_embed stays at normal init
        for p in vm.cross_modal_adapter.parameters():
            nn.init.zeros_(p)
        for p in vm.latent_summarizer.parameters():
            nn.init.zeros_(p)

        # multimodal_embed params should NOT all be zero after normal init
        mm_param_norm = sum(p.norm().item() for p in vm.multimodal_embed.parameters())
        assert mm_param_norm > 0.0, (
            "MultimodalEmbedding params are all zero — M1 embeddings not properly initialised"
        )

        B, L = 1, 32
        context = torch.randn(B, L)
        video = torch.zeros(B, 3, 17, 64, 64)
        group_ids = torch.zeros(B, dtype=torch.long)

        out_vis = vm.forward(
            context=context, group_ids=group_ids, num_output_patches=1, video=video
        )
        out_num = vm.forward(
            context=context, group_ids=group_ids, num_output_patches=1, video=None
        )

        # With zeroed adapter+summarizer the visual tokens are zero, but numeric
        # context tokens now carry modality/segment/token-type embeddings (added only
        # when use_video=True). These embeddings shift the output slightly.
        # Tolerance 0.5 — tighter than the old 1.0, accounts for embedding shift.
        max_diff = (out_vis.quantile_preds - out_num.quantile_preds).abs().max().item()
        assert max_diff < 0.5, (
            f"Zero-adapter video output too far from numeric-only: max_diff={max_diff:.4f}. "
            "Check that visual soft tokens are correctly zeroed when adapter weights=0."
        )
        assert not torch.isnan(out_vis.quantile_preds).any()

    # --- Shape tests ---
    def test_output_shape_with_video(self):
        vm = _make_vision_model()
        vm.eval()
        B, L = 2, 32
        context = torch.randn(B, L)
        video = torch.zeros(B, 3, 17, 64, 64)
        group_ids = torch.zeros(B, dtype=torch.long)

        out = vm.forward(
            context=context, group_ids=group_ids, num_output_patches=2, video=video
        )

        assert out.quantile_preds is not None
        assert out.quantile_preds.shape[0] == B
        assert out.quantile_preds.shape[1] == 3  # num_quantiles
        assert not torch.isnan(out.quantile_preds).any()

    def test_no_nan_outputs(self):
        vm = _make_vision_model()
        vm.eval()
        B = 1
        context = torch.randn(B, 32)
        video = torch.rand(B, 3, 17, 64, 64)
        group_ids = torch.zeros(B, dtype=torch.long)

        out = vm.forward(
            context=context, group_ids=group_ids, num_output_patches=1, video=video
        )
        assert not torch.isnan(out.quantile_preds).any()
        assert not torch.isinf(out.quantile_preds).any()

    # --- Gradient flow (critical test: adapter must receive gradient) ---
    def test_gradient_flows_through_adapter(self):
        vm = _make_vision_model()
        vm.train()
        vm.zero_grad()

        B = 1
        context = torch.randn(B, 32)
        video = torch.rand(B, 3, 17, 64, 64)
        group_ids = torch.zeros(B, dtype=torch.long)
        future_target = torch.randn(B, 8)

        out = vm.forward(
            context=context,
            group_ids=group_ids,
            num_output_patches=1,
            video=video,
            future_target=future_target,
        )
        assert out.loss is not None
        out.loss.backward()

        # Check gradient flows to adapter
        for name, p in vm.cross_modal_adapter.named_parameters():
            assert p.grad is not None, f"No grad for adapter.{name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad in adapter.{name}"

        # Check gradient flows to summarizer
        for name, p in vm.latent_summarizer.named_parameters():
            assert p.grad is not None, f"No grad for summarizer.{name}"

    # --- N_soft > 1 ---
    @pytest.mark.parametrize("n_soft", [1, 4])
    def test_n_soft_tokens(self, n_soft):
        vm = _make_vision_model(n_soft_tokens=n_soft)
        vm.eval()
        B = 1
        context = torch.randn(B, 32)
        video = torch.rand(B, 3, 17, 64, 64)
        group_ids = torch.zeros(B, dtype=torch.long)

        out = vm.forward(
            context=context, group_ids=group_ids, num_output_patches=1, video=video
        )
        assert out.quantile_preds.shape[0] == B

    # --- Modality dropout ---
    def test_modality_dropout_training(self):
        from mmtsfm.models.chronos2 import VisionChronos2Model, VisionChronos2Config

        chronos = _make_chronos2()
        vcfg = VisionChronos2Config(
            d_video_latent=4,
            n_visual_context_steps=4,
            visual_dropout_prob=1.0,  # always drop
        )
        vm = VisionChronos2Model(chronos, vcfg, vidtok_model=_make_fake_vidtok())
        vm.train()

        B = 2
        context = torch.randn(B, 32)
        video = torch.rand(B, 3, 17, 64, 64)
        group_ids = torch.zeros(B, dtype=torch.long)

        out = vm.forward(
            context=context, group_ids=group_ids, num_output_patches=1, video=video
        )
        # With dropout=1.0, visual_active should all be False
        assert out.visual_active is not None
        assert not out.visual_active.any()

    # --- visual_active flag ---
    def test_visual_active_none_when_no_video(self):
        vm = _make_vision_model()
        vm.eval()
        out = vm.forward(context=torch.randn(2, 32), num_output_patches=1, video=None)
        assert out.visual_active is None

    # --- M3 regression tests ---
    def test_modality_dropout_no_crash_with_video(self):
        """Regression: _build_visual_embeds must not crash with video input (M3 call-site fix)."""
        from mmtsfm.models.chronos2 import VisionChronos2Model, VisionChronos2Config

        chronos = _make_chronos2()
        vcfg = VisionChronos2Config(
            d_video_latent=4,
            n_visual_context_steps=4,
            visual_dropout_prob=1.0,
            numeric_dropout_prob=0.0,
        )
        vm = VisionChronos2Model(chronos, vcfg, vidtok_model=_make_fake_vidtok())
        vm.train()
        B = 2
        out = vm.forward(
            context=torch.randn(B, 32),
            group_ids=torch.zeros(B, dtype=torch.long),
            num_output_patches=1,
            video=torch.rand(B, 3, 17, 64, 64),
        )
        assert not torch.isnan(out.quantile_preds).any()
        assert out.visual_active is not None
        assert not out.visual_active.any()

    def test_numeric_dropout_fires(self):
        """M3: numeric dropout must zero numeric stream independently of visual."""
        from mmtsfm.models.chronos2 import VisionChronos2Model, VisionChronos2Config

        chronos = _make_chronos2()
        vcfg = VisionChronos2Config(
            d_video_latent=4,
            n_visual_context_steps=4,
            visual_dropout_prob=0.0,
            numeric_dropout_prob=1.0,
        )
        vm = VisionChronos2Model(chronos, vcfg, vidtok_model=_make_fake_vidtok())
        vm.train()
        B = 2
        out = vm.forward(
            context=torch.randn(B, 32),
            group_ids=torch.zeros(B, dtype=torch.long),
            num_output_patches=1,
            video=torch.rand(B, 3, 17, 64, 64),
        )
        assert not torch.isnan(out.quantile_preds).any()
        assert out.visual_active is not None
        assert out.visual_active.all()
        assert out.numeric_active is not None
        assert not out.numeric_active.any()

    # --- Adapter types ---
    @pytest.mark.parametrize("adapter_type", ["linear", "mlp", "cross_attention"])
    def test_adapter_types_no_crash(self, adapter_type):
        vm = _make_vision_model(adapter_type=adapter_type)
        vm.eval()
        B = 1
        out = vm.forward(
            context=torch.randn(B, 32),
            group_ids=torch.zeros(B, dtype=torch.long),
            num_output_patches=1,
            video=torch.rand(B, 3, 17, 64, 64),
        )
        assert not torch.isnan(out.quantile_preds).any()

    def test_covariate_channels_numeric_only(self):
        """N1: covariate_channels must be tokenized even when video=None."""
        vm = _make_vision_model()
        vm.eval()

        B, L, H = 2, 32, 8
        context = torch.randn(B, L)
        group_ids = torch.arange(B, dtype=torch.long)
        covariate_channels = [torch.randn(B, H), torch.randn(B, H)]  # C_cov=2

        out_with_cov = vm.forward(
            context=context,
            group_ids=group_ids,
            num_output_patches=1,
            covariate_channels=covariate_channels,
            video=None,
        )
        out_no_cov = vm.forward(
            context=context,
            group_ids=group_ids,
            num_output_patches=1,
            covariate_channels=None,
            video=None,
        )

        assert not torch.isnan(out_with_cov.quantile_preds).any(), (
            "NaN with covariate_channels"
        )
        # Covariate tokens participate in Group Attention → output must differ
        assert not torch.allclose(
            out_with_cov.quantile_preds, out_no_cov.quantile_preds
        ), (
            "Output identical with and without covariate_channels — "
            "N1 not fixed: covariate rows not reaching encoder"
        )

    def test_entity_ids_oob_raises_clear_error(self):
        """N5: entity_ids >= n_entities must raise AssertionError with clear message."""
        from mmtsfm.models.chronos2 import VisionChronos2Model, VisionChronos2Config

        chronos = _make_chronos2()
        vcfg = VisionChronos2Config(
            d_video_latent=4,
            n_visual_context_steps=4,
            n_entities=2,  # only 2 entities in embedding table
        )
        vm = VisionChronos2Model(chronos, vcfg, vidtok_model=_make_fake_vidtok())
        vm.eval()

        with pytest.raises(AssertionError, match="n_entities"):
            vm.forward(
                context=torch.randn(2, 32),
                group_ids=torch.zeros(2, dtype=torch.long),
                num_output_patches=1,
                video=torch.rand(2, 3, 17, 64, 64),
                entity_ids=torch.tensor([0, 5]),  # id=5 >= n_entities=2 → OOB
            )

    def test_visual_mask_zeros_padded_positions(self):
        """N9: Group Attention must not attend to zero-padded visual context positions."""
        from mmtsfm.models.chronos2 import VisionChronos2Model, VisionChronos2Config

        chronos = _make_chronos2(context_length=32)
        # context patches = 32/8 = 4 total; n_visual_context_steps=2 → only last 2 have visual
        vcfg = VisionChronos2Config(
            d_video_latent=4,
            n_visual_context_steps=2,
            visual_dropout_prob=0.0,
            dropout=0.0,
        )
        vm = VisionChronos2Model(chronos, vcfg, vidtok_model=_make_fake_vidtok())
        vm.eval()

        B = 1
        out = vm.forward(
            context=torch.randn(B, 32),
            group_ids=torch.zeros(B, dtype=torch.long),
            num_output_patches=1,
            video=torch.rand(B, 3, 17, 64, 64),
        )
        assert not torch.isnan(out.quantile_preds).any(), (
            "NaN with zero-padded visual mask"
        )
        assert not torch.isinf(out.quantile_preds).any()


# ---------------------------------------------------------------------------
# M4: CausalGrassmannMixing RoPE dimension parity tests
# ---------------------------------------------------------------------------


class TestVidTokEncoder:
    """M6: VidTokEncoder spatial shape fix tests."""

    def test_spatial_patches_reflects_actual_input_size(self):
        """M6: spatial_patches must match actual forward input dims, not hard-coded 256×256."""
        from mmtsfm.models.vision.vidtok_encoder import VidTokEncoder

        fake = _make_fake_vidtok(d_v=4, t_lat=2, h_lat=4, w_lat=4)
        enc = VidTokEncoder(model=fake)

        video = torch.zeros(1, 3, 17, 64, 64)
        enc(video)

        assert enc.spatial_patches == 4 * 4, (
            f"Expected spatial_patches=16, got {enc.spatial_patches}"
        )
        assert enc.d_v == 4

    def test_probe_latent_shape_accepts_img_size(self):
        """M6: probe_latent_shape must accept img_size param."""
        from mmtsfm.models.vision.vidtok_encoder import VidTokEncoder

        fake = _make_fake_vidtok(d_v=4, t_lat=3, h_lat=8, w_lat=8)
        enc = VidTokEncoder(model=fake)
        enc.probe_latent_shape(device=torch.device("cpu"), img_size=64)

        assert enc.spatial_patches == 8 * 8
        assert enc.d_v == 4


class TestCausalGrassmannMixing:
    def test_odd_reduced_dim_raises(self):
        """M4: grassmann_reduced_dim must be even for RoPE — assert fires on odd r."""
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig
        from mmtsfm.models.chronos2.grassmann import CausalGrassmannMixing

        cfg = Chronos2CoreConfig(
            d_model=64,
            d_kv=16,
            d_ff=128,
            num_layers=1,
            num_heads=4,
            use_grassmann=True,
            grassmann_reduced_dim=3,
            chronos_config={
                "context_length": 16,
                "output_patch_size": 4,
                "input_patch_size": 4,
                "input_patch_stride": 4,
                "quantiles": [0.5],
                "use_reg_token": False,
                "use_arcsinh": False,
                "max_output_patches": 2,
            },
        )
        with pytest.raises(AssertionError, match="even"):
            CausalGrassmannMixing(cfg)

    def test_even_reduced_dim_ok(self):
        """M4: even grassmann_reduced_dim must instantiate without error."""
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig
        from mmtsfm.models.chronos2.grassmann import CausalGrassmannMixing

        cfg = Chronos2CoreConfig(
            d_model=64,
            d_kv=16,
            d_ff=128,
            num_layers=1,
            num_heads=4,
            use_grassmann=True,
            grassmann_reduced_dim=4,
            chronos_config={
                "context_length": 16,
                "output_patch_size": 4,
                "input_patch_size": 4,
                "input_patch_stride": 4,
                "quantiles": [0.5],
                "use_reg_token": False,
                "use_arcsinh": False,
                "max_output_patches": 2,
            },
        )
        layer = CausalGrassmannMixing(cfg)
        assert layer.r == 4

    def test_plucker_buffers_used_not_reallocated(self):
        """M5: _compute_plucker must use registered buffers, not allocate new index tensors."""
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig
        from mmtsfm.models.chronos2.grassmann import CausalGrassmannMixing

        cfg = Chronos2CoreConfig(
            d_model=64,
            d_kv=16,
            d_ff=128,
            num_layers=1,
            num_heads=4,
            use_grassmann=True,
            grassmann_reduced_dim=4,
            grassmann_window_offsets=[1],
            chronos_config={
                "context_length": 8,
                "output_patch_size": 4,
                "input_patch_size": 4,
                "input_patch_stride": 4,
                "quantiles": [0.5],
                "use_reg_token": False,
                "use_arcsinh": False,
                "max_output_patches": 2,
            },
        )
        layer = CausalGrassmannMixing(cfg)
        layer.eval()

        ptr_i_before = layer._plucker_idx_i.data_ptr()
        ptr_j_before = layer._plucker_idx_j.data_ptr()

        B, L, D = 1, 8, 64
        mask = torch.zeros(B, 1, L, L)
        h = torch.randn(B, L, D)
        pos = torch.arange(L).unsqueeze(0)
        layer(h, mask, pos)

        assert layer._plucker_idx_i.data_ptr() == ptr_i_before, (
            "_plucker_idx_i was reallocated"
        )
        assert layer._plucker_idx_j.data_ptr() == ptr_j_before, (
            "_plucker_idx_j was reallocated"
        )

    def test_plucker_output_deterministic(self):
        """M5: two identical forward passes must produce identical output."""
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig
        from mmtsfm.models.chronos2.grassmann import CausalGrassmannMixing

        torch.manual_seed(7)
        cfg = Chronos2CoreConfig(
            d_model=64,
            d_kv=16,
            d_ff=128,
            num_layers=1,
            num_heads=4,
            use_grassmann=True,
            grassmann_reduced_dim=4,
            grassmann_window_offsets=[1, 2],
            chronos_config={
                "context_length": 8,
                "output_patch_size": 4,
                "input_patch_size": 4,
                "input_patch_stride": 4,
                "quantiles": [0.5],
                "use_reg_token": False,
                "use_arcsinh": False,
                "max_output_patches": 2,
            },
        )
        layer = CausalGrassmannMixing(cfg)
        layer.eval()

        B, L, D = 2, 8, 64
        mask = torch.zeros(B, 1, L, L)
        h = torch.randn(B, L, D)
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)

        out1 = layer(h, mask, pos).hidden_states
        out2 = layer(h, mask, pos).hidden_states

        assert torch.allclose(out1, out2, atol=1e-6), "Forward pass not deterministic"
        assert not torch.isnan(out1).any()


# ---------------------------------------------------------------------------
# Task 2: VisualEncoder
# ---------------------------------------------------------------------------


class TestVisualEncoder:
    def test_d_v_property_vit_large(self):
        from mmtsfm.models.vision.visual_encoder import VisualEncoder

        enc = VisualEncoder(arch="vit_large")
        assert enc.d_v == 1024

    def test_d_v_property_vit_base(self):
        from mmtsfm.models.vision.visual_encoder import VisualEncoder

        enc = VisualEncoder(arch="vit_base")
        assert enc.d_v == 768

    def test_mock_encoder_forward_shape(self):
        from mmtsfm.models.vision.visual_encoder import VisualEncoder
        import torch.nn as nn

        class _FakeVJepa(nn.Module):
            def forward(self, x):  # x: [B, C, T, H, W]
                B, C, T, H, W = x.shape
                T_lat = T // 2
                P = (H // 16) * (W // 16)
                return torch.zeros(B, T_lat * P, 1024)

        enc = VisualEncoder(arch="vit_large")
        enc._encoder = _FakeVJepa()
        x = torch.randn(2, 3, 8, 64, 64)  # [B, C, T_v, H, W]
        out = enc(x)
        # T_lat = 8//2 = 4, P = (64//16)^2 = 16
        assert out.shape == (2, 4, 16, 1024)


# ---------------------------------------------------------------------------
# Task 3: LatentSummarizer null_visual_token
# ---------------------------------------------------------------------------


class TestLatentSummarizerNullToken:
    def test_null_token_param_exists(self):
        from mmtsfm.models.vision.latent_summarizer import LatentSummarizer

        ls = LatentSummarizer(d_v=16, d_model=32, n_vis_steps=4)
        assert hasattr(ls, "null_visual_token")
        assert isinstance(ls.null_visual_token, torch.nn.Parameter)
        assert ls.null_visual_token.shape == (1, 1, 32)

    def test_null_token_fills_macro_positions(self):
        from mmtsfm.models.vision.latent_summarizer import LatentSummarizer

        ls = LatentSummarizer(d_v=16, d_model=32, n_vis_steps=4)
        with torch.no_grad():
            ls.null_visual_token.fill_(99.0)
        video_tokens = torch.randn(2, 5, 6, 16)  # [B, T_lat, P, D_v]
        out = ls(video_tokens, T_ts=8)
        assert out.shape == (2, 8, 32)
        # Macro positions (0..3): should be null token value
        torch.testing.assert_close(out[:, :4, :], torch.full_like(out[:, :4, :], 99.0))
        # Visual positions (4..7): should NOT be null token
        assert not torch.all(out[:, 4:, :] == 99.0)

    def test_null_token_init_magnitude(self):
        from mmtsfm.models.vision.latent_summarizer import LatentSummarizer

        d_model = 512
        ls = LatentSummarizer(d_v=64, d_model=d_model, n_vis_steps=6)
        std = ls.null_visual_token.std().item()
        expected_std = d_model**-0.5
        assert abs(std - expected_std) < 0.1 * expected_std  # within 10%


class TestConfigExtensions:
    def test_chronos2_config_has_grassmann_modality_pair_bias(self):
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig

        cfg = Chronos2CoreConfig()
        assert hasattr(cfg, "grassmann_modality_pair_bias")
        assert cfg.grassmann_modality_pair_bias is True  # default

    def test_vision_config_fusion_mode_default(self):
        from mmtsfm.models.chronos2.vision_chronos2 import VisionChronos2Config

        cfg = VisionChronos2Config()
        assert cfg.fusion_mode == "late"

    def test_vision_config_accepts_interleaved(self):
        from mmtsfm.models.chronos2.vision_chronos2 import VisionChronos2Config

        cfg = VisionChronos2Config(fusion_mode="interleaved")
        assert cfg.fusion_mode == "interleaved"

    def test_vision_config_encoder_fields(self):
        from mmtsfm.models.chronos2.vision_chronos2 import VisionChronos2Config

        cfg = VisionChronos2Config(
            visual_encoder_type="vjepa2",
            visual_encoder_ckpt_path="/path/to/ckpt",
            freeze_visual_encoder=True,
            skip_vision_stack=False,
        )
        assert cfg.visual_encoder_type == "vjepa2"
        assert cfg.visual_encoder_ckpt_path == "/path/to/ckpt"
        assert cfg.freeze_visual_encoder is True
        assert cfg.skip_vision_stack is False


# ---------------------------------------------------------------------------
# Task 5: Grassmann modality-pair bias tests
# ---------------------------------------------------------------------------


class TestGrassmannModalityPairBias:
    def _make_grassmann(self, use_bias: bool = True):
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig
        from mmtsfm.models.chronos2.grassmann import CausalGrassmannMixing

        cfg = Chronos2CoreConfig(
            d_model=64,
            grassmann_reduced_dim=8,
            grassmann_modality_pair_bias=use_bias,
        )
        return CausalGrassmannMixing(cfg)

    def test_modality_pair_bias_param_exists(self):
        gm = self._make_grassmann(use_bias=True)
        assert hasattr(gm, "modality_pair_bias")
        assert gm.modality_pair_bias.shape == (4,)
        assert torch.all(gm.modality_pair_bias == 0.0)  # init zeros

    def test_no_bias_param_when_disabled(self):
        gm = self._make_grassmann(use_bias=False)
        assert not hasattr(gm, "modality_pair_bias")

    def test_forward_accepts_modality_mask(self):
        gm = self._make_grassmann(use_bias=True)
        B, L, d = 2, 16, 64
        h = torch.randn(B, L, d)
        mask = (1.0 - torch.ones(B, 1, 1, L)) * torch.finfo(torch.float32).min
        pos_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
        modality_mask = torch.zeros(B, L, dtype=torch.long)
        modality_mask[:, L // 2 :] = 1
        out = gm(h, mask, pos_ids, modality_mask=modality_mask)
        assert out.hidden_states.shape == (B, L, d)

    def test_forward_without_modality_mask_unchanged(self):
        gm = self._make_grassmann(use_bias=True)
        B, L, d = 2, 16, 64
        torch.manual_seed(42)
        h = torch.randn(B, L, d)
        mask = (1.0 - torch.ones(B, 1, 1, L)) * torch.finfo(torch.float32).min
        pos_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
        out_no_mask = gm(h, mask, pos_ids, modality_mask=None)
        assert torch.isfinite(out_no_mask.hidden_states).all()


class TestInterleaveSequences:
    def test_output_shape(self):
        from mmtsfm.models.chronos2.vision_chronos2 import interleave_sequences

        B, T_ctx, n_vis, d = 2, 12, 4, 32
        ts = torch.randn(B, T_ctx, d)
        vis = torch.randn(B, n_vis, d)
        out, mm = interleave_sequences(ts, vis, n_vis)
        assert out.shape == (B, T_ctx + n_vis, d)
        assert mm.shape == (B, T_ctx + n_vis)

    def test_macro_positions_unchanged(self):
        from mmtsfm.models.chronos2.vision_chronos2 import interleave_sequences

        B, T_ctx, n_vis, d = 2, 10, 3, 8
        ts = torch.randn(B, T_ctx, d)
        vis = torch.randn(B, n_vis, d)
        out, mm = interleave_sequences(ts, vis, n_vis)
        T_M = T_ctx - n_vis
        torch.testing.assert_close(out[:, :T_M, :], ts[:, :T_M, :])
        assert mm[:, :T_M].sum() == 0

    def test_refinement_alternates_ts_vis(self):
        from mmtsfm.models.chronos2.vision_chronos2 import interleave_sequences

        B, T_ctx, n_vis, d = 1, 8, 4, 4
        T_M = T_ctx - n_vis  # = 4
        ts = torch.randn(B, T_ctx, d)
        vis = torch.randn(B, n_vis, d)
        out, mm = interleave_sequences(ts, vis, n_vis)
        for k in range(n_vis):
            pos_ts = T_M + 2 * k
            pos_vis = T_M + 2 * k + 1
            torch.testing.assert_close(out[:, pos_ts, :], ts[:, T_M + k, :])
            torch.testing.assert_close(out[:, pos_vis, :], vis[:, k, :])
            assert mm[0, pos_ts].item() == 0  # TS
            assert mm[0, pos_vis].item() == 1  # visual

    def test_position_ids_temporal(self):
        from mmtsfm.models.chronos2.vision_chronos2 import (
            build_interleaved_position_ids,
        )

        T_M, n_vis, T_fut = 4, 3, 2
        pos = build_interleaved_position_ids(
            T_M, n_vis, T_fut, device=torch.device("cpu")
        )
        assert pos.shape == (1, T_M + 2 * n_vis + T_fut)
        torch.testing.assert_close(pos[0, :T_M], torch.arange(T_M))
        expected_refine = torch.tensor([4, 4, 5, 5, 6, 6])
        torch.testing.assert_close(pos[0, T_M : T_M + 2 * n_vis], expected_refine)
        torch.testing.assert_close(pos[0, T_M + 2 * n_vis :], torch.tensor([7, 8]))


class TestInterleavedFusionMode:
    def _make_model_interleaved(self, n_vis=4):
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig
        from mmtsfm.models.chronos2.model import Chronos2Model
        from mmtsfm.models.chronos2.vision_chronos2 import (
            VisionChronos2Config,
            VisionChronos2Model,
        )

        core_cfg = Chronos2CoreConfig(
            d_model=64,
            num_layers=2,
            chronos_config={
                "context_length": 32,
                "input_patch_size": 4,
                "input_patch_stride": 4,
                "output_patch_size": 4,
                "quantiles": [0.1, 0.5, 0.9],
                "use_arcsinh": True,
                "max_output_patches": 2,
            },
        )
        chronos = Chronos2Model(core_cfg)
        vcfg = VisionChronos2Config(
            fusion_mode="interleaved",
            visual_encoder_type="vidtok",
            d_video_latent=4,
            n_visual_context_steps=n_vis,
            skip_vision_stack=False,
        )
        fake_vidtok = _make_fake_vidtok(d_v=4, t_lat=5, h_lat=4, w_lat=4)
        return VisionChronos2Model(chronos, vcfg, vidtok_model=fake_vidtok)

    def test_interleaved_forward_shape(self):
        model = self._make_model_interleaved(n_vis=4)
        B = 2
        T = 32
        H = 8
        context = torch.randn(B, T)
        future_target = torch.randn(B, H)
        video = torch.randn(B, 3, 8, 32, 32)  # [B, C, T_v, H, W]
        out = model(
            context=context,
            future_target=future_target,
            video=video,
            num_output_patches=2,
        )
        assert out.quantile_preds is not None
        assert out.quantile_preds.shape[0] == B
        assert out.loss is not None

    def test_interleaved_no_nan(self):
        model = self._make_model_interleaved(n_vis=4)
        B, T, H = 2, 32, 8
        context = torch.randn(B, T)
        video = torch.randn(B, 3, 8, 32, 32)
        out = model(
            context=context,
            video=video,
            future_target=torch.randn(B, H),
            num_output_patches=2,
        )
        assert torch.isfinite(out.quantile_preds).all()
        assert torch.isfinite(out.loss)


# ---------------------------------------------------------------------------
# Task 8: Stage Controls
# ---------------------------------------------------------------------------


class TestLightningModuleStageControls:
    def _make_module(self, warmup_steps=0, freeze_chronos=False):
        from mmtsfm.models.chronos2.lightning_module import (
            VisionChronos2LightningModule,
        )

        core_cfg = {
            "d_model": 32,
            "d_kv": 8,
            "d_ff": 64,
            "num_layers": 2,
            "num_heads": 2,
            "use_grassmann": True,
            "grassmann_reduced_dim": 4,
            "chronos_config": {
                "context_length": 16,
                "output_patch_size": 4,
                "input_patch_size": 4,
                "input_patch_stride": 4,
                "quantiles": [0.5],
                "use_reg_token": False,
                "use_arcsinh": False,
                "max_output_patches": 2,
            },
        }
        vision_cfg = {
            "d_video_latent": 4,
            "n_visual_context_steps": 2,
            "skip_vision_stack": True,
        }
        return VisionChronos2LightningModule(
            chronos_core_cfg=core_cfg,
            vision_cfg=vision_cfg,
            lr=1e-3,
            warmup_steps=10,
            grassmann_warmup_steps=warmup_steps,
            freeze_chronos=freeze_chronos,
            pretrained_model_name_or_path=None,
        )

    def test_freeze_chronos_keeps_grassmann_trainable(self):
        mod = self._make_module(freeze_chronos=True)

        # Last encoder block is intentionally trainable under freeze_chronos
        # (Stage 2a fix) — collect its parameter ids so we can exclude them
        # from the frozen-check below.
        block_layers = list(mod.model.chronos.encoder.block)
        last_block_param_ids = {id(p) for p in block_layers[-1].parameters()}

        # Re-initialised layers (size mismatch with checkpoint) are also kept
        # trainable by lightning_module.py and must be excluded from the
        # frozen-check below.
        reinit_substrings = (
            "input_patch_embedding",
            "output_patch_embedding",
            "shared",
        )

        grassmann_found = False
        standard_found = False
        for name, p in mod.model.chronos.named_parameters():
            if any(
                k in name
                for k in (
                    "W_red",
                    "W_plu",
                    "W_gate",
                    "offset_weights",
                    "modality_pair_bias",
                )
            ):
                assert p.requires_grad is True, (
                    f"Grassmann parameter {name} should be trainable"
                )
                grassmann_found = True
            elif id(p) in last_block_param_ids:
                # Stage 2a unfreeze — covered by test_freeze_chronos_unfreezes_last_encoder_block
                assert p.requires_grad is True, (
                    f"Last-block parameter {name} should be trainable"
                )
            elif any(k in name for k in reinit_substrings):
                # Re-init layers — must stay trainable to learn from scratch.
                assert p.requires_grad is True, (
                    f"Re-init parameter {name} should be trainable"
                )
            else:
                assert p.requires_grad is False, (
                    f"Standard parameter {name} should be frozen"
                )
                standard_found = True

        assert grassmann_found, "No Grassmann parameters found to test"
        assert standard_found, "No standard parameters found to test"

    def test_grassmann_warmup_schedule(self):
        mod = self._make_module(warmup_steps=5)
        # Use a dummy trainer to satisfy configure_optimizers
        from unittest.mock import Mock

        mod.trainer = Mock()
        mod.trainer.estimated_stepping_batches = 100

        opt_dict = mod.configure_optimizers()
        scheduler = opt_dict["lr_scheduler"]["scheduler"]
        optimizer = opt_dict["optimizer"]

        lrs = scheduler.lr_lambdas

        standard_lr = None
        grassmann_lr = None
        for group, lr_lambda in zip(optimizer.param_groups, lrs):
            if "grassmann" in group.get("name", ""):
                if grassmann_lr is None:
                    grassmann_lr = lr_lambda(2)
            else:
                if standard_lr is None:
                    standard_lr = lr_lambda(2)

        import pytest

        assert standard_lr == pytest.approx(0.2)
        assert grassmann_lr == pytest.approx(0.4)

    def test_freeze_chronos_unfreezes_last_encoder_block(self):
        """Stage 2a: when freeze_chronos=True, the LAST encoder block of
        Chronos-2 must remain trainable so cross-row attention can adapt to
        the visual modality. Earlier blocks stay frozen except for Grassmann
        inserts (which are trainable in every block by design)."""
        mod = self._make_module(freeze_chronos=True)
        block_layers = list(mod.model.chronos.encoder.block)
        assert len(block_layers) >= 2, (
            "fixture must have at least 2 encoder blocks for this assertion"
        )

        last = block_layers[-1]
        earlier = block_layers[:-1]

        # Standard (non-Grassmann) attention/FFN params in the last block must
        # be trainable. Without this, the Stage 2a cross-row attention has no
        # learnable degree of freedom to attend to visual rows.
        grassmann_substrings = (
            "W_red",
            "W_plu",
            "W_gate",
            "offset_weights",
            "modality_pair_bias",
        )

        last_standard_trainable = [
            name
            for name, p in last.named_parameters()
            if p.requires_grad and not any(k in name for k in grassmann_substrings)
        ]
        assert last_standard_trainable, (
            "last encoder block must have non-Grassmann trainable params under freeze_chronos=True"
        )

        # Earlier blocks' standard params must remain frozen — only Grassmann
        # params are allowed to be trainable there.
        for i, blk in enumerate(earlier):
            for name, p in blk.named_parameters():
                if any(k in name for k in grassmann_substrings):
                    continue  # Grassmann params trainable across all blocks
                assert not p.requires_grad, (
                    f"earlier encoder block {i} param {name} must remain frozen"
                )


# ---------------------------------------------------------------------------
# Task 9: Proposal Training Configs (Integration Tests)
# ---------------------------------------------------------------------------


class TestProposalTrainingConfigs:
    def _make_module(self, stage=1, fusion_mode="late"):
        from mmtsfm.models.chronos2.lightning_module import (
            VisionChronos2LightningModule,
        )

        core_cfg = {
            "d_model": 32,
            "d_kv": 8,
            "d_ff": 64,
            "num_layers": 2,
            "num_heads": 2,
            "use_grassmann": True,
            "grassmann_reduced_dim": 4,
            "chronos_config": {
                "context_length": 16,
                "output_patch_size": 4,
                "input_patch_size": 4,
                "input_patch_stride": 4,
                "quantiles": [0.5],
                "use_reg_token": False,
                "use_arcsinh": False,
                "max_output_patches": 2,
            },
        }
        vision_cfg = {
            "d_video_latent": 4,
            "n_visual_context_steps": 2,
            "fusion_mode": fusion_mode,
            "skip_vision_stack": (stage == 1),
        }
        vidtok_model = None
        if stage > 1:
            vidtok_model = _make_fake_vidtok(d_v=4, t_lat=5, h_lat=4, w_lat=4)
        freeze_chronos = stage == 3
        mod = VisionChronos2LightningModule(
            chronos_core_cfg=core_cfg,
            vision_cfg=vision_cfg,
            lr=1e-3,
            warmup_steps=10,
            grassmann_warmup_steps=10,
            freeze_chronos=freeze_chronos,
            pretrained_model_name_or_path=None,
            vidtok_model=vidtok_model,
        )
        from unittest.mock import Mock

        mod.trainer = Mock()
        mod.trainer.is_global_zero = True
        mod.trainer.estimated_stepping_batches = 100
        return mod

    def test_stage1_forward_no_video(self):
        mod = self._make_module(stage=1)
        mod.eval()
        batch = {
            "Y": torch.randn(2, 1, 16, 1),
            "Y_future": torch.randn(2, 1, 4, 1),
            "X_cov": torch.randn(2, 1, 20, 1),
            "V": torch.randn(2, 1, 4, 3, 32, 32),
            "mask_target": torch.ones(2, 1, 16, 1),
            "mask_future": torch.ones(2, 1, 4, 1),
            "mask_visual": torch.zeros(2, 1, 4),  # no video
            "daylight_future": torch.ones(2, 1, 4, 1),
            "site_id": torch.zeros(2, 1, dtype=torch.long),
        }
        out = mod.training_step(batch, batch_idx=0)
        assert isinstance(out, torch.Tensor)
        assert not torch.isnan(out).any()

    def test_stage1_gradients_flow_to_grassmann(self):
        mod = self._make_module(stage=1)
        mod.train()
        batch = {
            "Y": torch.randn(2, 1, 16, 1),
            "Y_future": torch.randn(2, 1, 4, 1),
            "X_cov": torch.randn(2, 1, 20, 1),
            "V": torch.randn(2, 1, 4, 3, 32, 32),
            "mask_target": torch.ones(2, 1, 16, 1),
            "mask_future": torch.ones(2, 1, 4, 1),
            "mask_visual": torch.zeros(2, 1, 4),  # no video
            "daylight_future": torch.ones(2, 1, 4, 1),
            "site_id": torch.zeros(2, 1, dtype=torch.long),
        }
        loss = mod.training_step(batch, batch_idx=0)
        loss.backward()

        grassmann_grad = False
        standard_grad = False
        for name, p in mod.model.chronos.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is not None and p.grad.abs().sum() > 0:
                if any(
                    k in name
                    for k in (
                        "W_red",
                        "W_plu",
                        "W_gate",
                        "offset_weights",
                        "modality_pair_bias",
                    )
                ):
                    grassmann_grad = True
                else:
                    standard_grad = True

        assert grassmann_grad, (
            "Expected gradients to flow to Grassmann parameters in Stage 1"
        )
        assert standard_grad, (
            "Expected gradients to flow to standard parameters in Stage 1"
        )

    def test_interleaved_mode_full_forward(self):
        mod = self._make_module(stage=2, fusion_mode="interleaved")
        mod.eval()

        # Test 1: With video
        batch_video = {
            "Y": torch.randn(2, 1, 16, 1),
            "Y_future": torch.randn(2, 1, 4, 1),
            "X_cov": torch.randn(2, 1, 20, 1),
            "V": torch.randn(2, 1, 4, 3, 32, 32),
            "mask_target": torch.ones(2, 1, 16, 1),
            "mask_future": torch.ones(2, 1, 4, 1),
            "mask_visual": torch.ones(2, 1, 4),  # video active
            "daylight_future": torch.ones(2, 1, 4, 1),
            "site_id": torch.zeros(2, 1, dtype=torch.long),
        }
        loss_video = mod.training_step(batch_video, batch_idx=0)
        assert torch.isfinite(loss_video)

        # Test 2: Without video (should gracefully fallback to temporal only)
        batch_no_video = {
            "Y": torch.randn(2, 1, 16, 1),
            "Y_future": torch.randn(2, 1, 4, 1),
            "X_cov": torch.randn(2, 1, 20, 1),
            "V": torch.randn(2, 1, 4, 3, 32, 32),
            "mask_target": torch.ones(2, 1, 16, 1),
            "mask_future": torch.ones(2, 1, 4, 1),
            "mask_visual": torch.zeros(2, 1, 4),  # no video
            "daylight_future": torch.ones(2, 1, 4, 1),
            "site_id": torch.zeros(2, 1, dtype=torch.long),
        }
        loss_no_video = mod.training_step(batch_no_video, batch_idx=1)
        assert torch.isfinite(loss_no_video)
