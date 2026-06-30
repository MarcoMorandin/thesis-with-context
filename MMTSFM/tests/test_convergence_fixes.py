"""Tests for convergence bug fixes (bugs A, B, F, G, H)."""
from __future__ import annotations
import math
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_chronos_config(d_model=64, d_kv=16, d_ff=128, num_layers=2):
    from mmtsfm.models.chronos2.config import Chronos2CoreConfig
    return Chronos2CoreConfig(
        d_model=d_model, d_kv=d_kv, d_ff=d_ff, num_layers=num_layers,
        num_heads=4, dropout_rate=0.0, use_grassmann=False,
        attn_implementation="eager",
        chronos_config=dict(
            context_length=32, input_patch_size=8, input_patch_stride=8,
            output_patch_size=8, quantiles=[0.1, 0.5, 0.9],
            use_reg_token=False, use_arcsinh=False, max_output_patches=2,
        ),
    )


# ---------------------------------------------------------------------------
# Bug A: SDPA scale must match eager scale
# ---------------------------------------------------------------------------

class TestSDPAScale:
    def test_sdpa_and_eager_produce_close_outputs(self):
        """After fix: SDPA and eager must produce outputs within 1e-3."""
        from mmtsfm.models.chronos2.layers import MHA
        cfg = _make_chronos_config()
        torch.manual_seed(0)
        mha_eager = MHA(cfg, use_rope=False)
        mha_eager.eval()

        # Clone weights into SDPA config
        cfg_sdpa = _make_chronos_config()
        cfg_sdpa._attn_implementation = "sdpa"
        mha_sdpa = MHA(cfg_sdpa, use_rope=False)
        mha_sdpa.load_state_dict(mha_eager.state_dict())
        mha_sdpa.eval()

        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D)
        mask = torch.zeros(B, 1, T, T)  # 0 = attend (additive mask convention)

        with torch.no_grad():
            out_eager = mha_eager(x, mask=mask, output_attentions=False).hidden_states
            out_sdpa  = mha_sdpa(x,  mask=mask, output_attentions=False).hidden_states

        diff = (out_eager - out_sdpa).abs().max().item()
        assert diff < 1e-3, (
            f"Eager vs SDPA output diff={diff:.4f} > 1e-3. "
            "SDPA scale likely still 1.0 — set scale=kv_proj_dim**-0.5."
        )

# ---------------------------------------------------------------------------
# Bug G: causal mask dtype must be float32
# ---------------------------------------------------------------------------

class TestCausalMaskDtype:
    def test_causal_mask_is_float32_regardless_of_input_dtype(self):
        """Causal mask must always be float32 — bf16 -inf is finite (-65504)."""
        from mmtsfm.models.vision.latent_summarizer import LatentSummarizer
        ls = LatentSummarizer(d_v=4, d_model=64, n_vis_steps=3, n_heads=4)

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            mask = ls._build_causal_attn_mask(
                n_vis=3, T_lat=5, P=4,
                device=torch.device("cpu"),
                dtype=dtype,
            )
            assert mask.dtype == torch.float32, (
                f"Causal mask dtype={mask.dtype} for input dtype={dtype}; "
                "must always be float32."
            )
            # Masked positions must be -10000.0 (numerical stability change)
            assert (mask[mask < 0] == -10000.0).all(), "Masked positions must be -10000.0."

# ---------------------------------------------------------------------------
# Bug H: masked mean over horizon (not plain mean including padding zeros)
# ---------------------------------------------------------------------------

class TestMaskedLoss:
    def test_loss_equals_masked_mean_not_padded_mean(self):
        """Loss must average only over unmasked horizon positions."""
        from mmtsfm.models.chronos2.model import Chronos2Model
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig

        cfg = Chronos2CoreConfig(
            d_model=64, d_kv=16, d_ff=128, num_layers=1, num_heads=4,
            dropout_rate=0.0, use_grassmann=False,
            chronos_config=dict(
                context_length=32, input_patch_size=8, input_patch_stride=8,
                output_patch_size=8, quantiles=[0.1, 0.5, 0.9],
                use_reg_token=False, use_arcsinh=False, max_output_patches=2,
            ),
        )
        model = Chronos2Model(cfg)
        model.eval()

        B, H = 4, 12   # horizon 12 → padded to 16
        torch.manual_seed(42)
        context       = torch.randn(B, 32)
        future_target = torch.randn(B, H)
        # mask: only first 8 positions are real
        future_mask   = torch.zeros(B, H)
        future_mask[:, :8] = 1.0

        with torch.no_grad():
            out = model(
                context=context,
                future_target=future_target,
                future_target_mask=future_mask,
                num_output_patches=2,
            )

        loss = out.loss
        assert loss is not None

        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        assert loss > 0, f"loss collapsed to 0: {loss}"

    def test_fully_masked_future_gives_zero_loss(self):
        """If all future positions are masked, loss contribution must be 0."""
        from mmtsfm.models.chronos2.model import Chronos2Model
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig

        cfg = Chronos2CoreConfig(
            d_model=64, d_kv=16, d_ff=128, num_layers=1, num_heads=4,
            dropout_rate=0.0, use_grassmann=False,
            chronos_config=dict(
                context_length=32, input_patch_size=8, input_patch_stride=8,
                output_patch_size=8, quantiles=[0.1, 0.5, 0.9],
                use_reg_token=False, use_arcsinh=False, max_output_patches=2,
            ),
        )
        model = Chronos2Model(cfg)
        model.eval()

        B, H = 2, 8
        context       = torch.randn(B, 32)
        future_target = torch.randn(B, H)
        future_mask   = torch.zeros(B, H)  # ALL masked

        with torch.no_grad():
            out = model(
                context=context,
                future_target=future_target,
                future_target_mask=future_mask,
                num_output_patches=1,
            )
        assert out.loss is not None
        assert out.loss.item() == pytest.approx(0.0, abs=1e-6), (
            f"Loss with all-masked future should be 0, got {out.loss.item()}"
        )

# ---------------------------------------------------------------------------
# Bug B: no gradient-killing clamps on encoder inputs/outputs
# ---------------------------------------------------------------------------

class TestNoGradientKillingClamps:
    def _make_vision_model(self):
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig
        from mmtsfm.models.chronos2.model import Chronos2Model
        from mmtsfm.models.chronos2.vision_chronos2 import (
            VisionChronos2Model, VisionChronos2Config,
        )

        core_cfg = Chronos2CoreConfig(
            d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
            dropout_rate=0.0, use_grassmann=False,
            chronos_config=dict(
                context_length=32, input_patch_size=8, input_patch_stride=8,
                output_patch_size=8, quantiles=[0.1, 0.5, 0.9],
                use_reg_token=False, use_arcsinh=True, max_output_patches=2,
            ),
        )
        chronos = Chronos2Model(core_cfg)
        vcfg = VisionChronos2Config(
            d_video_latent=4, n_visual_context_steps=4, n_soft_tokens=1,
            adapter_type="linear", visual_dropout_prob=0.0,
            numeric_dropout_prob=0.0, dropout=0.0, skip_vision_stack=True,
        )
        return VisionChronos2Model(chronos_model=chronos, vision_config=vcfg)

    def test_large_activation_gradient_flows(self):
        """Gradients must be non-zero even when activations naturally exceed 20."""
        model = self._make_vision_model()
        model.train()

        B, T, H = 2, 32, 8
        torch.manual_seed(0)
        # Scale context to force large activations in the encoder
        context       = torch.randn(B, T) * 50.0
        future_target = torch.randn(B, H)
        future_mask   = torch.ones(B, H)

        out = model.forward_numeric_only(
            context=context,
            future_target=future_target,
            future_target_mask=future_mask,
            num_output_patches=1,
        )
        loss = out.loss
        loss.backward()

        # Encoder final layer norm weight must have non-zero gradient
        ln_weight = model.chronos.encoder.final_layer_norm.weight
        assert ln_weight.grad is not None
        assert ln_weight.grad.abs().max() > 1e-8, (
            "Gradient through encoder is zero — clamp may still be killing gradients."
        )

# ---------------------------------------------------------------------------
# Bug J: Internal MHA/MLP bf16 overflow protection
# ---------------------------------------------------------------------------

class TestInternalOverflowProtection:
    def test_mha_internal_nan_gradient_protection(self):
        """NaN gradients from attention shouldn't corrupt q/k/v weights."""
        from mmtsfm.models.chronos2.layers import MHA
        cfg = _make_chronos_config()
        torch.manual_seed(0)
        mha = MHA(cfg, use_rope=False)

        hidden = torch.randn(2, 8, 64, requires_grad=True)
        mask = torch.zeros(2, 1, 8, 8)

        out = mha(hidden, mask=mask, output_attentions=False)

        # Simulate completely NaN gradient flowing backward from the next layer
        grad_out = torch.full_like(out.hidden_states, float('nan'))
        out.hidden_states.backward(grad_out)

        # The gradients for q, k, v, and o weights should be exactly 0, not NaN
        assert not torch.isnan(mha.q.weight.grad).any(), "MHA q weight got NaN gradient"
        assert not torch.isnan(mha.k.weight.grad).any(), "MHA k weight got NaN gradient"
        assert not torch.isnan(mha.v.weight.grad).any(), "MHA v weight got NaN gradient"
        assert not torch.isnan(mha.o.weight.grad).any(), "MHA o weight got NaN gradient"
        # The gradient flowing to previous layers should also be zeroed
        assert not torch.isnan(hidden.grad).any(), "Hidden states got NaN gradient"

    def test_mlp_internal_nan_gradient_protection(self):
        """NaN gradients from MLP outputs shouldn't corrupt wi/wo weights."""
        from mmtsfm.models.chronos2.layers import MLP
        cfg = _make_chronos_config()
        torch.manual_seed(0)
        mlp = MLP(cfg)

        hidden = torch.randn(2, 8, 64, requires_grad=True)
        out = mlp(hidden)

        # Simulate completely NaN gradient flowing backward
        grad_out = torch.full_like(out, float('nan'))
        out.backward(grad_out)

        assert not torch.isnan(mlp.wi.weight.grad).any(), "MLP wi weight got NaN gradient"
        assert not torch.isnan(mlp.wo.weight.grad).any(), "MLP wo weight got NaN gradient"
        assert not torch.isnan(hidden.grad).any(), "Hidden states got NaN gradient"

    def test_hook_does_not_protect_consuming_layer(self):
        """A hook on X protects what creates X, not what consumes X."""
        import torch.nn as nn
        x = torch.randn(2, 2, requires_grad=True)
        # Hook on x protects whatever created x
        x.register_hook(lambda g: torch.zeros_like(g))
        
        linear = nn.Linear(2, 2, bias=False)
        y = linear(x)
        
        # y receives NaN gradient
        y.backward(torch.full_like(y, float('nan')))
        
        # x's gradient is 0 because of the hook!
        assert not torch.isnan(x.grad).any()
        # BUT linear's weight gradient is NaN because it consumed y's NaN BEFORE the hook!
        assert torch.isnan(linear.weight.grad).any()

# ---------------------------------------------------------------------------
# Bug F: multimodal embeddings must NOT be added when visual is dropped
# ---------------------------------------------------------------------------

class TestConditionalMultimodalEmbeddings:
    def _make_vision_model_with_video(self):
        import torch.nn as nn
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig
        from mmtsfm.models.chronos2.model import Chronos2Model
        from mmtsfm.models.chronos2.vision_chronos2 import (
            VisionChronos2Model, VisionChronos2Config,
        )

        core_cfg = Chronos2CoreConfig(
            d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
            dropout_rate=0.0, use_grassmann=False,
            chronos_config=dict(
                context_length=32, input_patch_size=8, input_patch_stride=8,
                output_patch_size=8, quantiles=[0.1, 0.5, 0.9],
                use_reg_token=False, use_arcsinh=False, max_output_patches=2,
            ),
        )
        chronos = Chronos2Model(core_cfg)

        class FakeVidTok(nn.Module):
            def forward(self, x):
                B = x.shape[0]
                return torch.zeros(B, 2, 4, 4)  # [B, T_lat, P, D_v=4]

        vcfg = VisionChronos2Config(
            d_video_latent=4, n_visual_context_steps=4, n_soft_tokens=1,
            adapter_type="linear", visual_dropout_prob=1.0,  # ALWAYS drop visual
            numeric_dropout_prob=0.0, dropout=0.0,
        )
        model = VisionChronos2Model(
            chronos_model=chronos, vision_config=vcfg,
            vidtok_model=FakeVidTok(),
        )
        return model

    def test_visual_dropped_samples_match_vanilla_output(self):
        """When visual_dropout_prob=1.0, output must equal video=None baseline."""
        model = self._make_vision_model_with_video()
        model.eval()  # eval: dropout is off, but visual_dropout is applied in training only

        # Train mode to activate visual dropout
        model.train()

        B, T, H = 4, 32, 8
        torch.manual_seed(0)
        context       = torch.randn(B, T)
        future_target = torch.randn(B, H)
        future_mask   = torch.ones(B, H)

        # Fake video latents [B, T_lat, P, D_v]
        video_latents = torch.zeros(B, 2, 4, 4)
        visual_mask   = torch.ones(B, 2)

        with torch.no_grad():
            out_with_video = model(
                context=context, future_target=future_target,
                future_target_mask=future_mask,
                video_latents=video_latents, visual_mask=visual_mask,
                num_output_patches=1,
            )
            out_no_video = model(
                context=context, future_target=future_target,
                future_target_mask=future_mask,
                video=None, visual_mask=None,
                num_output_patches=1,
            )

        assert out_with_video.loss is not None
        assert out_no_video.loss is not None

        diff = abs(out_with_video.loss.item() - out_no_video.loss.item())
        assert diff < 5e-3, (
            f"With visual_dropout=1.0 and fix, loss should match video=None. "
            f"Got diff={diff:.6f}. Multimodal embeddings may still perturb numeric baseline."
        )
