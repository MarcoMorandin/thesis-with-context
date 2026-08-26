"""Regression tests for the CausalGrassmannMixing offset-mixing branch.

Leonardo job 50574535 (curriculum stage s2a, uk_pv) reported, on EVERY optimizer
step for 6.5 GPU-hours:

    [NaN-grad] nan_enc_params=['3:layer.0.offset_weights(6/6)']

i.e. all six ``offset_weights`` gradients non-finite in one encoder block while
every other parameter in the model stayed finite. That signature identifies the
offset-logit branch precisely: it is a gradient SINK — nothing reads it except
``offset_weights`` / ``modality_pair_bias`` — so a blow-up there cannot show up
anywhere else, and softmax's backward (``y*(ĝ - Σ y·ĝ)``) turns a single +inf
into NaN across *all* entries.

The amplifier was ``g = g_sum / clamp(weight_sum, min=1e-6)``: at a position
whose every offset pair is invalid, ``weight_sum`` is 0, the clamp makes the
denominator 1e-6, and ``dL/dg_sum`` is scaled by 1e6 before feeding the sink's
reduction over ``d_model``. Weights are now renormalised over the valid pairs
*before* accumulation, which is algebraically identical and keeps 1/Σ out of
that backward path.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

from mmtsfm.models.chronos2 import Chronos2CoreConfig
from mmtsfm.models.chronos2.grassmann import CausalGrassmannMixing

OFFSETS = [1, 2, 4, 8, 12, 16]


def _cfg(**over):
    base = dict(
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        use_grassmann=True,
        grassmann_reduced_dim=16,
        grassmann_window_offsets=OFFSETS,
        grassmann_modality_pair_bias=True,
    )
    base.update(over)
    return Chronos2CoreConfig(**base)


def _mixer(seed=0, **over):
    torch.manual_seed(seed)
    mix = CausalGrassmannMixing(_cfg(**over))
    mix.offset_weights.data.copy_(
        torch.tensor([1.38, 1.18, 0.69, 0.79, 0.96, 0.81])  # uk_pv_s2a/last.ckpt
    )
    mix.modality_pair_bias.data.zero_()
    mix.eval()
    return mix


def _inputs(B=3, L=40, D=64, left_pad=0, visual_tail=2):
    torch.manual_seed(1)
    h = torch.randn(B, L, D)
    attn = torch.zeros(B, 1, L, L)
    if left_pad:
        attn[:, :, :, :left_pad] = -1e9
    pos = torch.arange(L).unsqueeze(0).expand(B, L)
    mod = torch.zeros(B, L, dtype=torch.long)
    if visual_tail:
        mod[:, -visual_tail:] = 1
    return h, attn, pos, mod


def _layer_internals(mix, h, pos):
    """Replicate forward()'s pre-mixing steps: layer norm, W_red, RoPE."""
    from mmtsfm.models.chronos2.layers import RoPE

    h_ln = mix.layer_norm(h)
    z = mix.W_red(h_ln)
    cos, sin = mix.rope_embed(z.unsqueeze(1), pos)
    z_rope, _ = RoPE.apply_rotary_pos_emb(
        z.unsqueeze(1), z.unsqueeze(1), cos, sin, unsqueeze_dim=1
    )
    return h_ln, z_rope.squeeze(1)


def _reference_g(mix, h_ln, z, valid_mask, weights, valid_offsets):
    """Explicit Σ(v·w·g)/Σ(v·w) — the accumulate-then-divide form, in fp64."""
    B, L, D = h_ln.shape
    num = torch.zeros(B, L, D, dtype=torch.float64)
    den = torch.zeros(B, L, 1, dtype=torch.float64)
    for i, delta in enumerate(valid_offsets):
        L_eff = L - delta
        plucker = mix._compute_plucker(z[:, :L_eff], z[:, delta:])
        g = mix.W_plu(plucker).double()
        v = (valid_mask[:, :L_eff] & valid_mask[:, delta:]).unsqueeze(-1).double()
        w = weights[:, delta:, i : i + 1].double()
        num[:, delta:] += g * v * w
        den[:, delta:] += v * w
    return num / den.clamp(min=1e-6)


def _accumulate_then_project(mix, z, mixing, valid_offsets, B, L):
    """The shipped path: accumulate in Plücker space, then one W_plu."""
    p_sum = torch.zeros(B, L, mix.plucker_dim, dtype=z.dtype)
    z_i, z_j = mix._pair_components(z)
    for i, delta in enumerate(valid_offsets):
        mix._accumulate_offset(z_i, z_j, delta, mixing[:, :, i : i + 1], p_sum)
    g = mix.W_plu(p_sum)
    return g * (mixing.sum(dim=-1, keepdim=True) > 0).to(g.dtype)


def _project_then_accumulate(mix, z, mixing, valid_offsets, B, L):
    """The pre-refactor path: one W_plu PER offset, accumulate in d_model space."""
    g_sum = torch.zeros(B, L, mix.W_plu.out_features, dtype=torch.float64)
    for i, delta in enumerate(valid_offsets):
        L_eff = L - delta
        plucker = mix._compute_plucker(z[:, :L_eff], z[:, delta:])
        g = mix.W_plu(plucker).double()
        g_sum[:, delta:, :] += g * mixing[:, delta:, i : i + 1].double()
    return g_sum


def _reference_forward(mix, h, attn, pos, mod):
    """Whole layer, computed with the PRE-REFACTOR per-offset projection.

    Must be compared against the real ``mix(...)`` — a helper that reimplements
    the shipped masking would pass whether or not the shipped code does it.
    Requires ``mix.eval()`` and dropout_rate=0, both true of ``_mixer()``.
    """
    B, L, _ = h.shape
    h_ln, z = _layer_internals(mix, h, pos)
    valid_mask = attn[:, 0, 0, :] > -1.0
    valid_offsets = [d for d in OFFSETS if d < L]
    mixing, _ = mix._offset_weights_for(valid_offsets, valid_mask, mod, L, h.device)
    g = _project_then_accumulate(mix, z, mixing, valid_offsets, B, L).to(h_ln.dtype)
    alpha = torch.sigmoid(mix.W_gate(torch.cat([h_ln, g], dim=-1)))
    return h + (alpha * h_ln + (1 - alpha) * g)


class TestNormalisationEquivalence:
    """Weight-first renormalisation must equal the old accumulate-then-divide."""

    @pytest.mark.parametrize("left_pad", [0, 5, 20])
    def test_matches_accumulate_then_divide(self, left_pad):
        mix = _mixer()
        h, attn, pos, mod = _inputs(left_pad=left_pad)

        out = mix(h, attn, pos, output_attentions=True, modality_mask=mod)
        weights = out.attn_weights["offset_weights"]  # [B, L, n] softmax, fp32

        B, L, _ = h.shape
        h_ln, z = _layer_internals(mix, h, pos)
        valid_mask = attn[:, 0, 0, :] > -1.0
        valid_offsets = [d for d in OFFSETS if d < L]

        g_ref = _reference_g(
            mix, h_ln, z, valid_mask, weights.expand(B, L, -1), valid_offsets
        )
        mixing, _ = mix._offset_weights_for(valid_offsets, valid_mask, mod, L, h.device)
        g_new = _accumulate_then_project(mix, z, mixing, valid_offsets, B, L)

        assert torch.allclose(g_new.double(), g_ref, atol=1e-6), (
            (g_new.double() - g_ref).abs().max()
        )

    def test_all_invalid_position_contributes_zero(self):
        """A position with no valid pair must still produce g = 0 exactly."""
        mix = _mixer()
        h, attn, pos, mod = _inputs(L=40, left_pad=0)
        attn[:, :, :, :] = -1e9  # nothing is valid anywhere
        valid_mask = attn[:, 0, 0, :] > -1.0
        valid_offsets = [d for d in OFFSETS if d < h.shape[1]]
        mixing, _ = mix._offset_weights_for(
            valid_offsets, valid_mask, mod, h.shape[1], h.device
        )
        assert torch.equal(mixing, torch.zeros_like(mixing))


class TestOffsetWeightGradientIsFinite:
    """The failure mode from job 50574535, reduced to a unit test."""

    # 1e34 and 1e36 are the discriminating cases: with the old
    # accumulate-then-divide the clamped 1e-6 denominator scaled dL/dg_sum by
    # 1e6, so the sink overflowed there — four orders of magnitude before fp32
    # itself runs out (~3.4e38). Both now stay finite.
    @pytest.mark.parametrize("upstream", [1.0, 1e20, 1e34, 1e36])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_padded_positions_do_not_amplify(self, upstream, dtype):
        """Positions whose offset pairs are ALL invalid used to divide by a
        clamped 1e-6, scaling dL/dg_sum by 1e6 into a d_model-wide reduction
        that only ``offset_weights`` and ``modality_pair_bias`` see."""
        mix = _mixer()
        # Left-pad past the largest offset so the first valid positions have no
        # valid partner at any delta — the exact all-invalid case.
        h, attn, pos, mod = _inputs(B=3, L=40, left_pad=20)
        h.requires_grad_(True)

        with torch.autocast("cpu", dtype=dtype, enabled=dtype is torch.bfloat16):
            out = mix(h, attn, pos, modality_mask=mod)
        (out.hidden_states.float().sum() * upstream).backward()

        assert torch.isfinite(mix.offset_weights.grad).all(), (
            f"offset_weights grad non-finite at upstream={upstream:g} "
            f"dtype={dtype}: {mix.offset_weights.grad}"
        )
        assert torch.isfinite(mix.modality_pair_bias.grad).all()

    def test_no_inf_logits_when_offsets_out_of_range(self):
        """Short sequences drop out-of-range offsets by INDEXING, not by
        substituting -inf into the softmax input."""
        mix = _mixer()
        h, attn, pos, mod = _inputs(B=2, L=6)
        h.requires_grad_(True)
        out = mix(h, attn, pos, output_attentions=True, modality_mask=mod)
        w = out.attn_weights["offset_weights"]
        assert torch.isfinite(w).all()
        assert w.shape[-1] == len([d for d in OFFSETS if d < 6])
        out.hidden_states.sum().backward()
        assert torch.isfinite(mix.offset_weights.grad).all()
        # Out-of-range offsets must receive exactly zero gradient.
        assert torch.equal(
            mix.offset_weights.grad[3:], torch.zeros_like(mix.offset_weights.grad[3:])
        )


class TestSingleProjectionEquivalence:
    """One W_plu over the accumulated Plücker sum == one W_plu per offset.

    W_plu is affine and `mixing` sums to 1 at every position with a valid pair,
    so Σ_δ w_δ·(A·p_δ + b) = W_plu(Σ_δ w_δ·p_δ) exactly. Collapsing the six
    projections into one removes the layer's dominant cost (C(r,2)·d_model per
    offset) and is what moves the FLOP crossover against self-attention below the
    ~44 tokens this protocol actually produces. These tests pin the equivalence
    so a later edit cannot quietly reintroduce the per-offset form, nor drop the
    all-invalid masking that keeps the bias out of empty positions.
    """

    @staticmethod
    def _setup(left_pad):
        mix = _mixer()
        # A ZERO bias would let the two forms agree even WITHOUT the all-invalid
        # mask, so force a large one — that is the discriminating case. Note
        # model.py zero-inits this bias, but it trains away from zero.
        mix.W_plu.bias.data.fill_(3.0)
        return (mix, *_inputs(left_pad=left_pad))

    @staticmethod
    def _internals(mix, h, attn, pos, mod):
        B, L, _ = h.shape
        _, z = _layer_internals(mix, h, pos)
        valid_mask = attn[:, 0, 0, :] > -1.0
        valid_offsets = [d for d in OFFSETS if d < L]
        mixing, _ = mix._offset_weights_for(valid_offsets, valid_mask, mod, L, h.device)
        return z, mixing, valid_offsets, B, L

    @pytest.mark.parametrize("left_pad", [0, 5, 20])
    def test_forward_matches_per_offset_projection(self, left_pad):
        """Drives the real forward, not a reimplementation of it."""
        mix, h, attn, pos, mod = self._setup(left_pad)

        new = mix(h, attn, pos, modality_mask=mod).hidden_states
        old = _reference_forward(mix, h, attn, pos, mod)

        assert torch.allclose(new, old, atol=1e-5), (new - old).abs().max()

    def test_bias_does_not_leak_into_all_invalid_positions(self):
        """Positions with no valid pair at any offset must contribute g = 0.

        `mixing` is all-zero there, so the per-offset form contributed 0. A bare
        W_plu(0) contributes the bias instead — this is what the `any_valid`
        re-zeroing in forward exists for. left_pad=20 with max offset 16
        guarantees such a position: index 20 is itself valid but has no valid
        partner at any δ. With g = 0 the gated fusion degenerates to
        α·h_ln + residual, which is what this asserts.
        """
        mix, h, attn, pos, mod = self._setup(left_pad=20)
        h_ln, _ = _layer_internals(mix, h, pos)
        zero_g = torch.zeros_like(h_ln)
        alpha = torch.sigmoid(mix.W_gate(torch.cat([h_ln, zero_g], dim=-1)))
        expected = h + alpha * h_ln

        out = mix(h, attn, pos, modality_mask=mod).hidden_states

        assert torch.allclose(out[:, 20], expected[:, 20], atol=1e-6), (
            "W_plu bias leaked into the all-invalid position 20, max diff "
            f"{(out[:, 20] - expected[:, 20]).abs().max():.4e}"
        )

    @pytest.mark.parametrize("left_pad", [0, 20])
    def test_gradients_match_per_offset_projection(self, left_pad):
        """Equivalence must hold in backward too — training is what consumes it."""
        mix, h, attn, pos, mod = self._setup(left_pad)
        z, mixing, valid_offsets, B, L = self._internals(mix, h, attn, pos, mod)
        # `mixing` carries its own graph back to offset_weights; both paths below
        # would try to traverse that same graph. Its gradients are covered by
        # TestOffsetWeightGradientIsFinite — here only W_plu and z are at stake.
        mixing = mixing.detach()

        z_new = z.detach().clone().requires_grad_(True)
        out_new = _accumulate_then_project(mix, z_new, mixing, valid_offsets, B, L)
        out_new.sum().backward()
        g_w_new = mix.W_plu.weight.grad.double().clone()
        g_b_new = mix.W_plu.bias.grad.double().clone()
        mix.zero_grad(set_to_none=True)

        z_old = z.detach().clone().requires_grad_(True)
        out_old = _project_then_accumulate(mix, z_old, mixing, valid_offsets, B, L)
        out_old.sum().backward()

        assert torch.allclose(z_new.grad, z_old.grad, atol=1e-5), (
            (z_new.grad - z_old.grad).abs().max()
        )
        assert torch.allclose(g_w_new, mix.W_plu.weight.grad.double(), atol=1e-5), (
            (g_w_new - mix.W_plu.weight.grad.double()).abs().max()
        )
        assert torch.allclose(g_b_new, mix.W_plu.bias.grad.double(), atol=1e-5)

    def test_w_plu_is_applied_exactly_once_per_forward(self):
        """The point of the refactor: six offsets, one projection."""
        mix = _mixer()
        h, attn, pos, mod = _inputs()
        calls = []
        real = mix.W_plu.forward

        def _spy(x, _real=real, _calls=calls):
            _calls.append(tuple(x.shape))
            return _real(x)

        mix.W_plu.forward = _spy
        mix(h, attn, pos, modality_mask=mod)

        n_offsets = len([d for d in OFFSETS if d < h.shape[1]])
        assert n_offsets == 6, "test setup no longer exercises multiple offsets"
        assert len(calls) == 1, f"W_plu applied {len(calls)}x - expected 1: {calls}"
        assert calls[0][-1] == mix.plucker_dim
