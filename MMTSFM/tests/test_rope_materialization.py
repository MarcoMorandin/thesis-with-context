"""Regression test: RoPE must survive the meta + to_empty() materialisation that
``from_pretrained`` uses.

``inv_freq`` used to be ``register_buffer(..., persistent=False)``. That made it
absent from the checkpoint, so on the from_pretrained path — build on ``meta``,
materialise with ``to_empty()``, then ``load_state_dict`` — nothing ever wrote a
correct value into it:

  * ``to_empty()`` allocates every buffer, persistent or not, with uninitialised
    storage;
  * the checkpoint has no ``inv_freq`` key, so ``load_state_dict`` skips it and
    does not even report it missing (the curriculum logs showed
    ``Warm start: loaded (missing=0)`` while the buffer was garbage);
  * ``_init_weights`` covers ``CausalGrassmannMixing``'s Linears and parameters,
    not the ``RoPE`` it owns.

Both garbage regimes are damaging and only one is loud: NaN/Inf makes the
post-RoPE activations non-finite (which the per-block guards in model.py:75-100
then hide from the loss, freezing training silently), while finite garbage such
as zeros collapses RoPE to the identity and removes all positional information
without any warning.

Curriculum jobs 52364214 / 52409960 / 52410223 / 52411561 alternated clean and
NaN on identical code, config and seed — the allocation-dependent signature of
exactly this.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

from mmtsfm.models.chronos2.layers import RoPE

DIM = 32
BASE = 10000.0


def _materialised_like_from_pretrained() -> RoPE:
    """Build on `meta` and materialise, exactly as from_pretrained does."""
    with torch.device("meta"):
        m = RoPE(dim=DIM, base=BASE)
    m.to_empty(device="cpu")
    return m


class TestInvFreqSurvivesMaterialisation:
    def test_not_a_buffer(self):
        """A buffer is what made this unrecoverable — it must not be one."""
        m = RoPE(dim=DIM, base=BASE)
        assert "inv_freq" not in dict(m.named_buffers())
        assert "inv_freq" not in m.state_dict()

    def test_correct_after_meta_to_empty(self):
        reference = RoPE(dim=DIM, base=BASE)
        m = _materialised_like_from_pretrained()
        assert torch.allclose(m.inv_freq, reference.inv_freq)

    def test_expected_values(self):
        m = _materialised_like_from_pretrained()
        expected = 1.0 / (
            BASE ** (torch.arange(0, DIM, 2, dtype=torch.int64).float() / DIM)
        )
        assert torch.allclose(m.inv_freq, expected)
        assert float(m.inv_freq[0]) == pytest.approx(1.0)

    def test_forward_matches_reference_after_materialisation(self):
        """The failure was in the forward, so assert on cos/sin, not just the tensor."""
        reference = RoPE(dim=DIM, base=BASE)
        m = _materialised_like_from_pretrained()

        B, L = 2, 24
        x = torch.randn(B, 1, L, DIM)
        pos = torch.arange(L).unsqueeze(0).expand(B, L)

        cos_ref, sin_ref = reference(x, pos)
        cos_new, sin_new = m(x, pos)
        assert torch.allclose(cos_ref, cos_new)
        assert torch.allclose(sin_ref, sin_new)
        assert torch.isfinite(cos_new).all() and torch.isfinite(sin_new).all()

    def test_rope_is_not_the_identity(self):
        """Zero-garbage inv_freq gives cos=1, sin=0 — positionally inert and
        completely silent. Assert RoPE actually rotates."""
        m = _materialised_like_from_pretrained()
        B, L = 1, 16
        x = torch.randn(B, 1, L, DIM)
        pos = torch.arange(L).unsqueeze(0).expand(B, L)

        cos, sin = m(x, pos)
        assert not torch.allclose(cos, torch.ones_like(cos)), (
            "cos is all ones — inv_freq collapsed to zero and RoPE is inert"
        )
        assert sin.abs().max() > 1e-3, "sin is all zeros — RoPE is inert"

        q, _ = RoPE.apply_rotary_pos_emb(x, x, cos, sin, unsqueeze_dim=1)
        assert not torch.allclose(q, x), "apply_rotary_pos_emb is a no-op"

    def test_positions_are_distinguished(self):
        """Different positions must receive different rotations."""
        m = _materialised_like_from_pretrained()
        pos = torch.arange(8).unsqueeze(0)
        x = torch.randn(1, 1, 8, DIM)
        cos, _ = m(x, pos)
        assert not torch.allclose(cos[0, 0], cos[0, 5])


class TestGrassmannRoPEAfterMaterialisation:
    """The blocks that actually broke: each CausalGrassmannMixing owns a RoPE."""

    def test_grassmann_forward_finite_after_materialisation(self):
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig
        from mmtsfm.models.chronos2.grassmann import CausalGrassmannMixing

        cfg = Chronos2CoreConfig(
            d_model=64,
            d_kv=16,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            dropout_rate=0.0,
            use_grassmann=True,
            grassmann_reduced_dim=DIM,
            grassmann_window_offsets=[1, 2, 4],
            grassmann_modality_pair_bias=True,
        )
        with torch.device("meta"):
            mix = CausalGrassmannMixing(cfg)
        mix.to_empty(device="cpu")
        # Materialised parameters are garbage by definition; give them the values
        # _init_weights would. The point of the test is the DERIVED inv_freq.
        for p in mix.parameters():
            torch.nn.init.normal_(p, std=0.02)
        mix.offset_weights.data.fill_(1.0)
        mix.modality_pair_bias.data.zero_()

        B, L, D = 2, 20, 64
        h = torch.randn(B, L, D, requires_grad=True)
        attn = torch.zeros(B, 1, L, L)
        pos = torch.arange(L).unsqueeze(0).expand(B, L)
        mod = torch.zeros(B, L, dtype=torch.long)
        mod[:, -2:] = 1

        out = mix(h, attn, pos, modality_mask=mod)
        assert torch.isfinite(out.hidden_states).all()

        out.hidden_states.float().pow(2).mean().backward()
        for n, p in mix.named_parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), (
                f"{n} has non-finite grad after meta+to_empty materialisation"
            )
