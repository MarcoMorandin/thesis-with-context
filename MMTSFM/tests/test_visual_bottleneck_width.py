"""Widening the visual bottleneck: `n_soft_tokens` in the INTERLEAVED path.

Vision helps aggregate NMAE (13/14 plants, t=7.1) and does nothing on ramps
(10/14, t=0.9). A model-free probe on the raw crops traced that to pooling: a
1x1 pooled feature is neutral-to-NEGATIVE on ramp steps while an NxN grid helps
monotonically (see .scratch/ramp-gap/issues/13-widen-the-visual-bottleneck.md).

But `n_soft_tokens` was inert in interleaved fusion — `CrossModalAdapter` was
built only for `fusion_mode == "late"`, and the interleaved branch fed the
LatentSummarizer's one-token-per-step output straight into `interleave_sequences`.
Every ramp number on the map was produced at an effective N=1 that no config
could have changed.

These tests pin the two halves of the fix:
  * N=1 stays bit-identical, so every existing checkpoint and result stands;
  * N>1 actually puts N visual tokens per refined step, with the attention mask,
    position IDs and modality mask all following.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_vision_chronos2 import _make_fake_video_encoder  # noqa: E402

import pytest
import torch

from mmtsfm.models.chronos2.vision_chronos2 import (
    build_interleaved_position_ids,
    interleave_sequences,
)

B, T_CTX, N_VIS, D = 2, 7, 3, 8
T_M = T_CTX - N_VIS


def _ts():
    torch.manual_seed(0)
    return torch.randn(B, T_CTX, D)


def _vis(n_soft):
    torch.manual_seed(1)
    return torch.randn(B, N_VIS, n_soft, D)


class TestBackwardCompatibleAtNSoftOne:
    """N=1 must reproduce the original pairwise interleave exactly."""

    def test_three_dim_input_matches_four_dim_with_n_soft_one(self):
        ts = _ts()
        v3 = torch.randn(B, N_VIS, D)
        a, ma = interleave_sequences(ts, v3, N_VIS)
        b, mb = interleave_sequences(ts, v3.unsqueeze(2), N_VIS)
        assert torch.equal(a, b)
        assert torch.equal(ma, mb)

    def test_reproduces_the_original_pairwise_layout(self):
        """The pre-change implementation, written out literally."""
        ts, vis = _ts(), torch.randn(B, N_VIS, D)
        pairs = torch.stack([ts[:, T_M:, :], vis], dim=2)
        expected = torch.cat([ts[:, :T_M, :], pairs.reshape(B, 2 * N_VIS, D)], dim=1)

        got, mask = interleave_sequences(ts, vis, N_VIS)

        assert got.shape == (B, T_CTX + N_VIS, D)
        assert torch.equal(got, expected)
        exp_pos = T_M + 1 + torch.arange(N_VIS) * 2
        assert torch.equal(torch.nonzero(mask[0]).flatten(), exp_pos)

    def test_position_ids_unchanged_at_n_soft_one(self):
        old = torch.cat(
            [
                torch.arange(T_M),
                torch.stack([torch.arange(T_M, T_M + N_VIS)] * 2, dim=1).reshape(
                    2 * N_VIS
                ),
                torch.arange(T_M + N_VIS, T_M + N_VIS + 4),
            ]
        ).unsqueeze(0)
        new = build_interleaved_position_ids(T_M, N_VIS, 4, torch.device("cpu"))
        assert torch.equal(old, new)


class TestWidenedBottleneck:
    @pytest.mark.parametrize("n_soft", [1, 2, 4, 16])
    def test_each_refined_step_gets_n_soft_visual_tokens(self, n_soft):
        ts, vis = _ts(), _vis(n_soft)
        out, modality = interleave_sequences(ts, vis, N_VIS)

        assert out.shape == (B, T_CTX + N_VIS * n_soft, D)
        # macro region is untouched TS
        assert torch.equal(out[:, :T_M], ts[:, :T_M])
        # every block: one TS token then exactly n_soft visual tokens
        for k in range(N_VIS):
            s = T_M + k * (1 + n_soft)
            assert torch.equal(out[:, s], ts[:, T_M + k]), f"block {k} TS token"
            assert torch.equal(out[:, s + 1 : s + 1 + n_soft], vis[:, k]), f"block {k}"
        assert int(modality.sum(1)[0]) == N_VIS * n_soft
        assert torch.equal(modality[:, :T_M], torch.zeros(B, T_M, dtype=torch.long))

    @pytest.mark.parametrize("n_soft", [1, 2, 4, 16])
    def test_modality_mask_marks_exactly_the_visual_tokens(self, n_soft):
        """A visual token must never be labelled TS, or the Grassmann
        modality-pair bias and any modality-conditioned mixing silently learn on
        the wrong pairs."""
        ts, vis = _ts(), _vis(n_soft)
        out, modality = interleave_sequences(ts, vis, N_VIS)
        for pos in range(out.shape[1]):
            if pos < T_M:
                assert modality[0, pos] == 0
                continue
            off = (pos - T_M) % (1 + n_soft)
            assert modality[0, pos] == (0 if off == 0 else 1), f"pos {pos}"

    @pytest.mark.parametrize("n_soft", [1, 2, 4, 16])
    def test_position_ids_are_co_temporal_within_each_block(self, n_soft):
        """A step and all its visual tokens share one position ID, so RoPE sees
        them as simultaneous and the N tokens carry no spurious ordering."""
        T_fut = 4
        pos = build_interleaved_position_ids(
            T_M, N_VIS, T_fut, torch.device("cpu"), n_soft=n_soft
        )[0]

        assert pos.shape == (T_M + N_VIS * (1 + n_soft) + T_fut,)
        assert torch.equal(pos[:T_M], torch.arange(T_M))
        for k in range(N_VIS):
            s = T_M + k * (1 + n_soft)
            block = pos[s : s + 1 + n_soft]
            assert torch.equal(block, torch.full((1 + n_soft,), T_M + k)), f"block {k}"
        tail = pos[T_M + N_VIS * (1 + n_soft) :]
        assert torch.equal(tail, torch.arange(T_M + N_VIS, T_M + N_VIS + T_fut))
        # positions must never run backwards
        assert torch.all(pos[1:] >= pos[:-1])

    @pytest.mark.parametrize("n_soft", [2, 4])
    def test_visual_tokens_are_distinct(self, n_soft):
        """Guards the whole point: N tokens that are copies of each other carry
        no more information than one, and would make the ablation a no-op."""
        ts, vis = _ts(), _vis(n_soft)
        out, _ = interleave_sequences(ts, vis, N_VIS)
        s = T_M + 1
        first, second = out[:, s], out[:, s + 1]
        assert not torch.allclose(first, second)


class TestAttentionMaskFollowsTheBlocks:
    """The mask built in the interleaved branch must line up with the tokens.

    Mirrors the shipped construction; a mismatch here reintroduces the bug
    ticket 03 fixed, where padded context patches were presented as valid.
    """

    @pytest.mark.parametrize("n_soft", [1, 2, 4])
    def test_mask_marks_ts_padding_and_keeps_visual_partners_valid(self, n_soft):
        ctx_mask = torch.ones(B, T_CTX)
        ctx_mask[:, : T_M + 1] = 0.0  # pad the macro region AND the first refined step

        refine_mask = torch.cat(
            [ctx_mask[:, T_M:, None], torch.ones(B, N_VIS, n_soft)], dim=2
        ).reshape(B, N_VIS * (1 + n_soft))
        full = torch.cat([ctx_mask[:, :T_M], refine_mask], dim=1)

        ts, vis = _ts(), _vis(n_soft)
        out, modality = interleave_sequences(ts, vis, N_VIS)
        assert full.shape[1] == out.shape[1]

        # every visual token valid; TS tokens carry their own mask
        assert torch.all(full[:, modality[0] == 1] == 1.0)
        for k in range(N_VIS):
            s = T_M + k * (1 + n_soft)
            assert full[0, s] == ctx_mask[0, T_M + k], f"block {k} TS mask"
        assert full[0, T_M] == 0.0  # the padded refined step stayed masked


class TestInterleavedForwardAtWidenedBottleneck:
    """End-to-end: the widened bottleneck must actually run, and must matter.

    Unit tests above pin the token layout. These pin the two things only a full
    forward can show: that the encoder accepts the longer sequence at all, and
    that N>1 is not silently equivalent to N=1 (which is what the code did
    before the fix — `CrossModalAdapter` was never built for interleaved fusion,
    so `n_soft_tokens` changed nothing).
    """

    @staticmethod
    def _module(n_soft):
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
        torch.manual_seed(0)
        mod = VisionChronos2LightningModule(
            chronos_core_cfg=core_cfg,
            vision_cfg={
                "n_visual_context_steps": 2,
                "n_soft_tokens": n_soft,
                "fusion_mode": "interleaved",
                "skip_vision_stack": False,
                "visual_dropout_prob": 0.0,
                "numeric_dropout_prob": 0.0,
                "dropout": 0.0,
            },
            lr=1e-3,
            warmup_steps=10,
            grassmann_warmup_steps=10,
            video_encoder=_make_fake_video_encoder(d_v=4, t_lat=5, h_lat=4, w_lat=4),
        )
        from unittest.mock import Mock

        mod.trainer = Mock()
        mod.trainer.is_global_zero = True
        mod.trainer.estimated_stepping_batches = 100
        mod.eval()
        return mod

    @staticmethod
    def _batch():
        torch.manual_seed(7)
        return {
            "Y": torch.randn(2, 1, 16, 1),
            "Y_future": torch.randn(2, 1, 4, 1),
            "X_cov": torch.randn(2, 1, 20, 1),
            "V": torch.randn(2, 1, 4, 3, 32, 32),
            "mask_target": torch.ones(2, 1, 16, 1),
            "mask_future": torch.ones(2, 1, 4, 1),
            "mask_visual": torch.ones(2, 1, 4),
            "daylight_future": torch.ones(2, 1, 4, 1),
            "site_id": torch.zeros(2, 1, dtype=torch.long),
        }

    @pytest.mark.parametrize("n_soft", [1, 4, 16])
    def test_forward_runs_and_loss_is_finite(self, n_soft):
        loss = self._module(n_soft).training_step(self._batch(), batch_idx=0)
        assert torch.isfinite(loss), f"non-finite loss at n_soft={n_soft}"

    def _encoder_seq(self, n_soft):
        """(sequence length, visual-token count) as the encoder actually sees it."""
        mod = self._module(n_soft)
        seen = {}
        real = mod.model.chronos.encoder.forward

        def spy(*a, **kw):
            emb = kw.get("inputs_embeds", a[0] if a else None)
            seen["len"] = emb.shape[1]
            seen["n_vis_tok"] = int(kw["modality_mask"][0].sum())
            return real(*a, **kw)

        mod.model.chronos.encoder.forward = spy
        mod.training_step(self._batch(), batch_idx=0)
        return seen["len"], seen["n_vis_tok"]

    @pytest.mark.parametrize("n_soft", [4, 16])
    def test_encoder_sees_n_vis_times_n_soft_extra_tokens(self, n_soft):
        """Sequence length must grow by exactly n_vis*(N-1) over the N=1 layout —
        direct evidence the bottleneck widened rather than the config being
        ignored. Asserted as an increment so it does not encode the patch
        geometry, which is a separate concern with its own tests."""
        n_vis = 2  # n_visual_context_steps in _module
        base_len, base_vis = self._encoder_seq(1)
        wide_len, wide_vis = self._encoder_seq(n_soft)

        assert base_vis == n_vis, base_vis
        assert wide_vis == n_vis * n_soft, wide_vis
        assert wide_len - base_len == n_vis * (n_soft - 1), (wide_len, base_len)

    def test_widening_changes_the_forecast(self):
        """N=16 must not silently reproduce N=1. Before the fix it did."""
        b = self._batch()
        one = self._module(1).training_step(b, batch_idx=0)
        many = self._module(16).training_step(b, batch_idx=0)
        assert not torch.isclose(one, many, atol=1e-6), (
            "n_soft_tokens had no effect on the interleaved forward — the "
            "adapter is not wired in"
        )

    def test_n_soft_one_adds_no_parameters(self):
        """Existing curriculum checkpoints must still load. At N=1 the adapter
        is not built, so the state_dict keys are unchanged."""
        keys = set(self._module(1).state_dict())
        assert not [k for k in keys if "cross_modal_adapter" in k]
        wide = set(self._module(16).state_dict())
        assert [k for k in wide if "cross_modal_adapter" in k]
        assert keys.issubset(wide)
