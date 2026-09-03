"""A29 — sub-resolution inside the LatentSummarizer.

S2b interleaves ONE pooled token per visual step. A13 tried to widen that with
`n_soft_tokens` and found nothing, but the null was guaranteed by construction:
`CrossModalAdapter` sits DOWNSTREAM of the summarizer bottleneck, so N copies of
one pooled d_model vector carry exactly as much information as one. No amount of
fan-out can move signal through a bottleneck that already threw it away.

A29 widens the bottleneck itself. Each visual TS step is split into
`n_time_slices` temporal slices x `spatial_grid**2` spatial blocks, and each of
the resulting `n_sub` queries is masked to its own block — so the extra tokens
CANNOT collapse onto a shared global average the way free queries can.

These tests pin three things:
  * n_sub == 1 is bit-identical, so the four published arms and their
    checkpoints are untouched;
  * the masking is real — a sub-token sees its own block and nothing else, and
    causality survives per (t_vis, sub) pair, not just per step;
  * the interleaved forward actually carries n_vis * n_sub visual tokens, and
    the widening changes the forecast.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_vision_chronos2 import _make_fake_video_encoder  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from mmtsfm.models.vision.latent_summarizer import LatentSummarizer  # noqa: E402

B, T_LAT, SIDE, D_V, D_MODEL = 2, 5, 4, 4, 16
P = SIDE * SIDE
N_VIS = 2


def _summarizer(n_time_slices=1, spatial_grid=1, seed=0):
    torch.manual_seed(seed)
    m = LatentSummarizer(
        d_v=D_V,
        d_model=D_MODEL,
        n_vis_steps=N_VIS,
        n_heads=2,
        dropout=0.0,
        n_time_slices=n_time_slices,
        spatial_grid=spatial_grid,
    )
    m.eval()
    return m


def _video(seed=3):
    torch.manual_seed(seed)
    return torch.randn(B, T_LAT, P, D_V)


def _block_of(patch, g):
    """Reference block map, written independently of the implementation."""
    r, c = patch // SIDE, patch % SIDE
    return ((r * g) // SIDE) * g + ((c * g) // SIDE)


def _frame_limit(t_vis):
    """Frames visible to query t_vis under the uniform-spacing causal rule."""
    return min(-(-(t_vis + 1) * T_LAT // N_VIS), T_LAT)


class TestBackwardCompatibleAtNSubOne:
    """Defaults must reproduce the pre-A29 module exactly — every published
    number was produced by it and every checkpoint has to keep loading."""

    def test_default_kwargs_are_n_sub_one(self):
        m = _summarizer()
        assert (m.n_time_slices, m.spatial_grid, m.n_sub) == (1, 1, 1)

    def test_latent_queries_shape_unchanged(self):
        assert _summarizer().latent_queries.shape == (1, N_VIS, D_MODEL)

    def test_state_dict_keys_unchanged_when_widened(self):
        """No new modules — the widening is masking plus a longer query bank,
        so nothing but `latent_queries` can even change shape."""
        base = set(_summarizer().state_dict())
        wide = set(_summarizer(4, 4).state_dict())
        assert base == wide

    def test_output_is_bit_identical_to_the_pre_a29_path(self):
        v = _video()
        a = _summarizer(seed=11)(video_tokens=v, T_ts=T_LAT)
        b = _summarizer(1, 1, seed=11)(video_tokens=v, T_ts=T_LAT)
        assert a.shape == (B, T_LAT, D_MODEL)
        assert torch.equal(a, b)

    def test_sub_mask_is_all_zero_at_n_sub_one(self):
        sub, spat = _summarizer()._build_sub_attn_mask(T_LAT, P, torch.device("cpu"))
        assert torch.equal(sub, torch.zeros_like(sub))
        assert torch.equal(spat, torch.zeros_like(spat))


class TestSubMaskGeometry:
    @pytest.mark.parametrize("n_t,g", [(2, 2), (4, 4), (1, 4), (5, 1)])
    def test_every_sub_query_has_a_non_empty_and_distinct_key_set(self, n_t, g):
        """Two sub-queries with the same key set are two copies of one token —
        exactly the failure mode A13 hit, and the thing A29 must not repeat."""
        m = _summarizer(n_t, g)
        sub, _ = m._build_sub_attn_mask(T_LAT, P, torch.device("cpu"))
        assert sub.shape == (m.n_sub, T_LAT * P)

        seen = set()
        for i in range(m.n_sub):
            allowed = (sub[i] == 0).nonzero().flatten()
            assert allowed.numel() > 0, f"sub {i} sees nothing"
            key = tuple(allowed.tolist())
            assert key not in seen, f"sub {i} duplicates an earlier key set"
            seen.add(key)

    @pytest.mark.parametrize("n_t,g", [(2, 2), (4, 4)])
    def test_blocks_partition_the_key_axis(self, n_t, g):
        """Union covers every (frame, patch); the blocks do not overlap. A gap
        would silently drop part of the field, an overlap would double-count."""
        m = _summarizer(n_t, g)
        sub, _ = m._build_sub_attn_mask(T_LAT, P, torch.device("cpu"))
        counts = (sub == 0).long().sum(0)
        assert torch.equal(counts, torch.ones_like(counts))

    @pytest.mark.parametrize("n_t,g", [(2, 2), (4, 4)])
    def test_sub_index_layout_is_time_major(self, n_t, g):
        """Layout is tau * n_spatial + s, matching how `latent_queries` is
        sliced; get it wrong and every query attends to the wrong block."""
        m = _summarizer(n_t, g)
        sub, _ = m._build_sub_attn_mask(T_LAT, P, torch.device("cpu"))
        for tau in range(n_t):
            for s in range(g * g):
                allowed = (sub[tau * (g * g) + s] == 0).nonzero().flatten()
                frames = {int(k) // P for k in allowed}
                blocks = {_block_of(int(k) % P, g) for k in allowed}
                assert blocks == {s}, (tau, s, blocks)
                lo = (tau * T_LAT) // n_t
                hi = max(((tau + 1) * T_LAT) // n_t, lo + 1)
                assert frames == set(range(lo, min(hi, T_LAT))), (tau, s, frames)

    def test_spatial_fallback_mask_keeps_the_block_drops_the_time(self):
        m = _summarizer(4, 2)
        sub, spat = m._build_sub_attn_mask(T_LAT, P, torch.device("cpu"))
        for i in range(m.n_sub):
            allowed = (spat[i] == 0).nonzero().flatten()
            assert {_block_of(int(k) % P, 2) for k in allowed} == {i % 4}
            assert {int(k) // P for k in allowed} == set(range(T_LAT))
            # the fallback is a relaxation, never a restriction
            assert torch.all(spat[i][sub[i] == 0] == 0)

    def test_non_square_patch_grid_is_refused(self):
        with pytest.raises(ValueError, match="square patch grid"):
            _summarizer(1, 2)._build_sub_attn_mask(T_LAT, 12, torch.device("cpu"))

    def test_grid_finer_than_the_native_patch_grid_is_refused(self):
        with pytest.raises(ValueError, match="exceeds the native patch grid"):
            _summarizer(1, 4)._build_sub_attn_mask(T_LAT, 4, torch.device("cpu"))


class TestSubTokensCarryRealInformation:
    """The behavioural half: masks are only worth something if they show up in
    the outputs. Perturb one block of the latent field, see exactly which
    sub-tokens move."""

    @staticmethod
    def _out(m, v):
        return m(video_tokens=v, T_ts=N_VIS)  # T_ts == n_vis → no null padding

    def test_shape_is_four_dim_when_widened(self):
        m = _summarizer(4, 4)
        out = self._out(m, _video())
        assert m.n_sub == 64
        assert out.shape == (B, N_VIS, 64, D_MODEL)

    def test_null_padded_macro_positions_carry_n_sub_copies(self):
        m = _summarizer(2, 2)
        out = m(video_tokens=_video(), T_ts=N_VIS + 3)
        assert out.shape == (B, N_VIS + 3, m.n_sub, D_MODEL)
        null = m.null_visual_token.view(1, 1, 1, D_MODEL)
        assert torch.equal(out[:, :3], null.expand(B, 3, m.n_sub, D_MODEL))

    def test_sub_tokens_within_a_step_are_distinct(self):
        """If these collapse, A29 degenerates into A13 and the arm is a no-op."""
        m = _summarizer(2, 2)
        out = self._out(m, _video())
        for i in range(m.n_sub):
            for j in range(i + 1, m.n_sub):
                assert not torch.allclose(out[:, -1, i], out[:, -1, j], atol=1e-6)

    def test_a_spatial_block_only_moves_its_own_sub_tokens(self):
        """Direct evidence the payload is spatially RESOLVED, not pooled — the
        one property that separates A29 from every other interleaved arm."""
        g, n_t = 2, 2
        m = _summarizer(n_t, g)
        v = _video()
        base = self._out(m, v)

        target = 3  # perturb every patch of block 3, all frames
        pert = v.clone()
        for p in range(P):
            if _block_of(p, g) == target:
                pert[:, :, p, :] += 5.0
        got = self._out(m, pert)

        moved = ~torch.isclose(base, got, atol=1e-6).all(-1)  # [B, n_vis, n_sub]
        for sub in range(m.n_sub):
            expect = (sub % (g * g)) == target
            assert bool(moved[:, :, sub].any()) == expect, sub

    def test_a_temporal_slice_only_moves_its_own_sub_tokens(self):
        """The other axis. n_t=2 is chosen so no sub-query is fully blocked by
        the causal rule, i.e. no row takes the spatial-only fallback."""
        g, n_t = 2, 2
        m = _summarizer(n_t, g)
        v = _video()
        base = self._out(m, v)

        pert = v.clone()
        pert[:, 0:2] += 5.0  # temporal slice 0 == frames [0, 2)
        got = self._out(m, pert)

        moved = ~torch.isclose(base, got, atol=1e-6).all(-1)
        for sub in range(m.n_sub):
            assert bool(moved[:, :, sub].any()) == (sub // (g * g) == 0), sub

    @pytest.mark.parametrize("n_t,g", [(1, 1), (2, 2), (4, 4), (5, 2)])
    def test_no_future_frame_leaks_into_any_sub_token(self, n_t, g):
        """Causality must hold for every (t_vis, sub) pair — including the rows
        whose temporal slice is entirely in the future of their step and fall
        back to spatial-only. n_t=5 with n_vis=2 exercises that fallback."""
        m = _summarizer(n_t, g)
        v = _video()
        base = self._out(m, v)

        for t_vis in range(N_VIS):
            for f in range(_frame_limit(t_vis), T_LAT):
                pert = v.clone()
                pert[:, f] += 5.0
                got = self._out(m, pert)
                assert torch.allclose(base[:, t_vis], got[:, t_vis], atol=1e-6), (
                    f"frame {f} leaked into step {t_vis} (limit {_frame_limit(t_vis)})"
                )

    def test_widened_output_is_finite_and_backpropagates(self):
        m = _summarizer(4, 4)
        m.train()
        out = m(video_tokens=_video(), T_ts=N_VIS)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(m.latent_queries.grad).all()
        assert m.latent_queries.grad.abs().sum() > 0


class TestWiredIntoTheInterleavedForward:
    """End-to-end through VisionChronos2: config keys reach the summarizer, the
    4-D output survives interleaving, and the encoder sees the wider payload."""

    @staticmethod
    def _module(n_t=1, g=1, n_soft=1, fusion="interleaved"):
        from unittest.mock import Mock

        from mmtsfm.models.chronos2.lightning_module import (
            VisionChronos2LightningModule,
        )

        core_cfg = {
            "d_model": 32,
            "d_kv": 8,
            "d_ff": 64,
            "num_layers": 2,
            "num_heads": 2,
            "use_grassmann": False,
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
                "fusion_mode": fusion,
                "skip_vision_stack": False,
                "visual_dropout_prob": 0.0,
                "numeric_dropout_prob": 0.0,
                "dropout": 0.0,
                "summarizer_time_slices": n_t,
                "summarizer_spatial_grid": g,
            },
            lr=1e-3,
            warmup_steps=10,
            grassmann_warmup_steps=10,
            video_encoder=_make_fake_video_encoder(
                d_v=D_V, t_lat=T_LAT, h_lat=SIDE, w_lat=SIDE
            ),
        )
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
            "daylight_future": torch.ones(2, 1, 4),
            "site_id": torch.zeros(2, 1, dtype=torch.long),
        }

    def test_config_keys_reach_the_summarizer(self):
        s = self._module(4, 4).model.latent_summarizer
        assert (s.n_time_slices, s.spatial_grid, s.n_sub) == (4, 4, 64)

    @pytest.mark.parametrize("n_t,g", [(2, 2), (4, 4)])
    def test_forward_runs_and_loss_is_finite(self, n_t, g):
        loss = self._module(n_t, g).training_step(self._batch(), batch_idx=0)
        assert torch.isfinite(loss)

    def _encoder_seq(self, n_t, g):
        mod = self._module(n_t, g)
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

    @pytest.mark.parametrize("n_t,g", [(2, 2), (4, 4)])
    def test_encoder_sees_n_vis_times_n_sub_visual_tokens(self, n_t, g):
        n_vis, n_sub = 2, n_t * g * g
        base_len, base_vis = self._encoder_seq(1, 1)
        wide_len, wide_vis = self._encoder_seq(n_t, g)

        assert base_vis == n_vis
        assert wide_vis == n_vis * n_sub
        assert wide_len - base_len == n_vis * (n_sub - 1)

    def test_widening_changes_the_forecast(self):
        b = self._batch()
        one = self._module(1, 1).training_step(b, batch_idx=0)
        many = self._module(4, 4).training_step(b, batch_idx=0)
        assert not torch.isclose(one, many, atol=1e-6)

    def test_only_latent_queries_scales_with_n_sub(self):
        """Widening adds no new modules and no new state_dict keys — only the
        query bank grows, by exactly n_sub. A checkpoint from a narrower arm
        therefore loads with one shape mismatch on `latent_queries` and nothing
        else, rather than a missing-key failure."""
        n_vis, n_sub = 2, 4 * 4 * 4  # slices x grid^2
        base = {k: v.shape for k, v in self._module(1, 1).state_dict().items()}
        wide = {k: v.shape for k, v in self._module(4, 4).state_dict().items()}
        assert set(base) == set(wide)

        q = "model.latent_summarizer.latent_queries"
        differ = [k for k in base if base[k] != wide[k]]
        assert differ == [q], differ
        assert base[q][1] == n_vis
        assert wide[q][1] == n_vis * n_sub


class TestConfoundGuards:
    """n_sub and n_soft would both widen the payload — one with information,
    one with copies — and no post-hoc analysis could separate them. Refuse the
    combination rather than run a confounded arm."""

    def test_sub_resolution_with_adapter_fanout_is_refused(self):
        with pytest.raises(ValueError):
            TestWiredIntoTheInterleavedForward._module(4, 4, n_soft=16)

    def test_sub_resolution_outside_interleaved_fusion_is_refused(self):
        with pytest.raises(ValueError):
            TestWiredIntoTheInterleavedForward._module(4, 4, fusion="late")
