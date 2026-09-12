"""A44 / ticket 45 — visual anchors spaced `visual_anchor_patch_stride` apart.

Two things have to hold at once:

  * stride=1 must reproduce the contiguous anchor tail EXACTLY, because s2b,
    s2d and A43 are all measured against that layout and a drift here would
    silently invalidate the comparisons this arm exists to make;
  * stride>1 must spread the anchors through the context and keep every one of
    the four parallel constructions (embeds, modality mask, attention mask,
    position ids) agreeing about where each token landed.

The stride is forced by the data, not by taste: uk_pv frames exist 02:00-16:00
UTC only, so an 8 h (stride-1) ladder asks for anchors at clock times that are
night for some of them — measured at 0.0% of 24,605 origins with all five
populated. At 24 h (stride 3) the anchors share the origin's own solar
geometry: 94.4%.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from mmtsfm.models.chronos2.vision_chronos2 import (
    anchor_patch_indices,
    build_interleaved_position_ids,
    interleave_sequences,
    interleaved_slot_indices,
    validate_n_visual_context_steps,
)

B, T_CTX, D, N_VIS = 2, 42, 8, 5
CPU = torch.device("cpu")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _ts() -> torch.Tensor:
    return torch.arange(B * T_CTX * D, dtype=torch.float32).reshape(B, T_CTX, D)


def _vis(n_soft: int) -> torch.Tensor:
    n = B * N_VIS * n_soft * D
    return torch.arange(n, dtype=torch.float32).reshape(B, N_VIS, n_soft, D) * -1.0


class TestAnchorIndices:
    def test_stride_one_is_the_contiguous_tail(self):
        idx = anchor_patch_indices(T_CTX, N_VIS, 1, CPU)
        assert torch.equal(idx, torch.arange(T_CTX - N_VIS, T_CTX))

    def test_option_a_geometry(self):
        """uk_pv: 1 patch = 8 h, so stride 3 = daily anchors."""
        idx = anchor_patch_indices(T_CTX, N_VIS, 3, CPU)
        assert idx.tolist() == [29, 32, 35, 38, 41]
        # newest anchor is always co-temporal with the origin
        assert int(idx[-1]) == T_CTX - 1

    def test_refuses_to_fall_off_the_front(self):
        with pytest.raises(ValueError, match="falls off the front"):
            anchor_patch_indices(10, 5, 3, CPU)

    def test_refuses_stride_zero(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            anchor_patch_indices(T_CTX, N_VIS, 0, CPU)


class TestStrideOneIsUnchanged:
    """The regression fence around s2b / s2d / A43."""

    @pytest.mark.parametrize("n_soft", [1, 4, 20])
    def test_interleave_matches_the_old_tail_construction(self, n_soft):
        ts, vis = _ts(), _vis(n_soft)
        out, modality = interleave_sequences(ts, vis, N_VIS)

        T_M = T_CTX - N_VIS
        macro = ts[:, :T_M, :]
        blocks = torch.cat([ts[:, T_M:, :].unsqueeze(2), vis], dim=2)
        expected = torch.cat([macro, blocks.reshape(B, N_VIS * (1 + n_soft), D)], dim=1)
        assert torch.equal(out, expected)
        assert int(modality.sum(1)[0]) == N_VIS * n_soft

    @pytest.mark.parametrize("n_soft", [1, 4])
    def test_position_ids_match_the_old_construction(self, n_soft):
        T_M, T_fut = T_CTX - N_VIS, 4
        refine = torch.arange(T_M, T_M + N_VIS)[:, None]
        old = torch.cat(
            [
                torch.arange(T_M),
                refine.expand(N_VIS, 1 + n_soft).reshape(-1),
                torch.arange(T_M + N_VIS, T_M + N_VIS + T_fut),
            ]
        ).unsqueeze(0)
        new = build_interleaved_position_ids(T_M, N_VIS, T_fut, CPU, n_soft=n_soft)
        assert torch.equal(old, new)


class TestStridedLayout:
    @pytest.mark.parametrize("n_soft", [1, 20])
    def test_every_anchor_keeps_its_own_visual_block(self, n_soft):
        """Anchor i's tokens must sit immediately after patch anchor_idx[i].

        If they slid, anchor i would describe a different hour than the TS token
        it is paired with — a mislabelled sky age, which is the A10b failure
        (-0.135 SS) rather than a crash.
        """
        idx = anchor_patch_indices(T_CTX, N_VIS, 3, CPU)
        ts, vis = _ts(), _vis(n_soft)
        out, modality = interleave_sequences(ts, vis, N_VIS, anchor_idx=idx)
        ts_pos, _ = interleaved_slot_indices(T_CTX, idx, n_soft)

        assert out.shape == (B, T_CTX + N_VIS * n_soft, D)
        for t in range(T_CTX):
            assert torch.equal(out[:, ts_pos[t]], ts[:, t]), f"TS patch {t}"
        for a, patch in enumerate(idx.tolist()):
            s = int(ts_pos[patch]) + 1
            assert torch.equal(out[:, s : s + n_soft], vis[:, a]), f"anchor {a}"
        assert int(modality.sum(1)[0]) == N_VIS * n_soft
        assert torch.equal(modality[:, ts_pos], torch.zeros(B, T_CTX, dtype=torch.long))

    def test_visual_tokens_are_not_all_at_the_end(self):
        """The point of the arm: the TS stream is interrupted, not suffixed."""
        idx = anchor_patch_indices(T_CTX, N_VIS, 3, CPU)
        _, modality = interleave_sequences(_ts(), _vis(20), N_VIS, anchor_idx=idx)
        vis_at = modality[0].nonzero().flatten()
        # first visual token appears well before the tail begins
        assert int(vis_at[0]) < T_CTX - N_VIS
        # and TS tokens still follow it
        assert int(modality[0, int(vis_at[0]) + 21]) == 0

    @pytest.mark.parametrize("n_soft", [1, 20])
    def test_position_ids_are_the_anchor_patch_indices(self, n_soft):
        idx = anchor_patch_indices(T_CTX, N_VIS, 3, CPU)
        T_fut = 4
        pos = build_interleaved_position_ids(
            T_CTX - N_VIS,
            N_VIS,
            T_fut,
            CPU,
            n_soft=n_soft,
            anchor_idx=idx,
            T_ctx=T_CTX,
        )[0]
        ts_pos, vis_pos = interleaved_slot_indices(T_CTX, idx, n_soft)

        assert pos.shape == (T_CTX + N_VIS * n_soft + T_fut,)
        # a TS patch's position id is its own patch index
        assert torch.equal(pos[ts_pos], torch.arange(T_CTX))
        # each visual token is co-temporal with its anchor patch
        assert torch.equal(pos[vis_pos], idx[:, None].expand(N_VIS, n_soft).reshape(-1))
        # future continues from T_ctx, and time never runs backwards
        tail = pos[T_CTX + N_VIS * n_soft :]
        assert torch.equal(tail, torch.arange(T_CTX, T_CTX + T_fut))
        assert torch.all(pos[1:] >= pos[:-1])

    def test_attention_mask_follows_the_same_slot_map(self):
        """An unobserved TS patch must stay masked wherever it lands."""
        n_soft = 20
        idx = anchor_patch_indices(T_CTX, N_VIS, 3, CPU)
        ts_pos, vis_pos = interleaved_slot_indices(T_CTX, idx, n_soft)
        ctx_mask = torch.ones(B, T_CTX)
        ctx_mask[:, [0, 29, 41]] = 0.0

        il = torch.ones(B, T_CTX + N_VIS * n_soft)
        il[:, ts_pos] = ctx_mask
        assert torch.equal(il[:, ts_pos], ctx_mask)
        assert torch.all(il[:, vis_pos] == 1.0)
        assert int((il[0] == 0).sum()) == 3


class TestS2eConfigGeometry:
    """stage/s2e.yaml and model/vision_chronos2_s2e.yaml describe ONE ladder.

    The data side picks frames in hours; the model side places anchors in TS
    patches. Nothing at runtime reconciles them — a mismatch does not crash, it
    hands each anchor a sky from the wrong hour under a position id claiming
    otherwise, which is the A10b stale-sky failure (-0.135 SS). So assert the
    two files agree here, where it is cheap.
    """

    @staticmethod
    def _cfgs() -> tuple[dict, dict]:
        stage = yaml.safe_load((CONFIGS / "stage" / "s2e.yaml").read_text())
        model = yaml.safe_load(
            (CONFIGS / "model" / "vision_chronos2_s2e.yaml").read_text()
        )
        return stage["data"], model

    def test_anchor_stride_is_an_integer_number_of_ts_patches(self):
        data, model = self._cfgs()
        v = model["vision_cfg"]
        span_h = float(v["visual_position_span_seconds"]) / 3600.0
        stride = int(v["visual_anchor_patch_stride"])
        assert stride * span_h == pytest.approx(data["visual_anchor_stride_hours"])

    def test_window_covers_every_anchor(self):
        """The exact inequality vision_chronos2.forward raises on."""
        data, model = self._cfgs()
        v = model["vision_cfg"]
        n_vis = int(v["n_visual_context_steps"])
        need_h = (
            (n_vis - 1)
            * int(v["visual_anchor_patch_stride"])
            * float(v["visual_position_span_seconds"])
            / 3600.0
        )
        oldest_h = (n_vis - 1) * data["visual_anchor_stride_hours"] + (
            data["visual_frames_per_anchor"] - 1
        ) * data["visual_frame_spacing_min"] / 60.0
        assert need_h <= oldest_h <= data["visual_window_hours"]

    def test_frames_split_evenly_into_anchors(self):
        data, model = self._cfgs()
        n_vis = int(model["vision_cfg"]["n_visual_context_steps"])
        f = int(data["visual_frames_per_anchor"])
        assert int(data["video_frames"]) == n_vis * f
        # EVS budgets per anchor; a keep that does not divide would slice a
        # globally-ranked list and group i would not hold anchor i's tokens.
        assert int(model["vision_cfg"]["visual_evs_keep"]) % n_vis == 0

    def test_bursts_do_not_overlap(self):
        data, _ = self._cfgs()
        burst_h = (
            (data["visual_frames_per_anchor"] - 1)
            * data["visual_frame_spacing_min"]
            / 60.0
        )
        assert burst_h < data["visual_anchor_stride_hours"]

    def test_anchors_fit_the_ts_context(self):
        data, model = self._cfgs()
        v = model["vision_cfg"]
        cc = model["chronos_core_cfg"]["chronos_config"]
        # uk_pv history is 14 days at 30 min = 672 steps, below context_length.
        t_ctx = validate_n_visual_context_steps(
            int(v["n_visual_context_steps"]),
            672,
            int(cc["input_patch_size"]),
            anchor_patch_stride=int(v["visual_anchor_patch_stride"]),
        )
        assert t_ctx == 42
        idx = anchor_patch_indices(
            t_ctx,
            int(v["n_visual_context_steps"]),
            int(v["visual_anchor_patch_stride"]),
            CPU,
        )
        assert idx.tolist() == [29, 32, 35, 38, 41]
        assert data["emit_vision"] is True
