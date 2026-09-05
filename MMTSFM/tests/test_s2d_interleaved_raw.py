"""s2d: interleaved vision with NO resampler.

Every arm through s2b pushes the V-JEPA patch field through `LatentSummarizer`
— a Perceiver resampler whose ~800:1 learned-query pool the latent probe showed
destroys the spatial structure that carries ramp signal (ramp R^2 0.0060 at 1x1
vs 0.0512 at 4x4, t+30). s2d deletes that module from the graph entirely and
uses the `encoder -> MLP projector -> decoder` shape of NVIDIA Nemotron 3 Nano
Omni (arXiv 2604.24954v2 §2): pixel shuffle, MLP projector, spatial cell
embedding, EVS pruning.

The properties worth asserting are the ones that are silently satisfiable by a
broken implementation:

  1. **The summarizer is absent, not bypassed.** A `latent_summarizer` left on
     the module would still be checkpointed and still appear in grad logs, and
     the arm would be describing an architecture it does not have.
  2. **Visual tokens sit at FRACTIONAL positions inside the last context
     patch.** At `input_patch_size` 16 on 30-min uk_pv one TS token spans 8 h,
     so all four latents fall inside patch 41 — the interleaving is sub-patch.
     If the positions collapsed to integers, s2d would be s2b with more tokens,
     and A09 would be inert again.
  3. **A09 frame-shuffle is falsifiable here.** `_apply_eval_control` permutes
     `video_latents` but NOT `video_delta_t`, so the clock stays put while the
     content moves. On s2b that control is structurally inert; on s2d it must
     change the forecast, or the positional claim above is untestable.
  4. **s2b is byte-identical.** Every change is behind `raw_visual`; if it were
     not, the s2d-vs-s2b delta would not be attributable to the visual path.

Run with: uv run pytest tests/test_s2d_interleaved_raw.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

from mmtsfm.models.chronos2.vision_chronos2 import (  # noqa: E402
    build_subpatch_position_ids,
    reduce_delta_t_to_latents,
)
from mmtsfm.models.vision.patch_projector import (  # noqa: E402
    VisualPatchProjector,
    evs_select,
    pixel_shuffle_tokens,
)
from tests.test_vision_chronos2 import (  # noqa: E402
    _make_chronos2,
    _make_fake_video_encoder,
)

D_MODEL = 64
D_V = 4
T_LAT = 4
GRID0 = 8  # fake encoder emits 8x8 -> P=64; r=2 -> 4x4 = 16 cells
N_CELLS = 16
CTX_LEN = 64  # /8 input_patch_size -> T_ctx = 8, so T_M = 7


def _make_raw_model(evs_keep: int = 8, n_vis: int = 1, n_soft: int = 1):
    from mmtsfm.models.chronos2 import VisionChronos2Config, VisionChronos2Model

    chronos = _make_chronos2(d_model=D_MODEL, context_length=CTX_LEN)
    vcfg = VisionChronos2Config(
        fusion_mode="interleaved_raw",
        n_visual_context_steps=n_vis,
        n_soft_tokens=n_soft,
        visual_shuffle_r=2,
        visual_n_cells=N_CELLS,
        visual_evs_keep=evs_keep,
        visual_position_span_seconds=28800.0,
        visual_dropout_prob=0.0,
        dropout=0.0,
    )
    return VisionChronos2Model(
        chronos_model=chronos,
        vision_config=vcfg,
        video_encoder=_make_fake_video_encoder(
            d_v=D_V, t_lat=T_LAT, h_lat=GRID0, w_lat=GRID0
        ),
    )


def _inputs(B: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    return {
        "context": torch.randn(B, CTX_LEN),
        "group_ids": torch.arange(B),
        "num_output_patches": 1,
        "video_latents": torch.randn(B, T_LAT, GRID0 * GRID0, D_V),
        "visual_mask": torch.ones(B, T_LAT),
        # Newest frame last, 30-min spacing, all inside the 8 h span.
        "video_delta_t": torch.tensor(
            [[10800.0, 7200.0, 3600.0, 1800.0]] * B, dtype=torch.float32
        ),
    }


def _median(out):
    q = out.quantile_preds.float()
    return q[:, q.shape[1] // 2, :].clone()


# ---------------------------------------------------------------------------
# Pixel shuffle: space-to-depth, never averaging
# ---------------------------------------------------------------------------


class TestPixelShuffle:
    def test_shape(self):
        x = torch.randn(2, 4, 196, 1024)
        assert tuple(pixel_shuffle_tokens(x, 2).shape) == (2, 4, 49, 4096)

    def test_is_a_permutation_not_a_pool(self):
        """Every input value must survive. An averaging merge would lose 3 of 4.

        This is the whole hypothesis: the arm exists to stop pooling the patch
        field, so a shuffle that quietly averaged would silently reintroduce the
        bottleneck it was built to remove.
        """
        x = torch.randn(1, 2, 16, 3)
        y = pixel_shuffle_tokens(x, 2)
        assert torch.allclose(x.flatten().sort().values, y.flatten().sort().values)

    def test_rejects_non_square_grid(self):
        with pytest.raises(ValueError, match="not a square grid"):
            pixel_shuffle_tokens(torch.randn(1, 1, 15, 4), 2)

    def test_rejects_indivisible_factor(self):
        with pytest.raises(ValueError, match="not divisible"):
            pixel_shuffle_tokens(torch.randn(1, 1, 49, 4), 2)


# ---------------------------------------------------------------------------
# EVS: temporal novelty, frame 0 pinned
# ---------------------------------------------------------------------------


class TestEVS:
    def test_static_frames_are_dropped_and_anchor_survives(self):
        tok = torch.randn(2, 4, 5, 8)
        tok[:, 1] = tok[:, 0]  # frame 1 adds nothing new
        kept, frame_idx, cell_idx = evs_select(tok, 10)
        assert tuple(kept.shape) == (2, 10, 8)
        assert int((frame_idx == 0).sum(1).min()) == 5, (
            "anchor frame must be kept whole"
        )
        assert int((frame_idx == 1).sum()) == 0, "duplicate frame must be dropped"

    def test_indices_stay_in_frame_cell_order(self):
        """Kept tokens keep their sequence order, so positions stay monotone."""
        kept, f, c = evs_select(torch.randn(2, 4, 5, 8), 10)
        flat = (f * 5 + c).diff(dim=1)
        assert bool((flat > 0).all())

    def test_gathered_tokens_are_the_source_tokens(self):
        tok = torch.randn(2, 4, 5, 8)
        kept, f, c = evs_select(tok, 10)
        src = tok.reshape(2, 20, 8).gather(
            1, (f * 5 + c).unsqueeze(-1).expand(2, 10, 8)
        )
        assert torch.equal(kept, src)

    def test_keep_at_or_above_total_is_a_noop(self):
        kept, f, _ = evs_select(torch.randn(2, 4, 5, 8), 20)
        assert tuple(kept.shape) == (2, 20, 8) and int(f.max()) == 3


class TestProjector:
    def test_output_shape_and_no_learned_queries(self):
        proj = VisualPatchProjector(
            d_v=D_V, d_model=D_MODEL, shuffle_r=2, n_cells=N_CELLS, evs_keep=8
        )
        tok, f, c = proj(torch.randn(2, T_LAT, GRID0 * GRID0, D_V))
        assert tuple(tok.shape) == (2, 8, D_MODEL)
        names = dict(proj.named_parameters())
        assert not [n for n in names if "latent_quer" in n]
        # Spatial identity only: a temporal-slice embedding would give A09 a
        # second encoding of the axis the fractional positions already carry.
        assert "cell_embed" in names and names["cell_embed"].shape == (N_CELLS, D_MODEL)

    def test_cell_count_mismatch_is_loud(self):
        proj = VisualPatchProjector(
            d_v=D_V, d_model=D_MODEL, shuffle_r=2, n_cells=99, evs_keep=8
        )
        with pytest.raises(ValueError, match="n_cells=99"):
            proj(torch.randn(1, T_LAT, GRID0 * GRID0, D_V))


# ---------------------------------------------------------------------------
# Δt reduction and fractional positions
# ---------------------------------------------------------------------------


class TestSubpatchPositions:
    def test_delta_t_reduces_to_the_oldest_frame_per_tubelet(self):
        """Must match LatentSummarizer's amax, or s2b and s2d read different clocks."""
        raw = torch.tensor(
            [[12600.0, 10800.0, 9000.0, 7200.0, 5400.0, 3600.0, 1800.0, 0.0]]
        )
        assert torch.equal(
            reduce_delta_t_to_latents(raw, 4),
            torch.tensor([[12600.0, 9000.0, 5400.0, 1800.0]]),
        )

    def test_passthrough_when_already_per_latent(self):
        dt = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        assert torch.equal(reduce_delta_t_to_latents(dt, 4), dt)

    def test_rejects_ragged_length(self):
        with pytest.raises(ValueError, match="not a multiple"):
            reduce_delta_t_to_latents(torch.zeros(1, 7), 4)

    def test_visual_tokens_land_strictly_inside_the_last_context_patch(self):
        T_M, T_fut, K = 7, 1, 6
        frame_idx = torch.tensor([[0, 0, 1, 2, 3, 3]] * 2)
        dt = torch.tensor([[10800.0, 7200.0, 3600.0, 1800.0]] * 2)
        pos = build_subpatch_position_ids(T_M, T_fut, frame_idx, dt, 28800.0)

        assert tuple(pos.shape) == (2, T_M + 1 + K + T_fut)
        ctx, vis, fut = pos[:, : T_M + 1], pos[:, T_M + 1 : -T_fut], pos[:, -T_fut:]
        assert torch.equal(ctx[0], torch.arange(T_M + 1, dtype=torch.float32))
        # After the co-temporal TS token, strictly before the first future token.
        assert bool((vis > T_M).all()), (
            "visual tokens must follow the TS token they share a patch with"
        )
        assert bool((vis < fut.min()).all()), "visual tokens must precede the forecast"
        assert bool((vis % 1 != 0).all()), (
            "positions must be fractional, not collapsed to T_M"
        )
        # Older frame -> smaller position.
        assert bool((vis.diff(dim=1) >= 0).all())
        assert torch.equal(fut[0], torch.tensor([float(T_M + 1)]))

    def test_missing_delta_t_pins_every_token_to_the_patch_boundary(self):
        pos = build_subpatch_position_ids(
            7, 1, torch.zeros(2, 6, dtype=torch.long), None, 28800.0
        )
        assert bool((pos[:, 8:-1] == 7.0).all())


# ---------------------------------------------------------------------------
# Model wiring
# ---------------------------------------------------------------------------


class TestRawVisualModel:
    def test_summarizer_is_absent_not_bypassed(self):
        m = _make_raw_model()
        assert m.raw_visual is True
        assert m.latent_summarizer is None
        assert m.cross_modal_adapter is None
        assert isinstance(m.patch_projector, VisualPatchProjector)
        keys = m.state_dict().keys()
        assert not [k for k in keys if "latent_summarizer" in k]
        assert [k for k in keys if "patch_projector" in k]

    def test_interleaved_arm_keeps_its_summarizer(self):
        """s2b must be untouched — every s2d change sits behind `raw_visual`."""
        from mmtsfm.models.chronos2 import VisionChronos2Config, VisionChronos2Model

        m = VisionChronos2Model(
            chronos_model=_make_chronos2(d_model=D_MODEL, context_length=CTX_LEN),
            vision_config=VisionChronos2Config(
                fusion_mode="interleaved", n_visual_context_steps=1, dropout=0.0
            ),
            video_encoder=_make_fake_video_encoder(d_v=D_V),
        )
        assert m.raw_visual is False
        assert m.patch_projector is None
        assert m.latent_summarizer is not None

    def test_soft_token_fanout_is_rejected(self):
        """n_soft_tokens fans a POOLED vector out; with no pool it is meaningless
        and A13 already showed the fan-out adds no information."""
        with pytest.raises(ValueError):
            _make_raw_model(n_soft=4)

    def test_more_than_one_visual_context_step_is_rejected(self):
        """Checked at forward time, not init: `n_vis` is clamped to T_ctx there,
        so the constructor cannot know the effective value yet."""
        m = _make_raw_model(n_vis=2)
        with pytest.raises(ValueError, match="n_visual_context_steps=1"):
            m.forward(**_inputs())

    def test_sequence_length_is_ts_plus_kept_visual_plus_future(self):
        m = _make_raw_model(evs_keep=8)
        m.eval()
        seen = {}

        def hook(_mod, _args, kwargs):
            emb = kwargs.get("inputs_embeds")
            pid = kwargs.get("position_ids")
            if torch.is_tensor(emb):
                seen["seq"] = emb.shape[1]
            if torch.is_tensor(pid):
                seen["frac"] = int((pid % 1 != 0).sum())

        m.chronos.encoder.register_forward_pre_hook(hook, with_kwargs=True)
        with torch.no_grad():
            m.forward(**_inputs())

        T_ctx = CTX_LEN // 8
        assert seen["seq"] == T_ctx + 8 + 1, (
            f"expected {T_ctx} TS + 8 visual + 1 future, got {seen['seq']}"
        )
        assert seen["frac"] > 0, "no fractional positions — the arm degenerated to s2b"

    def test_disabling_evs_keeps_every_token(self):
        m = _make_raw_model(evs_keep=0)
        m.eval()
        seen = {}
        m.chronos.encoder.register_forward_pre_hook(
            lambda _m, _a, kw: seen.update(seq=kw["inputs_embeds"].shape[1]),
            with_kwargs=True,
        )
        with torch.no_grad():
            m.forward(**_inputs())
        assert seen["seq"] == CTX_LEN // 8 + T_LAT * N_CELLS + 1

    def test_vision_changes_the_forecast(self):
        m = _make_raw_model()
        m.eval()
        kw = _inputs()
        with torch.no_grad():
            on = _median(m.forward(**kw))
            off = _median(m.forward(**kw, force_vision_off=True))
        assert (on - off).abs().max().item() > 1e-6, "visual path is inert"

    def test_frame_shuffle_changes_the_forecast(self):
        """A09 falsifiability. `_apply_eval_control` permutes `video_latents` but
        not `video_delta_t`, so a shuffle moves content against a fixed clock. If
        this were invariant, the fractional-position claim would be untestable —
        which is exactly the state s2b is stuck in."""
        m = _make_raw_model()
        m.eval()
        kw = _inputs()
        shuffled = dict(kw)
        shuffled["video_latents"] = kw["video_latents"].flip(1)
        with torch.no_grad():
            base = _median(m.forward(**kw))
            perm = _median(m.forward(**shuffled))
        assert (base - perm).abs().max().item() > 1e-6, (
            "frame shuffle is inert — s2d has no temporal ordering to falsify"
        )

    def test_frame_mask_is_live(self):
        """Masked frames have their token embeddings zeroed *in place* — the
        tokens keep their slots, positions and modality embeds, so this is NOT
        equivalent to `force_vision_off`. What must hold is that the mask is
        wired at all: EVS pins frame 0 as an anchor, so an unavailable frame
        survives selection and would leak its content if the mask were dropped.
        """
        m = _make_raw_model()
        m.eval()
        kw = _inputs()
        masked = dict(kw, visual_mask=torch.zeros_like(kw["visual_mask"]))
        with torch.no_grad():
            seen = _median(m.forward(**kw))
            hidden = _median(m.forward(**masked))
        assert (seen - hidden).abs().max().item() > 1e-6, "visual_mask is ignored"


# ---------------------------------------------------------------------------
# Stage 0: projector-only LR warmup (Nemotron §3.1.1)
# ---------------------------------------------------------------------------


def _lightning_module(projector_warmup_steps: int):
    import types

    from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule

    lm = VisionChronos2LightningModule(
        chronos_core_cfg={
            "d_model": D_MODEL,
            "d_kv": 16,
            "d_ff": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout_rate": 0.0,
            "use_grassmann": False,
            "chronos_config": {
                "context_length": CTX_LEN,
                "output_patch_size": 8,
                "input_patch_size": 8,
                "input_patch_stride": 8,
                "quantiles": [0.1, 0.5, 0.9],
                "use_reg_token": False,
                "use_arcsinh": False,
                "max_output_patches": 4,
            },
        },
        vision_cfg={
            "fusion_mode": "interleaved_raw",
            "n_visual_context_steps": 1,
            "skip_vision_stack": False,
            "visual_shuffle_r": 2,
            "visual_n_cells": N_CELLS,
            "visual_evs_keep": 8,
            "visual_dropout_prob": 0.0,
            "dropout": 0.0,
        },
        pretrained_model_name_or_path=None,
        # Without this the model builds a real V-JEPA ViT-L from the hub.
        video_encoder=_make_fake_video_encoder(
            d_v=D_V, t_lat=T_LAT, h_lat=GRID0, w_lat=GRID0
        ),
        projector_warmup_steps=projector_warmup_steps,
        warmup_steps=100,
        freeze_chronos=True,
        horizon=8,
    )
    # `_total_steps` reads `trainer.estimated_stepping_batches`. Stub the trainer
    # rather than patching the property: this file runs inside the full suite, and
    # a class-level patch would leak a fixed step count into every other test.
    lm._trainer = types.SimpleNamespace(estimated_stepping_batches=2000, max_epochs=1)
    return lm


def _lr_multipliers(lm, step: int):
    cfg = lm.configure_optimizers()
    opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
    names = [g.get("name", "?") for g in opt.param_groups]
    vals = [lmb(step) for lmb in sched.lr_lambdas]
    proj = [v for n, v in zip(names, vals) if "projector" in n]
    rest = [v for n, v in zip(names, vals) if "projector" not in n]
    return proj, rest


class TestStageZero:
    def test_projector_gets_its_own_param_group(self):
        lm = _lightning_module(500)
        names = {
            g.get("name") for g in lm.configure_optimizers()["optimizer"].param_groups
        }
        assert {"projector_decay", "projector_nodecay"} <= names

    def test_only_the_projector_moves_during_stage_zero(self):
        lm = _lightning_module(500)
        for step in (0, 1, 250, 499):
            proj, rest = _lr_multipliers(lm, step)
            assert all(v == 0.0 for v in rest), f"step {step}: backbone must be frozen"
            assert all(v == pytest.approx(step / 500) for v in proj)

    def test_backbone_still_gets_its_warmup_after_stage_zero(self):
        """The schedule is SHIFTED, not fast-forwarded. Releasing the backbone
        straight to full LR is the failure warmup exists to prevent."""
        lm = _lightning_module(500)
        assert _lr_multipliers(lm, 500)[1][0] == pytest.approx(0.0)
        assert _lr_multipliers(lm, 550)[1][0] == pytest.approx(0.5)
        assert _lr_multipliers(lm, 600)[1][0] == pytest.approx(1.0)

    def test_projector_holds_full_lr_instead_of_re_warming(self):
        lm = _lightning_module(500)
        for step in (500, 550, 600):
            assert all(v == pytest.approx(1.0) for v in _lr_multipliers(lm, step)[0])

    def test_zero_warmup_leaves_every_other_arm_bit_identical(self):
        """p_warmup=0 must be the identity, or enabling s2d silently rewrites
        s2b's and s2c's schedules."""
        lm = _lightning_module(0)
        for step, expected in ((0, 0.0), (50, 0.5), (100, 1.0)):
            _, rest = _lr_multipliers(lm, step)
            assert all(v == pytest.approx(expected) for v in rest)
