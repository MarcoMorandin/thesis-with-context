"""Ablation switches: embedding leave-one-out, lead-time gate, eval controls.

Every switch here exists to make one architectural component removable without
touching the training path of any other arm. The tests that matter most are the
*negative* ones: with the switch off, the code path must be byte-identical to
what produced the numbers already on disk.
"""

from __future__ import annotations

import types

import pytest
import torch

from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule
from mmtsfm.models.chronos2.vision_chronos2 import (
    MultimodalEmbedding,
    VisionChronos2Config,
)


# ----------------------------------------------------------------------
# Config defaults — the switches must be inert unless asked for
# ----------------------------------------------------------------------


def test_ablation_switch_defaults_are_inert():
    cfg = VisionChronos2Config()
    assert cfg.use_lead_time_embed is True
    assert cfg.disable_modality_embed is False
    assert cfg.disable_segment_embed is False
    assert cfg.disable_token_type_embed is False
    assert cfg.disable_entity_embed is False


# ----------------------------------------------------------------------
# A24 / A25 / A26 — MultimodalEmbedding leave-one-out
# ----------------------------------------------------------------------


def _embed(**flags) -> MultimodalEmbedding:
    return MultimodalEmbedding(d_model=8, n_entities=4, **flags)


def test_disabled_embeddings_keep_state_dict_shape():
    """A disabled run must still warm-start from an existing checkpoint.

    The tables are constructed either way; only the additive contribution is
    suppressed. If this ever breaks, every ablation below needs a fresh s1.
    """
    on = _embed()
    off = _embed(
        disable_modality_embed=True,
        disable_segment_embed=True,
        disable_token_type_embed=True,
        disable_entity_embed=True,
    )
    off.load_state_dict(on.state_dict())  # raises on any shape/key mismatch


@pytest.mark.parametrize(
    "flag,method,args",
    [
        ("disable_modality_embed", "add_modality", (1,)),
        ("disable_segment_embed", "add_segment", (0,)),
        ("disable_token_type_embed", "add_token_type", (2,)),
        ("disable_entity_embed", "add_entity", (torch.tensor([0, 1]),)),
    ],
)
def test_each_embedding_channel_is_individually_removable(flag, method, args):
    x = torch.randn(2, 3, 8)
    assert not torch.equal(getattr(_embed(), method)(x, *args), x), (
        f"{method} is a no-op even when ENABLED — the channel is already dead"
    )
    off = _embed(**{flag: True})
    assert torch.equal(getattr(off, method)(x, *args), x)


def test_disabling_one_channel_leaves_the_others_active():
    x = torch.randn(2, 3, 8)
    m = _embed(disable_token_type_embed=True)
    assert torch.equal(m.add_token_type(x, 2), x)
    assert not torch.equal(m.add_modality(x, 1), x)
    assert not torch.equal(m.add_segment(x, 0), x)


# ----------------------------------------------------------------------
# A09 / A10 / A10b — test-time negative controls
# ----------------------------------------------------------------------


class _ControlShim(VisionChronos2LightningModule):
    """Bare shell exposing ``_apply_eval_control`` without building a model.

    ``hparams`` is a read-only property on LightningModule, so it is shadowed
    here rather than assigned.
    """

    def __init__(self, control: str, active: bool = True, allow_inert: bool = False):  # noqa: D107
        self._hp = types.SimpleNamespace(
            eval_control=control, seed=42, eval_control_allow_inert=allow_inert
        )
        self._eval_control_active = active

    @property
    def hparams(self):
        return self._hp


B, T_LAT, P, D_V = 4, 6, 9, 5

# Four rows, two plants: the shape the shuffled test loader produces, and the
# minimum needed for a cross-plant swap to be possible at all.
TWO_PLANTS = {"site_id": ["a", "a", "b", "b"]}
ONE_PLANT = {"site_id": ["a"] * B}


def _inputs():
    return dict(
        video=None,
        video_latents=torch.arange(B * T_LAT * P * D_V, dtype=torch.float32).reshape(
            B, T_LAT, P, D_V
        ),
        visual_mask=torch.ones(B, T_LAT),
        video_delta_t=torch.arange(B * T_LAT, dtype=torch.float32).reshape(B, T_LAT),
        context=torch.randn(B, 4),
    )


def test_control_none_is_the_identity():
    inp = _inputs()
    assert _ControlShim("none")._apply_eval_control(inp, ONE_PLANT) is inp


def test_control_is_inert_outside_the_test_loop():
    """Training and validation must never see a corrupted visual stream."""
    inp = _inputs()
    shim = _ControlShim("shuffle_frames", active=False)
    assert shim._apply_eval_control(inp, TWO_PLANTS) is inp


def test_shuffle_frames_permutes_within_each_sample_only():
    inp = _inputs()
    out = _ControlShim("shuffle_frames")._apply_eval_control(inp, ONE_PLANT)
    src, got = inp["video_latents"], out["video_latents"]
    assert got.shape == src.shape
    assert not torch.equal(got, src), "shuffle produced the identity permutation"
    for b in range(B):
        before = {tuple(src[b, t].flatten().tolist()) for t in range(T_LAT)}
        after = {tuple(got[b, t].flatten().tolist()) for t in range(T_LAT)}
        assert before == after, f"sample {b} received another sample's frames"


def test_shuffle_frames_leaves_delta_t_claiming_the_old_order():
    """Δt is deliberately NOT permuted: the timestamps must go stale."""
    inp = _inputs()
    out = _ControlShim("shuffle_frames")._apply_eval_control(inp, ONE_PLANT)
    assert torch.equal(out["video_delta_t"], inp["video_delta_t"])


def test_shuffle_frames_permutes_the_mask_on_the_cached_latent_path():
    """Regression: the mask gate used to require `video`, which is None here.

    A frame moved into slot t while the mask still describes slot t marks a
    present frame absent (or vice versa), so the control would have been
    corrupting availability as well as order — a second, unlabelled change.
    """
    inp = _inputs()
    inp["visual_mask"] = torch.stack(
        [torch.arange(T_LAT, dtype=torch.float32) for _ in range(B)]
    )
    out = _ControlShim("shuffle_frames")._apply_eval_control(inp, ONE_PLANT)
    assert not torch.equal(out["visual_mask"], inp["visual_mask"])
    # The mask must follow the SAME permutation as the frames it describes.
    src, got = inp["video_latents"], out["video_latents"]
    for b in range(B):
        for t in range(T_LAT):
            origin = int(out["visual_mask"][b, t].item())
            assert torch.equal(got[b, t], src[b, origin]), (b, t)


def test_shuffle_frames_is_reproducible():
    inp = _inputs()
    a = _ControlShim("shuffle_frames")._apply_eval_control(inp, ONE_PLANT)
    b = _ControlShim("shuffle_frames")._apply_eval_control(inp, ONE_PLANT)
    assert torch.equal(a["video_latents"], b["video_latents"])


def test_shuffle_frames_handles_the_raw_video_path():
    """Cache miss: frames arrive as [B, C, T, H, W] with T on axis 2."""
    video = torch.randn(B, 3, T_LAT, 8, 8)
    out = _ControlShim("shuffle_frames")._apply_eval_control(
        dict(
            video=video,
            video_latents=None,
            visual_mask=torch.ones(B, T_LAT),
            video_delta_t=None,
        ),
        ONE_PLANT,
    )
    assert out["video"].shape == video.shape
    assert not torch.equal(out["video"], video)


# --- A10: mismatched plant --------------------------------------------------


def test_swap_plant_frames_gives_every_row_a_different_plants_sky():
    inp = _inputs()
    out = _ControlShim("swap_plant_frames")._apply_eval_control(inp, TWO_PLANTS)
    sites = TWO_PLANTS["site_id"]
    src = inp["video_latents"]
    for b in range(B):
        origin = next(
            j for j in range(B) if torch.equal(out["video_latents"][b], src[j])
        )
        assert sites[origin] != sites[b], (
            f"row {b} (plant {sites[b]}) was given plant {sites[origin]}'s sky — "
            "this control is only a control if the donor is a DIFFERENT site"
        )


def test_swap_plant_frames_moves_every_visual_tensor_as_one_unit():
    """Frames, mask and Δt must stay mutually consistent — the wrong site's."""
    inp = _inputs()
    out = _ControlShim("swap_plant_frames")._apply_eval_control(inp, TWO_PLANTS)
    src = inp["video_latents"]
    for b in range(B):
        origin = next(
            j for j in range(B) if torch.equal(out["video_latents"][b], src[j])
        )
        for key in ("visual_mask", "video_delta_t"):
            assert torch.equal(out[key][b], inp[key][origin]), (key, b)


def test_swap_plant_frames_refuses_a_single_plant_batch():
    """The ordered test loader is series-major: a batch is ONE plant.

    Rolling inside it measures staleness, not identity. That silent
    substitution is exactly what the pre-2026-09 A10 runs recorded.
    """
    with pytest.raises(RuntimeError, match="shuffle_test"):
        _ControlShim("swap_plant_frames")._apply_eval_control(_inputs(), ONE_PLANT)


def test_swap_plant_frames_refuses_a_batch_with_no_site_id():
    # entity_ids are positional (0..N-1) and cannot identify a plant, so a
    # missing site_id has to fail rather than fall back to them.
    with pytest.raises(RuntimeError, match="site_id"):
        _ControlShim("swap_plant_frames")._apply_eval_control(_inputs(), {})


def test_swap_plant_frames_is_reproducible():
    inp = _inputs()
    a = _ControlShim("swap_plant_frames")._apply_eval_control(inp, TWO_PLANTS)
    b = _ControlShim("swap_plant_frames")._apply_eval_control(inp, TWO_PLANTS)
    assert torch.equal(a["video_latents"], b["video_latents"])


# --- A10b: stale sky --------------------------------------------------------


def test_stale_sky_rolls_every_visual_tensor_together():
    inp = _inputs()
    out = _ControlShim("stale_sky")._apply_eval_control(inp, ONE_PLANT)
    for key in ("video_latents", "visual_mask", "video_delta_t"):
        assert torch.equal(out[key], torch.roll(inp[key], 1, 0)), key


def test_stale_sky_refuses_a_shuffled_loader():
    """Rolling a shuffled batch crosses sites and horizons — not a control."""
    shuffled = {"site_id": ["a", "b", "c", "d"]}
    with pytest.raises(RuntimeError, match="shuffle_test=false"):
        _ControlShim("stale_sky")._apply_eval_control(_inputs(), shuffled)


def test_stale_sky_runs_without_site_ids():
    # A dataset that cannot report site_id gets no assertion, not a crash.
    out = _ControlShim("stale_sky")._apply_eval_control(_inputs(), {})
    assert out["video_latents"].shape == (B, T_LAT, P, D_V)


# --- shared ------------------------------------------------------------------


def test_controls_never_touch_the_numeric_stream():
    inp = _inputs()
    for control in ("shuffle_frames", "swap_plant_frames", "stale_sky"):
        batch = ONE_PLANT if control != "swap_plant_frames" else TWO_PLANTS
        out = _ControlShim(control)._apply_eval_control(inp, batch)
        assert torch.equal(out["context"], inp["context"]), control


def test_control_is_a_no_op_when_there_is_no_vision():
    """s1 has no visual tensors; the control must not invent a failure."""
    inp = dict(video=None, video_latents=None, context=torch.randn(B, 4))
    assert _ControlShim("shuffle_frames")._apply_eval_control(inp, ONE_PLANT) == inp


def test_unknown_control_is_rejected_loudly():
    with pytest.raises(ValueError, match="eval_control"):
        _ControlShim("rotate_frames")._apply_eval_control(_inputs(), ONE_PLANT)


# ----------------------------------------------------------------------
# A09 falsifiability guard — a control that cannot degrade must not run
# ----------------------------------------------------------------------


def _armed(control: str, fusion_mode: str, n_vis_steps=None, allow_inert=False):
    shim = _ControlShim(control, allow_inert=allow_inert)
    summarizer = (
        None if n_vis_steps is None else types.SimpleNamespace(n_vis_steps=n_vis_steps)
    )
    shim.model = types.SimpleNamespace(
        vcfg=types.SimpleNamespace(fusion_mode=fusion_mode),
        latent_summarizer=summarizer,
    )
    return shim


def test_shuffle_frames_is_refused_on_the_s2c_visual_kv_path():
    """s2c's visual KV is one flat, position-free key set under a zero mask.

    Permuting T_lat is invisible to softmax over an unordered key set, so the
    run would write delta = 0.000 that reads exactly like the empirical finding
    "motion does not matter". It is a fact about the wiring instead.
    """
    with pytest.raises(RuntimeError, match="no-op by construction"):
        _armed("shuffle_frames", "future_query")._assert_eval_control_is_falsifiable()


def test_shuffle_frames_is_refused_when_the_summarizer_sees_one_step():
    # n_vis_steps=1: no positional encoding on K/V and the single query's causal
    # threshold admits every frame, so the summary is a set function.
    with pytest.raises(RuntimeError, match="n_visual_context_steps"):
        _armed(
            "shuffle_frames", "late", n_vis_steps=1
        )._assert_eval_control_is_falsifiable()


def test_shuffle_frames_runs_where_frame_order_is_representable():
    _armed(
        "shuffle_frames", "late", n_vis_steps=4
    )._assert_eval_control_is_falsifiable()


def test_allow_inert_records_the_architectural_null_on_purpose(capsys):
    _armed(
        "shuffle_frames", "future_query", allow_inert=True
    )._assert_eval_control_is_falsifiable()
    out = capsys.readouterr().out
    assert "INERT" in out and "ARCHITECTURAL null" in out


def test_the_guard_only_polices_shuffle_frames():
    # A10/A10b operate on the batch axis and are unaffected by KV ordering.
    for control in ("none", "swap_plant_frames", "stale_sky"):
        _armed(control, "future_query")._assert_eval_control_is_falsifiable()


# ----------------------------------------------------------------------
# A10 needs a shuffled test loader to be possible at all
# ----------------------------------------------------------------------


def _dm(**over):
    from mmtsfm.data.datamodule import MMTSFMDataModule

    cfg = dict(
        dataset_name="synthetic",
        batch_size=4,
        num_workers=0,
        num_entities=1,
        hist_steps=32,
        horizon=4,
        img_size=8,
        video_frames=4,
        num_samples_test=32,
    )
    cfg.update(over)
    dm = MMTSFMDataModule(**cfg)
    dm.setup("test")
    return dm


def test_test_loader_keeps_the_protocol_order_by_default():
    from torch.utils.data import SequentialSampler

    assert isinstance(_dm().test_dataloader().sampler, SequentialSampler)


def test_shuffle_test_is_opt_in_and_seeded():
    """A10 needs it; being seeded is what makes that control reproducible."""
    from torch.utils.data import RandomSampler

    a = _dm(shuffle_test=True).test_dataloader()
    assert isinstance(a.sampler, RandomSampler)
    b = _dm(shuffle_test=True).test_dataloader()
    assert list(iter(a.sampler)) == list(iter(b.sampler))
    c = _dm(shuffle_test=True, shuffle_test_seed=7).test_dataloader()
    assert list(iter(a.sampler)) != list(iter(c.sampler))
