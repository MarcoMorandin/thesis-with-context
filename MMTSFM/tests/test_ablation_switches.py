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
# A09 / A10 — test-time negative controls
# ----------------------------------------------------------------------


class _ControlShim(VisionChronos2LightningModule):
    """Bare shell exposing ``_apply_eval_control`` without building a model.

    ``hparams`` is a read-only property on LightningModule, so it is shadowed
    here rather than assigned.
    """

    def __init__(self, control: str, active: bool = True):  # noqa: D107
        self._hp = types.SimpleNamespace(eval_control=control, seed=42)
        self._eval_control_active = active

    @property
    def hparams(self):
        return self._hp


B, T_LAT, P, D_V = 4, 6, 9, 5


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
    assert _ControlShim("none")._apply_eval_control(inp) is inp


def test_control_is_inert_outside_the_test_loop():
    """Training and validation must never see a corrupted visual stream."""
    inp = _inputs()
    shim = _ControlShim("shuffle_frames", active=False)
    assert shim._apply_eval_control(inp) is inp


def test_shuffle_frames_permutes_within_each_sample_only():
    inp = _inputs()
    out = _ControlShim("shuffle_frames")._apply_eval_control(inp)
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
    out = _ControlShim("shuffle_frames")._apply_eval_control(inp)
    assert torch.equal(out["video_delta_t"], inp["video_delta_t"])


def test_shuffle_frames_is_reproducible():
    inp = _inputs()
    a = _ControlShim("shuffle_frames")._apply_eval_control(inp)["video_latents"]
    b = _ControlShim("shuffle_frames")._apply_eval_control(inp)["video_latents"]
    assert torch.equal(a, b)


def test_shuffle_frames_handles_the_raw_video_path():
    """Cache miss: frames arrive as [B, C, T, H, W] with T on axis 2."""
    video = torch.randn(B, 3, T_LAT, 8, 8)
    out = _ControlShim("shuffle_frames")._apply_eval_control(
        dict(
            video=video,
            video_latents=None,
            visual_mask=torch.ones(B, T_LAT),
            video_delta_t=None,
        )
    )
    assert out["video"].shape == video.shape
    assert not torch.equal(out["video"], video)


def test_swap_plant_frames_rolls_every_visual_tensor_together():
    inp = _inputs()
    out = _ControlShim("swap_plant_frames")._apply_eval_control(inp)
    for key in ("video_latents", "visual_mask", "video_delta_t"):
        assert torch.equal(out[key], torch.roll(inp[key], 1, 0)), key


def test_controls_never_touch_the_numeric_stream():
    inp = _inputs()
    for control in ("shuffle_frames", "swap_plant_frames"):
        out = _ControlShim(control)._apply_eval_control(inp)
        assert torch.equal(out["context"], inp["context"]), control


def test_control_is_a_no_op_when_there_is_no_vision():
    """s1 has no visual tensors; the control must not invent a failure."""
    inp = dict(video=None, video_latents=None, context=torch.randn(B, 4))
    assert _ControlShim("shuffle_frames")._apply_eval_control(inp) == inp


def test_unknown_control_is_rejected_loudly():
    with pytest.raises(ValueError, match="eval_control"):
        _ControlShim("rotate_frames")._apply_eval_control(_inputs())
