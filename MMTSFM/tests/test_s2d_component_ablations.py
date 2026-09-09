"""Component ablations on the s2d path: A37, A38, A43 — and ticket 44's guard.

Each of the three arms below is a *foil*: it holds everything about s2d fixed
except one thing, so that a metric difference is attributable to that thing.
The tests here assert the "everything else fixed" half — the part a plausible
implementation can silently get wrong, and the part no cluster run can check.

  * **A37** (``pool_mode="avg"``, ticket 41) — mean-pool the r*r block instead
    of concatenating it. Must keep the token COUNT and layout identical to s2d;
    if it did not, the arm would confound "sub-cell detail" with "sequence
    length", which is the exact confound it exists to remove.
  * **A38** (``evs_mode="random"``, ticket 42) — same ``keep``, uniform random
    subset. Must be parameter-free, or it could not be evaluated on an existing
    s2d checkpoint and the cheap eval-first plan collapses into a retrain.
  * **A43** (``fusion_mode="late_raw"``, ticket 28) — s2d's payload at s2b's
    position. Must produce the same sequence length as s2d with *zero*
    fractional positions; that difference is the entire experiment.

Ticket 44 is the fourth block: ``force_vision_off=True`` must preserve sequence
length. If it did not, marginal gain would be a length delta plus a content
delta, and every Δ quoted by tickets 28, 32 and 40-43 would be contaminated by
the same confound that made A30-d's q-sweep uninterpretable.

Run with: uv run pytest tests/test_s2d_component_ablations.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

from mmtsfm.models.vision.patch_projector import (  # noqa: E402
    VisualPatchProjector,
    avg_pool_tokens,
    evs_select,
    pixel_shuffle_tokens,
)
from tests.test_s2d_interleaved_raw import (  # noqa: E402
    CTX_LEN,
    D_MODEL,
    D_V,
    GRID0,
    N_CELLS,
    T_LAT,
    _inputs,
    _make_raw_model,
)
from tests.test_vision_chronos2 import (  # noqa: E402
    _make_chronos2,
    _make_fake_video_encoder,
)


def _seq_len_hook(model, store):
    def hook(_mod, _args, kwargs):
        emb = kwargs.get("inputs_embeds")
        pid = kwargs.get("position_ids")
        if torch.is_tensor(emb):
            store.setdefault("seq", []).append(emb.shape[1])
        if torch.is_tensor(pid):
            store.setdefault("frac", []).append(int((pid % 1 != 0).sum()))

    model.chronos.encoder.register_forward_pre_hook(hook, with_kwargs=True)


def _make_late_raw(evs_keep: int = 8):
    from mmtsfm.models.chronos2 import VisionChronos2Config, VisionChronos2Model

    vcfg = VisionChronos2Config(
        fusion_mode="late_raw",
        n_visual_context_steps=1,
        n_soft_tokens=1,
        visual_shuffle_r=2,
        visual_n_cells=N_CELLS,
        visual_evs_keep=evs_keep,
        visual_position_span_seconds=28800.0,
        visual_dropout_prob=0.0,
        dropout=0.0,
    )
    return VisionChronos2Model(
        chronos_model=_make_chronos2(d_model=D_MODEL, context_length=CTX_LEN),
        vision_config=vcfg,
        video_encoder=_make_fake_video_encoder(
            d_v=D_V, t_lat=T_LAT, h_lat=GRID0, w_lat=GRID0
        ),
    )


# ---------------------------------------------------------------------------
# A37 — average pooling as the foil for pixel shuffle (ticket 41)
# ---------------------------------------------------------------------------


class TestAvgPool:
    def test_shape_matches_shuffle_in_token_count_only(self):
        x = torch.randn(2, 4, 196, 1024)
        avg, shuf = avg_pool_tokens(x, 2), pixel_shuffle_tokens(x, 2)
        assert avg.shape[:3] == shuf.shape[:3] == (2, 4, 49)
        assert avg.shape[3] == 1024 and shuf.shape[3] == 4096

    def test_is_the_mean_of_the_block(self):
        """Explicit: this arm is *supposed* to destroy sub-cell detail."""
        x = torch.randn(1, 1, 4, 3)  # one 2x2 block
        assert torch.allclose(avg_pool_tokens(x, 2)[0, 0, 0], x[0, 0].mean(0))

    def test_shares_the_shuffle_grid_validation(self):
        with pytest.raises(ValueError, match="not divisible"):
            avg_pool_tokens(torch.randn(1, 1, 49, 4), 2)

    def test_projector_input_dim_drops_to_d_v(self):
        """The reason A37 cannot warm-start from an s2d checkpoint."""
        avg = VisualPatchProjector(
            d_v=D_V, d_model=D_MODEL, n_cells=N_CELLS, evs_keep=8, pool_mode="avg"
        )
        shuf = VisualPatchProjector(
            d_v=D_V, d_model=D_MODEL, n_cells=N_CELLS, evs_keep=8
        )
        assert avg.proj[0].in_features == D_V
        assert shuf.proj[0].in_features == D_V * 4

    def test_token_count_is_held_fixed_across_pool_modes(self):
        """The confound A37 exists to remove: same K, same d_model, same layout."""
        x = torch.randn(2, T_LAT, GRID0 * GRID0, D_V)
        shapes = set()
        for mode in ("shuffle", "avg"):
            proj = VisualPatchProjector(
                d_v=D_V, d_model=D_MODEL, n_cells=N_CELLS, evs_keep=8, pool_mode=mode
            )
            tok, f, c = proj(x)
            shapes.add((tuple(tok.shape), tuple(f.shape), tuple(c.shape)))
        assert len(shapes) == 1, f"pool_mode changed the token layout: {shapes}"

    def test_unknown_pool_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown pool_mode"):
            VisualPatchProjector(d_v=D_V, d_model=D_MODEL, pool_mode="maxpool")


# ---------------------------------------------------------------------------
# A38 — random EVS as the length-matched foil for novelty (ticket 42)
# ---------------------------------------------------------------------------


class TestRandomEVS:
    def test_same_shapes_and_ordering_as_novelty(self):
        tok = torch.randn(2, 4, 5, 8)
        kept, f, c = evs_select(tok, 10, mode="random")
        assert tuple(kept.shape) == (2, 10, 8)
        assert bool(((f * 5 + c).diff(dim=1) > 0).all()), "must stay in sequence order"

    def test_gathered_tokens_are_the_source_tokens(self):
        tok = torch.randn(2, 4, 5, 8)
        kept, f, c = evs_select(tok, 10, mode="random")
        src = tok.reshape(2, 20, 8).gather(
            1, (f * 5 + c).unsqueeze(-1).expand(2, 10, 8)
        )
        assert torch.equal(kept, src)

    def test_it_actually_ignores_novelty(self):
        """With one duplicate frame, novelty drops it and random does not always.

        Novelty is deterministic here (frame 1 == frame 0 scores 0 dissimilarity
        and is dropped every time); if random reproduced that, the two arms would
        be the same experiment.
        """
        tok = torch.randn(1, 4, 5, 8)
        tok[:, 1] = tok[:, 0]
        torch.manual_seed(0)
        picks = {
            tuple(evs_select(tok, 10, mode="random")[1][0].tolist()) for _ in range(8)
        }
        assert len(picks) > 1, "random selection is not varying"
        nov = tuple(evs_select(tok, 10)[1][0].tolist())
        assert picks != {nov}

    def test_random_is_parameter_free(self):
        """Why A38 can run as eval-only on an s2d checkpoint: same state_dict."""
        kw = dict(d_v=D_V, d_model=D_MODEL, n_cells=N_CELLS, evs_keep=8)
        nov = VisualPatchProjector(**kw)
        rnd = VisualPatchProjector(**kw, evs_mode="random")
        assert set(nov.state_dict()) == set(rnd.state_dict())
        rnd.load_state_dict(nov.state_dict())  # must not raise

    def test_unknown_evs_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown EVS mode"):
            evs_select(torch.randn(1, 2, 2, 4), 2, mode="topk")
        with pytest.raises(ValueError, match="unknown evs_mode"):
            VisualPatchProjector(d_v=D_V, d_model=D_MODEL, evs_mode="topk")


# ---------------------------------------------------------------------------
# A43 — late_raw: s2d's payload at s2b's position (ticket 28)
# ---------------------------------------------------------------------------


class TestLateRaw:
    def test_it_is_the_raw_path_without_the_resampler(self):
        m = _make_late_raw()
        assert m.raw_visual is True and m.late_raw is True
        assert isinstance(m.patch_projector, VisualPatchProjector)
        assert m.latent_summarizer is None and m.cross_modal_adapter is None

    def test_same_sequence_length_as_s2d_but_no_fractional_positions(self):
        """The whole experiment in one assertion: identical tokens, integer slot.

        If the length differed, A43-vs-s2d would confound placement with budget;
        if any position were fractional, A43 would *be* s2d.
        """
        store_s2d, store_late = {}, {}
        s2d, late = _make_raw_model(evs_keep=8), _make_late_raw(evs_keep=8)
        for m, store in ((s2d, store_s2d), (late, store_late)):
            m.eval()
            _seq_len_hook(m, store)
            with torch.no_grad():
                m.forward(**_inputs())

        assert store_late["seq"] == store_s2d["seq"] == [CTX_LEN // 8 + 8 + 1]
        assert store_s2d["frac"][0] > 0
        assert store_late["frac"] == [0], "late_raw must use integer positions"

    def test_vision_still_changes_the_forecast(self):
        m = _make_late_raw()
        m.eval()
        kw = _inputs()
        with torch.no_grad():
            on = m.forward(**kw).quantile_preds.float()
            off = m.forward(**kw, force_vision_off=True).quantile_preds.float()
        assert (on - off).abs().max().item() > 1e-6, "visual path is inert"


# ---------------------------------------------------------------------------
# Ticket 44 — vision-off must not change the sequence length
# ---------------------------------------------------------------------------


class TestVisionOffPreservesLength:
    """`force_vision_off` zeroes the visual embeddings; it must not remove them.

    `use_video` is derived from the *presence* of `video_latents`
    (vision_chronos2.py:937-941), not from `force_vision_off`, which only
    reaches `_modality_dropout` (:1208) and returns a zeroed tensor early
    (:635-644). The zeroed tokens still enter `interleave_sequences` and
    `refine_mask` still marks them attendable, so the length is preserved by
    construction — this test is the guard that keeps it that way.

    It matters because `compute_marginal_gain` differences the two passes. If
    vision-off shortened the sequence, the marginal gain would fold in an
    attention-budget change, and every Δ in tickets 28, 32 and 40-43 would carry
    the same length confound that made A30-d's q-sweep uninterpretable.
    """

    @pytest.mark.parametrize("evs_keep", [8, 0])
    def test_length_and_positions_identical_with_vision_off(self, evs_keep):
        m = _make_raw_model(evs_keep=evs_keep)
        m.eval()
        store = {}
        _seq_len_hook(m, store)
        kw = _inputs()
        with torch.no_grad():
            m.forward(**kw)
            m.forward(**kw, force_vision_off=True)
        assert store["seq"][0] == store["seq"][1], (
            f"vision-off changed sequence length {store['seq']} — marginal gain "
            "would be a length delta, not a content delta"
        )
        assert store["frac"][0] == store["frac"][1]

    def test_late_raw_too(self):
        m = _make_late_raw()
        m.eval()
        store = {}
        _seq_len_hook(m, store)
        kw = _inputs()
        with torch.no_grad():
            m.forward(**kw)
            m.forward(**kw, force_vision_off=True)
        assert store["seq"][0] == store["seq"][1]
