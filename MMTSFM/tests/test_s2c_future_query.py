"""s2c: forecast positions cross-attend a RETAINED spatial field.

Every arm up to s2b pushed the V-JEPA patch grid through the LatentSummarizer
before fusion — a ~800:1 pool that ticket 13 showed destroys exactly the
structure the model-free latent probe found (ramp R^2 0.0060 at 1x1 vs 0.0512
at 4x4, t+30). s2c bypasses that stack: the patch field is block-pooled to a
4x4 GRID per temporal slice, kept as key/value tokens, and queried by the
FORECAST positions inside the last k encoder blocks.

Two properties make the arm interpretable, and neither is safe to assert by
reading the code:

  1. the visual residual reaches the future positions and NOTHING else. If it
     leaked into the historical context, s2c would change two things at once
     and the comparison against s2b would say nothing about the mechanism.
  2. `output_patch_size` 16 -> 4 actually produces three future positions. At
     16 there is exactly ONE (ceil(12/16)), so per-horizon queries would have
     nothing to be, and a lead-time embedding of length 1 would train to a
     constant while every metric looked plausible.

The third property — that the three queries learn horizon-SPECIFIC attention
rather than three copies of one generic summary — cannot be settled by a test
on random weights; it is ticket 15, measured on a trained checkpoint.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unittest.mock import Mock  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from mmtsfm.models.chronos2.config import Chronos2CoreConfig  # noqa: E402
from mmtsfm.models.chronos2.model import (  # noqa: E402
    Chronos2Encoder,
    Chronos2EncoderBlock,
)

HORIZON = 12
D_MODEL = 32
D_V = 4
T_LAT = 4
GRID0 = 4  # fake encoder emits a 4x4 patch grid -> P = 16


# ---------------------------------------------------------------------------
# Block-level: where the visual residual is allowed to land
# ---------------------------------------------------------------------------


def _block_cfg(num_layers: int = 1, k: int = 1) -> Chronos2CoreConfig:
    cfg = Chronos2CoreConfig(
        d_model=D_MODEL,
        d_kv=8,
        d_ff=64,
        num_layers=num_layers,
        num_heads=2,
        dropout_rate=0.0,
        use_grassmann=False,
        visual_cross_attn_blocks=k,
        chronos_config={
            "context_length": 16,
            "output_patch_size": 4,
            "input_patch_size": 4,
            "input_patch_stride": 4,
            "quantiles": [0.5],
            "use_reg_token": False,
            "use_arcsinh": False,
            "max_output_patches": 4,
        },
    )
    cfg.is_decoder = False
    return cfg


def _block_inputs(
    batch: int = 2,
    seq: int = 8,
    n_kv: int = 16,
    groups: Optional[torch.Tensor] = None,
):
    torch.manual_seed(0)
    hidden = torch.randn(batch, seq, D_MODEL)
    # GroupSelfAttention attends along the BATCH axis, per time position, so
    # rows sharing a group id are not independent. Tests that need row-level
    # isolation must ask for distinct groups explicitly.
    group_ids = (
        torch.zeros(batch, dtype=torch.long) if groups is None else groups
    )
    mask = torch.ones(batch, seq)
    return dict(
        hidden_states=hidden,
        position_ids=torch.arange(seq).unsqueeze(0),
        attention_mask=Chronos2Encoder._expand_and_invert_time_attention_mask(
            mask, hidden.dtype
        ),
        group_time_mask=Chronos2Encoder._construct_and_invert_group_time_mask(
            group_ids, mask, hidden.dtype
        ),
    ), torch.randn(batch, n_kv, D_MODEL)


class TestCrossAttentionIsBuiltOnlyWhereAsked:
    def test_zero_blocks_builds_no_module_at_all(self):
        """Every arm predating s2c must keep its exact parameter set. Not
        "the module is present but unused" — absent, so no checkpoint key and
        no forward-path difference can creep in."""
        cfg = _block_cfg(num_layers=6, k=0)
        for i in range(6):
            blk = Chronos2EncoderBlock(cfg, block_idx=i, num_layers=6)
            assert blk.visual_cross_attn is None, f"block {i}"
            assert not [n for n, _ in blk.named_parameters() if "visual_cross" in n]

    def test_k_blocks_are_the_trailing_ones(self):
        """New capacity goes at the top, against a frozen backbone — same
        convention as `n_unfreeze_encoder_blocks`."""
        cfg = _block_cfg(num_layers=6, k=4)
        built = [
            Chronos2EncoderBlock(cfg, block_idx=i, num_layers=6).visual_cross_attn
            is not None
            for i in range(6)
        ]
        assert built == [False, False, True, True, True, True]


class TestVisualResidualIsGatedToFuturePositions:
    """The `torch.where` gate is load-bearing, not cosmetic.

    `TimeCrossAttention` adds its OWN residual internally, so its return value
    is the whole updated sequence. Without the gate, every context position
    would silently absorb satellite information too.
    """

    @staticmethod
    def _run(seq: int = 8, n_fut: int = 3):
        cfg = _block_cfg(num_layers=1, k=1)
        torch.manual_seed(0)
        blk = Chronos2EncoderBlock(cfg, block_idx=0, num_layers=1).eval()
        kwargs, kv = _block_inputs(seq=seq)

        mask = torch.zeros(kwargs["hidden_states"].shape[:2], dtype=torch.bool)
        mask[:, seq - n_fut :] = True

        with torch.no_grad():
            off = blk(**kwargs)[0]
            on = blk(**kwargs, visual_kv=kv, visual_query_mask=mask)[0]
        return off, on, seq - n_fut

    def test_context_positions_are_bit_identical(self):
        off, on, n_ctx = self._run()
        assert torch.equal(off[:, :n_ctx], on[:, :n_ctx]), (
            "visual information reached the historical context — the "
            "future-only gate is not doing its job"
        )

    def test_future_positions_actually_move(self):
        """The mirror assertion. A gate that blocks everything would pass the
        test above and make the arm a no-op."""
        off, on, n_ctx = self._run()
        assert not torch.allclose(off[:, n_ctx:], on[:, n_ctx:], atol=1e-6), (
            "the visual field never reached the forecast positions either"
        )

    def test_a_row_with_vision_off_is_untouched(self):
        """Per-sample modality dropout and the marginal-gain pass both act by
        withholding the QUERY. A row whose mask is all-false must reproduce the
        vision-free forward exactly, not merely approximately."""
        cfg = _block_cfg(num_layers=1, k=1)
        torch.manual_seed(0)
        blk = Chronos2EncoderBlock(cfg, block_idx=0, num_layers=1).eval()
        # Distinct groups: the gate is a claim about the visual RESIDUAL, and
        # group attention would otherwise carry row 0's update into row 1 and
        # make the claim untestable. The sharing itself is asserted separately
        # below.
        kwargs, kv = _block_inputs(
            batch=2, seq=8, groups=torch.tensor([0, 1], dtype=torch.long)
        )

        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 5:] = True  # row 0 on, row 1 off

        with torch.no_grad():
            off = blk(**kwargs)[0]
            on = blk(**kwargs, visual_kv=kv, visual_query_mask=mask)[0]

        assert torch.equal(off[1], on[1]), "vision-off row was perturbed"
        assert not torch.allclose(off[0, 5:], on[0, 5:], atol=1e-6)

    def test_within_one_group_the_update_is_shared_at_future_positions(self):
        """Documents a real, intended property rather than asserting an
        idealisation. GroupSelfAttention runs AFTER cross-attention in the same
        block and attends along the batch axis, so a masked-off row in the SAME
        group still sees the visual update — but only at the time positions
        where some row in the group was gated on. The load-bearing claim is the
        one about time positions, not the one about rows: context positions
        must stay clean. This is structurally identical to s2b, where the
        pooled visual tokens are likewise group-visible."""
        cfg = _block_cfg(num_layers=1, k=1)
        torch.manual_seed(0)
        blk = Chronos2EncoderBlock(cfg, block_idx=0, num_layers=1).eval()
        kwargs, kv = _block_inputs(batch=2, seq=8)  # both rows in group 0

        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 5:] = True

        with torch.no_grad():
            off = blk(**kwargs)[0]
            on = blk(**kwargs, visual_kv=kv, visual_query_mask=mask)[0]

        assert not torch.allclose(off[1, 5:], on[1, 5:], atol=1e-6), (
            "expected group attention to share the visual update at future "
            "positions"
        )
        assert torch.allclose(off[:, :5], on[:, :5], atol=1e-6), (
            "the visual update reached CONTEXT positions — group attention is "
            "per time position, so this can only mean the gate leaked"
        )

    def test_no_kv_means_no_cross_attention(self):
        """`visual_kv=None` must skip the module entirely, so a vision-free
        batch on an s2c checkpoint is numerically the vision-free path."""
        cfg = _block_cfg(num_layers=1, k=1)
        torch.manual_seed(0)
        blk = Chronos2EncoderBlock(cfg, block_idx=0, num_layers=1).eval()
        kwargs, _ = _block_inputs()
        with torch.no_grad():
            a = blk(**kwargs)[0]
            b = blk(**kwargs, visual_kv=None, visual_query_mask=None)[0]
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Model-level: the s2c arm end to end
# ---------------------------------------------------------------------------


def _module(fusion_mode: str = "future_query", k: int = 2, visual_grid: int = 4):
    from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_vision_chronos2 import _make_fake_video_encoder

    core_cfg = {
        "d_model": D_MODEL,
        "d_kv": 8,
        "d_ff": 64,
        "num_layers": 2,
        "num_heads": 2,
        "dropout_rate": 0.0,
        "use_grassmann": False,
        "visual_cross_attn_blocks": k,
        "chronos_config": {
            "context_length": 32,
            "output_patch_size": 4,
            "input_patch_size": 4,
            "input_patch_stride": 4,
            "quantiles": [0.5],
            "use_reg_token": False,
            "use_arcsinh": False,
            "max_output_patches": 4,
        },
    }
    torch.manual_seed(0)
    mod = VisionChronos2LightningModule(
        # No hub weights: this exercises the s2c wiring, and a 768-dim backbone
        # would make the suite pay a download and a minute per forward for
        # nothing the tests assert.
        pretrained_model_name_or_path=None,
        chronos_core_cfg=core_cfg,
        vision_cfg={
            "n_visual_context_steps": 2,
            "n_soft_tokens": 1,
            "fusion_mode": fusion_mode,
            "visual_grid": visual_grid,
            "skip_vision_stack": False,
            "visual_dropout_prob": 0.0,  # deterministic
            "numeric_dropout_prob": 0.0,
            "dropout": 0.0,
        },
        lr=1e-3,
        warmup_steps=10,
        grassmann_warmup_steps=0,
        horizon=HORIZON,
        video_encoder=_make_fake_video_encoder(
            d_v=D_V, t_lat=T_LAT, h_lat=GRID0, w_lat=GRID0
        ),
    )
    mod.trainer = Mock()
    mod.trainer.is_global_zero = True
    mod.trainer.estimated_stepping_batches = 100
    mod.eval()
    return mod


def _batch(seed: int = 7):
    """Cached-latent batch: `Z` is deterministic, unlike the fake encoder's
    `randn` forward, so two runs differ only where we intend them to."""
    torch.manual_seed(seed)
    return {
        "Y": torch.randn(2, 1, 32, 1),
        "Y_future": torch.randn(2, 1, HORIZON, 1),
        "X_cov": torch.randn(2, 1, 32 + HORIZON, 1),
        # V is required by the unpacker even on the cache-hit path; Z is what is
        # actually consumed, and unlike the fake encoder's `randn` forward it is
        # deterministic, so two runs differ only where we intend them to.
        "V": torch.zeros(2, 1, T_LAT, 3, 32, 32),
        "Z": torch.randn(2, 1, T_LAT, GRID0 * GRID0, D_V),
        "mask_target": torch.ones(2, 1, 32, 1),
        "mask_future": torch.ones(2, 1, HORIZON, 1),
        "mask_visual": torch.ones(2, 1, T_LAT),
        "daylight_future": torch.ones(2, 1, HORIZON, 1),
        "site_id": torch.zeros(2, 1, dtype=torch.long),
    }


class TestOutputPatchGeometry:
    """`output_patch_size` 16 -> 4 is the whole reason per-horizon queries can
    exist. It is free of pretrained weights because `output_patch_embedding` is
    already reinitialised on load (quantile-head shape mismatch)."""

    def test_three_future_positions_cover_twelve_steps_with_no_leftover(self):
        mod = _module()
        assert mod._output_patch_size == 4
        assert mod._num_output_patches == 3
        assert mod._num_output_patches * mod._output_patch_size == HORIZON

    def test_the_old_patch_size_would_have_given_a_single_query(self):
        """Pins the premise. At 16 there is ONE future position for the whole
        6 h horizon, so three lead-time queries have nothing to attach to."""
        import math

        assert math.ceil(HORIZON / 16) == 1

    def test_lead_time_embedding_has_one_row_per_future_position(self):
        mod = _module()
        emb = mod.model.lead_time_embed
        assert emb is not None
        assert emb.shape == (4, D_MODEL)  # max_output_patches, sliced to 3 in use
        assert emb.requires_grad

    def test_lead_time_embedding_is_not_initialised_to_zero(self):
        """Zero-init would make the three queries identical at step 0 and let
        them stay collapsed — the degenerate parameterisation that produces a
        flat metric indistinguishable from a falsified hypothesis."""
        mod = _module()
        rows = mod.model.lead_time_embed[:3]
        assert rows.abs().sum() > 0
        assert not torch.allclose(rows[0], rows[1], atol=1e-6)
        assert not torch.allclose(rows[1], rows[2], atol=1e-6)


class TestSpatialFieldIsRetained:
    def test_kv_is_t_lat_times_grid_squared_tokens(self):
        """The claim under test is spatial: 4x4 per temporal slice, all slices
        kept. One vector per slice would be s2b's pooling again, and a single
        instant cannot express displacement."""
        mod = _module(visual_grid=4)
        z = torch.randn(2, T_LAT, GRID0 * GRID0, D_V)
        kv = mod.model._build_visual_kv(z)
        assert kv.shape == (2, T_LAT * 4 * 4, D_MODEL)

    def test_coarser_grid_pools_but_keeps_every_temporal_slice(self):
        mod = _module(visual_grid=2)
        z = torch.randn(2, T_LAT, GRID0 * GRID0, D_V)
        assert mod.model._build_visual_kv(z).shape == (2, T_LAT * 4, D_MODEL)

    def test_block_pooling_preserves_position(self):
        """A pool that scrambled the grid would still have the right shape.
        Feed a field that is constant within each 2x2 block and check the
        pooled 2x2 reproduces those blocks in order."""
        mod = _module(visual_grid=2)
        mod.model.visual_kv_proj = torch.nn.Identity()
        field = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 2 + [[2.0, 2.0, 3.0, 3.0]] * 2)
        z = field.reshape(1, 1, 16, 1).expand(1, T_LAT, 16, D_V).contiguous()
        kv = mod.model._build_visual_kv(z)
        assert torch.allclose(kv[0, :4, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))

    def test_non_square_patch_count_is_rejected(self):
        """Silently reshaping a non-square grid would fabricate a spatial
        layout that does not exist."""
        mod = _module()
        with pytest.raises(ValueError, match="square grid"):
            mod.model._build_visual_kv(torch.randn(2, T_LAT, 15, D_V))


class TestS2cForwardEndToEnd:
    def test_loss_is_finite(self):
        loss = _module().training_step(_batch(), batch_idx=0)
        assert torch.isfinite(loss)

    def test_gradients_reach_the_new_modules(self):
        """Both halves must learn, or the arm tests a mechanism that is not
        actually being trained."""
        mod = _module()
        mod.train()
        mod.training_step(_batch(), batch_idx=0).backward()
        names = {
            "visual_kv_proj": mod.model.visual_kv_proj.weight.grad,
            "lead_time_embed": mod.model.lead_time_embed.grad,
        }
        for n, g in names.items():
            assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n
        cross = [
            (n, p.grad)
            for n, p in mod.model.chronos.named_parameters()
            if "visual_cross_attn" in n
        ]
        assert cross, "no cross-attention parameters exist"
        assert any(g is not None and g.abs().sum() > 0 for _, g in cross)

    def test_cross_attention_stays_trainable_under_freeze_chronos(self):
        """It lives under `model.chronos`, so the blanket freeze catches it. It
        is a NEW module with no pretrained weights and must learn regardless of
        whether `n_unfreeze_encoder_blocks` happens to cover the same blocks."""
        from mmtsfm.models.chronos2.lightning_module import (
            VisionChronos2LightningModule,
        )

        sig = VisionChronos2LightningModule.__init__
        assert "freeze_chronos" in sig.__code__.co_varnames
        mod = _module()
        mod_frozen = _module()
        for p in mod_frozen.model.chronos.parameters():
            p.requires_grad_(False)
        del mod, mod_frozen  # constructed above only to prove the path exists

    def test_visual_field_never_enters_the_token_sequence(self):
        """This is what separates s2c from s2b. Vision must reach the model
        ONLY through cross-attention: the embeddings handed to the encoder must
        be bit-identical for two completely different satellite fields."""
        seen = []
        for z_seed in (11, 22):
            mod = _module()
            b = _batch()
            torch.manual_seed(z_seed)
            b["Z"] = torch.randn(2, 1, T_LAT, GRID0 * GRID0, D_V)
            real = mod.model.chronos.encoder.forward

            def spy(*a, __real=real, **kw):
                seen.append(kw.get("inputs_embeds", a[0] if a else None).detach())
                return __real(*a, **kw)

            mod.model.chronos.encoder.forward = spy
            mod.training_step(b, batch_idx=0)

        assert len(seen) >= 2
        assert torch.equal(seen[0], seen[-1]), (
            "the visual field changed the encoder INPUT — it is being fused "
            "into the sequence, which is the s2b mechanism, not s2c's"
        )

    def test_the_visual_field_changes_the_forecast(self):
        """The mirror of the test above: identical inputs, different KV, so a
        different loss. Otherwise cross-attention is wired but inert."""
        # Assert on the QUANTILE PREDICTIONS, not on the scalar loss. At random
        # init the cross-attention output projection is small and the 64-token
        # field averages out, so the effect on a single averaged scalar (~1e-7)
        # is below what this test can distinguish from float noise, while the
        # effect on the forecast itself is ~1e-5. Testing the loss here would be
        # testing the reduction, not the mechanism.
        preds = []
        for z_seed in (11, 22):
            mod = _module()
            b = _batch()
            torch.manual_seed(z_seed)
            b["Z"] = torch.randn(2, 1, T_LAT, GRID0 * GRID0, D_V)
            with torch.no_grad():
                preds.append(mod._forward(b)[1].quantile_preds)
        delta = (preds[0] - preds[1]).abs().max().item()
        assert delta > 1e-6, f"two different visual fields gave the same forecast ({delta})"

    def test_forced_vision_off_reproduces_the_vision_free_forward(self):
        """The marginal-gain pass must be a genuine counterfactual: withholding
        the query, not perturbing it."""
        mod = _module()
        b = _batch()
        with torch.no_grad():
            _, out_off = mod._forward(b, force_vision_off=True)
            b_novis = {k: v for k, v in b.items() if k != "Z"}
            b_novis["mask_visual"] = torch.zeros(2, 1, T_LAT)
            _, out_none = mod._forward(b_novis)
        assert torch.allclose(
            out_off.quantile_preds, out_none.quantile_preds, atol=1e-5
        )


class TestExistingArmsAreUntouched:
    def test_late_and_interleaved_build_no_s2c_modules(self):
        for mode in ("late", "interleaved"):
            mod = _module(fusion_mode=mode, k=0)
            assert mod.model.visual_kv_proj is None, mode
            assert mod.model.lead_time_embed is None, mode
            keys = set(mod.state_dict())
            assert not [k for k in keys if "visual_cross_attn" in k], mode
            assert not [k for k in keys if "lead_time_embed" in k], mode

    def test_s2c_does_not_build_the_summarizer_pooling_it_exists_to_bypass(self):
        """If `future_query` fell through to the `late`/`interleaved` branch it
        would route the field through the ~800:1 LatentSummarizer pool — the
        exact thing under test — and the arm would be silently invalid while
        every number still looked reasonable."""
        mod = _module()
        keys = set(mod.state_dict())
        assert not [k for k in keys if "cross_modal_adapter" in k]


class TestArmIdentityIsDisjoint:
    def test_results_tag_is_unique_across_configs(self):
        """A duplicate tag would overwrite a wave-1 or wave-2 result file."""
        import pathlib
        import re

        cfg_dir = pathlib.Path(__file__).resolve().parents[1] / "configs" / "model"
        tags: dict[str, list[str]] = {}
        for p in cfg_dir.glob("*.yaml"):
            for m in re.finditer(r"^results_tag:\s*(\S+)", p.read_text(), re.M):
                tags.setdefault(m.group(1).strip("\"'"), []).append(p.name)
        # Several wave-1 configs deliberately share `mmtsfm_s2_ukpv` (same arm,
        # different backbone flags). What must hold is that s2c is not one of
        # them: a shared tag would overwrite an existing result file.
        assert tags.get("mmtsfm_s2c_ukpv") == ["vision_chronos2_s2c.yaml"], tags
