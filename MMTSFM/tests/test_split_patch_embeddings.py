"""s2c needs output_patch_size < input_patch_size, and nothing tested that the
model could be BUILT that way.

The first launch of the arm died at ``Chronos2Model.__init__`` on the cluster,
after queueing, because every s2c test so far constructs a ``Chronos2EncoderBlock``
or a ``SimpleNamespace`` stand-in and never the real model. Upstream's assert was
load-bearing: ``encode`` embeds the context patches AND the future patches with
the same ``input_patch_embedding``, and a future patch is
``[time_enc, covariates, mask]`` each *output*_patch_size wide. These tests pin
both halves of the fix — unequal sizes work, and equal sizes are untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmtsfm.models.chronos2.config import Chronos2CoreConfig  # noqa: E402
from mmtsfm.models.chronos2.model import Chronos2Model  # noqa: E402

S2C_CFG = Path(__file__).resolve().parents[1] / "configs/model/vision_chronos2_s2c.yaml"


def _core(chronos_config: dict) -> Chronos2CoreConfig:
    """A small backbone around a REAL chronos_config. The bug lives entirely in
    the patch-embedding widths, so the encoder is shrunk and cross-attention is
    off — this is about construction and the future path, not fusion."""
    return Chronos2CoreConfig(
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        use_grassmann=False,
        visual_cross_attn_blocks=0,
        chronos_config=chronos_config,
    )


def _shipped_chronos_config() -> dict:
    return yaml.safe_load(S2C_CFG.read_text())["chronos_core_cfg"]["chronos_config"]


def _forward(model, *, history=672, horizon=12, num_output_patches=3):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(
            context=torch.randn(2, history),
            future_target=torch.randn(2, horizon),
            future_target_mask=torch.ones(2, horizon),
            num_output_patches=num_output_patches,
        )


def test_the_shipped_s2c_config_builds_and_runs():
    """The exact configuration that was submitted to SLURM. Reads the YAML rather
    than restating it, so reverting output_patch_size to 16 breaks this test
    instead of silently collapsing the horizon back to one position."""
    cc = _shipped_chronos_config()
    assert cc["output_patch_size"] != cc["input_patch_size"], (
        "s2c needs unequal patch sizes; equal sizes give ceil(12/16) = 1 future "
        "position and the three lead-time queries have nothing to be"
    )
    out = _forward(Chronos2Model(_core(cc)).eval())
    assert out.loss is not None and torch.isfinite(out.loss)
    # 3 positions x 4 steps covers the 12 scored steps exactly.
    assert out.quantile_preds.shape[-1] == 12


def test_the_future_embedding_is_shaped_for_the_output_patch():
    model = Chronos2Model(_core(_shipped_chronos_config()))
    cc = model.chronos_config
    assert model.future_patch_embedding is not None
    assert model.future_patch_embedding.hidden_layer.in_features == (
        cc.output_patch_size * 3
    )
    # The precise mismatch that raised at the first future token: the context
    # embedding is 3x wider and cannot take this input at all.
    assert model.input_patch_embedding.hidden_layer.in_features == (
        cc.input_patch_size * 3
    )
    with pytest.raises(RuntimeError):
        model.input_patch_embedding(torch.randn(2, 3, cc.output_patch_size * 3))


def test_equal_patch_sizes_build_nothing_new():
    """s1/s2a/s2b/s3 must keep their exact parameter set and checkpoint keys. A
    module that exists but is unused would still change the state dict, so the
    same discipline visual_cross_attn_blocks: 0 follows applies here."""
    cc = dict(_shipped_chronos_config(), output_patch_size=16)
    model = Chronos2Model(_core(cc))
    assert model._split_patch_embeddings is False
    assert model.future_patch_embedding is None
    assert not [k for k in model.state_dict() if "future_patch_embedding" in k]


def test_the_split_model_carries_the_extra_keys():
    keys = Chronos2Model(_core(_shipped_chronos_config())).state_dict()
    assert [k for k in keys if k.startswith("future_patch_embedding.")]


def test_the_freeze_policy_does_not_leave_it_frozen():
    """Fresh weights with no pretrained and no warm-started counterpart. The
    substring list is matched by ``in``, and "input_patch_embedding" does NOT
    match "future_patch_embedding" — s2c would train a random projection on every
    future position, which is the whole forecast."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src/mmtsfm/models/chronos2/lightning_module.py"
    ).read_text()
    assert '"future_patch_embedding",' in src


def test_unequal_sizes_reach_the_future_positions():
    """Guards against a fix that builds the module and never calls it: zeroing
    the future embedding must change the forecast."""
    model = Chronos2Model(_core(_shipped_chronos_config())).eval()
    before = _forward(model).quantile_preds.clone()
    with torch.no_grad():
        for p in model.future_patch_embedding.parameters():
            p.zero_()
    after = _forward(model).quantile_preds
    assert not torch.allclose(before, after)


# --- warm start ------------------------------------------------------------
# s2c is chained off s1, and the second cluster failure was here, not in
# construction: strict=False skips ABSENT keys but raises on keys present at the
# wrong shape, and the resized quantile head is present in every s1 donor.


def _donor_state_dict() -> dict:
    """An s1-shaped donor: the same model with the original patch size."""
    cc = dict(_shipped_chronos_config(), output_patch_size=16)
    return Chronos2Model(_core(cc)).state_dict()


def test_an_s1_donor_warm_starts_into_s2c():
    from mmtsfm.train import drop_reshaped_tensors

    model = Chronos2Model(_core(_shipped_chronos_config()))
    sd, n = drop_reshaped_tensors(_donor_state_dict(), model.state_dict(), "donor.ckpt")
    assert n == 4, "expected exactly the quantile head's 4 tensors"
    assert all("output_patch_embedding" in k for k in _donor_state_dict() if k not in sd)
    # The load that raised on the cluster.
    missing, _ = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k.startswith("encoder.")], (
        "the backbone must still warm-start; only the resized head is dropped"
    )


def test_a_matching_donor_is_returned_untouched():
    from mmtsfm.train import drop_reshaped_tensors

    model = Chronos2Model(_core(_shipped_chronos_config()))
    sd = model.state_dict()
    out, n = drop_reshaped_tensors(dict(sd), sd, "donor.ckpt")
    assert n == 0 and out.keys() == sd.keys()


def test_a_mismatched_donor_is_refused_not_silently_discarded():
    """The failure mode this guard exists for: pointing INIT_CKPT at another
    arm would drop most of the tensors and train from scratch while every log
    line said the stage was chained."""
    from mmtsfm.train import drop_reshaped_tensors

    model = Chronos2Model(_core(_shipped_chronos_config()))
    wrong = {k: torch.zeros(3, 3) for k in model.state_dict()}
    with pytest.raises(RuntimeError, match="mismatched donor"):
        drop_reshaped_tensors(wrong, model.state_dict(), "wrong_arm.ckpt")


# --- the arm that actually runs --------------------------------------------
# Everything above builds a bare Chronos2Model, and the third cluster failure
# went straight past all of it: VisionChronos2Model overrides `forward` and
# inlines its own copy of the encode path, so fixing `Chronos2Model.encode` left
# the vision arm — the ONLY arm that sets output_patch_size != input_patch_size —
# still embedding future patches with the context embedding. It died in the
# sanity check on `mat1 and mat2 shapes cannot be multiplied (12x12 and
# 48x3072)`: 12 = output_patch_size * 3, 48 = input_patch_size * 3.


def _vision_model(video_encoder=None):
    from test_vision_chronos2 import _make_fake_video_encoder

    from mmtsfm.models.chronos2 import VisionChronos2Config, VisionChronos2Model

    return VisionChronos2Model(
        chronos_model=Chronos2Model(_core(_shipped_chronos_config())),
        vision_config=VisionChronos2Config(
            n_visual_context_steps=4, visual_dropout_prob=0.0, dropout=0.0
        ),
        video_encoder=video_encoder or _make_fake_video_encoder(d_v=4),
    ).eval()


@pytest.mark.parametrize("with_video", [False, True])
@pytest.mark.parametrize("with_cov", [False, True])
def test_the_vision_arm_forwards_at_the_shipped_patch_sizes(with_video, with_cov):
    """The forward that ran on the cluster. Three axes matter: the future
    embedding sits BEFORE the visual stream, so vision-off rows hit it too and
    `force_vision_off` makes the validation loop take that branch on purpose; and
    the covariate rows call `_prepare_patched_future` a SECOND time, so they are
    output_patch_size wide as well. uk_pv passes weather covariates, so the arm
    would have died there next."""
    model = _vision_model()
    b, horizon = 2, 12
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.forward(
            context=torch.randn(b, 672),
            future_target=torch.randn(b, horizon),
            future_target_mask=torch.ones(b, horizon),
            group_ids=torch.arange(b),
            covariate_channels=[torch.rand(b, horizon)] if with_cov else None,
            video=torch.rand(b, 3, 8, 32, 32) if with_video else None,
            num_output_patches=3,
        )
    assert out.quantile_preds.shape[-1] == horizon
    assert torch.isfinite(out.quantile_preds).all()


def test_the_vision_arm_uses_the_future_embedding_not_the_context_one():
    """Guards the specific line that was wrong. Zeroing future_patch_embedding
    must move the forecast; zeroing it while the vision path still called
    input_patch_embedding would leave the output untouched."""
    model = _vision_model()
    kw = dict(
        context=torch.randn(2, 672),
        future_target=torch.randn(2, 12),
        future_target_mask=torch.ones(2, 12),
        group_ids=torch.arange(2),
        covariate_channels=[torch.rand(2, 12)],
        num_output_patches=3,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        before = model.forward(**kw).quantile_preds.clone()
        for p in model.chronos.future_patch_embedding.parameters():
            p.zero_()
        after = model.forward(**kw).quantile_preds
    assert not torch.allclose(before, after)

