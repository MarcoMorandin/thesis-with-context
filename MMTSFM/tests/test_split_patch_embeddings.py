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
