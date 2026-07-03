"""Regression test: _init_weights must cover ALL CausalGrassmannMixing params.

from_pretrained materializes the model on `meta` + to_empty(); params missing
from the checkpoint (all grassmann params) only get values via _init_weights.
Any param it skips keeps uninitialized GPU garbage — observed as random NaN in
offset_weights / modality_pair_bias (blocks 0,5,7,11 of the
mmtsfm_grassmann_interleaved_ukpv run), which froze training via the
NaN-grad-zeroing safety hook.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from mmtsfm.models.chronos2 import Chronos2CoreConfig, Chronos2Model
from mmtsfm.models.chronos2.grassmann import CausalGrassmannMixing


def _make_model():
    cfg = Chronos2CoreConfig(
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        use_grassmann=True,
        grassmann_reduced_dim=8,
        grassmann_window_offsets=[1, 2, 4],
        grassmann_modality_pair_bias=True,
        chronos_config={
            "context_length": 64,
            "output_patch_size": 8,
            "input_patch_size": 8,
            "input_patch_stride": 8,
            "quantiles": [0.1, 0.5, 0.9],
            "use_reg_token": False,
            "use_arcsinh": False,
            "max_output_patches": 4,
        },
    )
    return Chronos2Model(cfg)


def test_init_weights_reinitializes_every_grassmann_param():
    """Poison every grassmann param with NaN (simulating to_empty() garbage),
    re-run _init_weights, and require every param to be finite again."""
    model = _make_model()
    mixers = [m for m in model.modules() if isinstance(m, CausalGrassmannMixing)]
    assert mixers, "config should build grassmann mixing layers"

    for mixer in mixers:
        for p in mixer.parameters():
            p.data.fill_(float("nan"))
        model._init_weights(mixer)
        # layer_norm is a child module initialized via its own _init_weights call
        model._init_weights(mixer.layer_norm)

    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not bad, f"_init_weights left uninitialized params: {bad}"


def test_init_weights_grassmann_values():
    """offset_weights must start as ones (uniform offset mixing) and
    modality_pair_bias as zeros (no initial pair-type preference)."""
    model = _make_model()
    mixer = next(m for m in model.modules() if isinstance(m, CausalGrassmannMixing))
    mixer.offset_weights.data.fill_(float("nan"))
    mixer.modality_pair_bias.data.fill_(float("nan"))
    model._init_weights(mixer)
    assert torch.equal(mixer.offset_weights.data, torch.ones_like(mixer.offset_weights))
    assert torch.equal(
        mixer.modality_pair_bias.data, torch.zeros_like(mixer.modality_pair_bias)
    )
