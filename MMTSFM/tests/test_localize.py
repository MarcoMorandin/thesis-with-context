from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from scripts.probes.localize import decompose_by_horizon, stratify_by_variability


def test_decompose_finds_a_short_horizon_only_effect():
    """The C5 signature: vision helps at h=0..1 and not beyond."""
    n, h = 500, 6
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n, h))
    pred_off = y + rng.normal(scale=0.5, size=(n, h))
    pred_on = pred_off.copy()
    pred_on[:, :2] = y[:, :2] + rng.normal(scale=0.1, size=(n, 2))  # much better early
    res = decompose_by_horizon(pred_on, pred_off, y, np.ones((n, h), bool))
    assert res["delta"][0] > 0.1 and res["delta"][1] > 0.1
    assert abs(res["delta"][4]) < 0.05 and abs(res["delta"][5]) < 0.05


def test_decompose_respects_the_mask():
    n, h = 100, 3
    y = np.zeros((n, h))
    pred_on = np.ones((n, h))
    pred_off = np.full((n, h), 2.0)
    mask = np.zeros((n, h), bool)
    mask[:, 0] = True
    res = decompose_by_horizon(pred_on, pred_off, y, mask)
    assert res["nmae_on"][0] == 1.0
    assert np.isnan(res["nmae_on"][1])


def test_stratify_orders_bins_by_variability():
    delta = np.concatenate([np.zeros(300), np.full(300, 0.2)])
    csi_var = np.concatenate([np.zeros(300), np.ones(300)])
    res = stratify_by_variability(delta, csi_var, n_bins=2)
    assert res["mean_delta"][0] < res["mean_delta"][1]
    assert res["counts"] == [300, 300]


def test_gate_stats_flags_an_untrained_modality_bias(tmp_path):
    import torch

    from scripts.probes.localize import gate_stats

    ckpt = tmp_path / "c.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.chronos.encoder.block.0.layer.0.modality_pair_bias": torch.zeros(
                    4
                ),
                "model.chronos.encoder.block.0.layer.0.W_gate.bias": torch.full(
                    (8,), 2.0
                ),
                "model.chronos.encoder.block.1.layer.0.modality_pair_bias": torch.tensor(
                    [0.3, -0.2, 0.1, 0.0]
                ),
                "model.chronos.encoder.block.1.layer.0.W_gate.bias": torch.zeros(8),
            }
        },
        ckpt,
    )
    res = gate_stats(str(ckpt))
    assert res["n_blocks_with_zero_modality_bias"] == 1
    assert res["per_block"]["0"]["w_gate_bias_mean"] == 2.0
    assert res["per_block"]["1"]["modality_pair_bias_absmean"] > 0.0
