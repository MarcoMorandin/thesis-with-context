"""Unit tests for the §4.5 statistics (DM test, block bootstrap, Holm)."""

from __future__ import annotations

import numpy as np

from common.aggregate import average_rank, geometric_mean_skill, win_rate
from common.stats import block_bootstrap_ci, dm_test, holm_bonferroni


def test_dm_equal_losses_not_significant():
    rng = np.random.default_rng(0)
    loss = rng.random(500)
    noise = loss + rng.normal(0, 1e-9, 500)
    result = dm_test(loss, noise)
    assert result["p_value"] > 0.5


def test_dm_clearly_better_model_is_significant():
    rng = np.random.default_rng(1)
    base = rng.random(500) + 1.0
    better = base - 0.5 + rng.normal(0, 0.05, 500)
    result = dm_test(better, base)
    assert result["stat"] < 0          # negative ⇒ first argument better
    assert result["p_value"] < 1e-6
    assert result["mean_diff"] < 0


def test_dm_t_distribution_tail_sane():
    """p-value for |t|=1.96 with large df ≈ 0.05 (normal limit)."""
    rng = np.random.default_rng(2)
    n = 2000
    d = rng.normal(0, 1, n)
    d = (d - d.mean()) / d.std() * 1.0
    d += 1.96 / np.sqrt(n)             # shift mean to put stat at ≈1.96
    result = dm_test(d + 1.0, np.ones(n))
    assert 0.02 < result["p_value"] < 0.10


def test_block_bootstrap_detects_difference():
    rng = np.random.default_rng(3)
    days = np.repeat(np.arange(50), 10)
    base = rng.random(500) + 1.0
    better = base - 0.3
    ci = block_bootstrap_ci(better, base, blocks=days, n_resamples=200, seed=0)
    assert ci["significant"]
    assert ci["ci_high"] < 0
    assert np.isclose(ci["mean_diff"], -0.3)


def test_block_bootstrap_no_difference():
    rng = np.random.default_rng(4)
    days = np.repeat(np.arange(50), 10)
    a = rng.random(500)
    ci = block_bootstrap_ci(a, a.copy(), blocks=days, n_resamples=100, seed=0)
    assert not ci["significant"]
    assert ci["mean_diff"] == 0.0


def test_holm_bonferroni_stepdown():
    p = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}
    out = holm_bonferroni(p, alpha=0.05)
    assert out["a"]["reject"]
    assert not out["d"]["reject"]
    # adjusted p-values are monotone in the original ordering
    assert out["a"]["p_adjusted"] <= out["b"]["p_adjusted"] \
        <= out["c"]["p_adjusted"] <= out["d"]["p_adjusted"]
    # smallest p multiplied by the family size
    assert np.isclose(out["a"]["p_adjusted"], 0.004)


def test_win_rate_and_geometric_skill():
    model = {"p1": 0.08, "p2": 0.12, "p3": 0.05}
    ref = {"p1": 0.10, "p2": 0.10, "p3": 0.10}
    assert win_rate(model, ref) == 2 / 3
    ss = geometric_mean_skill(model, ref)
    expected = 1.0 - np.exp(np.mean(np.log([0.8, 1.2, 0.5])))
    assert np.isclose(ss, expected)


def test_average_rank_with_ties():
    metric = {
        "m1": {"d1": 0.1, "d2": 0.2},
        "m2": {"d1": 0.2, "d2": 0.2},
        "m3": {"d1": 0.3, "d2": 0.1},
    }
    ranks = average_rank(metric)
    assert ranks["m1"] == (1 + 2.5) / 2
    assert ranks["m2"] == (2 + 2.5) / 2
    assert ranks["m3"] == (3 + 1) / 2
