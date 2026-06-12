"""Metric correctness against hand-computed values (§6.4-6.5)."""

import numpy as np

from common import metrics
from common.metrics import PerPlantAccumulator


def test_nmae_nrmse_hand_computed():
    y = np.array([[0.5, 0.5]])
    p = np.array([[0.7, 0.1]])
    mask = np.ones_like(y)
    assert np.isclose(metrics.nmae(y, p, mask), (0.2 + 0.4) / 2)
    assert np.isclose(metrics.nrmse(y, p, mask), np.sqrt((0.04 + 0.16) / 2))


def test_masking_excludes_steps():
    y = np.array([[0.5, 0.5]])
    p = np.array([[0.7, 99.0]])  # huge masked error must not contribute
    mask = np.array([[1.0, 0.0]])
    assert np.isclose(metrics.nmae(y, p, mask), 0.2)


def test_skill_score_reference_is_zero():
    assert metrics.skill_score(0.1, 0.1) == 0.0
    assert metrics.skill_score(0.05, 0.1) == 0.5


def test_pinball_median_is_half_mae():
    y = np.random.default_rng(0).random((4, 3))
    pred = np.random.default_rng(1).random((4, 3, 1))
    mask = np.ones((4, 3))
    pin = metrics.pinball_loss(y, pred, np.array([0.5]), mask)
    mae = np.abs(y - pred[..., 0]).mean()
    assert np.isclose(pin[0], 0.5 * mae)


def test_crps_perfect_forecast_is_zero():
    y = np.random.default_rng(0).random((5, 4))
    levels = np.array([0.1, 0.5, 0.9])
    perfect = np.repeat(y[..., None], 3, axis=-1)
    assert metrics.crps(y, perfect, levels, np.ones_like(y)) == 0.0


def test_coverage_hand_computed():
    y = np.array([[0.5, 0.5, 0.5, 0.5]])
    lo = np.array([[0.4, 0.6, 0.4, 0.4]])
    hi = np.array([[0.6, 0.7, 0.45, 0.6]])
    cov = metrics.empirical_coverage(y, lo, hi, np.ones_like(y))
    assert np.isclose(cov, 0.5)  # inside, below, above, inside


def test_quantile_ece_perfect_calibration():
    rng = np.random.default_rng(0)
    y = rng.random((10000, 1))
    levels = np.array([0.1, 0.5, 0.9])
    preds = np.broadcast_to(levels, (10000, 1, 3))  # true quantiles of U(0,1)
    ece = metrics.quantile_ece(y, preds, levels, np.ones((10000, 1)))
    assert ece < 0.02


def test_accumulator_macro_averages_per_plant():
    """A plant with many samples must not dominate the macro average."""
    acc = PerPlantAccumulator()
    big = np.zeros((100, 2))
    acc.update(np.array(["a"] * 100), big, big + 0.1, np.ones_like(big))
    small = np.zeros((1, 2))
    acc.update(np.array(["b"]), small, small + 0.3, np.ones_like(small))
    macro = acc.macro()
    assert np.isclose(macro["nmae"], (0.1 + 0.3) / 2)
    assert macro["n_plants"] == 2


def test_accumulator_crps_matches_direct():
    rng = np.random.default_rng(0)
    y = rng.random((8, 3))
    q = np.sort(rng.random((8, 3, 9)), axis=-1)
    mask = np.ones_like(y)
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    acc = PerPlantAccumulator()
    acc.update(np.array(["p"] * 8), y, q[..., 4], mask, quantile_preds=q)
    direct = metrics.crps(y, q, levels, mask)
    assert np.isclose(acc.per_plant()["p"]["crps"], direct)
