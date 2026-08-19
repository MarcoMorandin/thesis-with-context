from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from scripts.probes.fit_ceiling import fit_and_score


def _arrays(n, d_vis=32, d_cov=8, horizon=4, vis_weight=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X_vis = rng.normal(size=(n, d_vis)).astype(np.float32)
    X_cov = rng.normal(size=(n, d_cov)).astype(np.float32)
    # target = a covariate-driven part + an optional vision-only part
    base = X_cov[:, :1] * 0.5
    vis = (X_vis[:, :1] * vis_weight).astype(np.float32)
    Y = np.repeat(base + vis, horizon, axis=1).astype(np.float32)
    Y += rng.normal(scale=0.01, size=Y.shape).astype(np.float32)
    return {
        "X_vis": X_vis,
        "X_cov": X_cov,
        "Y": Y,
        "Y_mask": np.ones_like(Y, dtype=bool),
        "site": np.array([str(i % 7) for i in range(n)], dtype=object),
        "origin": np.arange(n, dtype=np.int64),
    }


def test_recovers_a_planted_visual_signal():
    """If vision genuinely carries signal, (c)-(b) must be clearly positive."""
    tr = _arrays(4000, vis_weight=1.0, seed=1)
    te = _arrays(1000, vis_weight=1.0, seed=2)
    res = fit_and_score(tr, te, horizon=4)
    assert min(res["conditional"]) > 0.05, res["conditional"]


def test_reports_no_conditional_signal_when_vision_is_noise():
    """Vision uncorrelated with the target must give (c)-(b) ~ 0."""
    tr = _arrays(4000, vis_weight=0.0, seed=3)
    te = _arrays(1000, vis_weight=0.0, seed=4)
    res = fit_and_score(tr, te, horizon=4)
    assert max(res["conditional"]) < 0.02, res["conditional"]


def test_covariate_probe_beats_trivial_baseline():
    """Sanity guard: if (b) cannot beat the mean predictor, feature assembly is wrong."""
    tr, te = _arrays(2000, seed=5), _arrays(500, seed=6)
    res = fit_and_score(tr, te, horizon=4)
    assert min(res["b"]["skill"]) > 0.0


def test_cv_spread_is_reported_per_horizon():
    tr, te = _arrays(2000, seed=7), _arrays(500, seed=8)
    res = fit_and_score(tr, te, horizon=4)
    assert len(res["cv_spread"]) == 4
    assert all(v >= 0.0 for v in res["cv_spread"])
    assert len(res["cv_spread_rel"]) == 4
    assert all(v >= 0.0 for v in res["cv_spread_rel"])


def test_conditional_rel_recovers_a_planted_visual_signal():
    """Reference-free companion to conditional: must also clear a positive bar.

    conditional_rel is a *fraction of (b)'s error removed*, so a correct
    implementation is bounded above by 1.0. Pins the denominator too: a
    mutation that divides by nmae_c instead of nmae_b inflates this value
    by ~2 orders of magnitude (still > 0.05, but blown through the <= 1.0
    ceiling and disagreeing with the direct algebraic recomputation below).
    """
    tr = _arrays(4000, vis_weight=1.0, seed=1)
    te = _arrays(1000, vis_weight=1.0, seed=2)
    res = fit_and_score(tr, te, horizon=4)
    for h, v in enumerate(res["conditional_rel"]):
        assert 0.05 < v <= 1.0, res["conditional_rel"]
        expected = (res["b"]["nmae"][h] - res["c"]["nmae"][h]) / res["b"]["nmae"][h]
        assert v == pytest.approx(expected), (h, v, expected)


def test_conditional_rel_is_near_zero_when_vision_is_noise():
    tr = _arrays(4000, vis_weight=0.0, seed=3)
    te = _arrays(1000, vis_weight=0.0, seed=4)
    res = fit_and_score(tr, te, horizon=4)
    assert max(res["conditional_rel"]) < 0.02, res["conditional_rel"]
    for h, v in enumerate(res["conditional_rel"]):
        expected = (res["b"]["nmae"][h] - res["c"]["nmae"][h]) / res["b"]["nmae"][h]
        assert v == pytest.approx(expected), (h, v, expected)


def test_conditional_is_nan_not_zero_when_a_horizon_is_fully_masked():
    """A horizon with no test coverage must read as NaN ('no data'), never 0.0
    ('no signal') -- n_test_valid is what lets a reader tell the two apart."""
    tr = _arrays(2000, seed=9)
    te = _arrays(500, seed=10)
    te["Y_mask"][:, 2] = False
    res = fit_and_score(tr, te, horizon=4)
    assert np.isnan(res["conditional"][2])
    assert res["n_test_valid"][2] == 0
    # untouched horizons still report real data counts
    assert res["n_test_valid"][0] == 500


def test_block_penalties_prevent_a_false_negative_from_overfitting():
    """A wide noise block must read as ~0, not as "vision hurts".

    This is the failure the first real G0 run hit: set (c) carries ~40x set
    (b)'s dimensionality, and at a single fixed alpha the extra columns fit
    noise, so conditional_rel comes back RELIABLY negative. Read literally that
    says vision degrades the forecast; it only says the probe overfit, and it
    would have stopped the study on a false negative.

    The second half re-runs the identical data at the old fixed alpha=1.0 and
    asserts the pathology is present there -- without it this test could pass
    against a probe that never had the problem to begin with, and would silently
    lose its power if the fixture were ever made easier.
    """
    tr = _arrays(1200, d_vis=400, vis_weight=0.0, seed=11)
    te = _arrays(600, d_vis=400, vis_weight=0.0, seed=12)

    selected = fit_and_score(tr, te, horizon=4)["conditional_rel"]
    fixed = fit_and_score(tr, te, horizon=4, alphas=[1.0])["conditional_rel"]

    assert min(fixed) < -0.10, ("fixture no longer overfits at alpha=1", fixed)
    assert min(selected) > -0.02, selected
    assert max(selected) < 0.02, selected


def test_standardization_makes_the_fit_invariant_to_feature_scaling():
    """Rescaling a whole block must not change the answer.

    Ridge penalizes every coefficient equally, so on RAW features the visual
    block's activation scale silently sets how hard it is regularized relative
    to the covariates -- an arbitrary property of the encoder deciding the gate.
    """
    tr, te = (
        _arrays(2000, vis_weight=1.0, seed=13),
        _arrays(500, vis_weight=1.0, seed=14),
    )
    base = fit_and_score(tr, te, horizon=4)

    for arrays in (tr, te):
        arrays["X_vis"] = arrays["X_vis"] * 1000.0
    scaled = fit_and_score(tr, te, horizon=4)

    assert scaled["conditional_rel"] == pytest.approx(
        base["conditional_rel"], rel=1e-4, abs=1e-6
    )
    assert scaled["c"]["nmae"] == pytest.approx(base["c"]["nmae"], rel=1e-4, abs=1e-6)


def test_selected_alpha_is_reported_for_every_set_and_horizon():
    """alpha_selected is a diagnostic, not bookkeeping: an alpha pinned to a grid
    edge means the grid was too narrow and the result cannot be trusted."""
    tr, te = _arrays(2000, seed=15), _arrays(500, seed=16)
    res = fit_and_score(tr, te, horizon=4)
    grid = res["alpha_grid"]
    for which in ("a", "b", "c"):
        chosen = res["alpha_selected"][which]
        assert len(chosen) == 4
        assert all(v in grid for v in chosen), (which, chosen)


def test_single_plant_falls_back_instead_of_crashing_groupkfold():
    """GroupKFold cannot split one group. The probe must record the fallback
    alpha and a NaN spread rather than raising or inventing a spread of 0."""
    tr, te = _arrays(400, seed=17), _arrays(200, seed=18)
    tr["site"] = np.array(["only"] * 400, dtype=object)
    res = fit_and_score(tr, te, horizon=4)
    assert all(np.isnan(v) for v in res["cv_spread"])
    assert all(np.isnan(v) for v in res["cv_spread_rel"])
    mid = res["alpha_grid"][len(res["alpha_grid"]) // 2]
    assert res["alpha_selected"]["c"] == [mid] * 4


def test_still_recovers_signal_when_the_visual_block_is_wide():
    """Power check on the same wide fixture as the false-negative test.

    Suppressing a wide NOISE block is only useful if a wide INFORMATIVE block
    still gets through. Without this, a probe that answered "zero" to everything
    would pass the whole suite and the study would stop on a dead instrument.
    """
    tr = _arrays(1200, d_vis=400, vis_weight=1.0, seed=11)
    te = _arrays(600, d_vis=400, vis_weight=1.0, seed=12)
    res = fit_and_score(tr, te, horizon=4)
    assert min(res["conditional_rel"]) > 0.05, res["conditional_rel"]
    assert min(res["conditional"]) > 0.05, res["conditional"]
