"""Unit tests for the Chronos-2 covariate task builder."""

from __future__ import annotations

import numpy as np

from common import config
from tier3.build_chronos_inputs import build_chronos_inputs

T, H, C = 6, 3, len(config.COV_COLS)


def _batch(n=2):
    rng = np.random.default_rng(0)
    return {
        "y_hist": rng.random((n, T)).astype(np.float32),
        "mask_hist": np.ones((n, T), np.float32),
        "y_future": rng.random((n, H)).astype(np.float32),
        "mask_future": np.ones((n, H), np.float32),
        "cov": rng.random((n, T + H, C)).astype(np.float32),
    }


def test_predict_exposes_only_selected_future_covariates():
    batch = _batch()
    det = config.DETERMINISTIC_COV_IDX
    rows = build_chronos_inputs(batch, det, mode="predict")

    assert len(rows) == 2
    r = rows[0]
    # target is history only
    assert r["target"].shape == (T,)
    # every covariate is available over the past
    assert set(r["past_covariates"]) == set(config.COV_COLS)
    assert all(v.shape == (T,) for v in r["past_covariates"].values())
    # only the deterministic columns are known into the future, horizon-length
    assert set(r["future_covariates"]) == {config.COV_COLS[c] for c in det}
    assert all(v.shape == (H,) for v in r["future_covariates"].values())
    # future-covariate keys must be a subset of past-covariate keys (Chronos-2 rule)
    assert set(r["future_covariates"]) <= set(r["past_covariates"])


def test_oracle_exposes_all_future_covariates():
    all_idx = tuple(range(C))
    rows = build_chronos_inputs(_batch(), all_idx, mode="predict")
    assert set(rows[0]["future_covariates"]) == set(config.COV_COLS)


def test_predict_masks_history_as_nan():
    batch = _batch(n=1)
    batch["mask_hist"][0, 0] = 0.0
    rows = build_chronos_inputs(batch, config.DETERMINISTIC_COV_IDX, mode="predict")
    assert np.isnan(rows[0]["target"][0])
    assert np.isfinite(rows[0]["target"][1:]).all()


def test_fit_uses_full_series_and_marks_future_keys_none():
    det = config.DETERMINISTIC_COV_IDX
    rows = build_chronos_inputs(_batch(), det, mode="fit")
    r = rows[0]
    # target spans history + horizon
    assert r["target"].shape == (T + H,)
    # past covariates span the full series
    assert all(v.shape == (T + H,) for v in r["past_covariates"].values())
    # future-covariate values are unused during fit (flag-only)
    assert set(r["future_covariates"]) == {config.COV_COLS[c] for c in det}
    assert all(v is None for v in r["future_covariates"].values())
