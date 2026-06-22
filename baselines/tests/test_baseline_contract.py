"""Per-baseline contract test (BASELINE_COMPARISON.md §6.1, §6.6).

Every registered baseline: consumes the canonical batch dict → returns
(N, H) float32, finite on valid steps, within the physical range, and
deterministically (same fitted model → identical repeated predictions;
trained models are fitted once per session on synthetic CPU data, <60 s).
"""

from __future__ import annotations

import numpy as np
import pytest

import tier0  # noqa: F401
import tier1  # noqa: F401
import tslib  # noqa: F401
import tier3  # noqa: F401
import tier4  # noqa: F401
from common.base import REGISTRY, build
from tslib.trainer import TrainerConfig

from .conftest import skip_if_deps_missing

SMALL_TRAINER = TrainerConfig(epochs=2, batch_size=64, device="cpu", patience=2)

SMALL_KWARGS: dict[str, dict] = {
    "lightgbm": dict(n_estimators=10, max_train_rows=2000),
    "mlp": dict(trainer=SMALL_TRAINER, hidden=32),
    "dlinear": dict(trainer=SMALL_TRAINER),
    "patchtst": dict(trainer=SMALL_TRAINER, d_model=32, n_heads=2, n_layers=1),
    "itransformer": dict(trainer=SMALL_TRAINER, d_model=32, n_heads=2, n_layers=1),
    "tft": dict(trainer=SMALL_TRAINER, d_model=32, n_heads=2),
    "chronos2_zs": dict(model_id="dummy"),
    "chronos2_oracle": dict(model_id="dummy"),
    "chronos2_ft": dict(model_id="dummy", num_steps=2, batch_size=16),
    "timesfm_zs": dict(model_id="dummy"),
    "tirex_zs": dict(model_id="dummy"),
    "ttm_zs": dict(model_id="dummy"),
    "ttm_ft": dict(model_id="dummy", epochs=2, batch_size=16, patience=1),
    # tier 4 contract-tested against a dependency-free tier-0 backbone;
    # real runs wrap chronos2_zs (see tier4 docstrings). TS-RAG / Cross-RAG are
    # cluster-only (vendored original code), not registry baselines.
    "cora": dict(backbone="persistence", epochs=2, batch_size=64,
                 device="cpu", patience=1, max_train_windows=500),
}

ALL_NAMES = sorted(REGISTRY)


@pytest.fixture(scope="session")
def fitted(fit_datasets):
    """Fit each trainable baseline once per session."""
    train, val, _ = fit_datasets
    cache: dict[str, object] = {}

    def get(name: str):
        if name not in cache:
            model = build(name, **SMALL_KWARGS.get(name, {}))
            if model.requires_fit:
                model.fit(train, val)
            cache[name] = model
        return cache[name]

    return get


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forecast_contract(name, fitted, fit_datasets):
    skip_if_deps_missing(name)
    _, _, test = fit_datasets
    batch = test.batch(list(range(min(32, len(test)))))
    model = fitted(name)
    forecast = model.predict(batch)

    point = forecast.point
    assert point.shape == batch["y_future"].shape
    assert point.dtype == np.float32
    valid = batch["mask_future"] == 1
    assert np.isfinite(point[valid]).all()
    assert point[valid].min() >= -1e-6
    assert point[valid].max() <= 1.0 + 1e-6

    if forecast.quantiles is not None:
        q = forecast.quantiles
        assert q.shape == (*point.shape, len(forecast.quantile_levels))
        assert np.isfinite(q[valid]).all()
        assert (np.diff(q, axis=-1) >= -1e-6).all()  # monotone in level
        assert model.supports_quantiles


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forecast_deterministic(name, fitted, fit_datasets):
    """Same fitted model, same batch → bit-identical forecasts (§6.6)."""
    skip_if_deps_missing(name)
    _, _, test = fit_datasets
    batch = test.batch(list(range(min(16, len(test)))))
    model = fitted(name)
    a = model.predict(batch).point
    b = model.predict(batch).point
    np.testing.assert_array_equal(a, b)


def test_registry_covers_tiers_0_to_4():
    names = set(REGISTRY)
    assert {"persistence", "smart_persistence", "climatology_hourly",
            "seasonal_naive"} <= names                        # tier 0
    assert "lightgbm" in names                                # tier 1
    assert {"mlp", "dlinear", "patchtst", "itransformer", "tft"} <= names  # tier 2
    assert {"chronos2_zs", "chronos2_oracle", "chronos2_ft", "timesfm_zs",
            "tirex_zs", "ttm_zs", "ttm_ft"} <= names  # tier 3
    assert "cora" in names                                    # tier 4 (ts_rag/cross_rag are cluster-only vendored code)
