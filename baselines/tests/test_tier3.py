"""Verification tests for Tier 3 time-series foundation models (§6.1, §6.6)."""

from __future__ import annotations

import numpy as np
import pytest

from common.base import build
from .conftest import windows_for, make_frame


def _batch(ds, n=16):
    return ds.batch(list(range(min(n, len(ds)))))


@pytest.fixture(scope="module")
def synthetic_dataset():
    df = make_frame(n_sites=1, days=4, nan_fraction=0.0)
    return windows_for(df, stride=1)


@pytest.mark.parametrize("model_name", ["chronos2_zs", "timesfm_zs", "tirex_zs", "ttm_zs"])
def test_tier3_zero_shot_dummy(model_name, synthetic_dataset):
    batch = _batch(synthetic_dataset)
    model = build(model_name, model_id="dummy")
    
    assert not model.requires_fit
    forecast = model.predict(batch)
    
    point = forecast.point
    assert point.shape == batch["y_future"].shape
    assert point.dtype == np.float32
    assert np.isfinite(point).all()
    assert point.min() >= 0.0
    assert point.max() <= 1.0

    if forecast.quantiles is not None:
        q = forecast.quantiles
        assert q.shape == (*point.shape, len(forecast.quantile_levels))
        assert np.isfinite(q).all()
        assert (np.diff(q, axis=-1) >= -1e-6).all()  # monotone in level
        assert model.supports_quantiles
    else:
        assert not model.supports_quantiles


@pytest.mark.parametrize("model_name", ["chronos2_ft", "ttm_ft"])
def test_tier3_fine_tune_dummy(model_name, synthetic_dataset):
    train = synthetic_dataset
    val = synthetic_dataset
    batch = _batch(synthetic_dataset)
    
    model = build(model_name, model_id="dummy", epochs=2, batch_size=4, patience=1)
    assert model.requires_fit
    
    model.fit(train, val)
    forecast = model.predict(batch)
    
    point = forecast.point
    assert point.shape == batch["y_future"].shape
    assert point.dtype == np.float32
    assert np.isfinite(point).all()
    assert point.min() >= 0.0
    assert point.max() <= 1.0

    if forecast.quantiles is not None:
        q = forecast.quantiles
        assert q.shape == (*point.shape, len(forecast.quantile_levels))
        assert np.isfinite(q).all()
        assert (np.diff(q, axis=-1) >= -1e-6).all()
        assert model.supports_quantiles
    else:
        assert not model.supports_quantiles
