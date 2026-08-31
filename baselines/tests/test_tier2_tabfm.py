"""Guards for the tier-2 TabFM arm — runs without the weights or the package.

`tabfm` is an optional git dependency and its checkpoint is a ~GB download, so
these tests inject a stub estimator in place of ``tabfm.TabFMRegressor`` and
assert only what this repo owns: the flattening round-trip
``(N, H, F) -> (N*H, F) -> (N, H)`` stays index-aligned, the output is float32
clipped to the physical range, and the arm advertises itself as point-only.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from common.base import build
from tier1.features import FEATURE_NAMES, build_features
from tier2.tabfm_model import TabFMEnsembleBaseline, TabFMRegressorBaseline

HFRAC = FEATURE_NAMES.index("horizon_frac")


class _StubRegressor:
    """Echoes one feature column back, so predictions are row-identifiable."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.is_ensemble = False
        self.fitted = None

    @classmethod
    def ensemble(cls, **kwargs):
        obj = cls(**kwargs)
        obj.is_ensemble = True
        return obj

    def fit(self, x, y):
        self.fitted = (np.asarray(x), np.asarray(y))
        return self

    def predict(self, x):
        # Deliberately out of [0, 1] so the clip is exercised in both directions.
        return np.asarray(x)[:, HFRAC] * 4.0 - 1.5


@pytest.fixture
def stub_tabfm(monkeypatch):
    """Install a fake ``tabfm`` package for the duration of one test."""
    loader = types.SimpleNamespace(
        load=lambda **kwargs: types.SimpleNamespace(kwargs=kwargs)
    )
    module = types.ModuleType("tabfm")
    module.TabFMRegressor = _StubRegressor
    module.tabfm_v1_0_0_pytorch = loader
    monkeypatch.setitem(sys.modules, "tabfm", module)
    return module


def _fit(cls, fit_datasets, **kwargs):
    train, val, _ = fit_datasets
    model = cls(max_context_rows=200, n_estimators=2, device="cpu", **kwargs)
    model.fit(train, val)
    return model


def test_predictions_stay_aligned_to_their_window_and_horizon(stub_tabfm, fit_datasets):
    """Row r of the flattened table must land back at (r // H, r % H)."""
    _, _, test = fit_datasets
    batch = test.batch(list(range(min(16, len(test)))))
    model = _fit(TabFMRegressorBaseline, fit_datasets)

    point = model.predict(batch).point
    feats = build_features(batch)
    expected = np.clip(feats[..., HFRAC] * 4.0 - 1.5, 0.0, 1.0).astype(np.float32)

    assert point.shape == batch["y_future"].shape
    np.testing.assert_array_equal(point, expected)


def test_output_is_float32_within_the_physical_range(stub_tabfm, fit_datasets):
    _, _, test = fit_datasets
    batch = test.batch(list(range(min(16, len(test)))))
    point = _fit(TabFMRegressorBaseline, fit_datasets).predict(batch).point

    assert point.dtype == np.float32
    assert np.isfinite(point).all()
    assert point.min() >= 0.0 and point.max() <= 1.0


def test_arm_is_point_only(stub_tabfm, fit_datasets):
    """No regression quantile head upstream — CRPS/coverage/ECE must be N/A."""
    _, _, test = fit_datasets
    batch = test.batch(list(range(min(8, len(test)))))
    model = _fit(TabFMRegressorBaseline, fit_datasets)

    assert model.supports_quantiles is False
    assert model.predict(batch).quantiles is None


def test_context_is_subsampled_to_max_context_rows(stub_tabfm, fit_datasets):
    model = _fit(TabFMRegressorBaseline, fit_datasets)
    x, y = model._model.fitted

    assert x.shape[0] == y.shape[0] <= 200
    assert x.shape[1] == len(FEATURE_NAMES)


def test_cache_is_exact_by_default(stub_tabfm, fit_datasets):
    """cache_context speeds up predict; quantized KV would change the numbers."""
    _fit(TabFMRegressorBaseline, fit_datasets)
    kwargs = _StubRegressor.last_kwargs

    assert kwargs["cache_context"] is True
    assert kwargs["maybe_quantize_kv_cache"] is False


def test_ens_arm_uses_the_ensemble_preset(stub_tabfm, fit_datasets):
    plain = _fit(TabFMRegressorBaseline, fit_datasets)
    ens = _fit(TabFMEnsembleBaseline, fit_datasets)

    assert plain._model.is_ensemble is False
    assert ens._model.is_ensemble is True


def test_both_arms_are_registered(stub_tabfm):
    assert isinstance(build("tabfm"), TabFMRegressorBaseline)
    assert isinstance(build("tabfm_ens"), TabFMEnsembleBaseline)
    assert build("tabfm").tier == 2
