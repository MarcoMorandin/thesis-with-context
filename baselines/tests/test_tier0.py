"""Exactness tests for the tier-0 reference models (§6.5 skill-score sanity)."""

import numpy as np

from common.base import build

from .conftest import STEPS_PER_DAY, make_frame, windows_for


def _batch(ds, n=16):
    return ds.batch(list(range(min(n, len(ds)))))


def test_persistence_repeats_last_valid():
    ds = windows_for(make_frame(n_sites=1, days=3, nan_fraction=0.0), stride=1)
    batch = _batch(ds)
    point = build("persistence").predict(batch).point
    assert (point == batch["y_hist"][:, -1:]).all()


def test_persistence_skips_trailing_nan():
    ds = windows_for(make_frame(n_sites=1, days=3, nan_fraction=0.0), stride=1)
    batch = _batch(ds, n=4)
    batch["mask_hist"][:, -1] = 0.0  # last step invalid → use the one before
    point = build("persistence").predict(batch).point
    assert (point == batch["y_hist"][:, -2:-1]).all()


def test_smart_persistence_exact_on_clear_day():
    """With constant clearness index, smart persistence is a perfect forecast
    on daylight steps — this is what makes SS=0 by construction (§6.5)."""
    ds = windows_for(make_frame(n_sites=1, days=3, kt=0.8, nan_fraction=0.0),
                     stride=1)
    batch = _batch(ds, n=64)
    point = build("smart_persistence").predict(batch).point
    t = batch["y_hist"].shape[1]
    daylight = batch["daylight_future"] > 0
    defined = batch["clearsky"][:, t - 1] >= 50.0  # rows where SP is active
    err = np.abs(point - batch["y_future"])[defined[:, None] & daylight]
    assert err.max() < 1e-5


def test_smart_persistence_zero_at_night():
    ds = windows_for(make_frame(n_sites=1, days=3, kt=0.8, nan_fraction=0.0),
                     stride=1)
    batch = _batch(ds, n=64)
    point = build("smart_persistence").predict(batch).point
    night = batch["daylight_future"] == 0
    assert (point[night] == 0).all()


def test_seasonal_naive_uses_yesterday():
    df = make_frame(n_sites=1, days=4, nan_fraction=0.0)
    df["norm_power"] = np.arange(len(df), dtype=np.float32)
    ds = windows_for(df, stride=1)
    batch = ds.batch([len(ds) - 1])  # last window — yesterday fully available
    point = build("seasonal_naive").predict(batch).point
    assert (point == batch["y_future"] - STEPS_PER_DAY).all()


def test_climatology_learns_train_mean():
    df = make_frame(n_sites=2, days=4, kt=0.5, nan_fraction=0.0)
    train = windows_for(df, stride=1)
    model = build("climatology_hourly")
    model.fit(train, train)
    batch = _batch(train, n=8)
    point = model.predict(batch).point
    # constant-kt synthetic data: climatology equals the true value
    daylight = batch["daylight_future"] > 0
    assert np.abs(point - batch["y_future"])[daylight].max() < 1e-5
