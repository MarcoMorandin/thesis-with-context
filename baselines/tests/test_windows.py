"""Window construction, leakage and split-disjointness tests (§6.2-6.3)."""

import numpy as np
import pytest

from common import config
from common.splits import assert_disjoint, make_plant_splits
from common.windows import WindowDataset, build_site_series

from .conftest import STEPS_PER_DAY, make_frame, windows_for


def test_history_future_alignment():
    """y_future must continue y_hist with no gap or overlap."""
    df = make_frame(n_sites=1, days=4, kt=0.8, nan_fraction=0.0)
    df["norm_power"] = np.arange(len(df), dtype=np.float32)  # ramp → exact check
    ds = windows_for(df, stride=1)
    item = ds[10]
    t = len(item["y_hist"])
    assert item["y_future"][0] == item["y_hist"][-1] + 1
    assert (np.diff(np.concatenate([item["y_hist"], item["y_future"]])) == 1).all()
    # timestamps cover T+H contiguous steps
    assert len(item["timestamps"]) == t + len(item["y_future"])
    assert (np.diff(item["timestamps"]) == 30 * 60).all()


def test_seasonal_reference_is_one_day_lag():
    df = make_frame(n_sites=1, days=4, kt=0.8, nan_fraction=0.0)
    df["norm_power"] = np.arange(len(df), dtype=np.float32)
    ds = windows_for(df, stride=1)
    series = ds.series[0]
    si, start = ds._index[-1]  # late window: previous day exists
    item = ds[len(ds) - 1]
    t = ds.history
    expected = series.y[start + t - STEPS_PER_DAY : start + t + ds.horizon - STEPS_PER_DAY]
    assert (item["y_seasonal"] == expected).all()
    assert item["mask_seasonal"].all()


def test_future_observed_weather_zeroed():
    """No lookahead: observed-weather covariates are 0 beyond history."""
    ds = windows_for(make_frame(), stride=7)
    item = ds[0]
    t = ds.history
    observed = [
        i for i in range(item["cov"].shape[1])
        if i not in config.DETERMINISTIC_COV_IDX
    ]
    assert (item["cov"][t:, observed] == 0).all()
    det = list(config.DETERMINISTIC_COV_IDX)
    assert np.abs(item["cov"][t:, det]).sum() > 0  # deterministic covs survive


def test_masks_flag_nan_targets():
    df = make_frame(n_sites=1, days=3, nan_fraction=0.3, seed=1)
    ds = windows_for(df, stride=1)
    item = ds[0]
    assert set(np.unique(item["mask_hist"])) <= {0.0, 1.0}
    assert np.isfinite(item["y_hist"]).all()  # NaN replaced, masked instead
    assert np.isfinite(item["y_future"]).all()


def test_plant_splits_disjoint_and_deterministic():
    df = make_frame(n_sites=10)
    s1 = make_plant_splits(df, seed=config.SEED)
    s2 = make_plant_splits(df, seed=config.SEED)
    assert s1 == s2
    all_sites = [s for parts in s1.values() for part in parts.values() for s in part]
    assert len(all_sites) == len(set(all_sites)) == 10


def test_split_overlap_fails_loud():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint({"d": {"train": ["a"], "val": ["a"], "test": ["b"]}})


def test_bad_sites_excluded_from_splits():
    df = make_frame(n_sites=4)
    df.loc[df.site_id == "site_0", "bad_site_flag"] = True
    splits = make_plant_splits(df)
    all_sites = {s for parts in splits.values() for p in parts.values() for s in p}
    assert "site_0" not in all_sites


def test_native_cadences_preserved():
    """15-min and 30-min sites must keep their own grids (no resampling)."""
    df30 = make_frame(n_sites=1, days=2)
    series = build_site_series(df30)
    assert series[0].steps_per_day == STEPS_PER_DAY
    assert (np.diff(series[0].timestamps) == 30 * 60).all()
