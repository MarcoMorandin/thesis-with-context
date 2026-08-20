from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.probes.ceiling_dataset import build_arrays, parse_cache_key


def test_parse_cache_key_splits_from_the_right():
    # site ids are numeric strings, dataset names contain underscores
    assert parse_cache_key("uk_pv_7239_1546300800") == ("uk_pv", "7239", 1546300800)
    assert parse_cache_key("goes_pvdaq_1202_1546300800") == (
        "goes_pvdaq",
        "1202",
        1546300800,
    )


@pytest.fixture
def tiny(tmp_path):
    """One site, 40 half-hourly rows, 2 cached windows, both fully in range."""
    t0 = 1546300800  # 2019-01-01T00:00:00Z
    n = 40
    rows = {
        "dataset": ["uk_pv"] * n,
        "site_id": ["9001"] * n,
        "timestamp_utc": pd.to_datetime(
            [t0 + 1800 * i for i in range(n)], unit="s", utc=True
        ),
        "norm_power": np.linspace(0.0, 1.0, n),
        "installed_power_w": [3000.0] * n,
    }
    from common import config

    for c in config.COV_COLS:
        rows[c] = np.arange(n, dtype=float)
    pq = tmp_path / "d.parquet"
    pd.DataFrame(rows).to_parquet(pq)

    cache = tmp_path / "cache"
    cache.mkdir()
    for origin_i in (10, 11):
        torch.save(
            torch.full((4, 196, 1024), float(origin_i), dtype=torch.float16),
            cache / f"uk_pv_9001_{t0 + 1800 * origin_i}.pt",
        )
    return cache, pq


def test_build_arrays_shapes_and_alignment(tiny):
    cache, pq = tiny
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    assert out["X_vis"].shape == (2, 4096)  # spatial mean-pool, time kept
    assert out["Y"].shape == (2, 12)
    assert out["Y_mask"].shape == (2, 12)
    assert out["X_cov"].shape[0] == 2
    # rows are sorted by origin, so row 0 is the earlier window
    assert out["origin"][0] < out["origin"][1]
    # X_vis row i must come from the file whose origin matches row i
    assert np.allclose(out["X_vis"][0], 10.0)
    assert np.allclose(out["X_vis"][1], 11.0)


def test_build_arrays_rejects_unknown_sites(tiny):
    cache, pq = tiny
    out = build_arrays(cache, pq, sites={"9999"}, horizon=12)
    assert out["X_vis"].shape[0] == 0


@pytest.fixture
def masking(tmp_path):
    """One site, 40 half-hourly rows, engineered to exercise every masking
    branch in one fixture:

    - origin_i=10: fully in range for horizon=12 (h=1..12 -> idx 11..22, all
      < n=40), except a NaN planted at idx 17 (= origin_i + h=7) so one
      interior step must be masked False for a reason other than truncation.
    - origin_i=35: runs off the end of the table (h=1..12 -> idx 36..47);
      only idx < 40 exist, so h=1..4 (idx 36..39) are valid and h=5..12
      (idx 40..47) must be masked False.

    Rows sort by origin, so output row 0 is always origin_i=10 and row 1 is
    always origin_i=35.
    """
    t0 = 1546300800  # 2019-01-01T00:00:00Z
    n = 40
    norm_power = np.linspace(0.0, 1.0, n)
    nan_idx = 17
    norm_power[nan_idx] = np.nan
    rows = {
        "dataset": ["uk_pv"] * n,
        "site_id": ["9001"] * n,
        "timestamp_utc": pd.to_datetime(
            [t0 + 1800 * i for i in range(n)], unit="s", utc=True
        ),
        "norm_power": norm_power,
        "installed_power_w": [3000.0] * n,
    }
    from common import config

    for c in config.COV_COLS:
        rows[c] = np.arange(n, dtype=float)
    pq = tmp_path / "d.parquet"
    pd.DataFrame(rows).to_parquet(pq)

    cache = tmp_path / "cache"
    cache.mkdir()
    for origin_i in (10, 35):
        torch.save(
            torch.full((4, 196, 1024), float(origin_i), dtype=torch.float16),
            cache / f"uk_pv_9001_{t0 + 1800 * origin_i}.pt",
        )
    return cache, pq, norm_power


def test_build_arrays_masks_missing_future(masking):
    cache, pq, _norm_power = masking
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    assert out["origin"][0] < out["origin"][1]

    # row 1 (origin_i=35) runs off the end of the 40-row table starting at
    # h=5 (idx 40 >= n=40): leading steps valid, trailing steps must be False.
    assert out["Y_mask"][1, :4].all()
    assert not out["Y_mask"][1, 4:].any()

    # row 0 (origin_i=10) stays in range for every h, except the NaN planted
    # at idx 17, which is h=7 (idx = origin_i + h = 10 + 7).
    expected_row0_mask = np.ones(12, dtype=bool)
    expected_row0_mask[6] = False
    np.testing.assert_array_equal(out["Y_mask"][0], expected_row0_mask)


def test_build_arrays_y_values_align_with_the_table(masking):
    cache, pq, norm_power = masking
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    # row 0 = origin_i=10; Y[row, h-1] must equal the table's norm_power at
    # exactly idx = origin_i + h. Checked at h=1 and h=12 so an off-by-one in
    # the `origin + h * step_seconds` arithmetic would fail this.
    assert out["Y"][0, 0] == pytest.approx(norm_power[11], rel=1e-5)  # h=1 -> idx 11
    assert out["Y"][0, 11] == pytest.approx(norm_power[22], rel=1e-5)  # h=12 -> idx 22


def test_build_arrays_keeps_site_and_origin_aligned_when_history_row_is_missing(
    tmp_path,
):
    """Regression: a cache file whose origin has no matching row in the table
    at all must still carry its own site/origin in the output. X_vis is
    written unconditionally before the history lookup; site/origin must be
    written just as unconditionally, or a row can carry real vision features
    next to a default site=None/origin=0 -- silently breaking the invariant
    that site[N]/origin[N] describe row N.
    """
    t0 = 1546300800
    n = 5
    rows = {
        "dataset": ["uk_pv"] * n,
        "site_id": ["9001"] * n,
        "timestamp_utc": pd.to_datetime(
            [t0 + 1800 * i for i in range(n)], unit="s", utc=True
        ),
        "norm_power": np.linspace(0.0, 1.0, n),
        "installed_power_w": [3000.0] * n,
    }
    from common import config

    for c in config.COV_COLS:
        rows[c] = np.arange(n, dtype=float)
    pq = tmp_path / "d.parquet"
    pd.DataFrame(rows).to_parquet(pq)

    cache = tmp_path / "cache"
    cache.mkdir()
    # origin index 999 has no corresponding row anywhere in the 5-row table.
    missing_origin = t0 + 1800 * 999
    torch.save(
        torch.full((4, 196, 1024), 7.0, dtype=torch.float16),
        cache / f"uk_pv_9001_{missing_origin}.pt",
    )

    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    assert out["X_vis"].shape[0] == 1
    assert np.allclose(out["X_vis"][0], 7.0)
    assert out["site"][0] == "9001"
    assert out["origin"][0] == missing_origin


@pytest.fixture
def night_gap(tmp_path):
    """One site, 40 half-hourly grid steps, but rows 15-19 are dropped from the
    parquet ENTIRELY (not present, not NaN in the source table) -- like uk_pv's
    genuine night gaps. `build_site_series` reconstructs a full 40-step regular
    grid from `date_range(times[0], times[-1], freq=median_step)` and reindexes,
    so the dropped rows come back as NaN rows *on the grid*, at exactly the
    timestamps they would have had. A cache origin sitting on one of those
    timestamps has no row in the raw parquet table -- the pre-fix exact-match
    join (`table.loc[(ds, site, origin)]`) raised KeyError and silently dropped
    the window. This is that regression, pinned.
    """
    t0 = 1546300800  # 2019-01-01T00:00:00Z
    n = 40
    drop = set(range(15, 20))
    keep = [i for i in range(n) if i not in drop]

    from common import config

    rows = {
        "dataset": ["uk_pv"] * len(keep),
        "site_id": ["9001"] * len(keep),
        "timestamp_utc": pd.to_datetime(
            [t0 + 1800 * i for i in keep], unit="s", utc=True
        ),
        "norm_power": [float(i) / n for i in keep],
        "installed_power_w": [3000.0] * len(keep),
    }
    for c in config.COV_COLS:
        rows[c] = [float(i) for i in keep]
    pq = tmp_path / "d.parquet"
    pd.DataFrame(rows).to_parquet(pq)

    cache = tmp_path / "cache"
    cache.mkdir()
    gap_origin_i = 17  # inside the dropped range: no parquet row, but a grid point
    origin = t0 + 1800 * gap_origin_i
    torch.save(
        torch.full((4, 196, 1024), 5.0, dtype=torch.float16),
        cache / f"uk_pv_9001_{origin}.pt",
    )
    return cache, pq, origin, drop, n


def test_build_arrays_recovers_a_window_whose_origin_falls_in_a_night_gap(night_gap):
    cache, pq, origin, drop, n = night_gap
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)

    assert out["n_skipped"] == 0
    assert out["X_vis"].shape[0] == 1
    assert np.allclose(out["X_vis"][0], 5.0)
    assert out["site"][0] == "9001"
    assert out["origin"][0] == origin

    # origin's grid index is 17 (gap_origin_i); h=1..12 -> future grid idx
    # 18..29. idx 18,19 are still inside the dropped range (15-19) so their
    # target is NaN -> masked False. idx 20..29 are real rows -> masked True.
    expected_mask = np.array(
        [(17 + h) not in drop and (17 + h) < n for h in range(1, 13)]
    )
    np.testing.assert_array_equal(out["Y_mask"][0], expected_mask)
    for h in range(1, 13):
        j = 17 + h
        if j not in drop and j < n:
            assert out["Y"][0, h - 1] == pytest.approx(j / n, rel=1e-5)

    # The origin itself (grid idx 17) sits inside the dropped range, so its
    # row is entirely NaN post-reindex, including every covariate column --
    # exactly the case that must be nan_to_num'd before landing in X_cov, or
    # this row alone would poison a downstream Ridge fit with NaN.
    assert np.isfinite(out["X_cov"]).all()


def test_build_arrays_reports_n_skipped_for_origins_outside_the_grid(tmp_path):
    """A cache origin beyond the site's grid range (later than the last table
    row) must increment n_skipped and must NOT be silently absorbed into a
    default all-False Y_mask with no trace anywhere -- n_skipped is the only
    signal a caller has that data was lost.
    """
    t0 = 1546300800
    n = 5
    rows = {
        "dataset": ["uk_pv"] * n,
        "site_id": ["9001"] * n,
        "timestamp_utc": pd.to_datetime(
            [t0 + 1800 * i for i in range(n)], unit="s", utc=True
        ),
        "norm_power": np.linspace(0.0, 1.0, n),
        "installed_power_w": [3000.0] * n,
    }
    from common import config

    for c in config.COV_COLS:
        rows[c] = np.arange(n, dtype=float)
    pq = tmp_path / "d.parquet"
    pd.DataFrame(rows).to_parquet(pq)

    cache = tmp_path / "cache"
    cache.mkdir()
    beyond_origin = t0 + 1800 * 999  # far past the grid's last timestamp
    torch.save(
        torch.full((4, 196, 1024), 3.0, dtype=torch.float16),
        cache / f"uk_pv_9001_{beyond_origin}.pt",
    )

    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    assert out["n_skipped"] == 1
    assert out["X_vis"].shape[0] == 1
    assert not out["Y_mask"][0].any()


@pytest.fixture
def ramp(tmp_path):
    """One site, 40 half-hourly steps, norm_power a known monotonic ramp
    (norm_power[i] == i), so power-lag values can be checked exactly.
    """
    t0 = 1546300800
    n = 40

    from common import config

    rows = {
        "dataset": ["uk_pv"] * n,
        "site_id": ["9001"] * n,
        "timestamp_utc": pd.to_datetime(
            [t0 + 1800 * i for i in range(n)], unit="s", utc=True
        ),
        "norm_power": np.arange(n, dtype=float),
        "installed_power_w": [3000.0] * n,
    }
    for c in config.COV_COLS:
        rows[c] = np.arange(n, dtype=float)
    pq = tmp_path / "d.parquet"
    pd.DataFrame(rows).to_parquet(pq)

    cache = tmp_path / "cache"
    cache.mkdir()
    # origin_i=2: some lag offsets (-3, -6, -12) run off the start of the table.
    # origin_i=20: every lag offset stays in range.
    for origin_i in (2, 20):
        torch.save(
            torch.full((4, 196, 1024), float(origin_i), dtype=torch.float16),
            cache / f"uk_pv_9001_{t0 + 1800 * origin_i}.pt",
        )
    return cache, pq


def test_build_arrays_power_lags_are_correct(ramp):
    cache, pq = ramp
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    # rows sort by origin: row 0 = origin_i=2, row 1 = origin_i=20.
    n_lags = 6
    lag_offsets = (0, -1, -2, -3, -6, -12)

    # row 0: idx=2, offsets -3/-6/-12 -> j=-1/-4/-10, all < 0 -> lag=0, indicator=0.
    lags0 = out["X_cov"][0, :n_lags]
    valid0 = out["X_cov"][0, n_lags : 2 * n_lags]
    for li, off in enumerate(lag_offsets):
        j = 2 + off
        if j < 0:
            assert lags0[li] == pytest.approx(0.0)
            assert valid0[li] == pytest.approx(0.0)
        else:
            assert lags0[li] == pytest.approx(float(j))
            assert valid0[li] == pytest.approx(1.0)

    # row 1: idx=20, every offset stays >= 0 -> all lags valid.
    lags1 = out["X_cov"][1, :n_lags]
    valid1 = out["X_cov"][1, n_lags : 2 * n_lags]
    for li, off in enumerate(lag_offsets):
        j = 20 + off
        assert lags1[li] == pytest.approx(float(j))
        assert valid1[li] == pytest.approx(1.0)


def test_build_arrays_history_covariates_match_model_scaling(ramp):
    """X_cov's history-covariate block must equal the raw parquet value divided
    by its COV_SCALES entry -- the same scaling `build_site_series` applies for
    the model -- so probe/model equivalence is pinned by a test, not a comment.
    """
    from common import config

    cache, pq = ramp
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    n_lags = 6
    cov_base = 2 * n_lags

    # row 1: origin_i=20 -> idx=20; the fixture set every COV_COLS value to
    # `float(i)` at row i, so the raw value at idx=20 is 20.0 for every column.
    c = config.COV_COLS[0]
    col = config.COV_COLS.index(c)
    expected = 20.0 / config.COV_SCALES[c]
    assert out["X_cov"][1, cov_base + col] == pytest.approx(expected, rel=1e-5)


@pytest.fixture
def future_nan_cov(tmp_path):
    """One site, 40 half-hourly rows, every row present (no row-level gaps),
    but one DETERMINISTIC covariate (clearsky_ghi) is NaN at a single future
    grid index while norm_power at that same index stays valid. This is an
    observed-value NaN, independent of the row-presence gaps build_site_series
    reconstructs -- it exercises the future-covariate nan_to_num path in
    isolation from the history-covariate / night-gap path.
    """
    t0 = 1546300800
    n = 40
    from common import config

    cov_data = {c: np.arange(n, dtype=float) for c in config.COV_COLS}
    nan_future_idx = 25
    cov_data["clearsky_ghi"][nan_future_idx] = np.nan

    rows = {
        "dataset": ["uk_pv"] * n,
        "site_id": ["9001"] * n,
        "timestamp_utc": pd.to_datetime(
            [t0 + 1800 * i for i in range(n)], unit="s", utc=True
        ),
        "norm_power": np.linspace(0.0, 1.0, n),
        "installed_power_w": [3000.0] * n,
        **cov_data,
    }
    pq = tmp_path / "d.parquet"
    pd.DataFrame(rows).to_parquet(pq)

    cache = tmp_path / "cache"
    cache.mkdir()
    origin_i = 13  # h=12 -> future grid idx 25 == nan_future_idx
    torch.save(
        torch.full((4, 196, 1024), 1.0, dtype=torch.float16),
        cache / f"uk_pv_9001_{t0 + 1800 * origin_i}.pt",
    )
    return cache, pq, nan_future_idx


def test_build_arrays_future_covariate_nan_does_not_leak_into_x_cov(future_nan_cov):
    cache, pq, nan_future_idx = future_nan_cov
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)

    assert out["n_skipped"] == 0
    # h=12 -> grid idx 25 == nan_future_idx; norm_power there is a real value
    # (np.linspace never NaNs), so this future step must still be masked
    # valid even though its clearsky_ghi covariate is NaN.
    assert out["Y_mask"][0, 11]
    assert np.isfinite(out["X_cov"]).all()


def _origin_arrays(hours, n_per_hour=4, d_vis=6, d_cov=3, horizon=2):
    """Rows at known UTC times of day, each row tagged by its index everywhere.

    Every per-row array carries the same tag so a filter that keeps the right
    ROWS but pairs them wrongly is caught -- that is the failure mode that
    matters here, and a shape-only assertion cannot see it.
    """
    origins, tags = [], []
    for hi, h in enumerate(hours):
        for k in range(n_per_hour):
            # Day k, at hour h. Distinct days so origins are never duplicated.
            origins.append(k * 86400 + int(round(h * 3600)))
            tags.append(hi * 100 + k)
    tag = np.array(tags, dtype=np.float32)
    n = len(tags)
    return {
        "X_vis": np.tile(tag[:, None], (1, d_vis)),
        "X_cov": np.tile(tag[:, None], (1, d_cov)),
        "Y": np.tile(tag[:, None], (1, horizon)),
        "Y_mask": np.ones((n, horizon), dtype=bool),
        "site": np.array([str(int(t)) for t in tags], dtype=object),
        "origin": np.array(origins, dtype=np.int64),
        "n_skipped": 7,
    }


def test_filter_by_origin_hour_keeps_only_the_named_times_of_day():
    """uk_pv caches exactly two origins per site per day and only 13:30 has a
    daylight visual window; pooling 07:30 in dilutes every fit with blank rows."""
    from scripts.probes.ceiling_dataset import filter_by_origin_hour

    arrays = _origin_arrays([7.5, 13.5], n_per_hour=4)
    out = filter_by_origin_hour(arrays, [13.5])

    assert out["n_kept"] == 4
    assert out["n_dropped_by_origin_hour"] == 4
    kept_hours = (out["origin"] % 86400) / 3600.0
    assert set(kept_hours.tolist()) == {13.5}


def test_filter_by_origin_hour_preserves_row_alignment():
    """X_vis, X_cov, Y, Y_mask, site and origin are PARALLEL arrays. Filtering
    them out of step would pair one window's features with another's target --
    silent, and fatal to every number the probe reports."""
    from scripts.probes.ceiling_dataset import filter_by_origin_hour

    arrays = _origin_arrays([7.5, 13.5], n_per_hour=4)
    out = filter_by_origin_hour(arrays, [13.5])

    tags = out["X_vis"][:, 0]
    assert np.array_equal(out["X_cov"][:, 0], tags)
    assert np.array_equal(out["Y"][:, 0], tags)
    assert np.array_equal(out["site"].astype(int), tags.astype(int))
    assert out["Y_mask"].shape[0] == len(tags)
    # Tags 100..103 are the second hour bucket (13.5), never the first.
    assert set(tags.astype(int).tolist()) == {100, 101, 102, 103}


def test_filter_by_origin_hour_does_not_filter_n_skipped():
    """n_skipped counts cache files that never became rows, so it is not a
    per-row quantity and must survive filtering unchanged."""
    from scripts.probes.ceiling_dataset import filter_by_origin_hour

    out = filter_by_origin_hour(_origin_arrays([7.5, 13.5]), [13.5])
    assert out["n_skipped"] == 7


def test_filter_by_origin_hour_tolerance_spans_the_grid_step():
    """Origins land on a 30-minute grid, so a request for 13.5 must still match
    a window a few minutes off rather than silently returning nothing."""
    from scripts.probes.ceiling_dataset import filter_by_origin_hour

    arrays = _origin_arrays([13.4, 13.5, 7.5], n_per_hour=2)
    out = filter_by_origin_hour(arrays, [13.5], tol_seconds=900)
    assert out["n_kept"] == 4  # 13.4 is 360s away, inside the 900s tolerance
    out_tight = filter_by_origin_hour(arrays, [13.5], tol_seconds=60)
    assert out_tight["n_kept"] == 2
