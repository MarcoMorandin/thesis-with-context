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
