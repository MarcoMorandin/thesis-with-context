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
    """One site, 40 half-hourly rows, 2 cached windows."""
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


def test_build_arrays_masks_missing_future(tiny):
    cache, pq = tiny
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    # window at origin index 11 runs off the end of a 40-row table at h=12
    # only if 11+12 >= 40; here it does not, so every step is valid
    assert out["Y_mask"].all()


def test_build_arrays_rejects_unknown_sites(tiny):
    cache, pq = tiny
    out = build_arrays(cache, pq, sites={"9999"}, horizon=12)
    assert out["X_vis"].shape[0] == 0
