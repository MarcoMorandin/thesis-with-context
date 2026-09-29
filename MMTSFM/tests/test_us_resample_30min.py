"""Tests for 30-minute resampling of US data (goes_pvdaq) and Smart Persistence.

Verifies:
1. PVRecordDataset with resample_cadence_min=30 converts 15-min goes_pvdaq to
   30-min cadence (48 steps/day, T=672, H=12), matching the UK PV trained curriculum.
2. MMTSFMDataModule propagates resample_cadence_min to PVRecordDataset.
3. baselines/run_eval.py filtering produces 30-min WindowDataset and valid
   SmartPersistence predictions for the reference denominator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_BL = Path(__file__).resolve().parents[2] / "baselines"
if str(_BL) not in sys.path:
    sys.path.insert(0, str(_BL))

from common import config
from common.base import build
from common.windows import dataset_for_sites


def _make_15min_parquet(path: Path, site: str = "1202", n_steps: int = 1500) -> None:
    """Build synthetic goes_pvdaq parquet with 15-min cadence."""
    start = pd.Timestamp("2019-01-01 00:00:00", tz="UTC")
    times = start + pd.to_timedelta(np.arange(n_steps) * 15, unit="m")
    d = {
        config.DATASET_COL: "goes_pvdaq",
        config.SITE_COL: site,
        config.TIME_COL: times,
        config.TARGET_COL: np.random.default_rng(42).uniform(0.1, 0.9, n_steps),
        config.CAPACITY_COL: 5000.0,
        config.CLEARSKY_COL: 500.0,
        config.BAD_SITE_COL: False,
        config.FRAME_INDEX_COL: -1,
    }
    for c in config.COV_COLS:
        d[c] = 10.0
    pd.DataFrame(d).to_parquet(path)


def test_pv_record_resample_30min(tmp_path):
    from mmtsfm.data.pv_record import PVRecordDataset

    parquet = tmp_path / "dataset_all.parquet"
    _make_15min_parquet(parquet, site="1202", n_steps=2000)

    # 1. Without resampling: native 15-min cadence -> spd=96, T=1344, H=24
    ds_native = PVRecordDataset(
        split="test",
        dataset_name="goes_pvdaq",
        data_path=str(parquet),
        emit_vision=False,
    )
    assert ds_native.T == 1344
    assert ds_native.H == 24

    # 2. With resample_cadence_min=30: 30-min cadence -> spd=48, T=672, H=12
    ds_30m = PVRecordDataset(
        split="test",
        dataset_name="goes_pvdaq",
        data_path=str(parquet),
        emit_vision=False,
        resample_cadence_min=30,
    )
    assert ds_30m.T == 672
    assert ds_30m.H == 12

    # Check window batch tensor shapes
    batch = ds_30m[0]
    assert batch["Y"].shape == (1, 672, 1)
    assert batch["Y_future"].shape == (1, 12, 1)
    assert batch["X_cov"].shape == (1, 684, 14)


def test_datamodule_propagates_resample(tmp_path):
    from mmtsfm.data.datamodule import MMTSFMDataModule

    parquet = tmp_path / "dataset_all.parquet"
    _make_15min_parquet(parquet, site="1202", n_steps=2000)

    dm = MMTSFMDataModule(
        dataset_name="goes_pvdaq",
        data_dir=str(tmp_path),
        resample_cadence_min=30,
        emit_vision=False,
    )
    test_ds = dm._make_dataset("test", 10)
    assert test_ds.T == 672
    assert test_ds.H == 12


def test_baselines_resample_30min_smart_persistence(tmp_path):
    parquet = tmp_path / "dataset_all.parquet"
    _make_15min_parquet(parquet, site="1202", n_steps=2000)

    df = pd.read_parquet(parquet)
    # Apply the same filter as run_eval.py --resample-cadence-min 30
    t = pd.to_datetime(df[config.TIME_COL])
    df_30 = df[(t.dt.minute % 30 == 0) & (t.dt.second == 0)].copy()

    eval_ds = dataset_for_sites(
        df_30,
        site_ids={"1202"},
        history_days=14.0,
        horizon_hours=6.0,
        stride=12,
    )

    assert len(eval_ds.series) == 1
    assert eval_ds.series[0].steps_per_day == 48
    assert eval_ds.history == 672
    assert eval_ds.horizon == 12

    # Run SmartPersistence on the 30-min eval dataset
    model = build("smart_persistence")
    batch = eval_ds.batch(list(range(min(4, len(eval_ds)))))
    forecast = model.predict(batch)
    assert forecast.point.shape == (len(batch["y_future"]), 12)
    assert np.all(np.isfinite(forecast.point))
