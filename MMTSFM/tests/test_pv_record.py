"""Smoke tests for the dataset-of-record backend (pv_record).

Builds a tiny synthetic ``dataset_all.parquet`` (and, where h5py is present, an
``images_all.h5``) over *real* committed uk_pv split sites, then asserts the
emitted canonical dict matches the MMTSFM schema and the baseline protocol:
14-day history / 6-hour horizon at 30-min cadence (672 / 12 steps) and KNOWN
future weather (future_cov="all").
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

from common import config  # noqa: E402

SPLITS = json.loads((_BL / "configs" / "splits.json").read_text())


def _make_parquet(path: Path, sites: list[str], n_steps: int = 800, with_frames: bool = False):
    rows = []
    start = pd.Timestamp("2020-01-01", tz="UTC")
    times = start + pd.to_timedelta(np.arange(n_steps) * 30, unit="m")
    for fi, site in enumerate(sites):
        d = {
            config.DATASET_COL: "uk_pv",
            config.SITE_COL: site,
            config.TIME_COL: times,
            config.TARGET_COL: np.random.default_rng(fi).uniform(0, 1, n_steps),
            config.CAPACITY_COL: 3000.0,
            config.CLEARSKY_COL: 600.0,  # always daylight → every window valid
            config.BAD_SITE_COL: False,
            # one frame per step in this site's private 0..n_steps-1 index space
            config.FRAME_INDEX_COL: (np.arange(n_steps) if with_frames else -1),
        }
        for c in config.COV_COLS:
            d[c] = 10.0  # nonzero raw → nonzero scaled cov (incl. future weather)
        rows.append(pd.DataFrame(d))
    pd.concat(rows, ignore_index=True).to_parquet(path)


def test_numerical_schema_and_future_weather(tmp_path):
    sites = SPLITS["uk_pv"]["train"][:3]
    parquet = tmp_path / "dataset_all.parquet"
    _make_parquet(parquet, sites)

    from mmtsfm.data.pv_record import PVRecordDataset

    ds = PVRecordDataset(
        split="train", dataset_name="uk_pv", data_path=str(parquet),
        img_size=32, img_channels=3, video_frames=8,
    )
    assert ds.T == 672 and ds.H == 12  # 14 d / 6 h at 30-min cadence
    assert len(ds) > 0

    item = ds[0]
    assert item["Y"].shape == (1, 672, 1)
    assert item["Y_future"].shape == (1, 12, 1)
    assert item["X_cov"].shape == (1, 684, len(config.COV_COLS))
    assert item["mask_target"].shape == (1, 672, 1)
    assert item["mask_future"].shape == (1, 12, 1)
    assert item["V"].shape == (1, 8, 3, 32, 32)

    # future weather is KNOWN: an observed-weather covariate (temperature_2m) must
    # be nonzero over the horizon rows, not zeroed out as in the deterministic mode.
    temp_idx = config.COV_COLS.index("temperature_2m")
    future_temp = item["X_cov"][0, 672:, temp_idx]
    assert float(future_temp.abs().sum()) > 0.0


def test_eval_split_is_nonoverlapping(tmp_path):
    sites = SPLITS["uk_pv"]["test"][:2]
    parquet = tmp_path / "dataset_all.parquet"
    _make_parquet(parquet, sites)

    from mmtsfm.data.pv_record import PVRecordDataset

    ds = PVRecordDataset(split="test", dataset_name="uk_pv", data_path=str(parquet))
    # eval stride defaults to H (12): non-overlapping forecast windows
    assert ds.win.horizon == 12
    starts = [start for _, start in ds.win._index]
    assert all((b - a) % 12 == 0 for a, b in zip(starts, starts[1:]) if b > a)


def test_datamodule_config_wires_through(tmp_path):
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "data" / "ukpv.yaml").read_text()
    )
    cfg.pop("_target_")
    parquet = tmp_path / "dataset_all.parquet"
    _make_parquet(parquet, SPLITS["uk_pv"]["train"][:2] + SPLITS["uk_pv"]["val"][:2])
    cfg.update(data_dir=str(parquet), num_workers=0, batch_size=2)

    from mmtsfm.data.datamodule import MMTSFMDataModule

    dm = MMTSFMDataModule(**cfg)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["Y"].shape == (2, 1, 672, 1)
    assert batch["Y_future"].shape == (2, 1, 12, 1)
    assert batch["X_cov"].shape == (2, 1, 684, 14)
    assert batch["V"].shape[:3] == (2, 1, 8)


def test_vision_frames_loaded(tmp_path):
    h5py = pytest.importorskip("h5py")
    sites = SPLITS["uk_pv"]["train"][:1]
    parquet = tmp_path / "dataset_all.parquet"
    h5 = tmp_path / "images_all.h5"
    _make_parquet(parquet, sites, with_frames=True)
    with h5py.File(h5, "w") as f:
        g = f.create_group(f"uk_pv_{sites[0]}")
        g.create_dataset("images", data=np.full((800, 128, 128), 200, np.uint8))

    from mmtsfm.data.pv_record import PVRecordDataset

    ds = PVRecordDataset(
        split="train", dataset_name="uk_pv", data_path=str(parquet),
        h5_path=str(h5), img_size=32, img_channels=3, video_frames=8,
    )
    item = ds[0]
    assert item["V"].shape == (1, 8, 3, 32, 32)
    assert float(item["mask_visual"].sum()) == 8.0  # all 8 recent frames present
    assert float(item["V"].abs().sum()) > 0.0
