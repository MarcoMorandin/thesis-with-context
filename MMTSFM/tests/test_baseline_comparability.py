"""Guard tests: MMTSFM must train/test on the SAME splits and seed as the
baselines so the reported numbers are comparable (BASELINE_PROTOCOL.md).

If a baseline changes the canonical seed or regenerates splits.json, these
tests fail loudly instead of letting MMTSFM drift onto a different protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]  # MMTSFM/
_SRC = _ROOT / "src"
_BL = _ROOT.parent / "baselines"
for _p in (_SRC, _BL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import config as bl_config  # noqa: E402
from common.splits import SPLITS_PATH, load_splits  # noqa: E402


def _cfg_seed(rel: str) -> int:
    return yaml.safe_load((_ROOT / "configs" / rel).read_text())["seed"]


def test_mmtsfm_seed_matches_baselines():
    """The Hydra run seed must equal the baselines' canonical seed."""
    assert bl_config.SEED == _cfg_seed("config.yaml")
    assert bl_config.SEED == _cfg_seed("experiment/ukpv.yaml")


def test_pv_record_reads_the_baseline_splits_file():
    """MMTSFM resolves the exact committed baselines/configs/splits.json."""
    assert SPLITS_PATH == (_BL / "configs" / "splits.json")
    splits = load_splits()
    assert "uk_pv" in splits and "goes_pvdaq" in splits


def _make_parquet(path: Path, sites: list[str], n_steps: int = 800):
    rows = []
    start = pd.Timestamp("2020-01-01", tz="UTC")
    times = start + pd.to_timedelta(np.arange(n_steps) * 30, unit="m")
    for fi, site in enumerate(sites):
        d = {
            bl_config.DATASET_COL: "uk_pv",
            bl_config.SITE_COL: site,
            bl_config.TIME_COL: times,
            bl_config.TARGET_COL: np.random.default_rng(fi).uniform(0, 1, n_steps),
            bl_config.CAPACITY_COL: 3000.0,
            bl_config.CLEARSKY_COL: 600.0,
            bl_config.BAD_SITE_COL: False,
            bl_config.FRAME_INDEX_COL: -1,
        }
        for c in bl_config.COV_COLS:
            d[c] = 10.0
        rows.append(pd.DataFrame(d))
    pd.concat(rows, ignore_index=True).to_parquet(path)


def test_train_dataset_never_includes_eval_plants(tmp_path):
    """A train PVRecordDataset uses only train plants — never val/test plants,
    even when they are present in the parquet (disjoint cross-plant protocol)."""
    from mmtsfm.data.pv_record import PVRecordDataset

    splits = load_splits()["uk_pv"]
    train_sites = splits["train"][:3]
    eval_sites = set(splits["val"][:2]) | set(splits["test"][:2])
    parquet = tmp_path / "dataset_all.parquet"
    _make_parquet(parquet, train_sites + list(eval_sites))

    ds = PVRecordDataset(split="train", dataset_name="uk_pv", data_path=str(parquet))
    used = {str(s.site_id) for s in ds.win.series}
    assert used.issubset(set(train_sites))
    assert used.isdisjoint(eval_sites)


def test_test_dataset_uses_only_test_plants(tmp_path):
    from mmtsfm.data.pv_record import PVRecordDataset

    splits = load_splits()["uk_pv"]
    test_sites = splits["test"][:3]
    train_sites = splits["train"][:2]
    parquet = tmp_path / "dataset_all.parquet"
    _make_parquet(parquet, test_sites + train_sites)

    ds = PVRecordDataset(split="test", dataset_name="uk_pv", data_path=str(parquet))
    used = {str(s.site_id) for s in ds.win.series}
    assert used.issubset(set(test_sites))
    assert used.isdisjoint(set(train_sites))
