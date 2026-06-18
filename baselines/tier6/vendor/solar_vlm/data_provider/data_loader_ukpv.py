# File: data_provider/data_loader_ukpv.py
"""Solar-VLM data loader for the thesis dataset of record (uk_pv multimodal).

Feeds Solar-VLM's *fixed multi-station* interface from our per-plant parquet,
preserving the disjoint cross-plant split: plants of one split partition
(``flag``) are clustered into co-located groups of exactly ``num_stations``
(see ``grouping.build_station_groups``); each group is one Solar-VLM "station
set". Train uses only train-split plants, test only test-split plants, so a test
group is unseen plants — the zero-shot cross-plant contract, lifted to groups.

Contract matches ``Dataset_SKIPPD`` (the experiment.py-compatible loader): a
5-tuple ``(seq_x, seq_y, seq_x_mark, seq_y_mark, ts_key)`` with
``seq_x = [seq_len, S, F]`` and per-window ``ts_key``. We diverge from SKIPPD on
scaling only: covariates use the suite's FIXED physical scalings
(``common.config.COV_SCALES``) and the target ``norm_power`` is already
capacity-normalized in [0, 1], so there is no fitted StandardScaler and no
cross-plant leakage — predictions come back in the same space the suite
evaluates (``inverse_transform`` is the identity).

``ts_key`` is group-scoped: ``"<group_idx>__<YYYYMMDDHHMM>"``. The vendored
``VisionFeatureStore`` is patched to split off the ``<group_idx>__`` prefix so a
single store serves disjoint groups that share wall-clock timestamps (see
tier6/vendor/VENDOR_NOTICE.md).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# baselines/ on path for the shared split + schema (fairness contract)
_BASELINES = Path(__file__).resolve().parents[4]
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

from common import config as cfg                # noqa: E402
from data_provider.grouping import build_station_groups  # noqa: E402

warnings.filterwarnings("ignore")


def _split_plants(flag: str, dataset: str = "uk_pv") -> list[str]:
    """uk_pv plant ids for a split part using the suite's canonical splits."""
    from common.splits import load_splits
    part = {"train": "train", "val": "val", "test": "test"}[flag]
    return [str(s) for s in load_splits()[dataset][part]]


def load_split_frame_df(data_path: str, flag: str, dataset: str,
                        cov_cols: list[str]) -> pd.DataFrame:
    """Frame-bearing rows of one split's plants (shared by loader + precompute)."""
    plants = set(_split_plants(flag, dataset))
    cols = sorted({
        cfg.DATASET_COL, cfg.SITE_COL, cfg.TIME_COL, cfg.TARGET_COL,
        cfg.FRAME_INDEX_COL, "latitude", "longitude", *cov_cols,
    })
    df = pd.read_parquet(data_path, columns=cols)
    df = df[df[cfg.DATASET_COL] == dataset].copy()
    df[cfg.SITE_COL] = df[cfg.SITE_COL].astype(str)
    df = df[df[cfg.SITE_COL].isin(plants)]
    df = df[df[cfg.FRAME_INDEX_COL].notna() & (df[cfg.FRAME_INDEX_COL] >= 0)]
    df[cfg.TIME_COL] = pd.to_datetime(df[cfg.TIME_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[cfg.TIME_COL])
    return df


def iter_usable_groups(df: pd.DataFrame, num_stations: int, min_len: int):
    """Yield ``(group, common_index)`` for each usable station group.

    Identical grouping/alignment for the loader and the vision precompute so the
    enumerate() position == the ``<group_idx>`` baked into every ts_key. A group
    is usable when its plants share at least ``min_len`` timestamps.
    """
    coords = {s: (float(g["latitude"].iloc[0]), float(g["longitude"].iloc[0]))
              for s, g in df.groupby(cfg.SITE_COL)}
    out = []
    for group in build_station_groups(coords, num_stations):
        members = list(dict.fromkeys(group))
        common = None
        for site in members:
            idx = (df[df[cfg.SITE_COL] == site][cfg.TIME_COL]
                   .drop_duplicates().sort_values())
            ix = pd.DatetimeIndex(idx)
            common = ix if common is None else common.intersection(ix)
        if common is None or len(common) < min_len:
            continue
        out.append((group, common.sort_values()))
    return out


class Dataset_UKPV(Dataset):
    """Grouped multi-station windows over our uk_pv plants (GNN-faithful)."""

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: Optional[List[int]] = None,
        features: str = "MS",
        target: str = "power",
        scale: bool = False,
        timeenc: int = 0,
        freq: str = "t",
        start_time: str = "",
        end_time: str = "",
        num_stations: int = 8,
        data_path: Optional[str] = None,
        dataset: str = "uk_pv",
    ):
        if size is None:
            self.seq_len, self.label_len, self.pred_len = (
                cfg.HISTORY_STEPS, cfg.HORIZON_STEPS, cfg.HORIZON_STEPS)
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.flag = flag
        self.features = features
        self.timeenc = timeenc
        self.freq = freq
        self.num_stations = num_stations
        self.dataset = dataset
        # fixed-scaling protocol: no fitted scaler, identity inverse
        self.scale = False
        self.cov_cols = list(cfg.COV_COLS)
        self.cov_scale = np.array([cfg.COV_SCALES[c] for c in self.cov_cols],
                                  dtype=np.float64)
        self.feature_dim = len(self.cov_cols) + 1   # covariates + norm_power

        self.data_path = data_path or os.environ.get(
            "DATA", cfg.DEFAULT_DATA_PATH)
        self.__read_data__()

    # ------------------------------------------------------------------
    def __read_data__(self):
        df = load_split_frame_df(self.data_path, self.flag, self.dataset,
                                 self.cov_cols)
        if df.empty:
            raise ValueError(f"[UKPV] no frame-bearing rows for split={self.flag}")
        usable = iter_usable_groups(df, self.num_stations,
                                    self.seq_len + self.pred_len)
        print(f"[UKPV] flag={self.flag}  usable_groups={len(usable)}  "
              f"S={self.num_stations}")

        # per-group aligned tensors + window index (gi == position in `usable`)
        self._gx, self._gy, self._gstamp, self._gts = [], [], [], []
        self._index = []   # (group_idx, local_start)
        for gi, (group, common) in enumerate(usable):
            x, y, stamp, tskeys = self._build_group(df, group, common, gi)
            self._gx.append(x); self._gy.append(y)
            self._gstamp.append(stamp); self._gts.append(tskeys)
            for s in range(max(0, len(x) - self.seq_len - self.pred_len + 1)):
                self._index.append((gi, s))
        if not self._index:
            raise ValueError(f"[UKPV] 0 windows for split={self.flag} "
                             f"(seq_len+pred_len={self.seq_len+self.pred_len})")
        print(f"[UKPV] flag={self.flag}  windows={len(self._index)}")

    def _build_group(self, df, group, common, gi):
        """Align a group's plants on the shared common index → [T,S,F]."""
        members = list(dict.fromkeys(group))
        per_plant = {site: (df[df[cfg.SITE_COL] == site]
                            .drop_duplicates(cfg.TIME_COL)
                            .set_index(cfg.TIME_COL).sort_index())
                     for site in members}
        S, T, F = self.num_stations, len(common), self.feature_dim
        x = np.zeros((T, S, F), dtype=np.float32)
        y = np.zeros((T, S, 1), dtype=np.float32)
        for si, site in enumerate(group):           # group may repeat (padding)
            g = per_plant[site].reindex(common)
            cov = g[self.cov_cols].to_numpy(np.float64) / self.cov_scale[None, :]
            tgt = g[[cfg.TARGET_COL]].to_numpy(np.float64)
            feats = np.nan_to_num(np.concatenate([cov, tgt], axis=1), nan=0.0)
            x[:, si, :] = feats.astype(np.float32)
            y[:, si, 0] = np.nan_to_num(tgt[:, 0], nan=0.0).astype(np.float32)

        stamp = self._time_features(common)
        tskeys = np.array([f"{gi}__{t.strftime('%Y%m%d%H%M')}" for t in common])
        return x, y, stamp, tskeys

    def _time_features(self, idx: pd.DatetimeIndex) -> np.ndarray:
        d = pd.DataFrame({"date": idx})
        if self.timeenc == 0:
            d["month"] = d.date.dt.month
            d["day"] = d.date.dt.day
            d["weekday"] = d.date.dt.weekday
            d["hour"] = d.date.dt.hour
            d["minute"] = d.date.dt.minute
            if self.freq == "t":
                d["minute"] = d["minute"] // 15
            return d.drop("date", axis=1).to_numpy(np.float32)
        from utils.timefeatures import time_features
        return time_features(pd.to_datetime(idx.values), freq=self.freq).transpose(1, 0).astype(np.float32)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        gi, s_begin = self._index[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self._gx[gi][s_begin:s_end]                       # [L,S,F]
        seq_y = self._gy[gi][r_begin:r_end].squeeze(-1)           # [ll+pl,S]
        seq_x_mark = self._gstamp[gi][s_begin:s_end]
        seq_y_mark = self._gstamp[gi][r_begin:r_end]
        ts_key = self._gts[gi][s_end]
        return (seq_x.astype(np.float32), seq_y.astype(np.float32),
                seq_x_mark.astype(np.float32), seq_y_mark.astype(np.float32), ts_key)

    def __len__(self) -> int:
        return len(self._index)

    def inverse_transform(self, data):
        """Identity: target is already capacity-normalized norm_power."""
        return data
