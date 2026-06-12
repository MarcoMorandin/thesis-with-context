"""
Dataset loader for Wollongong PV + sky-images (Mendeley DOI 10.17632/cb8t8np9z3).

Three days of data (2019-09-10/11/12) from two rooftop PV systems at
University of Wollongong, Australia:
  - Location 1 (Innovation Campus SBRC): column 'AVG power'
  - Location 3 (Main Campus B28):        column 'P_Avg[W]'

Power: 30-second resolution xlsx files (resampled to 1-min here).
Sky images: 1024x768 JPEG at 10-second cadence (Cameras 1 + 2; not used
in this loader — requires separate Qwen3-VL precompute pass).

This loader is *test-only* (zero-shot transfer evaluation).
Returns the same tensor shapes as Dataset_SKIPPD:
    seq_x     : [seq_len,  1, num_features]   float32
    seq_y     : [label_len+pred_len, 1]        float32
    seq_x_mark: [seq_len,  time_feat_dim]      float32
    seq_y_mark: [label_len+pred_len, time_feat_dim]
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Power-column → station-id mapping
# ---------------------------------------------------------------------------
_LOC1_POWER_COL = "AVG power"   # Location 1 — SBRC, Innovation Campus
_LOC3_POWER_COL = "P_Avg[W]"    # Location 3 — Building 28, Main Campus

# Approximate site coordinates (lon, lat)
_WOLLONGONG_COORDS = {
    'loc1': (150.90,  -34.40),   # Innovation Campus
    'loc3': (150.88,  -34.41),   # Main Campus
}


def _load_xlsx_day(path: Path) -> tuple[pd.DataFrame, str]:
    """Load one Wollongong xlsx, return (df with [time, pv], station_id)."""
    # The relevant sheet is the second one ('YYYY_MM_DD'), Sheet1 is unstable.
    xl = pd.ExcelFile(path)
    date_sheets = [s for s in xl.sheet_names if s != 'Sheet1']
    sheet = date_sheets[0] if date_sheets else xl.sheet_names[-1]
    df = pd.read_excel(path, sheet_name=sheet)

    # Detect station by power column name
    if _LOC1_POWER_COL in df.columns:
        station = 'loc1'
        time_col, pv_col = 'Date and Time', _LOC1_POWER_COL
    elif _LOC3_POWER_COL in df.columns:
        station = 'loc3'
        time_col, pv_col = 'Date', _LOC3_POWER_COL
    else:
        # Heuristic fallback
        time_col = next(c for c in df.columns
                        if 'date' in str(c).lower() or 'time' in str(c).lower())
        pv_col   = next(c for c in df.columns
                        if c != time_col and pd.api.types.is_numeric_dtype(df[c]))
        station  = 'unknown'

    df = df[[time_col, pv_col]].rename(columns={time_col: 'time', pv_col: 'pv'})
    df['time'] = pd.to_datetime(df['time'])
    df = df.dropna(subset=['time', 'pv']).sort_values('time').reset_index(drop=True)
    return df, station


def _build_combined_df(root: Path, station_filter: Optional[str] = None) -> pd.DataFrame:
    """Walk root recursively, find all xlsx files, concat per station."""
    refactored = root / "numerical" / "wollongong.parquet"
    if refactored.exists():
        df = pd.read_parquet(refactored)
        df = df.rename(columns={"timestamp_utc": "time", "power_w": "pv"})
        if station_filter:
            df = df[df["station_id"] == station_filter]
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
        return (
            df[["time", "pv"]]
            .dropna(subset=["time", "pv"])
            .sort_values("time")
            .reset_index(drop=True)
        )

    xlsx_files = sorted(root.rglob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found under {root}")

    per_station: dict[str, list[pd.DataFrame]] = {}
    for p in xlsx_files:
        df, station = _load_xlsx_day(p)
        per_station.setdefault(station, []).append(df)

    if station_filter and station_filter not in per_station:
        avail = sorted(per_station.keys())
        raise ValueError(f"station='{station_filter}' not found. Available: {avail}")

    if station_filter:
        merged = pd.concat(per_station[station_filter], ignore_index=True)
    else:
        # Default: use loc1 (Innovation Campus, has the camera 1 imagery)
        chosen = 'loc1' if 'loc1' in per_station else sorted(per_station)[0]
        print(f"[Wollongong] Using station='{chosen}' (available: {sorted(per_station)})")
        merged = pd.concat(per_station[chosen], ignore_index=True)

    # Deduplicate by timestamp (same minute might appear in multiple files)
    merged = (merged.drop_duplicates(subset='time')
                    .sort_values('time')
                    .reset_index(drop=True))
    return merged


class Dataset_WOLLONGONG(Dataset):
    """Wollongong PV — test-only cross-site evaluation."""

    def __init__(
        self,
        root_path: str,
        flag: str = "test",
        size: Optional[List[int]] = None,
        features: str = "MS",
        target: str = "pv",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "t",
        start_time: str = "2019-09-10 00:00",
        end_time:   str = "2019-09-13 00:00",
        feature_cols: Optional[List[str]] = None,
        station: str = "loc1",     # 'loc1' or 'loc3'
        resample_freq: str = "1min",
    ):
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 60, 30, 15
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.flag       = flag
        self.features   = features
        self.target     = target
        self.scale      = scale
        self.timeenc    = timeenc
        self.freq       = freq
        self.start_time = start_time
        self.end_time   = end_time
        self.root_path  = root_path
        self.station    = station
        self.resample_freq = resample_freq

        self.num_stations  = 1
        self.STATION_COORD = _WOLLONGONG_COORDS.get(station, (150.90, -34.40))
        self.STATION_NAME  = f"wollongong_{station}"

        self.__read_data__()

    def __read_data__(self):
        # 1) Load all xlsx for the chosen station
        df = _build_combined_df(Path(self.root_path), station_filter=self.station)
        print(f"[Wollongong] Loaded {len(df)} raw rows for station={self.station}")

        # 2) Resample 30s -> 1-min mean
        df = (df.set_index('time')
                .resample(self.resample_freq).mean()
                .interpolate(method='linear', limit=5)
                .reset_index())
        df = df.dropna(subset=['pv']).reset_index(drop=True)
        print(f"[Wollongong] After resample to {self.resample_freq}: {len(df)} rows  "
              f"range={df['time'].min()} → {df['time'].max()}")

        # 3) Build feature matrix — univariate pv (matches SKIPPD-trained model)
        feat_cols = [self.target]
        self.feature_cols = feat_cols
        self.feature_dim  = len(feat_cols)

        raw = df[feat_cols].astype(np.float64).values   # [T, 1]

        # 4) Target-domain scaler — fit on actual Wollongong daytime data.
        # Capacity-scaled SKIPPD stats caused systematic bias because Wollongong
        # capacity factors differ from SKIPPD's (SKIPPD CF≈0.454, loc1 CF≈0.363).
        self.scaler_x = StandardScaler()
        self.scaler_y = [StandardScaler()]
        if self.scale:
            # Mask out night zeros before fitting so scaler reflects daytime distribution
            daytime_mask = raw[:, 0] > 0
            fit_data = raw[daytime_mask] if daytime_mask.sum() > 10 else raw
            self.scaler_x.fit(fit_data)
            self.scaler_y[0].fit(fit_data)
            scaled = self.scaler_x.transform(raw)
            print(f"[Wollongong] Scaler mean={self.scaler_x.mean_[0]:.1f}W  "
                  f"std={self.scaler_x.scale_[0]:.1f}W  "
                  f"(fit on {daytime_mask.sum()} daytime rows)")
        else:
            scaled = raw

        self.data_x = scaled[:, np.newaxis, :]              # [T, 1, F]
        self.data_y = scaled[:, np.newaxis, :]              # [T, 1, 1]

        # 5) Vision-store keys (YYYYMMDDHHMM, UTC-naive)
        self.ts_keys = df['time'].dt.strftime('%Y%m%d%H%M').values

        # 6) Time features
        ts = pd.to_datetime(df['time'].values)
        df_stamp = pd.DataFrame({"date": ts})
        if self.timeenc == 0:
            df_stamp["month"]   = df_stamp.date.dt.month
            df_stamp["day"]     = df_stamp.date.dt.day
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            df_stamp["hour"]    = df_stamp.date.dt.hour
            df_stamp["minute"]  = df_stamp.date.dt.minute
            self.data_stamp = df_stamp.drop("date", axis=1).values
        elif self.timeenc == 1:
            self.data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            ).transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc={self.timeenc}")

        print(f"[Wollongong] data_x={self.data_x.shape}  "
              f"data_y={self.data_y.shape}  stamp={self.data_stamp.shape}")

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError("index out of range")
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end   - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x      = self.data_x[s_begin:s_end].astype(np.float32)
        seq_y      = self.data_y[r_begin:r_end].squeeze(-1).astype(np.float32)
        seq_x_mark = self.data_stamp[s_begin:s_end].astype(np.float32)
        seq_y_mark = self.data_stamp[r_begin:r_end].astype(np.float32)
        ts_key     = self.ts_keys[s_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, ts_key

    def __len__(self) -> int:
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        was_tensor = isinstance(data, torch.Tensor)
        if was_tensor:
            device  = data.device
            np_data = data.detach().cpu().numpy()
        else:
            np_data = np.array(data)
        orig_shape = np_data.shape
        restored = self.scaler_y[0].inverse_transform(np_data.reshape(-1, 1))
        restored = restored.reshape(orig_shape)
        if was_tensor:
            return torch.from_numpy(restored).to(device)
        return restored
