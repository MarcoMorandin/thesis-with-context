"""
Dataset loader for solarbench/SKIPPD (HuggingFace).

Loads three configs and joins on hourly timestamp:
  - 'default'  : sky image + time + pv  (1-min resolution → resampled to hourly)
  - 'ERA5'     : 26 ERA5 NWP covariates (already hourly)
  - 'labels'   : time + pv fallback

Returns tensors in the same format as Dataset_PV:
    seq_x     : [seq_len,  1, num_features]   float32
    seq_y     : [label_len+pred_len, 1]        float32
    seq_x_mark: [seq_len,  time_feat_dim]      float32
    seq_y_mark: [label_len+pred_len, time_feat_dim]

ERA5 nn1 features used (26 cols → appended before pv target):
    solar: ssrd, ssr, fdir, cdir, tisr, ssrc, ssrdc, tsrc, tsr
    cloud: tcc, hcc, mcc, lcc, cbh, tciw, tclw
    met:   t2m, d2m, u10, v10, u100, v100, fg10, i10fg, tp, fal
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
# ERA5 nn1 columns to use as covariates (ordered consistently)
# ---------------------------------------------------------------------------
ERA5_NN1_COLS = [
    # solar radiation
    'ssrd_nn1', 'ssr_nn1', 'fdir_nn1', 'cdir_nn1', 'tisr_nn1',
    'ssrc_nn1', 'ssrdc_nn1', 'tsrc_nn1', 'tsr_nn1',
    # cloud
    'tcc_nn1', 'hcc_nn1', 'mcc_nn1', 'lcc_nn1', 'cbh_nn1',
    'tciw_nn1', 'tclw_nn1',
    # meteorological
    't2m_nn1', 'd2m_nn1', 'u10_nn1', 'v10_nn1',
    'u100_nn1', 'v100_nn1', 'fg10_nn1', 'i10fg_nn1',
    'tp_nn1', 'fal_nn1',
]

_TIME_COLS = ["date_time", "timestamp", "time", "datetime", "valid_time",
              "Date", "Timestamp", "DateTime", "date"]
_TARGET_COLS = ["pv", "power", "GHI", "ghi", "irradiance", "normalized_power",
                "power_mw", "power_kw", "energy"]

HF_DATASET_ID = "solarbench/SKIPPD"


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def _read_flat_parquet(path) -> pd.DataFrame:
    """Read parquet dropping nested/binary columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    flat_cols = [
        schema.field(i).name for i in range(len(schema))
        if not (pa.types.is_struct(schema.field(i).type)
                or pa.types.is_list(schema.field(i).type)
                or pa.types.is_large_list(schema.field(i).type)
                or pa.types.is_large_binary(schema.field(i).type)
                or pa.types.is_map(schema.field(i).type))
    ]
    return pf.read(columns=flat_cols).to_pandas()


def _detect_time_col(df: pd.DataFrame) -> str:
    for c in _TIME_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c].iloc[:5])
            return c
        except Exception:
            pass
    raise ValueError(f"Cannot detect timestamp column. Columns: {list(df.columns)}")


def _to_utc_naive_hourly(series: pd.Series) -> pd.Series:
    """Convert tz-aware timestamps to UTC-naive, floored to hour."""
    s = pd.to_datetime(series, utc=True).dt.tz_convert(None)
    return s.dt.floor('1h')


# ---------------------------------------------------------------------------
# HF download & cache helpers
# ---------------------------------------------------------------------------

def _download_and_cache_config(cfg: str, cache: Path, hf_token=None) -> dict:
    """Download one HF config and save per-split parquets."""
    import datasets as hf_datasets
    cache.mkdir(parents=True, exist_ok=True)
    load_kwargs = dict(
        path=HF_DATASET_ID,
        name=cfg,
        cache_dir=str(cache / "hf_raw"),
    )
    if hf_token:
        load_kwargs["token"] = hf_token
    ds = hf_datasets.load_dataset(**load_kwargs)
    splits = {}
    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        out = cache / f"{cfg}_{split_name}.parquet"
        df.to_parquet(out, index=False)
        splits[split_name] = df
        print(f"  [{cfg}] {split_name}: {len(df)} rows -> {out.name}")
    return splits


def _load_or_download(cfg: str, cache: Path, hf_token=None) -> dict:
    """Return {split_name: DataFrame} for a given HF config."""
    # Check if already cached
    try:
        cached = {p.stem.split('_', 1)[1]: p
                  for p in cache.glob(f"{cfg}_*.parquet")}
        if cached:
            result = {}
            for split_name, p in cached.items():
                df = _read_flat_parquet(p)
                result[split_name] = df
            return result
    except Exception as e:
        print(f"Error scanning cache: {e}")

    try:
        return _download_and_cache_config(cfg, cache, hf_token)
    except Exception as e:
        # Debug helper: print directory contents to help locate the cache
        print("\n" + "="*80)
        print(f"OFFLINE CACHE ERROR: Failed to load or download SKIPPD configuration '{cfg}'.")
        print(f"Target Cache Directory: {cache.resolve()}")
        print(f"Directory exists: {cache.exists()}")
        if cache.exists():
            try:
                print(f"Directory contents: {[p.name for p in cache.iterdir()]}")
            except Exception as d_err:
                print(f"Failed to list cache contents: {d_err}")
        
        # Walk up and print parent directory contents
        try:
            parent = cache.parent
            print(f"Parent Directory: {parent.resolve()}")
            if parent.exists():
                print(f"Parent contents: {[p.name for p in parent.iterdir()]}")
                for child in parent.iterdir():
                    if child.is_dir() and "skippd" in child.name.lower():
                        print(f"  Subdirectory '{child.name}' contents: {[p.name for p in child.iterdir()]}")
                        sub_cache = child / "skippd_hf_cache"
                        if sub_cache.exists():
                            print(f"    Found skippd_hf_cache under {child.name}! Contents: {[p.name for p in sub_cache.iterdir()]}")
        except Exception as p_err:
            print(f"Failed to list parent contents: {p_err}")
            
        try:
            grandparent = cache.parent.parent
            print(f"Grandparent Directory: {grandparent.resolve()}")
            if grandparent.exists():
                print(f"Grandparent contents: {[p.name for p in grandparent.iterdir()]}")
        except Exception as gp_err:
            print(f"Failed to list grandparent contents: {gp_err}")
        print("="*80 + "\n")
        raise e


# ---------------------------------------------------------------------------
# Main join logic
# ---------------------------------------------------------------------------

def _build_merged_splits(cache: Path, hf_token=None) -> dict[str, pd.DataFrame]:
    """
    Merge ALL ERA5 + labels data (ignoring HF train/test split which is unbalanced),
    join on hour-floored timestamp, then do a temporal 70/15/15 split.
    Returns {'train': df, 'val': df, 'test': df}.
    """
    print("[SKIPPD] Loading ERA5 covariates ...")
    era5_splits  = _load_or_download('ERA5',   cache, hf_token)
    print("[SKIPPD] Loading labels ...")
    label_splits = _load_or_download('labels', cache, hf_token)

    # Concatenate all splits of each config
    era5_all  = pd.concat(list(era5_splits.values()),  ignore_index=True)
    label_all = pd.concat(list(label_splits.values()), ignore_index=True)

    # Normalize time → UTC-naive hour
    era5_tc   = _detect_time_col(era5_all)
    label_tc  = _detect_time_col(label_all)
    era5_all['_hour']  = _to_utc_naive_hourly(era5_all[era5_tc])
    label_all['_hour'] = _to_utc_naive_hourly(label_all[label_tc])

    # Resample pv to hourly mean
    pv_hourly = (label_all.set_index('_hour')['pv']
                 .resample('1h').mean()
                 .reset_index())

    # ERA5 nn1 cols
    era5_cols = [c for c in ERA5_NN1_COLS if c in era5_all.columns]
    era5_slim = era5_all[['_hour'] + era5_cols].drop_duplicates('_hour')

    # Inner join on hour
    merged = pd.merge(era5_slim, pv_hourly, on='_hour', how='inner')
    merged = (merged.sort_values('_hour')
                    .dropna(subset=['pv'])
                    .reset_index(drop=True)
                    .rename(columns={'_hour': 'time'}))

    n = len(merged)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    print(f"[SKIPPD] Total merged: {n} hourly rows, {len(era5_cols)} ERA5 features + pv")
    print(f"[SKIPPD] Temporal split: train={n_train}  val={n_val}  test={n-n_train-n_val}")

    return {
        'train': merged.iloc[:n_train].reset_index(drop=True),
        'val':   merged.iloc[n_train:n_train+n_val].reset_index(drop=True),
        'test':  merged.iloc[n_train+n_val:].reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Dataset_SKIPPD(Dataset):
    """HuggingFace solarbench/SKIPPD with ERA5 covariates — single station."""

    STATION_NAME  = "skippd_site"
    STATION_COORD = (-105.1786, 39.7392)  # (lon, lat) Denver area

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: Optional[List[int]] = None,
        features: str = "MS",
        target: str = "pv",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        start_time: str = "2010-01-01 00:00",
        end_time: str = "2023-12-31 23:00",
        feature_cols: Optional[List[str]] = None,
        hf_cache_dir: Optional[str] = None,
        use_hf_splits: bool = True,
        hf_token: Optional[str] = None,
        use_era5: bool = True,
    ):
        if size is None:
            self.seq_len   = 96
            self.label_len = 48
            self.pred_len  = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.flag         = flag
        self.features     = features
        self.target       = target
        self.scale        = scale
        self.timeenc      = timeenc
        self.freq         = freq
        self.start_time   = start_time
        self.end_time     = end_time
        self.root_path    = root_path
        self.feature_cols = feature_cols
        self.use_hf_splits = use_hf_splits
        self.hf_token     = hf_token
        self.use_era5     = use_era5
        self.num_stations = 1

        if hf_cache_dir:
            cache_root = hf_cache_dir
        else:
            # Check candidate cache paths to support offline run structures where the
            # cache might reside in sibling folders (e.g., 'dataset/skippd/skippd_hf_cache'
            # vs refactored 'dataset/refactored/skippd_hf_cache')
            candidates = [
                os.path.join(root_path, "skippd_hf_cache"),
                os.path.join(os.path.dirname(os.path.normpath(root_path)), "skippd", "skippd_hf_cache"),
                os.path.join(os.path.dirname(os.path.normpath(root_path)), "skippd_hf_cache"),
            ]
            cache_root = candidates[0]
            print("[SKIPPD] Searching for cached datasets. Checked candidates:")
            for cand in candidates:
                cand_path = Path(cand)
                exists = cand_path.exists()
                has_parquet = exists and any(cand_path.glob("*.parquet"))
                has_raw = exists and (cand_path / "hf_raw").exists()
                print(f"  - {cand} (exists: {exists}, has_parquet: {has_parquet}, has_raw: {has_raw})")
                if has_parquet or has_raw:
                    cache_root = cand
                    print(f"[SKIPPD] Found cached datasets in: {cand_path.resolve()}")
                    break
        self.hf_cache_dir = cache_root

        self.__read_data__()

    # ------------------------------------------------------------------
    def _load_splits(self) -> dict[str, pd.DataFrame]:
        cache = Path(self.hf_cache_dir)
        cache.mkdir(parents=True, exist_ok=True)

        refactored = Path(self.root_path) / "numerical" / "skippd.parquet"
        if refactored.exists() and not self.use_era5:
            print(f"[SKIPPD] Loading refactored numerical parquet: {refactored}")
            df = pd.read_parquet(refactored)
            df = df.rename(columns={"timestamp_utc": "time", "power_w": "pv"})
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
            df = df.dropna(subset=["time", "pv"]).sort_values("time").reset_index(drop=True)
            n = len(df)
            n_train = int(n * 0.70)
            n_val = int(n * 0.15)
            return {
                "train": df.iloc[:n_train].reset_index(drop=True),
                "val": df.iloc[n_train:n_train + n_val].reset_index(drop=True),
                "test": df.iloc[n_train + n_val:].reset_index(drop=True),
            }

        if self.use_era5:
            # Hourly ERA5 + labels join (2711 rows)
            sentinel = cache / "merged_train.parquet"
            prefix   = "merged_"
        else:
            # Full 1-min labels only (~363k rows)
            sentinel = cache / "labels1min_train.parquet"
            prefix   = "labels1min_"

        if sentinel.exists():
            print(f"[SKIPPD] Loading {'ERA5-merged' if self.use_era5 else '1-min labels'} cache")
            return {p.stem[len(prefix):]: pd.read_parquet(p)
                    for p in sorted(cache.glob(f"{prefix}*.parquet"))}

        if self.use_era5:
            splits = _build_merged_splits(cache, self.hf_token)
        else:
            splits = self._build_1min_splits(cache)

        for split_name, df in splits.items():
            df.to_parquet(cache / f"{prefix}{split_name}.parquet", index=False)
        return splits

    def _build_1min_splits(self, cache: Path) -> dict[str, pd.DataFrame]:
        """Load full 1-min labels (no ERA5), temporal 70/15/15 split."""
        label_splits = _load_or_download('labels', cache, self.hf_token)
        label_all    = pd.concat(list(label_splits.values()), ignore_index=True)

        tc = _detect_time_col(label_all)
        label_all[tc] = pd.to_datetime(label_all[tc], utc=True).dt.tz_convert(None)
        label_all = label_all.sort_values(tc).reset_index(drop=True)
        label_all = label_all.rename(columns={tc: "time"})
        label_all = label_all.dropna(subset=["pv"]).reset_index(drop=True)

        n = len(label_all)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        print(f"[SKIPPD-1min] Total: {n} rows  train={n_train}  val={n_val}  test={n-n_train-n_val}")
        return {
            "train": label_all.iloc[:n_train].reset_index(drop=True),
            "val":   label_all.iloc[n_train:n_train+n_val].reset_index(drop=True),
            "test":  label_all.iloc[n_train+n_val:].reset_index(drop=True),
        }

    # ------------------------------------------------------------------
    def __read_data__(self):
        splits = self._load_splits()

        name_map = {}
        for k in splits:
            lk = k.lower()
            if "train" in lk:   name_map["train"] = k
            elif "val"  in lk:
                name_map["val"] = k
                name_map["validation"] = k
            elif "test" in lk:  name_map["test"]   = k

        # _build_merged_splits always returns train/val/test
        df_split          = splits[name_map[self.flag]].copy()
        df_all_for_scaler = splits[name_map["train"]].copy()

        # ---------- feature columns ----------
        time_col = "time"
        all_numeric = [c for c in df_split.select_dtypes(include="number").columns
                       if c != self.target]

        if self.feature_cols is not None:
            feat_cols = [c for c in self.feature_cols if c in df_split.columns]
        elif self.use_era5:
            era5_present = [c for c in ERA5_NN1_COLS if c in df_split.columns]
            feat_cols = era5_present if era5_present else all_numeric
        else:
            # 1-min mode: univariate pv only
            feat_cols = [self.target]

        # Target always last
        feat_cols = [c for c in feat_cols if c != self.target] + [self.target]

        self.feature_cols = feat_cols
        self.feature_dim  = len(feat_cols)
        print(f"[SKIPPD] flag={self.flag}  target='{self.target}'  "
              f"n_features={len(feat_cols)}  rows={len(df_split)}")

        # ---------- sort & fill ----------
        df_split[time_col] = pd.to_datetime(df_split[time_col], errors="coerce")
        df_split = df_split.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        self.ts_keys = df_split[time_col].dt.strftime('%Y%m%d%H%M').values

        raw_df   = df_split[feat_cols].apply(pd.to_numeric, errors="coerce")
        raw_df   = raw_df.interpolate(method="linear", limit_direction="both").ffill().bfill()
        raw_data = raw_df.values.astype(np.float64)

        target_idx = feat_cols.index(self.target)
        raw_y = raw_data[:, target_idx:target_idx+1]

        # ---------- scaling ----------
        self.scaler_x  = StandardScaler()
        self.scaler_y  = [StandardScaler()]

        if self.scale:
            df_train_raw = df_all_for_scaler[feat_cols].apply(pd.to_numeric, errors="coerce")
            df_train_raw = df_train_raw.ffill().bfill()
            train_data   = df_train_raw.values.astype(np.float64)
            train_y      = train_data[:, target_idx:target_idx+1]
            self.scaler_x.fit(train_data)
            self.scaler_y[0].fit(train_y)
            scaled_x = self.scaler_x.transform(raw_data)
            scaled_y = self.scaler_y[0].transform(raw_y)
        else:
            scaled_x, scaled_y = raw_data, raw_y

        self.data_x = scaled_x[:, np.newaxis, :]   # [T, 1, F]
        self.data_y = scaled_y[:, np.newaxis, :]   # [T, 1, 1]

        # ---------- time features ----------
        ts = pd.to_datetime(df_split[time_col].values)
        df_stamp = pd.DataFrame({"date": ts})

        if self.timeenc == 0:
            df_stamp["month"]   = df_stamp.date.dt.month
            df_stamp["day"]     = df_stamp.date.dt.day
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            df_stamp["hour"]    = df_stamp.date.dt.hour
            df_stamp["minute"]  = df_stamp.date.dt.minute
            if self.freq == "t":
                df_stamp["minute"] = df_stamp["minute"] // 15
            self.data_stamp = df_stamp.drop("date", axis=1).values
        elif self.timeenc == 1:
            self.data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            ).transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc={self.timeenc}")

        print(f"[SKIPPD] data_x={self.data_x.shape}  "
              f"data_y={self.data_y.shape}  "
              f"stamp={self.data_stamp.shape}")

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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
