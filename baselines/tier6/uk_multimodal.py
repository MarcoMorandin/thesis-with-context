"""uk_pv multimodal bridge: numerical (Y, cov) + satellite frames (V).

Tier-6's PV-specialized models (SUNSET, CrossViViT) need **real frames**. The
uk_pv track carries them: each curated row has an `image_uk128_index` pointing
into `images_uk128.h5` (per-site group `uk_pv_<site>` → `images (N,128,128)`
uint8 + `timestamps (N,) S20`, 30-min daylight cadence, 2019-2020), per
DATASET_CONTRACT.md §1bis.

This module reuses `common.windows` for *all* numerical logic (the disjoint
plant splits, NaN handling, deterministic-future-covariate masking, seasonal
reference) and only adds the visual tensor `V` + `mask_visual` over the history
window, so the vendored runners stay thin and the fairness contract is shared
with Tiers 0-5. No ETL here — frames are read straight from the standardized
HDF5 by the canonical `image_uk128_index`.

    from tier6.uk_multimodal import UKMultimodalDataset
    ds = UKMultimodalDataset(site_ids={"10793"}, img_size=64)
    item = ds[0]            # item["V"] -> (T, 1, 64, 64) in [0,1]; item["mask_visual"] -> (T,)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # baselines/ on path

from common import config              # noqa: E402
from common.windows import dataset_for_sites  # noqa: E402

DATASET = "uk_pv"
DEFAULT_H5 = "/Volumes/SSD/standardized-dataset/images_uk128.h5"
FRAME_IDX_COL = "image_uk128_index"
RAW_IMG_SIDE = 128


def _to_unix_seconds(ts: pd.Series) -> np.ndarray:
    """Match common.windows' resolution-robust epoch conversion exactly."""
    t = pd.DatetimeIndex(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return ((t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")).to_numpy().astype(
        np.int64
    )


def _block_downsample(img: np.ndarray, side: int) -> np.ndarray:
    """Average-pool a square uint8 frame to (side, side) in [0,1], no extra deps."""
    if side == img.shape[0]:
        return img.astype(np.float32) / 255.0
    if img.shape[0] % side != 0:
        raise ValueError(f"{img.shape[0]} not divisible by target side {side}")
    f = img.shape[0] // side
    return (
        img.astype(np.float32)
        .reshape(side, f, side, f)
        .mean(axis=(1, 3))
        / 255.0
    )


class UKMultimodalDataset:
    """Sliding windows of (Y, cov, V) for uk_pv plants.

    Wraps a numerical `WindowDataset` (one split's plants) and attaches, per
    window, the satellite frames over the **history** window (T_v = history):
    `V` (T, 1, S, S) in [0,1] and `mask_visual` (T,) — 1 where a frame exists
    on that 30-min step (daylight), 0 otherwise (night/outage).
    """

    def __init__(
        self,
        site_ids: set[str] | list[str],
        data_path: str = config.DEFAULT_DATA_PATH,
        h5_path: str = DEFAULT_H5,
        time_range: tuple[float, float] | None = None,
        history: int = config.HISTORY_STEPS,
        horizon: int = config.HORIZON_STEPS,
        stride: int = 1,
        img_size: int = RAW_IMG_SIDE,
        future_cov: str = "deterministic",
    ):
        self.h5_path = h5_path
        self.history = history
        self.img_size = img_size
        site_ids = {str(s) for s in site_ids}

        cols = [
            config.DATASET_COL, config.SITE_COL, config.TIME_COL,
            config.TARGET_COL, config.CAPACITY_COL, config.CLEARSKY_COL,
            FRAME_IDX_COL, "latitude", "longitude", *config.COV_COLS,
        ]
        df = pd.read_parquet(data_path, columns=sorted(set(cols)))
        df = df[df[config.DATASET_COL] == DATASET].copy()
        df[config.SITE_COL] = df[config.SITE_COL].astype(str)
        df = df[df[config.SITE_COL].isin(site_ids)]
        if df.empty:
            raise ValueError(f"no uk_pv rows for sites {sorted(site_ids)}")

        # numerical windows (reuses the shared fairness/window logic)
        self.win = dataset_for_sites(
            df, site_ids, time_range,
            history=history, horizon=horizon, stride=stride,
            future_cov=future_cov,
        )

        # per-site {unix_ts -> frame_idx} from the canonical pointer column
        self.frame_maps: dict[str, dict[int, int]] = {}
        self.coords: dict[str, tuple[float, float]] = {}
        for site, g in df.groupby(config.SITE_COL):
            self.coords[str(site)] = (
                float(g["latitude"].iloc[0]), float(g["longitude"].iloc[0])
            )
            fg = g.dropna(subset=[FRAME_IDX_COL])
            ts = _to_unix_seconds(fg[config.TIME_COL])
            idx = fg[FRAME_IDX_COL].to_numpy().astype(np.int64)
            self.frame_maps[str(site)] = dict(zip(ts.tolist(), idx.tolist()))

        self._h5 = None  # opened lazily (h5py handles are not fork-safe)

    def __len__(self) -> int:
        return len(self.win)

    def _h5_group(self, site: str):
        import h5py

        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5[f"uk_pv_{site}"]

    def __getitem__(self, i: int) -> dict:
        item = self.win[i]
        T, S = self.history, self.img_size
        site = str(item["site_id"])
        fmap = self.frame_maps.get(site, {})
        hist_ts = item["timestamps"][:T]

        V = np.zeros((T, 1, S, S), dtype=np.float32)
        mask_v = np.zeros((T,), dtype=np.float32)
        wanted = {int(t): j for j, t in enumerate(hist_ts.tolist()) if int(t) in fmap}
        if wanted:
            g = self._h5_group(site)
            images = g["images"]
            for t, j in wanted.items():
                frame = images[fmap[t]]
                V[j, 0] = _block_downsample(np.asarray(frame), S)
                mask_v[j] = 1.0

        item["V"] = V
        item["mask_visual"] = mask_v
        item["latlon"] = np.array(self.coords.get(site, (np.nan, np.nan)), np.float32)
        return item

    def batch(self, indices: list[int]) -> dict:
        items = [self[i] for i in indices]
        keys_stack = (
            "y_hist", "mask_hist", "y_future", "mask_future", "daylight_future",
            "cov", "clearsky", "V", "mask_visual", "latlon", "timestamps",
        )
        out = {k: np.stack([it[k] for it in items]) for k in keys_stack}
        out["site_id"] = np.array([it["site_id"] for it in items])
        out["capacity"] = np.array([it["capacity"] for it in items], np.float32)
        return out

    def iter_batches(self, batch_size: int = 64):
        for lo in range(0, len(self), batch_size):
            yield self.batch(list(range(lo, min(lo + batch_size, len(self)))))


def sites_for_split(part: str) -> list[str]:
    """Plant ids for a split part ('train' | 'val' | 'test') of uk_pv."""
    from common.splits import load_splits

    return [str(s) for s in load_splits()[DATASET][part]]
