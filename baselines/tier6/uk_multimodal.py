"""Multimodal bridge: numerical (Y, cov) + satellite frames (V), both datasets.

Tier-6's PV-specialized models (SUNSET, CrossViViT) need **real frames**. The
dataset of record carries them: each row has an `image_h5_index` pointing into
`images_all.h5` (per-site group `<dataset>_<site>` → `images` uint8 +
`timestamps (N,) S20`), per DATASET_CONTRACT.md §1.0. `uk_pv` frames are
`(N,128,128)` grayscale (30-min daylight cadence), `goes_pvdaq` `(N,256,256,3)`
RGB (15-min); with ``to_gray`` (default) both reduce to one channel so a site set
can span both datasets in one run.

This module reuses `common.windows` for *all* numerical logic (the disjoint
plant splits, NaN handling, deterministic-future-covariate masking, seasonal
reference) and only adds the visual tensor `V` + `mask_visual` over the history
window, so the vendored runners stay thin and the fairness contract is shared
with Tiers 0-5. No ETL here — frames are read straight from `images_all.h5` by
the canonical `image_h5_index`.

    from tier6.uk_multimodal import UKMultimodalDataset, sites_for_split
    ds = UKMultimodalDataset(site_ids={"10793"}, img_size=64)         # uk_pv plant
    ds = UKMultimodalDataset(site_ids=sites_for_split("test"))        # both datasets
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

DEFAULT_H5 = config.DEFAULT_IMAGES_H5     # images_all.h5 (both datasets)
FRAME_IDX_COL = config.FRAME_INDEX_COL    # image_h5_index (local-to-group)
RAW_IMG_SIDE = 128


def _to_unix_seconds(ts: pd.Series) -> np.ndarray:
    """Match common.windows' resolution-robust epoch conversion exactly."""
    t = pd.DatetimeIndex(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return ((t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")).to_numpy().astype(
        np.int64
    )


def _downsample(img: np.ndarray, side: int, to_gray: bool = True) -> np.ndarray:
    """Average-pool a uint8 frame to (side, side, C) in [0,1], no extra deps.

    Handles both `uk_pv` grayscale `(H, W)` and `goes_pvdaq` RGB `(H, W, 3)`
    frames. With ``to_gray`` the channels are averaged to a single channel so
    the visual tensor `V` is homogeneous `(T, 1, S, S)` across datasets.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:                       # (H, W) -> (H, W, 1)
        img = img[..., None]
    h, w, c = img.shape
    if side != h or side != w:
        if h % side or w % side:
            raise ValueError(f"{(h, w)} not divisible by target side {side}")
        fy, fx = h // side, w // side
        img = img.reshape(side, fy, side, fx, c).mean(axis=(1, 3))  # (S, S, C)
    img = img / 255.0
    if to_gray and img.shape[-1] > 1:
        img = img.mean(axis=-1, keepdims=True)
    return img                              # (S, S, C')


class UKMultimodalDataset:
    """Sliding windows of (Y, cov, V) for PV plants (uk_pv and/or goes_pvdaq).

    Wraps a numerical `WindowDataset` (one split's plants, any dataset) and
    attaches, per window, the satellite frames over the **history** window
    (T_v = history): `V` (T, C, S, S) in [0,1] and `mask_visual` (T,) — 1 where a
    frame exists on that step (daylight), 0 otherwise (night/outage).

    Frames are read from `images_all.h5` by the canonical `image_h5_index`
    pointer (per-site group `<dataset>_<site>`). uk_pv frames are 128px
    grayscale, goes_pvdaq 256px RGB; with ``to_gray`` (default) both are reduced
    to a single channel so `V` is homogeneous `(T, 1, S, S)` and a site set may
    span both datasets in one model run.
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
        datasets: set[str] | list[str] | None = None,
        to_gray: bool = True,
    ):
        self.h5_path = h5_path
        self.history = history
        self.img_size = img_size
        self.to_gray = to_gray
        self.channels = 1 if to_gray else None  # native channels if not gray
        site_ids = {str(s) for s in site_ids}

        cols = [
            config.DATASET_COL, config.SITE_COL, config.TIME_COL,
            config.TARGET_COL, config.CAPACITY_COL, config.CLEARSKY_COL,
            FRAME_IDX_COL, "latitude", "longitude", *config.COV_COLS,
        ]
        df = pd.read_parquet(data_path, columns=sorted(set(cols)))
        if datasets is not None:
            df = df[df[config.DATASET_COL].isin({str(d) for d in datasets})]
        df = df.copy()
        df[config.SITE_COL] = df[config.SITE_COL].astype(str)
        df = df[df[config.SITE_COL].isin(site_ids)]
        # only rows that actually carry a frame pointer (>=0; -1 = no frame)
        df = df[df[FRAME_IDX_COL].notna() & (df[FRAME_IDX_COL] >= 0)]
        if df.empty:
            raise ValueError(f"no frame-bearing rows for sites {sorted(site_ids)}")

        # numerical windows (reuses the shared fairness/window logic)
        self.win = dataset_for_sites(
            df, site_ids, time_range,
            history=history, horizon=horizon, stride=stride,
            future_cov=future_cov,
        )

        # per-(dataset, site) {unix_ts -> frame_idx} from the canonical pointer
        self.frame_maps: dict[tuple[str, str], dict[int, int]] = {}
        self.coords: dict[tuple[str, str], tuple[float, float]] = {}
        for (ds, site), g in df.groupby([config.DATASET_COL, config.SITE_COL]):
            key = (str(ds), str(site))
            self.coords[key] = (
                float(g["latitude"].iloc[0]), float(g["longitude"].iloc[0])
            )
            ts = _to_unix_seconds(g[config.TIME_COL])
            idx = g[FRAME_IDX_COL].to_numpy().astype(np.int64)
            self.frame_maps[key] = dict(zip(ts.tolist(), idx.tolist()))

        self._h5 = None  # opened lazily (h5py handles are not fork-safe)

    def __len__(self) -> int:
        return len(self.win)

    def _h5_group(self, dataset: str, site: str):
        import h5py

        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5[f"{dataset}_{site}"]

    def __getitem__(self, i: int) -> dict:
        item = self.win[i]
        T, S = self.history, self.img_size
        key = (str(item["dataset"]), str(item["site_id"]))
        fmap = self.frame_maps.get(key, {})
        hist_ts = item["timestamps"][:T]

        wanted = {int(t): j for j, t in enumerate(hist_ts.tolist()) if int(t) in fmap}
        C = self.channels or 1
        V = None
        mask_v = np.zeros((T,), dtype=np.float32)
        if wanted:
            g = self._h5_group(*key)
            images = g["images"]
            for t, j in sorted(wanted.items(), key=lambda kv: kv[1]):
                frame = _downsample(np.asarray(images[fmap[t]]), S, self.to_gray)
                if V is None:                       # learn C from the first frame
                    C = frame.shape[-1]
                    V = np.zeros((T, C, S, S), dtype=np.float32)
                V[j] = np.transpose(frame, (2, 0, 1))
                mask_v[j] = 1.0
        if V is None:
            V = np.zeros((T, C, S, S), dtype=np.float32)

        item["V"] = V
        item["mask_visual"] = mask_v
        item["latlon"] = np.array(self.coords.get(key, (np.nan, np.nan)), np.float32)
        return item

    def batch(self, indices: list[int]) -> dict:
        items = [self[i] for i in indices]
        keys_stack = (
            "y_hist", "mask_hist", "y_future", "mask_future", "daylight_future",
            "cov", "clearsky", "V", "mask_visual", "latlon", "timestamps",
        )
        out = {k: np.stack([it[k] for it in items]) for k in keys_stack}
        out["site_id"] = np.array([it["site_id"] for it in items])
        out["dataset"] = np.array([it["dataset"] for it in items])
        out["capacity"] = np.array([it["capacity"] for it in items], np.float32)
        return out

    def iter_batches(self, batch_size: int = 64):
        for lo in range(0, len(self), batch_size):
            yield self.batch(list(range(lo, min(lo + batch_size, len(self)))))


def sites_for_split(part: str, dataset: str | None = None) -> list[str]:
    """Plant ids for a split part ('train' | 'val' | 'test').

    With ``dataset`` restrict to that source; otherwise return the plants of
    **every** dataset in the split (uk_pv + goes_pvdaq) so a model run can span
    the whole dataset.
    """
    from common.splits import load_splits

    splits = load_splits()
    names = [dataset] if dataset else list(splits)
    return [str(s) for d in names for s in splits[d][part]]
