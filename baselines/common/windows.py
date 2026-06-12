"""Sliding-window dataset over the curated parquet.

Produces, per window, the numerical subset of the canonical dict defined in
docs/context/DATASET_CONTRACT.md §4 (no visual tensors — tiers 0-2 are
TS/covariate-only):

    y_hist        (T,)      normalized target history, NaN→0
    mask_hist     (T,)      1 = valid history step
    y_future      (H,)      ground truth, NaN→0
    mask_future   (H,)      1 = valid future step
    daylight_future (H,)    1 = sun above horizon (clearsky_ghi > 0)
    cov           (T+H, C)  scaled covariates; future rows of observed-weather
                            covariates are zeroed (no lookahead) unless
                            future_cov="all"
    clearsky      (T+H,)    raw clear-sky GHI in W/m² (smart persistence ref)
    y_seasonal    (H,)      target at the same clock time on the previous day
    mask_seasonal (H,)      1 = seasonal reference available
    timestamps    (T+H,)    unix seconds
    site_id, dataset, capacity

Windows are indexed lazily: only (series, start) pairs are materialized up
front, tensors are built on access. Native cadences are preserved (uk_pv
30 min, goes_pvdaq 15 min); each site is reindexed onto its regular grid so
lag arithmetic is exact and gaps stay NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from . import config


@dataclass
class SiteSeries:
    """One plant's series on a regular time grid (gaps = NaN)."""

    site_id: str
    dataset: str
    capacity: float
    timestamps: np.ndarray  # (L,) int64 unix seconds
    y: np.ndarray           # (L,) float32, NaN = missing/untrusted
    cov: np.ndarray         # (L, C) float32, scaled, NaN = missing
    clearsky: np.ndarray    # (L,) float32, raw W/m²
    steps_per_day: int


def build_site_series(
    df: pd.DataFrame, cov_cols: tuple[str, ...] = config.COV_COLS
) -> list[SiteSeries]:
    """Reindex each site onto its native regular grid and extract arrays."""
    scales = np.array([config.COV_SCALES[c] for c in cov_cols], dtype=np.float32)
    out: list[SiteSeries] = []
    for (dataset, site_id), g in df.groupby(
        [config.DATASET_COL, config.SITE_COL], sort=True
    ):
        g = g.sort_values(config.TIME_COL)
        times = pd.DatetimeIndex(g[config.TIME_COL])
        step = times.to_series().diff().median()
        grid = pd.date_range(times[0], times[-1], freq=step)
        g = g.set_index(times).reindex(grid)
        steps_per_day = int(round(pd.Timedelta(days=1) / step))
        out.append(
            SiteSeries(
                site_id=str(site_id),
                dataset=str(dataset),
                capacity=float(g[config.CAPACITY_COL].dropna().iloc[0]),
                timestamps=(grid.asi8 // 10**9).astype(np.int64),
                y=g[config.TARGET_COL].to_numpy(dtype=np.float32),
                cov=g[list(cov_cols)].to_numpy(dtype=np.float32) / scales,
                clearsky=np.nan_to_num(
                    g[config.CLEARSKY_COL].to_numpy(dtype=np.float32)
                ),
                steps_per_day=steps_per_day,
            )
        )
    return out


class WindowDataset:
    """Lazy sliding-window view over a list of SiteSeries."""

    def __init__(
        self,
        series: list[SiteSeries],
        history: int = config.HISTORY_STEPS,
        horizon: int = config.HORIZON_STEPS,
        stride: int = 1,
        min_future_valid: int = 1,
        min_hist_valid: int = 1,
        future_cov: str = "deterministic",  # or "all"
    ):
        if future_cov not in ("deterministic", "all"):
            raise ValueError(f"unknown future_cov mode: {future_cov!r}")
        self.series = series
        self.history = history
        self.horizon = horizon
        self.future_cov = future_cov
        self._index: list[tuple[int, int]] = []  # (series idx, window start)
        T, H = history, horizon
        for si, s in enumerate(series):
            valid = ~np.isnan(s.y)
            daylight = s.clearsky > 0
            for start in range(0, len(s.y) - T - H + 1, stride):
                fut = slice(start + T, start + T + H)
                if (valid[fut] & daylight[fut]).sum() < min_future_valid:
                    continue
                if valid[start : start + T].sum() < min_hist_valid:
                    continue
                self._index.append((si, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        si, start = self._index[i]
        s = self.series[si]
        T, H = self.history, self.horizon
        hist = slice(start, start + T)
        fut = slice(start + T, start + T + H)
        win = slice(start, start + T + H)

        y_hist = s.y[hist]
        y_future = s.y[fut]
        mask_hist = (~np.isnan(y_hist)).astype(np.float32)
        mask_future = (~np.isnan(y_future)).astype(np.float32)

        cov = np.nan_to_num(s.cov[win]).copy()
        if self.future_cov == "deterministic":
            observed = [
                c for c in range(cov.shape[1])
                if c not in config.DETERMINISTIC_COV_IDX
            ]
            cov[T:, observed] = 0.0

        # Same clock time on the previous day, for the seasonal-naive reference
        seas_idx = np.arange(start + T, start + T + H) - s.steps_per_day
        y_seasonal = np.where(seas_idx >= 0, s.y[np.maximum(seas_idx, 0)], np.nan)
        mask_seasonal = (~np.isnan(y_seasonal)).astype(np.float32)

        return {
            "y_hist": np.nan_to_num(y_hist),
            "mask_hist": mask_hist,
            "y_future": np.nan_to_num(y_future),
            "mask_future": mask_future,
            "daylight_future": (s.clearsky[fut] > 0).astype(np.float32),
            "cov": cov,
            "clearsky": s.clearsky[win],
            "y_seasonal": np.nan_to_num(y_seasonal),
            "mask_seasonal": mask_seasonal,
            "timestamps": s.timestamps[win],
            "site_id": s.site_id,
            "dataset": s.dataset,
            "capacity": s.capacity,
        }

    _STACK_KEYS = (
        "y_hist", "mask_hist", "y_future", "mask_future", "daylight_future",
        "cov", "clearsky", "y_seasonal", "mask_seasonal", "timestamps",
    )

    def batch(self, indices: list[int]) -> dict:
        """Stack windows into a batch dict; scalar fields become arrays."""
        items = [self[i] for i in indices]
        out = {k: np.stack([it[k] for it in items]) for k in self._STACK_KEYS}
        out["site_id"] = np.array([it["site_id"] for it in items])
        out["dataset"] = np.array([it["dataset"] for it in items])
        out["capacity"] = np.array([it["capacity"] for it in items], dtype=np.float32)
        return out

    def iter_batches(self, batch_size: int = 256) -> Iterator[dict]:
        for lo in range(0, len(self), batch_size):
            yield self.batch(list(range(lo, min(lo + batch_size, len(self)))))


def dataset_for_sites(
    df: pd.DataFrame, site_ids: set[str], **kwargs
) -> WindowDataset:
    """Build a WindowDataset restricted to a plant subset (one split part)."""
    sub = df[df[config.SITE_COL].astype(str).isin(site_ids)]
    return WindowDataset(build_site_series(sub), **kwargs)
