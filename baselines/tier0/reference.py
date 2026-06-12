"""Tier 0 reference baselines (BASELINE_COMPARISON.md §1, Tier 0).

* Persistence          — repeat last valid observation (absolute floor)
* SmartPersistence     — persist the clear-sky index; the Skill-Score
                         denominator (§4.3). Exempt from the no-physics rule:
                         it *is* the physics reference.
* HourlyClimatology    — train-plant mean by (dataset, month, time-of-day);
                         detects leakage / trivial seasonality wins
* SeasonalNaive        — same clock time on the previous day (GIFT-Eval
                         standard reference)

All are zero-parameter (climatology fits a lookup table from train plants
only) and consume nothing beyond `Y` and the clear-sky covariate, per the
input-parity matrix (§3, T0 row).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import config
from common.base import Baseline, Forecast, register
from common.windows import WindowDataset


def _last_valid_idx(mask_hist: np.ndarray) -> np.ndarray:
    """Index of the last valid history step per row; -1 when none valid."""
    t = mask_hist.shape[1]
    return np.where(mask_hist > 0, np.arange(t), -1).max(axis=1)


def _last_valid(y_hist: np.ndarray, mask_hist: np.ndarray) -> np.ndarray:
    """Last valid value per row of (N, T); 0.0 when no history step is valid."""
    n = y_hist.shape[0]
    idx = _last_valid_idx(mask_hist)
    last = y_hist[np.arange(n), np.maximum(idx, 0)]
    return np.where(idx >= 0, last, 0.0).astype(np.float32)


@register
class Persistence(Baseline):
    """ŷ(t+h) = y(t) for all h."""

    name = "persistence"
    tier = 0

    def predict(self, batch: dict) -> Forecast:
        last = _last_valid(batch["y_hist"], batch["mask_hist"])
        horizon = batch["y_future"].shape[1]
        return Forecast(point=np.repeat(last[:, None], horizon, axis=1))


@register
class SmartPersistence(Baseline):
    """Persist the clear-sky index k = y(t) / y_clearsky(t).

    ŷ(t+h) = k · y_clearsky(t+h), with y_clearsky = clearsky_ghi / 1000
    (capacity-normalized clear-sky power proxy, the same convention as the
    `csi` column in the curated dataset). Falls back to plain persistence
    when the clear-sky reference at t is below 50 W/m² (dawn/dusk/night,
    where the index is numerically undefined).
    """

    name = "smart_persistence"
    tier = 0

    def predict(self, batch: dict) -> Forecast:
        T = batch["y_hist"].shape[1]
        n = batch["y_hist"].shape[0]
        last = _last_valid(batch["y_hist"], batch["mask_hist"])
        cs = batch["clearsky"] / config.STC_IRRADIANCE   # (N, T+H) norm proxy
        cs_future = cs[:, T:]

        # clearness index is anchored at the *last valid* history step, so
        # the (y, clear-sky) pair is taken at the same timestamp even when
        # trailing history steps are masked out
        anchor = np.maximum(_last_valid_idx(batch["mask_hist"]), 0)
        cs_anchor_raw = batch["clearsky"][np.arange(n), anchor]
        cs_anchor = cs[np.arange(n), anchor]

        defined = cs_anchor_raw >= config.SP_MIN_CLEARSKY
        k = np.divide(
            last, cs_anchor,
            out=np.zeros_like(last), where=cs_anchor > 0,
        )
        smart = np.clip(k[:, None] * cs_future, 0.0, 1.0)
        fallback = np.repeat(last[:, None], cs_future.shape[1], axis=1)
        point = np.where(defined[:, None], smart, fallback)
        # Night future steps have zero clear-sky power by construction
        point = np.where(cs_future > 0, point, 0.0)
        return Forecast(point=point.astype(np.float32))


@register
class HourlyClimatology(Baseline):
    """Train-plant mean norm_power per (dataset, month, time-of-day).

    Keyed at native sub-hourly resolution (hour, minute): pooling :00 and
    :30 steps of a 30-min grid into one hourly bucket would blur the diurnal
    ramp and weaken the reference.
    """

    name = "climatology_hourly"
    tier = 0
    requires_fit = True

    def __init__(self):
        self._table: dict[tuple[str, int, int, int], float] = {}
        self._global: float = 0.0

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        keys, values = [], []
        for s in train.series:
            valid = ~np.isnan(s.y)
            times = pd.to_datetime(s.timestamps[valid], unit="s", utc=True)
            keys.append(
                pd.DataFrame(
                    {"dataset": s.dataset, "month": times.month,
                     "hour": times.hour, "minute": times.minute}
                )
            )
            values.append(s.y[valid])
        frame = pd.concat(keys, ignore_index=True)
        frame["y"] = np.concatenate(values)
        self._global = float(frame["y"].mean())
        grouped = frame.groupby(["dataset", "month", "hour", "minute"])["y"].mean()
        self._table = {k: float(v) for k, v in grouped.items()}

    def predict(self, batch: dict) -> Forecast:
        T = batch["y_hist"].shape[1]
        times = pd.to_datetime(batch["timestamps"][:, T:].ravel(), unit="s", utc=True)
        datasets = np.repeat(batch["dataset"], batch["y_future"].shape[1])
        point = np.array(
            [
                self._table.get((d, mo, h, mi), self._global)
                for d, mo, h, mi in zip(
                    datasets, times.month, times.hour, times.minute
                )
            ],
            dtype=np.float32,
        ).reshape(batch["y_future"].shape)
        return Forecast(point=point)


@register
class SeasonalNaive(Baseline):
    """ŷ(t+h) = y(t+h − 1 day); persistence fallback where yesterday is missing."""

    name = "seasonal_naive"
    tier = 0

    def predict(self, batch: dict) -> Forecast:
        last = _last_valid(batch["y_hist"], batch["mask_hist"])
        horizon = batch["y_future"].shape[1]
        fallback = np.repeat(last[:, None], horizon, axis=1)
        point = np.where(batch["mask_seasonal"] > 0, batch["y_seasonal"], fallback)
        return Forecast(point=point.astype(np.float32))
