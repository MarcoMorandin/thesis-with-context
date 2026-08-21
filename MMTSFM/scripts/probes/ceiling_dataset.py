"""Assemble the G0 ceiling-probe design matrices from the V-JEPA latent cache.

Reads the cache and the parquet directly -- no datamodule, no model, no GPU. Cache
keys are `{dataset}_{site_id}_{origin}` (see PVRecordDataset._entity_cache_key),
so each cached window joins to a site's series on (dataset, site_id, origin).

The join target is `build_site_series` (baselines/common/windows.py), not the raw
parquet. `build_site_series` reindexes each site onto its own regular time grid
(uk_pv 30 min, goes_pvdaq 15 min); night gaps and other missing rows become NaN
rows *on that grid* rather than absent rows. `timestamps` produced by a cache
key's origin are grid points, so an exact-match lookup against the raw parquet
table silently misses every origin that falls on a gap -- on the real Leonardo
cache this was 50% of window origins, each one dropped with a bare `except
KeyError: continue` and no count anywhere. Reusing `build_site_series` instead of
re-deriving the join makes divergence between this probe and the model's own
data pipeline impossible, which matters because the probe's entire claim is that
it upper-bounds what the model could extract.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "baselines"))
from common import config  # noqa: E402
from common.windows import SiteSeries, build_site_series  # noqa: E402

# Power lags (and companion validity indicators) added to X_cov, as grid-index
# offsets relative to the window origin. 0 = now, -1..-12 = steps into the past
# on the site's own grid (uk_pv: 30 min, 1h, 1.5h, 3h, 6h ago for -1,-2,-3,-6,-12).
POWER_LAG_OFFSETS: tuple[int, ...] = (0, -1, -2, -3, -6, -12)

# Clear-sky irradiance below which the persistence ratio clearsky(t+h)/clearsky(t)
# is numerically meaningless. Same floor dataset.md uses to NaN `csi`.
CLEARSKY_FLOOR = 50.0


def parse_cache_key(name: str) -> tuple[str, str, int]:
    """`uk_pv_7239_1546300800` -> `("uk_pv", "7239", 1546300800)`.

    Split from the RIGHT: dataset names contain underscores, site ids do not.
    """
    dataset, site_id, origin = name.rsplit("_", 2)
    return dataset, site_id, int(origin)


def build_arrays(
    cache_dir: Path,
    parquet_path: Path,
    sites: set[str],
    horizon: int = 12,
    step_seconds: int = 1800,
    max_files: int | None = None,
) -> dict[str, np.ndarray]:
    """Design matrices for one split's plants.

    X_vis   [N, 4096]  V-JEPA latents, mean-pooled over the 196 spatial patches,
                       4 latent steps kept and flattened (feature variant F2).
    X_cov   [N, D]     `[6 power lags][6 lag-validity indicators]
                       [14 history COV_COLS at idx]
                       [horizon * len(DETERMINISTIC_COV_IDX) future deterministic
                       covs]` -- exactly what the model has access to (predictor
                       set (b) of knowledge/specs/2026-08-19-visual-fusion-diagnosis.md
                       §4.1: norm_power lags + history COV_COLS + future
                       DETERMINISTIC_COVS). Power lags and history/future
                       covariates are grid steps on the site's own cadence, taken
                       from the same `SiteSeries` the model consumes (COV_COLS are
                       already divided by COV_SCALES).
    Y       [N, H]     norm_power at grid index origin_idx + h, h = 1..H.
    Y_mask  [N, H]     False where that future step is absent (NaN) or beyond
                       the site's grid.
    site, origin       as cached (unconditionally written for every row).
    n_skipped: int     cache files whose origin has no exact position on its
                       site's grid (outside the grid range, or the site itself
                       has no series). Counted, never silently dropped -- this
                       is the counter that would have caught the original bug.

    `step_seconds` is accepted for API compatibility but is no longer used for
    indexing: horizon steps and power lags are grid-index offsets on each
    site's own native cadence (as returned by `build_site_series`), not a fixed
    physical-time offset.
    """
    cache_dir, parquet_path = Path(cache_dir), Path(parquet_path)

    files = sorted(cache_dir.glob("*.pt"))
    parsed = [(f, *parse_cache_key(f.stem)) for f in files]
    parsed = [p for p in parsed if p[2] in sites]
    parsed.sort(key=lambda p: (p[2], p[3]))
    if max_files is not None:
        parsed = parsed[:max_files]

    cols = sorted(
        {
            config.DATASET_COL,
            config.SITE_COL,
            config.TIME_COL,
            config.TARGET_COL,
            config.CAPACITY_COL,
            config.CLEARSKY_COL,
            *config.COV_COLS,
        }
    )
    df = pd.read_parquet(parquet_path, columns=cols)
    df[config.SITE_COL] = df[config.SITE_COL].astype(str)
    df = df[df[config.SITE_COL].isin(sites)]

    series_by_key: dict[tuple[str, str], SiteSeries] = {
        (s.dataset, s.site_id): s for s in build_site_series(df)
    }

    det_idx = list(config.DETERMINISTIC_COV_IDX)
    n_lags = len(POWER_LAG_OFFSETS)
    n_cov = len(config.COV_COLS)
    n = len(parsed)
    X_vis = np.zeros((n, 4 * 1024), dtype=np.float32)
    X_cov = np.zeros((n, 2 * n_lags + n_cov + horizon * len(det_idx)), dtype=np.float32)
    Y = np.zeros((n, horizon), dtype=np.float32)
    Y_mask = np.zeros((n, horizon), dtype=bool)
    site_out = np.empty(n, dtype=object)
    origin_out = np.zeros(n, dtype=np.int64)
    # Smart persistence: "the clouds do not change", i.e. today's power scaled by
    # the clear-sky ratio between origin and target. NaN where it is undefined.
    # It is NOT a reference here -- no reported skill is measured against it. It
    # is the RAMP LABEL: |Y - SP| is how far reality departed from an unchanged
    # sky, which is exactly the event vision is supposed to see coming. Plain
    # |Y(t+h) - Y(t)| would not do, because over a 6h horizon that quantity is
    # dominated by the diurnal cycle rather than by cloud.
    SP = np.full((n, horizon), np.nan, dtype=np.float32)
    Y_origin = np.full(n, np.nan, dtype=np.float32)

    n_skipped = 0

    for i, (path, ds, site, origin) in enumerate(parsed):
        z = torch.load(path, map_location="cpu", weights_only=True).float()
        X_vis[i] = z.mean(dim=1).reshape(-1).numpy()  # [4, 196, 1024] -> [4, 1024]
        site_out[i] = site
        origin_out[i] = origin

        s = series_by_key.get((ds, site))
        if s is None:
            n_skipped += 1
            continue

        idx = int(np.searchsorted(s.timestamps, origin))
        if idx >= len(s.timestamps) or s.timestamps[idx] != origin:
            n_skipped += 1
            continue

        # Power lags + validity indicators.
        for li, off in enumerate(POWER_LAG_OFFSETS):
            j = idx + off
            if j < 0 or np.isnan(s.y[j]):
                X_cov[i, li] = 0.0
                X_cov[i, n_lags + li] = 0.0
            else:
                X_cov[i, li] = float(s.y[j])
                X_cov[i, n_lags + li] = 1.0

        # History covariates at idx (already scaled by COV_SCALES). s.cov is NaN
        # at gap grid positions (the same gaps this fix's join now recovers), so
        # this must be nan_to_num'd -- matching WindowDataset (windows.py:150,
        # `np.nan_to_num(s.cov[win])`) -- or a gap origin poisons the ridge fit
        # with NaN instead of just contributing a zeroed, uninformative feature.
        cov_base = 2 * n_lags
        X_cov[i, cov_base : cov_base + n_cov] = np.nan_to_num(s.cov[idx])

        # Below this the clear-sky ratio is numerically meaningless and the
        # scaling explodes; dataset.md uses the same 50 W/m^2 floor to NaN `csi`.
        y0, cs0 = s.y[idx], s.clearsky[idx]
        Y_origin[i] = y0
        sp_ok = (not np.isnan(y0)) and float(cs0) >= CLEARSKY_FLOOR

        det_base = cov_base + n_cov
        for h in range(1, horizon + 1):
            j = idx + h
            if j >= len(s.timestamps):
                continue
            y = s.y[j]
            if np.isnan(y):
                continue
            Y[i, h - 1] = float(y)
            Y_mask[i, h - 1] = True
            if sp_ok and not np.isnan(s.clearsky[j]):
                SP[i, h - 1] = float(y0) * float(s.clearsky[j]) / float(cs0)
            off = det_base + (h - 1) * len(det_idx)
            # A target can be valid (norm_power present) while a covariate at
            # the same future step is NaN (e.g. an observed-weather column with
            # a gap independent of the power gap) -- nan_to_num for the same
            # reason as the history block above.
            X_cov[i, off : off + len(det_idx)] = np.nan_to_num(s.cov[j][det_idx])

    return {
        "X_vis": X_vis,
        "X_cov": X_cov,
        "Y": Y,
        "Y_mask": Y_mask,
        "site": site_out,
        "origin": origin_out,
        "SP": SP,
        "Y_origin": Y_origin,
        "n_skipped": n_skipped,
    }


# Rows are keyed by their origin's UTC time of day. Cache windows are sampled at
# stride 12 on a 30-minute grid, so on uk_pv there are exactly TWO origins per
# site per day -- 07:30 and 13:30 UTC -- and they are not interchangeable:
#
#   13:30  visual window (6 h backward) spans 07:30-13:30, daylight. Max
#          per-column std of the pooled latent across ~49 k train windows: 4.39.
#   07:30  visual window spans 01:30-07:30, dark for most of the UK year. Same
#          statistic: 0.0038 train, 0.00023 test -- constant to four orders of
#          magnitude, i.e. V-JEPA's embedding of a blank frame (non-zero, so a
#          zero-row check does NOT catch it).
#
# Measured 2026-08-20 by scripts/probes/diagnose_h6_cliff.py. The two populations
# also carry disjoint target coverage: only 07:30 origins keep valid targets past
# h=5 under the daylight mask, which is why the unfiltered G0 report went inert
# at h>=6. Pooling them dilutes any ceiling estimate by ~50% blank rows.
SECONDS_PER_DAY = 86400


def filter_by_origin_hour(
    arrays: dict, hours: Sequence[float], tol_seconds: int = 900
) -> dict:
    """Keep only rows whose origin falls at one of `hours` (UTC, fractional).

    Filters every per-row array together so the row alignment that X_vis, X_cov,
    Y, Y_mask, site and origin share is preserved -- these are parallel arrays,
    and filtering any subset of them independently would silently pair a
    window's features with another window's target.

    `n_skipped` is carried through unchanged: it counts cache files that never
    became rows, so it is not a per-row quantity and must not be filtered.
    """
    origin = arrays["origin"].astype(np.int64)
    sec = origin % SECONDS_PER_DAY
    keep = np.zeros(len(origin), dtype=bool)
    for h in hours:
        keep |= np.abs(sec - int(round(h * 3600))) <= tol_seconds
    out = {k: (v[keep] if k in _ROW_KEYS else v) for k, v in arrays.items()}
    out["n_kept"] = int(keep.sum())
    out["n_dropped_by_origin_hour"] = int((~keep).sum())
    return out


_ROW_KEYS = frozenset(
    {"X_vis", "X_cov", "Y", "Y_mask", "site", "origin", "SP", "Y_origin"}
)
