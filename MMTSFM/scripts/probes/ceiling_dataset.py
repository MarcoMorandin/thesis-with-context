"""Assemble the G0 ceiling-probe design matrices from the V-JEPA latent cache.

Reads the cache and the parquet directly — no datamodule, no model, no GPU. Cache
keys are `{dataset}_{site_id}_{origin}` (see PVRecordDataset._entity_cache_key),
so each cached window joins to the table on (dataset, site_id, timestamp==origin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "baselines"))
from common import config  # noqa: E402

# ASSUMED, not verified against the real cache (Task 1 Step 1 requires
# Leonardo access and was skipped). Origins are assumed to be whole seconds
# since the epoch. If this is wrong, the (dataset, site, epoch) join below
# will silently fail for nearly every row -- watch for a near-zero count of
# successful history/future lookups as the symptom of a wrong unit here.
EPOCH_UNIT = "s"
_DIVISOR = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[EPOCH_UNIT]


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
    X_cov   [N, D]     history covariates at the origin + future deterministic
                       covariates at each horizon step — exactly what the model
                       has access to.
    Y       [N, H]     norm_power at origin + h*step_seconds.
    Y_mask  [N, H]     False where that future step is absent from the table.
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
            *config.COV_COLS,
        }
    )
    df = pd.read_parquet(parquet_path, columns=cols)
    df[config.SITE_COL] = df[config.SITE_COL].astype(str)
    df["_epoch"] = (
        pd.to_datetime(df[config.TIME_COL], utc=True).astype("int64") // _DIVISOR
    )
    table = df.set_index([config.DATASET_COL, config.SITE_COL, "_epoch"]).sort_index()

    det_idx = list(config.DETERMINISTIC_COV_IDX)
    n = len(parsed)
    X_vis = np.zeros((n, 4 * 1024), dtype=np.float32)
    X_cov = np.zeros((n, len(config.COV_COLS) + horizon * len(det_idx)), np.float32)
    Y = np.zeros((n, horizon), dtype=np.float32)
    Y_mask = np.zeros((n, horizon), dtype=bool)
    site_out = np.empty(n, dtype=object)
    origin_out = np.zeros(n, dtype=np.int64)

    for i, (path, ds, site, origin) in enumerate(parsed):
        z = torch.load(path, map_location="cpu", weights_only=True).float()
        X_vis[i] = z.mean(dim=1).reshape(-1).numpy()  # [4, 196, 1024] -> [4, 1024]
        site_out[i] = site
        origin_out[i] = origin

        try:
            hist = table.loc[(ds, site, origin)]
        except KeyError:
            continue
        if isinstance(hist, pd.DataFrame):
            hist = hist.iloc[0]
        X_cov[i, : len(config.COV_COLS)] = [float(hist[c]) for c in config.COV_COLS]

        base = len(config.COV_COLS)
        for h in range(1, horizon + 1):
            ts = origin + h * step_seconds
            try:
                fut = table.loc[(ds, site, ts)]
            except KeyError:
                continue
            if isinstance(fut, pd.DataFrame):
                fut = fut.iloc[0]
            y = fut[config.TARGET_COL]
            if pd.isna(y):
                continue
            Y[i, h - 1] = float(y)
            Y_mask[i, h - 1] = True
            off = base + (h - 1) * len(det_idx)
            X_cov[i, off : off + len(det_idx)] = [
                float(fut[config.COV_COLS[j]]) for j in det_idx
            ]

    return {
        "X_vis": X_vis,
        "X_cov": X_cov,
        "Y": Y,
        "Y_mask": Y_mask,
        "site": site_out,
        "origin": origin_out,
    }
