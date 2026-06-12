"""Tabular feature construction for tier-1 models (flattened Y, X_cov).

One row per (window, horizon step) — the direct multi-horizon formulation
with the horizon index as a feature, so a single model covers all h. The
same builder feeds LightGBM and TabPFN to keep the tier internally fair.
"""

from __future__ import annotations

import numpy as np

from common import config

FEATURE_NAMES: list[str] = (
    [f"y_lag_{i}" for i in range(config.HISTORY_STEPS)]
    + ["y_mean", "y_std", "y_max", "y_last"]
    + [f"cov_hist_mean_{c}" for c in config.COV_COLS]
    + [f"cov_future_{c}" for c in config.DETERMINISTIC_COVS]
    + ["horizon_frac"]
)


def build_features(batch: dict) -> np.ndarray:
    """(N, H, F) feature tensor for a window batch."""
    y, m = batch["y_hist"], batch["mask_hist"]
    n, t = y.shape
    h = batch["y_future"].shape[1]

    counts = np.maximum(m.sum(axis=1, keepdims=True), 1.0)
    y_mean = (y * m).sum(axis=1, keepdims=True) / counts
    y_var = ((y - y_mean) ** 2 * m).sum(axis=1, keepdims=True) / counts
    y_stats = np.concatenate(
        [
            y_mean,
            np.sqrt(y_var),
            (y * m).max(axis=1, keepdims=True),
            _last_valid(y, m)[:, None],
        ],
        axis=1,
    )  # (N, 4)

    cov_hist_mean = batch["cov"][:, :t, :].mean(axis=1)          # (N, C)
    cov_future = batch["cov"][:, t:, list(config.DETERMINISTIC_COV_IDX)]  # (N, H, D)

    static = np.concatenate([y, y_stats, cov_hist_mean], axis=1)  # (N, S)
    static = np.repeat(static[:, None, :], h, axis=1)             # (N, H, S)
    horizon = np.broadcast_to(
        (np.arange(1, h + 1, dtype=np.float32) / h)[None, :, None], (n, h, 1)
    )
    return np.concatenate([static, cov_future, horizon], axis=2).astype(np.float32)


def _last_valid(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, t = y.shape
    idx = np.where(mask > 0, np.arange(t), -1).max(axis=1)
    last = y[np.arange(n), np.maximum(idx, 0)]
    return np.where(idx >= 0, last, 0.0).astype(np.float32)


def training_table(
    dataset, max_rows: int | None = None, seed: int = config.SEED, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten all windows into (X, y) keeping only valid daylight steps."""
    xs, ys = [], []
    for batch in dataset.iter_batches(batch_size):
        feats = build_features(batch)                      # (N, H, F)
        valid = (batch["mask_future"] * batch["daylight_future"]) > 0
        xs.append(feats[valid])
        ys.append(batch["y_future"][valid])
    x = np.concatenate(xs) if xs else np.empty((0, len(FEATURE_NAMES)), np.float32)
    y = np.concatenate(ys) if ys else np.empty((0,), np.float32)
    if max_rows is not None and len(x) > max_rows:
        keep = np.random.default_rng(seed).choice(len(x), max_rows, replace=False)
        x, y = x[keep], y[keep]
    return x, y
