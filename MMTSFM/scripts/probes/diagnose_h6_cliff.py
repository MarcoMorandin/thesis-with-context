"""Why the G0 report goes inert at h >= 6.

The run of 2026-08-19 selected the SMALLEST visual penalty in the grid (0.01,
i.e. essentially none) for every horizon h >= 6, and yet set (c) reproduced set
(b) to 1e-10 and set (a) scored exactly zero skill. Those three facts together
have one explanation: on the training rows that survive the h >= 6 target mask,
the visual design matrix has no variance, so its coefficients cannot move any
prediction and every penalty in the grid ties (argmin then returns index 0,
which is the 0.01 we see).

That is a property of the DATA, not of the probe, and it matters well beyond
this report: if a large block of the V-JEPA cache is zero-filled, the same rows
fed s2a and s2b training.

Answers three questions, cheapest first:
  1. how many windows carry an all-zero visual latent at all;
  2. whether those windows are the same ones that survive the h >= 6 mask;
  3. what separates the two populations in time-of-day, which is what makes
     n_test_valid halve at exactly h = 6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "baselines"))


def _pct(n: int, d: int) -> str:
    return f"{n:7d} ({100.0 * n / d:5.1f}%)" if d else f"{n:7d} (  n/a)"


def diagnose(arrays: dict, horizon: int, label: str) -> None:
    X, mask, origin = arrays["X_vis"], arrays["Y_mask"], arrays["origin"]
    n = X.shape[0]
    print(f"\n===== {label}: {n} windows, X_vis {X.shape} =====")

    # Q1 --------------------------------------------------------------------
    # Row-wise, not global: a zero-filled latent is zero in every dimension.
    row_absmax = np.abs(X).max(axis=1)
    zero_row = row_absmax == 0.0
    print(f"all-zero visual latents      : {_pct(int(zero_row.sum()), n)}")
    print(
        f"row |max| percentiles        : "
        f"p1={np.percentile(row_absmax, 1):.4g} "
        f"p50={np.percentile(row_absmax, 50):.4g} "
        f"p99={np.percentile(row_absmax, 99):.4g}"
    )

    # Q2 --------------------------------------------------------------------
    # The probe fits on rows where the TARGET at h is present, so that is the
    # subset whose feature variance decides whether the fit can move at all.
    print("\n  h  n_train_valid   all-zero X_vis   max per-column std of X_vis")
    for h in range(horizon):
        sel = mask[:, h]
        k = int(sel.sum())
        if k == 0:
            print(f" {h + 1:2d}  {k:13d}   {'-':>14}   {'-':>27}")
            continue
        sub = X[sel]
        # If this is ~0 the block is constant on these rows: centering wipes it
        # out, every penalty gives the same fit, and the horizon is unmeasured.
        std_max = float(sub.std(axis=0).max())
        print(
            f" {h + 1:2d}  {k:13d}   {_pct(int(zero_row[sel].sum()), k)}   {std_max:27.6g}"
        )

    # Q3 --------------------------------------------------------------------
    # n_test_valid halves at h = 6 and then stays flat, which is a binary
    # property of the window rather than a horizon that decays -- time-of-day
    # is the obvious candidate.
    late = mask[:, min(horizon - 1, 5)]
    hod = (origin.astype(np.int64) % 86400) / 3600.0
    for name, m in (("survives h=6", late), ("dropped at h=6", ~late)):
        if m.sum() == 0:
            continue
        print(
            f"\n  {name:<15} n={int(m.sum()):6d}  "
            f"hour-of-day UTC min={hod[m].min():5.2f} "
            f"med={np.median(hod[m]):5.2f} max={hod[m].max():5.2f}  "
            f"all-zero X_vis {_pct(int(zero_row[m].sum()), int(m.sum()))}"
        )


if __name__ == "__main__":
    import argparse

    from common.splits import load_splits

    from scripts.probes.ceiling_dataset import build_arrays

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--max-files", type=int, default=None)
    a = ap.parse_args()

    splits = load_splits()
    for split in ("train", "test"):
        sites = {str(s) for s in splits["uk_pv"][split]}
        arrays = build_arrays(
            Path(a.cache_dir), Path(a.parquet), sites, a.horizon, max_files=a.max_files
        )
        diagnose(arrays, a.horizon, f"uk_pv {split}")
