"""Why the G0 report goes inert at h >= 6.

The run of 2026-08-19 selected the SMALLEST visual penalty in the grid (0.01,
i.e. essentially none) for every horizon h >= 6, and yet set (c) reproduced set
(b) to 1e-10 and set (a) scored exactly zero skill against the train-mean
reference. Set (a) landing on EXACTLY zero skill means its fit predicted the
train mean, i.e. its coefficients came out zero, i.e. the centered visual block
was orthogonal to the centered target -- and set (c) agreeing with set (b) to
1e-10 tightens that to the visual columns being CONSTANT on the fitted rows.
A constant block makes every penalty tie on validation NMAE, and argmin then
returns index 0, which is precisely the 0.01 in the report.

So h >= 6 is not a measurement of vision at long horizons that came back null.
It is not a measurement at all, and the gate must be read on h = 1..5 only.

What is NOT yet established is why. `build_arrays` writes X_vis[i] from a real
`torch.load` unconditionally, before every skip path, so "the loader zero-filled
those rows" is not supported by the code and has to be measured rather than
assumed. It still matters well past this probe: whatever makes a large block of
latents degenerate would have fed s2a and s2b training too.

Two modes, because they cost three orders of magnitude apart:

  --scan-only N   Samples N cache files at RANDOM and reports whether the
                  latents are degenerate at all -- all-zero rows, per-file
                  spread, and the across-file spread that would collapse if the
                  cache were writing one latent many times. Needs no parquet and
                  no join, so it runs in minutes and answers the urgent
                  question: is the cache itself sound?

  (default)       Full cross-tab through `build_arrays`: per-horizon feature
                  variance on exactly the rows each probe fits, target variance
                  on the same rows, and the time-of-day split of the two
                  populations -- n_test_valid halves at exactly h = 6 and then
                  stays flat, which is a binary property of the window rather
                  than a horizon that decays. Reads every cache file for the
                  split, so budget what G0 itself needed (~4 h), not minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "baselines"))


def _pct(n: int, d: int) -> str:
    return f"{n:7d} ({100.0 * n / d:5.1f}%)" if d else f"{n:7d} (  n/a)"


def scan_cache(cache_dir: Path, n_sample: int, seed: int = 0) -> None:
    """Sample cache files at random and report whether the latents are degenerate.

    Random, not the first N: `build_arrays` sorts by (site, origin) and slices,
    so a head-of-list sample would cover a handful of plants and could easily
    miss a defect that is concentrated elsewhere.
    """
    import torch

    files = sorted(cache_dir.glob("*.pt"))
    rng = np.random.default_rng(seed)
    take = rng.choice(len(files), size=min(n_sample, len(files)), replace=False)
    print(f"\n===== cache scan: {len(take)} of {len(files)} files =====")

    pooled = np.zeros((len(take), 4 * 1024), dtype=np.float32)
    n_zero = 0
    for i, j in enumerate(take):
        z = torch.load(files[j], map_location="cpu", weights_only=True).float()
        # Same pooling build_arrays applies: [4, 196, 1024] -> [4, 1024] -> flat.
        pooled[i] = z.mean(dim=1).reshape(-1).numpy()
        if float(z.abs().max()) == 0.0:
            n_zero += 1

    print(f"all-zero raw latents         : {_pct(n_zero, len(take))}")
    print(
        f"all-zero after pooling       : {_pct(int((np.abs(pooled).max(axis=1) == 0).sum()), len(take))}"
    )
    # Across-file spread is the decisive one. If the cache wrote one latent many
    # times over, per-file spread stays healthy while this collapses.
    across = pooled.std(axis=0)
    within = pooled.std(axis=1)
    print(
        f"across-file std per dim      : min={across.min():.4g} med={np.median(across):.4g} max={across.max():.4g}"
    )
    print(
        f"within-file std per window   : min={within.min():.4g} med={np.median(within):.4g} max={within.max():.4g}"
    )
    print(
        f"distinct pooled latents      : {_pct(len(np.unique(pooled, axis=0)), len(take))}"
    )


def diagnose(arrays: dict, horizon: int, label: str) -> None:
    X, Y, mask = arrays["X_vis"], arrays["Y"], arrays["Y_mask"]
    origin = arrays["origin"]
    n = X.shape[0]
    print(f"\n===== {label}: {n} windows, X_vis {X.shape} =====")

    row_absmax = np.abs(X).max(axis=1)
    zero_row = row_absmax == 0.0
    print(f"all-zero visual latents      : {_pct(int(zero_row.sum()), n)}")

    # The probes fit on rows where the TARGET at h is present, so that subset's
    # feature variance is what decides whether the fit can move at all. A
    # max-over-columns std at ~0 means the block is constant there: centering
    # wipes it out, every penalty gives an identical fit, and h is unmeasured.
    print("\n  h  n_valid   all-zero X_vis      max std X_vis        std Y")
    for h in range(horizon):
        sel = mask[:, h]
        k = int(sel.sum())
        if k == 0:
            print(f" {h + 1:2d}  {k:7d}   {'-':>14}   {'-':>16}   {'-':>10}")
            continue
        std_x = float(X[sel].std(axis=0).max())
        std_y = float(Y[sel, h].std())
        print(
            f" {h + 1:2d}  {k:7d}   {_pct(int(zero_row[sel].sum()), k)}   "
            f"{std_x:16.6g}   {std_y:10.6g}"
        )

    late = mask[:, min(horizon - 1, 5)]
    hod = (origin.astype(np.int64) % 86400) / 3600.0
    for name, m in (("survives h=6", late), ("dropped at h=6", ~late)):
        if m.sum() == 0:
            continue
        print(
            f"\n  {name:<15} n={int(m.sum()):6d}  "
            f"hour-of-day UTC min={hod[m].min():5.2f} "
            f"med={np.median(hod[m]):5.2f} max={hod[m].max():5.2f}  "
            f"max std X_vis={float(X[m].std(axis=0).max()):.6g}  "
            f"all-zero {_pct(int(zero_row[m].sum()), int(m.sum()))}"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--parquet", default=None, help="required unless --scan-only")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument(
        "--scan-only",
        type=int,
        default=None,
        metavar="N",
        help="sample N cache files and stop; needs no parquet, runs in minutes",
    )
    a = ap.parse_args()

    if a.scan_only:
        scan_cache(Path(a.cache_dir), a.scan_only)
        sys.exit(0)

    if not a.parquet:
        ap.error("--parquet is required unless --scan-only is given")

    from common.splits import load_splits

    from scripts.probes.ceiling_dataset import build_arrays

    splits = load_splits()
    for split in ("train", "test"):
        sites = {str(s) for s in splits["uk_pv"][split]}
        arrays = build_arrays(
            Path(a.cache_dir), Path(a.parquet), sites, a.horizon, max_files=a.max_files
        )
        diagnose(arrays, a.horizon, f"uk_pv {split}")
