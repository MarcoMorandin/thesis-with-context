"""Baseline-contract checks for the vendored TS-RAG / Cross-RAG path.

Two phases, both offline and dependency-light (pandas/numpy only) so they can
gate a cluster run without loading the heavy upstream stack:

* ``--inputs DIR``      pre-flight: the exported uk_pv CSVs (export_ukpv.py) are
                        well-formed for the upstream ``Dataset_Custom_retrieve``
                        loader — a ``date`` column, an ``OT`` target, a dense
                        monotone 30-min grid, finite values in [0, 1].
* ``--predictions NPZ`` post-run: the model output obeys the same contract the
                        in-repo baselines are held to (test_baseline_contract.py):
                        ``point`` shape (N, H) [reshapeable to (N, H, 1)], float,
                        finite, within [0, 1]; optional quantiles monotone in level.

Exit code 0 = contract satisfied, non-zero = violation (so SLURM can `set -e`).

    uv run python tier4/vendor/contract_check.py --inputs /tmp/ukpv_rag
    python tier4/vendor/contract_check.py --predictions preds_3432.npz --horizon 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def check_inputs(csv_dir: Path) -> list[str]:
    import pandas as pd

    errs: list[str] = []
    manifest = csv_dir / "manifest.json"
    if not manifest.is_file():
        errs.append(f"missing manifest.json in {csv_dir}")
    csvs = sorted(csv_dir.glob("uk_pv_*.csv"))
    if not csvs:
        errs.append(f"no uk_pv_*.csv found in {csv_dir}")
    for path in csvs:
        df = pd.read_csv(path)
        if df.columns[0] != "date":
            errs.append(f"{path.name}: first column must be 'date'")
        if "OT" not in df.columns:
            errs.append(f"{path.name}: missing 'OT' target column")
        if df.drop(columns=["date"]).isna().any().any():
            errs.append(f"{path.name}: NaNs present (loader StandardScale will break)")
        ts = pd.to_datetime(df["date"])
        if not ts.is_monotonic_increasing:
            errs.append(f"{path.name}: 'date' not monotonically increasing")
        steps = ts.diff().dropna().dt.total_seconds().unique()
        if len(steps) != 1 or steps[0] != 1800.0:
            errs.append(f"{path.name}: non-uniform / non-30min grid: {steps[:3]}")
        vals = df.drop(columns=["date"]).to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            errs.append(f"{path.name}: non-finite values")
        if vals.min() < -1e-6 or vals.max() > 1.0 + 1e-6:
            errs.append(f"{path.name}: values outside [0,1] (expected norm_power)")
    return errs


def check_predictions(npz: Path, horizon: int) -> list[str]:
    errs: list[str] = []
    data = np.load(npz, allow_pickle=False)
    if "pred" not in data:
        return [f"{npz.name}: no 'pred' array (apply the §6 dump patch)"]
    point = np.asarray(data["pred"], dtype=np.float64)
    if point.ndim == 3 and point.shape[-1] == 1:
        point = point[..., 0]
    if point.ndim != 2:
        errs.append(f"pred must be (N,H) or (N,H,1); got {point.shape}")
    elif point.shape[1] != horizon:
        errs.append(f"pred horizon {point.shape[1]} != expected H={horizon}")
    if "true" in data:
        true = np.asarray(data["true"], dtype=np.float64)
        if true.shape[: point.ndim] != point.shape:
            errs.append(f"pred {point.shape} vs true {true.shape} shape mismatch")
    if not np.isfinite(point).all():
        errs.append("pred has non-finite values")
    if point.size and (point.min() < -1e-6 or point.max() > 1.0 + 1e-6):
        errs.append(f"pred outside [0,1]: [{point.min():.4f}, {point.max():.4f}]")
    if "quantiles" in data:
        q = np.asarray(data["quantiles"], dtype=np.float64)
        if (np.diff(q, axis=-1) < -1e-6).any():
            errs.append("quantiles not monotonically non-decreasing in level")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, help="exported uk_pv CSV dir")
    ap.add_argument("--predictions", type=Path, help="predictions .npz")
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()
    if not args.inputs and not args.predictions:
        ap.error("pass --inputs and/or --predictions")

    errs: list[str] = []
    if args.inputs:
        errs += [f"[inputs] {e}" for e in check_inputs(args.inputs)]
    if args.predictions:
        errs += [f"[predictions] {e}" for e in check_predictions(
            args.predictions, args.horizon)]

    if errs:
        print("CONTRACT VIOLATIONS:")
        for e in errs:
            print(f"  ✗ {e}")
        return 1
    print("✓ baseline contract satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
