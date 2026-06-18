"""Import dumped predictions (Tier-4 RAG / Tier-5 vendored runs) into our metrics.

Each `<...>_<site>_pred.npz` carries `pred` (N,H[,1]) and `true` (N,H[,1]) — the
per-window forecasts the vendored harnesses dump (Time-VLM, VisionTS++/run_ukpv,
and the RAG originals once patched). This reduces them to the SAME result JSON
the in-repo baselines write (`PerPlantAccumulator` → overall + per-plant
NMAE/NRMSE/SS/CRPS), so `make_tables.py` / `summarize_ukpv.py` pick up the row.

    uv run python scripts/import_predictions.py --model time_vlm --tag s2_ukpv \
        --glob 'tier5/vendor/time_vlm/results/*/uk_pv_test_*_pred.npz'

Caveats (written into the result manifest):
- Daylight mask = `true > 0` (night norm_power is exactly 0), a proxy for the
  exact clear-sky daylight mask used by Tiers 0-4 — daytime near-zero overcast
  steps may be dropped. Pass `--data <parquet>` is reserved for an exact mask.
- These run on each harness's NATIVE eval windows, not bit-aligned with Tiers 0-4,
  so there is no per-window loss sidecar (DM/bootstrap vs Smart Persistence needs
  aligned windows). Compare via SS / rank, not pooled raw metrics (§4.4).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config                              # noqa: E402
from common.metrics import PerPlantAccumulator          # noqa: E402
from common.runner import add_skill_scores, write_results  # noqa: E402


def _2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a


def site_of(path: Path) -> str:
    return path.name[: -len("_pred.npz")].split("_")[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="result stem, e.g. time_vlm")
    ap.add_argument("--glob", required=True, help="glob for *_<site>_pred.npz")
    ap.add_argument("--tag", default="s2_ukpv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--csv_dir", default=None,
                    help="Directory containing original test CSVs to fit inverse scaler")
    ap.add_argument("--reference", default=None,
                    help="Smart Persistence result json for SS "
                         "(default: <out>/smart_persistence_<tag>.json)")
    args = ap.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no npz matched: {args.glob}")

    acc = PerPlantAccumulator()
    for f in files:
        data = np.load(f, allow_pickle=False)
        pred, true = _2d(data["pred"]), _2d(data["true"])
        if pred.shape != true.shape:
            raise SystemExit(f"{f.name}: pred {pred.shape} != true {true.shape}")
        
        site = site_of(f)
        if args.csv_dir and args.model == "time_vlm":
            import pandas as pd
            csv_path = Path(args.csv_dir) / f"uk_pv_test_{site}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                target_vals = df["OT"].values
                border2 = 12 * 30 * 24
                train_vals = target_vals[0:border2]
                mean = train_vals.mean()
                std = train_vals.std(ddof=0)
                if std > 1e-8:
                    pred = pred * std + mean
                    true = true * std + mean
            else:
                print(f"WARN: CSV file not found: {csv_path}. Skipping inverse scaling for site {site}.")

        mask = (true > 0).astype(np.float64)            # daylight proxy
        q = _2d(data["quantiles"]) if "quantiles" in data else None
        acc.update(plants=np.array([site] * len(pred)),
                   y_true=true, y_pred=np.clip(pred, 0.0, 1.0),
                   mask=mask, quantile_preds=q)

    results = {"overall": acc.macro(), "per_plant": acc.per_plant()}

    ref_path = Path(args.reference or
                    f"{args.out}/smart_persistence_{args.tag}.json")
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())["results"]
        results = add_skill_scores(results, ref)
    else:
        print(f"WARN: no Smart Persistence reference at {ref_path}; SS omitted")

    run_config = {
        "model": args.model, "tag": args.tag, "source": "vendored harness",
        "glob": args.glob, "n_plants": len(acc.per_plant()),
        "daylight_mask": "proxy true>0 (not exact clear-sky mask)",
        "eval_windows": "native harness split — not aligned with tiers 0-4; "
                        "no DM/bootstrap sidecar (compare via SS/rank, §4.4)",
        "quantile_levels": config.QUANTILE_LEVELS,
    }
    path = write_results(args.out, f"{args.model}_{args.tag}", results, run_config)
    o = results["overall"]
    print(f"{args.model}: plants={len(acc.per_plant())} "
          f"NMAE={o.get('nmae', float('nan')):.4f} "
          f"NRMSE={o.get('nrmse', float('nan')):.4f} "
          f"SS={o.get('skill_score', float('nan')):.4f} → {path}")


if __name__ == "__main__":
    main()
