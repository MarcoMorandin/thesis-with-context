"""DM test + paired block bootstrap + Holm–Bonferroni over saved runs (§4.5).

Consumes the `<model>_losses.npz` sidecars written by run_eval.py
(aligned per-window masked-MAE losses). Compares one candidate model
against every other model found in the results directory, applies
Holm–Bonferroni across the family, and writes `significance_<model>.json`.

    uv run python scripts/significance.py --results results --model patchtst
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.stats import block_bootstrap_ci, dm_test, holm_bonferroni  # noqa: E402


def load_losses(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in ("loss", "plant", "day")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--model", required=True,
                        help="candidate model (losses npz stem)")
    parser.add_argument("--horizon", type=int, default=12,
                        help="forecast horizon h for the DM variance correction")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()

    results_dir = Path(args.results)
    candidate_path = results_dir / f"{args.model}_losses.npz"
    if not candidate_path.exists():
        raise SystemExit(f"missing {candidate_path} — rerun run_eval.py")
    cand = load_losses(candidate_path)

    comparisons: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for path in sorted(results_dir.glob("*_losses.npz")):
        other = path.stem.removesuffix("_losses")
        if other == args.model:
            continue
        base = load_losses(path)
        if base["loss"].shape != cand["loss"].shape or not np.array_equal(
            base["day"], cand["day"]
        ):
            print(f"skip {other}: eval windows not aligned with {args.model}")
            continue
        dm = dm_test(cand["loss"], base["loss"], h=args.horizon)
        boot = block_bootstrap_ci(
            cand["loss"], base["loss"], blocks=cand["day"],
            n_resamples=args.resamples,
        )
        comparisons[other] = {"dm": dm, "bootstrap": boot}
        p_values[other] = dm["p_value"]

    if not comparisons:
        raise SystemExit("no aligned baselines found to compare against")

    holm = holm_bonferroni(p_values, alpha=args.alpha)
    for other, h in holm.items():
        comparisons[other]["holm"] = h
        comparisons[other]["bold_ok"] = bool(
            h["reject"] and comparisons[other]["bootstrap"]["significant"]
        )

    out = results_dir / f"significance_{args.model}.json"
    out.write_text(json.dumps(comparisons, indent=2) + "\n")
    print(f"wrote {out}")
    for other, c in sorted(comparisons.items()):
        verdict = "SIGNIFICANT" if c["bold_ok"] else "n.s."
        print(f"  vs {other:24s} ΔNMAE={c['dm']['mean_diff']:+.5f} "
              f"p_adj={c['holm']['p_adjusted']:.4f} {verdict}")


if __name__ == "__main__":
    main()
