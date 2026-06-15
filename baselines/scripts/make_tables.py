"""Render results/ into the §7 markdown tables.

* Table 7.1 headline: NMAE / NRMSE / SS / CRPS (+ ramp columns from S6)
* §4.4 aggregation block: win rate vs smart persistence, geometric-mean
  skill, average rank across plants.

    uv run python scripts/make_tables.py --results results --out results/tables.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.aggregate import average_rank, geometric_mean_skill, win_rate  # noqa: E402

TIER_ORDER = [
    ("T0", ["persistence", "smart_persistence", "climatology_hourly",
            "seasonal_naive"]),
    ("T1", ["lightgbm", "tabpfn"]),
    ("T2", ["mlp", "dlinear", "patchtst", "itransformer", "tft"]),
    ("T3", ["chronos2_zs", "chronos2_ft", "timesfm_zs", "tirex_zs",
            "ttm_zs", "ttm_ft"]),
    # cora runs via run_eval; the *_orig/*_proto rows are imported from the
    # vendored original TS-RAG / Cross-RAG cluster runs (TIER4_RAG_INTEGRATION.md)
    ("T4", ["cora", "ts_rag_orig", "ts_rag_proto",
            "cross_rag_orig", "cross_rag_proto"]),
    # Tier 5: vendored original multimodal-TS baselines imported by file stem
    # (TIER5_INTEGRATION.md). time_vlm/visionts_pp = numerical track (runnable);
    # unicast/aurora = multimodal track (blocked on image+text data).
    ("T5", ["time_vlm", "visionts_pp", "unicast", "aurora"]),
]


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results)
    runs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.stem.startswith("significance_"):
            continue
        payload = json.loads(path.read_text())
        if "results" in payload and "overall" in payload.get("results", {}):
            runs[path.stem] = payload["results"]

    lines = ["# Baseline results", "",
             "## Headline (cross-plant, §7.1)", "",
             "| Tier | Model | NMAE ↓ | NRMSE ↓ | SS ↑ | CRPS ↓ "
             "| Ramp NMAE ↓ | Ramp NRMSE ↓ |",
             "|---|---|---|---|---|---|---|---|"]
    nrmse_by_model: dict[str, dict[str, float]] = {}
    for tier, names in TIER_ORDER:
        for name in names:
            if name not in runs:
                continue
            o = runs[name]["overall"]
            lines.append(
                f"| {tier} | {name} | {fmt(o.get('nmae'))} "
                f"| {fmt(o.get('nrmse'))} | {fmt(o.get('skill_score'), 3)} "
                f"| {fmt(o.get('crps'))} | {fmt(o.get('nmae_ramp'))} "
                f"| {fmt(o.get('nrmse_ramp'))} |"
            )
            nrmse_by_model[name] = {
                plant: row["nrmse"]
                for plant, row in runs[name]["per_plant"].items()
            }

    reference = nrmse_by_model.get("smart_persistence")
    if reference:
        lines += ["", "## Aggregation (§4.4, per-plant units)", "",
                  "| Model | Win rate vs SP ↑ | SS_geo ↑ | Avg rank ↓ |",
                  "|---|---|---|---|"]
        ranks = average_rank(nrmse_by_model)
        for name in nrmse_by_model:
            lines.append(
                f"| {name} | {fmt(win_rate(nrmse_by_model[name], reference), 3)} "
                f"| {fmt(geometric_mean_skill(nrmse_by_model[name], reference), 3)} "
                f"| {fmt(ranks[name], 2)} |"
            )

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
