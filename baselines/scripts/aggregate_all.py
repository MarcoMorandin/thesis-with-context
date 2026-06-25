"""Aggregate every results/*.json into ONE results file across all tiers.

Tolerant to the tag zoo the suite produces (``_s2``, ``_s2_ukpv``,
``_s2_ukpv_mm``, ``_s2_<model>``, ``_lopo_goes_pvdaq``, ``_seedNN``, ``_agg``):
each result file's stem is mapped back to its base model name and tier, so the
GPU-node orchestrator can drop one comprehensive table after the full sweep.

    uv run python scripts/aggregate_all.py --results results \
        --md results/ALL_RESULTS.md --json results/ALL_RESULTS.json

Headline row metrics (NMAE/NRMSE/SS/CRPS/ramp) come straight from each run's
``results.overall``; the §4.4 block (win rate / geo-mean skill / avg rank vs
Smart Persistence) is computed per tag-group when that group has a
``smart_persistence`` reference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.aggregate import average_rank, geometric_mean_skill, win_rate  # noqa: E402

# Base model name -> tier label. Longest-prefix match maps a tagged result
# filename (e.g. "chronos2_ft_s2_chronos2_ft_seed42") back to its base model.
MODEL_TIER: dict[str, str] = {
    "persistence": "T0", "smart_persistence": "T0",
    "climatology_hourly": "T0", "seasonal_naive": "T0",
    "lightgbm": "T1", "tabpfn": "T1",
    "mlp": "T2", "dlinear": "T2", "patchtst": "T2",
    "itransformer": "T2", "tft": "T2",
    "chronos2_zs": "T3", "chronos2_ft": "T3", "chronos2_oracle": "T3", "chronos2_oracle_ft": "T3", "timesfm_zs": "T3",
    "tirex_zs": "T3", "ttm_zs": "T3", "ttm_ft": "T3",
    "cora": "T4", "ts_rag": "T4", "cross_rag": "T4",
    "time_vlm": "T5", "visionts_pp": "T5", "unicast": "T5", "aurora": "T5",
    "crossvivit": "T6", "sunset": "T6", "solar_vlm": "T6",
}
TIER_RANK = {f"T{i}": i for i in range(7)}
# longest names first so "chronos2_ft" wins over "chronos2"
_NAMES = sorted(MODEL_TIER, key=len, reverse=True)


def base_and_tag(stem: str) -> tuple[str | None, str]:
    """Map a result-file stem to (base_model, tag-remainder)."""
    for name in _NAMES:
        if stem == name or stem.startswith(name + "_"):
            tag = stem[len(name):].lstrip("_")
            tag = re.sub(r"_seed\d+$", "", tag)
            tag = re.sub(r"_agg$", "", tag)
            return name, (tag or "—")
    return None, "—"


def fmt(v, d: int = 4) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def scenario_of(tag: str) -> tuple[str | None, str]:
    """Split a tag into (intra|extra|None, dataset-suffix) for transferability.

    S1 (same plants, held-out time) → intra-site; S2 (disjoint plants) → extra-
    site. Pairing the two for one model measures how much accuracy *transfers*
    across the plant shift (BASELINE_COMPARISON §4.1, transferability metric).
    """
    if tag.startswith("s1"):
        return "intra", tag[2:].lstrip("_") or "—"
    if tag.startswith("s2"):
        return "extra", tag[2:].lstrip("_") or "—"
    return None, tag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--md", default="results/ALL_RESULTS.md")
    ap.add_argument("--json", default="results/ALL_RESULTS.json")
    args = ap.parse_args()

    results_dir = Path(args.results)
    rows: list[dict] = []
    # tag-group -> {model: {plant: nrmse}} for the §4.4 aggregation
    nrmse_by_tag: dict[str, dict[str, dict[str, float]]] = {}

    # Trained models only ship *_seedNN.json (+ *_agg without per_plant). Pick a
    # representative file per (model, tag): a seedless run if present, else the
    # lowest seed. (model, tag) -> (rank_tuple, Path)
    chosen: dict[tuple[str, str], tuple[tuple[int, int], Path]] = {}
    for path in sorted(results_dir.glob("*.json")):
        stem = path.stem
        if stem.startswith(("significance_", "tables", "ALL_RESULTS")):
            continue
        if stem.endswith("_agg"):
            continue  # *_agg stores scalar mean±std only, no per_plant/overall
        model, tag = base_and_tag(stem)
        if model is None:
            continue
        sm = re.search(r"_seed(\d+)$", stem)
        seed = int(sm.group(1)) if sm else -1
        rank = (0, 0) if seed < 0 else (1, seed)   # seedless wins, else min seed
        key = (model, tag)
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, path)

    for (model, tag), (_, path) in chosen.items():
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        res = payload.get("results", payload)
        if not isinstance(res, dict) or "overall" not in res:
            continue
        o = res["overall"]
        rows.append({
            "tier": MODEL_TIER[model], "model": model, "tag": tag,
            "nmae": o.get("nmae"), "nrmse": o.get("nrmse"),
            "skill_score": o.get("skill_score"), "crps": o.get("crps"),
            "nmae_ramp": o.get("nmae_ramp"), "nrmse_ramp": o.get("nrmse_ramp"),
            "source": path.name,
        })
        if isinstance(res.get("per_plant"), dict):
            grp = nrmse_by_tag.setdefault(tag, {})
            grp[model] = {p: r["nrmse"] for p, r in res["per_plant"].items()
                          if isinstance(r, dict) and "nrmse" in r}

    # Reference / Tier-0 models get re-emitted under every run_eval tag (each
    # invocation rewrites smart_persistence). Collapse them to one row each so
    # the headline is not flooded; the per-tag aggregation below still uses the
    # tag-local reference.
    REF_MODELS = {"persistence", "smart_persistence",
                  "climatology_hourly", "seasonal_naive"}
    seen_ref: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        if r["model"] in REF_MODELS:
            if r["model"] in seen_ref:
                continue
            seen_ref.add(r["model"])
            r = {**r, "tag": "ref"}
        deduped.append(r)
    rows = deduped

    rows.sort(key=lambda r: (TIER_RANK[r["tier"]], r["model"], r["tag"]))

    lines = ["# All baseline results (cross-plant)", "",
             "_One row per (model, scenario tag). SS = 1 − NRMSE/NRMSE(Smart "
             "Persistence) within the same scenario._", "",
             "| Tier | Model | Tag | NMAE ↓ | NRMSE ↓ | SS ↑ | CRPS ↓ "
             "| Ramp NMAE ↓ | Ramp NRMSE ↓ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['tier']} | {r['model']} | {r['tag']} | {fmt(r['nmae'])} "
            f"| {fmt(r['nrmse'])} | {fmt(r['skill_score'], 3)} | {fmt(r['crps'])} "
            f"| {fmt(r['nmae_ramp'])} | {fmt(r['nrmse_ramp'])} |")

    # §4.4 aggregation per tag-group that carries a smart_persistence reference
    for tag, by_model in sorted(nrmse_by_tag.items()):
        ref = by_model.get("smart_persistence")
        if not ref or len(by_model) < 2:
            continue
        lines += ["", f"## Aggregation vs Smart Persistence — tag `{tag}` (§4.4)", "",
                  "| Model | Win rate ↑ | SS_geo ↑ | Avg rank ↓ |",
                  "|---|---|---|---|"]
        ranks = average_rank(by_model)
        for m in sorted(by_model, key=lambda x: (TIER_RANK[MODEL_TIER[x]], x)):
            lines.append(
                f"| {m} | {fmt(win_rate(by_model[m], ref), 3)} "
                f"| {fmt(geometric_mean_skill(by_model[m], ref), 3)} "
                f"| {fmt(ranks[m], 2)} |")

    # Transferability: pair each model's intra-site (S1) and extra-site (S2)
    # rows on the same dataset-suffix → generalization gap + retention ratio.
    transfer: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        scen, suffix = scenario_of(r["tag"])
        if scen is None:
            continue
        transfer.setdefault((r["model"], suffix), {})[scen] = r
    pairs = {k: v for k, v in transfer.items() if "intra" in v and "extra" in v}
    if pairs:
        lines += ["", "## Transferability — intra-site (S1) → extra-site (S2)", "",
                  "_Gap Δ = NRMSE(extra) − NRMSE(intra), lower = more transferable. "
                  "Retention R = SS(extra) / SS(intra), →1 = accuracy transfers._", "",
                  "| Model | Suffix | NRMSE intra | NRMSE extra | Gap Δ ↓ "
                  "| SS intra | SS extra | Retention R ↑ |",
                  "|---|---|---|---|---|---|---|---|"]
        for (model, suffix) in sorted(pairs, key=lambda k: (TIER_RANK[MODEL_TIER[k[0]]], k[0])):
            intra, extra = pairs[(model, suffix)]["intra"], pairs[(model, suffix)]["extra"]
            gap = (None if intra["nrmse"] is None or extra["nrmse"] is None
                   else extra["nrmse"] - intra["nrmse"])
            ss_i, ss_e = intra["skill_score"], extra["skill_score"]
            ret = (ss_e / ss_i if ss_i not in (None, 0) and ss_e is not None else None)
            lines.append(
                f"| {model} | {suffix} | {fmt(intra['nrmse'])} | {fmt(extra['nrmse'])} "
                f"| {fmt(gap)} | {fmt(ss_i, 3)} | {fmt(ss_e, 3)} | {fmt(ret, 3)} |")

    Path(args.md).write_text("\n".join(lines) + "\n")
    Path(args.json).write_text(json.dumps(
        {"rows": rows, "n_results": len(rows)}, indent=2) + "\n")
    print(f"aggregated {len(rows)} result rows across "
          f"{len({r['tier'] for r in rows})} tiers")
    print(f"  → {args.md}")
    print(f"  → {args.json}")


if __name__ == "__main__":
    main()
