"""Publication summary for the uk_pv S2 cross-plant runs.

Reads the `*_s2_ukpv*` artifacts written by run_eval.py and renders a single
self-contained markdown report (headline table with seed mean±std, §4.4
aggregation, Diebold–Mariano + block-bootstrap significance vs Smart
Persistence, and the daylit per-horizon NMAE curve).

    uv run python scripts/summarize_ukpv.py --tag s2_ukpv \
        --out ../docs/experiments/BASELINE_RESULTS_UKPV.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.aggregate import average_rank, geometric_mean_skill, win_rate  # noqa: E402
from common.stats import block_bootstrap_ci, dm_test  # noqa: E402

# (tier, display name, zero-shot?, trainable-parameter note)
MODELS: list[tuple[str, str, str, bool, str]] = [
    ("T0", "persistence", "Persistence", True, "0"),
    ("T0", "smart_persistence", "Smart Persistence (ref)", True, "0"),
    ("T0", "climatology_hourly", "Hourly climatology", True, "0 (lookup)"),
    ("T0", "seasonal_naive", "Seasonal-naive", True, "0"),
    ("T1", "lightgbm", "LightGBM (9× quantile)", False, "GBDT ensemble"),
    ("T2", "mlp", "MLP", False, "small"),
    ("T2", "dlinear", "DLinear", False, "linear"),
    ("T2", "patchtst", "PatchTST", False, "transformer"),
    ("T2", "itransformer", "iTransformer", False, "transformer"),
    ("T2", "tft", "TFT-lite (quantile)", False, "transformer"),
    # Tier 3/4/5 appear only when their cluster results land in results/.
    ("T3", "chronos2_zs", "Chronos-2 ZS", True, "0"),
    ("T3", "chronos2_ft", "Chronos-2 FT", False, "adapter"),
    ("T3", "timesfm_zs", "TimesFM 2.5 ZS", True, "0"),
    ("T3", "tirex_zs", "TiRex ZS", True, "0"),
    ("T4", "cora", "CoRA (Chronos-2)", False, "adapter"),
    ("T4", "ts_rag_orig", "TS-RAG (orig 512/64)", True, "mixer"),
    ("T4", "ts_rag_proto", "TS-RAG (proto 24/12)", False, "mixer"),
    ("T4", "cross_rag_orig", "Cross-RAG (orig)", True, "mixer"),
    ("T4", "cross_rag_proto", "Cross-RAG (proto)", False, "mixer"),
    # Tier 5 (vendored, native eval windows — see import_predictions.py caveats)
    ("T5", "time_vlm", "Time-VLM", False, "VLM+TSLib"),
    ("T5", "visionts_pp", "VisionTS++ (ZS)", True, "MAE"),
    ("T5", "unicast", "UniCast", False, "prompt"),
    ("T5", "aurora", "Aurora", True, "MTSFM"),
]


def fmt(v, d=4):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fmt_pm(mean, std, d=4):
    if mean is None:
        return "—"
    if std is None or std == 0:
        return f"{mean:.{d}f}"
    return f"{mean:.{d}f} ± {std:.{d}f}"


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def overall_and_std(results_dir: Path, name: str, tag: str):
    """Return (overall dict, std dict). Uses the *_agg seed mean±std when present."""
    agg = load_json(results_dir / f"{name}_{tag}_agg.json")
    if agg and "results" in agg and "overall_mean_std" in agg["results"]:
        ms = agg["results"]["overall_mean_std"]
        mean = {k: v["mean"] for k, v in ms.items()}
        std = {k: v["std"] for k, v in ms.items()}
        return mean, std
    single = load_json(results_dir / f"{name}_{tag}.json")
    if single and "results" in single:
        return single["results"]["overall"], {}
    return None, {}


def per_plant_nrmse(results_dir: Path, name: str, tag: str):
    """Per-plant NRMSE from the seed-42 (or single) run, for §4.4 aggregation."""
    for cand in (f"{name}_{tag}_seed42.json", f"{name}_{tag}.json"):
        j = load_json(results_dir / cand)
        if j and "results" in j and "per_plant" in j["results"]:
            return {p: row["nrmse"] for p, row in j["results"]["per_plant"].items()}
    return None


def per_horizon(results_dir: Path, name: str, tag: str):
    """Per-horizon NMAE list (agg files store scalars only, so use a seed run)."""
    for cand in (f"{name}_{tag}_seed42.json", f"{name}_{tag}.json"):
        j = load_json(results_dir / cand)
        if j and "results" in j:
            return j["results"]["overall"].get("nmae_per_horizon")
    return None


def losses_path(results_dir: Path, name: str, tag: str) -> Path | None:
    for cand in (f"{name}_{tag}_seed42_losses.npz", f"{name}_{tag}_losses.npz"):
        if (results_dir / cand).exists():
            return results_dir / cand
    return None


def load_losses(p: Path):
    with np.load(p, allow_pickle=False) as d:
        return {k: d[k] for k in ("loss", "plant", "day")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="s2_ukpv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()
    rd = Path(args.results)
    tag = args.tag

    present = [
        (tier, name, disp, zs, par)
        for tier, name, disp, zs, par in MODELS
        if (rd / f"{name}_{tag}.json").exists()
        or (rd / f"{name}_{tag}_agg.json").exists()
    ]

    L = ["# Baseline results — `uk_pv` cross-plant (S2)", "",
         "Generated by `baselines/scripts/summarize_ukpv.py`. Disjoint "
         "cross-plant protocol (BASELINE_PROTOCOL.md): 69 train / 15 val / 14 "
         "**test** plants, capacity-normalized `norm_power`, 30-min cadence, "
         "history T=24 (12 h), horizon H=12 (6 h). Metrics are macro-averaged "
         "over the 14 test plants on `mask_future · daylight` steps. "
         "Skill Score `SS = 1 − NRMSE/NRMSE_SmartPersistence` (Smart "
         "Persistence is the reference, SS≡0). Trained models report seed "
         "mean ± std over seeds {42, 43, 44}.", ""]

    # ---- headline ----
    L += ["## 1. Headline table", "",
          "| Tier | Model | NMAE ↓ | NRMSE ↓ | SS ↑ | CRPS ↓ | Ramp NMAE ↓ "
          "| Ramp NRMSE ↓ | Trainable | ZS |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    overalls, stds = {}, {}
    for tier, name, disp, zs, par in present:
        o, s = overall_and_std(rd, name, tag)
        overalls[name], stds[name] = o, s
        L.append(
            f"| {tier} | {disp} "
            f"| {fmt_pm(o.get('nmae'), s.get('nmae'))} "
            f"| {fmt_pm(o.get('nrmse'), s.get('nrmse'))} "
            f"| {fmt_pm(o.get('skill_score'), s.get('skill_score'), 3)} "
            f"| {fmt_pm(o.get('crps'), s.get('crps'))} "
            f"| {fmt(o.get('nmae_ramp'))} | {fmt(o.get('nrmse_ramp'))} "
            f"| {par} | {'✅' if zs else '❌'} |"
        )

    # ---- §4.4 aggregation ----
    ppn = {name: per_plant_nrmse(rd, name, tag) for _, name, *_ in present}
    ref = ppn.get("smart_persistence")
    if ref:
        valid = {n: v for n, v in ppn.items() if v}
        ranks = average_rank(valid)
        L += ["", "## 2. Cross-plant aggregation (§4.4, per-plant NRMSE)", "",
              "Win rate / geometric-mean skill vs Smart Persistence; average "
              "rank over the 14 test plants (lower is better).", "",
              "| Model | Win rate vs SP ↑ | SS_geo ↑ | Avg rank ↓ |",
              "|---|---|---|---|"]
        for _, name, disp, *_ in present:
            if name not in valid:
                continue
            L.append(
                f"| {disp} | {fmt(win_rate(valid[name], ref), 3)} "
                f"| {fmt(geometric_mean_skill(valid[name], ref), 3)} "
                f"| {fmt(ranks[name], 2)} |"
            )

    # ---- significance vs Smart Persistence ----
    sp_loss = losses_path(rd, "smart_persistence", tag)
    if sp_loss:
        base = load_losses(sp_loss)
        L += ["", "## 3. Significance vs Smart Persistence (§4.5)", "",
              "Diebold–Mariano test (HLN-corrected) and paired block bootstrap "
              "(block = day, 1000 resamples) on per-window masked-MAE "
              "differentials (candidate − Smart Persistence). Negative ΔMAE ⇒ "
              "candidate better. **Significant** = DM p<0.05 *and* bootstrap "
              "95 % CI excludes 0.", "",
              "| Model | ΔMAE | DM p | Bootstrap 95% CI | Significant |",
              "|---|---|---|---|---|"]
        for _, name, disp, *_ in present:
            if name == "smart_persistence":
                continue
            lp = losses_path(rd, name, tag)
            cand = load_losses(lp) if lp else None
            if not cand or cand["loss"].shape != base["loss"].shape \
                    or not np.array_equal(cand["day"], base["day"]):
                L.append(f"| {disp} | — | — | not aligned | — |")
                continue
            dm = dm_test(cand["loss"], base["loss"], h=args.horizon)
            boot = block_bootstrap_ci(cand["loss"], base["loss"],
                                      blocks=cand["day"], n_resamples=1000)
            sig = bool(dm["p_value"] < 0.05 and boot["significant"])
            L.append(
                f"| {disp} | {dm['mean_diff']:+.5f} | {fmt(dm['p_value'], 4)} "
                f"| [{boot['ci_low']:+.5f}, {boot['ci_high']:+.5f}] "
                f"| {'**yes**' if sig else 'no'} |"
            )

    # ---- per-horizon NMAE ----
    ph = {name: per_horizon(rd, name, tag) for _, name, *_ in present}
    H = next((len(v) for v in ph.values() if v), 0)
    daylit = [h for h in range(H)
              if any(ph[n] and ph[n][h] for _, n, *_ in present)]
    if daylit:
        head = " | ".join(f"h{h+1}" for h in daylit)
        L += ["", "## 4. Per-horizon NMAE (daylit steps)", "",
              "Night-only horizon positions (NMAE≡0 across all models, masked "
              "out) are omitted. Each step is 30 min ahead. Trained-model rows "
              "use seed 42.", "",
              f"| Model | {head} |",
              "|---|" + "---|" * len(daylit)]
        for _, name, disp, *_ in present:
            v = ph[name]
            if not v:
                continue
            cells = " | ".join(fmt(v[h], 4) for h in daylit)
            L.append(f"| {disp} | {cells} |")

    L += ["", "---", "",
          "*Source artifacts: `baselines/results/*_" + tag + "*`. "
          "Reproduce: `uv run python scripts/summarize_ukpv.py`.*"]

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
