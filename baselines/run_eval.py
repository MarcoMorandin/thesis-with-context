"""Run baselines on the disjoint cross-plant protocol (BASELINE_COMPARISON §4).

Scenario flags compose (presets live in scripts/run_suite.py):

    S1 in-domain        --in-domain            (train plants, last 20 % of time)
    S2 cross-plant      (default)              --split test
    S3 cross-dataset    --train-datasets uk_pv --eval-datasets goes_pvdaq
    S4 long-horizon     --horizon 24|48
    S5 data efficiency  --train-fraction 0.25
    S6 ramp subset      always reported (nmae_ramp / nrmse_ramp columns)
    §5 controls         --control zero_cov|low_history_4|low_history_8|...

Examples
--------
    uv run python -m common.splits                     # once, committed
    uv run python run_eval.py --model smart_persistence persistence
    uv run python run_eval.py --model lightgbm dlinear patchtst --seeds 42 43 44
    uv run python run_eval.py --model chronos2_zs ts_rag cora
    uv run python run_eval.py --model dlinear --horizon 48
    uv run python run_eval.py --model chronos2_zs --control low_history_8
"""

from __future__ import annotations

import argparse
import inspect
import json

import numpy as np
import pandas as pd

from common import config
from common.base import REGISTRY, build
from common.controls import CONTROLS, apply_control
from common.runner import (add_skill_scores, compute_ramp_thresholds,
                           evaluate_model, write_results)
from common.splits import (SPLITS_PATH, load_splits, make_plant_splits,
                           save_splits, sites_for)
from common.windows import dataset_for_sites

IN_DOMAIN_TRAIN_RANGE = (0.0, 0.8)
IN_DOMAIN_EVAL_RANGE = (0.8, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--data", default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--history", type=int, default=config.HISTORY_STEPS)
    parser.add_argument("--horizon", type=int, default=config.HORIZON_STEPS)
    parser.add_argument("--eval-stride", type=int, default=config.HORIZON_STEPS,
                        help="window stride on the eval split (H = non-overlapping)")
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[config.SEED],
                        help="≥3 seeds for trained models per §4.5")
    parser.add_argument("--out", default="results")
    parser.add_argument("--tag", default="",
                        help="suffix for result files (scenario id, control, ...)")
    # scenarios
    parser.add_argument("--in-domain", action="store_true",
                        help="S1: train plants, held-out time range")
    parser.add_argument("--train-datasets", nargs="+", default=None,
                        help="S3: restrict training plants to these datasets")
    parser.add_argument("--eval-datasets", nargs="+", default=None,
                        help="S3: restrict eval plants to these datasets")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="S5: fraction of train plants (seeded subsample)")
    parser.add_argument("--control", default="none",
                        choices=["none", *sorted(CONTROLS)],
                        help="§5 eval-time input control")
    parser.add_argument("--model-kwargs", default="{}",
                        help="JSON kwargs forwarded to every --model (A15 sweeps)")
    return parser.parse_args()


def _filter_datasets(df: pd.DataFrame, names: list[str] | None) -> pd.DataFrame:
    return df if not names else df[df[config.DATASET_COL].isin(names)]


def _subsample_plants(sites: set[str], fraction: float, seed: int) -> set[str]:
    if fraction >= 1.0:
        return sites
    ordered = sorted(sites)
    n = max(1, round(len(ordered) * fraction))
    pick = np.random.default_rng(seed).choice(len(ordered), n, replace=False)
    return {ordered[i] for i in pick}


def _accepts_seed(name: str) -> bool:
    return "seed" in inspect.signature(REGISTRY[name].__init__).parameters


def _aggregate_seeds(per_seed: list[dict]) -> dict:
    """Mean ± std over seeds for the scalar overall metrics."""
    keys = {
        k for r in per_seed for k, v in r["overall"].items()
        if isinstance(v, float)
    }
    agg = {}
    for k in sorted(keys):
        values = [r["overall"][k] for r in per_seed if k in r["overall"]]
        agg[k] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return {"overall_mean_std": agg, "n_seeds": len(per_seed)}


def _load_registry() -> None:
    """Import all tier packages so REGISTRY is fully populated."""
    import importlib

    for pkg in ("tier0", "tier1", "tslib", "tier3", "tier4"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            pass  # optional dependency group not installed


def main() -> None:
    args = parse_args()
    _load_registry()
    model_kwargs = json.loads(args.model_kwargs)
    unknown = [m for m in args.model if m not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; known: {sorted(REGISTRY)}")
    df = pd.read_parquet(args.data)
    if SPLITS_PATH.exists():
        splits = load_splits()
    else:
        splits = make_plant_splits(df, seed=config.SEED)
        save_splits(splits)
        print(f"generated plant splits → {SPLITS_PATH}")

    window_kwargs = dict(history=args.history, horizon=args.horizon)
    train_sites = _subsample_plants(
        sites_for(splits, "train"), args.train_fraction, config.SEED
    )
    val_sites = sites_for(splits, "val")
    train_df = _filter_datasets(df, args.train_datasets)
    eval_df = _filter_datasets(df, args.eval_datasets)

    if args.in_domain:  # S1: same plants, disjoint time
        eval_sites = train_sites
        eval_range, train_range = IN_DOMAIN_EVAL_RANGE, IN_DOMAIN_TRAIN_RANGE
    else:               # S2/S3/...: disjoint plants
        eval_sites = sites_for(splits, args.split)
        eval_range = train_range = None

    eval_ds = dataset_for_sites(
        eval_df, eval_sites, time_range=eval_range,
        stride=args.eval_stride, **window_kwargs,
    )
    print(f"eval: {len(eval_ds)} windows, {len(eval_ds.series)} plants "
          f"(in_domain={args.in_domain}, control={args.control})")

    ramp_thresholds = compute_ramp_thresholds(eval_ds)
    transform = (
        None if args.control == "none"
        else (lambda b: apply_control(args.control, b))
    )

    def run(model, collect=True):
        return evaluate_model(
            model, eval_ds, args.batch_size,
            ramp_thresholds=ramp_thresholds,
            collect_losses=collect, transform=transform,
        )

    # Smart Persistence always runs first: it is the Skill-Score denominator.
    reference = run(build("smart_persistence"), collect=True)

    train_ds = val_ds = None
    tag = f"_{args.tag}" if args.tag else ""
    run_config = vars(args) | {"quantile_levels": config.QUANTILE_LEVELS}
    write_results(args.out, f"smart_persistence{tag}",
                  add_skill_scores(dict(reference), reference),
                  run_config, args.data)

    for name in args.model:
        if name == "smart_persistence":
            continue  # already written as the reference
        per_seed = []
        seeds = args.seeds if (REGISTRY[name].requires_fit
                               and _accepts_seed(name)) else args.seeds[:1]
        for seed in seeds:
            kwargs = dict(model_kwargs)
            if _accepts_seed(name):
                kwargs.setdefault("seed", seed)
            model = build(name, **kwargs)
            if model.requires_fit:
                if train_ds is None:
                    train_ds = dataset_for_sites(
                        train_df, train_sites, time_range=train_range,
                        stride=args.train_stride, **window_kwargs,
                    )
                    val_ds = dataset_for_sites(
                        train_df, val_sites,
                        stride=args.train_stride, **window_kwargs,
                    )
                    print(f"train: {len(train_ds)} windows, "
                          f"val: {len(val_ds)} windows")
                model.fit(train_ds, val_ds)
            results = add_skill_scores(run(model), reference)
            per_seed.append(results)
            suffix = f"{tag}_seed{seed}" if len(seeds) > 1 else tag
            path = write_results(args.out, f"{name}{suffix}", results,
                                 run_config | {"seed": seed}, args.data)
            overall = results["overall"]
            print(f"{name}[{seed}]: NMAE={overall['nmae']:.4f} "
                  f"NRMSE={overall['nrmse']:.4f} "
                  f"SS={overall.get('skill_score', float('nan')):.4f} → {path}")
        if len(per_seed) > 1:
            write_results(args.out, f"{name}{tag}_agg",
                          _aggregate_seeds(per_seed), run_config, args.data)


if __name__ == "__main__":
    main()
