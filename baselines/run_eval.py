"""Run baselines on the disjoint cross-plant protocol (BASELINE_COMPARISON §4).

Scenario flags compose (presets live in scripts/run_suite.py):

    S1 in-domain        --in-domain            (train plants, last 20 % of time)
    S2 cross-plant      (default)              --split test
    S2 LOPO variant     --lopo-dataset goes_pvdaq   (mandatory for goes_pvdaq)
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
    uv run python run_eval.py --model chronos2_zs cora
    uv run python run_eval.py --model dlinear --horizon 48
    uv run python run_eval.py --model chronos2_zs --lopo-dataset goes_pvdaq
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
    # Windows default to PHYSICAL TIME (per-dataset, cadence-fair). Pass the
    # step-based --history/--horizon to override (e.g. S4 long-horizon --horizon 48).
    parser.add_argument("--history", type=int, default=None,
                        help="history in STEPS (overrides --history-days)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="horizon in STEPS (overrides --horizon-hours)")
    parser.add_argument("--history-days", type=float, default=config.HISTORY_DAYS,
                        help="context in days (default physical-time spec)")
    parser.add_argument("--horizon-hours", type=float, default=config.HORIZON_HOURS,
                        help="horizon in hours (default physical-time spec)")
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
    parser.add_argument("--lopo-dataset", default=None,
                        help="leave-one-plant-out rotation over this dataset "
                             "(mandatory for goes_pvdaq, see §4.1)")
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


def _scalar_overalls(results_list: list[dict]) -> dict[str, dict[str, float]]:
    """Mean ± std of the scalar overall metrics across runs (seeds or folds)."""
    keys = {
        k for r in results_list for k, v in r.items() if isinstance(v, float)
    }
    return {
        k: {
            "mean": float(np.mean([r[k] for r in results_list if k in r])),
            "std": float(np.std([r[k] for r in results_list if k in r])),
        }
        for k in sorted(keys)
    }


def _load_registry() -> None:
    """Import all tier packages so REGISTRY is fully populated."""
    import importlib

    for pkg in ("tier0", "tier1", "tslib", "tier3", "tier4"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            pass  # optional dependency group not installed


def evaluate_suite(
    args: argparse.Namespace,
    df: pd.DataFrame,
    train_sites: set[str],
    val_sites: set[str],
    eval_sites: set[str],
    train_range: tuple[float, float] | None,
    eval_range: tuple[float, float] | None,
    tag: str,
    model_kwargs: dict,
) -> dict[str, dict]:
    """Evaluate all requested models on one site configuration.

    Returns {model_name: seed-mean overall metrics} for fold aggregation.
    """
    # step args override physical-time spec when explicitly provided
    window_kwargs: dict = {}
    if args.history is not None:
        window_kwargs["history"] = args.history
    else:
        window_kwargs["history_days"] = args.history_days
    if args.horizon is not None:
        window_kwargs["horizon"] = args.horizon
    else:
        window_kwargs["horizon_hours"] = args.horizon_hours
    train_df = _filter_datasets(df, args.train_datasets)
    eval_df = _filter_datasets(df, args.eval_datasets)

    eval_ds = dataset_for_sites(
        eval_df, eval_sites, time_range=eval_range,
        stride=args.eval_stride, **window_kwargs,
    )
    print(f"[{tag or 'run'}] eval: {len(eval_ds)} windows, "
          f"{len(eval_ds.series)} plants (control={args.control})")

    ramp_thresholds = compute_ramp_thresholds(eval_ds)
    transform = (
        None if args.control == "none"
        else (lambda b: apply_control(args.control, b))
    )

    def run(model):
        return evaluate_model(
            model, eval_ds, args.batch_size,
            ramp_thresholds=ramp_thresholds,
            collect_losses=True, transform=transform,
        )

    # Smart Persistence always runs first: it is the Skill-Score denominator.
    reference = run(build("smart_persistence"))

    suffix = f"_{tag}" if tag else ""
    run_config = vars(args) | {"quantile_levels": config.QUANTILE_LEVELS}
    write_results(args.out, f"smart_persistence{suffix}",
                  add_skill_scores(dict(reference), reference),
                  run_config, args.data)

    train_ds = val_ds = None
    fold_overalls: dict[str, dict] = {
        "smart_persistence": reference["overall"] | {"skill_score": 0.0}
    }
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
                    print(f"[{tag or 'run'}] train: {len(train_ds)} windows, "
                          f"val: {len(val_ds)} windows")
                model.fit(train_ds, val_ds)
            results = add_skill_scores(run(model), reference)
            per_seed.append(results["overall"])
            seed_suffix = f"{suffix}_seed{seed}" if len(seeds) > 1 else suffix
            path = write_results(args.out, f"{name}{seed_suffix}", results,
                                 run_config | {"seed": seed}, args.data)
            overall = results["overall"]
            print(f"[{tag or 'run'}] {name}[{seed}]: "
                  f"NMAE={overall['nmae']:.4f} NRMSE={overall['nrmse']:.4f} "
                  f"SS={overall.get('skill_score', float('nan')):.4f} → {path}")
        if len(per_seed) > 1:
            write_results(args.out, f"{name}{suffix}_agg",
                          {"overall_mean_std": _scalar_overalls(per_seed),
                           "n_seeds": len(per_seed)},
                          run_config, args.data)
        # seed-mean scalars, used for fold aggregation in LOPO mode
        fold_overalls[name] = {
            k: float(np.mean([o[k] for o in per_seed if k in o]))
            for k in per_seed[0] if isinstance(per_seed[0][k], float)
        }
    return fold_overalls


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

    train_sites = _subsample_plants(
        sites_for(splits, "train"), args.train_fraction, config.SEED
    )
    val_sites = sites_for(splits, "val")

    if args.lopo_dataset:
        # Leave-one-plant-out rotation (§4.1): each plant of the dataset is
        # the test fold once, the next plant in rotation provides early-stop
        # validation, all remaining plants train. Other datasets keep their
        # committed split roles. Reported as mean ± std over folds.
        if args.lopo_dataset not in splits:
            raise SystemExit(f"unknown dataset {args.lopo_dataset!r}; "
                             f"known: {sorted(splits)}")
        if args.in_domain:
            raise SystemExit("--lopo-dataset and --in-domain are exclusive")
        plants = sorted({
            s for part in splits[args.lopo_dataset].values() for s in part
        })
        other_train = {
            s for ds, parts in splits.items()
            if ds != args.lopo_dataset for s in parts["train"]
        } & train_sites
        other_val = {
            s for ds, parts in splits.items()
            if ds != args.lopo_dataset for s in parts["val"]
        }
        per_fold: dict[str, list[dict]] = {}
        for i, plant in enumerate(plants):
            val_plant = plants[(i + 1) % len(plants)]
            fold_tag = f"{args.tag + '_' if args.tag else ''}" \
                       f"lopo_{args.lopo_dataset}_fold{i:02d}"
            overalls = evaluate_suite(
                args, df,
                train_sites=other_train | (set(plants) - {plant, val_plant}),
                val_sites=other_val | {val_plant},
                eval_sites={plant},
                train_range=None, eval_range=None,
                tag=fold_tag, model_kwargs=model_kwargs,
            )
            for name, overall in overalls.items():
                per_fold.setdefault(name, []).append(overall)
        run_config = vars(args) | {"quantile_levels": config.QUANTILE_LEVELS}
        for name, folds in per_fold.items():
            write_results(
                args.out,
                f"{name}_{args.tag + '_' if args.tag else ''}"
                f"lopo_{args.lopo_dataset}_agg",
                {"overall_mean_std": _scalar_overalls(folds),
                 "n_folds": len(folds)},
                run_config, args.data,
            )
        print(f"LOPO over {len(plants)} folds done → *_lopo_"
              f"{args.lopo_dataset}_agg.json")
        return

    if args.in_domain:  # S1: same plants, disjoint time
        eval_sites = train_sites
        eval_range, train_range = IN_DOMAIN_EVAL_RANGE, IN_DOMAIN_TRAIN_RANGE
    else:               # S2/S3/...: disjoint plants
        eval_sites = sites_for(splits, args.split)
        eval_range = train_range = None

    evaluate_suite(
        args, df,
        train_sites=train_sites, val_sites=val_sites, eval_sites=eval_sites,
        train_range=train_range, eval_range=eval_range,
        tag=args.tag, model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()
