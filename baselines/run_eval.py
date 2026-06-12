"""Run one baseline on the disjoint cross-plant protocol.

Examples
--------
    # generate the plant split once (committed to configs/splits.json)
    uv run python -m common.splits

    # tier 0 references (zero-shot, fast)
    uv run python run_eval.py --model smart_persistence
    uv run python run_eval.py --model persistence seasonal_naive climatology_hourly

    # trained baselines
    uv run python run_eval.py --model lightgbm dlinear patchtst itransformer mlp tft

    # long-horizon scenario S4
    uv run python run_eval.py --model dlinear --horizon 48
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import config
from common.base import build
from common.runner import add_skill_scores, evaluate_model, write_results
from common.splits import SPLITS_PATH, load_splits, make_plant_splits, save_splits, sites_for
from common.windows import dataset_for_sites


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--data", default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--history", type=int, default=config.HISTORY_STEPS)
    parser.add_argument("--horizon", type=int, default=config.HORIZON_STEPS)
    parser.add_argument("--eval-stride", type=int, default=config.HORIZON_STEPS,
                        help="window stride on the eval split (H = non-overlapping)")
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if SPLITS_PATH.exists():
        splits = load_splits()
    else:
        splits = make_plant_splits(df, seed=config.SEED)
        save_splits(splits)
        print(f"generated plant splits → {SPLITS_PATH}")

    window_kwargs = dict(history=args.history, horizon=args.horizon)
    eval_ds = dataset_for_sites(
        df, sites_for(splits, args.split), stride=args.eval_stride, **window_kwargs
    )
    print(f"eval split={args.split}: {len(eval_ds)} windows, "
          f"{len(eval_ds.series)} plants")

    train_ds = val_ds = None
    run_config = vars(args) | {"quantile_levels": config.QUANTILE_LEVELS}

    # Smart Persistence always runs first: it is the Skill-Score denominator.
    reference = evaluate_model(build("smart_persistence"), eval_ds, args.batch_size)

    for name in args.model:
        model = build(name)
        if model.requires_fit:
            if train_ds is None:
                train_ds = dataset_for_sites(
                    df, sites_for(splits, "train"),
                    stride=args.train_stride, **window_kwargs,
                )
                val_ds = dataset_for_sites(
                    df, sites_for(splits, "val"),
                    stride=args.train_stride, **window_kwargs,
                )
                print(f"train: {len(train_ds)} windows, val: {len(val_ds)} windows")
            model.fit(train_ds, val_ds)
        results = evaluate_model(model, eval_ds, args.batch_size)
        results = add_skill_scores(results, reference)
        path = write_results(args.out, name, results, run_config, args.data)
        overall = results["overall"]
        print(f"{name}: NMAE={overall['nmae']:.4f} NRMSE={overall['nrmse']:.4f} "
              f"SS={overall.get('skill_score', float('nan')):.4f} → {path}")


if __name__ == "__main__":
    main()
