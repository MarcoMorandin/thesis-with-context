"""Disjoint cross-plant splits (BASELINE_PROTOCOL.md §2).

Plants (site_ids) are partitioned per source dataset into train/val/test with
a seeded shuffle, after dropping sites flagged `bad_site_flag`. The split is
written once to ``configs/splits.json`` and committed, so every baseline run
uses the identical partition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

SPLITS_PATH = Path(__file__).resolve().parent.parent / "configs" / "splits.json"


def make_plant_splits(
    df: pd.DataFrame,
    seed: int = config.SEED,
    fractions: dict[str, float] = config.SPLIT_FRACTIONS,
) -> dict[str, dict[str, list[str]]]:
    """Return {dataset: {train: [...], val: [...], test: [...]}} of site_ids."""
    splits: dict[str, dict[str, list[str]]] = {}
    sites = (
        df.loc[~df[config.BAD_SITE_COL], [config.DATASET_COL, config.SITE_COL]]
        .drop_duplicates()
        .sort_values([config.DATASET_COL, config.SITE_COL])
    )
    rng = np.random.default_rng(seed)
    for dataset, group in sites.groupby(config.DATASET_COL):
        ids = sorted(group[config.SITE_COL].astype(str))
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, round(n * fractions["train"]))
        n_val = max(1, round(n * fractions["val"]))
        if n_train + n_val >= n:  # keep at least one test plant
            n_train = max(1, n - n_val - 1)
        splits[str(dataset)] = {
            "train": sorted(ids[:n_train]),
            "val": sorted(ids[n_train : n_train + n_val]),
            "test": sorted(ids[n_train + n_val :]),
        }
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict[str, dict[str, list[str]]]) -> None:
    """Fail loud on any train/val/test overlap (BASELINE_COMPARISON.md §6.2)."""
    for dataset, parts in splits.items():
        train, val, test = (set(parts[k]) for k in ("train", "val", "test"))
        overlap = (train & val) | (train & test) | (val & test)
        if overlap:
            raise ValueError(f"plant split overlap in {dataset!r}: {sorted(overlap)}")


def save_splits(splits: dict, path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n")


def load_splits(path: Path = SPLITS_PATH) -> dict[str, dict[str, list[str]]]:
    splits = json.loads(Path(path).read_text())
    assert_disjoint(splits)
    return splits


def sites_for(splits: dict, part: str) -> set[str]:
    """All site_ids of one split part across datasets."""
    return {s for parts in splits.values() for s in parts[part]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and save plant splits")
    parser.add_argument("--data", default=config.DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    frame = pd.read_parquet(
        args.data,
        columns=[config.DATASET_COL, config.SITE_COL, config.BAD_SITE_COL],
    )
    result = make_plant_splits(frame, seed=args.seed)
    save_splits(result)
    for ds, parts in result.items():
        sizes = {k: len(v) for k, v in parts.items()}
        print(f"{ds}: {sizes}")
    print(f"wrote {SPLITS_PATH}")
