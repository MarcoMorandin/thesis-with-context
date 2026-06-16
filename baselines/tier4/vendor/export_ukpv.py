"""Export the uk_pv numerical split to the format the vendored TS-RAG /
Cross-RAG code consumes (Informer-style CSV: ``date`` + value columns + ``OT``).

Numerical only — these are time-series foundation models, no images.

Honours the BASELINE_PROTOCOL.md §2 fairness contract: the retrieval datastore
is built from **train plants only** (`uk_pv_train.csv`); each **test** plant is a
separate query CSV (`uk_pv_test_<site>.csv`, `OT` = that plant). The vendored
`Dataset_Custom_retrieve` reads `date` + variable columns + the `OT` target and
StandardScale-normalises, so we export the capacity-normalised `norm_power` on a
dense common 30-min grid (gaps → 0, the physical night value) and ship a
`capacity.json` so predictions can be mapped back for our NMAE/SS metrics.

Runnable in the baselines venv (pandas only); the retrieval-DB build + zeroshot
run happen later in the upstream env on the cluster (see
docs/experiments/TIER4_RAG_INTEGRATION.md and scripts/login_node_prep.sh).

    uv run python tier4/vendor/export_ukpv.py \
        --data /Volumes/SSD/.../all_curated.parquet --out /tmp/ukpv_rag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # baselines/ on path

from common import config           # noqa: E402
from common.splits import load_splits  # noqa: E402

DATASET = "uk_pv"


def _grid_frame(df: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
    """Pivot the requested sites onto a dense common 30-min UTC grid."""
    sub = df[df[config.SITE_COL].isin(sites)]
    wide = sub.pivot_table(
        index=config.TIME_COL, columns=config.SITE_COL,
        values=config.TARGET_COL, aggfunc="first",
    ).sort_index()
    # dense regular grid (their loader assumes no gaps); night/outage → 0
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="30min", tz="UTC")
    wide = wide.reindex(full).fillna(0.0)
    wide.index.name = "date"
    # stable column order = the committed split order
    return wide[[s for s in sites if s in wide.columns]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=config.DEFAULT_DATA_PATH)
    ap.add_argument("--out", required=True, help="output dir for CSVs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits()[DATASET]
    cols = [config.DATASET_COL, config.SITE_COL, config.TIME_COL,
            config.TARGET_COL, config.CAPACITY_COL]
    df = pd.read_parquet(args.data, columns=cols)
    df = df[df[config.DATASET_COL] == DATASET].copy()
    df[config.SITE_COL] = df[config.SITE_COL].astype(str)

    # retrieval datastore source = train plants only (§3)
    train = _grid_frame(df, splits["train"])
    train_csv = train.reset_index()
    train_csv["OT"] = train_csv[splits["train"][0]]   # loader needs an OT column
    train_csv.to_csv(out / "uk_pv_train.csv", index=False)

    # one query CSV per test plant (OT = that plant)
    test_paths = []
    for site in splits["test"]:
        s = _grid_frame(df, [site]).reset_index()
        s = s.rename(columns={site: "OT"})
        path = out / f"uk_pv_test_{site}.csv"
        s.to_csv(path, index=False)
        test_paths.append(path.name)

    # stacked single-series training set: all train plants concatenated in time.
    # TSLib-style harnesses (Time-VLM, --features S) train ONE univariate `OT`
    # series; stacking exposes every train plant without editing vendored code.
    # Plant boundaries are mild discontinuities (documented), not a leak: still
    # train-plants-only, disjoint from the test CSVs.
    stacked = pd.concat([train[c] for c in train.columns], ignore_index=True)
    synth = pd.date_range("2019-01-01", periods=len(stacked), freq="30min", tz="UTC")
    pd.DataFrame({"date": synth, "OT": stacked.to_numpy()}).to_csv(
        out / "uk_pv_train_stacked.csv", index=False)

    # capacities (W) for de-normalising predictions back to physical scale
    caps = (df.drop_duplicates(config.SITE_COL)
              .set_index(config.SITE_COL)[config.CAPACITY_COL].to_dict())
    (out / "capacity.json").write_text(json.dumps(
        {str(k): float(v) for k, v in caps.items()}, indent=2, sort_keys=True))

    manifest = {
        "dataset": DATASET, "cadence_min": 30,
        "n_train_plants": len(splits["train"]),
        "n_test_plants": len(splits["test"]),
        "train_csv": "uk_pv_train.csv",
        "train_stacked_csv": "uk_pv_train_stacked.csv",
        "test_csvs": test_paths,
        "rows_train": int(len(train)),
        "rows_train_stacked": int(len(stacked)),
        "gap_fill": "0.0 (dense 30-min grid; night/outage = 0)",
        "target": "norm_power (capacity-normalised)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {1 + len(test_paths) + 2} files to {out}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
