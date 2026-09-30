"""Export goes_pvdaq test plants at 30-min cadence for vendored models (Time-VLM).

Pivots goes_pvdaq test plants onto a dense 30-min UTC grid and exports them as
Informer-style CSVs (date + OT).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config  # noqa: E402
from common.splits import load_splits  # noqa: E402


def _grid_frame(df: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
    """Pivot the requested sites onto a dense common 30-min UTC grid."""
    sub = df[df[config.SITE_COL].isin(sites)]
    wide = sub.pivot_table(
        index=config.TIME_COL,
        columns=config.SITE_COL,
        values=config.TARGET_COL,
        aggfunc="first",
    ).sort_index()
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="30min", tz="UTC")
    wide = wide.reindex(full).fillna(0.0)
    wide.index.name = "date"
    return wide[[s for s in sites if s in wide.columns]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=config.DEFAULT_DATA_PATH)
    ap.add_argument("--out", required=True, help="output directory for CSVs")
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train", "all"],
        help="Split to export",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits().get("goes_pvdaq", {})
    if args.split == "all":
        test_sites = sorted(
            {s for part in ("train", "val", "test") for s in splits.get(part, [])}
        )
    else:
        test_sites = splits.get(args.split, ["1202"])

    cols = [
        config.DATASET_COL,
        config.SITE_COL,
        config.TIME_COL,
        config.TARGET_COL,
        config.CAPACITY_COL,
    ]
    df = pd.read_parquet(args.data, columns=cols)
    df = df[df[config.DATASET_COL] == "goes_pvdaq"].copy()
    df[config.SITE_COL] = df[config.SITE_COL].astype(str)

    test_paths = []
    for site in test_sites:
        s = _grid_frame(df, [site]).reset_index()
        s = s.rename(columns={site: "OT"})
        path = out / f"goes_pvdaq_test_{site}.csv"
        s.to_csv(path, index=False)
        test_paths.append(path.name)

    caps = (
        df.drop_duplicates(config.SITE_COL)
        .set_index(config.SITE_COL)[config.CAPACITY_COL]
        .to_dict()
    )
    cap_file = out / "capacity_goes.json"
    cap_file.write_text(
        json.dumps({str(k): float(v) for k, v in caps.items()}, indent=2, sort_keys=True)
    )
    print(f"wrote {len(test_paths)} test CSVs to {out}")


if __name__ == "__main__":
    main()
