"""
Build unified train / val / test window index across all refactored solar datasets.

What this script does
─────────────────────
1. Reads the refactored timeseries parquets for SKIPP'D, Solarnet, GOES-16+NSRDB.
2. For each dataset, generates all valid sliding windows of length (hist_steps + horizon).
3. Applies chronological 70 / 15 / 15 % split per dataset.
4. Writes train_windows.parquet, val_windows.parquet, test_windows.parquet.
5. Writes schema.json describing the complete unified data schema.

Output layout  (data/refactored/unified/)
──────────────────────────────────────────
  schema.json
  train_windows.parquet
  val_windows.parquet
  test_windows.parquet

Each windows parquet has columns:
  dataset          (string)  — "skippd" | "solarnet" | "goes16_nsrdb"
  entity_id        (int32)   — entity within the dataset
  window_start     (int64)   — Unix seconds of first historical step
  window_end       (int64)   — Unix seconds of last horizon step
  hist_steps       (int32)   — T
  horizon          (int32)   — H
  has_visual       (int8)    — 1 if at least one frame exists in the visual window
  frame_dir        (string)  — relative path to frames/ directory
  ts_path          (string)  — relative path to the timeseries parquet

Usage:
  python scripts/solar/build_unified.py
    [--hist-steps  168]    (default 168 — 7 days at 1-hour cadence)
    [--horizon     24]     (default 24)
    [--visual-window 6]    (how many video frames per sample)
    [--data-root PATH]
    [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

def _build_windows(ts_list: list[int],
                   entity_id: int,
                   hist_steps: int,
                   horizon: int,
                   frame_ts_set: set[int],
                   visual_window: int) -> list[dict]:
    """
    Slide a (hist_steps + horizon) window over a sorted timestamp list.
    Returns list of window dicts.
    """
    W  = hist_steps + horizon
    n  = len(ts_list)
    windows = []
    for i in range(n - W + 1):
        win_ts = ts_list[i: i + W]
        # Visual window = last visual_window steps of the history portion
        vis_start = i + hist_steps - visual_window
        vis_end   = i + hist_steps
        vis_ts    = ts_list[max(vis_start, 0): vis_end]
        has_visual = int(any(t in frame_ts_set for t in vis_ts))
        windows.append({
            "entity_id":    entity_id,
            "window_start": win_ts[0],
            "window_end":   win_ts[-1],
            "hist_steps":   hist_steps,
            "horizon":      horizon,
            "has_visual":   has_visual,
        })
    return windows


def _split_windows(windows: list[dict],
                   train_frac: float = 0.70,
                   val_frac:   float = 0.15,
                   ) -> tuple[list[dict], list[dict], list[dict]]:
    """Chronological split (no shuffling to preserve temporal order)."""
    n      = len(windows)
    n_train = int(n * train_frac)
    n_val   = int(n * (train_frac + val_frac))
    return windows[:n_train], windows[n_train:n_val], windows[n_val:]


def _windows_to_table(windows: list[dict],
                      dataset: str,
                      frame_dir: str,
                      ts_path: str) -> pa.Table:
    n = len(windows)
    return pa.table({
        "dataset":       pa.array([dataset]         * n, pa.string()),
        "entity_id":     pa.array([w["entity_id"]   for w in windows], pa.int32()),
        "window_start":  pa.array([w["window_start"] for w in windows], pa.int64()),
        "window_end":    pa.array([w["window_end"]   for w in windows], pa.int64()),
        "hist_steps":    pa.array([w["hist_steps"]   for w in windows], pa.int32()),
        "horizon":       pa.array([w["horizon"]      for w in windows], pa.int32()),
        "has_visual":    pa.array([w["has_visual"]   for w in windows], pa.int8()),
        "frame_dir":     pa.array([frame_dir]        * n, pa.string()),
        "ts_path":       pa.array([ts_path]          * n, pa.string()),
    })


# ---------------------------------------------------------------------------
# Per-dataset loading
# ---------------------------------------------------------------------------

def _load_frame_ts(frame_index_path: Path) -> set[int]:
    """Return set of Unix timestamps that have a visual frame."""
    if not frame_index_path.exists():
        return set()
    tbl = pq.read_table(frame_index_path, columns=["timestamp_unix", "mask_visual"])
    mask = tbl.column("mask_visual").to_pylist()
    ts   = tbl.column("timestamp_unix").to_pylist()
    return {t for t, m in zip(ts, mask) if m}


def process_skippd(refactored_dir: Path,
                   hist_steps: int, horizon: int,
                   visual_window: int) -> tuple[list[dict], list[dict], list[dict]]:
    ts_path = refactored_dir / "skippd" / "timeseries.parquet"
    fi_path = refactored_dir / "skippd" / "frame_index.parquet"
    if not ts_path.exists():
        print("  SKIP SKIPP'D: timeseries.parquet not found (run refactor_skippd.py first).")
        return [], [], []

    tbl    = pq.read_table(ts_path, columns=["timestamp_unix"])
    ts_all = sorted(tbl.column("timestamp_unix").to_pylist())
    frame_ts = _load_frame_ts(fi_path)

    windows = _build_windows(ts_all, entity_id=0,
                             hist_steps=hist_steps, horizon=horizon,
                             frame_ts_set=frame_ts, visual_window=visual_window)
    print(f"  SKIPP'D: {len(ts_all):,} timestamps → {len(windows):,} windows")
    train, val, test = _split_windows(windows)
    print(f"    train={len(train):,}  val={len(val):,}  test={len(test):,}")

    rel_frame = "skippd/frames"
    rel_ts    = "skippd/timeseries.parquet"
    return (
        [_windows_to_table([w], "skippd", rel_frame, rel_ts) for w in train],
        [_windows_to_table([w], "skippd", rel_frame, rel_ts) for w in val],
        [_windows_to_table([w], "skippd", rel_frame, rel_ts) for w in test],
    )


def process_solarnet(refactored_dir: Path,
                     hist_steps: int, horizon: int,
                     visual_window: int) -> tuple[list, list, list]:
    ts_path = refactored_dir / "solarnet" / "targets.parquet"
    fi_path = refactored_dir / "solarnet" / "frame_index.parquet"
    if not ts_path.exists():
        print("  SKIP Solarnet: targets.parquet not found (run refactor_solarnet.py first).")
        return [], [], []

    tbl    = pq.read_table(ts_path, columns=["timestamp_unix"])
    ts_all = sorted(tbl.column("timestamp_unix").to_pylist())
    frame_ts = _load_frame_ts(fi_path)

    windows = _build_windows(ts_all, entity_id=0,
                             hist_steps=hist_steps, horizon=horizon,
                             frame_ts_set=frame_ts, visual_window=visual_window)
    print(f"  Solarnet: {len(ts_all):,} timestamps → {len(windows):,} windows")
    train, val, test = _split_windows(windows)
    print(f"    train={len(train):,}  val={len(val):,}  test={len(test):,}")

    rel_frame = "solarnet/frames"
    rel_ts    = "solarnet/timeseries.parquet"
    return (
        [_windows_to_table([w], "solarnet", rel_frame, rel_ts) for w in train],
        [_windows_to_table([w], "solarnet", rel_frame, rel_ts) for w in val],
        [_windows_to_table([w], "solarnet", rel_frame, rel_ts) for w in test],
    )


def process_goes16_nsrdb(refactored_dir: Path,
                         hist_steps: int, horizon: int,
                         visual_window: int) -> tuple[list, list, list]:
    ts_path = refactored_dir / "goes16_nsrdb" / "timeseries.parquet"
    fi_path = refactored_dir / "goes16_nsrdb" / "frame_index.parquet"
    if not ts_path.exists():
        print("  SKIP GOES-16+NSRDB: timeseries.parquet not found "
              "(run refactor_goes16_nsrdb.py first).")
        return [], [], []

    import pyarrow.compute as pc
    tbl = pq.read_table(ts_path, columns=["entity_id", "timestamp_unix"])

    # Get unique entity IDs
    entity_ids = sorted(set(tbl.column("entity_id").to_pylist()))

    # Frame timestamps (shared across all entities — one frame covers the full crop)
    frame_ts = _load_frame_ts(fi_path)

    train_list, val_list, test_list = [], [], []
    for eid in tqdm(entity_ids, desc="  GOES-16+NSRDB entities"):
        mask   = pc.equal(tbl.column("entity_id"), pa.scalar(eid, pa.int32()))
        subset = tbl.filter(mask)
        ts_all = sorted(subset.column("timestamp_unix").to_pylist())

        windows = _build_windows(ts_all, entity_id=eid,
                                 hist_steps=hist_steps, horizon=horizon,
                                 frame_ts_set=frame_ts,
                                 visual_window=visual_window)
        tr, va, te = _split_windows(windows)
        rel_frame = "goes16_nsrdb/frames"
        rel_ts    = "goes16_nsrdb/timeseries.parquet"
        train_list.extend([_windows_to_table([w], "goes16_nsrdb", rel_frame, rel_ts) for w in tr])
        val_list.extend(  [_windows_to_table([w], "goes16_nsrdb", rel_frame, rel_ts) for w in va])
        test_list.extend( [_windows_to_table([w], "goes16_nsrdb", rel_frame, rel_ts) for w in te])

    print(f"  GOES-16+NSRDB: {len(entity_ids):,} entities "
          f"→ train={len(train_list):,}  val={len(val_list):,}  test={len(test_list):,} windows")
    return train_list, val_list, test_list


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _write_schema(out_dir: Path, hist_steps: int, horizon: int,
                  visual_window: int) -> None:
    schema = {
        "version":       "1.0",
        "description":   "Unified Vision-time FM solar dataset index",
        "window_schema": {
            "dataset":      "string — skippd | solarnet | goes16_nsrdb",
            "entity_id":    "int32 — entity within the dataset",
            "window_start": "int64 — Unix seconds, first historical timestep",
            "window_end":   "int64 — Unix seconds, last horizon timestep",
            "hist_steps":   f"int32 — T = {hist_steps}",
            "horizon":      f"int32 — H = {horizon}",
            "has_visual":   "int8  — 1 if ≥1 frame in the visual refinement window",
            "frame_dir":    "string — relative path to frames/ directory",
            "ts_path":      "string — relative path to timeseries parquet",
        },
        "model_tensor_schema": {
            "Y":                    f"(N, T={hist_steps}, C_target=1)  float32",
            "X_cov":                f"(N, T+H={hist_steps+horizon}, C_cov)  float32",
            "V":                    f"(N, T_v={visual_window}, C, H, W)  float32",
            "mask_target":          f"(N, T={hist_steps}, 1)  int8",
            "mask_visual":          f"(N, T_v={visual_window})  int8",
            "mask_modality_dropout":"(N, 2)  int8",
            "adj_matrix":           "(N, N)  float32",
            "timestamps":           f"(T+H={hist_steps+horizon},)  int64",
            "entity_ids":           "(N,)  int32",
        },
        "datasets": {
            "skippd": {
                "target":     "pv_power (kW normalised)",
                "covariates": [],
                "entities":   1,
                "visual":     "sky camera JPEG",
            },
            "solarnet": {
                "target":     "ghi (W/m²)",
                "covariates": ["air_temp", "relhum", "press",
                               "windsp", "winddir", "max_windsp", "precipitation"],
                "entities":   1,
                "visual":     "sky camera JPEG (extracted from tar.bz2)",
            },
            "goes16_nsrdb": {
                "target":     "GHI (W/m²)",
                "covariates": ["Wind_Speed", "Temperature", "Pressure"],
                "entities":   "up to 4629 (NSRDB 0.5° grid)",
                "visual":     "GOES-16 ABI MCMIP bands C02/C07/C13 (float16 npz)",
            },
        },
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(hist_steps: int = 168, horizon: int = 24, visual_window: int = 6,
         data_root: Path | None = None) -> None:

    refactored = (data_root or PROJECT_ROOT / "data") / "refactored" / "solar"
    out_dir    = (data_root or PROJECT_ROOT / "data") / "refactored" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Unified dataset  —  build windows")
    print(f"  hist_steps    = {hist_steps}")
    print(f"  horizon       = {horizon}")
    print(f"  visual_window = {visual_window}")
    print(f"  src  : {refactored}")
    print(f"  dest : {out_dir}")
    print("=" * 60)

    all_train, all_val, all_test = [], [], []

    for fn, name in [
        (process_skippd,        "SKIPP'D"),
        (process_solarnet,      "Solarnet"),
        (process_goes16_nsrdb,  "GOES-16+NSRDB"),
    ]:
        print(f"\n  {name}…")
        tr, va, te = fn(refactored, hist_steps, horizon, visual_window)
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)

    def _concat_and_write(tables: list, path: Path, label: str) -> None:
        if not tables:
            print(f"  WARNING: no {label} windows — skipping.")
            return
        combined = pa.concat_tables(tables)
        pq.write_table(combined, path, compression="snappy")
        print(f"  {path.name}: {combined.num_rows:,} windows → {path}")

    print()
    _concat_and_write(all_train, out_dir / "train_windows.parquet", "train")
    _concat_and_write(all_val,   out_dir / "val_windows.parquet",   "val")
    _concat_and_write(all_test,  out_dir / "test_windows.parquet",  "test")

    _write_schema(out_dir, hist_steps, horizon, visual_window)
    print(f"  schema.json → {out_dir / 'schema.json'}")
    print("\n[Unified] Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified window index.")
    parser.add_argument("--hist-steps",    type=int,  default=168)
    parser.add_argument("--horizon",       type=int,  default=24)
    parser.add_argument("--visual-window", type=int,  default=6)
    parser.add_argument("--data-root",     type=Path, default=None)
    args = parser.parse_args()
    main(hist_steps=args.hist_steps, horizon=args.horizon,
         visual_window=args.visual_window, data_root=args.data_root)
