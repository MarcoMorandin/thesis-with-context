"""
Refactor ERA5 EU (2020-2021) → canonical MMTSFM format.

What this script does
──────────────────────
0. Extracts the CDS ZIP (era5_eu_2020_2021.nc) to data/raw/meteorology/era5/extracted/.
1. Loads lat / lon / time arrays; maps each grid point to an entity_id.
2. Two-pass processing:
   Pass 1 — load each variable array to compute global z-score params.
   Pass 2 — stream-write timeseries.parquet + frame NPZs in time chunks.
3. Grid graph: 8-connectivity with inverse-distance weights.
4. Metadata JSON.

Input
──────
  data/raw/meteorology/era5/era5_eu_2020_2021.nc   (ZIP from Copernicus CDS)
  Contains:
    data_stream-oper_stepType-instant.nc  — t2m, u10, v10, cape, tcwv
    data_stream-oper_stepType-accum.nc    — ssrd

  Dimensions: valid_time(2924), latitude(149), longitude(241)
  Period: 2020-01-01 to 2021-12-31, 6-hourly

Output layout  (data/refactored/meteorology/era5_eu/)
──────────────────────────────────────────────────────
  timeseries.parquet   — entity_id, timestamp_unix,
                         t2m, t2m_norm, ssrd, ssrd_norm,
                         u10, u10_norm, v10, v10_norm,
                         cape, cape_norm, tcwv, tcwv_norm,
                         mask_target, mask_cov
                         (written time-first; 35,909 entities × 2924 steps = ~105 M rows)
  frame_index.parquet  — entity_id=-1, timestamp_unix, rel_path, mask_visual
  frames/
    ERA5EU_{YYYYMMDD_HH}.npz  — float16 {'frame': (6, 149, 241)}
                                 channels: t2m, ssrd, u10, v10, cape, tcwv
  graph.json           — {nodes, edges}   (no adjacency_matrix — 35,909² is too large)
  metadata.json

Usage
──────
  uv run python scripts/meteorology/refactor_era5_eu.py
  uv run python scripts/meteorology/refactor_era5_eu.py \\
      --skip-frames \\
      --workers 4 \\
      --chunk 200 \\
      --data-root /custom/data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_ROOT,
    ERA5_EU_ACCUM_FILE,
    ERA5_EU_COVARIATE_COLS,
    ERA5_EU_EXTRACTED,
    ERA5_EU_FRAME_CHANNELS,
    ERA5_EU_INSTANT_FILE,
    ERA5_EU_RAW,
    ERA5_EU_REFACTORED,
    ERA5_EU_TARGET_COLS,
    ERA5_EU_TEMPORAL_STEP_H,
    ERA5_EU_WRITE_CHUNK,
    ERA5_EU_ZIP,
)

ALL_COLS = ERA5_EU_TARGET_COLS + ERA5_EU_COVARIATE_COLS

# NetCDF variable names by file type
_INSTANT_VARS = ["t2m", "u10", "v10", "cape", "tcwv"]
_ACCUM_VARS   = ["ssrd"]


# ---------------------------------------------------------------------------
# Step 0 — Extract ZIP
# ---------------------------------------------------------------------------

def ensure_extracted(zip_path: Path, out_dir: Path) -> None:
    """Unzip era5 CDS archive if not already extracted."""
    expected = [out_dir / ERA5_EU_INSTANT_FILE, out_dir / ERA5_EU_ACCUM_FILE]
    if all(p.exists() for p in expected):
        print(f"  Already extracted: {out_dir}")
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"ERA5 ZIP not found: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path.name} ({zip_path.stat().st_size / 1e9:.2f} GB) ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print(f"  Extracted to {out_dir}")


# ---------------------------------------------------------------------------
# Step 1 — Load grid metadata
# ---------------------------------------------------------------------------

def load_grid_meta(extract_dir: Path) -> dict:
    """
    Return lat (H,), lon (W,), timestamps (T,) arrays and entity metadata list.
    entity_id = row * n_lon + col.
    """
    import netCDF4 as nc
    d = nc.Dataset(extract_dir / ERA5_EU_INSTANT_FILE)
    lat = np.array(d.variables["latitude"][:], dtype=np.float32)   # (H,) descending
    lon = np.array(d.variables["longitude"][:], dtype=np.float32)  # (W,)
    ts  = np.array(d.variables["valid_time"][:], dtype=np.int64)   # (T,) Unix s
    d.close()

    H, W = len(lat), len(lon)
    n = H * W

    entity_meta = []
    for r in range(H):
        for c in range(W):
            entity_meta.append({
                "id":  r * W + c,
                "lat": float(lat[r]),
                "lon": float(lon[c]),
                "row": r,
                "col": c,
            })

    return {
        "lat": lat,
        "lon": lon,
        "timestamps": ts,
        "H": H,
        "W": W,
        "n_entities": n,
        "entity_meta": entity_meta,
    }


# ---------------------------------------------------------------------------
# Step 2 — Two-pass stats + write
# ---------------------------------------------------------------------------

def _compute_var_stats(arr: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) over all finite values of arr."""
    valid = arr[np.isfinite(arr)]
    mu  = float(valid.mean())
    std = max(float(valid.std()), 1e-6)
    return mu, std


def _save_frame(args: tuple) -> bool:
    out_path, frame_f16 = args
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, frame=frame_f16)
        return True
    except Exception:
        return False


def process_era5(
    extract_dir: Path,
    out_dir:     Path,
    grid:        dict,
    workers:     int,
    chunk_size:  int,
    skip_frames: bool,
) -> dict:
    """
    Pass 1: compute per-variable normalisation stats.
    Pass 2: stream-write timeseries.parquet + frames.
    Returns normalisation stats dict.
    """
    import netCDF4 as nc

    H, W  = grid["H"], grid["W"]
    ts    = grid["timestamps"]    # (T,) int64
    T     = len(ts)
    n_ent = grid["n_entities"]

    # ── Pass 1: normalisation stats ──────────────────────────────────────────
    print("  Pass 1: normalisation stats ...")
    norm_params: dict[str, tuple[float, float]] = {}
    stats: dict = {"targets": {}, "covariates": {}}

    # Load instant file
    d_inst = nc.Dataset(extract_dir / ERA5_EU_INSTANT_FILE)
    for var in _INSTANT_VARS:
        if var not in ALL_COLS:
            continue
        arr = np.array(d_inst.variables[var][:], dtype=np.float32)  # (T, H, W)
        mu, std = _compute_var_stats(arr)
        norm_params[var] = (mu, std)
        bucket = "targets" if var in ERA5_EU_TARGET_COLS else "covariates"
        stats[bucket][var] = {"mean": mu, "std": std}
        print(f"    {var}: mean={mu:.4g}  std={std:.4g}")
        del arr
    d_inst.close()

    # Load accum file
    d_accum = nc.Dataset(extract_dir / ERA5_EU_ACCUM_FILE)
    for var in _ACCUM_VARS:
        if var not in ALL_COLS:
            continue
        arr = np.array(d_accum.variables[var][:], dtype=np.float32)  # (T, H, W)
        arr = np.maximum(arr, 0.0)                # ssrd is non-negative by definition
        mu, std = _compute_var_stats(arr)
        norm_params[var] = (mu, std)
        bucket = "targets" if var in ERA5_EU_TARGET_COLS else "covariates"
        stats[bucket][var] = {"mean": mu, "std": std}
        print(f"    {var}: mean={mu:.4g}  std={std:.4g}")
        del arr
    d_accum.close()

    # ── Pass 2: stream-write ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if not skip_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    out_schema = pa.schema([
        pa.field("entity_id",      pa.int32()),
        pa.field("timestamp_unix", pa.int64()),
        *[f for col in ERA5_EU_TARGET_COLS
          for f in (pa.field(col, pa.float32()), pa.field(f"{col}_norm", pa.float32()))],
        *[f for col in ERA5_EU_COVARIATE_COLS
          for f in (pa.field(col, pa.float32()), pa.field(f"{col}_norm", pa.float32()))],
        pa.field("mask_target", pa.int8()),
        pa.field("mask_cov",    pa.int8()),
    ])

    fi_schema = pa.schema([
        pa.field("entity_id",      pa.int32()),
        pa.field("timestamp_unix", pa.int64()),
        pa.field("rel_path",       pa.string()),
        pa.field("mask_visual",    pa.int8()),
    ])

    ts_writer = pq.ParquetWriter(out_dir / "timeseries.parquet",  out_schema, compression="snappy")
    fi_writer = pq.ParquetWriter(out_dir / "frame_index.parquet", fi_schema,  compression="snappy")

    total_rows   = 0
    total_frames = 0
    frame_records: list[dict] = []

    # Pre-compute flat entity_id array for all grid points (constant per timestep)
    flat_ids = np.arange(n_ent, dtype=np.int32)   # (H*W,)

    print(f"\n  Pass 2: stream-write ({T} timesteps, chunk={chunk_size}, {workers} workers) ...")

    # Open NetCDF files for streaming
    d_inst  = nc.Dataset(extract_dir / ERA5_EU_INSTANT_FILE)
    d_accum = nc.Dataset(extract_dir / ERA5_EU_ACCUM_FILE)

    n_chunks = math.ceil(T / chunk_size)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk_idx in tqdm(range(n_chunks), desc="  Chunks"):
                t0 = chunk_idx * chunk_size
                t1 = min(t0 + chunk_size, T)
                chunk_ts = ts[t0:t1]              # (C,) Unix timestamps
                C = t1 - t0

                # Load variable arrays for this chunk: (C, H, W)
                var_arrays: dict[str, np.ndarray] = {}
                for var in _INSTANT_VARS:
                    raw = np.array(d_inst.variables[var][t0:t1, :, :], dtype=np.float32)
                    var_arrays[var] = raw
                for var in _ACCUM_VARS:
                    raw = np.array(d_accum.variables[var][t0:t1, :, :], dtype=np.float32)
                    var_arrays[var] = np.maximum(raw, 0.0)   # ssrd >= 0

                # Build flat column arrays for the whole chunk: (C * n_ent,)
                n_rows = C * n_ent

                entity_id_col = np.tile(flat_ids, C).astype(np.int32)
                ts_col        = np.repeat(chunk_ts, n_ent).astype(np.int64)

                arrays: dict = {
                    "entity_id":      pa.array(entity_id_col, pa.int32()),
                    "timestamp_unix": pa.array(ts_col,        pa.int64()),
                }

                mask_target_ok = np.ones(n_rows, dtype=np.int8)
                mask_cov_ok    = np.ones(n_rows, dtype=np.int8)

                for col in ERA5_EU_TARGET_COLS:
                    mu, std = norm_params[col]
                    raw    = var_arrays[col].reshape(-1).astype(np.float32)
                    nan_m  = ~np.isfinite(raw)
                    mask_target_ok = np.where(nan_m, np.int8(0), mask_target_ok)
                    raw[nan_m] = 0.0
                    normed = ((raw - mu) / std).astype(np.float32)
                    arrays[col]           = pa.array(raw,    pa.float32())
                    arrays[f"{col}_norm"] = pa.array(normed, pa.float32())

                for col in ERA5_EU_COVARIATE_COLS:
                    mu, std = norm_params[col]
                    raw    = var_arrays[col].reshape(-1).astype(np.float32)
                    nan_m  = ~np.isfinite(raw)
                    mask_cov_ok = np.where(nan_m, np.int8(0), mask_cov_ok)
                    raw[nan_m] = 0.0
                    normed = ((raw - mu) / std).astype(np.float32)
                    arrays[col]           = pa.array(raw,    pa.float32())
                    arrays[f"{col}_norm"] = pa.array(normed, pa.float32())

                arrays["mask_target"] = pa.array(mask_target_ok, pa.int8())
                arrays["mask_cov"]    = pa.array(mask_cov_ok,    pa.int8())

                ts_writer.write_table(pa.table(arrays, schema=out_schema))
                total_rows += n_rows

                # ── Frame NPZs (one per timestep in chunk) ────────────────────
                if not skip_frames:
                    frame_tasks = []
                    for i in range(C):
                        t_unix = int(chunk_ts[i])
                        dt_str = datetime.fromtimestamp(t_unix, tz=timezone.utc).strftime("%Y%m%d_%H")
                        fname  = f"ERA5EU_{dt_str}.npz"
                        out_p  = frames_dir / fname

                        if not out_p.exists():
                            channels = [var_arrays[var][i].astype(np.float16)
                                        for var in ERA5_EU_FRAME_CHANNELS]
                            frame = np.stack(channels, axis=0)   # (6, H, W) float16
                            frame_tasks.append((out_p, frame))

                        frame_records.append({
                            "ts_unix":  t_unix,
                            "rel_path": f"frames/{fname}",
                        })

                    futs = [executor.submit(_save_frame, t) for t in frame_tasks]
                    for fut in as_completed(futs):
                        if fut.result():
                            total_frames += 1

                del var_arrays

    finally:
        d_inst.close()
        d_accum.close()
        # Always finalise parquet files — even on early exit / KeyboardInterrupt
        frame_records.sort(key=lambda r: r["ts_unix"])
        fi_table = pa.table({
            "entity_id":      pa.array([-1] * len(frame_records), pa.int32()),
            "timestamp_unix": pa.array([r["ts_unix"]  for r in frame_records], pa.int64()),
            "rel_path":       pa.array([r["rel_path"] for r in frame_records], pa.string()),
            "mask_visual":    pa.array([1]             * len(frame_records),   pa.int8()),
        }, schema=fi_schema)
        fi_writer.write_table(fi_table)
        ts_writer.close()
        fi_writer.close()

    print(f"  timeseries.parquet  ({total_rows:,} rows)")
    print(f"  frame_index.parquet ({len(frame_records):,} records)")
    if not skip_frames:
        print(f"  frames written:     {total_frames:,}")

    return stats


# ---------------------------------------------------------------------------
# Step 3 — Grid graph (8-connectivity)
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def build_grid_graph(grid: dict) -> dict:
    """
    8-connectivity graph over the ERA5 EU grid.
    Adjacency matrix omitted (35,909² entries would be ~10 GB).
    """
    lat = grid["lat"]
    lon = grid["lon"]
    H, W = grid["H"], grid["W"]
    entity_meta = grid["entity_meta"]

    edges: list[dict] = []
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for r in range(H):
        for c in range(W):
            i = r * W + c
            for dr, dc in dirs:
                nr, nc_ = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc_ < W:
                    j   = nr * W + nc_
                    d   = _haversine_km(float(lat[r]), float(lon[c]),
                                        float(lat[nr]), float(lon[nc_]))
                    w   = 1.0 / max(d, 1e-3)
                    edges.append({
                        "src":     i,
                        "dst":     j,
                        "weight":  round(w, 6),
                        "dist_km": round(d, 3),
                    })

    return {
        "nodes": entity_meta,
        "edges": edges,
        # adjacency_matrix omitted — reconstruct from edges if needed
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    workers:       int         = 4,
    chunk_size:    int         = ERA5_EU_WRITE_CHUNK,
    skip_frames:   bool        = False,
    max_timesteps: int | None  = None,
    data_root:     Path | None = None,
) -> None:
    raw_dir  = (data_root / "raw"        / "meteorology" / "era5")    if data_root else ERA5_EU_RAW
    out_dir  = (data_root / "refactored" / "meteorology" / "era5_eu") if data_root else ERA5_EU_REFACTORED
    zip_path = raw_dir / "era5_eu_2020_2021.nc"
    ext_dir  = raw_dir / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ERA5 EU (2020-2021)  --  refactor")
    print(f"  src    : {zip_path}")
    print(f"  dest   : {out_dir}")
    print(f"  workers: {workers}   chunk: {chunk_size}")
    print("=" * 60)

    # ── Step 0: Extract ZIP ───────────────────────────────────────────────────
    print("\n[1/4] Extracting ZIP ...")
    ensure_extracted(zip_path, ext_dir)

    # ── Step 1: Grid metadata ─────────────────────────────────────────────────
    print("\n[2/4] Loading grid metadata ...")
    grid = load_grid_meta(ext_dir)
    H, W = grid["H"], grid["W"]
    T    = len(grid["timestamps"])
    if max_timesteps:
        grid["timestamps"] = grid["timestamps"][:max_timesteps]
        print(f"  [--max-timesteps {max_timesteps}] truncated for testing")

    T = len(grid["timestamps"])
    print(f"  Grid: {H} lat x {W} lon = {grid['n_entities']:,} entities")
    print(f"  Time: {T} steps  ({ERA5_EU_TEMPORAL_STEP_H}h cadence)")
    t0_dt = datetime.fromtimestamp(int(grid["timestamps"][0]),  tz=timezone.utc)
    t1_dt = datetime.fromtimestamp(int(grid["timestamps"][-1]), tz=timezone.utc)
    print(f"  Period: {t0_dt.date()} .. {t1_dt.date()}")

    # ── Step 2: Two-pass processing ───────────────────────────────────────────
    print("\n[3/4] Two-pass processing ...")
    stats = process_era5(ext_dir, out_dir, grid, workers, chunk_size, skip_frames)

    # ── Step 3: Graph + Metadata ──────────────────────────────────────────────
    print("\n[4/4] Graph + Metadata ...")
    print("  Building grid graph (8-connectivity) ...")
    graph = build_grid_graph(grid)
    (out_dir / "graph.json").write_text(json.dumps(graph, indent=2))
    print(f"  graph.json  ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)")

    frames_dir     = out_dir / "frames"
    frames_disk_gb = (
        sum(p.stat().st_size for p in frames_dir.glob("*.npz")) / 1e9
        if frames_dir.exists() else 0.0
    )

    metadata = {
        "dataset":           "ERA5 EU (2020-2021)",
        "period":            f"{t0_dt.date()} / {t1_dt.date()}",
        "temporal_step_h":   ERA5_EU_TEMPORAL_STEP_H,
        "grid_shape":        [H, W],
        "lat_range":         [float(grid["lat"][-1]), float(grid["lat"][0])],
        "lon_range":         [float(grid["lon"][0]),  float(grid["lon"][-1])],
        "entity_count":      grid["n_entities"],
        "timestep_count":    T,
        "timeseries_file":   "timeseries.parquet",
        "frame_index_file":  "frame_index.parquet",
        "graph_file":        "graph.json",
        "target_cols":       ERA5_EU_TARGET_COLS,
        "covariate_cols":    ERA5_EU_COVARIATE_COLS,
        "frame_channels":    ERA5_EU_FRAME_CHANNELS,
        "frame_shape":       [len(ERA5_EU_FRAME_CHANNELS), H, W],
        "frames_disk_gb":    round(frames_disk_gb, 3),
        "normalization":     stats,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  metadata.json -> {out_dir / 'metadata.json'}")
    print(f"  Frames on disk: {frames_disk_gb:.2f} GB")
    print("\n[ERA5 EU] Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactor ERA5 EU dataset.")
    parser.add_argument("--workers",     type=int,  default=4,
                        help="Parallel workers for frame writing.")
    parser.add_argument("--chunk",       type=int,  default=ERA5_EU_WRITE_CHUNK,
                        help=f"Timesteps per streaming write chunk (default {ERA5_EU_WRITE_CHUNK}).")
    parser.add_argument("--skip-frames",    action="store_true",
                        help="Skip spatial frame NPZ writing.")
    parser.add_argument("--max-timesteps", type=int, default=None,
                        help="Cap timesteps processed (for testing).")
    parser.add_argument("--data-root",     type=Path, default=None,
                        help="Override project data root.")
    args = parser.parse_args()
    main(
        workers=args.workers,
        chunk_size=args.chunk,
        skip_frames=args.skip_frames,
        max_timesteps=args.max_timesteps,
        data_root=args.data_root,
    )
