"""
Refactor EarthNet2021 → canonical MMTSFM format.

What this script does
──────────────────────
1. Enumerates all NPZ samples across requested splits.
2. Two-pass processing:
   Pass 1 — streaming stats for z-score normalisation (NDVI + ERA5 covariates).
   Pass 2 — writes timeseries.parquet, frame_index.parquet, and per-timestep
             Sentinel-2 frame NPZs.
3. Spatial graph: within-tile k-NN on patch pixel-centre coordinates.
4. Metadata JSON.

Input layout  (data/raw/meteorology/earthnet2021/)
──────────────────────────────────────────────────
  {split}/{MGRS_TILE}/{filename}.npz

  Each NPZ contains:
    highresdynamic : (128, 128, 7, 30)   float16  Sentinel-2 bands + masks
      ch 0 Blue (B02), 1 Green (B03), 2 Red (B04), 3 NIR (B8A),
      ch 4 cloud probability (%), 5 SCL, 6 binary clear-sky mask
    highresstatic  : (128, 128, 1)       float16  static high-res feature
    mesodynamic    : (80, 80, 5, 150)    float16  ERA5 (5-day steps)
      ch 0 precipitation, 1 SLP, 2 2m-temp, 3 cloud cover, 4 solar radiation
    mesostatic     : (80, 80, 1)         float16  static ERA5 feature

Output layout  (data/refactored/meteorology/earthnet2021/)
──────────────────────────────────────────────────────────
  timeseries.parquet   — entity_id, timestamp_unix,
                         ndvi, ndvi_norm,
                         era5_precip, era5_precip_norm, …(×5),
                         mask_target, mask_cov
  frame_index.parquet  — entity_id, timestamp_unix, rel_path, mask_visual
  frames/
    {entity_id:05d}/{t:02d}.npz   — float16 {'frame': (4, 128, 128)}
                                    channels: Blue, Green, Red, NIR
  graph.json           — {nodes, edges, adjacency_matrix}
  metadata.json

Usage
──────
  uv run python scripts/meteorology/refactor_earthnet2021.py
  uv run python scripts/meteorology/refactor_earthnet2021.py \\
      --splits train \\
      --workers 8 \\
      --max-samples 500 \\
      --data-root /custom/data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_ROOT,
    EARTHNET_COVARIATE_COLS,
    EARTHNET_ERA5_RATIO,
    EARTHNET_IMG_H,
    EARTHNET_IMG_W,
    EARTHNET_KNN,
    EARTHNET_MASK_IDX,
    EARTHNET_NIR_IDX,
    EARTHNET_RAW,
    EARTHNET_REFACTORED,
    EARTHNET_RED_IDX,
    EARTHNET_SENTINEL_BAND_INDICES,
    EARTHNET_SENTINEL_BAND_NAMES,
    EARTHNET_SPLITS,
    EARTHNET_T_ERA5,
    EARTHNET_T_SENTINEL,
    EARTHNET_TARGET_COLS,
)

ALL_COLS = EARTHNET_TARGET_COLS + EARTHNET_COVARIATE_COLS


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(stem: str) -> dict:
    """
    Parse EarthNet2021 NPZ stem.
    Format: {TILE}_{START}_{END}_{HR_R0}_{HR_R1}_{HR_C0}_{HR_C1}[_{MR_R0}_{MR_R1}_{MR_C0}_{MR_C1}]
    Returns dict with tile, start, end (datetime), hr_row_center, hr_col_center.
    """
    parts = stem.split("_")
    tile  = parts[0]
    start = datetime.strptime(parts[1], "%Y-%m-%d")
    end   = datetime.strptime(parts[2], "%Y-%m-%d")
    coords = [int(x) for x in parts[3:7]]          # HR pixel bbox
    hr_row_center = (coords[0] + coords[1]) / 2.0
    hr_col_center = (coords[2] + coords[3]) / 2.0
    return {
        "tile":          tile,
        "start":         start,
        "end":           end,
        "hr_row_center": hr_row_center,
        "hr_col_center": hr_col_center,
    }


def sample_timestamps(start: datetime, end: datetime, n: int) -> list[int]:
    """Return n evenly-spaced Unix timestamps (int64) between start and end."""
    t0 = int(start.replace(tzinfo=timezone.utc).timestamp())
    t1 = int(end.replace(tzinfo=timezone.utc).timestamp())
    if n == 1:
        return [t0]
    step = (t1 - t0) / (n - 1)
    return [t0 + round(i * step) for i in range(n)]


# ---------------------------------------------------------------------------
# Per-sample feature extraction
# ---------------------------------------------------------------------------

def extract_features(npz_path: Path) -> dict:
    """
    Load one EarthNet2021 NPZ and return:
      ndvi      : (T_sentinel,)   float32  spatial-mean NDVI
      era5      : (T_sentinel, 5) float32  spatial-mean ERA5 covariates
      mask_tgt  : (T_sentinel,)   int8     1 = >=10 % clear-sky pixels
      mask_cov  : (T_sentinel,)   int8     1 = no NaN in ERA5 window
      bands     : (4, H, W, T_sentinel) float16  B/G/R/NIR frames, clipped [0,1]
    """
    d  = np.load(npz_path, allow_pickle=True)
    hd = d["highresdynamic"].astype(np.float32)          # (H, W, 7, T)
    me = d["mesodynamic"].astype(np.float32)              # (80, 80, 5, T_era5)

    H, W, _, T = hd.shape

    # ── NDVI ─────────────────────────────────────────────────────────────────
    red    = np.clip(hd[:, :, EARTHNET_RED_IDX, :], 0.0, 1.0)   # (H, W, T)
    nir    = np.clip(hd[:, :, EARTHNET_NIR_IDX, :], 0.0, 1.0)   # (H, W, T)
    mask6  = hd[:, :, EARTHNET_MASK_IDX, :]                     # (H, W, T) binary
    valid  = mask6 > 0.5                                          # bool (H, W, T)

    eps = 1e-6
    ndvi_px = (nir - red) / (nir + red + eps)                    # (H, W, T)
    ndvi_px[~valid] = np.nan

    valid_frac  = valid.mean(axis=(0, 1))                         # (T,)
    mask_tgt    = (valid_frac >= 0.1).astype(np.int8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ndvi = np.nanmean(ndvi_px, axis=(0, 1)).astype(np.float32)  # (T,)
    ndvi        = np.where(np.isnan(ndvi), 0.0, ndvi)

    # ── ERA5 ─────────────────────────────────────────────────────────────────
    # me shape: (80, 80, 5, 150)
    era5_spatial = me.mean(axis=(0, 1))                          # (5, 150)
    # Resample 150 → T_sentinel=30 by grouping every ERA5_RATIO=5 steps
    n_groups = EARTHNET_T_SENTINEL
    era5_grouped = era5_spatial.reshape(5, n_groups, EARTHNET_ERA5_RATIO).mean(axis=-1)  # (5, T)
    era5_grouped = era5_grouped.T.astype(np.float32)            # (T, 5)

    has_nan      = np.isnan(era5_grouped).any(axis=1)            # (T,)
    mask_cov     = (~has_nan).astype(np.int8)
    era5_grouped = np.where(np.isnan(era5_grouped), 0.0, era5_grouped)

    # ── Sentinel bands for frames ─────────────────────────────────────────────
    band_indices = EARTHNET_SENTINEL_BAND_INDICES               # [0, 1, 2, 3]
    bands = np.stack(
        [np.clip(hd[:, :, ci, :], 0.0, 1.0) for ci in band_indices],
        axis=0,
    ).astype(np.float16)                                        # (4, H, W, T)
    # Replace NaN (cloud-masked) with 0 before saving
    bands = np.where(np.isnan(bands), np.float16(0.0), bands)

    return {
        "ndvi":     ndvi,       # (T,) float32
        "era5":     era5_grouped,  # (T, 5) float32
        "mask_tgt": mask_tgt,   # (T,) int8
        "mask_cov": mask_cov,   # (T,) int8
        "bands":    bands,      # (4, H, W, T) float16
    }


# ---------------------------------------------------------------------------
# Frame writing
# ---------------------------------------------------------------------------

def _write_frame(args: tuple) -> bool:
    out_path, frame_f16 = args
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, frame=frame_f16)
        return True
    except Exception:
        return False


def write_entity_frames(
    entity_id: int,
    bands: np.ndarray,       # (4, H, W, T) float16
    frames_dir: Path,
    executor: ThreadPoolExecutor,
) -> list[tuple]:
    """Submit frame-write tasks; return list of (future, t, out_path)."""
    futures = []
    for t in range(bands.shape[3]):
        frame    = bands[:, :, :, t]                  # (4, H, W)
        out_path = frames_dir / f"{entity_id:05d}" / f"{t:02d}.npz"
        fut      = executor.submit(_write_frame, (out_path, frame))
        futures.append((fut, t, out_path))
    return futures


# ---------------------------------------------------------------------------
# Step 1+2 — Two-pass main processing
# ---------------------------------------------------------------------------

def process_samples(
    npz_paths: list[Path],
    entity_meta: list[dict],
    out_dir: Path,
    workers: int,
) -> dict:
    """
    Pass 1: accumulate global normalisation stats.
    Pass 2: write timeseries.parquet, frame_index.parquet, frames.

    Returns normalisation stats dict.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: running stats ────────────────────────────────────────────────
    # running[col] = [count, sum, sum_sq]
    running: dict[str, list] = {c: [0, 0.0, 0.0] for c in ALL_COLS}

    print("  Pass 1: streaming normalisation stats …")
    for npz_path in tqdm(npz_paths, desc="  Stats"):
        try:
            feats = extract_features(npz_path)
        except Exception as e:
            print(f"    WARN {npz_path.name}: {e}")
            continue

        # NDVI
        ndvi = feats["ndvi"]  # (T,)
        valid_ndvi = ndvi[feats["mask_tgt"] == 1]
        n = len(valid_ndvi)
        if n > 0:
            running["ndvi"][0] += n
            running["ndvi"][1] += float(valid_ndvi.sum())
            running["ndvi"][2] += float((valid_ndvi ** 2).sum())

        # ERA5
        era5 = feats["era5"]   # (T, 5)
        for ci, col in enumerate(EARTHNET_COVARIATE_COLS):
            vals = era5[feats["mask_cov"] == 1, ci]
            n = len(vals)
            if n > 0:
                running[col][0] += n
                running[col][1] += float(vals.sum())
                running[col][2] += float((vals ** 2).sum())

    # Compute global mean/std
    norm_params: dict[str, tuple[float, float]] = {}
    stats: dict = {"targets": {}, "covariates": {}}
    for col in ALL_COLS:
        n, s, sq = running[col]
        mu  = s / n if n > 0 else 0.0
        var = max(sq / n - mu * mu, 0.0) if n > 0 else 1.0
        std = max(var ** 0.5, 1e-6)
        norm_params[col] = (mu, std)
        bucket = "targets" if col in EARTHNET_TARGET_COLS else "covariates"
        stats[bucket][col] = {"mean": mu, "std": std}
        print(f"    {col}: mean={mu:.4g}  std={std:.4g}")

    # ── Pass 2: normalise + write ────────────────────────────────────────────
    out_schema = pa.schema([
        pa.field("entity_id",      pa.int32()),
        pa.field("timestamp_unix", pa.int64()),
        *[f for col in EARTHNET_TARGET_COLS
          for f in (pa.field(col, pa.float32()), pa.field(f"{col}_norm", pa.float32()))],
        *[f for col in EARTHNET_COVARIATE_COLS
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

    ts_writer = pq.ParquetWriter(out_dir / "timeseries.parquet",   out_schema, compression="snappy")
    fi_writer = pq.ParquetWriter(out_dir / "frame_index.parquet",  fi_schema,  compression="snappy")

    ndvi_mu,  ndvi_std  = norm_params["ndvi"]

    total_rows = 0
    total_frames = 0

    print(f"\n  Pass 2: normalise + write ({len(npz_paths)} samples, {workers} workers) …")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for entity_id, (npz_path, meta) in enumerate(
            tqdm(zip(npz_paths, entity_meta), total=len(npz_paths), desc="  Write")
        ):
            try:
                feats = extract_features(npz_path)
            except Exception as e:
                print(f"    WARN entity {entity_id} {npz_path.name}: {e}")
                continue

            parsed = meta
            timestamps = sample_timestamps(
                parsed["start"], parsed["end"], EARTHNET_T_SENTINEL
            )
            T = EARTHNET_T_SENTINEL

            # ── timeseries rows ──────────────────────────────────────────────
            arrays: dict = {
                "entity_id":      pa.array([entity_id] * T, pa.int32()),
                "timestamp_unix": pa.array(timestamps,      pa.int64()),
            }

            # NDVI
            ndvi     = feats["ndvi"]
            ndvi_n   = ((ndvi - ndvi_mu) / ndvi_std).astype(np.float32)
            arrays["ndvi"]      = pa.array(ndvi,   pa.float32())
            arrays["ndvi_norm"] = pa.array(ndvi_n, pa.float32())

            # ERA5 covariates
            era5 = feats["era5"]  # (T, 5)
            for ci, col in enumerate(EARTHNET_COVARIATE_COLS):
                mu, std = norm_params[col]
                raw    = era5[:, ci]
                normed = ((raw - mu) / std).astype(np.float32)
                arrays[col]           = pa.array(raw,    pa.float32())
                arrays[f"{col}_norm"] = pa.array(normed, pa.float32())

            arrays["mask_target"] = pa.array(feats["mask_tgt"], pa.int8())
            arrays["mask_cov"]    = pa.array(feats["mask_cov"], pa.int8())

            ts_writer.write_table(pa.table(arrays, schema=out_schema))
            total_rows += T

            # ── frame_index rows ─────────────────────────────────────────────
            rel_paths  = [f"frames/{entity_id:05d}/{t:02d}.npz" for t in range(T)]
            fi_arrays  = {
                "entity_id":      pa.array([entity_id] * T, pa.int32()),
                "timestamp_unix": pa.array(timestamps,      pa.int64()),
                "rel_path":       pa.array(rel_paths,       pa.string()),
                "mask_visual":    pa.array(
                    feats["mask_tgt"].tolist(), pa.int8()
                ),
            }
            fi_writer.write_table(pa.table(fi_arrays, schema=fi_schema))

            # ── frame files (async) ──────────────────────────────────────────
            # bands: (4, H, W, T)
            bands = feats["bands"]
            futs = write_entity_frames(entity_id, bands, frames_dir, executor)
            for fut, t, _ in futs:
                if fut.result():
                    total_frames += 1

    ts_writer.close()
    fi_writer.close()

    print(f"  timeseries.parquet  ({total_rows:,} rows)")
    print(f"  frame_index.parquet ({total_frames:,} frames)")

    return stats, norm_params


# ---------------------------------------------------------------------------
# Step 3 — Spatial graph
# ---------------------------------------------------------------------------

def build_graph(entity_meta: list[dict], k: int = EARTHNET_KNN) -> dict:
    """
    Within-tile k-NN graph on patch pixel-centre coordinates.
    Patches from different tiles are not connected.
    """
    n   = len(entity_meta)
    k   = min(k, n - 1)
    adj = [[0.0] * n for _ in range(n)]
    edges: list[dict] = []

    # Group by tile for efficiency
    from collections import defaultdict
    tile_groups: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(entity_meta):
        tile_groups[m["tile"]].append(i)

    for tile, indices in tile_groups.items():
        tile_k = min(k, len(indices) - 1)
        if tile_k <= 0:
            continue
        for i in indices:
            ei = entity_meta[i]
            dists = []
            for j in indices:
                if i == j:
                    continue
                ej  = entity_meta[j]
                dr  = ei["hr_row_center"] - ej["hr_row_center"]
                dc  = ei["hr_col_center"] - ej["hr_col_center"]
                d   = math.sqrt(dr * dr + dc * dc)
                dists.append((d, j))
            dists.sort()
            for d, j in dists[:tile_k]:
                w = 1.0 / max(d, 1e-3)
                edges.append({
                    "src":     i,
                    "dst":     j,
                    "weight":  round(w, 6),
                    "dist_px": round(d, 3),
                    "tile":    tile,
                })
                adj[i][j] = round(w, 6)

    return {
        "nodes":            entity_meta,
        "edges":            edges,
        "adjacency_matrix": adj,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    splits:      list[str]  = ("train",),
    workers:     int        = 4,
    max_samples: int | None = None,
    data_root:   Path | None = None,
    skip_frames: bool       = False,
) -> None:
    raw_dir = (data_root / "raw" / "meteorology" / "earthnet2021") if data_root else EARTHNET_RAW
    out_dir = ((data_root or DATA_ROOT) / "refactored" / "meteorology" / "earthnet2021")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EarthNet2021  --  refactor")
    print(f"  src    : {raw_dir}")
    print(f"  dest   : {out_dir}")
    print(f"  splits : {splits}")
    if max_samples:
        print(f"  max    : {max_samples:,} samples (test mode)")
    print("=" * 60)

    # ── Enumerate NPZ samples ─────────────────────────────────────────────────
    print("\n[1/3] Enumerating samples …")
    npz_paths:   list[Path] = []
    entity_meta: list[dict] = []

    for split in splits:
        split_dir = raw_dir / split
        if not split_dir.exists():
            print(f"  SKIP {split} (not found at {split_dir})")
            continue
        split_paths = sorted(split_dir.rglob("*.npz"))
        print(f"  {split}: {len(split_paths):,} samples")
        for p in split_paths:
            try:
                parsed = parse_filename(p.stem)
            except (ValueError, IndexError) as e:
                print(f"    WARN: cannot parse {p.name}: {e}")
                continue
            parsed["split"] = split
            parsed["name"]  = p.stem
            parsed["start_str"] = parsed["start"].strftime("%Y-%m-%d")
            parsed["end_str"]   = parsed["end"].strftime("%Y-%m-%d")
            # Convert datetimes to strings for JSON serialisation
            npz_paths.append(p)
            entity_meta.append(parsed)

    if not npz_paths:
        raise RuntimeError(f"No .npz files found under {raw_dir} for splits {splits}")

    # Assign sequential entity_ids (already in order)
    for i, m in enumerate(entity_meta):
        m["id"] = i

    if max_samples:
        npz_paths   = npz_paths[:max_samples]
        entity_meta = entity_meta[:max_samples]

    print(f"  Total: {len(npz_paths):,} entities")

    # ── Two-pass processing ───────────────────────────────────────────────────
    print("\n[2/3] Processing samples (two-pass) …")
    stats, norm_params = process_samples(npz_paths, entity_meta, out_dir, workers)

    # ── Graph + Metadata ──────────────────────────────────────────────────────
    print("\n[3/3] Graph + Metadata …")

    # Serialisable entity list (replace datetime objects with strings)
    serial_meta = []
    for m in entity_meta:
        serial_meta.append({
            "id":            m["id"],
            "name":          m["name"],
            "tile":          m["tile"],
            "split":         m["split"],
            "start":         m["start_str"],
            "end":           m["end_str"],
            "hr_row_center": m["hr_row_center"],
            "hr_col_center": m["hr_col_center"],
        })

    graph = build_graph(entity_meta)
    # Replace entity_meta in graph nodes with serialisable version
    graph["nodes"] = serial_meta
    (out_dir / "graph.json").write_text(json.dumps(graph, indent=2))
    print(f"  graph.json  ({len(serial_meta)} nodes, {len(graph['edges'])} edges)")

    frames_dir     = out_dir / "frames"
    frames_disk_gb = (
        sum(p.stat().st_size for p in frames_dir.rglob("*.npz")) / 1e9
        if frames_dir.exists() else 0.0
    )

    metadata = {
        "dataset":           "EarthNet2021",
        "splits":            list(splits),
        "entity_count":      len(serial_meta),
        "entities":          serial_meta,
        "timeseries_file":   "timeseries.parquet",
        "frame_index_file":  "frame_index.parquet",
        "graph_file":        "graph.json",
        "target_cols":       EARTHNET_TARGET_COLS,
        "covariate_cols":    EARTHNET_COVARIATE_COLS,
        "sentinel_bands":    EARTHNET_SENTINEL_BAND_NAMES,
        "sentinel_channels": len(EARTHNET_SENTINEL_BAND_INDICES),
        "img_shape":         [EARTHNET_IMG_H, EARTHNET_IMG_W],
        "t_sentinel":        EARTHNET_T_SENTINEL,
        "t_era5":            EARTHNET_T_ERA5,
        "era5_ratio":        EARTHNET_ERA5_RATIO,
        "frames_disk_gb":    round(frames_disk_gb, 3),
        "normalization":     stats,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  metadata.json -> {out_dir / 'metadata.json'}")
    print(f"  Frames on disk: {frames_disk_gb:.2f} GB")
    print("\n[EarthNet2021] Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactor EarthNet2021 dataset.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=EARTHNET_SPLITS,
        help="Which EarthNet2021 splits to process (default: train).",
    )
    parser.add_argument("--workers",     type=int,  default=4,
                        help="Parallel workers for frame writing.")
    parser.add_argument("--max-samples", type=int,  default=None,
                        help="Cap samples processed (for testing).")
    parser.add_argument("--data-root",   type=Path, default=None,
                        help="Override project data root.")
    parser.add_argument("--skip-frames", action="store_true",
                        help="Skip frame NPZ writing (timeseries + index only).")
    args = parser.parse_args()
    main(
        splits=args.splits,
        workers=args.workers,
        max_samples=args.max_samples,
        data_root=args.data_root,
        skip_frames=args.skip_frames,
    )
