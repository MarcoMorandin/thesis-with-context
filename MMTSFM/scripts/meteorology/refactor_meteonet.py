"""
Refactor MeteoNet: raw CSVs + radar tar.gz archives → training-ready layout.

What this script does
─────────────────────
1. Station observations (NW + SE, 2016–2018 CSVs):
   • Parses per-station 6-min observations (temperature, precipitation, wind,
     humidity, dew point, sea-level pressure).
   • Assigns sequential entity_ids across both regions.
   • Z-score normalises targets and covariates.
   • Writes timeseries.parquet (one row per station-timestamp).

2. Radar grids (rainfall + reflectivity, NW + SE, 2016–2018):
   • Extracts NPZ chunks from monthly tar.gz archives.
   • Subsamples to every --radar-step frames (default=12 → hourly at 5-min native).
   • Converts to float16, saves one NPZ per timestamp: {region}_{YYYYMMDD_HHMM}.npz
   • Channels saved: 'rainfall' (mm/h × 0.1) and 'reflectivity' (dBZ × 0.01).

3. Spatial graph:
   • k-NN graph (k=8) over station lat/lon using Haversine distance.

4. Metadata: entity list, normalization stats, schema description.

Output layout  (data/refactored/meteorology/meteonet/)
───────────────────────────────────────────────────────
  timeseries.parquet   — entity_id, timestamp_unix,
                         t, t_norm, precip, precip_norm,
                         dd, dd_norm, ff, ff_norm, hu, hu_norm,
                         td, td_norm, psl, psl_norm,
                         mask_target, mask_cov
  frames/
    {region}_{YYYYMMDD_HHMM}.npz  — float16 arrays {'rainfall': (H,W), 'reflectivity': (H,W)}
  frame_index.parquet  — entity_id, timestamp_unix, rel_path, mask_visual
  graph.json           — {nodes, edges, adjacency_matrix}
  metadata.json

Usage:
  uv run python scripts/meteorology/refactor_meteonet.py
  uv run python scripts/meteorology/refactor_meteonet.py \\
      --radar-step 12 \\
      --workers 4 \\
      --skip-radar \\
      --data-root /custom/data
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tarfile
from collections import defaultdict
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
    METEONET_COVARIATE_COLS,
    METEONET_KNN,
    METEONET_RAW,
    METEONET_REFACTORED,
    METEONET_REGIONS,
    METEONET_TARGET_COLS,
    METEONET_YEARS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zscore(values: list[float]) -> tuple[list[float], float, float]:
    valid = [v for v in values if not math.isnan(v)]
    mu  = float(sum(valid) / len(valid)) if valid else 0.0
    var = sum((v - mu) ** 2 for v in valid) / len(valid) if valid else 1.0
    std = max(var ** 0.5, 1e-6)
    return [(v - mu) / std if not math.isnan(v) else 0.0 for v in values], mu, std


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _parse_meteonet_ts(date_str: str) -> int:
    """'20160101 00:06' → Unix seconds UTC."""
    dt = datetime.strptime(date_str.strip(), "%Y%m%d %H:%M")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


# ---------------------------------------------------------------------------
# Step 1 — Station timeseries
# ---------------------------------------------------------------------------

def _read_csv_arrow(csv_path: Path) -> pa.Table:
    import pyarrow.csv as pa_csv
    return pa_csv.read_csv(
        csv_path,
        read_options=pa_csv.ReadOptions(block_size=64 * 1024 * 1024),
        convert_options=pa_csv.ConvertOptions(
            column_types={"number_sta": pa.int64(), "date": pa.string()}
        ),
    )


def _ts_array(date_col: pa.Array) -> pa.Array:
    """Parse 'YYYYMMDD HH:MM' string column → int64 Unix seconds UTC."""
    import pyarrow.compute as pc
    ts = pc.strptime(date_col, format="%Y%m%d %H:%M", unit="s")
    return pc.cast(ts, pa.int64())


def process_station_csvs(raw_dir: Path, out_dir: Path) -> tuple[dict, list[dict]]:
    """
    Two-pass streaming approach — avoids holding all 130 M rows in RAM at once.

    Pass 1: read each CSV with pyarrow, accumulate per-column running stats
            (count / sum / sum-of-squares) and build the entity map.
    Pass 2: re-read each CSV, apply global z-score with numpy, write one
            row-group per file via ParquetWriter (constant memory).

    Returns:
      stats       – normalisation dict for metadata
      entity_meta – list of {id, number_sta, region, lat, lon, height_sta}
    """
    import pyarrow.compute as pc

    ALL_COLS = METEONET_TARGET_COLS + METEONET_COVARIATE_COLS

    # Collect CSV paths in order
    csv_paths: list[tuple[str, int, Path]] = []
    for region in METEONET_REGIONS:
        for year in METEONET_YEARS:
            p = raw_dir / f"{region}{year}.csv"
            if p.exists():
                csv_paths.append((region, year, p))
            else:
                print(f"  SKIP {region}{year}.csv (not found)")

    if not csv_paths:
        raise RuntimeError("No station CSVs found in " + str(raw_dir))

    # ── Pass 1: entity map + running stats ────────────────────────────────────
    entity_map:  dict[int, int]  = {}   # number_sta → entity_id
    entity_meta: list[dict]      = []
    next_id = 0
    # col → [count, sum, sum_sq]
    running: dict[str, list[float]] = {c: [0, 0.0, 0.0] for c in ALL_COLS}

    print("  Pass 1: entity map + normalization stats")
    for region, year, csv_path in csv_paths:
        print(f"    {csv_path.name}…")
        tbl = _read_csv_arrow(csv_path)

        # Entity map
        sta_col = tbl.column("number_sta").to_pylist()
        lat_col = tbl.column("lat").to_pylist()
        lon_col = tbl.column("lon").to_pylist()
        hgt_col = tbl.column("height_sta").to_pylist()
        for i, sta in enumerate(sta_col):
            if sta not in entity_map:
                entity_map[sta] = next_id
                entity_meta.append({
                    "id":         next_id,
                    "number_sta": int(sta),
                    "region":     region,
                    "lat":        float(lat_col[i]) if lat_col[i] is not None else 0.0,
                    "lon":        float(lon_col[i]) if lon_col[i] is not None else 0.0,
                    "height_sta": float(hgt_col[i]) if hgt_col[i] is not None else 0.0,
                })
                next_id += 1

        # Running stats via pyarrow compute (no Python list conversion)
        for col in ALL_COLS:
            if col not in tbl.schema.names:
                continue
            arr    = pc.cast(tbl.column(col), pa.float64())
            valid  = pc.drop_null(arr)
            n      = len(valid)
            if n == 0:
                continue
            s      = pc.sum(valid).as_py()
            sq     = pc.sum(pc.multiply(valid, valid)).as_py()
            running[col][0] += n
            running[col][1] += s
            running[col][2] += sq

        del tbl  # free immediately

    # Global mean / std
    norm_params: dict[str, tuple[float, float]] = {}
    stats: dict = {"targets": {}, "covariates": {}}
    for col in ALL_COLS:
        n, s, sq = running[col]
        mu  = s / n if n > 0 else 0.0
        var = max(sq / n - mu * mu, 0.0) if n > 0 else 1.0
        std = max(var ** 0.5, 1e-6)
        norm_params[col] = (mu, std)
        bucket = "targets" if col in METEONET_TARGET_COLS else "covariates"
        stats[bucket][col] = {"mean": mu, "std": std}

    print(f"  {len(entity_meta)} stations indexed across {len(csv_paths)} files")
    for col, (mu, std) in norm_params.items():
        print(f"    {col}: mean={mu:.4g}  std={std:.4g}")

    # ── Pass 2: normalise + stream-write parquet ───────────────────────────────
    # Output schema (fixed column order)
    out_schema = pa.schema([
        pa.field("entity_id",      pa.int32()),
        pa.field("timestamp_unix", pa.int64()),
        *[f for col in METEONET_TARGET_COLS
          for f in (pa.field(col, pa.float32()), pa.field(f"{col}_norm", pa.float32()))],
        *[f for col in METEONET_COVARIATE_COLS
          for f in (pa.field(col, pa.float32()), pa.field(f"{col}_norm", pa.float32()))],
        pa.field("mask_target", pa.int8()),
        pa.field("mask_cov",    pa.int8()),
    ])

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "timeseries.parquet"
    writer    = pq.ParquetWriter(dst, out_schema, compression="snappy")
    total_rows = 0

    print("  Pass 2: normalizing + writing parquet")
    for region, year, csv_path in csv_paths:
        print(f"    {csv_path.name}…")
        tbl = _read_csv_arrow(csv_path)
        n   = tbl.num_rows

        # entity_id column
        entity_id_arr = np.array(
            [entity_map[s] for s in tbl.column("number_sta").to_pylist()],
            dtype=np.int32,
        )

        # timestamp_unix column
        ts_arr = _ts_array(tbl.column("date")).to_numpy().astype(np.int64)

        # Value columns via numpy (fast vectorised zscore)
        arrays: dict[str, pa.Array] = {
            "entity_id":      pa.array(entity_id_arr, pa.int32()),
            "timestamp_unix": pa.array(ts_arr,        pa.int64()),
        }

        mask_target_ok = np.ones(n, dtype=np.int8)
        mask_cov_ok    = np.ones(n, dtype=np.int8)

        for col in METEONET_TARGET_COLS:
            mu, std = norm_params[col]
            if col in tbl.schema.names:
                raw = np.array(
                    pc.cast(tbl.column(col), pa.float64()).to_pylist(),
                    dtype=np.float64,
                )
            else:
                raw = np.zeros(n, dtype=np.float64)
            nan_mask          = np.isnan(raw)
            mask_target_ok   &= (~nan_mask).astype(np.int8)
            raw[nan_mask]     = 0.0
            normed            = ((raw - mu) / std).astype(np.float32)
            arrays[col]       = pa.array(raw.astype(np.float32), pa.float32())
            arrays[f"{col}_norm"] = pa.array(normed, pa.float32())

        for col in METEONET_COVARIATE_COLS:
            mu, std = norm_params[col]
            if col in tbl.schema.names:
                raw = np.array(
                    pc.cast(tbl.column(col), pa.float64()).to_pylist(),
                    dtype=np.float64,
                )
            else:
                raw = np.zeros(n, dtype=np.float64)
            nan_mask        = np.isnan(raw)
            mask_cov_ok    &= (~nan_mask).astype(np.int8)
            raw[nan_mask]   = 0.0
            normed          = ((raw - mu) / std).astype(np.float32)
            arrays[col]     = pa.array(raw.astype(np.float32), pa.float32())
            arrays[f"{col}_norm"] = pa.array(normed, pa.float32())

        arrays["mask_target"] = pa.array(mask_target_ok, pa.int8())
        arrays["mask_cov"]    = pa.array(mask_cov_ok,    pa.int8())

        batch = pa.table(arrays, schema=out_schema)
        writer.write_table(batch)
        total_rows += n
        del tbl, arrays, batch

    writer.close()
    print(f"  timeseries.parquet ({total_rows:,} rows, {len(entity_meta)} stations) → {dst}")

    return stats, entity_meta


# ---------------------------------------------------------------------------
# Step 2 — Radar grids
# ---------------------------------------------------------------------------

def _extract_npz_chunk(npz_bytes: bytes) -> tuple[np.ndarray, list[datetime]]:
    """Load (data, dates) from NPZ bytes. data shape: (T, H, W) int16."""
    d = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
    return d["data"], list(d["dates"])


def _save_radar_frame(args: tuple) -> tuple[int, str, bool]:
    """Save one float16 radar frame NPZ. Returns (ts_unix, rel_path, ok)."""
    ts_unix, out_path, rainfall_arr, refl_arr = args
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            rainfall=rainfall_arr.astype(np.float16),
            reflectivity=refl_arr.astype(np.float16),
        )
        return ts_unix, str(out_path.name), True
    except Exception:
        return ts_unix, "", False


def _iter_radar_tars(
    raw_dir: Path,
    region: str,
    product: str,         # "rainfall" | "reflectivity"
    product_tag: str,     # e.g. "rainfall-NW-2016-01"
) -> list[Path]:
    """Glob monthly tar.gz paths for a given radar product + region."""
    if product == "rainfall":
        base = raw_dir / f"{region}_rainfall_{{}}"
        paths = []
        for year in METEONET_YEARS:
            d = raw_dir / f"{region}_rainfall_{year}"
            if d.exists():
                paths.extend(sorted(d.glob(f"rainfall-{region}-{year}-*.tar.gz")))
        return paths

    # reflectivity: old product 2016-2017, new product 2018
    paths = []
    for year in METEONET_YEARS:
        if year <= 2017:
            d = raw_dir / f"{region}_reflectivity_old_product_{year}"
            glob_pat = f"reflectivity-old-{region}-{year}-*.tar.gz"
        else:
            d = raw_dir / f"{region}_reflectivity_new_product_{year}"
            glob_pat = f"reflectivity-new-{region}-{year}-*.tar.gz"
        if d.exists():
            paths.extend(sorted(d.glob(glob_pat)))
    return paths


def extract_radar_frames(
    raw_dir: Path,
    out_dir: Path,
    radar_step: int,
    workers: int,
    max_frames: int | None,
) -> list[dict]:
    """
    Extract radar NPZ frames (rainfall + reflectivity) for all regions/years.
    radar_step=12 → hourly subsampling (native 5-min resolution).
    Returns frame_index records.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Build aligned rainfall + reflectivity maps per (region, ts) → arrays
    # We iterate region by region to keep memory bounded.
    frame_records: list[dict] = []

    for region in METEONET_REGIONS:
        print(f"\n  Radar: {region}")

        rain_tars = _iter_radar_tars(raw_dir, region, "rainfall", "")
        refl_tars = _iter_radar_tars(raw_dir, region, "reflectivity", "")

        if not rain_tars:
            print(f"    No rainfall archives for {region} — skip")
            continue

        # Build reflectivity lookup: ts_unix → float16 array
        print(f"    Loading reflectivity index ({len(refl_tars)} archives)…")
        refl_lookup: dict[int, np.ndarray] = {}
        for tar_path in tqdm(refl_tars, desc=f"    Refl {region}"):
            try:
                with tarfile.open(tar_path) as tf:
                    for m in tf.getmembers():
                        if not m.name.endswith(".npz"):
                            continue
                        raw = tf.extractfile(m).read()
                        data, dates = _extract_npz_chunk(raw)
                        for i in range(0, len(dates), radar_step):
                            dt = dates[i]
                            ts = int(dt.replace(tzinfo=timezone.utc).timestamp())
                            refl_lookup[ts] = data[i].astype(np.float16)
                            if max_frames and len(refl_lookup) >= max_frames:
                                break
                        if max_frames and len(refl_lookup) >= max_frames:
                            break
            except Exception as e:
                print(f"      WARN: {tar_path.name}: {e}")

        print(f"    {len(refl_lookup):,} reflectivity frames indexed")

        # Process rainfall archives and emit frames
        tasks: list[tuple] = []
        total_rain = 0

        print(f"    Processing rainfall ({len(rain_tars)} archives)…")
        for tar_path in tqdm(rain_tars, desc=f"    Rain {region}"):
            try:
                with tarfile.open(tar_path) as tf:
                    for m in tf.getmembers():
                        if not m.name.endswith(".npz"):
                            continue
                        raw = tf.extractfile(m).read()
                        data, dates = _extract_npz_chunk(raw)
                        for i in range(0, len(dates), radar_step):
                            if max_frames and total_rain >= max_frames:
                                break
                            dt   = dates[i]
                            ts   = int(dt.replace(tzinfo=timezone.utc).timestamp())
                            fname = f"{region}_{dt.strftime('%Y%m%d_%H%M')}.npz"
                            out_path = frames_dir / fname
                            if out_path.exists():
                                frame_records.append({"ts_unix": ts, "rel_path": f"frames/{fname}"})
                                total_rain += 1
                                continue
                            rain_arr = data[i].astype(np.float16)
                            refl_arr = refl_lookup.get(ts, np.zeros((data.shape[1], data.shape[2]), np.float16))
                            tasks.append((ts, out_path, rain_arr, refl_arr))
                            total_rain += 1
                        if max_frames and total_rain >= max_frames:
                            break
            except Exception as e:
                print(f"      WARN: {tar_path.name}: {e}")

        print(f"    Writing {len(tasks):,} new frames (radar_step={radar_step})…")
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futs = [exe.submit(_save_radar_frame, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"    Frames {region}"):
                ts_val, fname, ok = fut.result()
                if ok:
                    frame_records.append({"ts_unix": ts_val, "rel_path": f"frames/{fname}"})

        # Free memory
        del refl_lookup

    return frame_records


# ---------------------------------------------------------------------------
# Step 3 — Spatial graph
# ---------------------------------------------------------------------------

def build_station_graph(entity_meta: list[dict], k: int = METEONET_KNN) -> dict:
    """k-NN spatial graph (Haversine) over station coordinates."""
    n   = len(entity_meta)
    k   = min(k, n - 1)
    adj = [[0.0] * n for _ in range(n)]
    edges: list[dict] = []

    for i, ei in enumerate(entity_meta):
        dists = []
        for j, ej in enumerate(entity_meta):
            if i == j:
                continue
            d = _haversine_km(ei["lat"], ei["lon"], ej["lat"], ej["lon"])
            dists.append((d, j))
        dists.sort()
        for d, j in dists[:k]:
            w = 1.0 / max(d, 1e-3)
            edges.append({"src": i, "dst": j, "weight": round(w, 6), "dist_km": round(d, 3)})
            adj[i][j] = round(w, 6)

    return {
        "nodes":             entity_meta,
        "edges":             edges,
        "adjacency_matrix":  adj,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    radar_step: int        = 12,
    workers:    int        = 4,
    skip_radar: bool       = False,
    skip_stations: bool    = False,
    data_root:  Path | None = None,
    max_frames: int | None  = None,
) -> None:

    raw_dir = (data_root / "raw" / "meteorology" / "meteonet") if data_root else METEONET_RAW
    out_dir = ((data_root or DATA_ROOT) / "refactored" / "meteorology" / "meteonet")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MeteoNet  --  refactor")
    print(f"  src : {raw_dir}")
    print(f"  dest: {out_dir}")
    print(f"  radar_step: {radar_step} ({radar_step * 5}-min intervals)")
    if max_frames:
        print(f"  max_frames: {max_frames:,}")
    print("=" * 60)

    # ── Step 1: Stations ──────────────────────────────────────────────────────
    stats:       dict      = {}
    entity_meta: list[dict] = []

    if not skip_stations:
        print("\n[1/3] Station observations")
        stats, entity_meta = process_station_csvs(raw_dir, out_dir)
    else:
        print("[1/3] --skip-stations: loading entity_meta from existing metadata.json")
        meta_path = out_dir / "metadata.json"
        if meta_path.exists():
            entity_meta = json.loads(meta_path.read_text())["entities"]
        else:
            print("  WARNING: no metadata.json — graph and metadata will be incomplete.")

    # ── Step 2: Radar ─────────────────────────────────────────────────────────
    frame_records: list[dict] = []

    if not skip_radar:
        print("\n[2/3] Radar grids (rainfall + reflectivity)")
        frame_records = extract_radar_frames(raw_dir, out_dir, radar_step, workers, max_frames)

        frame_records.sort(key=lambda r: r["ts_unix"])
        fi_table = pa.table({
            "entity_id":      pa.array([-1] * len(frame_records), pa.int32()),
            "timestamp_unix": pa.array([r["ts_unix"]  for r in frame_records], pa.int64()),
            "rel_path":       pa.array([r["rel_path"] for r in frame_records], pa.string()),
            "mask_visual":    pa.array([1]            * len(frame_records),    pa.int8()),
        })
        fi_path = out_dir / "frame_index.parquet"
        pq.write_table(fi_table, fi_path, compression="snappy")
        print(f"  frame_index.parquet ({len(frame_records):,} frames) → {fi_path}")
    else:
        print("[2/3] --skip-radar: skipped")

    # ── Step 3: Graph + Metadata ──────────────────────────────────────────────
    print("\n[3/3] Graph + Metadata")
    graph = build_station_graph(entity_meta) if entity_meta else {"nodes": [], "edges": [], "adjacency_matrix": []}
    (out_dir / "graph.json").write_text(json.dumps(graph, indent=2))
    print(f"  graph.json ({len(entity_meta)} nodes, {len(graph['edges'])} edges)")

    frames_dir     = out_dir / "frames"
    frames_disk_gb = (
        sum(p.stat().st_size for p in frames_dir.glob("*.npz")) / 1e9
        if frames_dir.exists() else 0.0
    )
    metadata = {
        "dataset":          "MeteoNet (France NW + SE)",
        "regions":          METEONET_REGIONS,
        "years":            METEONET_YEARS,
        "entity_count":     len(entity_meta),
        "entities":         entity_meta,
        "timeseries_file":  "timeseries.parquet",
        "frame_index_file": "frame_index.parquet",
        "graph_file":       "graph.json",
        "target_cols":      METEONET_TARGET_COLS,
        "covariate_cols":   METEONET_COVARIATE_COLS,
        "radar_step":       radar_step,
        "radar_step_min":   radar_step * 5,
        "radar_shape":      [565, 784],
        "radar_channels":   ["rainfall", "reflectivity"],
        "frames_extracted": len(frame_records),
        "frames_disk_gb":   round(frames_disk_gb, 3),
        "normalization":    stats,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  metadata.json → {out_dir / 'metadata.json'}")
    print(f"  Radar frames on disk: {frames_disk_gb:.2f} GB")
    print("\n[MeteoNet] Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactor MeteoNet dataset.")
    parser.add_argument("--radar-step",     type=int,  default=12,
                        help="Subsample every N-th 5-min radar frame (default 12 = hourly).")
    parser.add_argument("--workers",        type=int,  default=4)
    parser.add_argument("--skip-radar",     action="store_true")
    parser.add_argument("--skip-stations",  action="store_true")
    parser.add_argument("--data-root",      type=Path, default=None)
    parser.add_argument("--max-frames",     type=int,  default=None,
                        help="Cap radar frames extracted per region (testing).")
    args = parser.parse_args()
    main(
        radar_step=args.radar_step,
        workers=args.workers,
        skip_radar=args.skip_radar,
        skip_stations=args.skip_stations,
        data_root=args.data_root,
        max_frames=args.max_frames,
    )
