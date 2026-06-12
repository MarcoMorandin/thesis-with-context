"""
Refactor Solarnet (Folsom, CA): raw CSVs + tar.bz2 archives -> cleaned layout.

What this script does
─────────────────────
1. Reads irradiance + weather CSVs, z-score normalises targets and covariates,
   merges on timestamp_unix, writes a single timeseries.parquet.
2. Extracts sky-camera images from the three tar.bz2 archives (2014–2016),
   resizes to --img-size × --img-size, saves as JPEG.
3. Writes frame_index.parquet aligning each frame to its UTC timestamp.
4. Writes metadata.json with normalization stats.

Output layout  (data/refactored/solar/solarnet/)
─────────────────────────────────────────────────
  timeseries.parquet   — entity_id, timestamp_unix,
                         ghi, ghi_norm, dni, dni_norm, dhi, dhi_norm,
                         air_temp, air_temp_norm, relhum, relhum_norm,
                         press, press_norm, windsp, windsp_norm,
                         winddir, winddir_norm, max_windsp, max_windsp_norm,
                         precipitation, precipitation_norm,
                         mask_target, mask_cov
  frames/
    {YYYYMMDD_HHMM}.jpg
  frame_index.parquet  — entity_id, timestamp_unix, rel_path, mask_visual
  metadata.json

Expected size: ~5–8 GB at img_size=224  (~150 K daylight frames × ~40 KB JPEG).

Notes on sky-image filename format
────────────────────────────────────
Solarnet archives use one of two naming schemes depending on year.
This script tries both and falls back to a sequential counter if neither parses.
  Scheme A: IMG_YYYYMMDD_HHMMSS.jpg
  Scheme B: YYYYMMDD/HHMMSS.jpg  (subdirectory per day)

Usage:
  python scripts/solar/refactor_solarnet.py [--img-size 224] [--quality 90]
                                            [--data-root PATH] [--max-frames N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROJECT_ROOT,
    SOLARNET_COVARIATE_COLS,
    SOLARNET_RAW,
    SOLARNET_TARGET_COLS,
)

_ENTITY_ID = 0

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

# Supported filename patterns (tried in order)
_TS_PATTERNS: list[tuple[str, str]] = [
    # IMG_YYYYMMDD_HHMMSS.jpg
    (r"IMG_(\d{8})_(\d{6})", "%Y%m%d %H%M%S"),
    # YYYYMMDD_HHMMSS.jpg
    (r"(\d{8})_(\d{6})", "%Y%m%d %H%M%S"),
    # YYYYMMDD/HHMMSS.jpg  (subdirectory + stem)
    (r"(\d{8})[/\\](\d{6})", "%Y%m%d %H%M%S"),
    # YYYYMMDD_HHMM.jpg (no seconds)
    (r"(\d{8})_(\d{4})(?!\d)", "%Y%m%d %H%M"),
]


def _parse_ts_from_path(path_in_archive: str) -> int | None:
    """Parse Unix timestamp (seconds UTC) from archive member path."""
    for pattern, fmt in _TS_PATTERNS:
        m = re.search(pattern, path_in_archive)
        if m:
            try:
                dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", fmt)
                dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# CSV normalization
# ---------------------------------------------------------------------------

def _zscore(values: list[float | None]) -> tuple[list[float], float, float]:
    valid = [v for v in values if v is not None]
    mu  = float(sum(valid) / len(valid)) if valid else 0.0
    variance = sum((v - mu) ** 2 for v in valid) / len(valid) if valid else 1.0
    std = max(variance ** 0.5, 1e-6)
    normed = [(v - mu) / std if v is not None else 0.0 for v in values]
    return normed, mu, std


def _read_csv(path: Path) -> pa.Table:
    return pa_csv.read_csv(path,
                           read_options=pa_csv.ReadOptions(encoding="utf-8"),
                           parse_options=pa_csv.ParseOptions(delimiter=","))


def _parse_folsom_ts(table: pa.Table) -> pa.Array:
    """Convert timeStamp string column to Unix seconds."""
    raw = table.column("timeStamp")

    # If already a timestamp, cast to int64 directly
    if pa.types.is_timestamp(raw.type):
        return pc.cast(raw, pa.int64())

    # Otherwise, parse from string
    ts  = pc.strptime(raw, format="%Y-%m-%d %H:%M:%S", unit="s")
    return pc.cast(ts, pa.int64())


def process_timeseries(solarnet_raw: Path, out_dir: Path) -> dict:
    """
    Write a single timeseries.parquet merging targets + covariates.
    Returns normalization stats dict.

    Output schema (matches SKIPPD / GOES-16 convention):
      entity_id, timestamp_unix,
      {target_cols}, {target_cols}_norm,
      {cov_cols}, {cov_cols}_norm,
      mask_target, mask_cov
    """
    stats: dict = {}

    # ── Targets (irradiance) ─────────────────────────────────────────────────
    irr_path = solarnet_raw / "Folsom_irradiance.csv"
    print(f"  Reading {irr_path.name}…")
    irr = _read_csv(irr_path)
    ts_irr = _parse_folsom_ts(irr).to_pylist()
    n_irr  = irr.num_rows
    schema_lower = {c.lower(): c for c in irr.schema.names}

    target_arrays: dict[str, list] = {}
    target_norm_arrays: dict[str, list] = {}
    target_stats: dict[str, dict] = {}
    for col in SOLARNET_TARGET_COLS:
        src = schema_lower.get(col.lower(), col)
        vals = irr.column(src).to_pylist() if src in irr.schema.names else [0.0] * n_irr
        normed, mu, std = _zscore(vals)
        target_arrays[col]            = [float(v) if v is not None else 0.0 for v in vals]
        target_norm_arrays[f"{col}_norm"] = normed
        target_stats[col]             = {"mean": mu, "std": std}
    stats["targets"] = target_stats

    # ── Covariates (weather) ─────────────────────────────────────────────────
    wx_path = solarnet_raw / "Folsom_weather.csv"
    cov_by_ts:  dict[int, dict[str, float]] = {}  # ts → {col: val}
    cov_stats: dict[str, dict] = {}

    if wx_path.exists():
        print(f"  Reading {wx_path.name}…")
        wx = _read_csv(wx_path)
        ts_wx = _parse_folsom_ts(wx).to_pylist()
        schema_lower_wx = {c.lower(): c for c in wx.schema.names}

        raw_cov: dict[str, list] = {}
        for col in SOLARNET_COVARIATE_COLS:
            src = schema_lower_wx.get(col.lower(), col)
            vals_wx = wx.column(src).to_pylist() if src in wx.schema.names else [0.0] * wx.num_rows
            raw_cov[col] = [float(v) if v is not None else 0.0 for v in vals_wx]

        # Normalize covariates
        norm_cov: dict[str, list] = {}
        for col, vals_wx in raw_cov.items():
            normed, mu, std = _zscore(vals_wx)
            norm_cov[f"{col}_norm"] = normed
            cov_stats[col]          = {"mean": mu, "std": std}

        # Build ts → covariate lookup for merge
        for i, t in enumerate(ts_wx):
            cov_by_ts[t] = {col: raw_cov[col][i] for col in SOLARNET_COVARIATE_COLS}
            cov_by_ts[t].update({f"{col}_norm": norm_cov[f"{col}_norm"][i]
                                  for col in SOLARNET_COVARIATE_COLS})
    else:
        print(f"  SKIP {wx_path.name} (not found) — covariate columns will be zeros.")

    stats["covariates"] = cov_stats

    # ── Merge on timestamp_unix and write timeseries.parquet ─────────────────
    # Target timestamps drive the output rows (left join).
    cov_zero = {col: 0.0 for col in SOLARNET_COVARIATE_COLS}
    cov_zero.update({f"{col}_norm": 0.0 for col in SOLARNET_COVARIATE_COLS})

    merged_rows: dict[str, list] = {
        "entity_id":      [],
        "timestamp_unix": [],
        **{col: [] for col in SOLARNET_TARGET_COLS},
        **{f"{col}_norm": [] for col in SOLARNET_TARGET_COLS},
        **{col: [] for col in SOLARNET_COVARIATE_COLS},
        **{f"{col}_norm": [] for col in SOLARNET_COVARIATE_COLS},
        "mask_target": [],
        "mask_cov":    [],
    }

    for i, ts in enumerate(ts_irr):
        cov_row = cov_by_ts.get(ts, cov_zero)
        has_cov = int(ts in cov_by_ts)
        merged_rows["entity_id"].append(_ENTITY_ID)
        merged_rows["timestamp_unix"].append(ts)
        for col in SOLARNET_TARGET_COLS:
            merged_rows[col].append(target_arrays[col][i])
            merged_rows[f"{col}_norm"].append(target_norm_arrays[f"{col}_norm"][i])
        for col in SOLARNET_COVARIATE_COLS:
            merged_rows[col].append(cov_row.get(col, 0.0))
            merged_rows[f"{col}_norm"].append(cov_row.get(f"{col}_norm", 0.0))
        merged_rows["mask_target"].append(1)
        merged_rows["mask_cov"].append(has_cov)

    out_table = pa.table({
        "entity_id":      pa.array(merged_rows["entity_id"],      pa.int32()),
        "timestamp_unix": pa.array(merged_rows["timestamp_unix"], pa.int64()),
        **{col:           pa.array(merged_rows[col], pa.float32())
           for col in SOLARNET_TARGET_COLS},
        **{f"{col}_norm": pa.array(merged_rows[f"{col}_norm"], pa.float32())
           for col in SOLARNET_TARGET_COLS},
        **{col:           pa.array(merged_rows[col], pa.float32())
           for col in SOLARNET_COVARIATE_COLS},
        **{f"{col}_norm": pa.array(merged_rows[f"{col}_norm"], pa.float32())
           for col in SOLARNET_COVARIATE_COLS},
        "mask_target": pa.array(merged_rows["mask_target"], pa.int8()),
        "mask_cov":    pa.array(merged_rows["mask_cov"],    pa.int8()),
    })

    dst = out_dir / "timeseries.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, dst, compression="snappy")
    print(f"  timeseries.parquet ({len(ts_irr):,} rows, "
          f"{len(SOLARNET_TARGET_COLS) + len(SOLARNET_COVARIATE_COLS)} value cols) -> {dst}")

    return stats


# ---------------------------------------------------------------------------
# Image extraction from tar.bz2
# ---------------------------------------------------------------------------

def _save_frame(args: tuple) -> tuple[int | None, str | None, bool]:
    """Decode + resize one image from raw bytes. Returns (ts_unix, rel_path, ok)."""
    path_in_archive, raw_bytes, out_path, img_size, quality = args
    ts = _parse_ts_from_path(path_in_archive)
    try:
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.LANCZOS)
        img.save(out_path, "JPEG", quality=quality, optimize=True)
        return ts, str(out_path.name), True
    except Exception as e:
        return ts, None, False


def extract_archives(solarnet_raw: Path, out_dir: Path,
                     img_size: int, quality: int,
                     max_frames: int | None,
                     workers: int) -> list[dict]:
    """
    Extract sky images from the three tar.bz2 archives.
    Returns list of {ts_unix, rel_path} dicts for the frame index.
    """
    archives = sorted(solarnet_raw.glob("Folsom_sky_images_*.tar.bz2"))
    if not archives:
        print("  No sky image archives found — skipping frame extraction.")
        return []

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records: list[dict] = []
    total_extracted = 0
    seq_counter = 0  # fallback sequential counter for unparseable timestamps

    for arc in archives:
        year = arc.stem.split("_")[-1]
        print(f"\n  Opening {arc.name} ({year})…")

        try:
            tf = tarfile.open(arc, "r:bz2")
        except Exception as e:
            print(f"    ERROR: cannot open {arc.name}: {e}")
            continue

        # Collect image members
        members = [m for m in tf.getmembers()
                   if m.isfile() and m.name.lower().endswith((".jpg", ".jpeg", ".png"))]

        if max_frames is not None:
            remaining = max_frames - total_extracted
            if remaining <= 0:
                tf.close()
                break
            members = members[:remaining]

        print(f"    {len(members):,} images")

        tasks = []
        for m in members:
            ts = _parse_ts_from_path(m.name)
            if ts is None:
                ts_str  = f"seq{seq_counter:08d}"
                seq_counter += 1
                fname = f"{year}_{ts_str}.jpg"
            else:
                # Use UTC timestamp as filename: YYYYMMDD_HHMM
                dt    = datetime.fromtimestamp(ts, tz=timezone.utc)
                fname = f"{dt.strftime('%Y%m%d_%H%M%S')}.jpg"

            out_path = frames_dir / fname
            if out_path.exists():
                frame_records.append({"ts_unix": ts, "rel_path": f"frames/{fname}"})
                continue

            try:
                raw_bytes = tf.extractfile(m).read()
            except Exception:
                continue

            tasks.append((m.name, raw_bytes, out_path, img_size, quality))

        tf.close()

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futs = [exe.submit(_save_frame, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc=f"    Extracting {year}"):
                ts_val, fname, ok = fut.result()
                if ok and fname:
                    frame_records.append({
                        "ts_unix":  ts_val,
                        "rel_path": f"frames/{fname}",
                    })

        total_extracted += len(members)
        print(f"    {total_extracted:,} total frames extracted so far")

    return frame_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(img_size: int = 224, quality: int = 90,
         data_root: Path | None = None,
         max_frames: int | None = None,
         workers: int = 4) -> None:

    raw_dir = (data_root / "raw" / "solar" / "solarnet") if data_root else SOLARNET_RAW
    out_dir = ((data_root or PROJECT_ROOT / "data") / "refactored" / "solar" / "solarnet")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Solarnet  —  refactor")
    print(f"  src : {raw_dir}")
    print(f"  dest: {out_dir}")
    print(f"  img : {img_size}×{img_size} JPEG quality={quality}")
    if max_frames:
        print(f"  max frames: {max_frames:,}")
    print("=" * 60)

    stats = process_timeseries(raw_dir, out_dir)

    frame_records = extract_archives(raw_dir, out_dir, img_size, quality,
                                     max_frames, workers)

    # Legacy split files no longer written — remove if they exist from an old run
    for stale in ("targets.parquet", "covariates.parquet"):
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
            print(f"  Removed stale {stale}")

    # ── Frame index ─────────────────────────────────────────────────────────
    if frame_records:
        frame_records.sort(key=lambda r: r["ts_unix"] or 0)
        fi_table = pa.table({
            "entity_id":      pa.array([_ENTITY_ID] * len(frame_records), pa.int32()),
            "timestamp_unix": pa.array([r["ts_unix"]  or -1 for r in frame_records], pa.int64()),
            "rel_path":       pa.array([r["rel_path"]        for r in frame_records], pa.string()),
            "mask_visual":    pa.array([1 if r["ts_unix"] else 0
                                        for r in frame_records], pa.int8()),
        })
        fi_path = out_dir / "frame_index.parquet"
        pq.write_table(fi_table, fi_path, compression="snappy")
        print(f"\n  frame_index.parquet ({len(frame_records):,} frames) -> {fi_path}")
    else:
        print("\n  No frames extracted — frame_index.parquet not written.")

    # ── Metadata ─────────────────────────────────────────────────────────────
    frames_dir = out_dir / "frames"
    frames_disk_gb = (
        sum(p.stat().st_size for p in frames_dir.glob("*.jpg")) / 1e9
        if frames_dir.exists() else 0.0
    )
    metadata = {
        "dataset":          "Solarnet (Folsom, CA)",
        "entity_count":     1,
        "entities":         [{"id": 0, "name": "folsom_site",
                              "lat": 38.68, "lon": -121.17}],
        "timeseries_file":  "timeseries.parquet",
        "frame_index_file": "frame_index.parquet",
        "target_cols":      SOLARNET_TARGET_COLS,
        "target_units":     "W/m²",
        "covariate_cols":   SOLARNET_COVARIATE_COLS,
        "img_size":         img_size,
        "img_format":       "JPEG",
        "jpeg_quality":     quality,
        "frames_extracted": len(frame_records),
        "frames_disk_gb":   round(frames_disk_gb, 3),
        "normalization":    stats,
        "graph":            {"nodes": [{"id": 0, "name": "folsom_site",
                                        "lat": 38.68, "lon": -121.17}],
                             "edges": [], "adjacency_matrix": [[1.0]]},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  metadata.json -> {out_dir / 'metadata.json'}")
    print(f"\n  Frames on disk: {frames_disk_gb:.2f} GB")
    print("\n[Solarnet] Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactor Solarnet dataset.")
    parser.add_argument("--img-size",   type=int,  default=224)
    parser.add_argument("--quality",    type=int,  default=90)
    parser.add_argument("--data-root",  type=Path, default=None)
    parser.add_argument("--max-frames", type=int,  default=None,
                        help="Cap total extracted frames (useful for testing).")
    parser.add_argument("--workers",    type=int,  default=4)
    args = parser.parse_args()
    main(img_size=args.img_size, quality=args.quality,
         data_root=args.data_root, max_frames=args.max_frames,
         workers=args.workers)
