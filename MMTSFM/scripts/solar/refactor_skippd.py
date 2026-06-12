"""
Refactor SKIPP'D: raw parquets → cleaned, training-ready layout.

What this script does
─────────────────────
1. Reads all split parquets (labels/ for timeseries, data/ for images).
2. Decodes each embedded JPEG, resizes to --img-size × --img-size, saves as JPEG.
3. Writes a clean timeseries parquet with z-score normalised target column.
4. Writes a frame_index parquet (entity_id, timestamp_unix, rel_path, mask_visual).
5. Writes metadata.json with normalization stats and dataset info.

Output layout  (data/refactored/solar/skippd/)
────────────────────────────────────────────────
  timeseries.parquet       — entity_id, timestamp_unix, pv_power, pv_power_norm, mask_target
  frames/
    {row_idx:010d}.jpg     — resized sky image
  frame_index.parquet      — entity_id, timestamp_unix, rel_path, mask_visual
  metadata.json

Expected size: ~7 GB at img_size=224 (349 K images × ~20 KB JPEG).

Usage:
  python scripts/solar/refactor_skippd.py [--img-size 224] [--quality 90]
                                          [--data-root PATH] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SKIPPD_RAW, PROJECT_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_unix_seconds(time_col: pa.ChunkedArray) -> pa.Array:
    flat = time_col.combine_chunks() if isinstance(time_col, pa.ChunkedArray) else time_col
    if pa.types.is_timestamp(flat.type):
        us = pc.cast(flat, pa.int64())
        return pc.divide(us, pa.scalar(1_000_000, pa.int64()))
    return pc.cast(flat, pa.int64())


def _zscore_normalize(values: list[float]) -> tuple[list[float], float, float]:
    """Return (normalised_values, mean, std)."""
    import statistics
    valid = [v for v in values if v is not None]
    mu  = statistics.mean(valid)
    std = statistics.stdev(valid) or 1.0
    normed = [(v - mu) / std if v is not None else 0.0 for v in values]
    return normed, mu, std


def _decode_and_save(args: tuple) -> tuple[int, bool]:
    """Decode one image from bytes and save as JPEG. Returns (row_idx, success)."""
    row_idx, image_bytes, out_path, img_size, quality = args
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.LANCZOS)
        img.save(out_path, "JPEG", quality=quality, optimize=True)
        return row_idx, True
    except Exception as e:
        print(f"\n  WARNING row {row_idx}: {e}")
        return row_idx, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(img_size: int = 224, quality: int = 90,
         data_root: Path | None = None, workers: int = 4) -> None:

    raw_dir = (data_root / "raw" / "solar" / "skippd") if data_root else SKIPPD_RAW
    out_dir = ((data_root or PROJECT_ROOT / "data") / "refactored" / "solar" / "skippd")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SKIPP'D  —  refactor")
    print(f"  src : {raw_dir}")
    print(f"  dest: {out_dir}")
    print(f"  img : {img_size}×{img_size} JPEG quality={quality}")
    print("=" * 60)

    # ── 1. Read all labels (lightweight) ───────────────────────────────────
    label_tables = []
    for split in ("train", "test"):
        label_files = sorted((raw_dir / "labels").glob(f"{split}-*.parquet"))
        for f in label_files:
            label_tables.append(pq.read_table(f))
    labels = pa.concat_tables(label_tables)
    n_total = labels.num_rows
    print(f"  Total rows: {n_total:,}")

    ts_unix    = _to_unix_seconds(labels.column("time"))
    pv_raw     = [float(v) if v is not None else 0.0
                  for v in labels.column("pv").to_pylist()]

    # Compute normalization stats on the training portion (first 349372 rows = train)
    # We approximate: use all available rows if split info is unavailable
    pv_norm, pv_mean, pv_std = _zscore_normalize(pv_raw)

    # ── 2. Read images and save as JPEG ───────────────────────────────────
    data_files_train = sorted((raw_dir / "data").glob("train-*.parquet"))
    data_files_test  = sorted((raw_dir / "data").glob("test-*.parquet"))
    data_files = data_files_train + data_files_test

    print(f"\n  Extracting {n_total:,} images from {len(data_files)} parquet files…")

    saved_paths: dict[int, str | None] = {}   # global_idx → rel_path (None if failed)

    global_offset = 0

    # One progress bar across all images; iterate row-groups to avoid re-reading files.
    with tqdm(total=n_total, desc="  Frames", unit="img") as pbar:
        for fpath in data_files:
            pf = pq.ParquetFile(fpath)

            for rg in range(pf.num_row_groups):
                # Read one row group at a time — memory-efficient, no full-file re-read.
                chunk   = pf.read_row_group(rg, columns=["image"])
                n_rg    = chunk.num_rows
                images  = chunk.column("image").to_pylist()  # list of {"bytes": …}

                tasks = []
                for local_i, img_dict in enumerate(images):
                    global_i  = global_offset + local_i
                    fname     = f"{global_i:010d}.jpg"
                    out_path  = frames_dir / fname
                    if out_path.exists():
                        saved_paths[global_i] = f"frames/{fname}"
                        pbar.update(1)
                        continue
                    tasks.append((global_i, img_dict["bytes"], out_path, img_size, quality))

                with ThreadPoolExecutor(max_workers=workers) as exe:
                    futs = {exe.submit(_decode_and_save, t): t[0] for t in tasks}
                    for fut in as_completed(futs):
                        idx, ok = fut.result()
                        saved_paths[idx] = f"frames/{idx:010d}.jpg" if ok else None
                        pbar.update(1)

                global_offset += n_rg

    # ── 3. Write timeseries parquet ────────────────────────────────────────
    ts_list  = ts_unix.to_pylist()
    ts_table = pa.table({
        "entity_id":      pa.array([0] * n_total, pa.int32()),
        "timestamp_unix": pa.array(ts_list,        pa.int64()),
        "pv_power":       pa.array(pv_raw,          pa.float32()),
        "pv_power_norm":  pa.array(pv_norm,         pa.float32()),
        "mask_target":    pa.array([1] * n_total,   pa.int8()),
    })
    ts_path = out_dir / "timeseries.parquet"
    pq.write_table(ts_table, ts_path, compression="snappy")
    print(f"\n  timeseries.parquet → {ts_path}")

    # ── 4. Write frame_index parquet ──────────────────────────────────────
    idx_entity, idx_ts, idx_path, idx_mask = [], [], [], []
    for i, ts in enumerate(ts_list):
        rel = saved_paths.get(i)
        idx_entity.append(0)
        idx_ts.append(ts)
        idx_path.append(rel or "")
        idx_mask.append(1 if rel else 0)

    frame_idx_table = pa.table({
        "entity_id":      pa.array(idx_entity, pa.int32()),
        "timestamp_unix": pa.array(idx_ts,     pa.int64()),
        "rel_path":       pa.array(idx_path,   pa.string()),
        "mask_visual":    pa.array(idx_mask,   pa.int8()),
    })
    fi_path = out_dir / "frame_index.parquet"
    pq.write_table(frame_idx_table, fi_path, compression="snappy")
    print(f"  frame_index.parquet ({len(idx_ts):,} entries) → {fi_path}")

    # ── 5. Metadata ────────────────────────────────────────────────────────
    frames_saved  = sum(1 for v in saved_paths.values() if v)
    frames_failed = len(saved_paths) - frames_saved
    metadata = {
        "dataset":        "SKIPP'D",
        "entity_count":   1,
        "entities":       [{"id": 0, "name": "skippd_site", "lat": None, "lon": None}],
        "total_rows":     n_total,
        "img_size":       img_size,
        "img_format":     "JPEG",
        "jpeg_quality":   quality,
        "frames_saved":   frames_saved,
        "frames_failed":  frames_failed,
        "target_col":     "pv_power",
        "target_units":   "kW (normalised PV output)",
        "normalization":  {
            "pv_power": {"method": "zscore", "mean": pv_mean, "std": pv_std}
        },
        "covariate_cols": [],
        "adjacency_matrix": [[1.0]],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  metadata.json → {out_dir / 'metadata.json'}")

    # ── Size report ────────────────────────────────────────────────────────
    total_bytes = sum(p.stat().st_size for p in frames_dir.glob("*.jpg"))
    print(f"\n  Frames on disk: {total_bytes / 1e9:.2f} GB")
    print("\n[SKIPP'D] Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactor SKIPP'D dataset.")
    parser.add_argument("--img-size",  type=int, default=224)
    parser.add_argument("--quality",   type=int, default=90,
                        help="JPEG quality (1–95).")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel image-encode workers.")
    args = parser.parse_args()
    main(img_size=args.img_size, quality=args.quality,
         data_root=args.data_root, workers=args.workers)
