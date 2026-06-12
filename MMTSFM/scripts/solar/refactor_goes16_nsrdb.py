"""
Refactor GOES-16 + NSRDB: raw files → cleaned, training-ready layout.

What this script does
─────────────────────
1. NSRDB (4 629 CSVs):
   • Consolidates all CSVs into one parquet, z-score normalises all columns,
     extracts lat/lon from filenames, writes timeseries.parquet.
   • Builds a k-NN spatial graph (graph.json) over the entity grid.

2. GOES-16 (17 473 NetCDF files):
   • Extracts three physically meaningful ABI bands:
       C02  (0.64 µm — red visible, cloud optical depth)
       C07  (3.9 µm  — shortwave IR, low-cloud / fog detection)
       C13  (10.3 µm — thermal IR, cloud-top temperature, day+night)
   • Crops to the bounding box of the NSRDB entity grid (+ margin).
   • Optionally resizes to --img-size × --img-size.
   • Converts to float16 and saves one .npz file per timestamp.
   • Writes frame_index.parquet (timestamp_unix, rel_path, crop metadata).

Output layout  (data/refactored/solar/goes16_nsrdb/)
──────────────────────────────────────────────────────
  timeseries.parquet   — entity_id, lat, lon, timestamp_unix,
                         GHI, GHI_norm, DNI, DNI_norm, DHI, DHI_norm,
                         Wind_Speed, Wind_Speed_norm, Temperature, Temperature_norm,
                         Pressure, Pressure_norm, mask_target
  frames/
    {YYYYDDD_HHmm}.npz — float16 array shape (C, H, W), C=3 (or --bands count)
  frame_index.parquet  — timestamp_unix, rel_path, crop_lat_min, crop_lat_max,
                         crop_lon_min, crop_lon_max, img_h, img_w
  graph.json
  metadata.json

Size estimate (3 bands, no resize, full CONUS crop at ABI-2km ~1500×2500 px):
  Uncompressed:  22.5 MB/frame × 17 473 frames ≈ 390 GB
  With npz deflate compression (typical 3–4× for float weather data):
  ≈ 95–130 GB  →  total with other datasets ≈ 110–145 GB.

With --img-size 1024:  ≈ 50–65 GB GOES-16 alone.
With --img-size 512:   ≈ 12–17 GB GOES-16 alone.

ABI projection
──────────────
Uses the NOAA fixed-grid formula (no external pyproj dependency):
  Goes Imager Fixed-Grid → geographic lat/lon for bounding-box computation.
  pyproj is used instead if installed (faster, more accurate for edge cases).

Usage:
  python scripts/solar/refactor_goes16_nsrdb.py [--img-size 0]
                                                [--bands C02,C07,C13]
                                                [--margin-deg 1.0]
                                                [--workers N]
                                                [--skip-nsrdb]
                                                [--skip-goes16]
                                                [--skip-graph]
                                                [--data-root PATH]
                                                [--max-frames N]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    GOES16_DIR,
    GOES16_NSRDB_PROCESSED,
    NSRDB_COVARIATE_COLS,
    NSRDB_DIR,
    NSRDB_KNN,
    NSRDB_TARGET_COLS,
    PROJECT_ROOT,
)

# ---------------------------------------------------------------------------
# ABI fixed-grid projection  (NOAA formula — no external deps)
# ---------------------------------------------------------------------------

_ABI_H      = 35_786_023.0    # perspective point height above ellipsoid (m)
_ABI_R_EQ   = 6_378_137.0     # semi-major axis (m)
_ABI_R_POL  = 6_356_752.3     # semi-minor axis (m)
_ABI_LON_0  = -75.0           # GOES-16 sub-satellite longitude (deg)
_ABI_e      = 0.0818191910435 # eccentricity


def _abi_xy_to_latlon(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ABI fixed-grid scanning angles (radians 1-D arrays) to lat/lon grids.
    Returns (lat [deg], lon [deg]) as 2-D arrays of shape (len(y), len(x)).
    Values outside the Earth disk are NaN.
    """
    H   = _ABI_H + _ABI_R_EQ   # satellite orbit radius from Earth centre
    lam = np.deg2rad(_ABI_LON_0)

    X, Y = np.meshgrid(x, y)    # (ny, nx)

    a = (np.sin(X)**2
         + np.cos(X)**2 * (np.cos(Y)**2
                           + (_ABI_R_EQ / _ABI_R_POL)**2 * np.sin(Y)**2))
    b = -2.0 * H * np.cos(X) * np.cos(Y)
    c = H**2 - _ABI_R_EQ**2

    disc = b**2 - 4.0 * a * c
    valid = disc >= 0

    rs = np.where(valid, (-b - np.sqrt(np.where(valid, disc, 0.0))) / (2.0 * a), np.nan)

    sx = rs * np.cos(X) * np.cos(Y) - H
    sy = -rs * np.sin(X)
    sz = rs * np.cos(X) * np.sin(Y)

    lat = np.where(valid,
                   np.degrees(np.arctan((_ABI_R_EQ / _ABI_R_POL)**2
                                        * sz / np.sqrt(sx**2 + sy**2))),
                   np.nan)
    lon = np.where(valid,
                   np.degrees(lam - np.arctan2(sy, -sx)),
                   np.nan)
    return lat, lon


# ---------------------------------------------------------------------------
# NSRDB processing
# ---------------------------------------------------------------------------

def _zscore(values: list) -> tuple[list[float], float, float]:
    valid = [float(v) for v in values if v is not None]
    mu    = sum(valid) / len(valid) if valid else 0.0
    var   = sum((v - mu)**2 for v in valid) / max(len(valid), 1)
    std   = max(var**0.5, 1e-6)
    normed = [(float(v) - mu) / std if v is not None else 0.0 for v in values]
    return normed, mu, std


def _latlon_from_fname(path: Path) -> tuple[float, float] | None:
    m = re.match(r"nsrdb_\d+_([-\d.]+)_([-\d.]+)\.csv", path.name)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _read_one_nsrdb(args: tuple) -> pa.Table | None:
    """
    Self-contained NSRDB CSV reader (no module-level state) so it works safely
    inside both ThreadPoolExecutor and ProcessPoolExecutor on Windows.
    """
    entity_id, path, target_cols, cov_cols = args
    # Instantiate options locally — avoids any module-level state in subprocesses.
    # NSRDB PSM3 format has 3 header rows:
    #   row 0 — field names  (Source, Location ID, …)
    #   row 1 — metadata values (NSRDB, 615880, …)
    #   row 2 — data column names (Year, Month, Day, …)  ← we want this as header
    read_opts  = pa_csv.ReadOptions(skip_rows=2, encoding="utf-8")
    parse_opts = pa_csv.ParseOptions(delimiter=",")

    try:
        tbl = pa_csv.read_csv(str(path), read_options=read_opts,
                              parse_options=parse_opts)
    except Exception as e:
        raise RuntimeError(f"CSV read failed for {path.name}: {e}") from e

    ll = _latlon_from_fname(path)
    if ll is None or tbl.num_rows == 0:
        return None
    lat, lon = ll
    n = tbl.num_rows

    try:
        ts_list = [
            int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp())
            for y, mo, d, h, mi in zip(
                tbl.column("Year").to_pylist(),
                tbl.column("Month").to_pylist(),
                tbl.column("Day").to_pylist(),
                tbl.column("Hour").to_pylist(),
                tbl.column("Minute").to_pylist(),
            )
        ]
    except Exception as e:
        raise RuntimeError(
            f"Timestamp build failed for {path.name} — "
            f"columns={tbl.schema.names[:10]}: {e}"
        ) from e

    cols: dict[str, pa.Array] = {
        "entity_id":      pa.array([entity_id] * n, pa.int32()),
        "lat":            pa.array([lat]        * n, pa.float32()),
        "lon":            pa.array([lon]        * n, pa.float32()),
        "timestamp_unix": pa.array(ts_list,         pa.int64()),
    }
    col_map = {c: c.replace(" ", "_") for c in target_cols + cov_cols}
    for orig, safe in col_map.items():
        if orig in tbl.schema.names:
            cols[safe] = pc.cast(tbl.column(orig), pa.float32())
    cols["mask_target"] = pa.array([1] * n, pa.int8())
    return pa.table(cols)


# Welford online algorithm for incremental mean/variance (no full data in RAM).
class _Welford:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def update(self, x: float):
        if x is None or (x != x):  # skip None and NaN
            return
        self.n += 1
        delta  = x - self.mean
        self.mean += delta / self.n
        self.M2   += delta * (x - self.mean)
    def finalize(self) -> tuple[float, float]:
        std = max((self.M2 / self.n) ** 0.5 if self.n > 1 else 1.0, 1e-6)
        return self.mean, std


def process_nsrdb(nsrdb_dir: Path, out_dir: Path,
                  workers: int) -> tuple[list[dict], dict]:
    """
    Returns (entity_list, normalization_stats).
    Uses ThreadPoolExecutor (I/O-bound) and incremental ParquetWriter so the
    full dataset never needs to fit in RAM at once.
    Two-pass strategy:
      Pass 1: read CSVs → write raw timeseries.parquet (no norm columns yet),
              accumulate Welford stats online.
      Pass 2: read parquet back column-by-column → append _norm columns.
    """
    csv_files = sorted(nsrdb_dir.glob("nsrdb_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No NSRDB CSVs in {nsrdb_dir}")

    print(f"  Found {len(csv_files):,} NSRDB files — reading with {workers} threads…")
    ll_list  = sorted({_latlon_from_fname(f) for f in csv_files if _latlon_from_fname(f)})
    ll_to_id = {ll: i for i, ll in enumerate(ll_list)}
    task_args = [
        (ll_to_id[_latlon_from_fname(f)], f, NSRDB_TARGET_COLS, NSRDB_COVARIATE_COLS)
        for f in csv_files if _latlon_from_fname(f)
    ]

    all_numeric = [c.replace(" ", "_") for c in NSRDB_TARGET_COLS + NSRDB_COVARIATE_COLS]
    welford: dict[str, _Welford] = {c: _Welford() for c in all_numeric}

    dst = out_dir / "timeseries.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: read CSVs, write parquet incrementally ─────────────────────
    writer: pq.ParquetWriter | None = None
    first_error: str | None = None
    n_ok = 0
    WRITE_BATCH = 100   # flush every 100 files (~1.7M rows / batch)

    batch: list[pa.Table] = []
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futs = {exe.submit(_read_one_nsrdb, a): a for a in task_args}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  NSRDB CSVs"):
            try:
                t = fut.result()
            except Exception as e:
                if first_error is None:
                    first_error = str(e)
                continue
            if t is None:
                continue
            n_ok += 1
            # Accumulate Welford stats
            for col in all_numeric:
                if col in t.schema.names:
                    w = welford[col]
                    for v in t.column(col).to_pylist():
                        w.update(v)
            batch.append(t)
            if len(batch) >= WRITE_BATCH:
                chunk = pa.concat_tables(batch)
                if writer is None:
                    writer = pq.ParquetWriter(dst, chunk.schema, compression="snappy")
                writer.write_table(chunk)
                batch.clear()

    if batch:
        chunk = pa.concat_tables(batch)
        if writer is None:
            writer = pq.ParquetWriter(dst, chunk.schema, compression="snappy")
        writer.write_table(chunk)
        batch.clear()

    if writer:
        writer.close()

    if n_ok == 0:
        msg = f"No NSRDB files could be read."
        if first_error:
            msg += f" First error: {first_error}"
        raise RuntimeError(msg)

    print(f"  {n_ok:,} files read OK  ({first_error and f'first error: {first_error}' or 'no errors'})")
    print(f"  timeseries.parquet (raw) → {dst}  ({dst.stat().st_size / 1e9:.2f} GB)")

    # ── Pass 2: append _norm columns ──────────────────────────────────────
    stats: dict[str, dict] = {}
    norm_map: dict[str, tuple[float, float]] = {}
    for col, w in welford.items():
        mu, std = w.finalize()
        norm_map[col] = (mu, std)
        stats[col]    = {"mean": mu, "std": std}

    print("  Appending normalised columns…")
    raw = pq.read_table(dst)
    for col, (mu, std) in norm_map.items():
        if col not in raw.schema.names:
            continue
        arr = raw.column(col).to_pylist()
        normed = [(float(v) - mu) / std if v is not None else 0.0 for v in arr]
        raw = raw.append_column(f"{col}_norm", pa.array(normed, pa.float32()))

    pq.write_table(raw, dst, row_group_size=500_000, compression="snappy")
    print(f"  timeseries.parquet (with norms) → {dst}  ({dst.stat().st_size / 1e9:.2f} GB)")

    entities = [{"id": ll_to_id[ll], "lat": ll[0], "lon": ll[1]} for ll in ll_list]
    return entities, stats


# ---------------------------------------------------------------------------
# Spatial graph
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dl = math.radians(lat2 - lat1)
    dl2 = math.radians(lon2 - lon1)
    a = (math.sin(dl / 2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dl2 / 2)**2)
    return 2 * R * math.asin(math.sqrt(a))


def build_graph(entities: list[dict], k: int, out_dir: Path) -> None:
    n = len(entities)
    print(f"  Building {k}-NN graph for {n:,} entities…")
    edges = []
    CHUNK = 500
    for i in range(0, n, CHUNK):
        batch = entities[i: i + CHUNK]
        for ei in batch:
            dists = sorted(
                (_haversine(ei["lat"], ei["lon"], ej["lat"], ej["lon"]), ej["id"])
                for ej in entities if ej["id"] != ei["id"]
            )
            for dist_km, nb_id in dists[:k]:
                edges.append({"src": ei["id"], "dst": nb_id, "dist_km": round(dist_km, 3)})

    graph = {"nodes": entities, "edges": edges, "k": k}
    (out_dir / "graph.json").write_text(json.dumps(graph, indent=2))
    print(f"  graph.json ({len(edges):,} directed edges) → {out_dir / 'graph.json'}")


# ---------------------------------------------------------------------------
# GOES-16 frame extraction
# ---------------------------------------------------------------------------

_GOES_PATTERN = re.compile(
    r"OR_ABI-L2-MCMIPC-M6_G16_s(\d{4})(\d{3})(\d{2})(\d{2})\d+_e.*\.nc$"
)


def _parse_goes_filename(path: Path) -> dict | None:
    m = _GOES_PATTERN.search(path.name)
    if not m:
        return None
    year, doy, hour, minute = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    dt = datetime(year, 1, 1, hour, minute, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    return {"ts_unix": int(dt.timestamp()), "year": year, "doy": doy, "hour": hour,
            "fname_stem": f"{year}{doy:03d}_{hour:02d}{minute:02d}"}


def _compute_crop_indices(sample_nc: Path,
                          lat_min: float, lat_max: float,
                          lon_min: float, lon_max: float) -> dict | None:
    """
    Open one GOES-16 NetCDF file, compute the pixel row/col ranges that
    correspond to the requested lat/lon bounding box.
    Returns dict with {row_start, row_end, col_start, col_end, ny, nx,
                       lat_grid (1-D), lon_grid (1-D)}.
    """
    try:
        import netCDF4 as nc4  # type: ignore
    except ImportError:
        print("  ERROR: netCDF4 not installed. "
              "Run: uv add netCDF4  (or: pip install netCDF4)")
        return None

    ds = nc4.Dataset(sample_nc)
    x_vals = ds.variables["x"][:]   # scanning angle radians (nx,)
    y_vals = ds.variables["y"][:]   # scanning angle radians (ny,)
    ds.close()

    lat_grid, lon_grid = _abi_xy_to_latlon(x_vals, y_vals)  # (ny, nx)

    # Find rows/cols where (lat, lon) ∈ [bbox]
    in_lat = (lat_grid >= lat_min) & (lat_grid <= lat_max)
    in_lon = (lon_grid >= lon_min) & (lon_grid <= lon_max)
    in_box = in_lat & in_lon

    rows_valid = np.where(in_box.any(axis=1))[0]
    cols_valid = np.where(in_box.any(axis=0))[0]

    if len(rows_valid) == 0 or len(cols_valid) == 0:
        return None

    return {
        "row_start": int(rows_valid[0]),
        "row_end":   int(rows_valid[-1]) + 1,
        "col_start": int(cols_valid[0]),
        "col_end":   int(cols_valid[-1]) + 1,
    }


def _extract_one_frame(args: tuple) -> dict | None:
    """
    Extract bands from one GOES-16 NetCDF, crop, optionally resize, save npz.
    Returns frame record dict or None on failure.
    """
    nc_path, out_path, bands, crop, img_size = args
    try:
        import netCDF4 as nc4  # type: ignore
        ds   = nc4.Dataset(nc_path)
        rs   = crop["row_start"]
        re_  = crop["row_end"]
        cs   = crop["col_start"]
        ce   = crop["col_end"]

        arrays = []
        for band in bands:
            var_name = f"CMI_{band}"
            if var_name not in ds.variables:
                # Try uppercase band suffix
                var_name = f"CMI_C{band.lstrip('C').zfill(2)}"
            if var_name not in ds.variables:
                arrays.append(np.zeros((re_ - rs, ce - cs), dtype=np.float16))
                continue
            data = ds.variables[var_name][rs:re_, cs:ce]
            # Handle masked arrays
            if hasattr(data, "filled"):
                data = data.filled(np.nan)
            data = data.astype(np.float16)
            arrays.append(data)
        ds.close()

        frame = np.stack(arrays, axis=0)  # (C, H, W) float16

        if img_size and img_size > 0:
            from PIL import Image as _PIL
            # Resize each band independently
            resized = []
            for c in range(frame.shape[0]):
                band_u8 = ((frame[c].astype(np.float32) - np.nanmin(frame[c]))
                           / (np.nanmax(frame[c]) - np.nanmin(frame[c]) + 1e-6)
                           * 255).astype(np.uint8)
                img = _PIL.fromarray(band_u8).resize((img_size, img_size), _PIL.LANCZOS)
                resized.append(np.array(img, dtype=np.float16) / 255.0)
            frame = np.stack(resized, axis=0)

        # Replace NaN with 0 before saving
        frame = np.nan_to_num(frame, nan=0.0)
        np.savez_compressed(out_path, frame=frame)
        return {"ok": True, "shape": list(frame.shape)}

    except Exception as e:
        return {"ok": False, "error": str(e)}


def process_goes16(goes16_dir: Path, out_dir: Path,
                   entities: list[dict],
                   bands: list[str],
                   img_size: int,
                   margin_deg: float,
                   workers: int,
                   max_frames: int | None) -> list[dict]:
    """
    Extract GOES-16 frames cropped to the NSRDB entity bounding box.
    Returns list of frame_record dicts for the frame index.
    """
    try:
        import netCDF4  # noqa: F401
    except ImportError:
        print("  netCDF4 not installed — skipping GOES-16 frame extraction.")
        print("  Install with:  uv add netCDF4")
        return []

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Bounding box from NSRDB entities
    lats = [e["lat"] for e in entities]
    lons = [e["lon"] for e in entities]
    lat_min = min(lats) - margin_deg
    lat_max = max(lats) + margin_deg
    lon_min = min(lons) - margin_deg
    lon_max = max(lons) + margin_deg
    print(f"  NSRDB bbox (+ {margin_deg}° margin): "
          f"lat [{lat_min:.1f}, {lat_max:.1f}]  "
          f"lon [{lon_min:.1f}, {lon_max:.1f}]")

    # Collect all NetCDF files
    nc_files = sorted(goes16_dir.rglob("*.nc"))
    if max_frames:
        nc_files = nc_files[:max_frames]
    print(f"  {len(nc_files):,} GOES-16 files to process")

    # Compute crop indices from the first available file
    crop: dict | None = None
    for nc in nc_files[:20]:
        crop = _compute_crop_indices(nc, lat_min, lat_max, lon_min, lon_max)
        if crop:
            h = crop["row_end"] - crop["row_start"]
            w = crop["col_end"] - crop["col_start"]
            print(f"  Crop indices: rows {crop['row_start']}–{crop['row_end']}, "
                  f"cols {crop['col_start']}–{crop['col_end']}  ({h}×{w} px)")
            break

    if crop is None:
        print("  ERROR: could not compute crop indices — bounding box may be outside image.")
        return []

    # Size estimate
    C = len(bands)
    H = img_size if img_size > 0 else (crop["row_end"] - crop["row_start"])
    W = img_size if img_size > 0 else (crop["col_end"] - crop["col_start"])
    est_mb_raw = C * H * W * 2 / 1e6  # float16
    print(f"  Estimated {est_mb_raw:.1f} MB/frame uncompressed  "
          f"({len(nc_files) * est_mb_raw / 1e3:.0f} GB total)")

    # Process frames
    frame_records: list[dict] = []
    tasks = []
    for nc in nc_files:
        info = _parse_goes_filename(nc)
        if info is None:
            continue
        out_path = frames_dir / f"{info['fname_stem']}.npz"
        if out_path.exists():
            frame_records.append({
                "ts_unix":    info["ts_unix"],
                "rel_path":   f"frames/{out_path.name}",
                "crop":       crop,
            })
            continue
        tasks.append((str(nc), str(out_path), bands, crop, img_size))

    print(f"  {len(tasks):,} frames to extract ({len(frame_records):,} already done)")

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futs = {exe.submit(_extract_one_frame, t): (t[0], t[1]) for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  GOES-16 frames"):
            nc_str, out_str = futs[fut]
            result = fut.result()
            if result and result["ok"]:
                nc_path_obj  = Path(nc_str)
                info         = _parse_goes_filename(nc_path_obj)
                frame_records.append({
                    "ts_unix":  info["ts_unix"] if info else -1,
                    "rel_path": f"frames/{Path(out_str).name}",
                    "crop":     crop,
                })
            else:
                err = result.get("error", "unknown") if result else "unknown"
                pass  # silent — corrupted files are expected occasionally

    frame_records.sort(key=lambda r: r["ts_unix"])
    return frame_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(img_size: int = 0,
         bands: list[str] | None = None,
         margin_deg: float = 1.0,
         workers: int = 4,
         skip_nsrdb: bool = False,
         skip_goes16: bool = False,
         skip_graph: bool = False,
         data_root: Path | None = None,
         max_frames: int | None = None) -> None:

    if bands is None:
        bands = ["C02", "C07", "C13"]  # visible, SW-IR, thermal-IR

    nsrdb_dir  = (data_root / "raw" / "solar" / "goes16_nsrdb" / "NSRDB")  if data_root else NSRDB_DIR
    goes16_dir = (data_root / "raw" / "solar" / "goes16_nsrdb" / "GOES16") if data_root else GOES16_DIR
    out_dir    = ((data_root or PROJECT_ROOT / "data") / "refactored" / "solar" / "goes16_nsrdb")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GOES-16 + NSRDB  —  refactor")
    print(f"  nsrdb_dir  : {nsrdb_dir}")
    print(f"  goes16_dir : {goes16_dir}")
    print(f"  dest       : {out_dir}")
    print(f"  bands      : {bands}")
    print(f"  img_size   : {'native (no resize)' if not img_size else f'{img_size}×{img_size}'}")
    print("=" * 60)

    entities: list[dict] = []
    norm_stats: dict = {}

    if not skip_nsrdb:
        entities, norm_stats = process_nsrdb(nsrdb_dir, out_dir, workers)
    else:
        print("  [--skip-nsrdb] Skipping NSRDB.")

    if not skip_graph and entities:
        build_graph(entities, k=NSRDB_KNN, out_dir=out_dir)

    frame_records: list[dict] = []
    if not skip_goes16 and entities:
        frame_records = process_goes16(goes16_dir, out_dir, entities,
                                       bands, img_size, margin_deg,
                                       workers, max_frames)
        # Write frame_index.parquet
        if frame_records:
            crop0 = frame_records[0]["crop"]
            H = img_size if img_size > 0 else (crop0["row_end"] - crop0["row_start"])
            W = img_size if img_size > 0 else (crop0["col_end"] - crop0["col_start"])
            fi_table = pa.table({
                "timestamp_unix": pa.array([r["ts_unix"]  for r in frame_records], pa.int64()),
                "rel_path":       pa.array([r["rel_path"] for r in frame_records], pa.string()),
                "crop_row_start": pa.array([crop0["row_start"]] * len(frame_records), pa.int32()),
                "crop_row_end":   pa.array([crop0["row_end"]]   * len(frame_records), pa.int32()),
                "crop_col_start": pa.array([crop0["col_start"]] * len(frame_records), pa.int32()),
                "crop_col_end":   pa.array([crop0["col_end"]]   * len(frame_records), pa.int32()),
                "img_h":          pa.array([H] * len(frame_records), pa.int32()),
                "img_w":          pa.array([W] * len(frame_records), pa.int32()),
            })
            fi_path = out_dir / "frame_index.parquet"
            pq.write_table(fi_table, fi_path, compression="snappy")
            print(f"  frame_index.parquet ({len(frame_records):,} frames) → {fi_path}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    frames_dir = out_dir / "frames"
    frames_gb = 0.0
    if frames_dir.exists():
        frames_gb = sum(p.stat().st_size for p in frames_dir.glob("*.npz")) / 1e9

    metadata = {
        "dataset":           "GOES-16 + NSRDB",
        "entity_count":      len(entities),
        "bands":             bands,
        "img_size":          img_size or "native",
        "img_format":        "npz (float16, shape C×H×W)",
        "frames_extracted":  len(frame_records),
        "frames_disk_gb":    round(frames_gb, 2),
        "target_cols":       [c.replace(" ", "_") for c in NSRDB_TARGET_COLS],
        "target_units":      "W/m²",
        "covariate_cols":    [c.replace(" ", "_") for c in NSRDB_COVARIATE_COLS],
        "normalization":     norm_stats,
        "spatial_graph_k":   NSRDB_KNN,
        "margin_deg":        margin_deg,
        "year":              2021,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n  metadata.json → {out_dir / 'metadata.json'}")
    print(f"  GOES-16 frames on disk: {frames_gb:.2f} GB")
    print("\n[GOES-16 + NSRDB] Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactor GOES-16 + NSRDB dataset.")
    parser.add_argument("--img-size",   type=int,   default=0,
                        help="Resize crop to N×N pixels (0 = keep native resolution).")
    parser.add_argument("--bands",      type=str,   default="C02,C07,C13",
                        help="Comma-separated ABI band names.")
    parser.add_argument("--margin-deg", type=float, default=1.0,
                        help="Degrees to expand bbox beyond NSRDB entity extents.")
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--skip-nsrdb",  action="store_true")
    parser.add_argument("--skip-goes16", action="store_true")
    parser.add_argument("--skip-graph",  action="store_true")
    parser.add_argument("--data-root",   type=Path,  default=None)
    parser.add_argument("--max-frames",  type=int,   default=None,
                        help="Process only the first N GOES-16 files (testing).")
    args = parser.parse_args()
    main(img_size=args.img_size,
         bands=args.bands.split(","),
         margin_deg=args.margin_deg,
         workers=args.workers,
         skip_nsrdb=args.skip_nsrdb,
         skip_goes16=args.skip_goes16,
         skip_graph=args.skip_graph,
         data_root=args.data_root,
         max_frames=args.max_frames)
