"""Re-extract UK satellite crops at 128x128 px (~128 km) into a single HDF5.

Derived from main.py with three changes:
  1. patch_radius = 64  ->  128x128 crops (was 16 -> 32x32). At ~1 km/px HRV
     over the UK this is ~128 km of context, enough for 1-4 h cloud advection.
  2. No PV yield / Hugging Face download: site ids + coordinates come from the
     already-built standardized parquet, so crops align with existing rows.
  3. Output is a single uint8 HDF5 (images_uk128.h5) written straight from the
     float32 memmaps - the 81 GB merged float32 NetCDF step is skipped. The
     SSD is exFAT with 128 KiB clusters, so millions of small files are not an
     option; one big file is mandatory.

Resume-safe: the per-batch progress marker and on-disk memmaps survive kills,
matching main.py behaviour.

Layout of images_uk128.h5 (mirrors standardized-dataset/images.h5):
    /uk_pv_<id>/images      uint8 (N, 128, 128)
    /uk_pv_<id>/timestamps  bytes ISO-8601 UTC, sorted

Run from this directory:  uv run python extract_128.py
"""

import os
import time as _time

import gcsfs
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
import numcodecs
import ocf_blosc2

numcodecs.registry.register_codec(ocf_blosc2.Blosc2)

PARQUET = "/Volumes/SSD/standardized-dataset/numerical/all.parquet"
WORK_DIR = "/Volumes/SSD/uk-extraction-128"
OUT_H5 = "/Volumes/SSD/standardized-dataset/images_uk128.h5"

PATCH_RADIUS = 64
PATCH_SIZE = 2 * PATCH_RADIUS
START_TIME = "2019-01-01T00:00:00"
END_TIME = "2020-12-31T23:30:00"
TIME_BATCH = 64  # frames per network fetch; 128x128 window is ~16x the data of 32x32
YEARS = [2019, 2020]


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    print("--- Step 1: Site list from standardized parquet ---")
    df = pd.read_parquet(PARQUET, columns=["dataset", "site_id", "latitude", "longitude"])
    sites = (
        df[df.dataset == "uk_pv"]
        .groupby("site_id")[["latitude", "longitude"]]
        .first()
        .reset_index()
    )
    print(f"{len(sites)} uk_pv sites")

    all_timestamps = pd.date_range(start=START_TIME, end=END_TIME, freq="30min")
    target_timestamps = all_timestamps[all_timestamps.indexer_between_time("08:00", "16:00")]
    n_t = len(target_timestamps)
    print(f"{n_t} target timestamps")

    print("--- Step 2: Opening EUMETSAT HRV Zarr streams ---")
    fs = gcsfs.GCSFileSystem(token="anon")
    sat_datasets = []
    for year in YEARS:
        path = (f"gs://public-datasets-eumetsat-solar-forecasting/satellite/"
                f"EUMETSAT/SEVIRI_RSS/v4/{year}_hrv.zarr")
        print(f" -> {year}...")
        ds = xr.open_zarr(fs.get_mapper(path), consolidated=True, chunks={})
        sat_datasets.append(ds)
    sat_ds = xr.concat(sat_datasets, dim="time", join="override")
    sat_var = sat_ds[list(sat_ds.data_vars)[0]]

    print("--- Step 3: Projection + per-site pixel windows ---")
    attrs = sat_ds.attrs.get("projection", {
        "proj": "geos", "lon_0": 9.5, "h": 35785831, "x_0": 0, "y_0": 0,
        "a": 6378169, "rf": 295.488065897014,
    })
    proj = (f"+proj={attrs['proj']} +lon_0={attrs['lon_0']} +h={attrs['h']} "
            f"+x_0={attrs['x_0']} +y_0={attrs['y_0']} +a={attrs['a']} +rf={attrs['rf']}")
    transformer = Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    n_x = sat_ds.sizes["x_geostationary"]
    n_y = sat_ds.sizes["y_geostationary"]
    x_coords = sat_ds.x_geostationary.values
    y_coords = sat_ds.y_geostationary.values

    plant_info = []
    for _, row in sites.iterrows():
        sat_x, sat_y = transformer.transform(row.longitude, row.latitude)
        x_idx = int(np.abs(x_coords - sat_x).argmin())
        y_idx = int(np.abs(y_coords - sat_y).argmin())
        x0 = max(0, min(x_idx - PATCH_RADIUS, n_x - PATCH_SIZE))
        y0 = max(0, min(y_idx - PATCH_RADIUS, n_y - PATCH_SIZE))
        plant_info.append((row.site_id, x0, y0))

    bx0 = min(x0 for _, x0, _ in plant_info)
    bx1 = max(x0 + PATCH_SIZE for _, x0, _ in plant_info)
    by0 = min(y0 for _, _, y0 in plant_info)
    by1 = max(y0 + PATCH_SIZE for _, _, y0 in plant_info)
    region = sat_var.isel(
        x_geostationary=slice(bx0, bx1),
        y_geostationary=slice(by0, by1),
    ).squeeze(drop=True).transpose("time", "y_geostationary", "x_geostationary")
    print(f"Shared window: {by1 - by0} x {bx1 - bx0} px for {len(plant_info)} plants")

    print("--- Step 4: Streaming time batches ---")
    time_idx = sat_ds.indexes["time"].get_indexer(target_timestamps, method="nearest")

    memmaps = {}
    for sid, _, _ in plant_info:
        mpath = os.path.join(WORK_DIR, f"pv_{sid}.patches.npy")
        mode = "r+" if os.path.exists(mpath) else "w+"
        memmaps[sid] = np.lib.format.open_memmap(
            mpath, mode=mode, dtype=np.float32,
            shape=(n_t, PATCH_SIZE, PATCH_SIZE),
        )

    progress_path = os.path.join(WORK_DIR, "_progress.txt")
    done_batches = 0
    if os.path.exists(progress_path):
        try:
            done_batches = int(open(progress_path).read().strip())
        except Exception:
            done_batches = 0

    n_batches = (n_t + TIME_BATCH - 1) // TIME_BATCH
    print(f"{n_batches} batches of {TIME_BATCH} frames (resuming from {done_batches})")

    for b, start in enumerate(range(0, n_t, TIME_BATCH)):
        if b < done_batches:
            continue
        end = min(start + TIME_BATCH, n_t)
        t0 = _time.time()
        block = region.isel(time=time_idx[start:end]).values
        block = np.ascontiguousarray(block, dtype=np.float32)
        for sid, x0, y0 in plant_info:
            ly, lx = y0 - by0, x0 - bx0
            memmaps[sid][start:end] = block[:, ly:ly + PATCH_SIZE, lx:lx + PATCH_SIZE]
        for mm in memmaps.values():
            mm.flush()
        with open(progress_path, "w") as f:
            f.write(str(b + 1))
        dt = _time.time() - t0
        eta = dt * (n_batches - b - 1)
        print(f"batch {b + 1}/{n_batches} in {dt:.0f}s | ~{eta / 3600:.1f} h left",
              flush=True)

    print("--- Step 5: Writing uint8 HDF5 ---")
    iso_strs = target_timestamps.strftime("%Y-%m-%dT%H:%M:%SZ").values
    with h5py.File(OUT_H5, "w") as h5:
        for sid, _, _ in plant_info:
            patches = np.asarray(memmaps[sid])
            valid = ~np.isnan(patches).all(axis=(1, 2))
            arr = np.nan_to_num(patches[valid], nan=0.0)
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            grp = h5.create_group(f"uk_pv_{sid}")
            grp.create_dataset("images", data=arr, compression="lzf",
                               chunks=(4,) + arr.shape[1:])
            grp.create_dataset("timestamps",
                               data=np.array(iso_strs[valid], dtype="S20"))
            print(f"  uk_pv_{sid}: {valid.sum()} valid frames")

    print(f"\nDone: {OUT_H5} ({os.path.getsize(OUT_H5) / 1e9:.1f} GB)")
    print(f"Memmap work dir {WORK_DIR} (~81 GB) can now be deleted.")


if __name__ == "__main__":
    main()
