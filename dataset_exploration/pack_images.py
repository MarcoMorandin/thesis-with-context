"""Pack the raw image tree into a single HDF5 file (images_all.h5, dataset of record).

Why: the SSD is exFAT with 128 KiB allocation blocks; 2.5M tiny PNGs (+ macOS
``._`` sidecars) burn ~305 GB of disk for ~3 GB of pixels. One HDF5 file
removes the cluster waste and is faster for training DataLoaders.

Layout of images_all.h5:
    /<site_key>/images      uint8, (N, 128, 128) for uk_pv, (N, 256, 256, 3) for goes
    /<site_key>/timestamps  bytes, ISO-8601 UTC, sorted, aligned with images

uk_pv images are rebuilt straight from the source netCDF (single 4.7 GB read)
using the exact transform of dataset_builder/build_dataset.py
(nan_to_num -> clip(0,1) -> *255 uint8), instead of 2.5M slow exFAT reads.
goes images are read from their PNGs (only ~15k files).

After packing, a verification pass samples rows and compares HDF5 arrays
against the original PNG pixels — must match exactly before the PNG tree may
be deleted.

Adds ``image_h5_index`` (row index within the site's image stack) to both
all.parquet and dataset_all.parquet.
"""

import os
import glob

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

# Dataset-build script: packs the raw frame tree (external) into the packed
# archive of record `thesis-dataset/images_all.h5` (DATASET_CONTRACT §1.0).
DATA_DIR = "/Volumes/SSD/thesis-dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
H5_PATH = os.path.join(DATA_DIR, "images_all.h5")
UK_NC = "/Volumes/SSD/useless-stuff-dataset/uk-data/uk_pv_local_paired_dataset.nc"
PARQUETS = [
    os.path.join(DATA_DIR, "all.parquet"),
    os.path.join(DATA_DIR, "dataset_all.parquet"),
]
N_VERIFY = 300

rng = np.random.default_rng(42)


def iso(ts_array):
    return pd.to_datetime(ts_array).strftime("%Y-%m-%dT%H:%M:%SZ")


print("Opening", H5_PATH)
h5 = h5py.File(H5_PATH, "w")

# ------------------------------------------------------------------ goes PNGs

goes_dirs = sorted(
    d for d in os.listdir(IMAGES_DIR)
    if d.startswith("goes_") and not d.startswith("._")
)
print(f"Packing {len(goes_dirs)} goes sites from PNGs...")
for site_key in goes_dirs:
    files = sorted(
        f for f in glob.glob(os.path.join(IMAGES_DIR, site_key, "*.png"))
        if not os.path.basename(f).startswith("._")
    )
    imgs = np.stack([np.asarray(Image.open(f)) for f in files])
    # filename format 2019-06-01T10-15-00Z -> ISO 2019-06-01T10:15:00Z
    names = [os.path.basename(f)[:-4] for f in files]
    stamps = [n[:13] + n[13:].replace("-", ":") for n in names]
    grp = h5.create_group(site_key)
    grp.create_dataset("images", data=imgs, compression="lzf",
                       chunks=(1,) + imgs.shape[1:])
    grp.create_dataset("timestamps", data=np.array(stamps, dtype="S20"))
    print(f"  {site_key}: {len(files)} images {imgs.shape[1:]}")

# ------------------------------------------------------------------ uk from nc

print("Packing uk_pv sites from source netCDF...")
ds = xr.open_dataset(UK_NC)
pv_ids = ds["pv_id"].values
times = pd.to_datetime(ds["time"].values)
iso_strs = times.strftime("%Y-%m-%dT%H:%M:%SZ").values

for i, pv_id in enumerate(pv_ids):
    patches = ds["satellite_hrv"].isel(pv_id=i).values  # (T, 32, 32) float32
    valid = ~np.isnan(patches).all(axis=(1, 2))
    arr = np.nan_to_num(patches[valid], nan=0.0)
    arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    site_key = f"uk_pv_{pv_id}"
    grp = h5.create_group(site_key)
    grp.create_dataset("images", data=arr, compression="lzf",
                       chunks=(16,) + arr.shape[1:])
    grp.create_dataset("timestamps",
                       data=np.array(iso_strs[valid], dtype="S20"))
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(pv_ids)} uk sites packed")
ds.close()
h5.close()
print(f"Packed. Size: {os.path.getsize(H5_PATH) / 1e9:.2f} GB")

# ------------------------------------------------------------------ verify

print(f"\nVerifying {N_VERIFY} random rows against original PNGs...")
df = pd.read_parquet(PARQUETS[0],
                     columns=["dataset", "site_id", "timestamp_utc", "image_path"])
sample = df.sample(N_VERIFY, random_state=42)

h5 = h5py.File(H5_PATH, "r")
fails = 0
for _, row in sample.iterrows():
    site_key = f"{row.dataset}_{row.site_id}"
    ts = row.timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    stamps = h5[site_key]["timestamps"][:]
    idx = np.searchsorted(stamps, ts)
    if idx >= len(stamps) or stamps[idx] != ts:
        print(f"  MISSING in h5: {site_key} {ts}")
        fails += 1
        continue
    h5_img = h5[site_key]["images"][idx]
    png = np.asarray(Image.open(os.path.join(DATA_DIR, row.image_path)))
    if h5_img.shape != png.shape or not np.array_equal(h5_img, png):
        print(f"  MISMATCH: {site_key} {ts}")
        fails += 1

if fails:
    h5.close()
    raise SystemExit(f"VERIFICATION FAILED: {fails}/{N_VERIFY} rows differ. "
                     "Do NOT delete the PNG tree.")
print(f"VERIFICATION PASSED: {N_VERIFY}/{N_VERIFY} rows identical.")

# ------------------------------------------------------------------ index col

print("\nAdding image_h5_index to parquets...")
index_maps = {}
for site_key in h5.keys():
    stamps = h5[site_key]["timestamps"][:].astype(str)
    index_maps[site_key] = {s: i for i, s in enumerate(stamps)}
h5.close()

for path in PARQUETS:
    full = pd.read_parquet(path)
    keys = full["dataset"] + "_" + full["site_id"]
    stamps = full["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    full["image_h5_index"] = [
        index_maps.get(k, {}).get(s, -1) for k, s in zip(keys, stamps)
    ]
    n_miss = (full["image_h5_index"] < 0).sum()
    full.to_parquet(path, index=False)
    print(f"  {os.path.basename(path)}: indexed, {n_miss} rows without image")

print("\nDone. PNG tree may now be deleted to reclaim ~300 GB:")
print(f"  rm -rf {IMAGES_DIR}")
