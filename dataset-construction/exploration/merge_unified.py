"""Merge the standardized dataset into ONE images h5 + ONE numerical parquet.

Produces:
  images_all.h5     one group per site:
                      uk_pv_*      (N, 128, 128)     uint8  (copied from images_uk128.h5)
                      goes_pvdaq_* (N, 256, 256, 3)  uint8  (copied from images.h5)
  numerical/dataset_all.parquet
                    = all_curated.parquet (all 34 features) + one canonical
                      `image_index` column that points into images_all.h5.

Why no reindex: each group is copied verbatim from its source h5, so the row
order inside every group is unchanged. The existing per-row indices therefore
stay valid:
    goes rows  -> image_h5_index    (their group came from images.h5)
    uk rows    -> image_uk128_index (their group came from images_uk128.h5)
so image_index = where(goes, image_h5_index, image_uk128_index). Validated: every
row has a valid (>=0) index in its source, so no row is dropped.

Run (after extend_goes_standardized.py has updated the goes groups):
    /Volumes/SSD/dataset-exploration/.venv/bin/python \
        /Volumes/SSD/dataset-exploration/merge_unified.py
"""

import os

import h5py
import numpy as np
import pandas as pd

STD = "/Volumes/SSD/standardized-dataset"
IMAGES_H5 = os.path.join(STD, "images.h5")          # goes 256x256x3 (+ uk 32, unused)
UK128_H5 = os.path.join(STD, "images_uk128.h5")     # uk 128x128
CUR_PATH = os.path.join(STD, "numerical/all_curated.parquet")

OUT_H5 = os.path.join(STD, "images_all.h5")
OUT_PARQUET = os.path.join(STD, "numerical/dataset_all.parquet")

N_VERIFY = 400
rng = np.random.default_rng(42)

# ----------------------------------------------------------------- 1. parquet
print("[1] Loading curated parquet")
df = pd.read_parquet(CUR_PATH)
df["site_id"] = df["site_id"].astype(str)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
is_goes = df.dataset == "goes_pvdaq"

bad = ((is_goes & (df.image_h5_index < 0)) |
       (~is_goes & (df.image_uk128_index < 0)))
if bad.any():
    raise SystemExit(f"{int(bad.sum())} rows lack an image in their source h5; "
                     "resolve before merging (would point image_index at -1)")

df["image_index"] = np.where(is_goes, df.image_h5_index,
                             df.image_uk128_index).astype("int64")
print(f"  rows: {len(df):,}  (goes {int(is_goes.sum()):,} | uk {int((~is_goes).sum()):,})")

# ----------------------------------------------------------------- 2. unified h5
print("[2] Building images_all.h5 (verbatim group copies)")
goes_keys = sorted("goes_pvdaq_" + df.loc[is_goes, "site_id"].unique())
uk_keys = sorted("uk_pv_" + df.loc[~is_goes, "site_id"].unique())

if os.path.exists(OUT_H5):
    os.remove(OUT_H5)
with h5py.File(OUT_H5, "w") as dst:
    with h5py.File(IMAGES_H5, "r") as src:
        for k in goes_keys:
            if k not in src:
                raise SystemExit(f"goes group {k} missing from images.h5")
            src.copy(k, dst)
            print(f"  goes {k}: {src[k]['images'].shape}")
    with h5py.File(UK128_H5, "r") as src:
        for i, k in enumerate(uk_keys):
            if k not in src:
                raise SystemExit(f"uk group {k} missing from images_uk128.h5")
            src.copy(k, dst)
            if (i + 1) % 25 == 0:
                print(f"  uk {i + 1}/{len(uk_keys)} copied")
print(f"  size: {os.path.getsize(OUT_H5) / 1e9:.2f} GB")

# ----------------------------------------------------------------- 3. write parquet
print("[3] Writing dataset_all.parquet")
df.to_parquet(OUT_PARQUET, index=False)
print(f"  {OUT_PARQUET}: {len(df):,} rows x {len(df.columns)} cols")

# ----------------------------------------------------------------- 4. verify
print(f"[4] Verifying {N_VERIFY} rows: image_index + timestamp align to source")
sample = df.sample(min(N_VERIFY, len(df)), random_state=42)
fails = 0
with h5py.File(OUT_H5, "r") as out, \
     h5py.File(IMAGES_H5, "r") as src_g, \
     h5py.File(UK128_H5, "r") as src_u:
    for _, row in sample.iterrows():
        site_key = f"{row.dataset}_{row.site_id}"
        idx = int(row.image_index)
        out_img = out[site_key]["images"][idx]
        out_ts = out[site_key]["timestamps"][idx]
        ref = (src_g if row.dataset == "goes_pvdaq" else src_u)[site_key]
        ref_img = ref["images"][idx]
        want_ts = row.timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ").encode()
        if out_ts != want_ts:
            print(f"  TS MISMATCH {site_key} idx={idx}: h5={out_ts} parquet={want_ts}")
            fails += 1
        if out_img.shape != ref_img.shape or not np.array_equal(out_img, ref_img):
            print(f"  PIXEL MISMATCH {site_key} idx={idx}")
            fails += 1
if fails:
    raise SystemExit(f"VERIFICATION FAILED: {fails} problems in {len(sample)} rows")
print(f"VERIFICATION PASSED: {len(sample)}/{len(sample)} rows aligned + identical")
print("\nDone.")
print(f"  images:    {OUT_H5}")
print(f"  numerical: {OUT_PARQUET}  (use the `image_index` column)")
