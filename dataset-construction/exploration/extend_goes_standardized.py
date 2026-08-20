"""Incrementally enlarge the goes_pvdaq portion of the standardized dataset.

The standardized dataset currently holds goes_pvdaq images/rows for June 2019
only (~15k rows). solar-satellite-v2 now has Jan-Oct 2019 (~105k rows). This
script replaces ONLY the goes_pvdaq portion (rows + images.h5 groups) with the
larger v2 build. The uk_pv portion (1.2M rows, h5 groups, images_uk128.h5) is
left completely untouched.

Pipeline (goes-only mirror of build_dataset -> add_metadata -> curate -> pack):

  0. Back up all.parquet / all_curated.parquet / images.h5 to *.bak (once).
  1. Load existing standardized parquets; keep the non-goes (uk_pv) rows as-is.
  2. New goes base rows = concat of every v2 chunk parquet (1:1 with PNG on disk).
  3. lat/lon/installed_power_w reused from the existing June goes rows (per site).
  4. Weather (open-meteo archive) fetched fresh for each goes site's full span and
     merged nearest-time per site. (add_metadata's per-site skip would have left
     the new rows weatherless, so we fetch directly here.)
  5. Curate goes rows (capacity audit, QC flags, solar geometry, clear-sky) with
     the exact logic of curate_dataset.py, on the goes subset.
  6. Repack the 10 goes groups in images.h5 from the v2 PNGs (uk groups untouched).
  7. image_h5_index recomputed from the new h5 timestamps; image_uk128_index = -1.
  8. Concat uk (unchanged) + goes (new), write all.parquet + all_curated.parquet.
  9. Verify random goes rows: h5 pixels must equal the source PNG.

Run:
    /Volumes/SSD/dataset-exploration/.venv/bin/python \
        /Volumes/SSD/dataset-exploration/extend_goes_standardized.py

Idempotent: backups are written once and reused; rerunning rebuilds from the
v2 chunks + the (untouched) uk rows in the .bak parquets.
"""

import glob
import os
import shutil
import time

import h5py
import numpy as np
import pandas as pd
import pvlib
import requests
from PIL import Image

# ----------------------------------------------------------------- paths
STD_DIR = "/Volumes/SSD/standardized-dataset"
ALL_PATH = os.path.join(STD_DIR, "numerical/all.parquet")
CUR_PATH = os.path.join(STD_DIR, "numerical/all_curated.parquet")
H5_PATH = os.path.join(STD_DIR, "images.h5")

V2_DIR = "/Volumes/SSD/solar-satellite-v2/refactored"
V2_CHUNKS = os.path.join(V2_DIR, "numerical/chunks")
V2_IMAGES = os.path.join(V2_DIR, "images")

WEATHER_COLS = [
    "temperature_2m", "shortwave_radiation", "direct_radiation",
    "diffuse_radiation", "direct_normal_irradiance", "cloudcover",
    "windspeed_10m", "precipitation",
]

# curation constants (identical to curate_dataset.py)
CAPACITY_TOLERANCE = 1.05
CSI_MIN_CLEARSKY = 50.0
STC_IRRADIANCE = 1000.0
BAD_SITE_CORR = 0.6
STUCK_RUN_LEN = 6

N_VERIFY = 300
rng = np.random.default_rng(42)


# ----------------------------------------------------------------- helpers
def backup_once(path):
    bak = path + ".bak"
    if os.path.exists(bak):
        print(f"  backup already exists, keeping: {bak}")
        return
    print(f"  backing up {os.path.basename(path)} -> {os.path.basename(bak)}")
    shutil.copy2(path, bak)


def fetch_weather_open_meteo(lat, lon, start_date_str, end_date_str):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date_str, "end_date": end_date_str,
        "hourly": ",".join(WEATHER_COLS), "timezone": "UTC",
    }
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"    rate limited, retry in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})
            if not hourly:
                return pd.DataFrame()
            out = {"time": pd.to_datetime(hourly["time"], utc=True)}
            for c in WEATHER_COLS:
                out[c] = hourly.get(c, [])
            return pd.DataFrame(out)
        except Exception as e:
            print(f"    open-meteo error (try {attempt + 1}/5): {e}")
            time.sleep(10)
    print(f"    FAILED weather for lat={lat} lon={lon}")
    return pd.DataFrame()


# ----------------------------------------------------------------- 0. backups
print("[0] Backups")
for p in (ALL_PATH, CUR_PATH, H5_PATH):
    backup_once(p)

# ----------------------------------------------------------------- 1. existing
print("[1] Loading existing standardized parquets (from .bak)")
all_old = pd.read_parquet(ALL_PATH + ".bak")
cur_old = pd.read_parquet(CUR_PATH + ".bak")
all_old["timestamp_utc"] = pd.to_datetime(all_old["timestamp_utc"], utc=True)
cur_old["timestamp_utc"] = pd.to_datetime(cur_old["timestamp_utc"], utc=True)
all_old["site_id"] = all_old["site_id"].astype(str)
cur_old["site_id"] = cur_old["site_id"].astype(str)

uk_all = all_old[all_old.dataset != "goes_pvdaq"].copy()
uk_cur = cur_old[cur_old.dataset != "goes_pvdaq"].copy()
goes_old = all_old[all_old.dataset == "goes_pvdaq"].copy()
print(f"  uk rows kept: {len(uk_all):,}  | old goes rows replaced: {len(goes_old):,}")

# per-site lat/lon/installed_power from existing June goes rows
site_meta = (goes_old.groupby("site_id")[["latitude", "longitude",
                                          "installed_power_w"]]
             .first().to_dict("index"))

# ----------------------------------------------------------------- 2. v2 chunks
print("[2] Concatenating v2 goes chunk parquets")
chunks = sorted(f for f in glob.glob(os.path.join(V2_CHUNKS, "goes_pvdaq_*__*.parquet"))
                if not os.path.basename(f).startswith("._"))
frames = [pd.read_parquet(c) for c in chunks]
frames = [f for f in frames if len(f)]
goes = pd.concat(frames, ignore_index=True)
goes["timestamp_utc"] = pd.to_datetime(goes["timestamp_utc"], utc=True)
goes["site_id"] = goes["site_id"].astype(str)
goes = (goes.drop_duplicates(subset=["site_id", "timestamp_utc"])
        .sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True))
print(f"  new goes rows from {len(chunks)} chunks: {len(goes):,}")

# ----------------------------------------------------------------- 3. metadata
print("[3] Attaching lat/lon/installed_power (reused from June goes rows)")
missing_meta = sorted(set(goes.site_id) - set(site_meta))
if missing_meta:
    raise SystemExit(f"no existing metadata for goes sites {missing_meta}; "
                     "add an NREL fetch before continuing")
goes["latitude"] = goes.site_id.map(lambda s: site_meta[s]["latitude"])
goes["longitude"] = goes.site_id.map(lambda s: site_meta[s]["longitude"])
goes["installed_power_w"] = goes.site_id.map(lambda s: site_meta[s]["installed_power_w"])

# ----------------------------------------------------------------- 4. weather
print("[4] Fetching open-meteo weather per goes site (full span)")
for c in WEATHER_COLS:
    goes[c] = np.nan
weather_frames = []
for site_id, g in goes.groupby("site_id"):
    lat, lon = site_meta[site_id]["latitude"], site_meta[site_id]["longitude"]
    s = g.timestamp_utc.min().strftime("%Y-%m-%d")
    e = g.timestamp_utc.max().strftime("%Y-%m-%d")
    print(f"  site {site_id}: {s} -> {e}")
    w = fetch_weather_open_meteo(lat, lon, s, e)
    if w.empty:
        raise SystemExit(f"weather fetch failed for site {site_id}; aborting "
                         "(rerun is safe, backups untouched)")
    w["site_id"] = site_id
    weather_frames.append(w)
all_weather = pd.concat(weather_frames, ignore_index=True).sort_values("time")

print("  merging weather (nearest time per site)")
goes = goes.sort_values("timestamp_utc")
goes = goes.drop(columns=WEATHER_COLS)
goes = pd.merge_asof(goes, all_weather, left_on="timestamp_utc",
                     right_on="time", by="site_id", direction="nearest")
if "time" in goes.columns:
    goes = goes.drop(columns=["time"])
goes["image_uk128_index"] = -1
goes["image_h5_index"] = -1  # filled after packing

# align goes base to existing all.parquet column order
goes_all = goes[uk_all.columns].copy()

# ----------------------------------------------------------------- 5. curate
print("[5] Curating goes rows (capacity / flags / geometry / clear-sky)")
c = goes_all.copy().sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)
n0 = len(c)

p995 = c.groupby("site_id")["power_w"].quantile(0.995)
declared = c.groupby("site_id")["installed_power_w"].first()
needs_fix = p995 > CAPACITY_TOLERANCE * declared
fixed_capacity = declared.where(~needs_fix, p995)
c["capacity_fixed"] = c.site_id.map(needs_fix)
c["installed_power_w"] = c.site_id.map(fixed_capacity)
print(f"  capacity raised for {int(needs_fix.sum())} sites: "
      f"{list(declared.index[needs_fix])}")
c["power_w"] = c.power_w.clip(lower=0, upper=c.installed_power_w)

c["outage_flag"] = (c.power_w == 0) & (c.shortwave_radiation > 200)
daytime = c.shortwave_radiation > 50
stuck = pd.Series(False, index=c.index)
for _, s in c[daytime].groupby("site_id")["power_w"]:
    runs = (s != s.shift()).cumsum()
    run_len = s.groupby(runs).transform("size")
    stuck.loc[s.index] = (run_len >= STUCK_RUN_LEN) & (s > 0)
c["stuck_flag"] = stuck
c["night_clamped"] = (c.shortwave_radiation == 0) & (c.power_w > 0)
c.loc[c.night_clamped, "power_w"] = 0.0
c["norm_power"] = c.power_w / c.installed_power_w
site_corr = (c[c.shortwave_radiation > 10].groupby("site_id")
             .apply(lambda g: g["norm_power"].corr(g["shortwave_radiation"]),
                    include_groups=False))
bad_sites = site_corr.index[site_corr < BAD_SITE_CORR]
c["bad_site_flag"] = c.site_id.isin(bad_sites)
mask = c.outage_flag | c.stuck_flag
c.loc[mask, ["power_w", "norm_power"]] = np.nan
print(f"  outage={int(c.outage_flag.sum())} stuck={int(c.stuck_flag.sum())} "
      f"night={int(c.night_clamped.sum())} bad_sites={list(bad_sites)} "
      f"masked={int(mask.sum())}")

zenith = np.empty(len(c)); azimuth = np.empty(len(c)); clearsky = np.empty(len(c))
for site, g in c.groupby("site_id"):
    times = pd.DatetimeIndex(g.timestamp_utc)
    pos = pvlib.solarposition.get_solarposition(times, g.latitude.iloc[0],
                                                 g.longitude.iloc[0])
    zenith[g.index] = pos["apparent_zenith"].values
    azimuth[g.index] = pos["azimuth"].values
    clearsky[g.index] = pvlib.clearsky.haurwitz(pos["apparent_zenith"])["ghi"].values
c["solar_zenith"] = zenith
c["solar_azimuth"] = azimuth
c["clearsky_ghi"] = clearsky
ok = c.clearsky_ghi >= CSI_MIN_CLEARSKY
c["kt"] = np.where(ok, c.shortwave_radiation / c.clearsky_ghi, np.nan)
c["csi"] = np.where(ok, c.power_w / (c.installed_power_w * c.clearsky_ghi
                                     / STC_IRRADIANCE), np.nan)
doy = c.timestamp_utc.dt.dayofyear
c["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
c["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
c["solar_time"] = (c.timestamp_utc.dt.hour + c.timestamp_utc.dt.minute / 60
                   + c.longitude / 15.0) % 24
assert len(c) == n0, "curation changed row count"
goes_cur = c

# ----------------------------------------------------------------- 6. pack h5
print("[6] Repacking 10 goes groups in images.h5 (uk groups untouched)")
goes_keys = sorted(goes_all.site_id.map(lambda s: f"goes_pvdaq_{s}").unique())
with h5py.File(H5_PATH, "a") as h5:
    for site_key in goes_keys:
        if site_key in h5:
            del h5[site_key]
        files = sorted(f for f in glob.glob(os.path.join(V2_IMAGES, site_key, "*.png"))
                       if not os.path.basename(f).startswith("._"))
        imgs = np.stack([np.asarray(Image.open(f)) for f in files])
        names = [os.path.basename(f)[:-4] for f in files]
        stamps = [n[:13] + n[13:].replace("-", ":") for n in names]
        grp = h5.create_group(site_key)
        grp.create_dataset("images", data=imgs, compression="lzf",
                           chunks=(1,) + imgs.shape[1:])
        grp.create_dataset("timestamps", data=np.array(stamps, dtype="S20"))
        print(f"  {site_key}: {len(files)} images {imgs.shape[1:]}")

# ----------------------------------------------------------------- 7. h5 index
print("[7] Computing image_h5_index from new h5 timestamps")
index_maps = {}
with h5py.File(H5_PATH, "r") as h5:
    for site_key in goes_keys:
        stamps = h5[site_key]["timestamps"][:].astype(str)
        index_maps[site_key] = {s: i for i, s in enumerate(stamps)}


def h5_index(df):
    keys = "goes_pvdaq_" + df.site_id.astype(str)
    iso = df.timestamp_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return np.array([index_maps.get(k, {}).get(s, -1)
                     for k, s in zip(keys, iso)])


goes_all["image_h5_index"] = h5_index(goes_all)
goes_cur["image_h5_index"] = h5_index(goes_cur)
goes_cur["image_uk128_index"] = -1
n_miss = int((goes_all.image_h5_index < 0).sum())
print(f"  goes rows without an image in h5: {n_miss} (expected 0)")
if n_miss:
    raise SystemExit("some goes rows have no matching image; aborting")

# ----------------------------------------------------------------- 8. combine
print("[8] Combining uk (unchanged) + goes (new) and writing parquets")
goes_all = goes_all[uk_all.columns]
goes_cur = goes_cur[uk_cur.columns]
new_all = pd.concat([uk_all, goes_all], ignore_index=True)
new_cur = pd.concat([uk_cur, goes_cur], ignore_index=True)
new_all.to_parquet(ALL_PATH, index=False)
new_cur.to_parquet(CUR_PATH, index=False)
print(f"  all.parquet: {len(all_old):,} -> {len(new_all):,} rows")
print(f"  all_curated.parquet: {len(cur_old):,} -> {len(new_cur):,} rows")
print(f"  goes: {len(goes_old):,} -> {len(goes_all):,} rows")

# ----------------------------------------------------------------- 9. verify
print(f"[9] Verifying {N_VERIFY} random goes rows (h5 == source PNG)")
sample = goes_all.sample(min(N_VERIFY, len(goes_all)), random_state=42)
fails = 0
with h5py.File(H5_PATH, "r") as h5:
    for _, row in sample.iterrows():
        site_key = f"goes_pvdaq_{row.site_id}"
        idx = int(row.image_h5_index)
        h5_img = h5[site_key]["images"][idx]
        png_name = row.timestamp_utc.strftime("%Y-%m-%dT%H-%M-%SZ") + ".png"
        png = np.asarray(Image.open(os.path.join(V2_IMAGES, site_key, png_name)))
        if h5_img.shape != png.shape or not np.array_equal(h5_img, png):
            print(f"  MISMATCH {site_key} idx={idx}")
            fails += 1
if fails:
    raise SystemExit(f"VERIFICATION FAILED: {fails}/{len(sample)} mismatched")
print(f"VERIFICATION PASSED: {len(sample)}/{len(sample)} rows identical")
print("\nDone. Backups at *.bak — delete once you trust the result.")
