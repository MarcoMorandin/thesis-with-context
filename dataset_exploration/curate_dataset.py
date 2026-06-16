"""Curate the standardized PV dataset for model training.

Reads the raw  all.parquet  and writes  dataset_all.parquet  (dataset of record) with:

  1. Capacity audit/fix: effective capacity = p99.5 of observed power when it
     exceeds the declared capacity by >5% (metadata error, not bad readings);
     power then clipped to [0, capacity].
  2. Quality flags (rows are kept, never deleted):
       outage_flag      zero power under shortwave_radiation > 200 W/m²
       stuck_flag       identical non-zero power for >=6 consecutive daytime steps
       night_clamped    power forced to 0 where irradiance is 0
       bad_site_flag    site-level corr(norm_power, shortwave_radiation) < 0.6
     power_w / norm_power are set to NaN on outage and stuck rows so models
     treat them as missing instead of real zeros.
  3. norm_power = power_w / installed_power_w  (post-fix capacity).
  4. Clear-sky columns (pvlib, Haurwitz model — zenith-only, no external
     turbidity tables): clearsky_ghi, kt (clear-sky index of irradiance) and
     csi (clear-sky index of power). Both indices are NaN when
     clearsky_ghi < 50 W/m² (dawn/dusk, numerically unstable).
  5. Solar geometry features: solar_zenith, solar_azimuth, doy_sin, doy_cos,
     solar_time.

No resampling, no gap interpolation: native 30-min (uk_pv) and 15-min
(goes_pvdaq) grids are preserved and missing rows stay missing.
"""

import os

import numpy as np
import pandas as pd
import pvlib

# Dataset-build script: reads the raw extraction (external) and writes the
# curated table of record. The raw `all.parquet` is an upstream artifact;
# the output is `thesis-dataset/dataset_all.parquet` (DATASET_CONTRACT §1.0).
DATA_DIR = "/Volumes/SSD/thesis-dataset"
IN_PATH = os.path.join(DATA_DIR, "all.parquet")
OUT_PATH = os.path.join(DATA_DIR, "dataset_all.parquet")

CAPACITY_TOLERANCE = 1.05   # declared capacity considered wrong above this
CSI_MIN_CLEARSKY = 50.0     # W/m² floor below which kt/csi are undefined
STC_IRRADIANCE = 1000.0     # W/m² standard test condition for csi denominator
BAD_SITE_CORR = 0.6
STUCK_RUN_LEN = 6

print("Loading", IN_PATH)
df = pd.read_parquet(IN_PATH)
df = df.sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)
n0 = len(df)

# ------------------------------------------------------------------ 1. capacity

print("Auditing installed capacity (p99.5 of observed power per site)...")
p995 = df.groupby("site_id")["power_w"].quantile(0.995)
declared = df.groupby("site_id")["installed_power_w"].first()
needs_fix = p995 > CAPACITY_TOLERANCE * declared
fixed_capacity = declared.where(~needs_fix, p995)
df["capacity_fixed"] = df.site_id.map(needs_fix)
df["installed_power_w"] = df.site_id.map(fixed_capacity)
print(f"  capacity raised for {needs_fix.sum()} sites: "
      f"{list(declared.index[needs_fix])}")

neg = (df.power_w < 0).sum()
over = (df.power_w > df.installed_power_w).sum()
df["power_w"] = df.power_w.clip(lower=0, upper=df.installed_power_w)
print(f"  clipped {neg} negative and {over} over-capacity readings")

# ------------------------------------------------------------------ 2. flags

print("Flagging outages, stuck sensors, night production, bad sites...")

df["outage_flag"] = (df.power_w == 0) & (df.shortwave_radiation > 200)

daytime = df.shortwave_radiation > 50
stuck = pd.Series(False, index=df.index)
for _, s in df[daytime].groupby("site_id")["power_w"]:
    runs = (s != s.shift()).cumsum()
    run_len = s.groupby(runs).transform("size")
    stuck.loc[s.index] = (run_len >= STUCK_RUN_LEN) & (s > 0)
df["stuck_flag"] = stuck

df["night_clamped"] = (df.shortwave_radiation == 0) & (df.power_w > 0)
df.loc[df.night_clamped, "power_w"] = 0.0

df["norm_power"] = df.power_w / df.installed_power_w

site_corr = (
    df[df.shortwave_radiation > 10]
    .groupby("site_id")
    .apply(lambda g: g["norm_power"].corr(g["shortwave_radiation"]),
           include_groups=False)
)
bad_sites = site_corr.index[site_corr < BAD_SITE_CORR]
df["bad_site_flag"] = df.site_id.isin(bad_sites)
print(f"  outage rows:   {df.outage_flag.sum():,}")
print(f"  stuck rows:    {df.stuck_flag.sum():,}")
print(f"  night clamped: {df.night_clamped.sum():,}")
print(f"  bad sites (corr < {BAD_SITE_CORR}): {list(bad_sites)}")

# mask the target on rows we cannot trust — missing, not zero
mask = df.outage_flag | df.stuck_flag
df.loc[mask, ["power_w", "norm_power"]] = np.nan
print(f"  target masked (NaN) on {mask.sum():,} rows")

# ------------------------------------------------------------------ 3. clear-sky + geometry

print("Computing solar position and clear-sky GHI per site (pvlib)...")
zenith = np.empty(len(df))
azimuth = np.empty(len(df))
clearsky = np.empty(len(df))
for site, g in df.groupby("site_id"):
    times = pd.DatetimeIndex(g.timestamp_utc)
    pos = pvlib.solarposition.get_solarposition(
        times, g.latitude.iloc[0], g.longitude.iloc[0])
    zenith[g.index] = pos["apparent_zenith"].values
    azimuth[g.index] = pos["azimuth"].values
    cs = pvlib.clearsky.haurwitz(pos["apparent_zenith"])
    clearsky[g.index] = cs["ghi"].values

df["solar_zenith"] = zenith
df["solar_azimuth"] = azimuth
df["clearsky_ghi"] = clearsky

ok = df.clearsky_ghi >= CSI_MIN_CLEARSKY
df["kt"] = np.where(ok, df.shortwave_radiation / df.clearsky_ghi, np.nan)
df["csi"] = np.where(
    ok,
    df.power_w / (df.installed_power_w * df.clearsky_ghi / STC_IRRADIANCE),
    np.nan,
)
print(f"  kt/csi defined on {ok.sum():,} rows "
      f"({ok.mean():.1%}; rest below {CSI_MIN_CLEARSKY} W/m² clear-sky)")

doy = df.timestamp_utc.dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
df["solar_time"] = (
    df.timestamp_utc.dt.hour
    + df.timestamp_utc.dt.minute / 60
    + df.longitude / 15.0
) % 24

# ------------------------------------------------------------------ save

assert len(df) == n0, "curation must not add or drop rows"
df.to_parquet(OUT_PATH, index=False)
print(f"\nWrote {OUT_PATH}")
print(f"  rows: {len(df):,}  columns: {len(df.columns)}")
print("  new columns: capacity_fixed, outage_flag, stuck_flag, night_clamped, "
      "bad_site_flag, norm_power, solar_zenith, solar_azimuth, clearsky_ghi, "
      "kt, csi, doy_sin, doy_cos, solar_time")
