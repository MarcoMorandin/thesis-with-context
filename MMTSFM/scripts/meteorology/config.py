"""
Shared paths and constants for meteorology data processing scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Raw dataset paths
# ---------------------------------------------------------------------------
METEONET_RAW        = DATA_ROOT / "raw"        / "meteorology" / "meteonet"
METEONET_REFACTORED = DATA_ROOT / "refactored" / "meteorology" / "meteonet"

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
METEONET_TARGET_COLS    = ["t", "precip"]        # temperature (K), precipitation (mm/h)
METEONET_COVARIATE_COLS = ["dd", "ff", "hu", "td", "psl"]
# dd=wind dir (°), ff=wind speed (m/s), hu=rel. humidity (%), td=dew point (K), psl=SLP (Pa)

METEONET_REGIONS = ["NW", "SE"]
METEONET_YEARS   = [2016, 2017, 2018]

# k-nearest neighbours for spatial station graph
METEONET_KNN = 8

# Radar grid dimensions (fixed by MeteoNet)
RADAR_H, RADAR_W = 565, 784

# ---------------------------------------------------------------------------
# EarthNet2021
# ---------------------------------------------------------------------------
EARTHNET_RAW        = DATA_ROOT / "raw"        / "meteorology" / "earthnet2021"
EARTHNET_REFACTORED = DATA_ROOT / "refactored" / "meteorology" / "earthnet2021"

# highresdynamic channels (0-indexed within axis=2)
# Channel layout: Blue(B02), Green(B03), Red(B04), NIR(B8A), CloudProb(%), SCL, BinaryMask
EARTHNET_SENTINEL_BAND_INDICES = [0, 1, 2, 3]          # B, G, R, NIR saved as frames
EARTHNET_SENTINEL_BAND_NAMES   = ["blue", "green", "red", "nir"]
EARTHNET_RED_IDX   = 2                                  # Red channel index
EARTHNET_NIR_IDX   = 3                                  # NIR channel index
EARTHNET_MASK_IDX  = 6                                  # binary clear-sky mask channel
EARTHNET_N_BANDS   = 7                                  # total highresdynamic channels
EARTHNET_IMG_H     = 128
EARTHNET_IMG_W     = 128
EARTHNET_T_SENTINEL = 30                                # Sentinel-2 timesteps per sample
EARTHNET_T_ERA5     = 150                               # ERA5 timesteps per sample
EARTHNET_ERA5_RATIO = EARTHNET_T_ERA5 // EARTHNET_T_SENTINEL  # 5 ERA5 per Sentinel step

# mesodynamic channels (ERA5)
EARTHNET_ERA5_COLS = [
    "era5_precip",   # total precipitation (normalised [0,1] by earthnet)
    "era5_psl",      # sea level pressure
    "era5_temp",     # 2m temperature
    "era5_cloud",    # total cloud cover fraction
    "era5_srad",     # surface solar radiation
]

EARTHNET_TARGET_COLS    = ["ndvi"]
EARTHNET_COVARIATE_COLS = EARTHNET_ERA5_COLS

EARTHNET_SPLITS = [
    "train",
    "iid_test_split",
    "ood_test_split",
    "extreme_test_split",
    "seasonal_test_split",
]

# k-NN for spatial graph (within-tile patch proximity)
EARTHNET_KNN = 8

# ---------------------------------------------------------------------------
# ERA5 EU
# ---------------------------------------------------------------------------
ERA5_EU_RAW        = DATA_ROOT / "raw"        / "meteorology" / "era5"
ERA5_EU_REFACTORED = DATA_ROOT / "refactored" / "meteorology" / "era5_eu"

ERA5_EU_ZIP         = ERA5_EU_RAW / "era5_eu_2020_2021.nc"   # ZIP from CDS
ERA5_EU_EXTRACTED   = ERA5_EU_RAW / "extracted"               # decompressed NetCDFs

ERA5_EU_INSTANT_FILE = "data_stream-oper_stepType-instant.nc"
ERA5_EU_ACCUM_FILE   = "data_stream-oper_stepType-accum.nc"

ERA5_EU_TARGET_COLS    = ["t2m", "ssrd"]     # 2m temperature (K), solar radiation (J/m²)
ERA5_EU_COVARIATE_COLS = ["u10", "v10", "cape", "tcwv"]
# u10/v10: 10m wind components (m/s), cape: CAPE (J/kg), tcwv: water vapour (kg/m²)

# Frame channels order (must match target + covariate for consistent indexing)
ERA5_EU_FRAME_CHANNELS = ERA5_EU_TARGET_COLS + ERA5_EU_COVARIATE_COLS
# = ["t2m", "ssrd", "u10", "v10", "cape", "tcwv"]

ERA5_EU_TEMPORAL_STEP_H = 6    # 6-hourly
ERA5_EU_WRITE_CHUNK     = 100  # timesteps per streaming write chunk
