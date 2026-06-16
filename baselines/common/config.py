"""Shared experimental constants for all baselines.

Single source of truth for window sizes, quantile levels, covariate scaling
and column names, mirroring docs/experiments/BASELINE_PROTOCOL.md §3 and
docs/context/DATASET_CONTRACT.md.
"""

from __future__ import annotations

SEED = 42

# Temporal configuration (BASELINE_PROTOCOL.md §3)
HISTORY_STEPS = 24          # T
HORIZON_STEPS = 12          # H (primary); long-horizon variants below
LONG_HORIZONS = (12, 24, 48)

# Probabilistic evaluation (BASELINE_COMPARISON.md §4.3)
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Dataset columns (dataset_all.parquet)
TARGET_COL = "norm_power"           # capacity-normalized power in [0, 1]
CAPACITY_COL = "installed_power_w"
CLEARSKY_COL = "clearsky_ghi"       # W/m², Haurwitz, zenith-only
SITE_COL = "site_id"
DATASET_COL = "dataset"
TIME_COL = "timestamp_utc"
BAD_SITE_COL = "bad_site_flag"

STC_IRRADIANCE = 1000.0     # W/m²; norm clear-sky power proxy = clearsky_ghi / 1000
SP_MIN_CLEARSKY = 50.0      # W/m²; below this the clearness index is undefined
                            # (matches kt/csi NaN rule in curate_dataset.py) and
                            # smart persistence falls back to plain persistence.

# Covariates available to T1/T2 models, with fixed physical scalings.
# Fixed scalings (instead of train-set statistics) make the normalizer
# trivially leakage-free across plant splits.
COV_SCALES: dict[str, float] = {
    "temperature_2m": 40.0,
    "cloudcover": 100.0,
    "windspeed_10m": 20.0,
    "precipitation": 10.0,
    "shortwave_radiation": 1000.0,
    "direct_radiation": 1000.0,
    "diffuse_radiation": 1000.0,
    "direct_normal_irradiance": 1000.0,
    "solar_zenith": 90.0,
    "solar_azimuth": 360.0,
    "doy_sin": 1.0,
    "doy_cos": 1.0,
    "solar_time": 24.0,
    "clearsky_ghi": 1000.0,
}
COV_COLS: tuple[str, ...] = tuple(COV_SCALES)

# Covariates that are known for future timestamps without any lookahead
# (pure solar geometry / calendar). All other covariates are observed
# weather: their values are only provided over the history window.
DETERMINISTIC_COVS: tuple[str, ...] = (
    "solar_zenith",
    "solar_azimuth",
    "doy_sin",
    "doy_cos",
    "solar_time",
    "clearsky_ghi",
)
DETERMINISTIC_COV_IDX: tuple[int, ...] = tuple(
    COV_COLS.index(c) for c in DETERMINISTIC_COVS
)

# Plant-split fractions per dataset (disjoint cross-plant protocol)
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}

# Dataset of record (DATASET_CONTRACT.md §1.0): one flat numerical table +
# packed frames, covering both uk_pv and goes_pvdaq.
DEFAULT_DATA_PATH = "/Volumes/SSD/thesis-dataset/dataset_all.parquet"
DEFAULT_IMAGES_H5 = "/Volumes/SSD/thesis-dataset/images_all.h5"
FRAME_INDEX_COL = "image_h5_index"   # canonical local-to-group frame pointer
