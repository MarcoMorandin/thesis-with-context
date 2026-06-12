"""
Shared paths and constants for solar data processing scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"
RAW_SOLAR    = DATA_ROOT / "raw"  / "solar"
PROCESSED_SOLAR = DATA_ROOT / "processed" / "solar"

# ---------------------------------------------------------------------------
# Raw dataset paths
# ---------------------------------------------------------------------------
SKIPPD_RAW       = RAW_SOLAR / "skippd"
SOLARNET_RAW     = RAW_SOLAR / "solarnet"
GOES16_NSRDB_RAW = RAW_SOLAR / "goes16_nsrdb"
GOES16_DIR       = GOES16_NSRDB_RAW / "GOES16"
NSRDB_DIR        = GOES16_NSRDB_RAW / "NSRDB"

# ---------------------------------------------------------------------------
# Processed dataset paths
# ---------------------------------------------------------------------------
SKIPPD_PROCESSED       = PROCESSED_SOLAR / "skippd"
SOLARNET_PROCESSED     = PROCESSED_SOLAR / "solarnet"
GOES16_NSRDB_PROCESSED = PROCESSED_SOLAR / "goes16_nsrdb"

# ---------------------------------------------------------------------------
# Schema constants  (must match src/mmtsfm/data/dataset.py)
# ---------------------------------------------------------------------------
# SKIPP'D
SKIPPD_TARGET_COL = "pv_power"

# Solarnet
SOLARNET_TARGET_COLS    = ["ghi", "dni", "dhi"]
SOLARNET_COVARIATE_COLS = ["air_temp", "relhum", "press", "windsp", "winddir",
                            "max_windsp", "precipitation"]

# GOES-16 / NSRDB
NSRDB_TARGET_COLS    = ["GHI", "DNI", "DHI"]
NSRDB_COVARIATE_COLS = ["Wind Speed", "Temperature", "Pressure"]
NSRDB_KNN            = 8   # k-nearest-neighbour spatial graph
NSRDB_TIME_STEP_S    = 1800  # 30-minute intervals in seconds
