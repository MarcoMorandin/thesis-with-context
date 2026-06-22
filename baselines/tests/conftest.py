"""Synthetic PV data for contract tests — no SSD dataset required."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from common import config
from common.windows import WindowDataset, build_site_series

CADENCE_MIN = 30
STEPS_PER_DAY = 24 * 60 // CADENCE_MIN

# Optional third-party deps per baseline; tests skip instead of erroring
# when a dependency group is not installed (e.g. `uv sync --group tier3`).
# The chronos2 baselines use the official ``chronos-forecasting`` package
# (imported as ``chronos``).
OPTIONAL_BASELINE_DEPS: dict[str, tuple[str, ...]] = {
    "tabpfn": ("tabpfn",),
    "chronos2_zs": ("chronos",),
    "chronos2_ft": ("chronos",),
    "chronos2_oracle": ("chronos",),
}


def skip_if_deps_missing(name: str) -> None:
    missing = [
        mod for mod in OPTIONAL_BASELINE_DEPS.get(name, ())
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        pytest.skip(f"{name}: optional dependencies not installed: {missing}")


def clearsky_curve(hours: np.ndarray) -> np.ndarray:
    """Simple smooth daytime bump: 1000 W/m² at noon, 0 outside 6h-18h."""
    return 1000.0 * np.clip(np.sin(np.pi * (hours - 6.0) / 12.0), 0.0, None)


def make_frame(
    n_sites: int = 3,
    days: int = 6,
    seed: int = 0,
    kt: float | None = None,
    nan_fraction: float = 0.02,
) -> pd.DataFrame:
    """Curated-parquet-shaped frame with plausible synthetic PV physics.

    With ``kt`` fixed, norm_power == kt * clearsky/1000 exactly (clear-sky
    day) — useful for exactness tests on smart persistence.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_sites):
        times = pd.date_range(
            "2021-06-01", periods=days * STEPS_PER_DAY, freq=f"{CADENCE_MIN}min",
            tz="UTC",
        )
        hours = times.hour + times.minute / 60.0
        cs = clearsky_curve(hours.to_numpy())
        if kt is None:
            # slow cloud random walk per site, clipped to [0.2, 1.0]
            walk = np.clip(
                0.7 + np.cumsum(rng.normal(0, 0.03, len(times))), 0.2, 1.0
            )
        else:
            walk = np.full(len(times), kt)
        power = walk * cs / 1000.0
        if nan_fraction > 0:
            drop = rng.random(len(times)) < nan_fraction
            power = np.where(drop, np.nan, power)
        doy = times.dayofyear.to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "dataset": "synth",
                    "site_id": f"site_{s}",
                    "timestamp_utc": times,
                    "norm_power": power.astype(np.float32),
                    "installed_power_w": 1000.0,
                    "bad_site_flag": False,
                    "temperature_2m": 15.0 + 10 * np.sin(np.pi * hours / 24),
                    "cloudcover": (1 - walk) * 100.0,
                    "windspeed_10m": rng.uniform(0, 10, len(times)),
                    "precipitation": 0.0,
                    "shortwave_radiation": walk * cs,
                    "direct_radiation": 0.7 * walk * cs,
                    "diffuse_radiation": 0.3 * walk * cs,
                    "direct_normal_irradiance": 0.8 * walk * cs,
                    "solar_zenith": 90.0 - 70.0 * cs / 1000.0,
                    "solar_azimuth": (hours / 24.0) * 360.0,
                    "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                    "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                    "solar_time": hours,
                    "clearsky_ghi": cs,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def windows_for(df: pd.DataFrame, **kwargs) -> WindowDataset:
    kwargs.setdefault("stride", 4)
    return WindowDataset(build_site_series(df), **kwargs)


@pytest.fixture(scope="session")
def synth_windows() -> WindowDataset:
    return windows_for(make_frame())


@pytest.fixture(scope="session")
def fit_datasets() -> tuple[WindowDataset, WindowDataset, WindowDataset]:
    """Disjoint-plant train/val/test windows from synthetic data."""
    df = make_frame(n_sites=5, days=6)
    sites = lambda *ids: df[df.site_id.isin(ids)]  # noqa: E731
    return (
        windows_for(sites("site_0", "site_1", "site_2")),
        windows_for(sites("site_3")),
        windows_for(sites("site_4")),
    )
