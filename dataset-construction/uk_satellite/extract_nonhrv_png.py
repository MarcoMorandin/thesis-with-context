"""uk_pv satellite crops from EUMETSAT SEVIRI non-HRV, as 24h-alive RGB PNGs.

Replaces extract_128.py, which read `{year}_hrv.zarr`. HRV is High Resolution
*Visible*: reflected sunlight, black at night by physics, and extract_128.py then
dropped the all-NaN night patches it produced. Because uk_pv windows sit at
stride 12 on a 30-minute grid there are exactly two origins per site per day
(07:30 and 13:30 UTC) against a 6h BACKWARD visual window, so half of every
visual batch was V-JEPA's embedding of a blank frame -- non-zero, therefore
invisible to an all-zero check. See ../README.md for the measurements.

This reads `{year}_nonhrv.zarr`: same bucket, same path, same 104,807 timesteps,
11 channels of which 8 are emissive and so alive at night.

Three decisions, each measured rather than assumed (../README.md):

* **Difference-free RGB.** The standard EUMETSAT 24h-Microphysics recipe uses
  channel differences over 6 K beams, but OCF stores [0,1] float16, which
  quantises IR_120 at 0.325 K -- several percent of beam width, and visibly
  speckled (speckle index 0.451 day / 0.313 night, vs 0.129 / 0.057 here). Three
  emissive channels taken straight avoid the amplification entirely.
* **Same recipe day and night.** All three channels are emissive, so there is no
  sunrise discontinuity for the encoder to model around.
* **128x128 at ~3 km/px is ~384 km**, against ~128 km for the old HRV crop.
  Clouds advect 120-360 km over the 6h lookback, so the old crop mostly excluded
  the air that was about to arrive.

Output mirrors the GOES layout exactly -- `uk_pv_<site>/<ISO>Z.png`, uint8 RGB --
so pack_images.py can consume both with one code path.

    uv run python extract_nonhrv_png.py --out /Volumes/dataset/uk_nonhrv_png
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dask
import gcsfs
import numcodecs
import numpy as np
import ocf_blosc2
import pandas as pd
import xarray as xr
from PIL import Image
from pyproj import Transformer

numcodecs.registry.register_codec(ocf_blosc2.Blosc2)

BUCKET = (
    "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/"
    "SEVIRI_RSS/v4/{year}_nonhrv.zarr"
)

# openclimatefix/Satip, satip/scale_to_zero_to_one.py -- the affine OCF applied
# before writing. Values in the store are NOT kelvin; the stretch ranges below
# are, so this has to be undone first. Verified: IR_087/108/120 then agree at
# 268-271 K on the same pixel.
_ORDER = [
    "HRV",
    "IR_016",
    "IR_039",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
    "VIS006",
    "VIS008",
    "WV_062",
    "WV_073",
]
_MINS = [
    -1.2278595,
    -2.5118103,
    -64.83977,
    63.404694,
    2.844452,
    199.10002,
    -17.254883,
    -26.29155,
    -1.1009827,
    -2.4184198,
    199.57048,
    198.95093,
]
_MAXS = [
    103.90016,
    69.60857,
    339.15588,
    340.26526,
    317.86752,
    313.2767,
    315.99194,
    274.82297,
    93.786545,
    101.34922,
    249.91806,
    286.96323,
]
DENORM = {n: (_MINS[i], _MAXS[i]) for i, n in enumerate(_ORDER)}

# (channel, low_K, high_K) -> R, G, B. All emissive.
#
# Ranges are the p0.5-p99.5 of the ACTUAL brightness temperatures over the UK
# bounding box, sampled across 28 scans spanning both years and all seasons --
# not the textbook SEVIRI ranges, which are set for the full disc and run far too
# warm here. The first run used those textbook values and paid for it: measured
# on 1,500 written frames, WV_073 had 26% of its pixels pinned at 0, IR_108 8%,
# WV_062 16%. For a water-vapour channel a LOW brightness temperature means
# moisture and cloud high in the atmosphere, so clipping the cold end destroys
# exactly the structure the probe is meant to find, in every frame.
#
#   channel   actual p0.5-p99.5   old range   clipped low
#   IR_108      212.7 - 293.2      235-300       7.4%
#   WV_073      212.6 - 261.4      240-275      23.6%
#   WV_062      210.7 - 243.7      225-245      15.9%
#
# The cost is a coarser quantisation step (WV_073: 0.14 -> 0.20 K per level), but
# that is still only ~2x the store's own float16 step of 0.086 K, and losing
# resolution is recoverable where clipping is not.
RECIPE = [
    ("IR_108", 212.0, 294.0),  # thermal window: cloud-top temperature
    ("WV_073", 212.0, 262.0),  # low/mid-level moisture
    ("WV_062", 210.0, 244.0),  # upper-level flow
]
CHANNELS = [c for c, _, _ in RECIPE]

CROP = 128  # px per side -> ~384 km at ~3 km/px
STEP_MIN = 30  # match the PV grid
LOOKBACK_H = 6  # visual_window_hours
GEOS_PROJ = "+proj=geos +lon_0=9.5 +h=35785831 +x_0=0 +y_0=0 +a=6378169 +b=6356583.8"


def stretch(k: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((k - lo) / (hi - lo), 0.0, 1.0)


def to_kelvin(norm: np.ndarray, name: str) -> np.ndarray:
    lo, hi = DENORM[name]
    return norm.astype("float32") * (hi - lo) + lo


def site_table(parquet: Path) -> pd.DataFrame:
    """One row per uk_pv plant with its coordinates, from the dataset of record.

    Taken from the parquet rather than the HF metadata so the plant set and the
    coordinates cannot drift from the table the model is actually scored on.
    """
    df = pd.read_parquet(
        parquet, columns=["dataset", "site_id", "latitude", "longitude"]
    )
    df = df[df["dataset"] == "uk_pv"]
    df["site_id"] = df["site_id"].astype(str)
    return df.groupby("site_id", as_index=False)[["latitude", "longitude"]].first()


def needed_timestamps(
    parquet: Path, step_min: int = STEP_MIN, lookback_h: float = LOOKBACK_H
) -> np.ndarray:
    """Every slot some PV row's visual window reaches back into.

    Extracting a flat 24h/day would fetch frames no window ever reads. Taking the
    union of [t - lookback_h, t] over the real PV timestamps keeps the download
    to what the model can actually consume, and stays correct if the origin
    sampling is changed later (the union only grows toward 24h).

    `step_min` is the frame CADENCE, not the model's frame spacing. Extract at
    the finest cadence you might want and choose the spacing at training time by
    subsampling: the store is 5-minutely and its time chunks are 12 steps, all of
    which are fetched whether or not they are used, so a finer cadence is nearly
    free on the wire (it costs PNG count and disk, not download). Extracting at
    30 min and later wanting 15 means downloading everything again.
    """
    df = pd.read_parquet(parquet, columns=["dataset", "timestamp_utc"])
    t = pd.to_datetime(df.loc[df["dataset"] == "uk_pv", "timestamp_utc"], utc=True)
    step = pd.Timedelta(minutes=step_min)
    back = int(round(lookback_h * 60 / step_min))
    base = t.dt.floor(f"{step_min}min").unique()
    slots = {b - step * k for b in base for k in range(back + 1)}
    return np.array(sorted(slots), dtype="datetime64[ns]")


def geo_index(ds: xr.Dataset, lats, lons) -> tuple[np.ndarray, np.ndarray]:
    tf = Transformer.from_crs("EPSG:4326", GEOS_PROJ, always_xy=True)
    gx, gy = tf.transform(np.asarray(lons), np.asarray(lats))
    xs = ds.x_geostationary.values
    ys = ds.y_geostationary.values
    ix = np.array([int(np.abs(xs - v).argmin()) for v in np.atleast_1d(gx)])
    iy = np.array([int(np.abs(ys - v).argmin()) for v in np.atleast_1d(gy)])
    return iy, ix


def load_done(out: Path) -> set[str]:
    """Slots already fully written, from the completion log.

    A slot is 100 files; probing all of them costs 2.1M stat calls over a full
    run, and probing only the first would call a half-written slot complete.
    The log is appended AFTER a slot's files are all closed, so a crash loses at
    most one slot -- which is then simply redone.
    """
    f = out / "_completed.txt"
    return set(f.read_text().split()) if f.exists() else set()


def mark_done(fh, stamp: str) -> None:
    fh.write(stamp + "\n")
    fh.flush()


def build_rgb(slab: np.ndarray, ch_pos: dict[str, int]) -> tuple[np.ndarray, int]:
    """slab [y, x, variable] -> (uint8 [y, x, 3], count of non-finite inputs).

    A handful of archive scans carry NaN pixels. np.clip PROPAGATES NaN and
    casting NaN to uint8 is undefined behaviour in C, so those pixels used to
    become unpredictable bytes with nothing but a RuntimeWarning to show for it.
    They are now pinned to the low end explicitly, and counted, so the run can
    report how much of the output was affected instead of leaving it to a warning
    that Python de-duplicates after the first occurrence.
    """
    planes, n_bad = [], 0
    for name, lo, hi in RECIPE:
        k = to_kelvin(slab[..., ch_pos[name]], name)
        bad = ~np.isfinite(k)
        n_bad += int(bad.sum())
        planes.append(stretch(np.where(bad, lo, k), lo, hi))
    return (np.dstack(planes) * 255.0).astype(np.uint8), n_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="/Users/marcomorandin/Desktop/thesis-dataset/dataset_all.parquet",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", default="2019,2020")
    ap.add_argument("--max-sites", type=int, default=None)
    ap.add_argument(
        "--limit-slots",
        type=int,
        default=None,
        help="stop after N time slots; for smoke tests",
    )
    ap.add_argument(
        "--block",
        type=int,
        default=192,
        help="time slots per network read. Bigger is faster: one isel is one dask\n"
        "graph, so its chunks are fetched concurrently. Measured 0.50 slot/s at\n"
        "48 vs 1.05 at 192; 192 holds ~380 MB of float16 in memory.",
    )
    ap.add_argument("--workers", type=int, default=16, help="PNG encoder threads")
    ap.add_argument(
        "--step-min",
        type=int,
        default=STEP_MIN,
        help="frame cadence in minutes. Must divide 5 evenly into the store's\n"
        "5-minutely grid. Finer is nearly free on the wire (whole 12-step chunks\n"
        "are fetched regardless) but multiplies PNG count and disk.",
    )
    ap.add_argument(
        "--lookback-h",
        type=float,
        default=LOOKBACK_H,
        help="how far back a visual window may reach. Extract the WIDEST you may\n"
        "want; the model can always use a shorter span by taking fewer frames.",
    )
    ap.add_argument(
        "--fetch-workers",
        type=int,
        default=32,
        help="concurrent chunk fetches. Chunks are 2.64 MB behind object-store\n"
        "latency, so throughput is concurrency-bound, not bandwidth-bound, and\n"
        "dask's default of one thread per core badly under-drives it. Measured\n"
        "on a 1 GB slab: 8->29.7 MB/s, 32->88.2, 64->82.3, 128->77.7. Past ~32\n"
        "the connections contend and it gets slower again.",
    )
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sites = site_table(Path(a.parquet))
    if a.max_sites:
        sites = sites.head(a.max_sites)
    slots = needed_timestamps(Path(a.parquet), a.step_min, a.lookback_h)
    if a.limit_slots:
        slots = slots[: a.limit_slots]
    print(
        f"{len(sites)} plants, {len(slots)} time slots ({slots[0]} .. {slots[-1]})",
        flush=True,
    )

    fs = gcsfs.GCSFileSystem(token="anon")
    written = skipped = n_nonfinite = 0
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=a.workers)
    done_fh = (out / "_completed.txt").open("a")

    for year in [int(y) for y in a.years.split(",")]:
        t_year = time.time()
        ds = xr.open_zarr(
            fs.get_mapper(BUCKET.format(year=year)), consolidated=True, chunks={}
        )
        names = [str(v) for v in np.asarray(ds.variable)]
        missing = [c for c in CHANNELS if c not in names]
        assert not missing, f"{year}: channels absent from the store: {missing}"
        # The block read below relies on .sel(variable=CHANNELS) returning the
        # planes in CHANNELS order, which is what indexes RECIPE. If that ever
        # stopped holding, the RGB would silently swap channels -- still a
        # plausible-looking image, and undetectable downstream.
        got = [str(v) for v in np.asarray(ds.sel(variable=CHANNELS).variable)]
        assert got == CHANNELS, f"sel reordered channels: {got} != {CHANNELS}"
        iy, ix = geo_index(ds, sites["latitude"].values, sites["longitude"].values)
        r = CROP // 2
        # One bounding box over every plant, so a time block is fetched once and
        # sliced 100 ways rather than re-fetched per plant.
        y0, y1 = int(iy.min() - r), int(iy.max() + r)
        x0, x1 = int(ix.min() - r), int(ix.max() + r)
        print(
            f"  {year}: bbox y[{y0}:{y1}] x[{x0}:{x1}] = {y1 - y0}x{x1 - x0} px",
            flush=True,
        )

        stamps = np.asarray(ds.time)
        want = slots[(slots >= stamps[0]) & (slots <= stamps[-1])]
        # Snap to the nearest available scan; RSS is 5-minutely so a 30-minute
        # slot always has one, but gaps in the archive must not shift the crop
        # silently onto a distant time.
        pos = np.searchsorted(stamps, want)
        pos = np.clip(pos, 0, len(stamps) - 1)
        delta = np.abs(stamps[pos] - want).astype("timedelta64[m]").astype(int)
        # Half a cadence step: at 15-min cadence a 15-min tolerance would let
        # two adjacent slots snap onto the SAME scan and silently write the
        # same picture under two different timestamps.
        tol = max(1, a.step_min // 2)
        ok = delta <= tol
        if (~ok).any():
            print(
                f"  {year}: {int((~ok).sum())} slots have no scan within "
                f"{tol} min; skipped",
                flush=True,
            )
        want, pos = want[ok], pos[ok]

        for sid in sites["site_id"].values:
            (out / f"uk_pv_{sid}").mkdir(parents=True, exist_ok=True)
        site_ids = sites["site_id"].values
        done = load_done(out)

        # Read in BLOCKS, not per slot. The zarr time chunk is 12 steps (1 hour
        # at 5-minutely RSS) and the wanted 30-minute slots land ~1.93 per hour,
        # so fetching one timestep at a time pulled every chunk about twice and
        # discarded 11 of its 12 steps. A block-sized isel is one dask graph, so
        # its chunks are also fetched concurrently instead of strictly serially.
        for lo in range(0, len(want), a.block):
            hi = min(lo + a.block, len(want))
            blk_ts, blk_pos = want[lo:hi], pos[lo:hi]
            stamps_out = [
                pd.Timestamp(t).strftime("%Y-%m-%dT%H-%M-%SZ") for t in blk_ts
            ]
            todo = [i for i, s in enumerate(stamps_out) if s not in done]
            if not todo:
                skipped += len(stamps_out) * len(site_ids)
                continue
            t_lo, t_hi = int(blk_pos[todo[0]]), int(blk_pos[todo[-1]]) + 1
            with dask.config.set(scheduler="threads", num_workers=a.fetch_workers):
                block = np.asarray(
                    ds["data"]
                    .isel(
                        time=slice(t_lo, t_hi),
                        y_geostationary=slice(y0, y1),
                        x_geostationary=slice(x0, x1),
                    )
                    .sel(variable=CHANNELS)
                    .compute()
                )
            local = {n: i for i, n in enumerate(CHANNELS)}
            for i in todo:
                slab = block[int(blk_pos[i]) - t_lo]
                stamp = stamps_out[i]
                jobs = []
                for k, sid in enumerate(site_ids):
                    cy, cx = int(iy[k] - y0), int(ix[k] - x0)
                    win = slab[cy - r : cy + r, cx - r : cx + r, :]
                    if win.shape[:2] != (CROP, CROP):
                        continue
                    rgb, nb = build_rgb(win, local)
                    n_nonfinite += nb
                    jobs.append((out / f"uk_pv_{sid}" / f"{stamp}.png", rgb))
                # PIL's PNG encoder releases the GIL, so threads genuinely
                # overlap here; encoding 2.1M frames serially would cost hours.
                list(pool.map(lambda j: Image.fromarray(j[1]).save(j[0]), jobs))
                written += len(jobs)
                mark_done(done_fh, stamp)
            n = hi
            if (lo // a.block) % 20 == 0:
                # Against t_year, not the run-wide t0. `n` restarts at 0 for each
                # year, so dividing it by run-wide elapsed makes the first tick of
                # the second year report the whole of the first year's runtime as
                # if it produced 384 slots: 0.06 slot/s and an 87 h ETA, which
                # reads as a catastrophic stall rather than a fresh counter.
                el = time.time() - t_year
                rate = n / el if el else 0
                print(
                    f"  {year} {n}/{len(want)} slots  {written} png  "
                    f"{rate:.2f} slot/s  eta {(len(want) - n) / max(rate, 1e-9) / 3600:.1f} h",
                    flush=True,
                )

    print(f"non-finite source pixels pinned to range low: {n_nonfinite}", flush=True)
    print(
        f"done: {written} written, {skipped} already present, "
        f"{(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
