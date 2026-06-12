"""
Full meteorology data pipeline: raw → training-ready.

Pipeline
────────
  Step 1  refactor_meteonet.py      raw CSVs + tar.gz → timeseries.parquet + radar frames
  Step 2  refactor_earthnet2021.py  NPZ samples → timeseries.parquet + Sentinel-2 frames
  Step 3  refactor_era5_eu.py       CDS ZIP → timeseries.parquet + spatial grid frames

Output roots
────────────
  data/refactored/meteorology/meteonet/
  data/refactored/meteorology/earthnet2021/
  data/refactored/meteorology/era5_eu/

Shared output schema
─────────────────────
  timeseries.parquet   — entity_id, timestamp_unix, {target_cols}, {target_cols}_norm,
                         {cov_cols}, {cov_cols}_norm, mask_target, mask_cov
  frames/              — NPZ float16 grids
  frame_index.parquet  — entity_id, timestamp_unix, rel_path, mask_visual
  graph.json           — {nodes, edges, adjacency_matrix}
  metadata.json        — entity list, schema, normalization stats

Usage
─────
  uv run python scripts/meteorology/build_all.py
  uv run python scripts/meteorology/build_all.py \\
      --skip-meteonet \\
      --workers 8 \\
      --radar-step 12 \\
      --max-frames 500 \\
      --earthnet-splits train iid_test_split \\
      --max-samples 500 \\
      --era5-chunk 200 \\
      --data-root /custom/data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_METEO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_METEO_DIR))


def _hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _banner(step: int, total: int, name: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  STEP {step} / {total}  --  {name}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full meteorology data pipeline: raw → training-ready."
    )
    parser.add_argument("--data-root",   type=Path, default=None,
                        help="Override project data root (default: auto-detected).")
    parser.add_argument("--workers",     type=int,  default=4,
                        help="Parallel workers for radar frame extraction.")
    parser.add_argument("--radar-step",  type=int,  default=12,
                        help="Subsample every N-th 5-min radar frame (default 12 = hourly).")
    parser.add_argument("--max-frames",  type=int,  default=None,
                        help="Cap radar frames per region (for testing).")
    parser.add_argument("--skip-meteonet",    action="store_true")
    parser.add_argument("--skip-radar",       action="store_true",
                        help="Skip radar extraction within MeteoNet step.")
    parser.add_argument("--skip-stations",    action="store_true",
                        help="Skip station CSV processing within MeteoNet step.")
    parser.add_argument("--skip-earthnet",    action="store_true",
                        help="Skip EarthNet2021 refactoring step.")
    parser.add_argument("--earthnet-splits",  nargs="+", default=["train"],
                        help="EarthNet2021 splits to process (default: train).")
    parser.add_argument("--max-samples",      type=int, default=None,
                        help="Cap EarthNet2021 samples processed (for testing).")
    parser.add_argument("--skip-en-frames",   action="store_true",
                        help="Skip Sentinel-2 frame NPZ writing in EarthNet step.")
    parser.add_argument("--skip-era5",        action="store_true",
                        help="Skip ERA5 EU refactoring step.")
    parser.add_argument("--skip-era5-frames", action="store_true",
                        help="Skip spatial frame NPZ writing in ERA5 step.")
    parser.add_argument("--era5-chunk",       type=int, default=100,
                        help="Timesteps per streaming write chunk for ERA5 (default 100).")
    parser.add_argument("--era5-max-ts",      type=int, default=None,
                        help="Cap ERA5 timesteps processed (for testing).")
    args = parser.parse_args()

    total_steps = sum([not args.skip_meteonet, not args.skip_earthnet, not args.skip_era5])
    step = 0
    wall = time.time()

    # ── Step 1: MeteoNet ──────────────────────────────────────────────────────
    if not args.skip_meteonet:
        step += 1
        _banner(step, total_steps, "MeteoNet  (France NW + SE, stations + radar)")
        t0 = time.time()
        import refactor_meteonet
        refactor_meteonet.main(
            radar_step=args.radar_step,
            workers=args.workers,
            skip_radar=args.skip_radar,
            skip_stations=args.skip_stations,
            data_root=args.data_root,
            max_frames=args.max_frames,
        )
        print(f"  Elapsed: {_hms(time.time() - t0)}")
    else:
        print("  [--skip-meteonet] Step skipped.")

    # ── Step 2: EarthNet2021 ──────────────────────────────────────────────────
    if not args.skip_earthnet:
        step += 1
        _banner(step, total_steps, "EarthNet2021  (Sentinel-2 + ERA5 patches)")
        t0 = time.time()
        import refactor_earthnet2021
        refactor_earthnet2021.main(
            splits=args.earthnet_splits,
            workers=args.workers,
            max_samples=args.max_samples,
            data_root=args.data_root,
            skip_frames=args.skip_en_frames,
        )
        print(f"  Elapsed: {_hms(time.time() - t0)}")
    else:
        print("  [--skip-earthnet] Step skipped.")

    # ── Step 3: ERA5 EU ───────────────────────────────────────────────────────
    if not args.skip_era5:
        step += 1
        _banner(step, total_steps, "ERA5 EU 2020-2021  (6-hourly gridded analysis)")
        t0 = time.time()
        import refactor_era5_eu
        refactor_era5_eu.main(
            workers=args.workers,
            chunk_size=args.era5_chunk,
            skip_frames=args.skip_era5_frames,
            max_timesteps=args.era5_max_ts,
            data_root=args.data_root,
        )
        print(f"  Elapsed: {_hms(time.time() - t0)}")
    else:
        print("  [--skip-era5] Step skipped.")

    print(f"\n{'=' * 60}")
    print(f"  Meteorology pipeline complete.  Total: {_hms(time.time() - wall)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
