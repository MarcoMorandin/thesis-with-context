"""
Full solar data pipeline: raw → training-ready → unified window index.

Pipeline
────────
  Step 1  refactor_skippd.py        raw parquets → timeseries.parquet + frame_index
  Step 2  refactor_solarnet.py      raw CSVs + tar.bz2 → timeseries.parquet + frames
  Step 3  refactor_goes16_nsrdb.py  NSRDB CSVs + GOES-16 NetCDF → timeseries.parquet + frames
  Step 4  build_unified.py          sliding-window index across all datasets

Output roots
────────────
  data/refactored/solar/skippd/
  data/refactored/solar/solarnet/
  data/refactored/solar/goes16_nsrdb/
  data/refactored/unified/          ← train/val/test_windows.parquet + schema.json

Shared output schema (per dataset)
───────────────────────────────────
  timeseries.parquet   — entity_id, timestamp_unix, {target_cols}, {target_cols}_norm,
                         [{cov_cols}, {cov_cols}_norm], mask_target, [mask_cov]
  frames/              — JPEG (sky cam) or NPZ float16 (satellite) indexed by timestamp
  frame_index.parquet  — entity_id, timestamp_unix, rel_path, mask_visual
  graph.json           — {nodes, edges, [adjacency_matrix]}
  metadata.json        — entity list, schema, normalization stats

Usage
─────
  python scripts/solar/build_all.py            # all datasets, default settings
  python scripts/solar/build_all.py \\
      --skip-skippd \\
      --skip-solarnet \\
      --skip-unified \\
      --workers 8 \\
      --img-size 224 \\
      --goes16-bands C02,C07,C13 \\
      --max-goes16-frames 500 \\
      --data-root /custom/data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the solar scripts directory is on the path for sibling imports
_SOLAR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SOLAR_DIR))


def _hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _banner(step: int, total: int, name: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  STEP {step} / {total}  —  {name}")
    print("━" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full solar data pipeline: raw → training-ready → unified."
    )
    parser.add_argument("--data-root",      type=Path, default=None,
                        help="Override project data root (default: auto-detected).")
    parser.add_argument("--workers",        type=int, default=4,
                        help="Parallel workers for image/CSV processing.")
    parser.add_argument("--img-size",       type=int, default=224,
                        help="Resize images to N×N pixels (SKIPPD + Solarnet).")
    parser.add_argument("--jpeg-quality",   type=int, default=90,
                        help="JPEG quality for sky-cam frames.")
    # GOES-16 specific
    parser.add_argument("--goes16-bands",   type=str, default="C02,C07,C13",
                        help="Comma-separated GOES-16 ABI band names.")
    parser.add_argument("--goes16-img-size",type=int, default=0,
                        help="Resize GOES-16 crops (0 = keep native resolution).")
    parser.add_argument("--max-goes16-frames", type=int, default=None,
                        help="Cap GOES-16 frames processed (for testing).")
    # Unified index
    parser.add_argument("--hist-steps",    type=int, default=168,
                        help="Historical context length in timesteps.")
    parser.add_argument("--horizon",       type=int, default=24,
                        help="Forecast horizon in timesteps.")
    parser.add_argument("--visual-window", type=int, default=6,
                        help="Number of visual frames per sample.")
    # Skip flags
    parser.add_argument("--skip-skippd",    action="store_true")
    parser.add_argument("--skip-solarnet",  action="store_true")
    parser.add_argument("--skip-goes16",    action="store_true")
    parser.add_argument("--skip-unified",   action="store_true")
    args = parser.parse_args()

    total_steps = sum([
        not args.skip_skippd,
        not args.skip_solarnet,
        not args.skip_goes16,
        not args.skip_unified,
    ])
    step = 0
    wall = time.time()

    # ── Step 1: SKIPP'D ───────────────────────────────────────────────────────
    if not args.skip_skippd:
        step += 1
        _banner(step, total_steps, "SKIPP'D  (sky cam + PV power)")
        t0 = time.time()
        import refactor_skippd
        refactor_skippd.main(
            img_size=args.img_size,
            quality=args.jpeg_quality,
            data_root=args.data_root,
            workers=args.workers,
        )
        print(f"  Elapsed: {_hms(time.time() - t0)}")
    else:
        print("  [--skip-skippd] Step skipped.")

    # ── Step 2: Solarnet ──────────────────────────────────────────────────────
    if not args.skip_solarnet:
        step += 1
        _banner(step, total_steps, "Solarnet  (Folsom CA, sky cam + irradiance)")
        t0 = time.time()
        import refactor_solarnet
        refactor_solarnet.main(
            img_size=args.img_size,
            quality=args.jpeg_quality,
            data_root=args.data_root,
            workers=args.workers,
        )
        print(f"  Elapsed: {_hms(time.time() - t0)}")
    else:
        print("  [--skip-solarnet] Step skipped.")

    # ── Step 3: GOES-16 + NSRDB ───────────────────────────────────────────────
    if not args.skip_goes16:
        step += 1
        _banner(step, total_steps, "GOES-16 + NSRDB  (satellite + irradiance grid)")
        t0 = time.time()
        import refactor_goes16_nsrdb
        refactor_goes16_nsrdb.main(
            img_size=args.goes16_img_size,
            bands=args.goes16_bands.split(","),
            workers=args.workers,
            data_root=args.data_root,
            max_frames=args.max_goes16_frames,
        )
        print(f"  Elapsed: {_hms(time.time() - t0)}")
    else:
        print("  [--skip-goes16] Step skipped.")

    # ── Step 4: Unified window index ──────────────────────────────────────────
    if not args.skip_unified:
        step += 1
        _banner(step, total_steps, "Unified window index  (train/val/test splits)")
        t0 = time.time()
        import build_unified
        build_unified.main(
            hist_steps=args.hist_steps,
            horizon=args.horizon,
            visual_window=args.visual_window,
            data_root=args.data_root,
        )
        print(f"  Elapsed: {_hms(time.time() - t0)}")
    else:
        print("  [--skip-unified] Step skipped.")

    print(f"\n{'=' * 60}")
    print(f"  Solar pipeline complete.  Total: {_hms(time.time() - wall)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
