"""
Precompute Qwen3-VL features for Wollongong sky-camera images.

Image layout (per camera, per day):
  Sky Images at Location {1,2}/{DD}_{MM}_{YYYY}/
    Time.txt                # rows: 'idx HH MM SS' (one row per image)
    1 to 999/{N}.jpg
    1000 to 1999/{N}.jpg
    2000 to 2999/{N}.jpg
    (Loc2 may use 'New folder', 'New folder.1', etc. — recursive glob handles this)

Strategy:
  For each day folder:
    1. Parse Time.txt -> {N: (HH, MM, SS)}
    2. Recursive-glob all *.jpg, extract N from filename
    3. For each (HH, MM), keep only the first image of the minute
    4. Encode batch with Qwen3-VL, save per-minute features

Output: one .npy file per minute, shape [1, D] (S=1 single station),
        keyed by `YYYYMMDDHHMM` (matches vision_store lookup format).

Usage (GPU node):
    python tools/precompute_vision_feats_wollongong.py \
        --image_root $SCRATCH/dataset/wollongong \
        --camera     1 \
        --out_dir    $SCRATCH/vision_feats_wollongong_qwen3vl/cam1 \
        --qwen_path  $SCRATCH/QwenQwen3-VL-Embedding-2B
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Project root on sys.path so `src` package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


def _parse_time_txt(path: Path) -> dict[int, tuple[int, int, int]]:
    """Parse 'idx HH MM SS' rows from Time.txt -> {idx: (H, M, S)}."""
    times = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0]); h = int(parts[1])
                m = int(parts[2]);   s = int(parts[3])
                times[idx] = (h, m, s)
            except ValueError:
                continue
    return times


def _parse_day_dir(day_dir: Path) -> str:
    """Day folder name '10_09_2019' -> '20190910'."""
    m = re.match(r'(\d{1,2})_(\d{1,2})_(\d{4})', day_dir.name)
    if not m:
        raise ValueError(f"Cannot parse day from folder: {day_dir.name}")
    dd, mm, yyyy = m.groups()
    return f"{yyyy}{int(mm):02d}{int(dd):02d}"


def _collect_first_image_per_minute(day_dir: Path) -> dict[str, Path]:
    """Return {YYYYMMDDHHMM: image_path} — one image per minute (earliest in minute)."""
    time_txt = day_dir / "Time.txt"
    if not time_txt.exists():
        print(f"  [WARN] No Time.txt in {day_dir}, skipping")
        return {}
    times = _parse_time_txt(time_txt)
    date_str = _parse_day_dir(day_dir)

    # Map filename N -> path
    by_idx: dict[int, Path] = {}
    for p in day_dir.rglob("*.jpg"):
        try:
            n = int(p.stem)
        except ValueError:
            continue
        # Keep first found (shouldn't have duplicates per N, but safe)
        by_idx.setdefault(n, p)

    # Group by minute, keep earliest
    per_minute: dict[str, tuple[int, Path]] = {}   # key -> (seconds_in_minute, path)
    for n, path in by_idx.items():
        if n not in times:
            continue
        h, m, s = times[n]
        dt = datetime.strptime(f"{date_str}{h:02d}{m:02d}", "%Y%m%d%H%M") + timedelta(hours=7)
        key = dt.strftime("%Y%m%d%H%M")
        if key not in per_minute or s < per_minute[key][0]:
            per_minute[key] = (s, path)

    return {k: v[1] for k, v in per_minute.items()}


def load_qwen_embedder(qwen_path: str, device, dtype):
    try:
        from src.SolarVLM.qwen3_vl_embedding import Qwen3VLEmbedder
    except ImportError as e:
        raise RuntimeError(f"Cannot import Qwen3VLEmbedder: {e}")
    embedder = Qwen3VLEmbedder(qwen_path, torch_dtype=dtype)
    embedder.model.to(device).eval()
    return embedder


@torch.no_grad()
def encode_batch(embedder, images, normalize=True):
    inputs = [{"image": im, "text": ""} for im in images]
    emb = embedder.process(inputs, normalize=normalize)
    return emb.detach().cpu().float().numpy().astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", type=str, required=True,
                    help="Root of Wollongong dataset (contains 'Sky Images at Location 1/2')")
    ap.add_argument("--camera",     type=int, choices=[1, 2], required=True,
                    help="Which camera (1 = SBRC near Loc1 PV; 2 = Main Campus near Loc3 PV)")
    ap.add_argument("--out_dir",    type=str, required=True,
                    help="Output dir for .npy features (one per minute)")
    ap.add_argument("--qwen_path",  type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device",     type=str, default="cuda")
    ap.add_argument("--fp16",       type=int, default=1)
    ap.add_argument("--resize",     type=int, default=256,
                    help="Resize images to this (square) before encoding (0=no resize)")
    args = ap.parse_args()

    root = Path(args.image_root)
    # Check if this is the refactored directory (pointing directly to cam1/cam2) or the raw directory
    is_refactored = False
    if root.name in ("cam1", "cam2") or (root / "2019-09-10").exists():
        is_refactored = True
        cam_dir = root
    else:
        cam_dir = root / f"Sky Images at Location {args.camera}"
        if not cam_dir.exists():
            # Fallback check: maybe they passed the parent directory of cam1/cam2 in the refactored layout
            fallback_dir = root / "images" / "wollongong" / f"cam{args.camera}"
            if fallback_dir.exists():
                is_refactored = True
                cam_dir = fallback_dir
            else:
                # Second fallback: maybe they passed refactored root directly
                fallback_dir2 = root / f"images/wollongong/cam{args.camera}"
                if fallback_dir2.exists():
                    is_refactored = True
                    cam_dir = fallback_dir2
                else:
                    raise FileNotFoundError(f"Camera dir not found: {cam_dir} (also checked fallback: {fallback_dir})")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Build minute->image map across all day folders
    minute_map: dict[str, Path] = {}
    if is_refactored:
        print(f"[1/3] Scanning refactored {cam_dir} for images ...")
        per_minute: dict[str, tuple[int, Path]] = {}
        for p in cam_dir.rglob("*.jpg"):
            stem = p.name[:-4] # strip .jpg
            parts = stem.split('_')
            if len(parts) >= 2:
                date_str = parts[0]
                time_str = parts[1]
                if len(date_str) == 8 and len(time_str) == 6:
                    key = f"{date_str}{time_str[:4]}"
                    s = int(time_str[4:6])
                    if key not in per_minute or s < per_minute[key][0]:
                        per_minute[key] = (s, p)
        minute_map = {k: v[1] for k, v in per_minute.items()}
        print(f"  Found {len(minute_map)} unique minutes in refactored layout")
    else:
        print(f"[1/3] Scanning raw {cam_dir} for day folders ...")
        day_dirs = sorted(d for d in cam_dir.iterdir() if d.is_dir())
        for dd in day_dirs:
            m = _collect_first_image_per_minute(dd)
            minute_map.update(m)
            print(f"  {dd.name}: {len(m)} unique minutes")
        print(f"  Total: {len(minute_map)} minutes to encode")

    # Skip already-cached
    todo = [(k, p) for k, p in sorted(minute_map.items())
            if not (out / f"{k}.npy").exists()]
    print(f"  Cached: {len(minute_map) - len(todo)}  |  To encode: {len(todo)}")
    if not todo:
        print("All features already cached. Exiting.")
        return

    # 2) Load Qwen
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.fp16 else torch.float32
    print(f"[2/3] Loading Qwen3-VL on {device} ...")
    embedder = load_qwen_embedder(args.qwen_path, device, dtype)

    # 3) Encode in batches
    print(f"[3/3] Encoding {len(todo)} images ...")
    batch_imgs, batch_keys, saved = [], [], 0

    def flush():
        nonlocal saved
        feats = encode_batch(embedder, batch_imgs)
        for key, feat in zip(batch_keys, feats):
            np.save(out / f"{key}.npy", feat[np.newaxis, :])  # [1, D]
            saved += 1
        batch_imgs.clear(); batch_keys.clear()

    for i, (key, path) in enumerate(todo):
        try:
            img = Image.open(path).convert("RGB")
            if args.resize > 0:
                img = img.resize((args.resize, args.resize), Image.BILINEAR)
            img.load()
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (args.resize or 64, args.resize or 64), 0)
        batch_imgs.append(img)
        batch_keys.append(key)

        if len(batch_imgs) >= args.batch_size:
            flush()
            if saved % 200 == 0:
                print(f"  {saved}/{len(todo)} encoded ...")

    if batch_imgs:
        flush()

    print(f"\nDone. {saved} features saved to {out}")


if __name__ == "__main__":
    main()
