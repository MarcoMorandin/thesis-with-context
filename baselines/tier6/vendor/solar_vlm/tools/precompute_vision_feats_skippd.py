"""
Precompute Qwen3-VL vision features for SKIPPD sky-camera images.

SKIPPD images are 64x64 RGB PNGs, one per minute, single station.
Saves each timestep as {YYYYMMDDHHMM}.npy with shape [1, D] (S=1, D=embed_dim).
The vision_store.get_sequence() assembles n_frames by looking up hourly keys.

Usage (GPU node):
    python tools/precompute_vision_feats_skippd.py \
        --image_dir  $SCRATCH/dataset/skippd/images \
        --out_dir    $SCRATCH/vision_feats_skippd_qwen3vl \
        --qwen_path  $SCRATCH/QwenQwen3-VL-Embedding-2B \
        --batch_size 256
"""

import argparse
import glob
import os
import re
import sys

# Ensure project root is on path so `src` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


def extract_ts_key(path: str) -> str:
    """Parse YYYYMMDDHHMM from filenames like '20170309_0647000800.jpg'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    # Remove separators, take first 12 digits = YYYYMMDDHHMM
    digits = re.sub(r'\D', '', stem)
    return digits[:12] if len(digits) >= 12 else digits


def load_qwen_embedder(qwen_path: str, device: torch.device, dtype: torch.dtype):
    try:
        from src.SolarVLM.qwen3_vl_embedding import Qwen3VLEmbedder
    except ImportError as e:
        raise RuntimeError(f"Cannot import Qwen3VLEmbedder: {e}")
    embedder = Qwen3VLEmbedder(qwen_path, torch_dtype=dtype)
    embedder.model.to(device).eval()
    return embedder


@torch.no_grad()
def encode_batch(embedder, images, normalize: bool = True) -> np.ndarray:
    inputs = [{"image": img, "text": ""} for img in images]
    emb = embedder.process(inputs, normalize=normalize)
    return emb.detach().cpu().float().numpy().astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir",  type=str, required=True)
    ap.add_argument("--out_dir",    type=str, required=True)
    ap.add_argument("--qwen_path",  type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device",     type=str, default="cuda")
    ap.add_argument("--fp16",       type=int, default=1)
    ap.add_argument("--normalize",  type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) +
                       glob.glob(os.path.join(args.image_dir, "*.png")))
    assert img_files, f"No images found in {args.image_dir}"
    
    print(f"[1/3] Found {len(img_files)} images")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.fp16 else torch.float32
    
    print(f"[2/3] Loading Qwen3-VL from {args.qwen_path} on {device} ...")
    embedder = load_qwen_embedder(args.qwen_path, device=device, dtype=dtype)

    print(f"[3/3] Encoding {len(img_files)} images ...")
    batch_imgs, batch_keys = [], []
    saved = 0

    def flush():
        nonlocal saved
        feats = encode_batch(embedder, batch_imgs, normalize=bool(args.normalize))
        # feats: [B, D]
        for key, feat in zip(batch_keys, feats):
            out_path = os.path.join(args.out_dir, f"{key}.npy")
            # Shape [1, D] — S=1 (single station)
            np.save(out_path, feat[np.newaxis, :])
            saved += 1
        batch_imgs.clear()
        batch_keys.clear()

    for i, path in enumerate(img_files):
        ts_key = extract_ts_key(path)
        out_path = os.path.join(args.out_dir, f"{ts_key}.npy")
        if os.path.exists(out_path):
            continue
        try:
            img = Image.open(path).convert("RGB")
            img.load()
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (64, 64), 0)

        batch_imgs.append(img)
        batch_keys.append(ts_key)

        if len(batch_imgs) >= args.batch_size:
            prev = saved
            flush()
            if (saved // 5000) > (prev // 5000):
                pct = saved / len(img_files) * 100
                print(f"  {saved}/{len(img_files)} ({pct:.1f}%) saved ...", flush=True)

    if batch_imgs:
        flush()

    print(f"Done. {saved} features saved to {args.out_dir}")


if __name__ == "__main__":
    main()
