# tools/precompute_vision_feats_ukpv.py
"""Offline Qwen3-VL vision features for the uk_pv multimodal track.

Unlike the original precompute (crop a ROI around each station's lat/lon from one
large shared sky image), our dataset of record already stores ONE satellite frame
PER PLANT in images_all.h5 (group ``uk_pv_<site>`` → ``images`` uint8, indexed by
the canonical ``image_h5_index``). So each plant's frame already *is* its ROI —
we simply encode it with Qwen3-VL-Embedding-2B.

Grouping/alignment is shared with the data loader (``iter_usable_groups``), so the
``<group_idx>`` and timestamps written here match the loader's ts_keys exactly.
For each usable group ``gi`` and each timestamp ``t`` in its common index we save
``<out_dir>/<gi>__<YYYYMMDDHHMM>.npy`` of shape ``[num_stations, D]`` (the S group
members' frame embeddings). Train-time multi-frame context is reconstructed by
``VisionFeatureStore.get_sequence`` walking back ``num_frames`` timestamps.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOLARVLM = HERE.parent
sys.path.insert(0, str(SOLARVLM))                 # solar_vlm pkg root
sys.path.insert(0, str(SOLARVLM.parents[2]))      # baselines/

from common import config as cfg                                  # noqa: E402
from data_provider.data_loader_ukpv import (                      # noqa: E402
    load_split_frame_df, iter_usable_groups)

EMB_DIM_FALLBACK = 2048


def _frame_to_rgb(arr: np.ndarray):
    """uk_pv frame (H,W) uint8 grayscale or (H,W,3) → PIL RGB."""
    from PIL import Image
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    return Image.fromarray(a.astype(np.uint8), mode="RGB")


@torch.no_grad()
def _encode(embedder, images, batch_size: int, normalize: bool) -> np.ndarray:
    outs = []
    for s in range(0, len(images), batch_size):
        chunk = images[s:s + batch_size]
        emb = embedder.process([{"image": im, "text": ""} for im in chunk],
                               normalize=normalize)
        outs.append(emb.detach().cpu())
    return torch.cat(outs, dim=0).float().numpy().astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset_all.parquet")
    ap.add_argument("--h5", required=True, help="images_all.h5")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--qwen_path", required=True)
    ap.add_argument("--flag", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--dataset", default="uk_pv")
    ap.add_argument("--num_stations", type=int, default=8)
    ap.add_argument("--min_len", type=int, default=cfg.HISTORY_STEPS + cfg.HORIZON_STEPS)
    ap.add_argument("--batch_images", type=int, default=32)
    ap.add_argument("--normalize", type=int, default=1)
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.float32
    from src.SolarVLM.qwen3_vl_embedding import Qwen3VLEmbedder
    embedder = Qwen3VLEmbedder(args.qwen_path, torch_dtype=dtype)
    embedder.model.to(device).eval()

    import h5py
    flags = ["train", "val", "test"] if args.flag == "all" else [args.flag]
    h5 = h5py.File(args.h5, "r")
    for flag in flags:
        df = load_split_frame_df(args.data, flag, args.dataset, list(cfg.COV_COLS))
        if df.empty:
            print(f"[precompute] {flag}: no frame-bearing rows, skip")
            continue
        usable = iter_usable_groups(df, args.num_stations, args.min_len)
        print(f"[precompute] {flag}: {len(usable)} usable groups")
        for gi, (group, common) in enumerate(usable):
            # per (site) timestamp -> frame index
            fmap = {}
            for site in dict.fromkeys(group):
                g = df[df[cfg.SITE_COL] == site]
                fmap[site] = dict(zip(g[cfg.TIME_COL],
                                      g[cfg.FRAME_INDEX_COL].astype(int)))
            for t in common:
                key = f"{gi}__{t.strftime('%Y%m%d%H%M')}"
                out_path = os.path.join(args.out_dir, f"{key}.npy")
                if os.path.exists(out_path):
                    continue
                imgs = []
                for site in group:                     # length S (padded repeats)
                    grp = h5[f"{args.dataset}_{site}"]
                    fi = fmap[site].get(t)
                    if fi is None:
                        imgs.append(_frame_to_rgb(np.zeros((128, 128), np.uint8)))
                    else:
                        imgs.append(_frame_to_rgb(grp["images"][fi]))
                try:
                    feats = _encode(embedder, imgs, args.batch_images,
                                    bool(args.normalize))           # [S, D]
                except Exception as e:                              # noqa: BLE001
                    print(f"[precompute] encode failed {key}: {e}")
                    feats = np.zeros((args.num_stations, EMB_DIM_FALLBACK), np.float32)
                np.save(out_path, feats.astype("float32"))
        print(f"[precompute] {flag}: done → {args.out_dir}")
    h5.close()
    print("✓ UKPV vision features precomputed.")


if __name__ == "__main__":
    main()
