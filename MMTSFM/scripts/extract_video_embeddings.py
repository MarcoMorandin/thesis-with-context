"""Pre-extract visual latents (VidTok or V-JEPA 2.1) for a multimodal dataset.

Reads frame_index.parquet, loads each frame, runs the chosen encoder, and
caches the latent tensor [T_lat, P, D_v] to disk.  Training runs load cached
latents directly (skipping the encoder) when the dataset is configured with
``vidtok_cache_dir`` (the param name is generic — accepts any cache dir).

Cache layout (per encoder, to avoid collisions):
    data/refactored/{domain}/{dataset}/vidtok_cache/{key}.pt   (VidTok)
    data/refactored/{domain}/{dataset}/vjepa_cache/{key}.pt    (V-JEPA)

Key convention (matches MMTSFMDataset._compute_cache_key):
    - EarthNet2021:            {entity_id}_{window_offset:05d}.pt
    - All other datasets:      {window_start_timestamp}.pt

Usage
-----
    # V-JEPA 2.1 (matches training config defaults from configs/data/default.yaml):
    uv run python scripts/extract_video_embeddings.py \\
        --encoder vjepa2 --vjepa-arch vit_large \\
        --dataset skippd --split train \\
        --video-frames 17 --img-size 224 --hist-steps 24 --horizon 12 \\
        --imagenet-norm \\
        --data-dir /leonardo_scratch/fast/IscrC_MTSFM/data \\
        --batch-size 8

    # VidTok (original path):
    uv run python scripts/extract_video_embeddings.py \\
        --encoder vidtok --dataset skippd \\
        --vidtok-cfg /path/to/vidtok_kl_causal_488_4chn.yaml \\
        --vidtok-ckpt /path/to/model.ckpt \\
        --data-dir ./data --batch-size 8

    # Dry-run (validates frames load correctly, no encoder required):
    uv run python scripts/extract_video_embeddings.py \\
        --encoder vjepa2 --dataset earthnet2021 --dry-run

Supported datasets
------------------
    skippd, solarnet, goes16_nsrdb, earthnet2021, era5_eu, meteonet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Ensure src/ is importable
_root = Path(__file__).resolve().parents[1]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from mmtsfm.data.dataset import MMTSFMDataset  # noqa: E402

# Per-dataset defaults (hist_steps, horizon, img_channels, video_frames, img_size)
_DATASET_DEFAULTS = {
    "skippd":        dict(hist_steps=168, horizon=24, img_channels=3, video_frames=8,  img_size=64,  num_entities=1,  covariate_dim=1),
    "solarnet":      dict(hist_steps=720, horizon=60, img_channels=3, video_frames=8,  img_size=64,  num_entities=1,  covariate_dim=7),
    "goes16_nsrdb":  dict(hist_steps=48,  horizon=16, img_channels=3, video_frames=6,  img_size=64,  num_entities=10, covariate_dim=3),
    "earthnet2021":  dict(hist_steps=20,  horizon=8,  img_channels=4, video_frames=4,  img_size=128, num_entities=1,  covariate_dim=5),
    "era5_eu":       dict(hist_steps=24,  horizon=8,  img_channels=6, video_frames=4,  img_size=128, num_entities=10, covariate_dim=5),
    "meteonet":      dict(hist_steps=48,  horizon=12, img_channels=2, video_frames=6,  img_size=128, num_entities=5,  covariate_dim=5),
}

# Domain prefix for data/refactored/
_DOMAIN = {
    "skippd": "solar", "solarnet": "solar", "goes16_nsrdb": "solar",
    "earthnet2021": "meteorology", "era5_eu": "meteorology", "meteonet": "meteorology",
}


def _build_vidtok_encoder(cfg_path: str, ckpt_path: str, vidtok_root: str | None,
                          is_causal: bool, device: torch.device):
    from mmtsfm.models.vision.vidtok_encoder import VidTokEncoder
    enc = VidTokEncoder(
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        vidtok_root=vidtok_root,
        is_causal=is_causal,
    ).to(device)
    enc.probe_latent_shape(device)
    return enc


def _build_vjepa_encoder(arch: str, device: torch.device):
    from mmtsfm.models.vision.visual_encoder import VisualEncoder
    enc = VisualEncoder(arch=arch, freeze=True).to(device)
    return enc


def _cache_dir_for(data_dir: Path, dataset_name: str, encoder: str) -> Path:
    domain = _DOMAIN[dataset_name]
    subdir = "vjepa_cache" if encoder == "vjepa2" else "vidtok_cache"
    return data_dir / "refactored" / domain / dataset_name / subdir


def extract(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    # Per-dataset defaults, with CLI overrides applied on top so the cached
    # shapes match the training config (see configs/data/default.yaml).
    cfg = dict(_DATASET_DEFAULTS[args.dataset])
    for key, val in [
        ("hist_steps",   args.hist_steps),
        ("horizon",      args.horizon),
        ("video_frames", args.video_frames),
        ("img_size",     args.img_size),
    ]:
        if val is not None:
            cfg[key] = val

    print(f"[extract] encoder={args.encoder}  dataset={args.dataset}  "
          f"split={args.split}  device={device}")
    print(f"[extract] shape cfg: hist_steps={cfg['hist_steps']} horizon={cfg['horizon']} "
          f"video_frames={cfg['video_frames']} img_size={cfg['img_size']}")

    # Build dataset (no cache; loads raw frames)
    ds = MMTSFMDataset(
        data_dir=str(data_dir),
        dataset_name=args.dataset,
        split=args.split,
        imagenet_norm=args.imagenet_norm,
        **cfg,
    )
    print(f"[extract] {len(ds)} samples")
    if len(ds) == 0:
        print("[extract] WARNING: dataset is empty — check hist_steps + horizon vs sequence length.")
        return

    if not args.dry_run:
        if args.encoder == "vjepa2":
            enc = _build_vjepa_encoder(arch=args.vjepa_arch, device=device)
        else:
            enc = _build_vidtok_encoder(
                cfg_path=args.vidtok_cfg,
                ckpt_path=args.vidtok_ckpt,
                vidtok_root=args.vidtok_root,
                is_causal=not args.non_causal,
                device=device,
            )
        enc.eval()

    cache_dir = _cache_dir_for(data_dir, args.dataset, args.encoder)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] cache dir: {cache_dir}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(not args.dry_run))

    n_done = 0
    n_skip = 0
    t0 = time.time()

    pbar = tqdm(loader, total=len(loader), unit="batch",
                desc=f"{args.dataset}/{args.split}", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        BS = batch["Y"].shape[0]
        N  = batch["Y"].shape[1]

        # Compute cache keys for each sample in the batch
        # idx range: batch_idx*BS ... (batch_idx+1)*BS
        base_idx = batch_idx * args.batch_size
        keys = [ds._compute_cache_key(base_idx + i) for i in range(BS)]

        # Skip already-cached samples
        missing_local = [i for i, k in enumerate(keys)
                         if not (cache_dir / f"{k}.pt").exists()]
        if not missing_local:
            n_skip += BS
            pbar.set_postfix(done=n_done, skip=n_skip, refresh=False)
            continue

        if args.dry_run:
            # Just verify frames loaded; no VidTok
            n_done += len(missing_local)
            pbar.set_postfix(done=n_done, skip=n_skip, refresh=False)
            continue

        # V: [BS, N, T_v, C, H, W] → [BS*N, C, T_v, H, W]
        V = batch["V"]
        T_v, C, H_img, W_img = V.shape[2], V.shape[3], V.shape[4], V.shape[5]
        video = V.reshape(BS * N, T_v, C, H_img, W_img).permute(0, 2, 1, 3, 4)
        video = video.to(device, non_blocking=True)

        with torch.no_grad():
            # [BS*N, T_lat, P, D_v]
            z = enc(video)

        # Reshape back to [BS, N, T_lat, P, D_v]
        z = z.reshape(BS, N, *z.shape[1:]).cpu()

        # Save only missing samples
        for local_i in missing_local:
            key = keys[local_i]
            z_i = z[local_i]           # [N, T_lat, P, D_v]
            # C5 fix: always save [N, T_lat, P, D_v] regardless of N.
            # Old code squeezed N=1 → [T_lat, P, D_v], breaking the consumer
            # which unconditionally expects the N dimension after collation.
            torch.save(z_i, cache_dir / f"{key}.pt")
            n_done += 1

        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        pbar.set_postfix(done=n_done, skip=n_skip, sps=f"{rate:.1f}", refresh=False)

    pbar.close()

    elapsed = time.time() - t0
    print(f"[extract] DONE  done={n_done}  skip={n_skip}  total_time={elapsed:.1f}s")
    if not args.dry_run:
        print(f"[extract] cache: {cache_dir}")
        print(f"[extract] To use: set vidtok_cache_dir={cache_dir} in your data config.")


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-extract visual latents (VidTok or V-JEPA).")
    p.add_argument("--encoder", default="vidtok", choices=["vidtok", "vjepa2"],
                   help="Which visual encoder to run")
    p.add_argument("--dataset", required=True,
                   choices=list(_DATASET_DEFAULTS), help="Dataset name")
    p.add_argument("--split", default="train",
                   choices=["train", "val", "test"],
                   help="Which split to extract (run once per split)")
    # Shape overrides — must match training config (configs/data/default.yaml).
    p.add_argument("--hist-steps",   type=int, default=None)
    p.add_argument("--horizon",      type=int, default=None)
    p.add_argument("--video-frames", type=int, default=None)
    p.add_argument("--img-size",     type=int, default=None)
    p.add_argument("--imagenet-norm", action="store_true",
                   help="Apply ImageNet mean/std (default for V-JEPA training)")
    # V-JEPA-specific
    p.add_argument("--vjepa-arch", default="vit_large",
                   choices=["vit_large", "vit_base"])
    # VidTok-specific
    p.add_argument("--vidtok-cfg",  default="",
                   help="Path to VidTok YAML config (vidtok only)")
    p.add_argument("--vidtok-ckpt", default="",
                   help="Path to VidTok .ckpt checkpoint (vidtok only)")
    p.add_argument("--vidtok-root", default=None,
                   help="VidTok repo root added to sys.path (vidtok only)")
    p.add_argument("--non-causal", action="store_true",
                   help="Use non-causal VidTok variant (default: causal 4×8×8)")
    # I/O
    p.add_argument("--data-dir", default="./data", help="Project data root")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="Load frames only; skip encoder (validates data pipeline)")
    args = p.parse_args()

    if not args.dry_run and args.encoder == "vidtok" and (not args.vidtok_cfg or not args.vidtok_ckpt):
        p.error("--vidtok-cfg and --vidtok-ckpt are required for --encoder vidtok unless --dry-run is set.")

    extract(args)


if __name__ == "__main__":
    main()
