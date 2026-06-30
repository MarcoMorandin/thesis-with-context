"""Pre-extract V-JEPA 2.1 visual latents for protocol PV datasets.

The supported datasets are the dataset-of-record tracks used by the baseline
contract: ``uk_pv`` and ``goes_pvdaq``. Frames are read from
``dataset_all.parquet`` + ``images_all.h5`` through ``PVRecordDataset`` and
cached per plant/window so training can pass ``data.vjepa_cache_dir``.

Usage:
    uv run python scripts/extract_video_embeddings.py \
        --dataset uk_pv --split train \
        --data-dir /leonardo_scratch/fast/IscrC_MTSFM/data \
        --batch-size 8 --imagenet-norm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_root = Path(__file__).resolve().parents[1]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from mmtsfm.data.pv_record import PVRecordDataset  # noqa: E402

_DATASET_DEFAULTS = {
    "uk_pv": dict(horizon=12, img_channels=3, video_frames=8, img_size=224),
    "goes_pvdaq": dict(horizon=24, img_channels=3, video_frames=8, img_size=224),
}


def _build_vjepa_encoder(arch: str, device: torch.device):
    from mmtsfm.models.vision.visual_encoder import VisualEncoder

    return VisualEncoder(arch=arch, freeze=True).to(device)


def _cache_dir_for(data_dir: Path, dataset_name: str) -> Path:
    return data_dir / "vjepa_cache" / dataset_name


def _cache_keys_for(ds: PVRecordDataset, sample_idx: int) -> list[str]:
    return [ds._entity_cache_key(ds.win[w]) for w in ds.groups[sample_idx]]


def extract(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    cfg = dict(_DATASET_DEFAULTS[args.dataset])
    for key, val in [
        ("hist_steps", args.hist_steps),
        ("horizon", args.horizon),
        ("video_frames", args.video_frames),
        ("img_size", args.img_size),
    ]:
        if val is not None:
            cfg[key] = val

    print(
        f"[extract] encoder=vjepa2 dataset={args.dataset} "
        f"split={args.split} device={device}"
    )
    print(
        f"[extract] shape cfg: hist_steps={cfg.get('hist_steps')} "
        f"horizon={cfg['horizon']} video_frames={cfg['video_frames']} "
        f"img_size={cfg['img_size']}"
    )

    ds = PVRecordDataset(
        data_dir=str(data_dir),
        dataset_name=args.dataset,
        split=args.split,
        imagenet_norm=args.imagenet_norm,
        num_entities=args.num_entities,
        **cfg,
    )
    print(f"[extract] {len(ds)} samples")
    if len(ds) == 0:
        print("[extract] WARNING: dataset is empty")
        return

    if not args.dry_run:
        enc = _build_vjepa_encoder(arch=args.vjepa_arch, device=device)
        enc.eval()

    cache_dir = _cache_dir_for(data_dir, args.dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] cache dir: {cache_dir}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.dry_run,
    )

    n_done = 0
    n_skip = 0
    t0 = time.time()
    pbar = tqdm(loader, total=len(loader), unit="batch", desc=f"{args.dataset}/{args.split}", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        batch_size = batch["Y"].shape[0]
        n_entities = batch["Y"].shape[1]
        base_idx = batch_idx * args.batch_size
        keys = [_cache_keys_for(ds, base_idx + i) for i in range(batch_size)]

        missing = [
            (i, j, key)
            for i, sample_keys in enumerate(keys)
            for j, key in enumerate(sample_keys)
            if not (cache_dir / f"{key}.pt").exists()
        ]
        if not missing:
            n_skip += batch_size * n_entities
            pbar.set_postfix(done=n_done, skip=n_skip, refresh=False)
            continue

        if args.dry_run:
            n_done += len(missing)
            pbar.set_postfix(done=n_done, skip=n_skip, refresh=False)
            continue

        video = batch["V"]
        t_v, c_img, h_img, w_img = video.shape[2], video.shape[3], video.shape[4], video.shape[5]
        video = video.reshape(batch_size * n_entities, t_v, c_img, h_img, w_img)
        video = video.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)

        with torch.no_grad():
            z = enc(video)

        z = z.reshape(batch_size, n_entities, *z.shape[1:]).cpu()
        for sample_i, entity_i, key in missing:
            torch.save(z[sample_i, entity_i], cache_dir / f"{key}.pt")
            n_done += 1

        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0.0
        pbar.set_postfix(done=n_done, skip=n_skip, sps=f"{rate:.1f}", refresh=False)

    pbar.close()
    elapsed = time.time() - t0
    print(f"[extract] DONE done={n_done} skip={n_skip} total_time={elapsed:.1f}s")
    if not args.dry_run:
        print(f"[extract] cache: {cache_dir}")
        print(f"[extract] To use: set data.vjepa_cache_dir={cache_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-extract V-JEPA visual latents.")
    p.add_argument("--encoder", default="vjepa2", choices=["vjepa2"])
    p.add_argument("--dataset", required=True, choices=list(_DATASET_DEFAULTS))
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--hist-steps", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--video-frames", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--num-entities", type=int, default=1)
    p.add_argument("--imagenet-norm", action="store_true")
    p.add_argument("--vjepa-arch", default="vit_large", choices=["vit_large", "vit_base"])
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
