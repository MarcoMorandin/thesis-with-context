"""Prune + recompress an existing V-JEPA latent cache IN PLACE (no GPU).

Why: the original extractor saved ``z[i, j]`` — a *view* — and ``torch.save``
serializes the whole underlying batch storage, so every cache file carried 8×
the data (25.7 MB instead of 1.6 MB fp16). At uk_pv train stride-1 that blew
the 1 TB scratch quota (32.84 TB observed). The latents themselves are valid.

This walks the cache dir and, against the key set the training runs will
actually request (train at ``--train-stride``, val/test at protocol stride H):
  - WANTED files are rewritten as contiguous fp16 (atomic tmp+rename) when
    oversized or not fp16 already;
  - UNWANTED files are deleted (only with ``--apply``).

Default is a dry run that just reports counts/sizes. CPU/IO-bound → login node.

Usage:
    uv run python scripts/prune_recompress_vjepa_cache.py \
        --dataset uk_pv --data-dir /leonardo_scratch/fast/IscrC_MTSFM/data \
        --cache-dir <data-dir>/vjepa_cache/uk_pv/vit_large_f8_s224 \
        --train-stride 12 --workers 16 [--apply]
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

_root = Path(__file__).resolve().parents[1]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from extract_video_embeddings import _DATASET_DEFAULTS  # noqa: E402
from mmtsfm.data.pv_record import PVRecordDataset  # noqa: E402


def wanted_keys(args) -> set[str]:
    """Every cache key the training runs will look up (all splits)."""
    keys: set[str] = set()
    cfg = dict(_DATASET_DEFAULTS[args.dataset])
    for split in args.splits.split(","):
        stride = args.train_stride if split == "train" else None  # val/test: H
        ds = PVRecordDataset(
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            split=split,
            stride=stride,
            num_entities=1,
            **cfg,
        )
        keys.update(ds._entity_cache_key(ds.win[w]) for w in range(len(ds.win)))
        print(f"[prune] split={split} stride={stride or 'H'} → {len(ds.win)} keys")
    return keys


def recompress(path: Path, apply: bool) -> int:
    """Rewrite as contiguous fp16 if beneficial; return bytes saved."""
    size = path.stat().st_size
    z = torch.load(path, map_location="cpu", weights_only=True)
    target = z.numel() * 2  # fp16 payload
    if z.dtype == torch.float16 and size <= int(target * 1.2):
        return 0  # already compact
    if apply:
        tmp = path.with_suffix(".pt.tmp")
        torch.save(z.to(torch.float16).clone().contiguous(), tmp)
        os.replace(tmp, path)
        return size - path.stat().st_size
    return size - int(target * 1.05)  # dry-run estimate (+save overhead)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=list(_DATASET_DEFAULTS))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--train-stride", type=int, default=None)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--apply", action="store_true", help="actually delete/rewrite")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    keep = wanted_keys(args)
    files = [f for f in cache_dir.iterdir() if f.suffix == ".pt"]
    # stale tmp files from an interrupted earlier pass
    for tmp in cache_dir.glob("*.pt.tmp"):
        if args.apply:
            tmp.unlink()
    kept = [f for f in files if f.stem in keep]
    drop = [f for f in files if f.stem not in keep]
    missing = len(keep) - len(kept)
    print(
        f"[prune] cache files={len(files)}  keep={len(kept)}  delete={len(drop)}  "
        f"missing-from-cache={missing} (extractor will fill these)"
    )

    freed = 0
    if drop:
        for f in tqdm(drop, desc="delete", unit="file", disable=not args.apply):
            freed += f.stat().st_size
            if args.apply:
                f.unlink()
    saved = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for s in tqdm(
            ex.map(lambda f: recompress(f, args.apply), kept),
            total=len(kept),
            desc="recompress",
            unit="file",
        ):
            saved += s

    mode = "APPLIED" if args.apply else "DRY RUN (re-run with --apply)"
    per_file = 0
    if kept:
        z = torch.load(kept[0], map_location="cpu", weights_only=True)
        per_file = z.numel() * 2  # contiguous fp16 payload
    print(
        f"[prune] {mode}: delete frees {freed / 1e12:.2f} TB, "
        f"recompress saves {saved / 1e12:.2f} TB, "
        f"final cache ≈ {len(kept) * per_file / 1e9:.0f} GB in {len(kept)} files"
    )


if __name__ == "__main__":
    main()
