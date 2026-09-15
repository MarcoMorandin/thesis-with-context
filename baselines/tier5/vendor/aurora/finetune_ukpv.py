"""Fine-tune Aurora on UK-PV train plants, protocol-aligned with `run_ukpv.py`.

Aurora ships a fine-tune path (`train_from_scratch.py --mode fine_tune`), but it
is wired to `generate_pretrain_dataset`, which reads a CSV plus a text JSON and
emits no `vision_ids` — the text modality this thesis excludes, with no plant
split and no target mask. This script keeps `AuroraForPrediction` and swaps in
the shared UK-PV windows instead, so the fine-tuned checkpoint can be scored by
`run_ukpv.py` without touching the evaluation path at all:

    uv run python finetune_ukpv.py ckpt_path=<pretrained> out=<ft_dir>
    uv run python run_ukpv.py ckpt_path=<ft_dir>/checkpoint vision_mode=real

Protocol: training sees **only** train plants, early stopping reads **only** val
plants, and the test plants are never loaded here. The vision mode is fixed by
config before training rather than selected on results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

BASELINES = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aurora.modeling_aurora import AuroraForPrediction  # noqa: E402

from run_ukpv import latest_real_frames  # noqa: E402
from ukpv_ft_dataset import (  # noqa: E402
    aurora_subbatches,
    build_split_dataset,
    iter_batches,
    iter_shuffled_batches,
    pinned_rng,
    set_finetune_mode,
    trainable_parameters,
)


def _batch_loss(model, batch, device, *, vision_mode):
    """Mean loss over one window batch, weighting each vision group by its size."""
    total, seen = None, 0
    for kwargs, count in aurora_subbatches(
        batch, device, vision_mode=vision_mode, latest_real_frames=latest_real_frames
    ):
        loss = model(**kwargs).loss
        if not torch.isfinite(loss):
            continue
        total = loss * count if total is None else total + loss * count
        seen += count
    return None if seen == 0 else total / seen


@torch.no_grad()
def validate(model, ds, cfg, device) -> float:
    """Mean objective over the val plants --- the early-stopping signal.

    Runs in **train** mode on purpose. `AuroraEmbedding.forward` short-circuits
    to `_predict` under `eval()`, which returns no masked copies, so `x_rec` is
    None and the reconstruction term of the loss does not exist --- the training
    objective is only defined in train mode. `set_finetune_mode` already holds
    BatchNorm on its running statistics, and `pinned_rng` makes the masking,
    dropout and flow-matching draws identical every epoch, so epoch-to-epoch
    movement in this number comes from the weights alone.

    This is the selection signal only. The reported metrics come from
    `run_ukpv.py`, which generates in `eval()` and is untouched by any of this.
    """
    losses = []
    with pinned_rng(cfg.seed):
        for batch in iter_batches(ds, cfg.batch_size, cfg.max_val_batches):
            loss = _batch_loss(model, batch, device, vision_mode=cfg.vision_mode)
            if loss is not None:
                losses.append(float(loss))
    if not losses:
        raise ValueError("no scorable validation windows")
    return float(np.mean(losses))


def train_epoch(model, ds, cfg, device, optimizer, rng) -> float:
    """One pass over a shuffled slice of the train windows."""
    losses = []
    for batch in iter_shuffled_batches(ds, cfg.batch_size, rng, cfg.max_train_batches):
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, device, vision_mode=cfg.vision_mode)
        if loss is None:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), cfg.grad_clip)
        optimizer.step()
        losses.append(float(loss))
    if not losses:
        raise ValueError("no scorable training windows")
    return float(np.mean(losses))


@hydra.main(
    version_base=None, config_path="../../../configs/tier5", config_name="aurora_ft"
)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AuroraForPrediction.from_pretrained(cfg.ckpt_path).to(device)
    set_finetune_mode(model)
    params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    train_ds = build_split_dataset(cfg, "train", stride=cfg.train_stride)
    val_ds = build_split_dataset(cfg, "val", stride=cfg.train_stride)

    out = Path(cfg.out)
    ckpt = out / "checkpoint"
    out.mkdir(parents=True, exist_ok=True)

    history, best, bad = [], float("inf"), 0
    started = time.time()
    for epoch in range(cfg.max_epochs):
        train_loss = train_epoch(model, train_ds, cfg, device, optimizer, rng)
        val_loss = validate(model, val_ds, cfg, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best - cfg.min_delta:
            best, bad = val_loss, 0
            model.save_pretrained(ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if not ckpt.exists():  # never improved; still score something reproducible
        model.save_pretrained(ckpt)

    gpu_hours = (time.time() - started) / 3600.0
    summary = {
        "checkpoint": str(ckpt),
        "vision_mode": cfg.vision_mode,
        "trainable_parameters": int(sum(p.numel() for p in params)),
        "epochs_run": len(history),
        "best_val_loss": best,
        "gpu_hours": gpu_hours,
        "history": history,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
