"""Bridge `UKMultimodalDataset` windows to Aurora's fine-tuning batch dict.

Upstream's `utils/pretrain_dataset.py` emits `input_ids / labels / text_*` from a
bare CSV: no `vision_ids`, no plant splits, no daylight mask. That is the wrong
modality (text, which this thesis deliberately excludes) and the wrong protocol.
This module reuses the same dataset the zero-shot runner evaluates on, so the
window spec (`history` / `horizon` / `stride`), the plant split and the NaN
handling are shared between the fine-tune and the test run.

`loss_masks` carries `mask_future`, so a horizon step with no observed power
contributes no gradient — `common.windows` fills those with 0, and training on
the placeholder would teach the model the fill value. Night steps are masked out
along with genuinely missing ones; scoring excludes them too (the test mask is
``mask_future * daylight_future``). The reconstruction term stays unmasked: a
zero-filled night history step is physically correct for PV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

BASELINES = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASELINES))

from tier6.uk_multimodal import UKMultimodalDataset, sites_for_split  # noqa: E402


def build_split_dataset(cfg, split: str, stride: int | None = None):
    """One dataset over every plant of a split, windowed like the test run."""
    return UKMultimodalDataset(
        site_ids=sites_for_split(split, dataset=cfg.dataset),
        data_path=cfg.data_path,
        h5_path=cfg.h5_path,
        history=cfg.history,
        horizon=cfg.horizon,
        stride=cfg.stride if stride is None else stride,
        img_size=cfg.image_size,
        datasets=[cfg.dataset],
        to_gray=True,
        visual_history_steps=cfg.visual_history_steps,
    )


def iter_shuffled_batches(ds, batch_size: int, rng: np.random.Generator, limit=None):
    """Yield shuffled window batches, at most `limit` of them per pass."""
    order = rng.permutation(len(ds))
    for n, lo in enumerate(range(0, len(order), batch_size)):
        if limit is not None and n >= limit:
            return
        yield ds.batch(order[lo : lo + batch_size].tolist())


def iter_batches(ds, batch_size: int, limit=None):
    """Yield window batches in order, at most `limit` of them (validation)."""
    for n, batch in enumerate(ds.iter_batches(batch_size)):
        if limit is not None and n >= limit:
            return
        yield batch


def aurora_subbatches(batch: dict, device, *, vision_mode: str, latest_real_frames):
    """Split one window batch into Aurora forward kwargs, grouped by frame availability.

    `AuroraForPrediction.forward` takes a single `vision_ids` for the whole call,
    so windows that carry a real frame and windows that do not cannot share a
    pass — the same split the zero-shot runner makes at generation time. Windows
    whose horizon holds no observed power at all are dropped: their forecast term
    would be empty.

    Yields ``(kwargs, count)`` so the caller can weight each group's loss by how
    many windows it covers.
    """
    scored = batch["mask_future"].sum(axis=1) > 0
    if not scored.any():
        return
    histories = torch.from_numpy(batch["y_hist"][scored]).float()
    labels = torch.from_numpy(batch["y_future"][scored]).float()
    masks = torch.from_numpy(batch["mask_future"][scored]).float()
    frames, available = latest_real_frames(
        batch["V"][scored], batch["mask_visual"][scored]
    )
    if vision_mode == "pseudo":
        available = torch.zeros_like(available)
    for use_real in (True, False):
        keep = available == use_real
        if not keep.any():
            continue
        yield (
            {
                "input_ids": histories[keep].to(device),
                "labels": labels[keep].to(device),
                "loss_masks": masks[keep].to(device),
                "vision_ids": frames[keep].to(device) if use_real else None,
            },
            int(keep.sum()),
        )


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    """Aurora freezes its ViT and BERT encoders; train everything else."""
    return [p for p in model.parameters() if p.requires_grad]


def set_finetune_mode(model):
    """Put the model in train mode but keep BatchNorm on its running statistics.

    Aurora's decoder norms are `BatchNorm1d` (`norm_mode: 'batch'`), and upstream
    freezes exactly these when adapting the model to a downstream dataset --- see
    `EPF/exp/exp_long_term_forecasting.py::_build_model`. Two reasons it matters
    here:

    * `aurora_subbatches` splits a window batch by frame availability, so a
      forward pass routinely carries a single window. Batch statistics are
      undefined for one sample and `F.batch_norm` raises outright.
    * `run_ukpv.py` scores in `eval()`, i.e. on running statistics. Training on
      batch statistics computed over a handful of windows --- and separately for
      the real- and pseudo-frame groups of the same step --- would optimise
      against a normalisation the evaluation never applies.

    The affine weight and bias stay trainable, so a shifted input distribution is
    still absorbed; only the running mean/var are held fixed.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    return model
