"""Single trainer for all tier-2 models (one port covers the tier, §8.2).

Masked L1 loss for point models, masked pinball loss for quantile models
(TFT). Early stopping on validation loss, fixed seed, best-state restore.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from common import config
from common.windows import WindowDataset


@dataclass
class TrainerConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = config.SEED
    device: str | None = None  # auto: cuda > mps > cpu
    num_workers: int = 0
    max_batches_per_epoch: int | None = None  # cap for very large train sets


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchWindows(Dataset):
    """Tensor view over a WindowDataset for the DataLoader."""

    KEYS = ("y_hist", "mask_hist", "y_future", "mask_future",
            "daylight_future", "cov")

    def __init__(self, windows: WindowDataset):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        item = self.windows[i]
        return {k: torch.as_tensor(item[k], dtype=torch.float32) for k in self.KEYS}


def quantile_loss(pred: Tensor, target: Tensor, mask: Tensor,
                  levels: tuple[float, ...]) -> Tensor:
    q = torch.tensor(levels, device=pred.device, dtype=pred.dtype)
    err = target.unsqueeze(-1) - pred
    loss = torch.where(err >= 0, q * err, (q - 1) * err)
    return (loss * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1.0) / len(levels)


def point_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def _epoch_loss(model: nn.Module, loader: DataLoader, device: torch.device,
                optimizer: torch.optim.Optimizer | None,
                max_batches: int | None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch["y_hist"], batch["cov"], batch["mask_hist"])
            mask = batch["mask_future"] * batch["daylight_future"]
            if model.output_quantiles:
                loss = quantile_loss(pred, batch["y_future"], mask,
                                     config.QUANTILE_LEVELS)
            else:
                loss = point_loss(pred, batch["y_future"], mask)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach())
            count += 1
    return total / max(count, 1)


def train_model(
    model: nn.Module,
    train: WindowDataset,
    val: WindowDataset,
    cfg: TrainerConfig,
) -> nn.Module:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = model.to(device)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        TorchWindows(train), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, generator=generator,
    )
    val_loader = DataLoader(
        TorchWindows(val), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(cfg.epochs):
        _epoch_loss(model, train_loader, device, optimizer,
                    cfg.max_batches_per_epoch)
        val_loss = _epoch_loss(model, val_loader, device, None, None)
        if val_loss < best_loss - 1e-6:
            best_loss, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
