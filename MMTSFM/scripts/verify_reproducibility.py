"""Verify that set_seed produces identical val losses across two runs with the same seed.

Usage:
    uv run python scripts/verify_reproducibility.py
"""

import sys
sys.path.insert(0, "src")

import torch
from torch.utils.data import DataLoader

from mmtsfm.data.dataset import MMTSFMDataset
from mmtsfm.models.base import MMTSFMBaseModel
from utils.reproducibility import set_seed


def run_one_epoch(seed: int, input_dim: int = 24) -> float:
    """Seed everything, init model + data, run one val epoch, return mean loss."""
    set_seed(seed)

    # Model init — weights are drawn from seeded RNG
    model = MMTSFMBaseModel(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=input_dim,
        learning_rate=1e-3,
        weight_decay=0.0,
    )
    model.eval()

    # Data — synthetic tensors drawn from seeded RNG
    # target_dim=1 is the only supported mode; use Y shape (entities, T, 1)
    dataset = MMTSFMDataset(
        num_samples=16,
        num_entities=1,
        hist_steps=input_dim,
        horizon=8,
        target_dim=1,
        covariate_dim=2,
        video_frames=2,
        img_size=8,
        dataset_name="synthetic",
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            # Y: (batch, entities=1, T, C=1) → flatten to (batch, input_dim)
            x = batch["Y"].view(batch["Y"].shape[0], -1)
            y_hat = model(x)
            total_loss += criterion(y_hat, x).item()
            n_batches += 1

    return total_loss / n_batches


def main():
    SEED = 42
    ALT_SEED = 99

    print("Running epoch with seed=42 (run 1)...", flush=True)
    loss_a = run_one_epoch(SEED)

    print("Running epoch with seed=42 (run 2)...", flush=True)
    loss_b = run_one_epoch(SEED)

    print("Running epoch with seed=99 (control)...", flush=True)
    loss_c = run_one_epoch(ALT_SEED)

    print(f"\n  seed=42 run 1 val loss : {loss_a:.8f}")
    print(f"  seed=42 run 2 val loss : {loss_b:.8f}")
    print(f"  seed=99 val loss       : {loss_c:.8f}")

    assert loss_a == loss_b, (
        f"FAIL: same seed produced different losses: {loss_a} != {loss_b}"
    )
    assert loss_a != loss_c, (
        f"WARN: different seeds produced identical losses (unlikely but possible): {loss_a}"
    )

    print("\nPASS: identical seeds → identical val loss.")


if __name__ == "__main__":
    main()
