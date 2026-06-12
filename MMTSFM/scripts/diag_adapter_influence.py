"""Stage 2a diagnostic: measure cross-modal adapter influence on loss.

Runs the same forward pass twice on a fixed batch with the SAME random
``cross_modal_adapter`` weights, scaled once by 1.0 and once by 0.0 (the
"no-adapter" baseline). The two losses should differ if the adapter has
a non-trivial gradient path to the loss. If they match to <1e-3 relative
delta, the visual signal is dead in the current configuration.

Usage:
    uv run python scripts/diag_adapter_influence.py \
        model=vision_chronos2_timeselfattn \
        data.dataset_name=skippd \
        data.data_dir=$DATA_DIR \
        data.num_workers=0 \
        data.num_samples_train=8
"""
from __future__ import annotations

import sys
import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

load_dotenv()


def _scale_adapter(model, factor: float) -> None:
    adapter = getattr(model.model, "cross_modal_adapter", None)
    if adapter is None:
        raise RuntimeError("cross_modal_adapter is None; only late-fusion "
                           "configurations are supported by this probe.")
    for p in adapter.parameters():
        p.data.mul_(factor)


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print("=== diag_adapter_influence: config ===")
    print(OmegaConf.to_yaml(cfg))

    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    torch.manual_seed(0)
    model = instantiate(cfg.model)
    model.eval()  # disable dropout so the probe is deterministic

    # --- Path 1: full adapter (scale=1.0, identity) -----------------------
    with torch.no_grad():
        loss_full = model._step(batch, "val").item()

    # Snapshot adapter weights so we can restore for path 2.
    snapshot = {n: p.detach().clone() for n, p in
                model.model.cross_modal_adapter.named_parameters()}

    # --- Path 2: zero adapter -------------------------------------------------
    _scale_adapter(model, factor=0.0)
    with torch.no_grad():
        loss_zero = model._step(batch, "val").item()

    # Restore.
    for n, p in model.model.cross_modal_adapter.named_parameters():
        p.data.copy_(snapshot[n])

    delta = loss_full - loss_zero
    rel = abs(delta) / max(abs(loss_full), 1e-9)
    print("\n=== adapter influence on val loss (no grad) ===")
    print(f"  loss(adapter ON)   = {loss_full:.6f}")
    print(f"  loss(adapter OFF)  = {loss_zero:.6f}")
    print(f"  Δ                  = {delta:+.6f}  (rel = {rel:.2e})")
    if rel < 1e-3:
        print("  VERDICT: adapter has no measurable influence on loss.")
        print("           Stage 2a will not learn visual alignment as configured.")
        sys.exit(2)
    else:
        print("  VERDICT: adapter influences loss; gradient path is alive.")


if __name__ == "__main__":
    main()
