"""One forward-pass diagnostic for loss explosion debugging.

Accepts the same Hydra overrides as slurm_train.sh.  Instantiates the real
model and datamodule, grabs 1 batch, and runs training_step.  The DIAG print
in _compute_loss fires automatically.

Usage (local):
    uv run python scripts/diag_forward_pass.py model=vision_chronos2

Usage (cluster — same overrides as slurm_train.sh):
    srun ... uv run python scripts/diag_forward_pass.py \\
        model=vision_chronos2 \\
        data.dataset_name=skippd \\
        data.data_dir=$DATA_DIR \\
        data.num_workers=0 \\
        model.vision_cfg.vidtok_cfg_path=$VIDTOK_CFG \\
        model.vision_cfg.vidtok_ckpt_path=$VIDTOK_CKPT \\
        model.vision_cfg.vidtok_root=$VIDTOK_ROOT
"""
from __future__ import annotations

import sys
import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import torch

load_dotenv()


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print("=== diag_forward_pass: config ===")
    print(OmegaConf.to_yaml(cfg))

    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("\n=== batch shapes ===")
    for k, v in batch.items():
        shape = v.shape if hasattr(v, "shape") else type(v)
        print(f"  {k}: {shape}")

    model = instantiate(cfg.model)
    model.train()

    print("\n=== running training_step (watch for [DIAG _compute_loss] lines) ===")
    loss = model.training_step(batch, 0)

    print(f"\n=== result ===")
    print(f"loss={loss.item():.6f}  isnan={torch.isnan(loss).item()}  isinf={torch.isinf(loss).item()}")


if __name__ == "__main__":
    main()
