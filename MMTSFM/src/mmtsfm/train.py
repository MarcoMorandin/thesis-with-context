import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import logging

# Ensure project root is accessible
import sys
import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import torch

torch.set_float32_matmul_precision("high")  # Tensor Core acceleration on A100 / Ampere

load_dotenv()

log = logging.getLogger(__name__)


def _allowlist_lightning_checkpoint_globals() -> None:
    """Patch torch.load to use weights_only=False for trusted internal checkpoints.

    PyTorch 2.6 changed the default to weights_only=True. Lightning checkpoints
    contain omegaconf/dict/list globals that are not allowlisted by default.
    Since these checkpoints are generated internally, weights_only=False is safe.
    """
    _orig = torch.load

    def _load(*args, **kwargs):
        # Default only — callers that explicitly pass weights_only (e.g. the
        # latent-cache loader uses weights_only=True) keep their choice.
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load


def _best_finite_checkpoint_path(trainer) -> str | None:
    """Return best checkpoint only when monitored score exists and is finite."""
    callbacks = getattr(trainer, "checkpoint_callbacks", None) or []
    for callback in callbacks:
        best_path = getattr(callback, "best_model_path", "")
        best_score = getattr(callback, "best_model_score", None)
        monitor = getattr(callback, "monitor", None)
        if not monitor or not best_path or best_score is None:
            continue
        score = torch.as_tensor(best_score)
        if torch.isfinite(score).all() and os.path.exists(best_path):
            return best_path
    return None


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    from utils.reproducibility import set_seed

    set_seed(cfg.seed)
    log.info(OmegaConf.to_yaml(cfg))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    # Thread the run seed into the model so the protocol results manifest
    # records the actual seed (matches the baselines' run_config provenance).
    model = instantiate(cfg.model, seed=cfg.seed)

    log.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger = instantiate(cfg.logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate(cfg.trainer, logger=logger)

    # Optional: Log hyperparameters to WandB
    if logger:
        # Save hydra config safely to WandB
        # Convert dictionary and avoid omegaconf specific types during logging
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    _allowlist_lightning_checkpoint_globals()

    # Weights-only warm start for chaining curriculum stages. Distinct from
    # ckpt_path (full-state resume): load only the prior stage's model weights so
    # the new stage starts at epoch 0 with a fresh optimizer whose param groups
    # match the CURRENT freezing. strict=False tolerates added/removed modules
    # (e.g. S1 numeric-only -> S2a with vision) and the reinit'd output head.
    init_ckpt = cfg.get("init_ckpt", None) or None
    if init_ckpt:
        log.info(f"Warm-starting weights from: {init_ckpt}")
        state = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info(
            f"Warm start: loaded (missing={len(missing)}, unexpected={len(unexpected)})"
        )

    if cfg.get("train", True):
        log.info("Starting training!")
        ckpt_path = cfg.get("ckpt_path", None) or None
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        # Export a stable best.ckpt so the curriculum runner can thread stages
        # deterministically (+ckpt_path=<stage_dir>/best.ckpt). Lightning's own
        # filename embeds val/loss and is not a fixed path.
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            best = _best_finite_checkpoint_path(trainer)
            if best is not None:
                import shutil

                stable = os.path.join(trainer.default_root_dir, "best.ckpt")
                if os.path.abspath(best) != os.path.abspath(stable):
                    shutil.copyfile(best, stable)
                log.info(f"Exported best checkpoint → {stable}")

    if cfg.get("test", True):
        ckpt_path = "best"
        if cfg.get("train", True):
            ckpt_path = _best_finite_checkpoint_path(trainer)
            if ckpt_path is None:
                log.warning("Skipping testing: no finite best checkpoint was produced.")
        elif cfg.get("ckpt_path", None):
            # test-only mode (train=false): score an existing checkpoint —
            # "best" is meaningless without a fit in this process.
            ckpt_path = cfg.ckpt_path
        if ckpt_path is not None:
            log.info(f"Starting testing from checkpoint: {ckpt_path}")
            trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
                weights_only=False,
            )

    # wandb.finish() only on rank 0 — other ranks never called wandb.init()
    import wandb

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
