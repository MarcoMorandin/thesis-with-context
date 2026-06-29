"""VisionChronos2 Lightning training module (Task 3.5).

Implements the full training loop from the roadmap:
  - Mixed precision via Lightning ``precision="16-mixed"``
  - Gradient clipping via Lightning ``gradient_clip_val=1.0``
  - AdamW + linear LR warmup + cosine decay
  - Early stopping on val loss (patience 7 epochs, via callback)
  - Checkpoint: best val loss + every 5 epochs (via callback)
  - W&B logging: train/val loss, lr, gradient norm, per-modality breakdown

Batch schema (from MMTSFMDataset, after DataLoader collation):
    Y                [BS, N, T, 1]
    Y_future         [BS, N, H, 1]
    X_cov            [BS, N, T+H, C_cov]
    V                [BS, N, T_v, C, H_img, W_img]   [0,1]
    mask_target      [BS, N, T, 1]
    mask_future      [BS, N, H, 1]
    mask_visual      [BS, N, T_v]
    mask_modality_dropout [BS, N, 2]
    entity_ids       [BS, N]
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .model import Chronos2Model
from .config import Chronos2CoreConfig
from .vision_chronos2 import VisionChronos2Model, VisionChronos2Config


class VisionChronos2LightningModule(LightningModule):
    """Lightning wrapper for VisionChronos2Model.

    Parameters
    ----------
    chronos_core_cfg:
        Dict of kwargs forwarded to ``Chronos2CoreConfig``.
    vision_cfg:
        Dict of kwargs forwarded to ``VisionChronos2Config``.
    lr:
        Peak learning rate for AdamW.
    weight_decay:
        AdamW weight decay.
    warmup_steps:
        Number of linear warmup steps before cosine decay begins.
    min_lr_ratio:
        ``min_lr = lr * min_lr_ratio`` at end of cosine schedule.
    horizon:
        Forecast horizon (H); used to compute ``num_output_patches``.
    freeze_chronos:
        If True, only vision modules (adapter + summarizer) are trained.
        Chronos-2 backbone is frozen.
    """

    def __init__(
        self,
        chronos_core_cfg: Dict[str, Any],
        vision_cfg: Dict[str, Any],
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 500,
        min_lr_ratio: float = 0.1,
        horizon: int = 12,
        freeze_chronos: bool = False,
        n_unfreeze_encoder_blocks: int = 1,
        backbone_lr_ratio: float = 0.1,
        grassmann_warmup_steps: int = 0,
        vidtok_model: Optional[nn.Module] = None,
        pretrained_model_name_or_path: Optional[str] = "amazon/chronos-2",
        # Protocol evaluation (BASELINE_PROTOCOL.md §5): NMAE/NRMSE/SS written in
        # the baselines results schema so aggregate_all.py ingests MMTSFM too.
        results_dir: str = "results",
        results_tag: str = "mmtsfm_s2_ukpv",
        sp_reference_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vidtok_model"])
        self._protocol_eval = None
        self.grassmann_warmup_steps = grassmann_warmup_steps
        self.n_unfreeze_encoder_blocks = n_unfreeze_encoder_blocks
        self._last_loss = None

        # Build Chronos-2 core
        core_config = Chronos2CoreConfig(**chronos_core_cfg)

        if pretrained_model_name_or_path:
            # Load the pretrained config, then override architecture fields from core_config.
            # This ensures settings like use_grassmann=False actually take effect — calling
            # from_pretrained() without a config restores the checkpoint's saved config
            # (which defaults use_grassmann=True), silently ignoring the YAML override.
            pretrained_config = Chronos2CoreConfig.from_pretrained(pretrained_model_name_or_path)
            pretrained_config.use_grassmann = core_config.use_grassmann
            pretrained_config.grassmann_reduced_dim = core_config.grassmann_reduced_dim
            pretrained_config.grassmann_window_offsets = core_config.grassmann_window_offsets
            pretrained_config.grassmann_plucker_eps = core_config.grassmann_plucker_eps
            # Propagate nested chronos_config overrides (use_arcsinh, max_output_patches, quantiles, …).
            # Without this, YAML values silently lose to the values stored in the HF checkpoint —
            # e.g., use_arcsinh stays False and (target-loc)/scale becomes unbounded → loss overflow.
            yaml_chronos_cfg = getattr(core_config, "chronos_config", None)
            if yaml_chronos_cfg is not None:
                if not isinstance(pretrained_config.chronos_config, dict):
                    pretrained_config.chronos_config = dict(pretrained_config.chronos_config)
                pretrained_config.chronos_config.update(dict(yaml_chronos_cfg))
            chronos_model = Chronos2Model.from_pretrained(
                pretrained_model_name_or_path,
                config=pretrained_config,
                ignore_mismatched_sizes=True,
            )
            # FIX: HF from_pretrained ignores the buffer shape mismatch but reinitializes it with zeros.
            # We must restore our intended quantiles array.
            chronos_model.quantiles.data.copy_(
                torch.tensor(pretrained_config.chronos_config["quantiles"], dtype=chronos_model.dtype)
            )

        else:
            chronos_model = Chronos2Model(core_config)

        # Build vision config
        vcfg = VisionChronos2Config(**vision_cfg)

        # Build full model
        self.model = VisionChronos2Model(
            chronos_model=chronos_model,
            vision_config=vcfg,
            vidtok_model=vidtok_model,
        )

        if freeze_chronos:
            for p in self.model.chronos.parameters():
                p.requires_grad_(False)

            keep_trainable_substrings = (
                "W_red", "W_plu", "W_gate", "offset_weights", "modality_pair_bias",
                # Re-initialised due to checkpoint size mismatch — must learn.
                "input_patch_embedding", "output_patch_embedding", "shared",
            )
            for name, p in self.model.chronos.named_parameters():
                if any(k in name for k in keep_trainable_substrings):
                    p.requires_grad_(True)

            # Unfreeze the last n_unfreeze_encoder_blocks encoder blocks so that
            # group self-attention in those blocks can learn to attend to visual
            # modality rows. More unfrozen blocks = stronger gradient signal to
            # the visual adapter; 1 block causes gradient starvation.
            encoder_blocks = getattr(self.model.chronos.encoder, "block", None)
            if encoder_blocks is not None and len(encoder_blocks) > 0:
                n = min(self.n_unfreeze_encoder_blocks, len(encoder_blocks))
                for block in list(encoder_blocks)[-n:]:
                    for p in block.parameters():
                        p.requires_grad_(True)

        self._output_patch_size: int = self.model.chronos.chronos_config.output_patch_size
        self._num_output_patches: int = max(1, math.ceil(horizon / self._output_patch_size))

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle checkpoint compatibility across architecture changes.

        Drops model keys absent from the current architecture and resets
        optimizer/scheduler state when param groups no longer match
        (e.g. resuming a late-fusion ckpt with interleaved-fusion model).
        """
        import logging
        _log = logging.getLogger(__name__)

        # 1. Drop stale model state_dict keys
        current_keys = set(self.state_dict().keys())
        ckpt_state = checkpoint.get("state_dict", {})
        stale = [k for k in list(ckpt_state.keys()) if k not in current_keys]
        for k in stale:
            del ckpt_state[k]
        if stale:
            _log.warning(
                f"on_load_checkpoint: dropped {len(stale)} stale model keys "
                f"(architecture changed): {stale}"
            )

        # 2. Rebuild optimizer state when architecture changed
        # The old ckpt has param groups for modules that no longer exist.
        # PyTorch's load_state_dict checks param-count-per-group; it fails if they
        # don't match. We build a fresh minimal state that matches the CURRENT
        # param group layout (same structure as configure_optimizers) with an empty
        # `state` dict so Adam moments start from zero.
        if stale:
            bd, bn, nd, nn_ = [], [], [], []
            idx = 0
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                no_decay = any(kw in name for kw in self._NO_DECAY_KWS)
                is_backbone = name.startswith("model.chronos.")
                if is_backbone:
                    (bn if no_decay else bd).append(idx)
                else:
                    (nn_ if no_decay else nd).append(idx)
                idx += 1

            lr  = self.hparams.lr
            blr = lr * self.hparams.backbone_lr_ratio
            wd  = self.hparams.weight_decay

            new_groups = []
            for params, _lr, _wd in [
                (bd,  blr, wd),
                (bn,  blr, 0.0),
                (nd,  lr,  wd),
                (nn_, lr,  0.0),
            ]:
                if params:
                    new_groups.append({
                        "params": params, "lr": _lr, "initial_lr": _lr,
                        "weight_decay": _wd, "betas": (0.9, 0.999), "eps": 1e-8,
                        "amsgrad": False, "maximize": False, "foreach": None,
                        "capturable": False, "differentiable": False, "fused": None,
                    })

            checkpoint["optimizer_states"] = [{"state": {}, "param_groups": new_groups}]

            # LambdaLR.load_state_dict pops "lr_lambdas" then calls super().
            # Lambdas are not serializable so the checkpoint stores None per group.
            all_lrs = [g["lr"] for g in new_groups]
            checkpoint["lr_schedulers"] = [{
                "last_epoch": 0,
                "_step_count": 1,
                "verbose": "deprecated",
                "base_lrs": all_lrs,
                "_last_lr": all_lrs,
                "lr_lambdas": [None] * len(new_groups),
            }]
            # Reset EarlyStopping patience so the new architecture gets a full
            # budget. Without this, the inherited wait_count fires after 1-2 epochs.
            for cb_state in checkpoint.get("callbacks", {}).values():
                if "wait_count" in cb_state:
                    cb_state["wait_count"] = 0
                if "best_score" in cb_state:
                    cb_state["best_score"] = torch.tensor(float("inf"))

            _log.warning(
                "on_load_checkpoint: rebuilt optimizer/scheduler state and reset "
                "EarlyStopping patience for changed architecture."
            )

    # ------------------------------------------------------------------
    # Batch → model inputs
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch: Dict[str, torch.Tensor]):
        """Flatten entity dim into batch dim and convert to Chronos2 input format.

        Returns a dict ready for ``VisionChronos2Model.forward()``.
        """
        Y        = batch["Y"]           # [BS, N, T, 1]
        Y_future = batch["Y_future"]    # [BS, N, H, 1]
        X_cov    = batch["X_cov"]       # [BS, N, T+H, C_cov]
        V        = batch["V"]           # [BS, N, T_v, C, H_img, W_img]
        mask_tgt = batch["mask_target"] # [BS, N, T, 1]
        mask_fut = batch["mask_future"] # [BS, N, H, 1]
        mask_vis = batch["mask_visual"] # [BS, N, T_v]

        BS, N, T, _ = Y.shape
        H = Y_future.shape[2]

        # Flatten [BS, N] → [BS*N]
        context        = Y.reshape(BS * N, T)
        context_mask   = mask_tgt.reshape(BS * N, T)
        future_target  = Y_future.reshape(BS * N, H)
        future_mask    = mask_fut.reshape(BS * N, H)
        visual_mask    = mask_vis.reshape(BS * N, -1)

        # M2 fix: extract per-channel covariate tensors instead of mean-collapsing.
        # Each channel becomes its own token in the encoder (token-type=covariate).
        # The first channel is also passed as ``future_covariates`` to preserve the
        # Chronos-2 loss path which requires a single [B, H] tensor.
        C_cov = X_cov.shape[-1]
        future_cov_slices = X_cov[:, :, T:, :]   # [BS, N, H, C_cov]
        covariate_channels: list[torch.Tensor] = [
            future_cov_slices[..., c].reshape(BS * N, H)   # [BS*N, H]
            for c in range(C_cov)
        ]
        # Primary covariate (first channel) used by Chronos-2 loss internals.
        future_covariates = covariate_channels[0] if C_cov > 0 else torch.zeros(BS * N, H, device=Y.device)
        # C1 fix: pass a zero mask so the model treats these as *unknown* covariates.
        # Without this, model.py auto-builds mask=all-1s (no NaNs after .mean()),
        # which sets inv_future_covariate_mask=all-0s and collapses loss to 0.
        future_covariates_mask = torch.zeros_like(future_covariates)

        # Group IDs: entities within same sample share group
        group_ids = torch.arange(BS, device=Y.device, dtype=torch.long).repeat_interleave(N)

        # Entity position indices 0..N-1 (consistent ordering per dataset)
        entity_ids = (
            torch.arange(N, device=Y.device, dtype=torch.long)
            .unsqueeze(0).expand(BS, -1)
            .reshape(BS * N)
        )

        # Pre-computed VidTok latents (optional cache hit)
        # Z: [BS, N, T_lat, P, D_v] → flatten to [BS*N, T_lat, P, D_v]
        Z_raw = batch.get("Z")
        video_latents: Optional[torch.Tensor] = None
        video: Optional[torch.Tensor] = None

        if Z_raw is not None and Z_raw.numel() > 0:
            # C5 fix: normalise shape before reshape.
            # New producer always saves [N, T_lat, P, D_v] → collated [BS, N, T_lat, P, D_v] (5-D).
            # Old producer squeezed N=1 → saved [T_lat, P, D_v] → collated [BS, T_lat, P, D_v] (4-D).
            # Detect the old format by checking ndim and unsqueeze the missing N dim so both
            # cases arrive here as [BS, N, T_lat, P, D_v] before the reshape below.
            if Z_raw.ndim == 4:
                # Old cache file: [BS, T_lat, P, D_v] → [BS, 1, T_lat, P, D_v]
                Z_raw = Z_raw.unsqueeze(1)
            # Z_raw is now guaranteed [BS, N, T_lat, P, D_v]
            video_latents = Z_raw.reshape(BS * N, *Z_raw.shape[2:])
        else:
            # Cache miss: pass raw frames to VidTokEncoder
            T_v   = V.shape[2]
            C     = V.shape[3]
            H_img = V.shape[4]
            W_img = V.shape[5]
            video = V.reshape(BS * N, T_v, C, H_img, W_img).permute(0, 2, 1, 3, 4)
            # [BS*N, C, T_v, H_img, W_img]

        visual_available = visual_mask.any(dim=-1)  # [BS*N]
        if not visual_available.any():
            video         = None
            video_latents = None

        return dict(
            context=context,
            context_mask=context_mask,
            future_target=future_target,
            future_target_mask=future_mask,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,  # C1 fix: zero mask → treat as unknown
            covariate_channels=covariate_channels,          # M2 fix: per-channel list [BS*N, H] each
            group_ids=group_ids,
            entity_ids=entity_ids,
            video=video,
            visual_mask=visual_mask if (video is not None or video_latents is not None) else None,
            video_latents=video_latents,
            num_output_patches=self._num_output_patches,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test
    # ------------------------------------------------------------------

    def _forward(self, batch: Dict[str, torch.Tensor]):
        """Run the fp32 forward once; return (unpacked inputs, model output)."""
        inputs = self._unpack_batch(batch)
        device_type = self.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            fp32_inputs = {key: self._to_float32(value) for key, value in inputs.items()}
            out = self.model.forward(**fp32_inputs)
        return inputs, out

    def _step(self, batch: Dict[str, torch.Tensor], stage: str):
        inputs, out = self._forward(batch)

        loss = out.loss
        assert loss is not None, "Loss is None — check future_target in batch"
        if not torch.isfinite(loss):
            with torch.no_grad():
                context = inputs["context"]
                future_target = inputs["future_target"]
                raise FloatingPointError(
                    f"Non-finite {stage}/loss={loss.item()} "
                    f"context_finite={torch.isfinite(context).float().mean().item():.4f} "
                    f"future_finite={torch.isfinite(future_target).float().mean().item():.4f}"
                )

        # Log fraction of samples with active visual stream (replaces spurious duplicate loss logs)
        if out.visual_active is not None:
            with torch.no_grad():
                visual_frac = out.visual_active.float().mean()
                self.log(f"{stage}/visual_fraction", visual_frac,
                         on_step=(stage == "train"), on_epoch=True,
                         prog_bar=False, sync_dist=True)

        self.log(f"{stage}/loss", loss,
                 on_step=(stage == "train"), on_epoch=True,
                 prog_bar=True, sync_dist=True)
        if stage == "train":
            self.log("train/loss_epoch", loss,
                     on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True)
        return loss

    @staticmethod
    def _to_float32(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.float()
        if isinstance(value, list):
            return [
                item.float() if isinstance(item, torch.Tensor) and item.is_floating_point() else item
                for item in value
            ]
        return value

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        self._last_loss = loss
        if self.trainer.is_global_zero and batch_idx == 0:
            ep = self.trainer.current_epoch
            print(f"[train] epoch={ep} step={self.trainer.global_step} loss={loss.item():.4f}", flush=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_test_start(self):
        from eval.protocol_eval import ProtocolEvaluator

        self._protocol_eval = ProtocolEvaluator(
            horizon=self.hparams.horizon,
            reference_path=self.hparams.sp_reference_path,
        )

    def test_step(self, batch, batch_idx):
        inputs, out = self._forward(batch)
        loss = out.loss
        if loss is not None and torch.isfinite(loss):
            self.log("test/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if self._protocol_eval is not None and out.quantile_preds is not None:
            self._accumulate_protocol(batch, inputs, out)

    def _accumulate_protocol(self, batch, inputs, out):
        """Collect masked daylight predictions for NMAE/NRMSE/SS (protocol §5)."""
        H = self.hparams.horizon
        q = out.quantile_preds.detach().float()          # [B, Q, H_out]
        q50 = self.model.chronos.num_quantiles // 2
        median = q[:, q50, :H]                            # [B, H]
        quantiles = q[:, :, :H].permute(0, 2, 1)         # [B, H, Q]
        y = inputs["future_target"][:, :H].float()       # [B, H]
        mask = inputs["future_target_mask"][:, :H].float()
        daylight = batch["daylight_future"].reshape(y.shape[0], -1)[:, :H].float()
        site_ids = [str(s) for s in batch["site_id"]]
        self._protocol_eval.update(
            site_ids=site_ids,
            y_true=y.cpu().numpy(),
            median=median.cpu().numpy(),
            mask=(mask * daylight).cpu().numpy(),
            quantiles=quantiles.cpu().numpy(),
        )

    def on_test_epoch_end(self):
        if self._protocol_eval is None or not self.trainer.is_global_zero:
            return
        results = self._protocol_eval.finalize()
        overall = results.get("overall", {})
        for k in ("nmae", "nrmse", "skill_score", "crps"):
            if k in overall:
                self.log(f"test/{k}", float(overall[k]), rank_zero_only=True)
        try:
            from omegaconf import OmegaConf

            run_cfg = {"seed": getattr(self.hparams, "seed", 42),
                       "model": "mmtsfm", "quantile_levels": None}
            path = self._protocol_eval.write(
                self.hparams.results_dir, self.hparams.results_tag, run_cfg
            )
            print(f"[protocol-eval] NMAE={overall.get('nmae'):.4f} "
                  f"NRMSE={overall.get('nrmse'):.4f} "
                  f"SS={overall.get('skill_score', float('nan')):.4f} → {path}", flush=True)
        except Exception as e:  # never fail the run on a results-write hiccup
            print(f"[protocol-eval] results write skipped: {e}", flush=True)

    # ------------------------------------------------------------------
    # Gradient norm logging (before clipping)
    # ------------------------------------------------------------------

    _GRAD_GROUPS: tuple[tuple[str, str], ...] = (
        ("vision_adapter",          "model.cross_modal_adapter"),
        ("latent_summarizer",       "model.latent_summarizer"),
        ("multimodal_embed",        "model.multimodal_embed"),
        ("output_patch_embedding",  "model.chronos.output_patch_embedding"),
        ("input_patch_embedding",   "model.chronos.input_patch_embedding"),
        ("shared",                  "model.chronos.shared"),
    )

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm per param group before Lightning applies clipping.

        Per-group breakdown is the diagnostic signal we use to detect
        gradient starvation in the visual adapter / summarizer (Stage 2a).
        """
        total_sq = 0.0
        per_group_sq: dict[str, float] = {name: 0.0 for name, _ in self._GRAD_GROUPS}
        # Index by parameter id so each grad is counted in exactly one group.
        group_by_param_id: dict[int, str] = {}
        for group_name, prefix in self._GRAD_GROUPS:
            module = self
            for attr in prefix.split("."):
                module = getattr(module, attr, None)
                if module is None:
                    break
            if module is None:
                continue
            for p in module.parameters():
                group_by_param_id[id(p)] = group_name

        for p in self.model.parameters():
            if p.grad is None:
                continue
            sq = p.grad.detach().pow(2).sum().item()
            total_sq += sq
            group = group_by_param_id.get(id(p))
            if group is not None:
                per_group_sq[group] += sq

        grad_norm = total_sq ** 0.5

        if self.trainer.global_step % 500 == 0:
            loss_val = self._last_loss.item() if self._last_loss is not None else float("nan")
            if self.trainer.is_global_zero:
                print(f"[train] step={self.trainer.global_step} loss={loss_val:.4f} grad_norm={grad_norm:.4f}", flush=True)
            self.log("train/loss_500", loss_val, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("train/grad_norm_500", grad_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        self.log("train/grad_norm", grad_norm,
                 on_step=True, on_epoch=False, prog_bar=False)
        for name, sq in per_group_sq.items():
            self.log(f"train/grad_norm/{name}", sq ** 0.5,
                     on_step=True, on_epoch=False, prog_bar=False)

        # ---- NaN/Inf safety ------------------------------------------------
        # bf16 master + fp32 forward still occasionally emits non-finite grads
        # through quantile loss / instance_norm.inverse. Rather than letting
        # the NaN pollute AdamW's moment buffers (which would taint *all*
        # subsequent steps), zero every grad on this rank for this step.
        # Lightning's gradient_clip_val=1.0 then clips a no-op vector.
        if not math.isfinite(grad_norm):
            self.log("train/grad_skipped", 1.0,
                     on_step=True, on_epoch=False, prog_bar=False)
            # Diagnostic: log ALL param groups that carry NaN/Inf grad.
            # Fires only on rank 0, at most once per epoch to avoid log spam.
            # Reading both the first offending param AND which high-level
            # groups are NaN lets us trace whether NaN enters via the visual
            # path (vision_adapter/latent_summarizer) or the encoder itself.
            if self.trainer.is_global_zero:
                epoch = self.trainer.current_epoch
                step  = self.trainer.global_step
                if not hasattr(self, "_nan_logged_epoch") or self._nan_logged_epoch != epoch:
                    self._nan_logged_epoch = epoch
                    # Which high-level groups have NaN?
                    nan_groups = []
                    for gname, sq in per_group_sq.items():
                        if not math.isfinite(sq):
                            nan_groups.append(gname)
                    # Also check unfrozen encoder blocks individually — log
                    # the specific param name inside the block so we can
                    # tell which sub-op (time_attn/group_attn/ffn) is the source.
                    enc_blocks = getattr(
                        getattr(getattr(self.model, "chronos", None), "encoder", None),
                        "block", None,
                    )
                    nan_blocks = []
                    if enc_blocks is not None:
                        for bi, blk in enumerate(enc_blocks):
                            for pname, p in blk.named_parameters():
                                if p.grad is not None and not torch.isfinite(p.grad).all():
                                    n = (~torch.isfinite(p.grad)).sum().item()
                                    nan_blocks.append(f"{bi}:{pname}({n}/{p.grad.numel()})")
                                    break
                    print(
                        f"[NaN-grad] epoch={epoch} step={step} "
                        f"nan_groups={nan_groups} nan_enc_params={nan_blocks}",
                        flush=True,
                    )
                    # Also print first offending leaf param for fine detail
                    for pname, p in self.model.named_parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            n_nan = (~torch.isfinite(p.grad)).sum().item()
                            print(
                                f"[NaN-grad] first_param={pname} "
                                f"n_nonfinite={n_nan}/{p.grad.numel()}",
                                flush=True,
                            )
                            break
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.detach().zero_()
        else:
            self.log("train/grad_skipped", 0.0,
                     on_step=True, on_epoch=False, prog_bar=False)

    # ------------------------------------------------------------------
    # LR logging
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx):
        sch = self.lr_schedulers()
        if sch is not None:
            lr = sch.get_last_lr()[0]
            self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False)

    # ------------------------------------------------------------------
    # Optimizer + LR schedule
    # ------------------------------------------------------------------

    # Keywords whose parameters must NOT receive weight decay.
    _NO_DECAY_KWS: tuple[str, ...] = (
        "bias",
        "layer_norm",
        "LayerNorm",
        "embed",          # modality_embed, segment_embed, token_type_embed, entity_embed
        "null_visual_token",
        "latent_queries",
        "offset_weights",
        "modality_pair_bias",
    )

    def configure_optimizers(self):
        lr  = self.hparams.lr
        wd  = self.hparams.weight_decay
        blr = lr * self.hparams.backbone_lr_ratio

        backbone_decay:   list[torch.Tensor] = []
        backbone_nodecay: list[torch.Tensor] = []
        new_decay:        list[torch.Tensor] = []
        new_nodecay:      list[torch.Tensor] = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_no_decay = any(kw in name for kw in self._NO_DECAY_KWS)
            is_backbone = name.startswith("model.chronos.")
            if is_backbone:
                (backbone_nodecay if is_no_decay else backbone_decay).append(p)
            else:
                (new_nodecay if is_no_decay else new_decay).append(p)

        param_groups: list[dict] = []
        if backbone_decay:
            param_groups.append(
                {"params": backbone_decay,   "lr": blr, "weight_decay": wd,  "name": "backbone_decay"}
            )
        if backbone_nodecay:
            param_groups.append(
                {"params": backbone_nodecay, "lr": blr, "weight_decay": 0.0, "name": "backbone_nodecay"}
            )
        if new_decay:
            param_groups.append(
                {"params": new_decay,        "lr": lr,  "weight_decay": wd,  "name": "new_decay"}
            )
        if new_nodecay:
            param_groups.append(
                {"params": new_nodecay,      "lr": lr,  "weight_decay": 0.0, "name": "new_nodecay"}
            )
        if not param_groups:
            param_groups = [{"params": [], "lr": lr, "weight_decay": 0.0}]

        optimizer = AdamW(param_groups)

        total_steps = self._total_steps
        warmup    = self.hparams.warmup_steps
        min_ratio = self.hparams.min_lr_ratio

        def lr_schedule(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        lambdas = [lr_schedule for _ in param_groups]
        scheduler = LambdaLR(optimizer, lr_lambda=lambdas)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # ------------------------------------------------------------------
    # Total steps (for cosine decay)
    # ------------------------------------------------------------------

    @property
    def _steps_per_epoch(self) -> int:
        try:
            dl = self.trainer.train_dataloader
            if dl is not None:
                n = len(dl)
                accum = self.trainer.accumulate_grad_batches or 1
                return max(1, n // accum)
        except Exception:
            pass
        return 1000  # fallback

    @property
    def _total_steps(self) -> int:
        """Estimate total training steps for cosine decay endpoint."""
        trainer = self.trainer
        if trainer is None:
            return 10_000
        try:
            total = int(trainer.estimated_stepping_batches)
            if total > 0:
                return total
        except Exception:
            pass
        if trainer.max_epochs is None:
            return 10_000
        return trainer.max_epochs * self._steps_per_epoch
