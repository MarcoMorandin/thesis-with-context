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
        # s2d / A30 Stage 0 — "vision projector warmup" (Nemotron §3.1.1). For the
        # first N optimizer steps every param group EXCEPT the visual patch
        # projector is held at lr 0, so a freshly initialised projector cannot
        # inject noise into a warm-started Chronos-2 at step 0. Implemented as an
        # LR gate rather than a requires_grad toggle because flipping
        # requires_grad mid-run re-buckets DDP gradients. 0 disables it, which
        # leaves every existing arm's schedule bit-identical.
        projector_warmup_steps: int = 0,
        # Vision unfreeze schedule (curriculum). freeze_visual_encoder="partial"
        # (in vision_cfg) unfreezes the last ``n_visual_unfreeze_layers`` V-JEPA
        # blocks at construction (Stage 2a). ``progressive_vision_unfreeze`` adds
        # ``n_visual_unfreeze_layers`` more blocks every
        # ``progressive_unfreeze_interval`` epochs (Stage 3).
        n_visual_unfreeze_layers: int = 4,
        progressive_vision_unfreeze: bool = False,
        progressive_unfreeze_interval: int = 1,
        video_encoder: Optional[nn.Module] = None,
        pretrained_model_name_or_path: Optional[str] = "amazon/chronos-2",
        # Protocol evaluation (knowledge/protocol.md §5): NMAE/NRMSE/SS written in
        # the baselines results schema so aggregate_all.py ingests MMTSFM too.
        results_dir: str = "results",
        results_tag: str = "mmtsfm_s2_ukpv",
        sp_reference_path: Optional[str] = None,
        # Recorded in the results manifest for provenance; must equal the
        # baselines' seed (common.config.SEED) for comparable runs.
        seed: int = 42,
        # W6: run a second vision-off pass at test time and report the visual
        # marginal gain (Δ on/off). Off by default — doubles the test forward.
        compute_marginal_gain: bool = False,
        # Abort after this many CONSECUTIVE optimizer steps whose gradients were
        # non-finite. Each such step is zeroed (see on_before_optimizer_step), so
        # an unbroken streak means the run is not learning at all — job 50574535
        # burned 6.5 GPU-hours in exactly that state, val/loss pinned at 6.3977,
        # and would have poisoned every downstream curriculum stage.
        max_nonfinite_grad_steps: int = 50,
        # Test-time negative controls (registry A09/A10). "none" is the only value
        # that leaves the eval path byte-identical to every run recorded so far;
        # the others corrupt the visual input in a specific, structured way at
        # TEST TIME ONLY so that a non-zero marginal gain can be attributed to
        # genuine sky information rather than to a per-plant or per-day constant.
        #   "shuffle_frames"     — permute the temporal axis within each sample
        #   "swap_plant_frames"  — give each sample another PLANT's sky sequence
        #   "stale_sky"          — give each sample the SAME plant's sky from one
        #                          horizon earlier (persistence of the visual input)
        eval_control: str = "none",
        # Escape hatch for eval_control="shuffle_frames" on an arm whose visual
        # pathway is provably permutation-invariant (see
        # _assert_eval_control_is_falsifiable). Default False so a control that
        # CANNOT degrade refuses to run instead of writing a null that reads as
        # an empirical finding.
        eval_control_allow_inert: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["video_encoder"])
        # The frozen V-JEPA encoder is stripped from checkpoints (see
        # on_save_checkpoint); loading must therefore tolerate missing keys.
        self.strict_loading = False
        self._protocol_eval = None
        # A09/A10: armed in on_test_start only, so training and validation take
        # the untouched path no matter what eval_control says.
        self._eval_control_active = False
        self.grassmann_warmup_steps = grassmann_warmup_steps
        self.projector_warmup_steps = projector_warmup_steps
        self.n_unfreeze_encoder_blocks = n_unfreeze_encoder_blocks
        self._last_loss = None
        self._nonfinite_grad_streak = 0
        # Whether the V-JEPA encoder weights still equal the torch.hub baseline.
        # Once ANY stage fine-tunes it the weights diverge and must be persisted;
        # see on_save_checkpoint. Propagated across stages via the checkpoint.
        self._vjepa_finetuned = False

        # Build Chronos-2 core
        core_config = Chronos2CoreConfig(**chronos_core_cfg)

        if pretrained_model_name_or_path:
            # Load the pretrained config, then override architecture fields from core_config.
            # This ensures settings like use_grassmann=False actually take effect — calling
            # from_pretrained() without a config restores the checkpoint's saved config
            # (which defaults use_grassmann=True), silently ignoring the YAML override.
            pretrained_config = Chronos2CoreConfig.from_pretrained(
                pretrained_model_name_or_path
            )
            pretrained_config.use_grassmann = core_config.use_grassmann
            pretrained_config.grassmann_reduced_dim = core_config.grassmann_reduced_dim
            pretrained_config.grassmann_window_offsets = (
                core_config.grassmann_window_offsets
            )
            pretrained_config.grassmann_plucker_eps = core_config.grassmann_plucker_eps
            # The §8.1 no-modbias ablation flips this flag; without propagation the
            # hub config's default (True) silently wins and the ablation no-ops.
            pretrained_config.grassmann_modality_pair_bias = (
                core_config.grassmann_modality_pair_bias
            )
            # s2c: same trap as the flag above. `from_pretrained` restores the hub
            # config, whose default is 0, so without this line the YAML's
            # visual_cross_attn_blocks is discarded, NO cross-attention module is
            # built, and the arm trains as a plain s2b-shaped model while every log
            # line still says s2c.
            pretrained_config.visual_cross_attn_blocks = (
                core_config.visual_cross_attn_blocks
            )
            pretrained_config._attn_implementation = core_config._attn_implementation
            # Propagate nested chronos_config overrides (use_arcsinh, max_output_patches, quantiles, …).
            # Without this, YAML values silently lose to the values stored in the HF checkpoint —
            # e.g., use_arcsinh stays False and (target-loc)/scale becomes unbounded → loss overflow.
            yaml_chronos_cfg = getattr(core_config, "chronos_config", None)
            if yaml_chronos_cfg is not None:
                if not isinstance(pretrained_config.chronos_config, dict):
                    pretrained_config.chronos_config = dict(
                        pretrained_config.chronos_config
                    )
                pretrained_config.chronos_config.update(dict(yaml_chronos_cfg))
            chronos_model = Chronos2Model.from_pretrained(
                pretrained_model_name_or_path,
                config=pretrained_config,
                ignore_mismatched_sizes=True,
            )
            # FIX: HF from_pretrained ignores the buffer shape mismatch but reinitializes it with zeros.
            # We must restore our intended quantiles array.
            chronos_model.quantiles.data.copy_(
                torch.tensor(
                    pretrained_config.chronos_config["quantiles"],
                    dtype=chronos_model.dtype,
                )
            )

        else:
            chronos_model = Chronos2Model(core_config)

        # Build vision config
        vcfg = VisionChronos2Config(**vision_cfg)

        # Build full model
        self.model = VisionChronos2Model(
            chronos_model=chronos_model,
            vision_config=vcfg,
            video_encoder=video_encoder,
        )

        if freeze_chronos:
            for p in self.model.chronos.parameters():
                p.requires_grad_(False)

            keep_trainable_substrings = (
                "W_red",
                "W_plu",
                "W_gate",
                "offset_weights",
                "modality_pair_bias",
                # Re-initialised due to checkpoint size mismatch — must learn.
                "input_patch_embedding",
                "output_patch_embedding",
                # s2c only: built exactly when output_patch_size != input_patch_size,
                # because the future patch no longer fits `input_patch_embedding`.
                # Fresh weights with no pretrained or warm-started counterpart, so it
                # must learn; the substring above does not match this name.
                "future_patch_embedding",
                "shared",
                # s2c: the visual cross-attention sits inside `chronos.encoder.block`,
                # so the blanket freeze above catches it. It is a NEW module with no
                # pretrained weights — it must learn. Named here rather than left to
                # the `n_unfreeze_encoder_blocks` tail below, so that trainability does
                # not silently depend on two independent settings happening to agree.
                "visual_cross_attn",
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

        # Vision unfreeze policy (curriculum Stage 2a / 3). freeze_visual_encoder
        # may be True (frozen), False (fully trainable), or "partial" (unfreeze
        # the last n_visual_unfreeze_layers blocks now). The V-JEPA encoder is
        # built frozen when truthy; "partial" then reopens the top blocks.
        self._apply_vision_unfreeze_policy(
            vision_cfg.get("freeze_visual_encoder", True)
        )
        # partial/False opens encoder params for training, so from here on this
        # encoder is no longer the pristine hub baseline.
        _enc = getattr(self.model, "video_encoder", None)
        if _enc is not None and any(p.requires_grad for p in _enc.parameters()):
            self._vjepa_finetuned = True

        self._output_patch_size: int = (
            self.model.chronos.chronos_config.output_patch_size
        )
        self._num_output_patches: int = max(
            1, math.ceil(horizon / self._output_patch_size)
        )

    def _apply_vision_unfreeze_policy(self, freeze_visual_encoder) -> None:
        """Apply the initial V-JEPA freeze/unfreeze for the current stage."""
        enc = getattr(self.model, "video_encoder", None)
        if enc is None:
            return
        policy = str(freeze_visual_encoder).lower()
        if policy == "partial" and hasattr(enc, "partial_unfreeze"):
            enc.partial_unfreeze(self.hparams.n_visual_unfreeze_layers)
        elif policy in ("false", "0") and hasattr(enc, "set_freeze"):
            enc.set_freeze(False)
        # True / "true" → leave frozen as constructed.

    def on_train_epoch_start(self) -> None:
        """Stage 3 progressive unfreeze: reopen more V-JEPA blocks over epochs."""
        if not self.hparams.progressive_vision_unfreeze:
            return
        enc = getattr(self.model, "video_encoder", None)
        if enc is None or not hasattr(enc, "partial_unfreeze"):
            return
        interval = max(1, self.hparams.progressive_unfreeze_interval)
        step = self.hparams.n_visual_unfreeze_layers
        n_open = step * (self.current_epoch // interval + 1)
        enc.partial_unfreeze(n_open)

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Strip the V-JEPA encoder (~1.2 GB/ckpt) — ONLY while it is still
        bit-identical to the torch.hub baseline.

        Stripping is safe only because the encoder is rebuilt from the torch.hub
        cache at module init. That holds until some stage fine-tunes it; after
        that the weights are unique to this run, and dropping them silently
        substitutes the pretrained baseline on the next load.

        The old predicate asked "does any encoder param require grad RIGHT NOW",
        which is a different question, and it broke the uk_pv curriculum: s2a
        (freeze_visual_encoder=partial) tuned the last 4 blocks and kept them;
        s2b (=true) inherited those tuned weights and trained/tested with them
        for SS 0.5188, then wrote its checkpoint with the encoder stripped
        because everything was frozen at save time. s3 warm-started from that
        file, reported `missing=302` (the whole encoder), and silently ran on
        the pristine baseline — invalidating the stage and leaving s2b's score
        unreproducible from its own checkpoint.

        `vjepa_finetuned` is persisted so the answer survives across stages.
        """
        enc = getattr(self.model, "video_encoder", None)
        if enc is None:
            return
        if any(p.requires_grad for p in enc.parameters()):
            self._vjepa_finetuned = True
        checkpoint["vjepa_finetuned"] = self._vjepa_finetuned
        if self._vjepa_finetuned:
            return  # unique to this run — must be persisted
        prefix = "model.video_encoder."
        state = checkpoint.get("state_dict", {})
        for k in [k for k in state if k.startswith(prefix)]:
            del state[k]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle checkpoint compatibility across architecture changes.

        Drops model keys absent from the current architecture and resets
        optimizer/scheduler state when param groups no longer match
        (e.g. resuming a late-fusion ckpt with interleaved-fusion model).
        """
        import logging

        _log = logging.getLogger(__name__)

        # Sticky: once any stage has fine-tuned the encoder it stays finetuned.
        if checkpoint.get("vjepa_finetuned", False):
            self._vjepa_finetuned = True

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

            lr = self.hparams.lr
            blr = lr * self.hparams.backbone_lr_ratio
            wd = self.hparams.weight_decay

            new_groups = []
            for params, _lr, _wd in [
                (bd, blr, wd),
                (bn, blr, 0.0),
                (nd, lr, wd),
                (nn_, lr, 0.0),
            ]:
                if params:
                    new_groups.append(
                        {
                            "params": params,
                            "lr": _lr,
                            "initial_lr": _lr,
                            "weight_decay": _wd,
                            "betas": (0.9, 0.999),
                            "eps": 1e-8,
                            "amsgrad": False,
                            "maximize": False,
                            "foreach": None,
                            "capturable": False,
                            "differentiable": False,
                            "fused": None,
                        }
                    )

            checkpoint["optimizer_states"] = [{"state": {}, "param_groups": new_groups}]

            # LambdaLR.load_state_dict pops "lr_lambdas" then calls super().
            # Lambdas are not serializable so the checkpoint stores None per group.
            all_lrs = [g["lr"] for g in new_groups]
            checkpoint["lr_schedulers"] = [
                {
                    "last_epoch": 0,
                    "_step_count": 1,
                    "verbose": "deprecated",
                    "base_lrs": all_lrs,
                    "_last_lr": all_lrs,
                    "lr_lambdas": [None] * len(new_groups),
                }
            ]
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
        Y = batch["Y"]  # [BS, N, T, 1]
        Y_future = batch["Y_future"]  # [BS, N, H, 1]
        X_cov = batch["X_cov"]  # [BS, N, T+H, C_cov]
        V = batch["V"]  # [BS, N, T_v, C, H_img, W_img]
        mask_tgt = batch["mask_target"]  # [BS, N, T, 1]
        mask_fut = batch["mask_future"]  # [BS, N, H, 1]
        mask_vis = batch["mask_visual"]  # [BS, N, T_v]

        BS, N, T, _ = Y.shape
        H = Y_future.shape[2]

        # Flatten [BS, N] → [BS*N]
        context = Y.reshape(BS * N, T)
        context_mask = mask_tgt.reshape(BS * N, T)
        future_target = Y_future.reshape(BS * N, H)
        future_mask = mask_fut.reshape(BS * N, H)
        visual_mask = mask_vis.reshape(BS * N, -1)

        # M2 fix: extract per-channel covariate tensors instead of mean-collapsing.
        # Each channel becomes its own token in the encoder (token-type=covariate).
        # The first channel is also passed as ``future_covariates`` to preserve the
        # Chronos-2 loss path which requires a single [B, H] tensor.
        C_cov = X_cov.shape[-1]
        future_cov_slices = X_cov[:, :, T:, :]  # [BS, N, H, C_cov]
        covariate_channels: list[torch.Tensor] = [
            future_cov_slices[..., c].reshape(BS * N, H)  # [BS*N, H]
            for c in range(C_cov)
        ]
        # Primary covariate (first channel) used by Chronos-2 loss internals.
        future_covariates = (
            covariate_channels[0]
            if C_cov > 0
            else torch.zeros(BS * N, H, device=Y.device)
        )
        # C1 fix: pass a zero mask so the model treats these as *unknown* covariates.
        # Without this, model.py auto-builds mask=all-1s (no NaNs after .mean()),
        # which sets inv_future_covariate_mask=all-0s and collapses loss to 0.
        future_covariates_mask = torch.zeros_like(future_covariates)

        # Group IDs: entities within same sample share group
        group_ids = torch.arange(
            BS, device=Y.device, dtype=torch.long
        ).repeat_interleave(N)

        # Entity position indices 0..N-1 (consistent ordering per dataset)
        entity_ids = (
            torch.arange(N, device=Y.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(BS, -1)
            .reshape(BS * N)
        )

        # Pre-computed V-JEPA latents (optional cache hit)
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
            # Cache miss: pass raw frames to the V-JEPA video encoder
            T_v = V.shape[2]
            C = V.shape[3]
            H_img = V.shape[4]
            W_img = V.shape[5]
            video = V.reshape(BS * N, T_v, C, H_img, W_img).permute(0, 2, 1, 3, 4)
            # [BS*N, C, T_v, H_img, W_img]

        visual_available = visual_mask.any(dim=-1)  # [BS*N]
        if not visual_available.any():
            video = None
            video_latents = None

        # W5: per-frame Δt (seconds before forecast origin). Threaded to the
        # summarizer, which only uses it when its length matches the latent
        # temporal dim; otherwise it falls back to uniform spacing.
        vdt_raw = batch.get("video_delta_t")
        video_delta_t: Optional[torch.Tensor] = None
        if (
            vdt_raw is not None
            and vdt_raw.numel() > 0
            and (video is not None or video_latents is not None)
        ):
            video_delta_t = vdt_raw.reshape(BS * N, -1)

        return dict(
            context=context,
            context_mask=context_mask,
            future_target=future_target,
            future_target_mask=future_mask,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,  # C1 fix: zero mask → treat as unknown
            covariate_channels=covariate_channels,  # M2 fix: per-channel list [BS*N, H] each
            group_ids=group_ids,
            entity_ids=entity_ids,
            video=video,
            visual_mask=visual_mask
            if (video is not None or video_latents is not None)
            else None,
            video_latents=video_latents,
            video_delta_t=video_delta_t,
            num_output_patches=self._num_output_patches,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test
    # ------------------------------------------------------------------

    _EVAL_CONTROLS = ("none", "shuffle_frames", "swap_plant_frames", "stale_sky")

    @staticmethod
    def _row_site_ids(batch: Any, n_rows: int) -> Optional[list]:
        """Per-row plant identity, or None when the batch does not carry one.

        ``_unpack_batch`` entity_ids are POSITIONAL (0..N-1 within the sample),
        so they cannot identify a plant.  ``batch["site_id"]`` is the real one
        (pv_record.py), and it deliberately never enters the inputs dict — that
        dict is ``**``-splatted into ``model.forward`` and mapped through
        ``_to_float32``, neither of which accepts a string column.
        """
        if not isinstance(batch, dict):
            return None
        sid = batch.get("site_id")
        if sid is None:
            return None
        if isinstance(sid, torch.Tensor):
            sid = [str(v) for v in sid.reshape(-1).tolist()]
        elif isinstance(sid, str):
            sid = [sid]
        else:
            sid = [str(v) for v in sid]
        if len(sid) == n_rows:
            return sid
        # _unpack_batch flattens [BS, N] -> [BS*N]. N == 1 at val/test
        # (datamodule W4), so this only fires if that ever changes.
        if sid and n_rows % len(sid) == 0:
            reps = n_rows // len(sid)
            return [s for s in sid for _ in range(reps)]
        return None

    @staticmethod
    def _cross_site_donor(sites: list, generator: torch.Generator) -> Optional[list]:
        """Row index j for every row i such that ``sites[j] != sites[i]``.

        Rows are grouped by plant and the GROUPS are rotated under a seeded
        permutation, rather than each row drawing an independent random donor.
        A per-row draw on a batch of 32 leaves the donor distribution lumpy and
        can hand several rows the same sky; a rotation of distinct groups gives
        every row a different plant and spreads the donors evenly.

        Returns None when the batch holds a single plant — no row can then be
        given a *different* plant's sky, and silently rolling within one site
        would measure staleness while claiming to measure identity.
        """
        groups: Dict[str, list] = {}
        for i, s in enumerate(sites):
            groups.setdefault(s, []).append(i)
        keys = sorted(groups)
        if len(keys) < 2:
            return None
        perm = torch.randperm(len(keys), generator=generator).tolist()
        donor_of = {
            keys[perm[k]]: keys[perm[(k + 1) % len(keys)]] for k in range(len(keys))
        }
        idx = [0] * len(sites)
        for site, rows in groups.items():
            pool = groups[donor_of[site]]
            for r, i in enumerate(rows):
                idx[i] = pool[r % len(pool)]
        return idx

    def _apply_eval_control(
        self, inputs: Dict[str, Any], batch: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Corrupt the visual stream at TEST TIME for the negative controls.

        A non-zero marginal gain says only that the visual tokens carry
        *something* the numeric stream lacks — a per-plant bias or a
        time-of-day constant would produce exactly the same signature as real
        sky information.  Three controls separate those:

        ``shuffle_frames``
            Permutes the temporal axis within each sample.  Every frame the
            model sees still belongs to the right plant on the right day, so
            any per-plant or per-day constant survives untouched; only the
            *temporal ordering* — cloud advection — is destroyed.  The Δt
            vector is deliberately NOT permuted, so the claimed timestamps no
            longer describe the frames.

            Inert by construction on some arms: see
            ``_assert_eval_control_is_falsifiable``, which refuses the run
            rather than let it write a null.

        ``swap_plant_frames``
            Gives each sample a DIFFERENT plant's sky, matched row-to-row by
            ``batch["site_id"]``.  Destroys plant identity and site-specific
            bias as well as ordering.  Raises when the batch holds one plant —
            the protocol test loader is series-major and unshuffled, so whole
            batches are a single site and this control needs
            ``data.shuffle_test=true``.

        ``stale_sky``
            Gives each sample the SAME plant's sky from one horizon earlier
            (the previous window in series-major order).  Plant identity,
            site bias and internal frame ordering all survive; only the
            *currency* of the sky is destroyed.  This is persistence applied
            to the visual input, and it is the control that separates "the
            model reads THIS sky" from "the model reads A sky from this site".

        All three are pure test-time input transforms: no weights, no config,
        and no training path changes, so the model under control is
        bit-identical to the model under the headline number.
        """
        control = getattr(self.hparams, "eval_control", "none")
        if control == "none" or not self._eval_control_active:
            return inputs
        if control not in self._EVAL_CONTROLS:
            raise ValueError(
                f"eval_control={control!r} is not one of {self._EVAL_CONTROLS}"
            )
        video = inputs.get("video")
        latents = inputs.get("video_latents")
        if video is None and latents is None:
            return inputs  # vision-off arm: control is a no-op by construction

        out = dict(inputs)
        ref = latents if latents is not None else video
        B = ref.shape[0]
        seed = int(getattr(self.hparams, "seed", 42))

        if control == "shuffle_frames":
            # One independent permutation per sample, drawn from a generator
            # seeded off the run seed so the control is reproducible.
            gen = torch.Generator().manual_seed(seed)
            lat_perm = vid_perm = None
            if latents is not None:
                # [B, T_lat, P, D_v] — temporal axis 1
                lat_perm = torch.stack(
                    [torch.randperm(latents.shape[1], generator=gen) for _ in range(B)]
                ).to(latents.device)
                idx = lat_perm.view(B, -1, 1, 1).expand_as(latents)
                out["video_latents"] = torch.gather(latents, 1, idx)
            if video is not None:
                # [B, C, T_v, H, W] — temporal axis 2
                vid_perm = torch.stack(
                    [torch.randperm(video.shape[2], generator=gen) for _ in range(B)]
                ).to(video.device)
                idx = vid_perm.view(B, 1, -1, 1, 1).expand_as(video)
                out["video"] = torch.gather(video, 2, idx)
            # visual_mask is per-frame availability; permute it with the frames
            # so a shuffled-in frame is not simultaneously marked absent. Match
            # it against whichever tensor shares its frame count — on the
            # cached-latent path `video` is absent entirely, and gating on it
            # left the mask unpermuted for every V-JEPA run.
            vm = inputs.get("visual_mask")
            if vm is not None and vm.dim() >= 2:
                for perm, n in ((lat_perm, vm.shape[1]), (vid_perm, vm.shape[1])):
                    if perm is not None and perm.shape[1] == n:
                        out["visual_mask"] = torch.gather(vm, 1, perm.to(vm.device))
                        break
        elif control == "swap_plant_frames":
            sites = self._row_site_ids(batch, B)
            if sites is None:
                raise RuntimeError(
                    "eval_control='swap_plant_frames' needs batch['site_id'] to "
                    "match every row to a DIFFERENT plant, and this batch does "
                    "not carry it. entity_ids are positional (0..N-1) and cannot "
                    "identify a site. Score a pv_record dataset (uk_pv / "
                    "goes_pvdaq), not synthetic."
                )
            donor = self._cross_site_donor(sites, torch.Generator().manual_seed(seed))
            if donor is None:
                raise RuntimeError(
                    "eval_control='swap_plant_frames': every row of this batch is "
                    f"plant {sites[0]!r}, so no sample can be handed a DIFFERENT "
                    "plant's sky. The protocol test loader is unshuffled and its "
                    "windows are series-major (~984 per plant), so whole batches "
                    "are one site. Set data.shuffle_test=true — "
                    "configs/ablation/A10.yaml does. To score the same-plant "
                    "stale-sky control instead, use eval_control='stale_sky'."
                )
            idx = torch.as_tensor(donor, dtype=torch.long)
            for key in ("video", "video_latents", "visual_mask", "video_delta_t"):
                val = inputs.get(key)
                if val is not None:
                    out[key] = val.index_select(0, idx.to(val.device))
        else:  # stale_sky
            # Roll every visual tensor TOGETHER by one row so frames, mask and
            # Δt stay mutually consistent — they are simply one horizon stale.
            for key in ("video", "video_latents", "visual_mask", "video_delta_t"):
                val = inputs.get(key)
                if val is not None:
                    out[key] = torch.roll(val, shifts=1, dims=0)
            # "One window earlier at the same site" only holds while the loader
            # is series-major and unshuffled. Under data.shuffle_test=true the
            # same roll is an unlabelled mixture of sites and horizons, which is
            # not a control at all.
            sites = self._row_site_ids(batch, B)
            if sites is not None and B > 1:
                same = sum(sites[i] == sites[i - 1] for i in range(1, B))
                if same < (B - 1) // 2:
                    raise RuntimeError(
                        "eval_control='stale_sky' expects the ordered "
                        "series-major test loader, where rolling by one row is "
                        "the SAME plant one horizon earlier. Only "
                        f"{same}/{B - 1} adjacent rows share a plant here, so "
                        "the roll is crossing sites. Set data.shuffle_test=false."
                    )
        return out

    def _assert_eval_control_is_falsifiable(self) -> None:
        """Refuse a control the architecture cannot possibly react to.

        ``shuffle_frames`` is a claim about temporal ordering, and two of the
        visual pathways here are permutation-invariant over frames by
        construction.  Running it on one of those produces Δ = 0.000 that reads
        exactly like the empirical finding "motion does not matter", when it is
        in fact a statement about wiring that no checkpoint could falsify.

        Set ``model.eval_control_allow_inert=true`` to record that
        architectural null deliberately.
        """
        control = getattr(self.hparams, "eval_control", "none")
        if control != "shuffle_frames":
            return
        reasons = []
        vcfg = getattr(self.model, "vcfg", None)
        if vcfg is not None and getattr(vcfg, "fusion_mode", None) == "future_query":
            reasons.append(
                "fusion_mode='future_query' (s2c): _build_visual_kv block-pools "
                "[T_lat, g, g] and flattens it into ONE unordered key set with no "
                "temporal or spatial embedding, model.py attends to it under an "
                "all-zero kv_mask, and TimeCrossAttention builds its MHA with "
                "use_rope=False — so the KV carries no position at all. Softmax "
                "over an unordered key set is invariant to permuting T_lat, so "
                "shuffle_frames changes nothing bit-for-bit. Making it bite needs "
                "a temporal embedding on the visual KV — a TRAINING change, not "
                "an eval flag, and NOT n_visual_context_steps (this branch never "
                "reaches the LatentSummarizer)."
            )
        summ = getattr(self.model, "latent_summarizer", None)
        if summ is not None and int(getattr(summ, "n_vis_steps", 0)) <= 1:
            reasons.append(
                "LatentSummarizer has n_vis_steps=1: its K/V carry no positional "
                "encoding and the single query's causal threshold admits every "
                "frame, so the summary is a set function over the token bag and a "
                "frame permutation is invisible. n_visual_context_steps > 1 makes "
                "it falsifiable."
            )
        if not reasons:
            return
        if bool(getattr(self.hparams, "eval_control_allow_inert", False)):
            print(
                "[eval-control] shuffle_frames is INERT on this arm "
                "(allow_inert=true). The Δ it reports is an ARCHITECTURAL null, "
                "not an empirical one — report it as such:\n  - "
                + "\n  - ".join(reasons),
                flush=True,
            )
            return
        raise RuntimeError(
            "eval_control='shuffle_frames' cannot degrade this arm — it is a "
            "no-op by construction, so a null result would say nothing about "
            "whether the model reads cloud motion:\n  - "
            + "\n  - ".join(reasons)
            + "\nEither score an arm whose visual pathway encodes frame order, "
            "or set model.eval_control_allow_inert=true to record the "
            "architectural null on purpose."
        )

    def _forward(self, batch: Dict[str, torch.Tensor], force_vision_off: bool = False):
        """Run the fp32 forward once; return (unpacked inputs, model output)."""
        inputs = self._unpack_batch(batch)
        # `batch`, not `inputs`: site_id is the only real plant identity and it
        # must stay out of the dict that is splatted into model.forward.
        inputs = self._apply_eval_control(inputs, batch)
        device_type = self.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            fp32_inputs = {
                key: self._to_float32(value) for key, value in inputs.items()
            }
            out = self.model.forward(**fp32_inputs, force_vision_off=force_vision_off)
        return inputs, out

    def _step(
        self, batch: Dict[str, torch.Tensor], stage: str, force_vision_off: bool = False
    ):
        inputs, out = self._forward(batch, force_vision_off=force_vision_off)

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
                self.log(
                    f"{stage}/visual_fraction",
                    visual_frac,
                    on_step=(stage == "train"),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if stage == "train":
            self.log(
                "train/loss_epoch",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss

    @staticmethod
    def _to_float32(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.float()
        if isinstance(value, list):
            return [
                item.float()
                if isinstance(item, torch.Tensor) and item.is_floating_point()
                else item
                for item in value
            ]
        return value

    def on_train_start(self):
        """Fail loud on non-finite params before burning GPU hours.

        Runs after checkpoint restore, so it catches both uninitialized
        from_pretrained garbage (params missing from ckpt that _init_weights
        skipped) and NaN-poisoned resume checkpoints. Without this, the
        NaN-grad zeroing in on_before_optimizer_step silently freezes training
        at init quality for the whole run.
        """
        bad = [
            name
            for name, p in self.model.named_parameters()
            if p.is_floating_point() and not torch.isfinite(p).all()
        ]
        if bad:
            raise FloatingPointError(
                f"Non-finite parameters at train start (bad init or corrupt "
                f"checkpoint): {bad[:8]}{' ...' if len(bad) > 8 else ''}"
            )

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        self._last_loss = loss
        if self.trainer.is_global_zero and batch_idx == 0:
            ep = self.trainer.current_epoch
            print(
                f"[train] epoch={ep} step={self.trainer.global_step} loss={loss.item():.4f}",
                flush=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_test_start(self):
        from eval.protocol_eval import ProtocolEvaluator

        # A09/A10 arm here and nowhere else — training/validation never see it.
        self._eval_control_active = True
        control = getattr(self.hparams, "eval_control", "none")
        if control != "none":
            print(f"[eval-control] ACTIVE: {control}", flush=True)
            # Before a single batch is scored: a control that cannot degrade
            # this arm writes a null indistinguishable from an empirical one.
            self._assert_eval_control_is_falsifiable()

        self._protocol_eval = ProtocolEvaluator(
            horizon=self.hparams.horizon,
            reference_path=self.hparams.sp_reference_path,
            compute_marginal_gain=getattr(self.hparams, "compute_marginal_gain", False),
        )
        # s2c (ticket 15): capture is armed for EVAL ONLY. It forces the eager
        # attention path in the cross-attention modules, which is slower and would
        # perturb nothing numerically but has no business running during training.
        self._horizon_attn = None
        self._horizon_attn_blocks = self._cross_attn_blocks()
        for blk in self._horizon_attn_blocks:
            blk.capture_visual_attn = True

    def _cross_attn_blocks(self) -> list:
        """Encoder blocks that actually own a visual cross-attention module.

        Empty for every arm before s2c (visual_cross_attn_blocks defaults to 0), so
        the whole diagnostic is inert rather than conditional on a config flag.
        """
        enc = getattr(getattr(self.model, "chronos", None), "encoder", None)
        blocks = getattr(enc, "block", None)
        if blocks is None:
            return []
        return [b for b in blocks if getattr(b, "visual_cross_attn", None) is not None]

    def _capture_horizon_attention(self) -> None:
        """Harvest one batch of per-tau attention over the visual field.

        Called after the vision-ON pass only. The vision-off pass builds no
        visual_kv, so the cross-attention never runs and the stored maps would be
        stale; they are cleared here so a missed call fails loudly (None) instead of
        silently double-counting the previous batch.
        """
        blocks = getattr(self, "_horizon_attn_blocks", None)
        if not blocks:
            return
        maps = [getattr(b, "last_visual_attn", None) for b in blocks]
        active = getattr(self.model, "_last_visual_active", None)
        if any(m is None for m in maps) or active is None:
            return
        idx = torch.nonzero(active.reshape(-1).bool(), as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        t_fut = int(self._num_output_patches)
        # [rows, heads, seq, n_kv] -> head-average -> active target rows, future
        # positions only. Covariate rows live past index B and are masked out of the
        # residual anyway; taking them here would pollute the distribution.
        stack = [m.float().mean(dim=1)[idx][:, -t_fut:, :] for m in maps]
        arr = torch.stack(stack, dim=0).detach().cpu().numpy()
        if self._horizon_attn is None:
            from mmtsfm.models.chronos2.horizon_attention import (
                HorizonAttentionAccumulator,
            )

            self._horizon_attn = HorizonAttentionAccumulator(
                n_blocks=arr.shape[0], n_tau=arr.shape[2], n_kv=arr.shape[3]
            )
        self._horizon_attn.update(arr)
        for blk in blocks:
            blk.last_visual_attn = None

    def test_step(self, batch, batch_idx):
        # Pass 1: vision-on
        inputs, out = self._forward(batch)
        loss = out.loss
        if loss is not None and torch.isfinite(loss):
            self.log("test/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if self._protocol_eval is not None and out.quantile_preds is not None:
            self._accumulate_protocol(batch, inputs, out, vision_off=False)
        self._capture_horizon_attention()

        # Pass 2: vision-off (W6 visual marginal gain). Same window, forced
        # vision-masked forward; only run when marginal gain is requested.
        if self._protocol_eval is not None and getattr(
            self._protocol_eval, "compute_marginal_gain", False
        ):
            inputs_off, out_off = self._forward(batch, force_vision_off=True)
            if out_off.quantile_preds is not None:
                self._accumulate_protocol(batch, inputs_off, out_off, vision_off=True)

    def _accumulate_protocol(self, batch, inputs, out, vision_off: bool = False):
        """Collect masked daylight predictions for NMAE/NRMSE/SS (protocol §5)."""
        H = self.hparams.horizon
        q = out.quantile_preds.detach().float()  # [B, Q, H_out]
        q50 = self.model.chronos.num_quantiles // 2
        median = q[:, q50, :H]  # [B, H]
        quantiles = q[:, :, :H].permute(0, 2, 1)  # [B, H, Q]
        y = inputs["future_target"][:, :H].float()  # [B, H]
        mask = inputs["future_target_mask"][:, :H].float()
        # daylight/site_id come from the real PV loader; default gracefully so the
        # protocol path also runs on synthetic batches (smoke) — daylight=all-on,
        # site_id=row index.
        if "daylight_future" in batch:
            daylight = batch["daylight_future"].reshape(y.shape[0], -1)[:, :H].float()
        else:
            daylight = torch.ones_like(y)
        if "site_id" in batch:
            site_ids = [str(s) for s in batch["site_id"]]
        else:
            site_ids = [str(i) for i in range(y.shape[0])]
        # S6 ramp inputs — same rule as baselines/common/runner._future_deltas:
        # |Δy| vs the previous step (step 0 uses the last history step), validity
        # = mask_future·daylight·prev_mask. Thresholding happens in finalize().
        delta = delta_valid = None
        if "context" in inputs and "context_mask" in inputs:
            prev = torch.cat([inputs["context"][:, -1:].float(), y[:, :-1]], dim=1)
            prev_mask = torch.cat(
                [inputs["context_mask"][:, -1:].float(), mask[:, :-1]], dim=1
            )
            delta = (y - prev).abs()
            delta_valid = mask * daylight * prev_mask
        self._protocol_eval.update(
            site_ids=site_ids,
            y_true=y.cpu().numpy(),
            median=median.cpu().numpy(),
            mask=(mask * daylight).cpu().numpy(),
            quantiles=quantiles.cpu().numpy(),
            vision_off=vision_off,
            delta=None if delta is None else delta.cpu().numpy(),
            delta_valid=None if delta_valid is None else delta_valid.cpu().numpy(),
        )

    def _emit_horizon_attention(self) -> None:
        """Fold the s2c horizon-attention diagnostic into the results JSON.

        Runs whenever the arm HAS cross-attention blocks, even if nothing was
        captured: a missing key and a "not_measured" verdict are different claims,
        and the second one is the honest one when the machinery was present but
        never fired.
        """
        for blk in getattr(self, "_horizon_attn_blocks", []) or []:
            blk.capture_visual_attn = False
            blk.last_visual_attn = None
        if not getattr(self, "_horizon_attn_blocks", None):
            return
        acc = getattr(self, "_horizon_attn", None)
        if acc is None:
            self._protocol_eval.extra["horizon_attention"] = {
                "n_rows": 0,
                "verdict": "not_measured",
            }
            return
        emb = getattr(self.model, "lead_time_embed", None)
        tau = None if emb is None else emb.detach().float().cpu().numpy()
        self._protocol_eval.extra["horizon_attention"] = acc.report(tau_embed=tau)

    def _run_cfg(self) -> Dict[str, Any]:
        """Provenance record written into the results manifest (see knowledge/running-ablations.md §0).

        runner.write_results stores this as manifest["config"] AND derives
        config_hash from it. It used to hold three keys — seed, model,
        quantile_levels — none of which vary across architectures, so s1, s2a,
        s2b and s2c all landed on the single hash 18d5735b73123686 and no result
        JSON could say which model produced it. Recording the full resolved
        architecture makes every ablation self-identifying from its own file.
        """
        from omegaconf import OmegaConf

        def _plain(value):
            """DictConfig/ListConfig → plain containers; anything else as-is."""
            if OmegaConf.is_config(value):
                return OmegaConf.to_container(value, resolve=True)
            return value

        hp = self.hparams
        return {
            "seed": getattr(hp, "seed", 42),
            "model": "mmtsfm",
            "quantile_levels": None,
            "chronos_core_cfg": _plain(hp.chronos_core_cfg),
            "vision_cfg": _plain(hp.vision_cfg),
            "horizon": getattr(hp, "horizon", None),
            "pretrained_model_name_or_path": getattr(
                hp, "pretrained_model_name_or_path", None
            ),
            "train_strategy": {
                key: getattr(hp, key, None)
                for key in (
                    "lr",
                    "weight_decay",
                    "warmup_steps",
                    "min_lr_ratio",
                    "freeze_chronos",
                    "n_unfreeze_encoder_blocks",
                    "backbone_lr_ratio",
                    "grassmann_warmup_steps",
                    "projector_warmup_steps",
                    "n_visual_unfreeze_layers",
                    "progressive_vision_unfreeze",
                )
            },
            "eval_control": getattr(hp, "eval_control", "none"),
            # True marks a control that is a no-op on this architecture: the Δ
            # in this file is an ARCHITECTURAL null, not an empirical one, and
            # nothing downstream may read it as "the model ignores motion".
            "eval_control_allow_inert": bool(
                getattr(hp, "eval_control_allow_inert", False)
            ),
            "compute_marginal_gain": getattr(hp, "compute_marginal_gain", False),
        }

    def on_test_epoch_end(self):
        if self._protocol_eval is None or not self.trainer.is_global_zero:
            return
        self._emit_horizon_attention()
        results = self._protocol_eval.finalize()
        overall = results.get("overall", {})
        for k in (
            "nmae",
            "nrmse",
            "skill_score",
            "crps",
            "nmae_vision_on",
            "nmae_vision_off",
            "delta_nmae",
            "nrmse_vision_on",
            "nrmse_vision_off",
            "delta_nrmse",
        ):
            if k in overall:
                self.log(f"test/{k}", float(overall[k]), rank_zero_only=True)
        try:
            run_cfg = self._run_cfg()
            path = self._protocol_eval.write(
                self.hparams.results_dir, self.hparams.results_tag, run_cfg
            )
            msg = (
                f"[protocol-eval] NMAE={overall.get('nmae'):.4f} "
                f"NRMSE={overall.get('nrmse'):.4f} "
                f"SS={overall.get('skill_score', float('nan')):.4f}"
            )
            if "delta_nmae" in overall:
                msg += (
                    f" dNMAE={overall.get('delta_nmae'):.4f} "
                    f"dNRMSE={overall.get('delta_nrmse'):.4f}"
                )
            msg += f" → {path}"
            print(msg, flush=True)
        except Exception as e:  # never fail the run on a results-write hiccup
            print(f"[protocol-eval] results write skipped: {e}", flush=True)

    # ------------------------------------------------------------------
    # Gradient norm logging (before clipping)
    # ------------------------------------------------------------------

    _GRAD_GROUPS: tuple[tuple[str, str], ...] = (
        ("vision_adapter", "model.cross_modal_adapter"),
        ("latent_summarizer", "model.latent_summarizer"),
        # s2d's entire visual path. It is the ONLY module with a non-zero LR
        # during Stage 0, so its grad norm is how you tell the projector warmup
        # actually ran rather than silently no-op'd.
        ("patch_projector", "model.patch_projector"),
        ("multimodal_embed", "model.multimodal_embed"),
        ("output_patch_embedding", "model.chronos.output_patch_embedding"),
        ("input_patch_embedding", "model.chronos.input_patch_embedding"),
        # Resolves to None on every arm that does not build it, which the prefix
        # walk below skips. Worth logging separately: it is the only fresh module on
        # s2c's future path besides the cross-attention itself.
        ("future_patch_embedding", "model.chronos.future_patch_embedding"),
        ("shared", "model.chronos.shared"),
    )

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm per param group before Lightning applies clipping.

        Per-group breakdown is the diagnostic signal we use to detect
        gradient starvation in the visual adapter / summarizer (Stage 2a).
        """
        # Accumulate squared norms as TENSORS: calling .item() per parameter
        # forces a host-device sync for every tensor on every step (hundreds of
        # CUDA syncs/step). One .item()/isfinite at the end is enough.
        zero = torch.zeros((), device=self.device)
        total_sq = zero
        per_group_sq: dict[str, torch.Tensor] = {
            name: zero for name, _ in self._GRAD_GROUPS
        }
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
            sq = p.grad.detach().pow(2).sum()
            total_sq = total_sq + sq
            group = group_by_param_id.get(id(p))
            if group is not None:
                per_group_sq[group] = per_group_sq[group] + sq

        grad_norm = total_sq.sqrt()
        grad_finite = bool(torch.isfinite(grad_norm))

        if self.trainer.global_step % 500 == 0:
            loss_val = (
                self._last_loss.item() if self._last_loss is not None else float("nan")
            )
            if self.trainer.is_global_zero:
                print(
                    f"[train] step={self.trainer.global_step} loss={loss_val:.4f} grad_norm={grad_norm:.4f}",
                    flush=True,
                )
            self.log(
                "train/loss_500",
                loss_val,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/grad_norm_500",
                grad_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )

        self.log(
            "train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False
        )
        for name, sq in per_group_sq.items():
            self.log(
                f"train/grad_norm/{name}",
                sq**0.5,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        # ---- NaN/Inf safety ------------------------------------------------
        # bf16 master + fp32 forward still occasionally emits non-finite grads
        # through quantile loss / instance_norm.inverse. Rather than letting
        # the NaN pollute AdamW's moment buffers (which would taint *all*
        # subsequent steps), zero every grad on this rank for this step.
        # Lightning's gradient_clip_val=1.0 then clips a no-op vector.
        if not grad_finite:
            self._nonfinite_grad_streak += 1
            self.log(
                "train/grad_skipped", 1.0, on_step=True, on_epoch=False, prog_bar=False
            )
            # Diagnostic: log ALL param groups that carry NaN/Inf grad.
            # Fires only on rank 0, at most once per epoch to avoid log spam.
            # Reading both the first offending param AND which high-level
            # groups are NaN lets us trace whether NaN enters via the visual
            # path (vision_adapter/latent_summarizer) or the encoder itself.
            if self.trainer.is_global_zero:
                epoch = self.trainer.current_epoch
                step = self.trainer.global_step
                if (
                    not hasattr(self, "_nan_logged_epoch")
                    or self._nan_logged_epoch != epoch
                ):
                    self._nan_logged_epoch = epoch
                    # Which high-level groups have NaN?
                    nan_groups = []
                    for gname, sq in per_group_sq.items():
                        if not bool(torch.isfinite(sq)):
                            nan_groups.append(gname)
                    # Also check unfrozen encoder blocks individually — log
                    # the specific param name inside the block so we can
                    # tell which sub-op (time_attn/group_attn/ffn) is the source.
                    enc_blocks = getattr(
                        getattr(getattr(self.model, "chronos", None), "encoder", None),
                        "block",
                        None,
                    )
                    # List EVERY offending param in a block, not just the first:
                    # knowing whether the blow-up is confined to the
                    # offset_weights/modality_pair_bias sink or has also reached
                    # W_plu/W_red is what distinguishes "the offset-logit branch
                    # overflowed" from "the whole layer is exploding".
                    nan_blocks = []
                    if enc_blocks is not None:
                        for bi, blk in enumerate(enc_blocks):
                            peak = 0.0
                            offenders = []
                            for pname, p in blk.named_parameters():
                                if p.grad is None:
                                    continue
                                finite = torch.isfinite(p.grad)
                                if not finite.all():
                                    n = (~finite).sum().item()
                                    offenders.append(
                                        f"{bi}:{pname}({n}/{p.grad.numel()})"
                                    )
                                elif p.grad.numel():
                                    peak = max(peak, p.grad.abs().max().item())
                            if offenders:
                                nan_blocks.extend(offenders)
                                nan_blocks.append(f"{bi}:max_finite_grad={peak:.3e}")
                    print(
                        f"[NaN-grad] epoch={epoch} step={step} "
                        f"streak={self._nonfinite_grad_streak} "
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

            # An unbroken streak means every step is a no-op: the run is burning
            # GPU-hours without moving a single weight. Fail loudly instead of
            # letting the walltime expire on a frozen model.
            limit = self.hparams.max_nonfinite_grad_steps
            if limit and self._nonfinite_grad_streak >= limit:
                raise RuntimeError(
                    f"Non-finite gradients on {self._nonfinite_grad_streak} "
                    f"consecutive optimizer steps (limit "
                    f"max_nonfinite_grad_steps={limit}). Every one of them was "
                    "zeroed, so no weight has been updated — training is frozen. "
                    "See the [NaN-grad] lines above for the offending parameters."
                )
        else:
            self._nonfinite_grad_streak = 0
            self.log(
                "train/grad_skipped", 0.0, on_step=True, on_epoch=False, prog_bar=False
            )

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
        "embed",  # modality_embed, segment_embed, token_type_embed, entity_embed
        "null_visual_token",
        "latent_queries",
        "offset_weights",
        "modality_pair_bias",
    )

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = self.hparams.weight_decay
        blr = lr * self.hparams.backbone_lr_ratio

        backbone_decay: list[torch.Tensor] = []
        backbone_nodecay: list[torch.Tensor] = []
        new_decay: list[torch.Tensor] = []
        new_nodecay: list[torch.Tensor] = []
        projector_decay: list[torch.Tensor] = []
        projector_nodecay: list[torch.Tensor] = []
        grassmann_decay: list[torch.Tensor] = []
        grassmann_nodecay: list[torch.Tensor] = []

        grassmann_kws = (
            "W_red",
            "W_plu",
            "W_gate",
            "offset_weights",
            "modality_pair_bias",
        )

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_no_decay = any(kw in name for kw in self._NO_DECAY_KWS)
            is_grassmann = any(kw in name for kw in grassmann_kws)
            is_backbone = name.startswith("model.chronos.")
            # s2d: split out so Stage 0 can hold everything else at lr 0 while
            # these train. Falls back into `new_*` on every other arm, where the
            # module does not exist and this list stays empty.
            is_projector = name.startswith("model.patch_projector.")

            if is_projector:
                (projector_nodecay if is_no_decay else projector_decay).append(p)
            elif is_grassmann:
                (grassmann_nodecay if is_no_decay else grassmann_decay).append(p)
            elif is_backbone:
                (backbone_nodecay if is_no_decay else backbone_decay).append(p)
            else:
                (new_nodecay if is_no_decay else new_decay).append(p)

        param_groups: list[dict] = []
        if backbone_decay:
            param_groups.append(
                {
                    "params": backbone_decay,
                    "lr": blr,
                    "weight_decay": wd,
                    "name": "backbone_decay",
                }
            )
        if backbone_nodecay:
            param_groups.append(
                {
                    "params": backbone_nodecay,
                    "lr": blr,
                    "weight_decay": 0.0,
                    "name": "backbone_nodecay",
                }
            )
        if new_decay:
            param_groups.append(
                {"params": new_decay, "lr": lr, "weight_decay": wd, "name": "new_decay"}
            )
        if new_nodecay:
            param_groups.append(
                {
                    "params": new_nodecay,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": "new_nodecay",
                }
            )
        if grassmann_decay:
            param_groups.append(
                {
                    "params": grassmann_decay,
                    "lr": lr,
                    "weight_decay": wd,
                    "name": "grassmann_decay",
                }
            )
        if grassmann_nodecay:
            param_groups.append(
                {
                    "params": grassmann_nodecay,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": "grassmann_nodecay",
                }
            )
        if projector_decay:
            param_groups.append(
                {
                    "params": projector_decay,
                    "lr": lr,
                    "weight_decay": wd,
                    "name": "projector_decay",
                }
            )
        if projector_nodecay:
            param_groups.append(
                {
                    "params": projector_nodecay,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": "projector_nodecay",
                }
            )
        if not param_groups:
            param_groups = [{"params": [], "lr": lr, "weight_decay": 0.0}]
        optimizer = AdamW(param_groups)

        total_steps = self._total_steps
        warmup = self.hparams.warmup_steps
        g_warmup = self.grassmann_warmup_steps
        p_warmup = self.projector_warmup_steps
        min_ratio = self.hparams.min_lr_ratio

        # Stage 0 (s2d) holds every non-projector group at lr 0 for p_warmup steps.
        # The rest of the schedule is then SHIFTED by p_warmup rather than
        # fast-forwarded, so the backbone still gets its full linear warmup once it
        # is released — otherwise a p_warmup >= warmup makes the backbone LR jump
        # 0 -> full in one step, which is exactly what warmup exists to prevent.
        # p_warmup = 0 (every other arm) makes this the identity.
        eff_total = max(1, total_steps - p_warmup)

        def lr_schedule(step: int) -> float:
            if step < p_warmup:
                return 0.0  # Stage 0: only the projector moves
            s = step - p_warmup
            if s < warmup:
                return s / max(1, warmup)
            progress = (s - warmup) / max(1, eff_total - warmup)
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        def grassmann_lr_schedule(step: int) -> float:
            if step < p_warmup:
                return 0.0
            s = step - p_warmup
            if s < g_warmup:
                return s / max(1, g_warmup)
            progress = (s - g_warmup) / max(1, eff_total - g_warmup)
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        def projector_lr_schedule(step: int) -> float:
            # Its own linear ramp over Stage 0, then it HOLDS at full while the
            # backbone runs the warmup it was denied during Stage 0, then rejoins
            # the shared cosine. Holding rather than calling lr_schedule keeps the
            # curve continuous: the projector has already warmed up, and dropping
            # it back to 0 at step p_warmup to re-warm alongside the backbone would
            # undo Stage 0's whole purpose.
            if step < p_warmup:
                return step / max(1, p_warmup)
            s = step - p_warmup
            if s < warmup:
                return 1.0
            return lr_schedule(step)

        lambdas = []
        for g in param_groups:
            name = g.get("name", "")
            if "projector" in name:
                lambdas.append(projector_lr_schedule)
            elif "grassmann" in name:
                lambdas.append(grassmann_lr_schedule)
            else:
                lambdas.append(lr_schedule)

        scheduler = LambdaLR(optimizer, lr_lambda=lambdas)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

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
