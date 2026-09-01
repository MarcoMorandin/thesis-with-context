# shellcheck shell=bash
# =============================================================================
# Shared Hydra command builder for one MMTSFM run.
# =============================================================================
# Sourced by BOTH launchers so there is exactly one definition of "what command
# does a stage run":
#
#   scripts/curriculum_stage.sbatch  — one run per SLURM job (1 GPU, 1/4 node)
#   scripts/ablation_pack.sbatch     — many runs per SLURM job (4 GPUs, 1 node)
#
# If these two ever drift, an ablation is compared against a curriculum arm that
# was trained with a different command and nothing in the results says so. That
# is the same class of defect as the config_hash collision fixed in
# knowledge/running-ablations.md §0, so it gets one implementation, not two.
#
# Contract: set the environment, call `build_stage_cmd`, read the global array
# CMD. Nothing here executes anything.

# --- arm identity ------------------------------------------------------------
# Kept byte-identical to slurm_curriculum.sh: the canonical (grassmann, seed 42)
# arm keeps the historical bare tag because mmtsfm_<stage>_<ds>.json is
# referenced by the A03 gate, ALL_RESULTS and the manuscript.
variant_slug() {
  case "$1" in
    vision_chronos2_grassmann)    echo grassmann;;
    vision_chronos2_timeselfattn) echo selfattn;;
    vision_chronos2)              echo base;;
    *)                            echo "${1#vision_chronos2_}";;
  esac
}

arm_suffix() {
  local model_cfg="$1" seed="$2"
  if [[ "$model_cfg" == "vision_chronos2_grassmann" && "$seed" == "42" ]]; then
    echo ""
  else
    echo "_$(variant_slug "$model_cfg")_s${seed}"
  fi
}

# --- V-JEPA latent cache -----------------------------------------------------
# Every failure is fatal: silently continuing costs either ~10x the compute
# (live re-encode) or, worse, a whole wave trained on the wrong imagery.
validate_vjepa_cache() {
  local cache="$1" data_dir="$2"
  # 1. Requested but absent. The old code dropped the override and logged a WARN,
  #    so a typo'd VJEPA_CACHE_VER re-encoded V-JEPA live for every batch and the
  #    only evidence was one line in a log read after the fact.
  if [[ ! -d "$cache" ]]; then
    echo "FATAL: no V-JEPA latent cache at '${cache}'." >&2
    echo "  Refusing to fall back to live encoding, which is ~10x the cost." >&2
    echo "  Check VJEPA_CACHE_ROOT / VJEPA_CACHE_VER, or extract it first." >&2
    return 1
  fi
  # 2. Present but with no provenance record. Unknown imagery is not a cache we
  #    are willing to train on: caches built before the extractor wrote
  #    _extract_meta.txt cannot be checked by (3) at all, so an obsolete one
  #    sitting at the default path would be consumed without a word.
  if [[ ! -f "${cache}/_extract_meta.txt" ]]; then
    echo "FATAL: '${cache}' has no _extract_meta.txt, so the imagery it was" >&2
    echo "  built from cannot be established. Re-extract it, or delete it so the" >&2
    echo "  wrong cache cannot be picked up by a default path." >&2
    return 1
  fi
  # 3. Present, with provenance that disagrees with DATA_DIR. Frames and latents
  #    must come from the same place or mask_visual/video_delta_t -- computed live
  #    from the frame grid -- describe a different frame set than the latents encode.
  local meta_dir
  meta_dir="$(sed -n 's/.*data_dir=\([^|]*\).*/\1/p' "${cache}/_extract_meta.txt")"
  if [[ -n "$meta_dir" && "$meta_dir" != "$data_dir" ]]; then
    echo "FATAL: latent cache was built from '${meta_dir}' but DATA_DIR is" >&2
    echo "  '${data_dir}'. mask_visual/video_delta_t would describe a different" >&2
    echo "  frame set than the latents encode. Point both at the same data." >&2
    return 1
  fi
  return 0
}

# --- the command -------------------------------------------------------------
# MODE=train (default) — fit then test, warm-start from PREV_CKPT, resume from
#                        STAGE_DIR/last.ckpt.
# MODE=eval            — score SCORE_CKPT only. `train: false` comes from the
#                        ablation config (A09/A10), not from here, so the config
#                        stays the single statement of what the run does.
build_stage_cmd() {
  : "${STAGE:?STAGE not set}" "${DS:?DS not set}" "${DCFG:?DCFG not set}" "${TAG:?TAG not set}"
  : "${MODEL_CFG:=vision_chronos2_grassmann}"
  : "${DATA_DIR:?DATA_DIR not set}" "${CKPT_DIR:?CKPT_DIR not set}" "${RESULTS_DIR:?RESULTS_DIR not set}"
  : "${MAX_EPOCHS:=50}" "${BATCH_SIZE:=8}" "${ACCUM:=2}" "${NUM_WORKERS:=8}" "${SEED:=42}"
  : "${MODE:=train}"

  # The submitter owns arm identity (MODEL_CFG + SEED) and exports the resolved
  # STAGE_DIR. Recomputing it here would drop the arm suffix and put every chain
  # of a wave back in one checkpoint directory — the collision the tag fix exists
  # to prevent. Fall back only for a hand-run stage with no submitter.
  STAGE_DIR="${STAGE_DIR:-${CKPT_DIR}/${DS}_${STAGE}}"
  mkdir -p "$STAGE_DIR" logs/slurm

  CMD=(
    python -m mmtsfm.train
    "+stage=${STAGE}" "model=${MODEL_CFG}" "data=${DCFG}"
    trainer=slurm trainer.devices=1 trainer.strategy=auto
    "seed=${SEED}" "trainer.max_epochs=${MAX_EPOCHS}"
    "trainer.default_root_dir=${STAGE_DIR}"
    # GroupSelfAttention is O((BS*num_entities*(1+covariates))^2) in activations —
    # keep the micro-batch small and recover the effective batch via accumulation.
    "data.batch_size=${BATCH_SIZE}" "trainer.accumulate_grad_batches=${ACCUM}"
    "data.num_workers=${NUM_WORKERS}"
    "model.results_dir=${RESULTS_DIR}" "model.results_tag=${TAG}"
    'hydra.run.dir=logs/experiments/runs/${now:%Y-%m-%d_%H-%M-%S}_'"${TAG}"
  )

  if [[ "$MODE" == "eval" ]]; then
    # Scoring an existing checkpoint. A missing file here would silently score a
    # randomly-initialised model and write a plausible-looking results JSON.
    : "${SCORE_CKPT:?SCORE_CKPT not set (MODE=eval)}"
    [[ -f "$SCORE_CKPT" ]] || { echo "FATAL: SCORE_CKPT not found: ${SCORE_CKPT}" >&2; return 1; }
    CMD+=("ckpt_path=${SCORE_CKPT}")
  else
    # Cross-stage: WEIGHTS-ONLY warm start from the previous stage's best.ckpt
    # (fresh optimizer + epoch 0 — the freezing/param-groups differ per stage).
    [[ -n "${PREV_CKPT:-}" && -f "${PREV_CKPT}" ]] && CMD+=("init_ckpt=${PREV_CKPT}")
    # Requeue-safety: FULL-STATE resume of THIS same stage from its own last.ckpt.
    # Takes precedence — a mid-stage requeue continues where it left off.
    [[ "${RESUME:-1}" == "1" && -f "${STAGE_DIR}/last.ckpt" ]] && CMD+=("ckpt_path=${STAGE_DIR}/last.ckpt")
  fi

  # DATA_DIR was required by the stage script but never reached Hydra, so
  # `data=ukpv`'s default won and every run read the same directory regardless.
  # Harmless while one dataset existed; wrong the moment a second did.
  CMD+=("data.data_dir=${DATA_DIR}")
  if [[ -n "${VJEPA_CACHE:-}" ]]; then
    validate_vjepa_cache "$VJEPA_CACHE" "$DATA_DIR" || return 1
    CMD+=("data.vjepa_cache_dir=${VJEPA_CACHE}")
  fi
  # Forced vision-off pass at test time -> dNMAE/dNRMSE incl. ramp, beside the
  # normal metrics. Without it the on/off ramp decomposition is simply absent.
  [[ "${MARGINAL_GAIN:-0}" == "1" ]] && CMD+=("model.compute_marginal_gain=true")
  # n_visual_context_steps per dataset (patch=16 alignment) — see slurm_curriculum.sh:nvis_for.
  [[ -n "${N_VIS:-}" ]] && CMD+=("model.vision_cfg.n_visual_context_steps=${N_VIS}")
  [[ -n "${SP_REF:-}" ]] && CMD+=("model.sp_reference_path=${SP_REF}")
  [[ -n "${TRAIN_STRIDE:-}" ]] && CMD+=("+data.train_stride=${TRAIN_STRIDE}")
  [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]] && CMD+=("+trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}")
  # Appended LAST so `+ablation=<ID>` beats both the model and the stage config.
  # shellcheck disable=SC2206  -- intentional word-split: a list of overrides
  [[ -n "${EXTRA_OVERRIDES:-}" ]] && CMD+=(${EXTRA_OVERRIDES})
  return 0
}
