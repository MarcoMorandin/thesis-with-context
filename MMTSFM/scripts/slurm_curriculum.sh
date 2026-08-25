#!/bin/bash
# =============================================================================
# MMTSFM 4-STAGE CURRICULUM SUBMITTER  —  run on a Leonardo LOGIN node.
# =============================================================================
# Chains Stage 1 -> 2a -> 2b -> 3 as four dependency-linked SLURM jobs per
# dataset (afterok), threading each stage's best.ckpt into the next via
# +ckpt_path. Every stage runs its own test pass and writes a tagged protocol
# JSON (mmtsfm_<stage>_<ds>) into RESULTS_DIR so aggregate_all.py shows the
# per-stage progression next to the baselines.
#
# Prereq: scripts/precache_login.sh has staged the uv env, Chronos-2 + V-JEPA
# weights, the dataset, and (for S2a/S2b/S3) the pre-extracted V-JEPA latent
# cache. Submit from MMTSFM/.
#
#   bash scripts/slurm_curriculum.sh                       # uk_pv (protocol-compliant default)
#   START_STAGE=s2a bash scripts/slurm_curriculum.sh       # skip finished S1, warm-start S2a from S1/best.ckpt
#   SMOKE=1 bash scripts/slurm_curriculum.sh               # local CPU dry-run, no sbatch
# =============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/.."
[[ -f pyproject.toml && -d src/mmtsfm ]] || { echo "FATAL: run from MMTSFM/"; exit 1; }
REPO_ROOT="$(cd .. && pwd)"
mkdir -p logs/slurm

# ---- config (override via env) ---------------------------------------------
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints/curriculum}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/baselines/results}"
VJEPA_CACHE_ROOT="${VJEPA_CACHE_ROOT:-/leonardo_work/IscrC_MTSFM/vjepa_cache}"
VJEPA_ARCH="${VJEPA_ARCH:-vit_large}"
# Cache of record: v2 non-HRV frames, 8 frames over 6 h => 45-min spacing. The
# bare `${VJEPA_ARCH}_f8_s224` name still exists on /leonardo_work and holds the
# OBSOLETE v1 HRV latents, so defaulting to it would silently train a wave on
# grayscale daylight-only imagery. Verified 2026-08-25: this is the cache the
# s2b run of record consumed.
VJEPA_CACHE_VER="${VJEPA_CACHE_VER:-${VJEPA_ARCH}_f8_s224_nonhrv_sp45}"
MODEL_CFG="${MODEL_CFG:-vision_chronos2_grassmann}"
# uk_pv is the primary cross-plant benchmark and the only protocol-compliant
# curriculum target. goes_pvdaq is intentionally NOT run here: knowledge/protocol.md
# §2/§4.1 require leave-one-plant-out for it (test share = 1 plant), which this
# fixed-split runner does not implement. Pass DATASETS=goes_pvdaq only once a
# LOPO harness exists.
DATASETS="${DATASETS:-uk_pv}"
SEED="${SEED:-42}"
# Empty → use the per-stage ST_BATCH defaults below; set to force one batch size
# across all stages (e.g. BATCH_SIZE=2 for a very tight GPU).
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_STRIDE="${TRAIN_STRIDE:-12}"
ACCOUNT="${ACCOUNT:-IscrC_MTSFM}"
PARTITION="${PARTITION:-boost_usr_prod}"
# Job status via Slurm email notifications (CINECA-recommended) instead of
# polling squeue. Set MAIL_USER to receive BEGIN/END/FAIL mails; leave empty to
# disable. Never poll the controller in a loop (no `watch -n N squeue`).
MAIL_USER="${MAIL_USER:-}"
MAIL_TYPE="${MAIL_TYPE:-END,FAIL}"

# Per-stage max epochs + walltime (S1/S3 heavier than the two alignment stages).
#
# Walltimes are the boost_usr_prod cap (24 h) for every vision stage, because a
# TIMEOUT is far more expensive here than an over-long reservation: train.py only
# exports the stable best.ckpt and runs the test pass AFTER fit() returns, and the
# stages are chained with afterok — so a stage killed at walltime leaves no
# best.ckpt, writes no protocol JSON, and takes every downstream stage down with
# it (DependencyNeverSatisfied). Unused walltime costs nothing.
#
# Measured on uk_pv (batch 4, accum 4, train_stride 12, 1x A100):
#   s2a  69 min/epoch, EarlyStopping fired at epoch 13  → ~16 h
#   s2b  77 min/epoch (interleaved fusion is ~12% dearer than late)
# 20 epochs of s2b is ~26 h, i.e. past the cap, so that stage can still need one
# resume if EarlyStopping does not fire first. s3 (50 epochs, progressive vision
# unfreeze) will certainly need several. Both are safe: ModelCheckpoint writes
# last.ckpt every epoch and curriculum_stage.sbatch resumes from it by default
# (RESUME=1) — but the afterok chain must be re-submitted after any TIMEOUT.
declare -A ST_EPOCHS=( [s1]="${S1_EPOCHS:-40}" [s2a]="${S2A_EPOCHS:-20}" [s2b]="${S2B_EPOCHS:-20}" [s3]="${S3_EPOCHS:-50}" )
declare -A ST_TIME=(   [s1]="${S1_TIME:-20:00:00}" [s2a]="${S2A_TIME:-24:00:00}" [s2b]="${S2B_TIME:-24:00:00}" [s3]="${S3_TIME:-24:00:00}" )
# Per-stage micro-batch + grad accumulation (effective batch = batch*accum ≈ 16).
# GroupSelfAttention flattens BS*num_entities*(1+covariates) rows into one attention
# axis (uk_pv: 16*4*15 = 960 rows) → O(rows²) activations OOM a 64 GB A100 at
# batch 16. Small micro-batches keep it on-GPU; accumulation preserves the effective
# batch. Vision stages (s2a+) add V-JEPA + visual rows, so they go smaller. Set
# BATCH_SIZE to override all stages; drop to 2 (accum 8) if a stage still OOMs.
declare -A ST_BATCH=( [s1]="${S1_BATCH:-8}" [s2a]="${S2A_BATCH:-4}" [s2b]="${S2B_BATCH:-4}" [s3]="${S3_BATCH:-4}" )
declare -A ST_ACCUM=( [s1]="${S1_ACCUM:-2}" [s2a]="${S2A_ACCUM:-4}" [s2b]="${S2B_ACCUM:-4}" [s3]="${S3_ACCUM:-4}" )
STAGES=(s1 s2a s2b s3)

# START_STAGE: resume the curriculum from a later stage (e.g. after S1 is done).
# RUN_STAGES = STAGES from START_STAGE onward; STAGE_BEFORE_START = the stage just
# before it, whose best.ckpt warm-starts START_STAGE (no in-submission dependency).
# END_STAGE: stop the chain early. The curriculum terminates at s2b whenever s3 is
# a reported regression rather than a stage to improve — without this knob every
# arm pays ~24 h for a stage already known to hurt cross-plant generalization.
START_STAGE="${START_STAGE:-s1}"
END_STAGE="${END_STAGE:-s3}"
RUN_STAGES=(); STAGE_BEFORE_START=""; _seen=0; _past_end=0
for _st in "${STAGES[@]}"; do
  [[ "$_st" == "$START_STAGE" ]] && _seen=1
  if [[ $_seen -eq 1 && $_past_end -eq 0 ]]; then RUN_STAGES+=("$_st")
  elif [[ $_seen -eq 0 ]]; then STAGE_BEFORE_START="$_st"; fi
  [[ "$_st" == "$END_STAGE" ]] && _past_end=1
done
[[ ${#RUN_STAGES[@]} -gt 0 ]] || { echo "FATAL: START_STAGE='${START_STAGE}' not in [${STAGES[*]}]"; exit 1; }
printf '%s\n' "${STAGES[@]}" | grep -qx "$END_STAGE" || {
  echo "FATAL: END_STAGE='${END_STAGE}' not in [${STAGES[*]}]"; exit 1; }

# ---- arm identity -----------------------------------------------------------
# Wave runs submit several chains that differ ONLY in MODEL_CFG and SEED. Both
# must reach the run tag AND the checkpoint dir, or the chains share a results
# JSON and a checkpoint dir and silently clobber each other mid-run.
#
# The canonical (grassmann, seed 42) arm keeps the historical bare tag, because
# `mmtsfm_<stage>_<ds>.json` is referenced by the A03 gate, ALL_RESULTS and the
# manuscript; every other combination gets an explicit suffix.
variant_slug() {
  case "$1" in
    vision_chronos2_grassmann)    echo grassmann;;
    vision_chronos2_timeselfattn) echo selfattn;;
    vision_chronos2)              echo base;;
    *)                            echo "${1#vision_chronos2_}";;
  esac
}
ARM_SUFFIX=""
if [[ "$MODEL_CFG" != "vision_chronos2_grassmann" || "$SEED" != "42" ]]; then
  ARM_SUFFIX="_$(variant_slug "$MODEL_CFG")_s${SEED}"
fi

DRY_RUN="${DRY_RUN:-0}"

dcfg_for() { case "$1" in uk_pv) echo ukpv;; goes_pvdaq) echo goespvdaq;; *) echo "$1";; esac; }
# n_visual_context_steps per dataset for patch=16: the 6h visual window spans
# ceil(window_steps/16) TS patches — uk_pv (30-min) = ceil(12/16)=1, goes_pvdaq
# (15-min) = ceil(24/16)=2. Override globally with N_VIS for the vision ablation.
nvis_for() { [[ -n "${N_VIS:-}" ]] && { echo "$N_VIS"; return; }; case "$1" in uk_pv) echo 1;; goes_pvdaq) echo 2;; *) echo 1;; esac; }
sp_ref_for() {
  local f="${RESULTS_DIR}/smart_persistence_s2_$(dcfg_for "$1").json"
  [[ -f "$f" ]] && echo "$f" || echo ""
}

# ---------------------------------------------------------------------------
# SMOKE: run the whole chain locally (CPU, synthetic data, 1 step/stage) with
# NO sbatch — validates config composition + cross-stage ckpt threading.
# ---------------------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo ">>> SMOKE: local curriculum dry-run (synthetic, CPU)"
  # Full chain (s1 s2a s2b s3) needs cached V-JEPA weights for the vision stages;
  # on a dev box without them, run SMOKE_STAGES="s1" to validate the numeric path
  # + best.ckpt threading. Vision stages smoke on a login node (weights present).
  read -ra SMK_STAGES <<< "${SMOKE_STAGES:-s1 s2a s2b s3}"
  SMOKE_ROOT="${SMOKE_ROOT:-/tmp/mmtsfm_smoke}"; rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE_ROOT"
  prev=""
  for st in "${SMK_STAGES[@]}"; do
    sd="${SMOKE_ROOT}/${st}"
    declare -a C=(
      python -m mmtsfm.train "+stage=${st}" "model=${MODEL_CFG}" data=smoke
      trainer=default trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
      trainer.precision=32
      +trainer.limit_train_batches=2 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1
      logger=csv "+trainer.default_root_dir=${sd}"
      "model.results_dir=${SMOKE_ROOT}/results" "model.results_tag=smoke_${st}"
      "data.batch_size=2" "data.num_workers=0"
      'hydra.run.dir='"${sd}"'/hydra'
    )
    [[ -n "$prev" && -f "$prev" ]] && C+=("init_ckpt=${prev}")
    echo ">>> SMOKE stage ${st}  warm-start=${prev:-<none>}"
    uv run "${C[@]}" || { echo "SMOKE FAILED at ${st}"; exit 1; }
    prev="${sd}/best.ckpt"
  done
  echo ">>> SMOKE OK — stages [${SMK_STAGES[*]}] ran and threaded best.ckpt via init_ckpt"
  exit 0
fi

# ---------------------------------------------------------------------------
# Cluster: submit 4 dependency-linked jobs per dataset.
# ---------------------------------------------------------------------------
[[ "$DRY_RUN" == "1" ]] || command -v sbatch >/dev/null || {
  echo "FATAL: sbatch not found (run on a Leonardo login node, or SMOKE=1/DRY_RUN=1)"; exit 1; }

for ds in $DATASETS; do
  dcfg="$(dcfg_for "$ds")"
  nvis="$(nvis_for "$ds")"
  vjepa_cache="${VJEPA_CACHE_ROOT}/${ds}/${VJEPA_CACHE_VER}"
  sp_ref="$(sp_ref_for "$ds")"
  echo "=== dataset ${ds} (data=${dcfg}) ==="
  # The vision stages read a pre-extracted V-JEPA latent cache. If it is absent
  # the worker omits data.vjepa_cache_dir and V-JEPA runs LIVE per batch on the
  # GPU (correct, but re-encodes every step). Pre-extract once with
  # scripts/extract_video_embeddings.py (as run_all_mmtsfm.sh's PREEXTRACT does),
  # writing to VJEPA_CACHE_ROOT/<ds>/<arch>_f8_s224, and keep TRAIN_STRIDE aligned.
  if [[ ! -d "$vjepa_cache" ]]; then
    echo "  ! WARN: no V-JEPA latent cache at ${vjepa_cache} — s2a/s2b/s3 will encode live (slow)."
  fi
  # Seed the warm-start ckpt from the stage before START_STAGE (already trained in
  # a previous submission). prev_jid stays empty → no in-submission dependency.
  prev_jid=""; prev_ckpt=""
  if [[ -n "$STAGE_BEFORE_START" ]]; then
    prev_ckpt="${CKPT_DIR}/${ds}_${STAGE_BEFORE_START}${ARM_SUFFIX}/best.ckpt"
    [[ -f "$prev_ckpt" ]] || echo "  ! WARN: warm-start ckpt missing: ${prev_ckpt} — ${START_STAGE} will start from scratch"
  fi
  for st in "${RUN_STAGES[@]}"; do
    tag="mmtsfm_${st}_${dcfg}${ARM_SUFFIX}"
    stage_dir="${CKPT_DIR}/${ds}_${st}${ARM_SUFFIX}"
    bs="${BATCH_SIZE:-${ST_BATCH[$st]}}"    # global override else per-stage default
    accum="${ST_ACCUM[$st]}"
    declare -a DEP=(); [[ -n "$prev_jid" ]] && DEP=(--dependency="afterok:${prev_jid}")
    exports="ALL,STAGE=${st},DS=${ds},DCFG=${dcfg},MODEL_CFG=${MODEL_CFG},TAG=${tag}"
    exports+=",STAGE_DIR=${stage_dir}"
    exports+=",DATA_DIR=${DATA_DIR},CKPT_DIR=${CKPT_DIR},RESULTS_DIR=${RESULTS_DIR}"
    exports+=",MAX_EPOCHS=${ST_EPOCHS[$st]},BATCH_SIZE=${bs},ACCUM=${accum},NUM_WORKERS=${NUM_WORKERS}"
    exports+=",SEED=${SEED},TRAIN_STRIDE=${TRAIN_STRIDE},N_VIS=${nvis}"
    [[ -n "$prev_ckpt" ]] && exports+=",PREV_CKPT=${prev_ckpt}"
    [[ -n "$sp_ref" ]]   && exports+=",SP_REF=${sp_ref}"
    # S1 skips vision (emit_vision=false); no cache needed there.
    [[ "$st" != "s1" ]]  && exports+=",VJEPA_CACHE=${vjepa_cache}"
    # Forced vision-off pass at test time, reported as dNMAE/dNRMSE (incl. ramp)
    # beside the normal metrics. Wave runs need it on EVERY arm.
    [[ "${MARGINAL_GAIN:-0}" == "1" ]] && exports+=",MARGINAL_GAIN=1"

    declare -a MAIL=(); [[ -n "$MAIL_USER" ]] && MAIL=(--mail-type="${MAIL_TYPE}" --mail-user="${MAIL_USER}")
    if [[ "$DRY_RUN" == "1" ]]; then
      # Print the plan instead of submitting. Lets a wave be inspected for tag /
      # checkpoint-dir collisions before five afterok chains hit the queue, and
      # is the seam the wave-safety tests assert against.
      echo "  --- would submit ---"
      echo "    STAGE=${st}"
      echo "    TAG=${tag}"
      echo "    STAGE_DIR=${stage_dir}"
      echo "    MODEL_CFG=${MODEL_CFG}"
      echo "    SEED=${SEED}"
      echo "    EXPORTS=${exports}"
      prev_jid="dry"; prev_ckpt="${stage_dir}/best.ckpt"
      continue
    fi
    jid="$(sbatch --parsable "${DEP[@]}" "${MAIL[@]}" \
      --job-name="${tag}" --account="${ACCOUNT}" --partition="${PARTITION}" \
      --time="${ST_TIME[$st]}" --export="${exports}" \
      scripts/curriculum_stage.sbatch)" || { echo "sbatch failed for ${tag}"; exit 1; }
    echo "  ${st}: job ${jid}  ${DEP[*]:-<no dep>}"
    prev_jid="$jid"; prev_ckpt="${stage_dir}/best.ckpt"
  done
done
echo ">>> submitted. Job status arrives by email when MAIL_USER is set"
echo "    (BEGIN/END/FAIL). Do NOT poll the controller — no 'watch -n N squeue'."
echo "    One-off status check is fine: squeue -u \$USER   Results: ${RESULTS_DIR}/mmtsfm_*.json"
