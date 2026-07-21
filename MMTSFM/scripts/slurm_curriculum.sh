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
#   bash scripts/slurm_curriculum.sh                       # uk_pv + goes_pvdaq
#   DATASETS="uk_pv" bash scripts/slurm_curriculum.sh      # one dataset
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
VJEPA_CACHE_VER="${VJEPA_CACHE_VER:-${VJEPA_ARCH}_f8_s224}"
MODEL_CFG="${MODEL_CFG:-vision_chronos2_grassmann}"
DATASETS="${DATASETS:-uk_pv goes_pvdaq}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
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
declare -A ST_EPOCHS=( [s1]="${S1_EPOCHS:-40}" [s2a]="${S2A_EPOCHS:-20}" [s2b]="${S2B_EPOCHS:-20}" [s3]="${S3_EPOCHS:-50}" )
declare -A ST_TIME=(   [s1]="${S1_TIME:-12:00:00}" [s2a]="${S2A_TIME:-08:00:00}" [s2b]="${S2B_TIME:-08:00:00}" [s3]="${S3_TIME:-20:00:00}" )
STAGES=(s1 s2a s2b s3)

dcfg_for() { case "$1" in uk_pv) echo ukpv;; goes_pvdaq) echo goespvdaq;; *) echo "$1";; esac; }
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
command -v sbatch >/dev/null || { echo "FATAL: sbatch not found (run on a Leonardo login node, or SMOKE=1)"; exit 1; }

for ds in $DATASETS; do
  dcfg="$(dcfg_for "$ds")"
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
  prev_jid=""; prev_ckpt=""
  for st in "${STAGES[@]}"; do
    tag="mmtsfm_${st}_${dcfg}"
    stage_dir="${CKPT_DIR}/${ds}_${st}"
    declare -a DEP=(); [[ -n "$prev_jid" ]] && DEP=(--dependency="afterok:${prev_jid}")
    exports="ALL,STAGE=${st},DS=${ds},DCFG=${dcfg},MODEL_CFG=${MODEL_CFG},TAG=${tag}"
    exports+=",DATA_DIR=${DATA_DIR},CKPT_DIR=${CKPT_DIR},RESULTS_DIR=${RESULTS_DIR}"
    exports+=",MAX_EPOCHS=${ST_EPOCHS[$st]},BATCH_SIZE=${BATCH_SIZE},NUM_WORKERS=${NUM_WORKERS}"
    exports+=",SEED=${SEED},TRAIN_STRIDE=${TRAIN_STRIDE}"
    [[ -n "$prev_ckpt" ]] && exports+=",PREV_CKPT=${prev_ckpt}"
    [[ -n "$sp_ref" ]]   && exports+=",SP_REF=${sp_ref}"
    # S1 skips vision (emit_vision=false); no cache needed there.
    [[ "$st" != "s1" ]]  && exports+=",VJEPA_CACHE=${vjepa_cache}"

    declare -a MAIL=(); [[ -n "$MAIL_USER" ]] && MAIL=(--mail-type="${MAIL_TYPE}" --mail-user="${MAIL_USER}")
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
