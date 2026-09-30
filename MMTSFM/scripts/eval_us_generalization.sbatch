#!/bin/bash
#SBATCH --job-name=mmtsfm-eval-us
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# =============================================================================
# EVALUATION OF US ZERO-SHOT GENERALIZATION (goes_pvdaq)
# =============================================================================
# Evaluates models trained on UK PV (MMTSFM, chronos2_ft, itransformer, timevlm)
# zero-shot on US goes_pvdaq (resampled to 30-min cadence) without retraining.
#
# Usage: [SEEDS="42 43 44"] [TEST_SPLIT="all"] [ONLY="itransformer chronos2_ft timevlm s2d"] \
#        sbatch scripts/eval_us_generalization.sbatch
#        DRY_RUN=1 bash scripts/eval_us_generalization.sbatch
# =============================================================================
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -d src/mmtsfm ]] || { echo "FATAL: submit from MMTSFM/"; exit 1; }
[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}" HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd .. && pwd)"
mkdir -p logs/slurm

TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data_v2}"; CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints/curriculum}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/baselines/results}"
TEST_SPLIT="${TEST_SPLIT:-all}"; SPLIT_SFX=""; [[ "$TEST_SPLIT" != "test" ]] && SPLIT_SFX="_${TEST_SPLIT}"
SP_REF="${SP_REF:-${RESULTS_DIR}/smart_persistence_s2_goespvdaq${SPLIT_SFX}.json}"
VJEPA_CACHE_ROOT="${VJEPA_CACHE_ROOT:-/leonardo_work/IscrC_MTSFM/vjepa_cache}"; VJEPA_ARCH="${VJEPA_ARCH:-vit_large}"
VJEPA_CACHE_VER="${VJEPA_CACHE_VER:-${VJEPA_ARCH}_f8_s224_nonhrv_sp45}"
vjepa_cache="${VJEPA_CACHE:-${VJEPA_CACHE_ROOT}/goes_pvdaq/${VJEPA_CACHE_VER}}"

SEEDS="${SEEDS:-42 43 44}"; ONLY="${ONLY:-}"; SKIP_DONE="${SKIP_DONE:-1}"; DRY_RUN="${DRY_RUN:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"; TRAINER_CFG="${TRAINER_CFG:-slurm}"; VISUAL_SCALE="${VISUAL_SCALE:-0.5}"
EVS_KEEP_S2D="${EVS_KEEP_S2D:-14}"; EVS_KEEP_S2E="${EVS_KEEP_S2E:-20}"; EVS_KEEP_A46B="${EVS_KEEP_A46B:-70}"

is_selected() {
  [[ -z "$ONLY" ]] && return 0
  for i in $ONLY; do
    [[ "$i" == "$1" || ("$1" == "s1" && "$i" =~ ^chronos2_) || ("$1" == "timevlm" && "$i" == "time_vlm") ]] && return 0
  done
  return 1
}

# 1. US Smart Persistence Reference (30-min cadence)
if [[ ! -f "$SP_REF" ]]; then
  echo ">>> [SP] Generating US Smart Persistence baseline at 30-min cadence (${TEST_SPLIT})..."
  [[ "$DRY_RUN" != "1" ]] && (cd "${REPO_ROOT}/baselines" && uv run python run_eval.py \
    --model smart_persistence --eval-datasets goes_pvdaq --split "${TEST_SPLIT}" \
    --tag "s2_goespvdaq${SPLIT_SFX}" --resample-cadence-min 30 --eval-stride 12 \
    --out "${RESULTS_DIR}" --data "${DATA_DIR}/dataset_all.parquet")
fi

# 2. iTransformer (NeuralForecast, trained on UK PV)
eval_itransformer() {
  local seed="$1" tag="itransformer_nf_s2_goespvdaq${SPLIT_SFX}_seed${seed}"
  [[ "$SKIP_DONE" == "1" && -f "${RESULTS_DIR}/${tag}.json" ]] && { echo ">>> [SKIP] ${tag} exists"; return 0; }
  local dir="${CKPT_ITRANSFORMER:-${TEAM_SCRATCH}/checkpoints/itransformer_nf/seed${seed}}"
  local ckpt="${dir}/last.ckpt"
  [[ ! -f "$ckpt" ]] && { local c=( "${dir}"/epoch*.ckpt ); ckpt="${c[-1]:-${ckpt}}"; }
  echo ">>> [EVAL] model=itransformer seed=${seed} ckpt=${ckpt}"
  [[ "$DRY_RUN" == "1" ]] && { echo "    (cd ${REPO_ROOT}/baselines && uv run --group nf python tier2/train_itransformer_nf.py --eval-only --split ${TEST_SPLIT} --ckpt-path ${ckpt} ...)"; return 0; }
  [[ ! -f "$ckpt" ]] && { echo "ERROR: itransformer ckpt not found: ${ckpt}" >&2; return 1; }
  (cd "${REPO_ROOT}/baselines" && uv run --group nf python tier2/train_itransformer_nf.py \
    --eval-only --split "$TEST_SPLIT" --ckpt-path "$ckpt" --dataset goes_pvdaq --resample-cadence-min 30 \
    --data-dir "$DATA_DIR" --out "$RESULTS_DIR" --tag "$tag" --sp-reference "$SP_REF")
}

# 3. Time-VLM (Tier 5, trained on UK PV)
eval_timevlm() {
  local tag="time_vlm_s2_goespvdaq${SPLIT_SFX}"
  [[ "$SKIP_DONE" == "1" && -f "${RESULTS_DIR}/${tag}.json" ]] && { echo ">>> [SKIP] ${tag} exists"; return 0; }
  echo ">>> [EVAL] model=timevlm tag=${tag}"
  DATA="${DATA_DIR}/dataset_all.parquet" RESULTS_DIR="$RESULTS_DIR" SP_REF="$SP_REF" \
    TEST_SPLIT="$TEST_SPLIT" TAG="s2_goespvdaq${SPLIT_SFX}" DRY_RUN="$DRY_RUN" \
    bash "${REPO_ROOT}/baselines/scripts/eval_time_vlm_us.sh"
}

# 4. MMTSFM Curriculum & Controls (s1=chronos2_ft, s2a, s2d, s2e, A46b)
MODEL_SPECS=(
  "s1|s1|vision_chronos2_timeselfattn|uk_pv_s1_selfattn_s{seed}/best.ckpt||${BATCH_SIZE:-8}"
  "s2a|s2a|vision_chronos2_timeselfattn|uk_pv_s2a_selfattn_s{seed}/best.ckpt|model.compute_marginal_gain=true model.vision_cfg.visual_scale=${VISUAL_SCALE}|${BATCH_SIZE:-8}"
  "s2d|s2d|vision_chronos2_s2d|uk_pv_s2d_s2d_s{seed}/best.ckpt|model.compute_marginal_gain=true model.vision_cfg.visual_scale=${VISUAL_SCALE} model.vision_cfg.visual_evs_keep=${EVS_KEEP_S2D}|${BATCH_SIZE:-8}"
  "s2e|s2e|vision_chronos2_s2e|uk_pv_s2e_s2e_s{seed}/best.ckpt|model.compute_marginal_gain=true model.vision_cfg.visual_scale=${VISUAL_SCALE} model.vision_cfg.visual_evs_keep=${EVS_KEEP_S2E}|${BATCH_SIZE_S2E:-4}"
  "A46b|s2e|vision_chronos2_s2e|uk_pv_s2e_s2e_s{seed}_A46b/best.ckpt|+ablation=A46b model.compute_marginal_gain=true model.vision_cfg.visual_scale=${VISUAL_SCALE} model.vision_cfg.visual_evs_keep=${EVS_KEEP_A46B}|${BATCH_SIZE_A46B:-2}"
)

resolve_ckpt() {
  local model="$1" seed="$2" subpath="$3"
  local env_var="CKPT_$(echo "$model" | tr '[:lower:]' '[:upper:]')"
  [[ -n "${!env_var:-}" ]] && { echo "${!env_var}"; return; }
  local target="${CKPT_DIR}/${subpath//\{seed\}/$seed}"
  [[ ! -f "$target" && "$model" == "s1" && -f "${CKPT_DIR}/uk_pv_s1/best.ckpt" ]] && { echo "${CKPT_DIR}/uk_pv_s1/best.ckpt"; return; }
  echo "$target"
}

tag_for_model() {
  local model="$1" seed="$2" ckpt="$3"
  if [[ "$model" == "A46b" ]]; then echo "mmtsfm_A46b_s2e_goespvdaq${SPLIT_SFX}_s${seed}"
  elif [[ "$model" == "s1" && "$ckpt" =~ uk_pv_s1/best.ckpt$ ]]; then echo "mmtsfm_s1_goespvdaq_grassmann${SPLIT_SFX}_s${seed}"
  elif [[ "$model" == "s1" || "$model" == "s2a" ]]; then echo "mmtsfm_${model}_goespvdaq_selfattn${SPLIT_SFX}_s${seed}"
  else echo "mmtsfm_${model}_goespvdaq_${model}${SPLIT_SFX}_s${seed}"; fi
}

for seed in $SEEDS; do
  is_selected "itransformer" && eval_itransformer "$seed" || true

  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r name stage cfg default_subpath extra_args bs <<< "$spec"
    is_selected "$name" || continue
    ckpt="$(resolve_ckpt "$name" "$seed" "$default_subpath")"
    tag="$(tag_for_model "$name" "$seed" "$ckpt")"
    [[ "$name" == "s1" && "$ckpt" =~ uk_pv_s1/best.ckpt$ ]] && cfg="vision_chronos2_grassmann"
    [[ "$SKIP_DONE" == "1" && -f "${RESULTS_DIR}/${tag}.json" ]] && { echo ">>> [SKIP] ${tag} exists"; continue; }
    [[ ! -f "$ckpt" && "$DRY_RUN" != "1" ]] && { echo "WARN: ckpt not found: $ckpt"; continue; }

    declare -a CMD=(
      python -m mmtsfm.train train=false test=true "+stage=${stage}" "model=${cfg}" data=goespvdaq
      "data.data_dir=${DATA_DIR}" "data.batch_size=${bs}" "data.num_workers=${NUM_WORKERS}"
      "data.test_split=${TEST_SPLIT}"
      "trainer=${TRAINER_CFG}" trainer.devices=1 trainer.strategy=auto
      "seed=${seed}" "ckpt_path=${ckpt}" "model.results_dir=${RESULTS_DIR}" "model.results_tag=${tag}"
      'hydra.run.dir=logs/experiments/runs/${now:%Y-%m-%d_%H-%M-%S}_'"${tag}"
    )
    [[ -f "$SP_REF" ]] && CMD+=("model.sp_reference_path=${SP_REF}")
    [[ -d "$vjepa_cache" ]] && CMD+=("data.vjepa_cache_dir=${vjepa_cache}")
    [[ -n "$extra_args" ]] && { IFS=' ' read -r -a extras <<< "$extra_args"; CMD+=("${extras[@]}"); }

    echo ">>> [EVAL] model=${name} seed=${seed} tag=${tag}"
    [[ "$DRY_RUN" == "1" ]] && { echo "    uv run ${CMD[*]}"; continue; }
    uv run "${CMD[@]}"
  done
done

is_selected "timevlm" && eval_timevlm || true
