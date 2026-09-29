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
# EVALUATION OF REGION GENERALIZATION (US SET: goes_pvdaq)
# =============================================================================
# Evaluates trained UK PV checkpoints (s1, s2a, s2d, s2e, and A46b) zero-shot
# on the US goes_pvdaq dataset (15-min cadence, 24-step horizon) without fitting.
#
# Usage:
#   sbatch scripts/eval_us_generalization.sbatch
#   SEEDS="42 43 44" sbatch scripts/eval_us_generalization.sbatch
#   ONLY="s1 s2d" sbatch scripts/eval_us_generalization.sbatch
#   DRY_RUN=1 bash scripts/eval_us_generalization.sbatch
# =============================================================================
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -d src/mmtsfm ]] || { echo "FATAL: submit from MMTSFM/"; exit 1; }

[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}" HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd .. && pwd)"
mkdir -p logs/slurm

TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data_v2}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints/curriculum}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/baselines/results}"
VJEPA_CACHE_ROOT="${VJEPA_CACHE_ROOT:-/leonardo_work/IscrC_MTSFM/vjepa_cache}"
VJEPA_ARCH="${VJEPA_ARCH:-vit_large}"
VJEPA_CACHE_VER="${VJEPA_CACHE_VER:-${VJEPA_ARCH}_f8_s224_nonhrv_sp45}"

SEEDS="${SEEDS:-42}"
ONLY="${ONLY:-}"
SKIP_DONE="${SKIP_DONE:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
DRY_RUN="${DRY_RUN:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAINER_CFG="${TRAINER_CFG:-slurm}"

vjepa_cache="${VJEPA_CACHE:-${VJEPA_CACHE_ROOT}/goes_pvdaq/${VJEPA_CACHE_VER}}"

# Specs: model_name | stage | model_cfg | default_ckpt_subpath | extra_hydra_args | batch_size
MODEL_SPECS=(
  "s1|s1|vision_chronos2_timeselfattn|uk_pv_s1_selfattn_s{seed}/best.ckpt||${BATCH_SIZE:-8}"
  "s2a|s2a|vision_chronos2_timeselfattn|uk_pv_s2a_selfattn_s{seed}/best.ckpt|model.compute_marginal_gain=true|${BATCH_SIZE:-8}"
  "s2d|s2d|vision_chronos2_s2d|uk_pv_s2d_s2d_s{seed}/best.ckpt|model.compute_marginal_gain=true|${BATCH_SIZE:-8}"
  "s2e|s2e|vision_chronos2_s2e|uk_pv_s2e_s2e_s{seed}/best.ckpt|model.compute_marginal_gain=true|${BATCH_SIZE_S2E:-4}"
  "A46b|s2e|vision_chronos2_s2e|uk_pv_s2e_s2e_s{seed}_A46b/best.ckpt|+ablation=A46b model.compute_marginal_gain=true|${BATCH_SIZE_A46B:-2}"
)

resolve_ckpt() {
  local model="$1" seed="$2" subpath="$3"
  local upper_model
  upper_model="$(echo "$model" | tr '[:lower:]' '[:upper:]')"
  local env_var="CKPT_${upper_model}"
  if [[ -n "${!env_var:-}" ]]; then echo "${!env_var}"; return; fi
  local target="${CKPT_DIR}/${subpath//\{seed\}/$seed}"
  if [[ ! -f "$target" && "$model" == "s1" && -f "${CKPT_DIR}/uk_pv_s1/best.ckpt" ]]; then
    echo "${CKPT_DIR}/uk_pv_s1/best.ckpt"; return
  fi
  echo "$target"
}

tag_for_model() {
  local model="$1" seed="$2" ckpt="$3"
  if [[ "$model" == "A46b" ]]; then echo "mmtsfm_A46b_s2e_goespvdaq_s${seed}"
  elif [[ "$model" == "s1" && "$ckpt" =~ uk_pv_s1/best.ckpt$ ]]; then echo "mmtsfm_s1_goespvdaq_grassmann_s${seed}"
  elif [[ "$model" == "s1" || "$model" == "s2a" ]]; then echo "mmtsfm_${model}_goespvdaq_selfattn_s${seed}"
  else echo "mmtsfm_${model}_goespvdaq_${model}_s${seed}"; fi
}

for seed in $SEEDS; do
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r name stage cfg default_subpath extra_args bs <<< "$spec"
    if [[ -n "$ONLY" ]] && ! grep -qw "$name" <<< "$ONLY"; then continue; fi

    ckpt="$(resolve_ckpt "$name" "$seed" "$default_subpath")"
    tag="$(tag_for_model "$name" "$seed" "$ckpt")"
    [[ "$name" == "s1" && "$ckpt" =~ uk_pv_s1/best.ckpt$ ]] && cfg="vision_chronos2_grassmann"

    if [[ "$SKIP_DONE" == "1" && -f "${RESULTS_DIR}/${tag}.json" ]]; then
      echo ">>> [SKIP] ${tag} — ${RESULTS_DIR}/${tag}.json already exists"
      continue
    fi

    if [[ ! -f "$ckpt" ]]; then
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "    (notice: ckpt '${ckpt}' not found on disk; continuing dry run)"
      elif [[ "$FAIL_FAST" == "1" ]]; then
        echo "FATAL: Checkpoint for ${name} (seed ${seed}) not found at: ${ckpt}" >&2; exit 1
      else
        echo "ERROR: Checkpoint for ${name} (seed ${seed}) not found at: ${ckpt}. Skipping." >&2; continue
      fi
    fi

    declare -a CMD=(
      python -m mmtsfm.train
      train=false test=true
      "+stage=${stage}" "model=${cfg}" data=goespvdaq
      "data.data_dir=${DATA_DIR}" "data.batch_size=${bs}" "data.num_workers=${NUM_WORKERS}"
      "trainer=${TRAINER_CFG}" trainer.devices=1 trainer.strategy=auto
      "seed=${seed}" "ckpt_path=${ckpt}"
      "model.results_dir=${RESULTS_DIR}" "model.results_tag=${tag}"
      'hydra.run.dir=logs/experiments/runs/${now:%Y-%m-%d_%H-%M-%S}_'"${tag}"
    )
    [[ -d "$vjepa_cache" ]] && CMD+=("data.vjepa_cache_dir=${vjepa_cache}")
    [[ -n "$extra_args" ]] && { IFS=' ' read -r -a extras <<< "$extra_args"; CMD+=("${extras[@]}"); }

    echo ">>> [EVAL] model=${name} seed=${seed} tag=${tag}"
    echo "    ckpt=${ckpt}"
    echo "    uv run ${CMD[*]}"

    if [[ "$DRY_RUN" == "1" ]]; then
      continue
    fi

    uv run "${CMD[@]}"
  done
done
