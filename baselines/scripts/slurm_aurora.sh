#!/bin/bash
#SBATCH --job-name=t5-aurora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Tier-5 Aurora zero-shot on uk_pv. Validation plants choose between Aurora's
# native TS-rendered pseudo-image and the latest real satellite frame; that mode
# is frozen before test evaluation. Text is intentionally excluded.
#
#   sbatch scripts/slurm_aurora.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${TEAM_SCRATCH}/conda_pkgs}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-${TEAM_SCRATCH}/conda_envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TEAM_SCRATCH}/pip_cache}"
export UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"

VENV_NAME="${VENV_NAME:-aurora}"
AURORA_CKPT="${AURORA_CKPT:-${TEAM_SCRATCH}/weights/aurora-tsfm}"
[[ -d "$AURORA_CKPT" ]] || { echo "ERROR: AURORA_CKPT not a dir: $AURORA_CKPT"; exit 1; }
DATA="${DATA:-${TEAM_SCRATCH}/data_v2/dataset_all.parquet}"
H5="${H5:-${TEAM_SCRATCH}/data_v2/images_all.h5}"
CTX="${CTX:-672}"; PRED_LEN="${PRED_LEN:-12}"   # 14-day context / 6h horizon (uk_pv 30-min)
OUT="${OUT:-tier5/vendor/aurora/results_ukpv/${SLURM_JOB_ID:-manual}}"
VISION_MODE="${VISION_MODE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
VISUAL_HISTORY_STEPS="${VISUAL_HISTORY_STEPS:-8}"
REFERENCE="${REFERENCE:-results/smart_persistence_s2_ukpv.json}"
[[ -f "$DATA" ]] || { echo "ERROR: DATA parquet not found: $DATA"; exit 1; }
[[ -f "$H5" ]] || { echo "ERROR: image HDF5 not found: $H5"; exit 1; }
[[ -f "$REFERENCE" ]] || { echo "ERROR: Smart Persistence reference not found: $REFERENCE"; exit 1; }
[[ -x "$UV_ENVS_DIR/$VENV_NAME/bin/python" ]] || {
    echo "ERROR: Aurora uv env missing: $UV_ENVS_DIR/$VENV_NAME"; exit 1;
}

source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"

# ---- 1. validation selection + frozen-mode zero-shot test -------------------
echo ">>> Aurora ZERO-SHOT (uk_pv, ctx=$CTX pred=$PRED_LEN vision=$VISION_MODE)"
uv run --active --no-sync python tier5/vendor/aurora/run_ukpv.py \
    "data_path=$DATA" "h5_path=$H5" "ckpt_path=$AURORA_CKPT" \
    "history=$CTX" "horizon=$PRED_LEN" "vision_mode=$VISION_MODE" \
    "batch_size=$BATCH_SIZE" "num_samples=$NUM_SAMPLES" \
    "visual_history_steps=$VISUAL_HISTORY_STEPS" "out=$OUT"

# ---- 2. contract-check + import → NMAE/NRMSE/CRPS/SS results JSON -----------
# --active --no-sync: this job runs in the standalone aurora env, not the
# project .venv. Without the flags uv ignores VIRTUAL_ENV, tries to sync .venv
# against PyPI, and dies on a compute node with no internet (job 57819033).
shopt -s nullglob
for npz in "$OUT"/aurora_*_pred.npz; do
    uv run --active --no-sync python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN"
done
uv run --active --no-sync python scripts/import_predictions.py --model aurora --tag s2_ukpv \
    --glob "$OUT/aurora_*_pred.npz" \
    --reference "$REFERENCE" \
    --data "$DATA"
echo "✓ Aurora done → results/aurora_s2_ukpv.json"
