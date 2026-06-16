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

# Tier-5 Aurora (P2) on the uk_pv track — TRAIN (fine-tune) + EVAL.
# NOTE: Aurora's data pipeline (utils/pretrain_dataset.py::Aurora_Single_Dataset)
# is **time-series + TEXT**, NOT images — it reads a per-series CSV (date+value)
# plus a matching JSON weather-text list (BERT-tokenized). So uk *images* do not
# apply to Aurora; it was blocked on the per-window TEXT, which tier5/uk_export.py
# now generates (templated from the uk covariates) alongside the CSVs. The
# authors' runner.py consumes that layout unchanged.
#
#   sbatch --export=ALL,VENV_NAME=aurora,AURORA_CKPT=<dir>,\
#          DATA=<dataset_all.parquet>,IMAGES_H5=<images_all.h5>,MODE=eval \
#          scripts/slurm_aurora.sh                    # MODE=eval | finetune
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
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"

: "${VENV_NAME:?set VENV_NAME to the Aurora uv env (TIER5_INTEGRATION.md §1)}"
: "${AURORA_CKPT:?set AURORA_CKPT to the Aurora checkpoint dir (utils/download_ckpt.py)}"
[[ -d "$AURORA_CKPT" ]] || { echo "ERROR: AURORA_CKPT not a dir: $AURORA_CKPT"; exit 1; }
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
IMAGES_H5="${IMAGES_H5:-${TEAM_SCRATCH}/data/images_all.h5}"
MODE="${MODE:-eval}"; PRED_LEN="${PRED_LEN:-12}"
EXPORT="${EXPORT:-tier5/vendor/aurora/data_ukpv}"
[[ -f "$DATA" ]] || { echo "ERROR: DATA parquet not found: $DATA"; exit 1; }

# ---- 1. export uk_pv → Aurora layout (per-series CSV + weather-text JSON) ---
# (images_all.h5 only used to share the tier6 window builder; Aurora ignores V)
uv run --with h5py --with pillow python tier5/uk_export.py --model aurora \
    --out "$EXPORT" --data "$DATA" --h5 "$IMAGES_H5" --pred_len "$PRED_LEN"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
cd tier5/vendor/aurora

if [[ "$MODE" == "finetune" ]]; then
    echo ">>> Aurora fine-tune (uk_pv TS+text)"
    python runner.py --mode pretrain --model_path "$AURORA_CKPT" \
        --dataset "../../../$EXPORT" --prediction_length "$PRED_LEN"
fi
echo ">>> Aurora EVAL (uk_pv TS+text)"
python runner.py --mode eval --model_path "$AURORA_CKPT" \
    --dataset "../../../$EXPORT" --prediction_length "$PRED_LEN"

echo "✓ Aurora done. Dump predictions (runner eval output) → *_pred.npz, then"
echo "  uv run python scripts/import_predictions.py --model aurora --tag s2_ukpv_mm ... (TIER5_INTEGRATION.md §4)."
