#!/bin/bash
#SBATCH --job-name=t6-sunset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Tier-6 SUNSET (P0, domain SOTA) on the uk_pv MULTIMODAL track — TRAIN + EVAL.
# Nie et al. (Stanford) — canonical sky-image CNN: a stack of past sky frames +
# PV history → PV forecast. Runs the authors' ORIGINAL TF2/Keras model
# (faithfully transcribed from tier6/vendor/sunset/models/SUNSET_forecast.ipynb,
# MIT), adapted to our contract via run_ukpv.py — NOT reimplemented. Consumes the
# uk_pv numerical track (Y) + the satellite frames in images_all.h5 (V), wired
# by tier6/uk_multimodal.py through the canonical image_h5_index pointer.
#
#   sbatch --export=ALL,VENV_NAME=sunset,DATA=<dataset_all.parquet>,\
#          IMAGES_H5=<images_all.h5> scripts/slurm_sunset.sh
#
# Required: VENV_NAME (TF2 + h5py env, TIER6_INTEGRATION.md §1).
# Optional: DATA IMAGES_H5 PRED_LEN(12) EPOCHS(20) SEED(42) IMG_SIZE(64) STRIDE(3)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

export WANDB_MODE=offline TF_CPP_MIN_LOG_LEVEL=2
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${TEAM_SCRATCH}/conda_pkgs}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-${TEAM_SCRATCH}/conda_envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TEAM_SCRATCH}/pip_cache}"
export UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"

: "${VENV_NAME:?set VENV_NAME to the SUNSET TF2 uv env (TIER6_INTEGRATION.md §1)}"
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
IMAGES_H5="${IMAGES_H5:-${TEAM_SCRATCH}/data/images_all.h5}"
PRED_LEN="${PRED_LEN:-12}"; EPOCHS="${EPOCHS:-20}"; SEED="${SEED:-42}"
IMG_SIZE="${IMG_SIZE:-64}"; STRIDE="${STRIDE:-3}"
OUT="${OUT:-tier6/vendor/sunset/results_ukpv}"

[[ -f "$DATA" ]] || { echo "ERROR: DATA parquet not found: $DATA"; exit 1; }
[[ -f "$IMAGES_H5" ]] || { echo "ERROR: IMAGES_H5 not found: $IMAGES_H5 (uk_pv frames)"; exit 1; }

source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"

# ---- TRAIN + EVAL (run_ukpv.py = original SUNSET model on uk_pv multimodal) --
# run_ukpv.py transcribes the upstream Keras graph (SUNSET_forecast.ipynb), feeds
# it tier6.uk_multimodal windows, and dumps sunset_<site>_pred.npz per held-out
# plant in our baseline-contract format. See tier6/vendor/VENDOR_NOTICE.md.
echo ">>> TRAIN+EVAL SUNSET (uk_pv multimodal)"
python tier6/vendor/sunset/run_ukpv.py \
  --data "$DATA" --h5 "$IMAGES_H5" \
  --epochs "$EPOCHS" --seed "$SEED" --pred_len "$PRED_LEN" \
  --history 24 --img_size "$IMG_SIZE" --stride "$STRIDE" --out "$OUT"

# ---- baseline-contract check + import → our NMAE/NRMSE/SS results JSON ------
shopt -s nullglob
for npz in "$OUT"/sunset_*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done
uv run python scripts/import_predictions.py --model sunset --tag s2_ukpv_mm \
    --glob "$OUT/sunset_*_pred.npz" \
    --reference results/smart_persistence_s2_ukpv.json
echo "✓ SUNSET done → results/sunset_s2_ukpv_mm.json (make_tables / summarize_ukpv pick it up)."
