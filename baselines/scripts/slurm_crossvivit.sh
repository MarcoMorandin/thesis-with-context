#!/bin/bash
#SBATCH --job-name=t6-crossvivit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Tier-6 CrossViViT (P0, domain SOTA) on the uk_pv MULTIMODAL track — TRAIN + EVAL.
# Boussif et al., NeurIPS 2023 — deep satellite(V)+irradiance(Y) cross-attention.
# Drives the authors' ORIGINAL model src.models.cross_vivit.RoCrossViViT unchanged
# (tier6/vendor/crossvivit), via run_ukpv.py which feeds it our uk_pv multimodal
# windows (Y + images_all.h5 frames, wired by tier6/uk_multimodal.py). The model
# is NOT reimplemented; only the data adapter is ours, with documented
# approximations (single-channel 128px crops, synthetic per-pixel coords, no
# optical flow/elevation — see tier6/vendor/VENDOR_NOTICE.md).
#
#   sbatch --export=ALL,CONDA_ENV=crossvivit,DATA=<dataset_all.parquet>,\
#          IMAGES_H5=<images_all.h5> scripts/slurm_crossvivit.sh
#
# Required: CONDA_ENV (torch+einops+lightning env, TIER6_INTEGRATION.md §1).
# Optional: DATA IMAGES_H5 PRED_LEN(12) SEED(42) MAX_EPOCHS(50) IMG_SIZE(64) STRIDE(3)
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

: "${CONDA_ENV:?set CONDA_ENV to the CrossViViT conda env (TIER6_INTEGRATION.md §1)}"
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
IMAGES_H5="${IMAGES_H5:-${TEAM_SCRATCH}/data/images_all.h5}"
PRED_LEN="${PRED_LEN:-12}"; SEED="${SEED:-42}"; MAX_EPOCHS="${MAX_EPOCHS:-50}"
IMG_SIZE="${IMG_SIZE:-64}"; STRIDE="${STRIDE:-3}"
OUT="${OUT:-tier6/vendor/crossvivit/results_ukpv}"

[[ -f "$DATA" ]] || { echo "ERROR: DATA parquet not found: $DATA"; exit 1; }
[[ -f "$IMAGES_H5" ]] || { echo "ERROR: IMAGES_H5 not found: $IMAGES_H5 (uk_pv frames)"; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"

# ---- TRAIN + EVAL (run_ukpv.py drives the original RoCrossViViT on uk_pv) ----
echo ">>> TRAIN+EVAL CrossViViT (uk_pv multimodal)"
python tier6/vendor/crossvivit/run_ukpv.py \
  --data "$DATA" --h5 "$IMAGES_H5" \
  --epochs "$MAX_EPOCHS" --seed "$SEED" --pred_len "$PRED_LEN" \
  --img_size "$IMG_SIZE" --stride "$STRIDE" --out "$OUT"

# ---- baseline-contract check + import → our NMAE/NRMSE/SS results JSON ------
shopt -s nullglob
for npz in "$OUT"/crossvivit_*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done
uv run python scripts/import_predictions.py --model crossvivit --tag s2_ukpv_mm \
    --glob "$OUT/crossvivit_*_pred.npz" \
    --reference results/smart_persistence_s2_ukpv.json
echo "✓ CrossViViT done → results/crossvivit_s2_ukpv_mm.json (make_tables / summarize_ukpv pick it up)."
