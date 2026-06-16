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

# Tier-5 Aurora (P2, MULTIMODAL track) — TRAIN (fine-tune) + EVAL.
# Aurora is a generative multimodal TSFM (image+text) loaded via
# AuroraForPrediction.from_pretrained; zero-shot eval needs only the checkpoint,
# but multimodal forecasting needs the real frames+text from the multimodal track
# (skippd/goes16). Ready; fails loud until that dataset exists.
#
#   sbatch --export=ALL,CONDA_ENV=aurora,AURORA_CKPT=<dir>,MM_DATASET=<dir>,\
#          MODE=eval scripts/slurm_aurora.sh        # MODE=eval | finetune
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"

: "${CONDA_ENV:?set CONDA_ENV to the Aurora conda env (TIER5_INTEGRATION.md §1)}"
: "${AURORA_CKPT:?set AURORA_CKPT to the Aurora checkpoint dir (utils/download_ckpt.py)}"
[[ -d "$AURORA_CKPT" ]] || { echo "ERROR: AURORA_CKPT not a dir: $AURORA_CKPT"; exit 1; }
MODE="${MODE:-eval}"; PRED_LEN="${PRED_LEN:-12}"

# ---- multimodal-track guard ------------------------------------------------
[[ -n "${MM_DATASET:-}" && -d "${MM_DATASET:-/nonexistent}" ]] || {
    echo "ERROR: MM_DATASET unset/missing — Aurora multimodal forecasting needs the
  real frames+text dataset (multimodal track). Build it from skippd/goes16 once
  that data lands, then set MM_DATASET. See TIER5_INTEGRATION.md §0/§3."; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
cd tier5/vendor/aurora

if [[ "$MODE" == "finetune" ]]; then
    echo ">>> Aurora fine-tune"
    python runner.py --mode pretrain --model_path "$AURORA_CKPT" \
        --dataset "$MM_DATASET" --prediction_length "$PRED_LEN"
fi
echo ">>> Aurora EVAL (zero-shot/probabilistic)"
python runner.py --mode eval --model_path "$AURORA_CKPT" \
    --dataset "$MM_DATASET" --prediction_length "$PRED_LEN"

echo "✓ Aurora done. Dump predictions + import to our metrics per TIER5_INTEGRATION.md §4."
