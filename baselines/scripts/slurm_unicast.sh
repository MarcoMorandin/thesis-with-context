#!/bin/bash
#SBATCH --job-name=t5-unicast
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

# Tier-5 UniCast (P1, MULTIMODAL track) — TRAIN + EVAL.
# UniCast soft-prompts real vision+text into a Chronos backbone, so it needs the
# multimodal dataset (per-window frames + generated weather text) from the
# multimodal track (skippd / goes16_nsrdb) — NOT the numerical parquet. That data
# is still downloading; this script is ready and fails loud until it exists.
#
#   sbatch --export=ALL,CONDA_ENV=unicast,MM_TRAIN=<dir>,MM_TEST=<dir>,\
#          MM_TEXT=<text>,CHRONOS_PATH=<dir>,VISION_MODEL=<id>,TEXT_MODEL=<id> \
#          scripts/slurm_unicast.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"

: "${CONDA_ENV:?set CONDA_ENV to the UniCast conda env (TIER5_INTEGRATION.md §1)}"
PRED_LEN="${PRED_LEN:-12}"; EPOCHS="${EPOCHS:-10}"; LR="${LR:-1e-4}"
OUT="${OUT:-tier5/vendor/unicast/out_ukpv}"

# ---- multimodal-track guard (real image+text required) ---------------------
for v in MM_TRAIN MM_TEST MM_TEXT CHRONOS_PATH VISION_MODEL TEXT_MODEL; do
    [[ -n "${!v:-}" ]] || { echo "ERROR: $v unset — UniCast needs the MULTIMODAL dataset
  (per-window frames + weather text). Build it from skippd/goes16 once that data
  lands (DATASET_CONTRACT V/text), then set MM_TRAIN/MM_TEST/MM_TEXT and the
  backbone paths. See docs/experiments/TIER5_INTEGRATION.md §0/§3."; exit 2; }
done
[[ -d "$MM_TRAIN" && -d "$MM_TEST" ]] || { echo "ERROR: MM_TRAIN/MM_TEST not dirs"; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
cd tier5/vendor/unicast

# ---- TRAIN -----------------------------------------------------------------
echo ">>> TRAIN UniCast (multimodal)"
python train_multi_modal_chronos.py \
  --forecasting_length "$PRED_LEN" \
  --vision_model_name "$VISION_MODEL" --text_model_name "$TEXT_MODEL" \
  --chronos_path "$CHRONOS_PATH" \
  --dataset_path "$MM_TRAIN" --dataset_text "$MM_TEXT" \
  --output_dir "$OUT" --learning_rate "$LR" --train_epoch "$EPOCHS"

# ---- EVAL ------------------------------------------------------------------
echo ">>> EVAL UniCast (multimodal)"
python test_multi_modal_chronos.py \
  --forecasting_length "$PRED_LEN" \
  --vision_model_name "$VISION_MODEL" \
  --test_dataset_path "$MM_TEST" --dataset_text "$MM_TEXT" \
  --checkpoint_path "$OUT"

echo "✓ UniCast done. Dump predictions + import to our metrics per TIER5_INTEGRATION.md §4."
