#!/bin/bash
#SBATCH --job-name=pv-tier5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --account=IscrC_MTSFM
# Leonardo 'normal' QOS = up to 24 h. boost_qos_dbg (30 min) only for a smoke.
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Run the *original* vendored Tier-5 multimodal-TS baselines on uk_pv test plants.
# Each model has its own conda env (deps conflict with baselines/ and each other).
# Full recipe: docs/experiments/TIER5_INTEGRATION.md
#
#   sbatch --export=ALL,MODEL=time_vlm,CONDA_ENV=timevlm,UKPV_CSV_DIR=… scripts/slurm_tier5.sh
#   sbatch --export=ALL,MODEL=visionts_pp,CONDA_ENV=visionts,UKPV_GLUONTS=… scripts/slurm_tier5.sh
#
# Required: MODEL (time_vlm|visionts_pp|unicast|aurora), CONDA_ENV, CKPT (model weights).
# Time-VLM: UKPV_CSV_DIR (reuse tier4/vendor/export_ukpv.py output).
# Optional: SEQ_LEN(24) PRED_LEN(12) VLM_TYPE(CLIP) SEEDS("42 43 44").

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

MODEL="${MODEL:-time_vlm}"
SEQ_LEN="${SEQ_LEN:-24}"; PRED_LEN="${PRED_LEN:-12}"
VLM_TYPE="${VLM_TYPE:-CLIP}"; SEEDS="${SEEDS:-42 43 44}"
VENDOR="tier5/vendor/${MODEL}"

# ---- fully offline (compute node has no internet) --------------------------
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
[[ -d "$HF_HOME" ]] || { echo "ERROR: HF_HOME not found ($HF_HOME) — run login-node prep"; exit 1; }
[[ -d "$VENDOR" ]] || { echo "ERROR: vendored code missing: $VENDOR"; exit 1; }
: "${CONDA_ENV:?set CONDA_ENV to the per-model conda env (TIER5_INTEGRATION.md §1)}"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"

echo "=== Tier-5 $MODEL  (ctx=$SEQ_LEN pred=$PRED_LEN) ==="

case "$MODEL" in
  time_vlm)
    : "${UKPV_CSV_DIR:?Time-VLM needs UKPV_CSV_DIR (tier4/vendor/export_ukpv.py output)}"
    python tier4/vendor/contract_check.py --inputs "$UKPV_CSV_DIR"   # reuse input gate
    cd "$VENDOR"
    for csv in "$UKPV_CSV_DIR"/uk_pv_test_*.csv; do
      site=$(basename "$csv" .csv | sed 's/uk_pv_test_//')
      echo ">>> Time-VLM plant=$site"
      python run.py --task_name long_term_forecast --is_training 0 \
        --model TimeVLM --vlm_type "$VLM_TYPE" \
        --data custom --root_path "$UKPV_CSV_DIR" --data_path "$(basename "$csv")" \
        --features S --target OT --model_id "ukpv_${site}" \
        --seq_len "$SEQ_LEN" --label_len 0 --pred_len "$PRED_LEN" --gpu 0
    done ;;
  visionts_pp)
    : "${UKPV_GLUONTS:?VisionTS++ needs UKPV_GLUONTS (GluonTS export dir, §0)}"
    : "${CKPT:?set CKPT to the VisionTS++ MAE checkpoint}"
    cd "$VENDOR"
    python scripts/batch_evaluate.py --model_path "$CKPT" \
      --dataset "$UKPV_GLUONTS" --context_length "$SEQ_LEN" --prediction_length "$PRED_LEN" ;;
  unicast|aurora)
    echo "ERROR: $MODEL is a MULTIMODAL-track baseline (real image+text)."
    echo "       Blocked until the skippd/goes16 multimodal data + V/text channels land."
    echo "       See docs/experiments/TIER5_INTEGRATION.md §0/§3."
    exit 2 ;;
  *) echo "unknown MODEL: $MODEL"; exit 1 ;;
esac

echo "✓ $MODEL done. Dump predictions + import to our metrics per TIER5_INTEGRATION.md §4,"
echo "  validate with tier4/vendor/contract_check.py --predictions <npz> --horizon $PRED_LEN."
