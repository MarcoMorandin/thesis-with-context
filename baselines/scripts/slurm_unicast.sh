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

# Tier-5 UniCast (P1) on the uk_pv MULTIMODAL track — TRAIN + EVAL.
# UniCast soft-prompts real vision (CLIP/BLIP) + text into a frozen Chronos-Bolt
# backbone. It needs REAL frames — now available on uk_pv via images_uk128.h5.
# tier5/uk_export.py emits UniCast's native on-disk layout (inputs.pt/targets/img)
# from the uk multimodal windows; the authors' train/test code runs UNCHANGED on
# it (only a --dump_npz flag was added to the test script, see VENDOR_NOTICE.md).
#
#   sbatch --export=ALL,CONDA_ENV=unicast,VISION_MODEL=CLIP,\
#          VISION_MODEL_PATH=<clip_weights_dir>,CHRONOS_PATH=<chronos_bolt_dir>,\
#          DATA=<all_curated.parquet>,IMAGES_H5=<images_uk128.h5> \
#          scripts/slurm_unicast.sh
#
# Required: CONDA_ENV, VISION_MODEL(CLIP|BLIP), VISION_MODEL_PATH, CHRONOS_PATH
#           (gated backbone weights — cache on the login node).
# Optional: TEXT_MODEL(+TEXT_MODEL_PATH) DATA IMAGES_H5 PRED_LEN(12) EPOCHS(10) LR(1e-4)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"

: "${CONDA_ENV:?set CONDA_ENV to the UniCast conda env (TIER5_INTEGRATION.md §1)}"
: "${VISION_MODEL:?set VISION_MODEL=CLIP or BLIP}"
: "${VISION_MODEL_PATH:?set VISION_MODEL_PATH to the CLIP/BLIP weights dir (login-node cached)}"
: "${CHRONOS_PATH:?set CHRONOS_PATH to the Chronos-Bolt weights dir (login-node cached)}"
DATA="${DATA:-${TEAM_SCRATCH}/data/numerical/all_curated.parquet}"
IMAGES_H5="${IMAGES_H5:-${TEAM_SCRATCH}/data/images_uk128.h5}"
PRED_LEN="${PRED_LEN:-12}"; EPOCHS="${EPOCHS:-10}"; LR="${LR:-1e-4}"
EXPORT="${EXPORT:-tier5/vendor/unicast/data_ukpv}"
OUT="${OUT:-tier5/vendor/unicast/out_ukpv_${VISION_MODEL}}"   # name carries CLIP/BLIP (test autodetects)
RES="${RES:-tier5/vendor/unicast/results_ukpv}"

[[ -f "$DATA" && -f "$IMAGES_H5" ]] || { echo "ERROR: DATA/IMAGES_H5 not found"; exit 1; }

# ---- 1. export uk_pv multimodal → UniCast layout (reuses tier6 bridge) ------
uv run --with h5py --with pillow python tier5/uk_export.py --model unicast \
    --out "$EXPORT" --data "$DATA" --h5 "$IMAGES_H5" --pred_len "$PRED_LEN"
DATASET_TEXT="$(cat "$EXPORT/dataset_text.txt")"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
cd tier5/vendor/unicast

# ---- 2. TRAIN (train/ + val/ under the export root) ------------------------
echo ">>> TRAIN UniCast ($VISION_MODEL, uk_pv multimodal)"
python train_multi_modal_chronos.py \
  --forecasting_length "$PRED_LEN" \
  --vision_model_name "$VISION_MODEL" --vision_model_path "${VISION_MODEL_PATH}" \
  ${TEXT_MODEL:+--text_model_name "$TEXT_MODEL"} ${TEXT_MODEL_PATH:+--text_model_path "$TEXT_MODEL_PATH"} \
  --chronos_path "$CHRONOS_PATH" \
  --dataset_path "../../../$EXPORT" --dataset_text "$DATASET_TEXT" \
  --output_dir "../../../$OUT" --learning_rate "$LR" --train_epoch "$EPOCHS"

# ---- 3. EVAL each held-out test plant → per-site npz -----------------------
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
shopt -s nullglob
for d in "$EXPORT"/test_*; do
    site="${d##*/test_}"
    ( cd tier5/vendor/unicast && python test_multi_modal_chronos.py \
        --forecasting_length "$PRED_LEN" \
        --test_dataset_path "../../../$d" --dataset_text "$DATASET_TEXT" \
        --checkpoint_path "../../../$OUT" \
        --dump_npz "../../../$RES/unicast_${site}_pred.npz" )
done

# ---- 4. contract-check + import → our NMAE/NRMSE/SS results JSON ------------
for npz in "$RES"/unicast_*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done
uv run python scripts/import_predictions.py --model unicast --tag s2_ukpv_mm \
    --glob "$RES/unicast_*_pred.npz" \
    --reference results/smart_persistence_s2_ukpv.json
echo "✓ UniCast done → results/unicast_s2_ukpv_mm.json (make_tables / summarize_ukpv pick it up)."
