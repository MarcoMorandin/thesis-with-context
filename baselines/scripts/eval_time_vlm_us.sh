#!/bin/bash
# =============================================================================
# Zero-shot evaluation of UK-trained Time-VLM on US dataset (goes_pvdaq)
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env

TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"
VENV_NAME="${VENV_NAME:-timevlm}"
DATA="${DATA:-${TEAM_SCRATCH}/data_v2/dataset_all.parquet}"
UKPV_DIR="${UKPV_DIR:-${TEAM_SCRATCH}/data_v2/ukpv_rag}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results}"
SP_REF="${SP_REF:-${RESULTS_DIR}/smart_persistence_s2_goespvdaq.json}"
SEQ_LEN="${SEQ_LEN:-672}"
PRED_LEN="${PRED_LEN:-12}"
VLM_TYPE="${VLM_TYPE:-CLIP}"
MODEL_ID="${MODEL_ID:-ukpv_tvlm}"
VISION_MODEL_PATH="${VISION_MODEL_PATH:-${TEAM_SCRATCH}/weights/clip-vit-base-patch32}"
DRY_RUN="${DRY_RUN:-0}"

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline VISION_MODEL_PATH

echo ">>> [Time-VLM] Exporting goes_pvdaq test series to 30-min Informer CSV..."
if [[ "$DRY_RUN" != "1" ]]; then
  uv run python tier4/vendor/export_goes.py --data "$DATA" --out "$UKPV_DIR"
fi

common_args=(
  --task_name long_term_forecast --model TimeVLM --vlm_type "$VLM_TYPE"
  --data ukpv --features S --target OT --root_path "$UKPV_DIR"
  --enc_in 1 --dec_in 1 --c_out 1 --inverse
  --seq_len "$SEQ_LEN" --label_len 0 --pred_len "$PRED_LEN"
  --periodicity 48 --freq t --use_amp --seed 42
  --model_id "$MODEL_ID" --des Exp --itr 1 --gpu 0
)

echo ">>> [Time-VLM] Evaluating UK-trained checkpoint zero-shot on US test plant(s)..."
if [[ "$DRY_RUN" == "1" ]]; then
  echo "    source ${UV_ENVS_DIR}/${VENV_NAME}/bin/activate"
  echo "    python tier5/vendor/time_vlm/run.py --is_training 0 --data_path goes_pvdaq_test_1202.csv ${common_args[*]}"
  echo "    uv run python scripts/import_predictions.py --model time_vlm --tag s2_goespvdaq ..."
  exit 0
fi

if [[ -f "${UV_ENVS_DIR}/${VENV_NAME}/bin/activate" ]]; then
  source "${UV_ENVS_DIR}/${VENV_NAME}/bin/activate"
else
  echo "WARN: ${UV_ENVS_DIR}/${VENV_NAME}/bin/activate not found, trying local venv"
fi

(
  cd tier5/vendor/time_vlm
  for csv in "$UKPV_DIR"/goes_pvdaq_test_*.csv; do
    [[ -f "$csv" ]] || continue
    echo ">>> EVAL US plant $(basename "$csv")"
    python run.py --is_training 0 --data_path "$(basename "$csv")" "${common_args[@]}"
  done
)

echo ">>> [Time-VLM] Importing predictions against US Smart Persistence reference..."
uv run python scripts/import_predictions.py --model time_vlm --tag s2_goespvdaq \
  --glob 'tier5/vendor/time_vlm/results/*/goes_pvdaq_test_*_pred.npz' \
  --reference "$SP_REF" --ukpv_dir "$UKPV_DIR" --data "$DATA" --out "$RESULTS_DIR"
echo "✓ Time-VLM done → ${RESULTS_DIR}/time_vlm_s2_goespvdaq.json"
