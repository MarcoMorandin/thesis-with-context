#!/bin/bash
#SBATCH --job-name=t5-aurora-ft
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

# Tier-5 Aurora FINE-TUNED on uk_pv, then scored by the same runner the
# zero-shot arm uses. The zero-shot row is the H0 anchor; this row is the
# budget-matched opponent for MMTSFM s2d/s2e, which are themselves fine-tuned.
#
# Training sees only train plants, early stopping only val plants; the test
# plants are never loaded by finetune_ukpv.py. VISION_MODE is fixed before the
# run — it is not selected on results.
#
#   sbatch scripts/slurm_aurora_ft.sh                  # real-frame arm
#   VISION_MODE=pseudo sbatch scripts/slurm_aurora_ft.sh
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
RUN="${SLURM_JOB_ID:-manual}"
FT_OUT="${FT_OUT:-${TEAM_SCRATCH}/weights/aurora-ukpv-ft/${RUN}}"
OUT="${OUT:-tier5/vendor/aurora/results_ukpv_ft/${RUN}}"
VISION_MODE="${VISION_MODE:-real}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
VISUAL_HISTORY_STEPS="${VISUAL_HISTORY_STEPS:-8}"
TRAIN_STRIDE="${TRAIN_STRIDE:-12}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-500}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
TAG="${TAG:-s2_ukpv}"
REFERENCE="${REFERENCE:-results/smart_persistence_s2_ukpv.json}"
[[ -f "$DATA" ]] || { echo "ERROR: DATA parquet not found: $DATA"; exit 1; }
[[ -f "$H5" ]] || { echo "ERROR: image HDF5 not found: $H5"; exit 1; }
[[ -f "$REFERENCE" ]] || { echo "ERROR: Smart Persistence reference not found: $REFERENCE"; exit 1; }
[[ -x "$UV_ENVS_DIR/$VENV_NAME/bin/python" ]] || {
    echo "ERROR: Aurora uv env missing: $UV_ENVS_DIR/$VENV_NAME"; exit 1;
}

source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"

# ---- 1. fine-tune on train plants, early-stop on val plants -----------------
echo ">>> Aurora FINE-TUNE (uk_pv, ctx=$CTX pred=$PRED_LEN vision=$VISION_MODE)"
uv run --active --no-sync python tier5/vendor/aurora/finetune_ukpv.py \
    "data_path=$DATA" "h5_path=$H5" "ckpt_path=$AURORA_CKPT" \
    "history=$CTX" "horizon=$PRED_LEN" "vision_mode=$VISION_MODE" \
    "batch_size=$BATCH_SIZE" "train_stride=$TRAIN_STRIDE" \
    "max_epochs=$MAX_EPOCHS" "max_train_batches=$MAX_TRAIN_BATCHES" \
    "learning_rate=$LEARNING_RATE" \
    "visual_history_steps=$VISUAL_HISTORY_STEPS" "out=$FT_OUT"

# ---- 2. score the fine-tuned checkpoint with the zero-shot runner -----------
# vision_mode is passed explicitly so the runner skips its validation selection:
# the mode was already fixed when training started.
echo ">>> Aurora FINE-TUNED evaluation"
uv run --active --no-sync python tier5/vendor/aurora/run_ukpv.py \
    "data_path=$DATA" "h5_path=$H5" "ckpt_path=$FT_OUT/checkpoint" \
    "history=$CTX" "horizon=$PRED_LEN" "vision_mode=$VISION_MODE" \
    "batch_size=$EVAL_BATCH_SIZE" "num_samples=$NUM_SAMPLES" \
    "visual_history_steps=$VISUAL_HISTORY_STEPS" "out=$OUT"

# ---- 3. contract-check + import → NMAE/NRMSE/CRPS/SS results JSON -----------
# --active --no-sync: this job runs in the standalone aurora env, not the
# project .venv. Without the flags uv ignores VIRTUAL_ENV, tries to sync .venv
# against PyPI, and dies on a compute node with no internet (job 57819033).
shopt -s nullglob
for npz in "$OUT"/aurora_*_pred.npz; do
    uv run --active --no-sync python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN"
done
uv run --active --no-sync python scripts/import_predictions.py --model aurora_ft --tag "$TAG" \
    --glob "$OUT/aurora_*_pred.npz" \
    --reference "$REFERENCE" \
    --data "$DATA"

# knowledge/baselines.md requires a training budget alongside every fine-tuned row
uv run --active --no-sync python - "$FT_OUT/train_summary.json" "$TAG" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"✓ Aurora FT done → results/aurora_ft_{sys.argv[2]}.json")
print(f"  GPU-hours: {s['gpu_hours']:.2f}  epochs: {s['epochs_run']}  "
      f"trainable params: {s['trainable_parameters']:,}  "
      f"best val loss: {s['best_val_loss']:.5f}")
PY
