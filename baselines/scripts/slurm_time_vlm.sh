#!/bin/bash
#SBATCH --job-name=t5-timevlm
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

# Tier-5 Time-VLM (P0, numerical track) — TRAIN + EVAL on uk_pv, end-to-end.
# Everything is set up here: data export, train on the pooled train plants, eval
# each disjoint test plant (reusing the trained checkpoint), baseline-contract
# check. No edits to the vendored code at run time (already adapted on push).
#
#   sbatch --export=ALL,VENV_NAME=timevlm,DATA=/path/dataset_all.parquet,\
#          VLM_CKPT_OK=1 scripts/slurm_time_vlm.sh
#
# Required: VENV_NAME (Time-VLM env, TIER5_INTEGRATION.md §1).
# Optional: DATA, SEQ_LEN(24) PRED_LEN(12) VLM_TYPE(CLIP) EPOCHS(10) MODEL_ID(ukpv_tvlm)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

# ---- offline (compute node has no internet) --------------------------------
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${TEAM_SCRATCH}/conda_pkgs}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-${TEAM_SCRATCH}/conda_envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TEAM_SCRATCH}/pip_cache}"
export UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
[[ -d "$HF_HOME" ]] || { echo "ERROR: HF_HOME missing ($HF_HOME) — run login_node_prep.sh"; exit 1; }

: "${VENV_NAME:?set VENV_NAME to the Time-VLM uv env (TIER5_INTEGRATION.md §1)}"
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
UKPV_DIR="${UKPV_DIR:-${TEAM_SCRATCH}/data/ukpv_rag}"
SEQ_LEN="${SEQ_LEN:-24}"; PRED_LEN="${PRED_LEN:-12}"
VLM_TYPE="${VLM_TYPE:-CLIP}"; EPOCHS="${EPOCHS:-10}"; MODEL_ID="${MODEL_ID:-ukpv_tvlm}"
export VISION_MODEL_PATH="${VISION_MODEL_PATH:-${TEAM_SCRATCH}/weights/clip-vit-base-patch32}"

# ---- 1. export uk_pv → Informer CSVs (reuse the tier-4 bridge) --------------
uv run python tier4/vendor/export_ukpv.py --data "$DATA" --out "$UKPV_DIR"
uv run python tier4/vendor/contract_check.py --inputs "$UKPV_DIR"

source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"
cd tier5/vendor/time_vlm

common_args=(--task_name long_term_forecast --model TimeVLM --vlm_type "$VLM_TYPE"
  --data custom --features S --target OT --root_path "$UKPV_DIR"
  --enc_in 1 --dec_in 1 --c_out 1 --inverse
  --seq_len "$SEQ_LEN" --label_len 0 --pred_len "$PRED_LEN"
  --model_id "$MODEL_ID" --des Exp --itr 1 --gpu 0)

# ---- 2. TRAIN on the stacked train-plant series ----------------------------
echo ">>> TRAIN Time-VLM on uk_pv_train_stacked.csv"
python run.py --is_training 1 --data_path uk_pv_train_stacked.csv \
  --train_epochs "$EPOCHS" "${common_args[@]}"

# ---- 3. EVAL each disjoint test plant (reuses the trained checkpoint) -------
# setting is built from model_id (constant) not data_path, so is_training=0 loads
# the same checkpoint; predictions are dumped per data_path stem.
for csv in "$UKPV_DIR"/uk_pv_test_*.csv; do
    echo ">>> EVAL plant $(basename "$csv")"
    python run.py --is_training 0 --data_path "$(basename "$csv")" "${common_args[@]}"
done

# ---- 4. baseline-contract check on dumped predictions ----------------------
cd "${SLURM_SUBMIT_DIR:-$OLDPWD}"
shopt -s nullglob
for npz in tier5/vendor/time_vlm/results/*/uk_pv_test_*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done

# ---- 5. import predictions → our NMAE/NRMSE/SS results JSON -----------------
uv run python scripts/import_predictions.py --model time_vlm --tag s2_ukpv \
    --glob 'tier5/vendor/time_vlm/results/*/uk_pv_test_*_pred.npz' \
    --reference results/smart_persistence_s2_ukpv.json
echo "✓ Time-VLM done → results/time_vlm_s2_ukpv.json (make_tables / summarize_ukpv pick it up)."
