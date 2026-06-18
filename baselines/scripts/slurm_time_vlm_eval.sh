#!/bin/bash
#SBATCH --job-name=eval-timevlm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=01:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Evaluate Tier-5 Time-VLM using the already saved checkpoint.
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
[[ -d "$HF_HOME" ]] || { echo "ERROR: HF_HOME missing ($HF_HOME) — run login_node_prep.sh"; exit 1; }

: "${VENV_NAME:=timevlm}"
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
UKPV_DIR="${UKPV_DIR:-${TEAM_SCRATCH}/data/ukpv_rag_tvlm}"
SEQ_LEN="${SEQ_LEN:-24}"; PRED_LEN="${PRED_LEN:-12}"
VLM_TYPE="${VLM_TYPE:-CLIP}"; MODEL_ID="${MODEL_ID:-ukpv_tvlm}"
export VISION_MODEL_PATH="${VISION_MODEL_PATH:-${TEAM_SCRATCH}/weights/clip-vit-base-patch32}"

# ---- 1. Check/Export uk_pv → Informer CSVs --------------
uv run python tier4/vendor/export_ukpv.py --data "$DATA" --out "$UKPV_DIR"
uv run python tier4/vendor/contract_check.py --inputs "$UKPV_DIR"

source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"
cd tier5/vendor/time_vlm

common_args=(--task_name long_term_forecast --model TimeVLM --vlm_type "$VLM_TYPE"
  --data custom --features S --target OT --root_path "$UKPV_DIR"
  --enc_in 1 --dec_in 1 --c_out 1
  --seq_len "$SEQ_LEN" --label_len 0 --pred_len "$PRED_LEN"
  --model_id "$MODEL_ID" --des Exp --itr 1 --gpu 0)

# ---- 2. EVAL each disjoint test plant (reuses the trained checkpoint) -------
for csv in "$UKPV_DIR"/uk_pv_test_*.csv; do
    echo ">>> EVAL plant $(basename "$csv")"
    python run.py --is_training 0 --data_path "$(basename "$csv")" "${common_args[@]}"
done

# ---- 3. baseline-contract check on dumped predictions ----------------------
cd "${SLURM_SUBMIT_DIR:-$OLDPWD}"
shopt -s nullglob
for npz in tier5/vendor/time_vlm/results/*/uk_pv_test_*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done

# ---- 4. import predictions → our NMAE/NRMSE/SS results JSON -----------------
uv run python scripts/import_predictions.py --model time_vlm --tag s2_ukpv \
    --glob 'tier5/vendor/time_vlm/results/*/uk_pv_test_*_pred.npz' \
    --reference results/smart_persistence_s2_ukpv.json

# ---- 5. aggregate all results ----------------------------------------------
uv run python scripts/aggregate_all.py --results results \
    --md results/ALL_RESULTS.md --json results/ALL_RESULTS.json

echo "✓ Time-VLM evaluation and aggregation done!"
