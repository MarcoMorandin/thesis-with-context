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

# Tier-5 Aurora (decisionintelligence/Aurora, P2) — ZERO-SHOT on uk_pv.
# Aurora is a multimodal TS foundation model with a zero-shot generate() API; the
# old runner.py path was *training* (and a no-op — runner.py has no CLI). We use
# the unimodal TS path on the released DecisionIntelligence/Aurora checkpoint:
# feed each plant's power history, sample forecasts, average (run_ukpv.py mirrors
# the upstream TFB wrapper). No training, no images/text — like the other Tier-3/5
# zero-shot FMs. Dumps aurora_<site>_pred.npz → import_predictions.
#
#   sbatch --export=ALL,VENV_NAME=aurora,AURORA_CKPT=<DecisionIntelligence/Aurora dir>,\
#          DATA=<dataset_all.parquet> scripts/slurm_aurora.sh
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

: "${VENV_NAME:?set VENV_NAME to the Aurora uv env}"
: "${AURORA_CKPT:?set AURORA_CKPT to the DecisionIntelligence/Aurora checkpoint dir}"
[[ -d "$AURORA_CKPT" ]] || { echo "ERROR: AURORA_CKPT not a dir: $AURORA_CKPT"; exit 1; }
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
CTX="${CTX:-24}"; PRED_LEN="${PRED_LEN:-12}"
UKPV_DIR="${UKPV_DIR:-${TEAM_SCRATCH}/data/ukpv_rag_aurora}"
OUT="${OUT:-tier5/vendor/aurora/results_ukpv}"
[[ -f "$DATA" ]] || { echo "ERROR: DATA parquet not found: $DATA"; exit 1; }

# ---- 1. export uk_pv → per-plant test CSVs (date+OT), reuse the tier-4 bridge --
uv run python tier4/vendor/export_ukpv.py --data "$DATA" --out "$UKPV_DIR"

source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"

# ---- 2. zero-shot forecast on each uk_pv test plant -------------------------
echo ">>> Aurora ZERO-SHOT (uk_pv, ctx=$CTX pred=$PRED_LEN)"
python tier5/vendor/aurora/run_ukpv.py \
    --csv_dir "$UKPV_DIR" --ckpt_path "$AURORA_CKPT" \
    --context_len "$CTX" --pred_len "$PRED_LEN" --out "$OUT"

# ---- 3. contract-check + import → our NMAE/NRMSE/SS results JSON ------------
shopt -s nullglob
for npz in "$OUT"/aurora_*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done
uv run python scripts/import_predictions.py --model aurora --tag s2_ukpv \
    --glob "$OUT/aurora_*_pred.npz" \
    --reference results/smart_persistence_s2_ukpv.json
echo "✓ Aurora done → results/aurora_s2_ukpv.json (make_tables / summarize_ukpv pick it up)."
