#!/bin/bash
#SBATCH --job-name=t5-visionts
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Tier-5 VisionTS++ (P2, numerical track) on uk_pv, end-to-end.
# VisionTS++ is a *zero-shot* continual-pretrained MAE — there is no training
# stage (the "train" of train+eval is a no-op by design); we run the adapted
# zero-shot runner tier5/vendor/visionts_pp/run_ukpv.py and contract-check.
#
#   sbatch --export=ALL,VENV_NAME=visionts,MAE_CKPT=/path/visiontspp.ckpt,\
#          DATA=/path/dataset_all.parquet scripts/slurm_visionts_pp.sh
#
# Required: CONDA_ENV, MAE_CKPT (VisionTS++ checkpoint, HF Lefei/VisionTSpp).
# Optional: DATA, SEQ_LEN(24) PRED_LEN(12) PERIODICITY(48)
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
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
[[ -d "$HF_HOME" ]] || { echo "ERROR: HF_HOME missing ($HF_HOME) — run login_node_prep.sh"; exit 1; }

: "${VENV_NAME:?set VENV_NAME to the VisionTS++ uv env (TIER5_INTEGRATION.md §1)}"
: "${MAE_CKPT:?set MAE_CKPT to the VisionTS++ MAE checkpoint}"
[[ -f "$MAE_CKPT" ]] || { echo "ERROR: MAE_CKPT not found: $MAE_CKPT"; exit 1; }
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
UKPV_DIR="${UKPV_DIR:-${TEAM_SCRATCH}/data/ukpv_rag}"
SEQ_LEN="${SEQ_LEN:-24}"; PRED_LEN="${PRED_LEN:-12}"; PERIODICITY="${PERIODICITY:-48}"

# ---- 1. export uk_pv → CSVs (reuse the tier-4 bridge) ----------------------
uv run python tier4/vendor/export_ukpv.py --data "$DATA" --out "$UKPV_DIR"
uv run python tier4/vendor/contract_check.py --inputs "$UKPV_DIR"

source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"

# ---- 2. zero-shot forecast + dump predictions ------------------------------
OUT="tier5/vendor/visionts_pp/results_ukpv"
python tier5/vendor/visionts_pp/run_ukpv.py \
  --csv_dir "$UKPV_DIR" --ckpt_path "$MAE_CKPT" \
  --context_len "$SEQ_LEN" --pred_len "$PRED_LEN" --periodicity "$PERIODICITY" \
  --out "$OUT"

# ---- 3. baseline-contract check --------------------------------------------
shopt -s nullglob
for npz in "$OUT"/visionts_pp_*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done

# ---- 4. import predictions → our NMAE/NRMSE/SS results JSON -----------------
uv run python scripts/import_predictions.py --model visionts_pp --tag s2_ukpv \
    --glob "$OUT/visionts_pp_*_pred.npz" \
    --reference results/smart_persistence_s2_ukpv.json
echo "✓ VisionTS++ done → results/visionts_pp_s2_ukpv.json (make_tables / summarize_ukpv pick it up)."
