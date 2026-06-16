#!/bin/bash
#SBATCH --job-name=t6-crossvivit
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

# Tier-6 CrossViViT (P0, domain SOTA, MULTIMODAL track) — TRAIN + EVAL.
# Boussif et al., NeurIPS 2023 — deep satellite(V)+irradiance(Y) cross-attention.
# Runs the authors' ORIGINAL PyTorch-Lightning+Hydra code (tier6/vendor/crossvivit),
# adapted to our contract, NOT reimplemented. Needs the multimodal dataset
# (per-window satellite frames + Y) from the multimodal track (goes16_nsrdb /
# skippd) — NOT the numerical parquet. That data is still being wired; this
# script is ready and fails loud until it exists.
#
#   sbatch --export=ALL,CONDA_ENV=crossvivit,MM_DATA=<deeplake_or_dir>,\
#          EXPERIMENT=cross_vivit scripts/slurm_crossvivit.sh
#
# Required: CONDA_ENV, MM_DATA (multimodal-track frames+Y for the datamodule).
# Optional: EXPERIMENT(cross_vivit) PRED_LEN(12) SEED(42) MAX_EPOCHS(50)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"

: "${CONDA_ENV:?set CONDA_ENV to the CrossViViT conda env (TIER6_INTEGRATION.md §1)}"
EXPERIMENT="${EXPERIMENT:-cross_vivit}"
PRED_LEN="${PRED_LEN:-12}"; SEED="${SEED:-42}"; MAX_EPOCHS="${MAX_EPOCHS:-50}"
OUT="${OUT:-tier6/vendor/crossvivit/out_ukpv}"

# ---- multimodal-track guard (real satellite frames required) ---------------
[[ -n "${MM_DATA:-}" ]] || { echo "ERROR: MM_DATA unset — CrossViViT needs the MULTIMODAL
  dataset (per-window satellite frames + Y), NOT the numerical parquet. Build it
  from goes16_nsrdb/skippd once that data lands (DATASET_CONTRACT V), wire the
  tscontext_datamodule to it, then set MM_DATA. See docs/experiments/TIER6_INTEGRATION.md."; exit 2; }
[[ -e "$MM_DATA" ]] || { echo "ERROR: MM_DATA not found: $MM_DATA"; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
cd tier6/vendor/crossvivit

# ---- TRAIN + TEST (Hydra/Lightning; test_step dumps *_pred.npz) -------------
echo ">>> TRAIN+EVAL CrossViViT (experiment=$EXPERIMENT, multimodal)"
python main.py \
  experiment="$EXPERIMENT" seed="$SEED" \
  train=True test=True \
  datamodule.data_dir="$MM_DATA" \
  trainer.max_epochs="$MAX_EPOCHS" \
  paths.output_dir="$OUT"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

# ---- baseline-contract check + import → our NMAE/NRMSE/SS results JSON ------
shopt -s nullglob
for npz in "$OUT"/*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done
uv run python scripts/import_predictions.py --model crossvivit --tag s2_mm \
    --glob "$OUT/*_pred.npz" \
    --reference results/smart_persistence_s2_ukpv.json
echo "✓ CrossViViT done → results/crossvivit_s2_mm.json (make_tables / summarize_ukpv pick it up)."
