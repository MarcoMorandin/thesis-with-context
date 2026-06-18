#!/bin/bash
#SBATCH --job-name=dump-predictions
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=01:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_dump-predictions.out
#SBATCH --error=logs/slurm/%j_dump-predictions.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="/leonardo_scratch/fast/IscrC_MTSFM"
export UV_CACHE_DIR="${TEAM_SCRATCH}/uv_cache"
export CONDA_PKGS_DIRS="${TEAM_SCRATCH}/conda_pkgs"
export CONDA_ENVS_DIRS="${TEAM_SCRATCH}/conda_envs"
export PIP_CACHE_DIR="${TEAM_SCRATCH}/pip_cache"
export UV_ENVS_DIR="${TEAM_SCRATCH}/uv_envs"
export HF_HOME="${TEAM_SCRATCH}/hf_cache"
export TORCH_HOME="${TEAM_SCRATCH}/torch_cache"

DATA="${TEAM_SCRATCH}/data/dataset_all.parquet"

echo ">>> Generating predictions for all baselines..."
uv run --group tier3 python run_eval.py \
    --model climatology_hourly persistence seasonal_naive smart_persistence lightgbm dlinear mlp patchtst tft itransformer timesfm_zs ttm_zs chronos2_zs ttm_ft \
    --data "$DATA" --tag s2 --seeds 42

echo "✓ All predictions dumped!"
