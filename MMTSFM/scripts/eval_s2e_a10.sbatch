#!/bin/bash
#SBATCH --job-name=eval_s2e_wrong_plant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%j_A10_s2e.out
#SBATCH --error=logs/slurm/%j_A10_s2e.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm ../baselines/results

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec uv run python -m mmtsfm.train \
  +stage=s2e \
  model=vision_chronos2_s2e \
  +ablation=A10 \
  train=false \
  test=true \
  ckpt_path=/leonardo_scratch/fast/IscrC_MTSFM/checkpoints/curriculum/uk_pv_s2e_s2e_s42/best.ckpt \
  data.data_dir=/leonardo_scratch/fast/IscrC_MTSFM/data_v2 \
  data.vjepa_cache_dir=/leonardo_work/IscrC_MTSFM/vjepa_cache/vit_large_f20_s224_nonhrv_sp45_a24h \
  model.results_dir=../baselines/results \
  model.results_tag=mmtsfm_A10_s2e_ukpv_s42
