#!/bin/bash
#SBATCH --job-name=g0_ceiling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# G0 needs no GPU: it reads cached V-JEPA latents and fits ridge probes. Running
# it on the CPU partition keeps the whole diagnosis off the GPU allocation.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export UV_OFFLINE=1 UV_NO_SYNC=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

OUT=../baselines/results/probes
mkdir -p "$OUT"

uv run python -m scripts.probes.fit_ceiling \
  --cache-dir /leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224 \
  --parquet   /leonardo_scratch/fast/IscrC_MTSFM/data/dataset_all.parquet \
  --out       "$OUT/g0_ceiling_ukpv.json" \
  --horizon   12
