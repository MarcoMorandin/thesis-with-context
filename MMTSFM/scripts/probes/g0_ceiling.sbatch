#!/bin/bash
#SBATCH --job-name=g0_ceiling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 4 is conservative for lrd_all_serial's per-user CPU cap. Confirm the exact
# limit before raising it:
#   scontrol show partition lrd_all_serial
#   sacctmgr show qos format=name,maxtresperuser%40,maxwall
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# G0 needs no GPU: it reads cached V-JEPA latents and fits ridge probes. Running
# it on the CPU partition keeps the whole diagnosis off the GPU allocation.
#
# Optional MAX_FILES passthrough to subsample without editing this file, e.g.
# if the job is running long against the walltime:
#   MAX_FILES=25000 sbatch scripts/probes/g0_ceiling.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm
[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export UV_OFFLINE=1 UV_NO_SYNC=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# The probe is BLAS-bound (Gram accumulation + one dense solve per penalty).
# Whether numpy links MKL or OpenBLAS decides which of these it reads, and a
# library that reads neither falls back to ONE thread -- a silent 4x slowdown
# that would push the job past its walltime rather than fail visibly.
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

OUT=../baselines/results/probes
mkdir -p "$OUT"

# Env-driven so the same script serves every arm. Hardcoding these invited
# editing the file per run, which is how two arms end up overwriting one
# report and the surviving numbers get attributed to the wrong cache.
CACHE_DIR="${CACHE_DIR:-/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224}"
PARQUET="${PARQUET:-/leonardo_scratch/fast/IscrC_MTSFM/data/dataset_all.parquet}"
TAG="${TAG:-ukpv}"
REPORT="$OUT/g0_ceiling_${TAG}.json"

declare -a MAX_FILES_ARGS=()
[[ -n "${MAX_FILES:-}" ]] && MAX_FILES_ARGS=(--max-files "${MAX_FILES}")
declare -a HOURS_ARGS=()
[[ -n "${ORIGIN_HOURS:-}" ]] && HOURS_ARGS=(--origin-hours "${ORIGIN_HOURS}")

[[ -e "$REPORT" ]] && { echo "FATAL: $REPORT exists; set TAG" >&2; exit 1; }
echo ">>> cache=${CACHE_DIR}"
echo ">>> parquet=${PARQUET}"
echo ">>> report=${REPORT}  origin_hours=${ORIGIN_HOURS:-all}"
[[ -f "${CACHE_DIR}/_extract_meta.txt" ]] && \
  echo ">>> cache built with: $(cat "${CACHE_DIR}/_extract_meta.txt")"

uv run python -m scripts.probes.fit_ceiling \
  --cache-dir "$CACHE_DIR" \
  --parquet   "$PARQUET" \
  --out       "$REPORT" \
  --horizon   12 \
  "${MAX_FILES_ARGS[@]}" "${HOURS_ARGS[@]}"
