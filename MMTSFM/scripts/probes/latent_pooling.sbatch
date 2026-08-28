#!/bin/bash
#SBATCH --job-name=latent_pooling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 4 is conservative for lrd_all_serial's per-user CPU cap. Confirm before raising:
#   scontrol show partition lrd_all_serial
#   sacctmgr show qos format=name,maxtresperuser%40,maxwall
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Ticket 13 successor. Reads cached V-JEPA latents and fits ridge probes — no
# GPU, no model forward, so it stays off the boost_usr_prod allocation.
#
# Asks whether V-JEPA's PATCH GRID carries ramp signal that the LatentSummarizer's
# single query per step throws away. 1x1 (what the summarizer emits today) vs
# 2x2 / 4x4. Grid >> 1x1 on ramp steps means the summarizer rewrite has headroom;
# a tie means V-JEPA is the ceiling and the visual branch is a measured negative
# result rather than an unfinished one.
#
#   sbatch scripts/probes/latent_pooling.sbatch
#   MAX_PER_PLANT=400 sbatch scripts/probes/latent_pooling.sbatch   # fast look
#
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm
[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export UV_OFFLINE=1 UV_NO_SYNC=1
# The probe is BLAS-bound (randomized SVD + one dense solve per arm). Which of
# these numpy reads depends on whether it links MKL or OpenBLAS, and a build that
# reads neither silently falls back to ONE thread — a 4x slowdown that shows up
# as a walltime kill rather than an error.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Env-driven so one script serves every cache. Hardcoding invites editing the
# file per run, which is how two arms overwrite one report and the surviving
# numbers get attributed to the wrong cache.
CACHE_DIR="${CACHE_DIR:-/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224_nonhrv_sp45}"
PARQUET="${PARQUET:-/leonardo_scratch/fast/IscrC_MTSFM/data_v2/dataset_all.parquet}"

[[ -d "$CACHE_DIR" ]] || { echo "FATAL: no cache at ${CACHE_DIR}" >&2; exit 1; }
[[ -f "$PARQUET"  ]] || { echo "FATAL: no parquet at ${PARQUET}" >&2; exit 1; }

echo ">>> cache   = ${CACHE_DIR}"
echo ">>> parquet = ${PARQUET}"
# Provenance is not decoration here: the stale v1 HRV cache still exists at the
# old default path, and reading it would answer the question about the wrong
# imagery. Loud, not silent.
if [[ -f "${CACHE_DIR}/_extract_meta.txt" ]]; then
  echo ">>> cache built with: $(cat "${CACHE_DIR}/_extract_meta.txt")"
else
  echo "!!! WARN: no _extract_meta.txt — cache provenance unverified" >&2
fi

declare -a ARGS=()
[[ -n "${MAX_PER_PLANT:-}" ]] && ARGS+=(--max-per-plant "${MAX_PER_PLANT}")
[[ -n "${PCA_DIM:-}"       ]] && ARGS+=(--pca-dim       "${PCA_DIM}")

uv run python scripts/probes/latent_pooling_bottleneck.py \
  --cache   "$CACHE_DIR" \
  --parquet "$PARQUET" \
  "${ARGS[@]}"
