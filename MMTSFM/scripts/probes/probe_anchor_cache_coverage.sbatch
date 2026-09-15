#!/bin/bash
#SBATCH --job-name=anchor_cov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#
# Does the s2d latent cache already hold A46b's historical anchors?
#
# Filename scan only -- no parquet, no h5, no GPU, no torch import. It belongs
# on lrd_all_serial and off both the GPU allocation and the login node's CPU
# budget. One core is enough; the cost is metadata reads on a few hundred
# thousand directory entries, which is I/O-bound on the parallel FS.
#
#   sbatch scripts/probes/probe_anchor_cache_coverage.sbatch
#
# Point it at a different cache, or check more anchors, without editing:
#   CACHE_DIR=/path/to/cache ANCHORS=5 sbatch scripts/probes/probe_anchor_cache_coverage.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm
[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export UV_OFFLINE=1 UV_NO_SYNC=1

# The s2d control cache. Two names for it are in circulation -- g0_ceiling.sbatch
# defaults to f8 (which matches s2d's data.video_frames: 8) while ablations.md
# calls the 210 G control f20. Rather than pick one blind, a miss below prints
# the actual sibling directories and exits, so the run names the cache instead
# of guessing at it.
CACHE_ROOT="${CACHE_ROOT:-/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv}"
CACHE_DIR="${CACHE_DIR:-$CACHE_ROOT/vit_large_f8_s224_nonhrv_sp45}"
ANCHORS="${ANCHORS:-5}"
STRIDE_H="${STRIDE_H:-24.0}"

if [[ ! -d "$CACHE_DIR" ]]; then
  echo "CACHE_DIR does not exist: $CACHE_DIR"
  echo "Candidates under $CACHE_ROOT (re-submit with CACHE_DIR=<one of these>):"
  ls -1d "$CACHE_ROOT"/*/ 2>/dev/null || echo "  (cannot list $CACHE_ROOT)"
  exit 2
fi

echo "cache   : $CACHE_DIR"
echo "entries : $(ls -U1 "$CACHE_DIR" | head -400000 | wc -l) (capped count)"
echo "size    : $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"

# test first -- it is the split the protocol scores, and it runs at stride=H=6 h,
# so it is where a complete ladder is most likely. train uses TRAIN_STRIDE and is
# the one that can silently fail to divide the 24 h anchor step.
for SPLIT in test train; do
  echo
  echo "================ split=$SPLIT ================"
  uv run python scripts/probes/probe_anchor_cache_coverage.py \
    --cache-dir "$CACHE_DIR" \
    --split "$SPLIT" \
    --anchors "$ANCHORS" \
    --stride-hours "$STRIDE_H"
done
