#!/bin/bash
#SBATCH --job-name=mmtsfm-extract-vjepa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# =============================================================================
# Pre-extract V-JEPA 2.1 latents on a GPU node, so the curriculum's vision stages
# read a cache instead of re-encoding video every step. Writes one <key>.pt per
# plant/window under CACHE_DIR; re-running is idempotent (existing files skipped).
#
# The stride for the TRAIN split MUST equal the training TRAIN_STRIDE (default 12)
# or the cache keys won't match and training silently falls back to live encode.
#
#   sbatch scripts/extract_vjepa.sbatch                       # uk_pv, all splits
#   DATASET=uk_pv SPLITS="train val test" TRAIN_STRIDE=12 sbatch scripts/extract_vjepa.sbatch
# =============================================================================
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -d src/mmtsfm ]] || { echo "FATAL: submit from MMTSFM/"; exit 1; }
mkdir -p logs/slurm

[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1

TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data}"
DATASET="${DATASET:-uk_pv}"
SPLITS="${SPLITS:-train val test}"
VJEPA_ARCH="${VJEPA_ARCH:-vit_large}"
VIDEO_FRAMES="${VIDEO_FRAMES:-8}"
IMG_SIZE="${IMG_SIZE:-224}"
TRAIN_STRIDE="${TRAIN_STRIDE:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VJEPA_CACHE_ROOT="${VJEPA_CACHE_ROOT:-/leonardo_work/IscrC_MTSFM/vjepa_cache}"
CACHE_VER="${VJEPA_CACHE_VER:-${VJEPA_ARCH}_f${VIDEO_FRAMES}_s${IMG_SIZE}}"
CACHE_DIR="${CACHE_DIR:-${VJEPA_CACHE_ROOT}/${DATASET}/${CACHE_VER}}"

echo ">>> extract V-JEPA latents  dataset=${DATASET}  splits=[${SPLITS}]"
echo "    cache_dir=${CACHE_DIR}  train_stride=${TRAIN_STRIDE}  frames=${VIDEO_FRAMES}  img=${IMG_SIZE}"
mkdir -p "$CACHE_DIR"

for split in $SPLITS; do
  # val/test always use protocol stride (=H, handled by the loader); only the
  # TRAIN split takes an explicit stride, which must match training.
  declare -a STRIDE=(); [[ "$split" == "train" ]] && STRIDE=(--stride "$TRAIN_STRIDE")
  echo ">>> [${DATASET}/${split}]"
  uv run python scripts/extract_video_embeddings.py \
    --encoder vjepa2 --vjepa-arch "$VJEPA_ARCH" \
    --dataset "$DATASET" --split "$split" \
    --video-frames "$VIDEO_FRAMES" --img-size "$IMG_SIZE" --imagenet-norm \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" "${STRIDE[@]}" \
    || { echo "FATAL: extraction failed for ${DATASET}/${split}"; exit 1; }
done

echo ">>> DONE. ${DATASET} latents in ${CACHE_DIR}"
echo "    files: $(find "$CACHE_DIR" -name '*.pt' | wc -l)   size: $(du -sh "$CACHE_DIR" | cut -f1)"
