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
VISUAL_WINDOW_H="${VISUAL_WINDOW_H:-6.0}"
# Minutes between consecutive frames. Empty => spread VIDEO_FRAMES across
# VISUAL_WINDOW_H (6h / 8 = 45 min). Latents from one spacing are meaningless
# for another, so the spacing is part of the cache identity below.
VISUAL_SPACING_MIN="${VISUAL_SPACING_MIN:-}"
VJEPA_CACHE_ROOT="${VJEPA_CACHE_ROOT:-/leonardo_work/IscrC_MTSFM/vjepa_cache}"
CACHE_VER="${VJEPA_CACHE_VER:-${VJEPA_ARCH}_f${VIDEO_FRAMES}_s${IMG_SIZE}}"
CACHE_DIR="${CACHE_DIR:-${VJEPA_CACHE_ROOT}/${DATASET}/${CACHE_VER}}"

echo ">>> extract V-JEPA latents  dataset=${DATASET}  splits=[${SPLITS}]"
echo "    cache_dir=${CACHE_DIR}  train_stride=${TRAIN_STRIDE}  frames=${VIDEO_FRAMES}  img=${IMG_SIZE}"
echo "    data_dir=${DATA_DIR}  window_h=${VISUAL_WINDOW_H}  spacing_min=${VISUAL_SPACING_MIN:-auto}"
mkdir -p "$CACHE_DIR"

# ---------------------------------------------------------------------------
# Cache identity guard. Extraction is IDEMPOTENT: existing <key>.pt are skipped.
# The key encodes plant and window origin but NOT the data source, the frame
# spacing or the visual window -- so pointing a second configuration at a
# populated cache skips every file and leaves the FIRST configuration's latents
# in place, under a directory named for the second. Nothing errors, the run
# reports success, and every downstream number describes data that was never
# extracted. This is the exact shape of the HRV/non-HRV mix-up that would
# otherwise be waiting here: the pre-existing vit_large_f8_s224 cache holds HRV
# latents built from data/, and the default CACHE_VER would land on top of it.
# ---------------------------------------------------------------------------
META="${CACHE_DIR}/_extract_meta.txt"
SIG="data_dir=${DATA_DIR}|arch=${VJEPA_ARCH}|frames=${VIDEO_FRAMES}|img=${IMG_SIZE}"
SIG="${SIG}|window_h=${VISUAL_WINDOW_H}|spacing_min=${VISUAL_SPACING_MIN:-auto}"
SIG="${SIG}|train_stride=${TRAIN_STRIDE}"
if [[ -f "$META" ]]; then
  PREV="$(cat "$META")"
  if [[ "$PREV" != "$SIG" ]]; then
    echo "FATAL: ${CACHE_DIR} was built with different settings." >&2
    echo "  existing: ${PREV}" >&2
    echo "  requested: ${SIG}" >&2
    echo "  Re-running here would SKIP the existing .pt files and silently keep" >&2
    echo "  the old latents. Use a different VJEPA_CACHE_VER or CACHE_DIR." >&2
    exit 1
  fi
elif compgen -G "${CACHE_DIR}/*.pt" > /dev/null; then
  echo "FATAL: ${CACHE_DIR} already holds .pt files but no _extract_meta.txt," >&2
  echo "  so the settings that produced them are unknown and cannot be checked." >&2
  echo "  Use a fresh VJEPA_CACHE_VER / CACHE_DIR." >&2
  exit 1
fi
printf '%s' "$SIG" > "$META"

for split in $SPLITS; do
  # val/test always use protocol stride (=H, handled by the loader); only the
  # TRAIN split takes an explicit stride, which must match training.
  declare -a STRIDE=(); [[ "$split" == "train" ]] && STRIDE=(--stride "$TRAIN_STRIDE")
  declare -a SPACING=()
  [[ -n "$VISUAL_SPACING_MIN" ]] && SPACING=(--visual-frame-spacing-min "$VISUAL_SPACING_MIN")
  echo ">>> [${DATASET}/${split}]"
  uv run python scripts/extract_video_embeddings.py \
    --encoder vjepa2 --vjepa-arch "$VJEPA_ARCH" \
    --dataset "$DATASET" --split "$split" \
    --video-frames "$VIDEO_FRAMES" --img-size "$IMG_SIZE" --imagenet-norm \
    --visual-window-hours "$VISUAL_WINDOW_H" "${SPACING[@]}" \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" "${STRIDE[@]}" \
    || { echo "FATAL: extraction failed for ${DATASET}/${split}"; exit 1; }
done

echo ">>> DONE. ${DATASET} latents in ${CACHE_DIR}"
echo "    files: $(find "$CACHE_DIR" -name '*.pt' | wc -l)   size: $(du -sh "$CACHE_DIR" | cut -f1)"
