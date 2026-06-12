#!/bin/bash
#SBATCH --job-name=extract-vjepa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=12:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Pre-extract V-JEPA 2.1 latents for SKIPPD train+val+test splits.
# After this finishes, point training at the cache with:
#   data.vidtok_cache_dir=/leonardo_scratch/.../data/refactored/solar/skippd/vjepa_cache
#
# Shape params MUST match configs/data/default.yaml (video_frames=17, img_size=224,
# hist_steps=24, horizon=12, imagenet_norm=true).
#
# Override at submission time:
#   sbatch --export=ALL,DATASET=earthnet2021,VIDEO_FRAMES=4,IMG_SIZE=128 \
#          scripts/slurm_extract_vjepa.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source .env

export UV_OFFLINE=1
export UV_NO_SYNC=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TEAM_SCRATCH="/leonardo_scratch/fast/IscrC_MTSFM"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data}"
DATASET="${DATASET:-skippd}"
VJEPA_ARCH="${VJEPA_ARCH:-vit_large}"

# Must match training config — see configs/data/default.yaml
HIST_STEPS="${HIST_STEPS:-720}"   # 12h of 1-min PV; matches configs/data/default.yaml
HORIZON="${HORIZON:-12}"
VIDEO_FRAMES="${VIDEO_FRAMES:-17}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# V-JEPA weights cache (pre-populated by login_node_setup.sh)
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME}/hub}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"

mkdir -p logs/slurm

echo "============================================================"
echo " Job ID    : $SLURM_JOB_ID"
echo " Encoder   : vjepa2 / $VJEPA_ARCH"
echo " Dataset   : $DATASET"
echo " Data dir  : $DATA_DIR"
echo " Shape     : T=$HIST_STEPS  H=$HORIZON  T_v=$VIDEO_FRAMES  img=$IMG_SIZE"
echo "============================================================"

for split in train val test; do
    echo ""
    echo ">>> Extracting split: $split"
    uv run python scripts/extract_video_embeddings.py \
        --encoder      vjepa2          \
        --vjepa-arch   "$VJEPA_ARCH"   \
        --dataset      "$DATASET"      \
        --split        "$split"        \
        --hist-steps   "$HIST_STEPS"   \
        --horizon      "$HORIZON"      \
        --video-frames "$VIDEO_FRAMES" \
        --img-size     "$IMG_SIZE"     \
        --imagenet-norm                \
        --data-dir     "$DATA_DIR"     \
        --batch-size   "$BATCH_SIZE"   \
        --num-workers  "$NUM_WORKERS"
done

echo ""
echo "✓ Cache ready at: $DATA_DIR/refactored/solar/$DATASET/vjepa_cache"
echo "  Train with:    data.vidtok_cache_dir=$DATA_DIR/refactored/solar/$DATASET/vjepa_cache"
