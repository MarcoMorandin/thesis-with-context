#!/bin/bash
#SBATCH --job-name=mmtsfm-vjepa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=4-00:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# ---- environment ----------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source .env                      # WANDB_API_KEY and any cluster-specific vars

# ---- offline mode (compute nodes have no internet) -----------------------------
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_OFFLINE=1
export UV_NO_SYNC=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ---- distributed training setup ------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
export NCCL_TIMEOUT=180000        # 30-min barrier timeout instead of hanging forever
export NCCL_DEBUG=WARN          # surface NCCL errors without flooding logs

# ---- paths -- override via sbatch --export or environment -----------------------
TEAM_SCRATCH="/leonardo_scratch/fast/IscrC_MTSFM"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints/vjepa_proposal}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"

# ---- torch / HF cache dirs (pre-populated by login_node_setup.sh) -------------
# V-JEPA 2.1 loaded via torch.hub - weights cached under TORCH_HOME/hub/
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME}/hub}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"

# ---- full Hydra tracebacks - critical for diagnosing model init errors ---------
export HYDRA_FULL_ERROR=1


# ---- model variant selection --------------------------------------------------
# Options:
#   - vision_chronos2_timeselfattn  (SAFE Baseline: pretrained Chronos temporal weights)
#   - vision_chronos2_grassmann     (TARGET: novel O(L) Grassmann mixing)
#
# Override MODEL_NAME / DATASET / NUM_WORKERS / BATCH_SIZE / MAX_EPOCHS either
# via --export or by passing KEY=VALUE as script arguments (handled below).
MODEL_NAME="${MODEL_NAME:-vision_chronos2_timeselfattn}"
DATASET="${DATASET:-skippd}"

# ---- parse script args: intercept KEY=VALUE env-var overrides, forward the rest
# to Hydra as extra config overrides (e.g. model.vision_cfg.fusion_mode=late).
HYDRA_EXTRA=()
for arg in "$@"; do
    case "$arg" in
        MODEL_NAME=*)  MODEL_NAME="${arg#MODEL_NAME=}"  ;;
        DATASET=*)     DATASET="${arg#DATASET=}"        ;;
        NUM_WORKERS=*) NUM_WORKERS="${arg#NUM_WORKERS=}";;
        BATCH_SIZE=*)  BATCH_SIZE="${arg#BATCH_SIZE=}"  ;;
        MAX_EPOCHS=*)  MAX_EPOCHS="${arg#MAX_EPOCHS=}"  ;;
        CKPT_DIR=*)    CKPT_DIR="${arg#CKPT_DIR=}"      ;;
        # Skip bare equals or empty args that can confuse Hydra
        "=" | "")      continue ;;
        *)             HYDRA_EXTRA+=("$arg")            ;;
    esac
done

mkdir -p logs/slurm "$CKPT_DIR"

# ---- V-JEPA pre-computed latent cache (huge speedup, encoder is frozen) -------
# Auto-derive cache path from dataset name. Set VJEPA_CACHE_DIR="" to force-disable.
# Run scripts/slurm_extract_vjepa.sh once per dataset/split to populate.
case "$DATASET" in
    skippd|solarnet|goes16_nsrdb)   _DOMAIN="solar" ;;
    earthnet2021|era5_eu|meteonet)  _DOMAIN="meteorology" ;;
    *)                              _DOMAIN="" ;;
esac
if [[ -n "$_DOMAIN" ]]; then
    VJEPA_CACHE_DIR="${VJEPA_CACHE_DIR-${DATA_DIR}/refactored/${_DOMAIN}/${DATASET}/vjepa_cache}"
else
    VJEPA_CACHE_DIR="${VJEPA_CACHE_DIR-}"
fi

echo "Running MMTSFM Proposal Training"
echo "Model:   $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Hub cache: $TORCH_HUB_DIR"
if [[ -n "$VJEPA_CACHE_DIR" && -d "$VJEPA_CACHE_DIR" ]]; then
    echo "V-JEPA cache: $VJEPA_CACHE_DIR  [HIT — encoder forward will be skipped]"
elif [[ -n "$VJEPA_CACHE_DIR" ]]; then
    echo "V-JEPA cache: $VJEPA_CACHE_DIR  [MISS — directory not found, encoder will run on-the-fly]"
else
    echo "V-JEPA cache: disabled  [encoder will run on-the-fly]"
fi

# ---- run -----------------------------------------------------------------------
# Construct the command array to avoid issues with shell expansion
CMD=(
    "python" "-m" "mmtsfm.train"
    "model=$MODEL_NAME"
    "trainer=slurm"
    "trainer.max_epochs=$MAX_EPOCHS"
    "trainer.default_root_dir=$CKPT_DIR"
    "data.dataset_name=$DATASET"
    "data.data_dir=$DATA_DIR"
    "data.num_workers=$NUM_WORKERS"
    "data.batch_size=$BATCH_SIZE"
    "model.vision_cfg.visual_encoder_type=vjepa2"
)

# Only attach cache override when the directory actually exists. The dataset
# falls back to on-the-fly encoding for any missing keys, but an empty/stale
# path should not silently mask cache-extraction problems.
if [[ -n "$VJEPA_CACHE_DIR" && -d "$VJEPA_CACHE_DIR" ]]; then
    CMD+=("data.vidtok_cache_dir=$VJEPA_CACHE_DIR")
fi

# Append extra arguments
for arg in "${HYDRA_EXTRA[@]}"; do
    if [[ -n "$arg" ]]; then
        CMD+=("$arg")
    fi
done

echo "Executing: uv run ${CMD[@]}"
printf "Arguments: %q\n" "${CMD[@]}"

srun --ntasks="$SLURM_NTASKS" --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
    uv run "${CMD[@]}"

# Example usage:
#
# 1. SAFE PATH (TimeSelfAttention, default):
#    sbatch scripts/slurm_train_vjepa.sh
#    sbatch scripts/slurm_train_vjepa.sh MODEL_NAME=vision_chronos2_timeselfattn model.vision_cfg.fusion_mode=late
#
# 2. GRASSMANN PATH (CausalGrassmannMixing):
#    sbatch scripts/slurm_train_vjepa.sh MODEL_NAME=vision_chronos2_grassmann
#
# 3. Any Hydra override can be appended:
#    sbatch scripts/slurm_train_vjepa.sh BATCH_SIZE=16 model.chronos_core_cfg.num_layers=8
