#!/bin/bash
#SBATCH --job-name=solar-vlm-skippd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=4-00:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# ============================================================================
# SLURM job script — SolarVLM training on SKIPPD (GPU nodes, no internet)
# ============================================================================

set -euo pipefail

PROJECT_DIR="/leonardo/home/userexternal/mmorand1/Solar-VLM"
SCRATCH="/leonardo_scratch/fast/IscrC_MTSFM/SolarVLM"

export SOLARVLM_SCRATCH="$SCRATCH"
export PROJECT_DIR="$PROJECT_DIR"
export UNIFIED_ROOT="${SCRATCH}/dataset/refactored/"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_HUB_OFFLINE=1

echo "======================================================================"
echo "Node        : $(hostname)"
echo "GPUs        : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Working dir : ${PROJECT_DIR}"
echo "Scratch dir : ${SCRATCH}"
echo "Dataset     : ${UNIFIED_ROOT}"
echo "Date        : $(date)"
echo "======================================================================"

mkdir -p logs/slurm "${SCRATCH}/vision_feats_skippd_qwen3vl" \
         "${SCRATCH}/checkpoints" "${SCRATCH}/results" "${SCRATCH}/test_results"

cd "$PROJECT_DIR"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

uv run tools/precompute_vision_feats_skippd.py \
    --image_dir  "${UNIFIED_ROOT}/images/skippd" \
    --out_dir    "${SCRATCH}/vision_feats_skippd_qwen3vl" \
    --qwen_path  "${SCRATCH}/QwenQwen3-VL-Embedding-2B" \
    --batch_size 256 \
    --device     cuda \
    --fp16       1


echo ""
echo "==> Step 2: Training SolarVLM on SKIPPD using torchrun DDP"

# FIXES: Re-added explicit string boolean values for value-expected arguments.
# Standalone true/false switches (use_multi_gpu, use_offline_vision) remain bare.
uv run run_skippd.py \
    --is_training            1           \
    --model_id               skippd_v1   \
    --model                  SolarVLM    \
    --data                   SKIPPD      \
    --root_path              "${UNIFIED_ROOT}" \
    --features               MS          \
    --target                 pv          \
    --freq                   t           \
    --use_era5               False       \
    --seq_len                60          \
    --label_len              30          \
    --pred_len               15          \
    --enc_in                 1           \
    --dec_in                 1           \
    --c_out                  1           \
    --d_model                128         \
    --n_heads                8           \
    --e_layers               3           \
    --d_ff                   512         \
    --dropout                0.1         \
    --patch_len              12          \
    --stride                 8           \
    --padding                8           \
    --embed                  timeF       \
    --num_stations           1           \
    --num_frames             8           \
    --vlm_type               qwen3vl     \
    --vlm_embed_dim          2048        \
    --vision_feat_dir        "${SCRATCH}/vision_feats_skippd_qwen3vl" \
    --checkpoints            "${SCRATCH}/checkpoints" \
    --results_dir            "${SCRATCH}/results" \
    --test_results_dir       "${SCRATCH}/test_results" \
    --memory_bank_size       20          \
    --patch_memory_size      100         \
    --top_k                  5           \
    --periodicity            24          \
    --norm_const             0.4         \
    --use_mem_gate           True        \
    --disable_gnn            True        \
    --disable_cross_site_attn True       \
    --disable_visual         False       \
    --disable_text           False       \
    --nonnegative            False       \
    --warmup_epochs          3           \
    --multimodal_epochs      5           \
    --train_epochs           50          \
    --batch_size             32          \
    --learning_rate          0.0005      \
    --multimodal_lr_ratio    0.2         \
    --memory_loss_weight     0.05        \
    --multimodal_loss_weight 0.1         \
    --grad_clip_norm         1.0         \
    --lr_warmup_steps        300         \
    --patience               15          \
    --lradj                  cosine      \
    --num_workers            16          \
    --loss_type              mse         \
    --use_offline_vision                 \
    --seed                   2024

echo ""
echo "======================================================================"
echo "Done: $(date)"
echo "======================================================================"