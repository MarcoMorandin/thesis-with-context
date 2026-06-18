#!/bin/bash
#SBATCH --job-name=solar-vlm-generalization
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/leonardo/home/userexternal/mmorand1/Solar-VLM}"
SCRATCH="${SOLARVLM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM/SolarVLM}"
export SOLARVLM_SCRATCH="$SCRATCH"
export PROJECT_DIR="$PROJECT_DIR"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_HUB_OFFLINE=1

UNIFIED_ROOT="${SCRATCH}/dataset/refactored/"
MODE=${MODE:-all}
STATION="${STATION:-loc3}"
DISABLE_VISUAL="${DISABLE_VISUAL:-False}"

SKIPPD_FEATS="${SCRATCH}/vision_feats_skippd_qwen3vl"
case "$STATION" in
    loc1) CAMERA=1 ;;
    loc3) CAMERA=2 ;;
    *) echo "Unknown STATION='$STATION'. Use loc1 or loc3."; exit 1 ;;
esac
WOLL_FEATS="${SCRATCH}/vision_feats_wollongong_qwen3vl/cam${CAMERA}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRATCH}/checkpoints/long_term_forecast_skippd_v1_SolarVLM_SKIPPD_sl60_ll30_pl15_dm128_wu3_mm5_gc1.0_v1_0_abV0T0G1C1/checkpoint.pth}"

cd "$PROJECT_DIR"
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

mkdir -p logs/slurm "${SCRATCH}/results" "${SCRATCH}/test_results" "$SKIPPD_FEATS" "${SCRATCH}/vision_feats_wollongong_qwen3vl/cam1" "${SCRATCH}/vision_feats_wollongong_qwen3vl/cam2"

echo "======================================================================"
echo "Node        : $(hostname)"
echo "GPUs        : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Mode        : ${MODE}"
echo "Dataset     : ${UNIFIED_ROOT}"
echo "Checkpoint  : ${CHECKPOINT_PATH}"
echo "Station     : ${STATION} (camera ${CAMERA})"
echo "Date        : $(date)"
echo "======================================================================"

run_skippd_intra_site() {
    echo ""
    echo "==> Intra-site generalization: SKIPPD test split"
    if [ "${DISABLE_VISUAL}" != "True" ]; then
        echo "==> Precomputing Qwen3-VL vision features for SKIPPD..."
        uv run --no-sync python tools/precompute_vision_feats_skippd.py \
            --image_dir  "${UNIFIED_ROOT}/images/skippd" \
            --out_dir    "${SKIPPD_FEATS}" \
            --qwen_path  "${SCRATCH}/QwenQwen3-VL-Embedding-2B" \
            --batch_size 256 \
            --device     cuda \
            --fp16       1
    fi

    # FIXES: Enabled spatial cross-site components & matched training specs
    uv run --no-sync run_skippd.py \
        --is_training            0 \
        --model_id               skippd_v1 \
        --model                  SolarVLM \
        --data                   SKIPPD \
        --root_path              "${UNIFIED_ROOT}" \
        --features               MS \
        --target                 pv \
        --freq                   t \
        --seq_len                60 \
        --label_len              30 \
        --pred_len               15 \
        --enc_in                 1 \
        --dec_in                 1 \
        --use_era5               False \
        --load_checkpoint_path   "${CHECKPOINT_PATH}" \
        --checkpoints            "${SCRATCH}/checkpoints" \
        --results_dir            "${SCRATCH}/results" \
        --test_results_dir       "${SCRATCH}/test_results" \
        --d_model                128 \
        --n_heads                8 \
        --e_layers               3 \
        --d_ff                   512 \
        --dropout                0.1 \
        --patch_len              12 \
        --stride                 8 \
        --padding                8 \
        --embed                  timeF \
        --num_stations           1 \
        --num_frames             8 \
        --vlm_type               qwen3vl \
        --vlm_embed_dim          2048 \
        --qwen3_vl_model_path    "${SCRATCH}/QwenQwen3-VL-Embedding-2B" \
        --vision_feat_dir        "${SKIPPD_FEATS}" \
        --memory_bank_size       20 \
        --patch_memory_size      100 \
        --top_k                  5 \
        --periodicity            24 \
        --norm_const             0.4 \
        --use_mem_gate           True \
        --disable_gnn            True \
        --disable_cross_site_attn True \
        --disable_visual         "${DISABLE_VISUAL}" \
        --disable_text           False \
        --nonnegative            False \
        --warmup_epochs          3 \
        --multimodal_epochs      5 \
        --batch_size             32 \
        --num_workers            16 \
        --gpu                    0 \
        --seed                   2024

    echo ""
    echo "==> Running evaluation diagnostics for SKIPPD..."
    uv run tools/evaluate_generalization.py \
        --results_dir "${SCRATCH}/results" \
        --setting "long_term_forecast_skippd_v1_SolarVLM_SKIPPD_sl60_ll30_pl15_dm128_wu3_mm5_gc1.0_v1_0_abV0T0G1C1" \
        --data_dir "${UNIFIED_ROOT}" \
        --dataset "SKIPPD"
}

run_wollongong_extra_site() {
    for ST in loc1 loc3; do
        case "$ST" in
            loc1) CAM=1 ;;
            loc3) CAM=2 ;;
        esac
        W_FEATS="${SCRATCH}/vision_feats_wollongong_qwen3vl/cam${CAM}"

        echo ""
        echo "==> Extra-site generalization: Wollongong station=${ST} (camera ${CAM})"
        if [ "${DISABLE_VISUAL}" != "True" ]; then
            echo "==> Precomputing Qwen3-VL vision features for Wollongong camera ${CAM}..."
            uv run --no-sync python tools/precompute_vision_feats_wollongong.py \
                --image_root "${UNIFIED_ROOT}" \
                --camera     "${CAM}" \
                --out_dir    "${W_FEATS}" \
                --qwen_path  "${SCRATCH}/QwenQwen3-VL-Embedding-2B"
        fi

        # FIXES: Turned use_mem_gate to True to solve loading mismatch.
        # Turned spatial/cross-site modules back on and clipped negative artifacts.
        uv run --no-sync run_skippd.py \
            --is_training            0 \
            --model_id               wollongong_xfer_${ST} \
            --model                  SolarVLM \
            --data                   WOLLONGONG \
            --root_path              "${UNIFIED_ROOT}" \
            --features               MS \
            --target                 pv \
            --freq                   t \
            --seq_len                60 \
            --label_len              30 \
            --pred_len               15 \
            --enc_in                 1 \
            --dec_in                 1 \
            --use_era5               False \
            --wollongong_station     "${ST}" \
            --load_checkpoint_path   "${CHECKPOINT_PATH}" \
            --checkpoints            "${SCRATCH}/checkpoints" \
            --results_dir            "${SCRATCH}/results" \
            --test_results_dir       "${SCRATCH}/test_results" \
            --d_model                128 \
            --n_heads                8 \
            --e_layers               3 \
            --d_ff                   512 \
            --dropout                0.1 \
            --patch_len              12 \
            --stride                 8 \
            --padding                8 \
            --embed                  timeF \
            --num_stations           1 \
            --num_frames             8 \
            --vlm_type               qwen3vl \
            --vlm_embed_dim          2048 \
            --qwen3_vl_model_path    "${SCRATCH}/QwenQwen3-VL-Embedding-2B" \
            --vision_feat_dir        "${W_FEATS}" \
            --memory_bank_size       20 \
            --patch_memory_size      100 \
            --top_k                  5 \
            --periodicity            24 \
            --norm_const             0.4 \
            --use_mem_gate           True \
            --disable_gnn            True \
            --disable_cross_site_attn True \
            --disable_visual         "${DISABLE_VISUAL}" \
            --disable_text           False \
            --nonnegative            False \
            --warmup_epochs          3 \
            --multimodal_epochs      5 \
            --batch_size             32 \
            --num_workers            16 \
            --gpu                    0 \
            --seed                   2024

        echo ""
        echo "==> Running evaluation diagnostics for Wollongong station=${ST}..."
        uv run tools/evaluate_generalization.py \
            --results_dir "${SCRATCH}/results" \
            --setting "long_term_forecast_wollongong_xfer_${ST}_SolarVLM_WOLLONGONG_sl60_ll30_pl15_dm128_wu3_mm5_gc1.0_v1_0_abV0T0G1C1" \
            --data_dir "${UNIFIED_ROOT}" \
            --dataset "WOLLONGONG" \
            --station "${ST}"
    done
}

case "$MODE" in
    skippd) run_skippd_intra_site ;;
    wollongong) run_wollongong_extra_site ;;
    all)
        run_skippd_intra_site
        run_wollongong_extra_site
        ;;
    *) echo "Unknown MODE='$MODE'. Use skippd, wollongong, or all."; exit 1 ;;
esac

echo ""
echo "======================================================================"
echo "Done: $(date)"
echo "======================================================================"