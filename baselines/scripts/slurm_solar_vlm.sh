#!/bin/bash
#SBATCH --job-name=t6-solarvlm
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

# Tier-6 Solar-VLM (P0, domain SOTA) on our uk_pv MULTIMODAL track — TRAIN + EVAL.
# Solar-VLM is now vendored in-tree (tier6/vendor/solar_vlm). It forecasts a fixed
# co-located station SET jointly (GNN + cross-station attention); we keep the GNN
# on by clustering our uk_pv plants into spatial groups of `num_stations` (the
# disjoint cross-plant split is preserved at the group level — train groups use
# only train plants, test groups only unseen test plants). Vision is offline
# Qwen3-VL-Embedding-2B features over each plant's per-plant satellite frame from
# images_all.h5. The authors' model/Experiment run UNCHANGED; only the data adapter
# + a group-prefix in VisionFeatureStore are ours (see tier6/vendor/VENDOR_NOTICE.md).
#
#   sbatch --export=ALL,VENV_NAME=solar_vlm,QWEN_PATH=<qwen3vl_dir>,\
#          DATA=<dataset_all.parquet>,IMAGES_H5=<images_all.h5> \
#          scripts/slurm_solar_vlm.sh
#
# Required: VENV_NAME (Solar-VLM uv env), QWEN_PATH (Qwen3-VL weights dir),
#           IMAGES_H5 (uk_pv frames). Optional: DATA NUM_STATIONS(8) PRED_LEN(12)
#           EPOCHS(50) VFEAT_DIR.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env
BASELINES_DIR="$PWD"

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TEAM_SCRATCH}/pip_cache}"
export UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"

: "${VENV_NAME:?set VENV_NAME to the Solar-VLM uv env (precache_login.sh)}"
: "${QWEN_PATH:?set QWEN_PATH to the Qwen3-VL-Embedding-2B weights dir}"
[[ -d "$QWEN_PATH" ]] || { echo "ERROR: QWEN_PATH not a dir: $QWEN_PATH"; exit 1; }
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
IMAGES_H5="${IMAGES_H5:-${TEAM_SCRATCH}/data/images_all.h5}"
NUM_STATIONS="${NUM_STATIONS:-8}"; PRED_LEN="${PRED_LEN:-12}"; EPOCHS="${EPOCHS:-50}"
VFEAT_DIR="${VFEAT_DIR:-tier6/vendor/solar_vlm/vision_feats_ukpv}"
OUT="${OUT:-tier6/vendor/solar_vlm/results_ukpv}"
[[ -f "$DATA" && -f "$IMAGES_H5" ]] || { echo "ERROR: DATA/IMAGES_H5 not found"; exit 1; }

source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"
cd tier6/vendor/solar_vlm

# ---- 1. offline Qwen3-VL vision features (all splits, group-scoped) ---------
echo ">>> precompute Qwen3-VL vision features → $BASELINES_DIR/$VFEAT_DIR"
python tools/precompute_vision_feats_ukpv.py \
  --data "$DATA" --h5 "$IMAGES_H5" --out_dir "$BASELINES_DIR/$VFEAT_DIR" \
  --qwen_path "$QWEN_PATH" --num_stations "$NUM_STATIONS" --flag all

# ---- 2. TRAIN (train-split groups) + EVAL (unseen test-split groups) --------
echo ">>> TRAIN+EVAL Solar-VLM (uk_pv multimodal, cross-plant)"
python run_ukpv.py \
  --data_path "$DATA" --vision_feat_dir "$BASELINES_DIR/$VFEAT_DIR" \
  --qwen3_vl_model_path "$QWEN_PATH" --num_stations "$NUM_STATIONS" \
  --pred_len "$PRED_LEN" --train_epochs "$EPOCHS" --out "$BASELINES_DIR/$OUT"

# ---- 3. contract-check + import → our NMAE/NRMSE/SS results JSON ------------
cd "$BASELINES_DIR"
shopt -s nullglob
for npz in "$OUT"/preds/solar_vlm_*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done
uv run python scripts/import_predictions.py --model solar_vlm --tag s2_ukpv_mm \
    --glob "$OUT/preds/solar_vlm_*_pred.npz" \
    --reference results/smart_persistence_s2_ukpv.json
echo "✓ Solar-VLM done → results/solar_vlm_s2_ukpv_mm.json (make_tables / summarize_ukpv pick it up)."
