#!/bin/bash
#SBATCH --job-name=rag-original
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=04:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Run the *original* vendored TS-RAG / Cross-RAG code (baselines/tier4/vendor/)
# on the uk_pv test plants. This is NOT the uv/run_eval path — the upstream repos
# pin numpy==1.25 + chronos-forecasting + faiss-gpu and conflict with the
# baselines venv, so they run from a dedicated conda env and their own zeroshot.py.
#
# Full recipe + fairness mapping: docs/experiments/TIER4_RAG_INTEGRATION.md
#
# Produces the two rows per method agreed in that doc:
#   REGIME=orig  → native ctx-512/pred-64, released pretrained mixer checkpoint
#   REGIME=proto → T=24/H=12, mixer re-pretrained on uk_pv (input-parity)
#
# Usage (submit from baselines/):
#   sbatch --export=ALL,METHOD=ts_rag,REGIME=orig  scripts/slurm_rag_original.sh
#   sbatch --export=ALL,METHOD=cross_rag,REGIME=proto scripts/slurm_rag_original.sh
#
# Required overrides (no defaults — these are off-repo artifacts, see the doc):
#   CONDA_ENV     conda env name with the upstream requirements (e.g. tsrag/crossrag)
#   UKPV_CSV_DIR  dir of exported uk_pv CSVs + train retrieval DB (export_ukpv.py)
#   BASE_CKPT     Chronos-Bolt base weights dir (amazon/chronos-bolt-base)
#   MIXER_CKPT    pretrained ARM / cross-attn checkpoint (REGIME=orig only)
# Optional: METHOD (ts_rag|cross_rag), REGIME (orig|proto), TOP_K, SEEDS

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

METHOD="${METHOD:-ts_rag}"
REGIME="${REGIME:-orig}"
SEEDS="${SEEDS:-42 43 44}"

case "$METHOD" in
    ts_rag)    VENDOR_DIR="tier4/vendor/ts_rag/TS-RAG";    TOP_K="${TOP_K:-10}" ;;
    cross_rag) VENDOR_DIR="tier4/vendor/cross_rag/cross-rag"; TOP_K="${TOP_K:-15}" ;;
    *) echo "unknown METHOD: $METHOD (ts_rag|cross_rag)"; exit 1 ;;
esac
case "$REGIME" in
    orig)  SEQ_LEN=512; PRED_LEN=64 ;;
    proto) SEQ_LEN=24;  PRED_LEN=12 ;;
    *) echo "unknown REGIME: $REGIME (orig|proto)"; exit 1 ;;
esac

# ---- prerequisite guards (fail loud, never silently mis-run) ---------------
: "${CONDA_ENV:?set CONDA_ENV to the upstream conda env (see TIER4_RAG_INTEGRATION.md §1)}"
: "${UKPV_CSV_DIR:?set UKPV_CSV_DIR to the exported uk_pv data dir (§3)}"
: "${BASE_CKPT:?set BASE_CKPT to the Chronos-Bolt base weights dir (§2)}"
[[ -d "$VENDOR_DIR" ]] || { echo "ERROR: vendored code missing: $VENDOR_DIR"; exit 1; }
[[ -d "$UKPV_CSV_DIR" ]] || { echo "ERROR: UKPV_CSV_DIR not found: $UKPV_CSV_DIR (run export_ukpv.py)"; exit 1; }
if [[ "$REGIME" == "orig" ]]; then
    : "${MIXER_CKPT:?REGIME=orig needs the released mixer checkpoint MIXER_CKPT (§2)}"
    [[ -f "$MIXER_CKPT" ]] || { echo "ERROR: MIXER_CKPT not found: $MIXER_CKPT"; exit 1; }
fi

# ---- fully offline (compute node has no internet; prep on the login node) --
# Everything below reads local caches only — run scripts/login_node_prep.sh first.
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
[[ -d "$HF_HOME" ]] || { echo "ERROR: HF_HOME not found: $HF_HOME — run login_node_prep.sh"; exit 1; }
# zeroshot.py hardcodes amazon/chronos-t5-base for retrieval embeddings: must be cached.
if ! find "$HF_HOME" -path '*chronos-t5-base*' -name '*.safetensors' -o -path '*chronos-t5-base*' -name '*.bin' 2>/dev/null | grep -q .; then
    echo "ERROR: amazon/chronos-t5-base not in HF_HOME cache ($HF_HOME)."
    echo "       compute node is offline — cache it on the login node (login_node_prep.sh STAGE=rag)."
    exit 1
fi

# ---- upstream env (conda, NOT uv) ------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---- baseline-contract preflight (offline, no model; needs the env's pandas)
python tier4/vendor/contract_check.py --inputs "$UKPV_CSV_DIR"
if [[ "${CONTRACT_CHECK:-0}" == "1" ]]; then
    echo "✓ CONTRACT_CHECK gate passed (inputs valid); skipping the heavy run."
    exit 0
fi

echo "============================================================"
echo " Method  : $METHOD   Regime: $REGIME (ctx=$SEQ_LEN pred=$PRED_LEN)"
echo " Vendor  : $VENDOR_DIR"
echo " Data    : $UKPV_CSV_DIR    top_k=$TOP_K"
echo "============================================================"

ORIG_DIR="$PWD"
cd "$VENDOR_DIR"
mkdir -p results/forecast_evaluation

# proto regime: re-pretrain the mixer at 24/12 first (orig uses released ckpt)
if [[ "$REGIME" == "proto" ]]; then
    echo ">>> [proto] re-pretraining mixer at ${SEQ_LEN}/${PRED_LEN}"
    python pretrain.py \
        --context_length "$SEQ_LEN" --prediction_length "$PRED_LEN" \
        --retrieve_lookback_length "$SEQ_LEN" --top_k "$TOP_K" --augment_mode moe \
        --retrieval_database_path "$UKPV_CSV_DIR/ukpv_train_pairs.parquet" \
        --data_path "$UKPV_CSV_DIR/ukpv_pretrain_pairs" \
        --freeze_chronos_bolt --train_steps 10000 --batch_size 256 --learning_rate 3e-5
    MIXER_CKPT="$PWD/checkpoints/proto_${METHOD}/best.pth"
fi

# zero-shot over each exported uk_pv test plant
for csv in "$UKPV_CSV_DIR"/uk_pv_test_*.csv; do
    site=$(basename "$csv" .csv | sed 's/uk_pv_test_//')
    echo ">>> zeroshot ${METHOD} ${REGIME} plant=${site}"
    python zeroshot.py \
        --root_path "$UKPV_CSV_DIR" --data_path "$(basename "$csv")" \
        --data custom_retrieve --model ChronosBoltRetrieve --augment_mode moe \
        --model_id "ukpv_${site}_${REGIME}_retrieve_${PRED_LEN}" \
        --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" --label_len 0 \
        --lookback_length "$SEQ_LEN" --top_k "$TOP_K" \
        --pretrained_model_path "$BASE_CKPT" \
        --checkpoint_model_path "$MIXER_CKPT" \
        --retrieval_database_dir "$UKPV_CSV_DIR" \
        --metadata_frequency half_hourly --metadata_database_name uk_pv \
        --embedding_model_type chronos --dimension 768 \
        --batch_size 256 --gpu_loc 0 \
        --save_file_name "ukpv_${METHOD}_${REGIME}.txt"
done

cd "$ORIG_DIR"

# ---- post-run baseline-contract check on dumped predictions ----------------
# Requires the §6 dump_predictions patch (writes *_pred.npz next to results).
shopt -s nullglob
preds=("$VENDOR_DIR"/results/forecast_evaluation/*_pred.npz)
if (( ${#preds[@]} )); then
    echo ">>> predictions contract check (H=$PRED_LEN)"
    for npz in "${preds[@]}"; do
        python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN"
    done
else
    echo "NOTE: no *_pred.npz found — apply the §6 dump patch to enable the"
    echo "      output contract check + import into our NMAE/SS metrics."
fi

echo ""
echo "✓ ${METHOD}/${REGIME} done. Import predictions into our metrics per"
echo "  docs/experiments/TIER4_RAG_INTEGRATION.md §6 (dump_predictions patch),"
echo "  then regenerate scripts/summarize_ukpv.py."
