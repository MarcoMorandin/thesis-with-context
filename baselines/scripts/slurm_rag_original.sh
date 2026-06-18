#!/bin/bash
#SBATCH --job-name=rag-original
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --account=IscrC_MTSFM
# Leonardo 'normal' QOS = up to 24 h (REGIME=proto re-pretrains the mixer, so
# allow headroom). boost_qos_dbg is 30 min max — only for a CONTRACT_CHECK smoke.
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Run the *original* vendored TS-RAG / Cross-RAG code (baselines/tier4/vendor/)
# on the uk_pv test plants. This is NOT the uv/run_eval path — the upstream repos
# pin numpy==1.25 + chronos-forecasting + faiss-gpu and conflict with the
# baselines venv, so they run from a dedicated uv env and their own zeroshot.py.
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
#   VENV_NAME     uv env name with the upstream requirements (e.g. tsrag/crossrag)
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
    cross_rag) VENDOR_DIR="tier4/vendor/cross_rag/cross-rag"; TOP_K="${TOP_K:-15}"
               # Cross-RAG names its retrieval DB <...>_${RETRIEVE_SPACE}_space.pkl;
               # retrieve_X.py builds the X-space (do_retrieve_Z is disabled), but
               # zeroshot.py reads RETRIEVE_SPACE (default "") → looks for __space.pkl.
               # Pin it to X so the load matches the built _X_space.pkl.
               export RETRIEVE_SPACE="${RETRIEVE_SPACE:-X}" ;;
    *) echo "unknown METHOD: $METHOD (ts_rag|cross_rag)"; exit 1 ;;
esac
case "$REGIME" in
    orig)  SEQ_LEN=512; PRED_LEN=64 ;;
    proto) SEQ_LEN=24;  PRED_LEN=12 ;;
    *) echo "unknown REGIME: $REGIME (orig|proto)"; exit 1 ;;
esac

# ---- prerequisite guards (fail loud, never silently mis-run) ---------------
: "${VENV_NAME:?set VENV_NAME to the upstream uv env (see TIER4_RAG_INTEGRATION.md §1)}"
: "${UKPV_CSV_DIR:?set UKPV_CSV_DIR to the exported uk_pv data dir (§3)}"
: "${BASE_CKPT:?set BASE_CKPT to the Chronos-Bolt base weights dir (§2)}"
[[ -d "$VENDOR_DIR" ]] || { echo "ERROR: vendored code missing: $VENDOR_DIR"; exit 1; }
[[ -d "$UKPV_CSV_DIR" ]] || { echo "ERROR: UKPV_CSV_DIR not found: $UKPV_CSV_DIR (run export_ukpv.py)"; exit 1; }
# ts_rag orig loads the released ARM mixer; cross_rag has NO released cross-attention
# mixer (the Drive best.pth is TS-RAG's ARM) so it pretrains its own below — no
# MIXER_CKPT required for cross_rag.
if [[ "$REGIME" == "orig" && "$METHOD" != "cross_rag" ]]; then
    : "${MIXER_CKPT:?REGIME=orig needs the released mixer checkpoint MIXER_CKPT (§2)}"
    [[ -f "$MIXER_CKPT" ]] || { echo "ERROR: MIXER_CKPT not found: $MIXER_CKPT"; exit 1; }
fi

# ---- fully offline (compute node has no internet; prep on the login node) --
# Everything below reads local caches only — run scripts/login_node_prep.sh first.
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${TEAM_SCRATCH}/conda_pkgs}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-${TEAM_SCRATCH}/conda_envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TEAM_SCRATCH}/pip_cache}"
export UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
[[ -d "$HF_HOME" ]] || { echo "ERROR: HF_HOME not found: $HF_HOME — run login_node_prep.sh"; exit 1; }
# zeroshot.py hardcodes amazon/chronos-t5-base for retrieval embeddings: must be cached.
if ! find "$HF_HOME" -path '*chronos-t5-base*' -name '*.safetensors' -o -path '*chronos-t5-base*' -name '*.bin' 2>/dev/null | grep -q .; then
    echo "ERROR: amazon/chronos-t5-base not in HF_HOME cache ($HF_HOME)."
    echo "       compute node is offline — cache it on the login node (login_node_prep.sh STAGE=rag)."
    exit 1
fi

# ---- upstream env (conda, NOT uv) ------------------------------------------
source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"

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

# cross_rag orig: no released cross-attention mixer exists, so pretrain it natively
# at 512/64 on the HF nkh/TS-RAG-Data 50m pretrain pairs (mirrors Cross-RAG-pretrain.sh),
# then the zeroshot loop below uses it. Data staged by precache_login.sh.
if [[ "$METHOD" == "cross_rag" && "$REGIME" == "orig" ]]; then
    CR_DB="${CROSSRAG_PRETRAIN_DB:-${TEAM_SCRATCH}/crossrag_pretrain/retrieval_database_512.parquet}"
    CR_PAIRS="${CROSSRAG_PRETRAIN_PAIRS:-${TEAM_SCRATCH}/crossrag_pretrain/pretrain_pairs_ctx512}"
    CR_STEPS="${CROSSRAG_PRETRAIN_STEPS:-20000}"
    CR_MIXER="$PWD/checkpoints/pretrain_Chronos_lb512_k${TOP_K}_CrossRAG/best.pth"
    if [[ -f "$CR_MIXER" ]]; then
        echo ">>> [cross_rag orig] reusing pretrained mixer: $CR_MIXER"
    else
        [[ -f "$CR_DB" && -d "$CR_PAIRS" ]] || {
            echo "ERROR: Cross-RAG pretrain data missing ($CR_DB / $CR_PAIRS) — run precache_login.sh"; exit 1; }
        echo ">>> [cross_rag orig] native pretrain (512/64, ${CR_STEPS} steps) on HF 50m pairs"
        export RETRIEVE_SPACE=X TABPFN_DUAL_LAMBDA_INIT="${TABPFN_DUAL_LAMBDA_INIT:-0.7}"
        python pretrain.py \
            --model_id "pretrain_Chronos_lb512_k${TOP_K}_CrossRAG" \
            --top_k "$TOP_K" --retrieve_lookback_length 512 --augment_mode moe \
            --context_length 512 --prediction_length 64 \
            --retrieval_database_path "$CR_DB" --data_path "$CR_PAIRS" \
            --train_steps "$CR_STEPS" --evaluation_steps 10000 \
            --optimizer adamw --learning_rate 0.00003 --weight_decay 0.01 --tmax 20 \
            --drop_prob 0.0 --batch_size 256 --shuffle_buffer_length 10000 \
            --freeze_chronos_bolt --model ChronosBoltRetrieve \
            --pretrained_model_path "$BASE_CKPT" --checkpoints ./checkpoints/ --gpu_loc 0
        # pretrain.py writes model_steps*.pth; mirror the upstream "latest → best.pth" copy
        latest="$(ls -t "checkpoints/pretrain_Chronos_lb512_k${TOP_K}_CrossRAG"/model_steps*.pth 2>/dev/null | head -1)"
        [[ -n "$latest" ]] && cp -f "$latest" "$CR_MIXER"
        [[ -f "$CR_MIXER" ]] || { echo "ERROR: pretrain produced no mixer at $CR_MIXER"; exit 1; }
    fi
    MIXER_CKPT="$CR_MIXER"
fi

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
    # --freq 0 → zeroshot.py maps it to the 'h' pandas alias (its int default 1 is
    # not a valid offset and crashes time_features on the first data load). Hourly
    # time features capture PV's daily cycle; the half-hour granularity is moot.
    python zeroshot.py \
        --root_path "$UKPV_CSV_DIR" --data_path "$(basename "$csv")" \
        --data custom_retrieve --model ChronosBoltRetrieve --augment_mode moe \
        --model_id "ukpv_${site}_${REGIME}_retrieve_${PRED_LEN}" \
        --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" --label_len 0 \
        --freq 0 \
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
    echo ">>> importing predictions to results"
    python scripts/import_predictions.py --model "${METHOD}_${REGIME}" --tag s2_ukpv \
        --glob "$VENDOR_DIR/results/forecast_evaluation/*_pred.npz" \
        --reference results/smart_persistence_s2_ukpv.json
else
    echo "NOTE: no *_pred.npz found."
fi

echo ""
echo "✓ ${METHOD}/${REGIME} done."
