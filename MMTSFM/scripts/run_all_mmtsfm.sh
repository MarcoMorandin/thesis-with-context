#!/bin/bash
#SBATCH --job-name=run-all-mmtsfm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# =============================================================================
# MMTSFM OFFLINE ORCHESTRATOR — train ALL ablations in parallel on a GPU node.
# =============================================================================
# Trains + tests the protocol-aligned MMTSFM model (BASELINE_PROTOCOL.md) on each
# requested dataset of record and writes NMAE/NRMSE/Skill-Score into
# baselines/results (the baselines results schema), so aggregate_all.py lists
# every MMTSFM ablation next to every baseline.
#
# Saturates the node: the ablation matrix (datasets × ABLATIONS) is dispatched
# one run per GPU, $GPUS at a time (default = all GPUs on the node, Leonardo
# boost = 4× A100). Each run is pinned with CUDA_VISIBLE_DEVICES + trainer.devices=1.
#
# Ablations are MMTSFM-architecture variants ONLY. Variants already covered by the
# baselines suite are NOT re-run here — Chronos-2 zero-shot / fine-tune (tier3) and
# TS-RAG / Cross-RAG (tier4) live in baselines/ and are evaluated there.
#
# Prereq: scripts/precache_login.sh has run on the login node (uv env, V-JEPA 2.1
# + Chronos-2 weights, data staged to $DATA_DIR). This script can also pre-extract
# V-JEPA latents before training, so no separate extraction Slurm script is needed.
#
#   sbatch scripts/run_all_mmtsfm.sh
#   DATASETS="uk_pv goes_pvdaq" sbatch scripts/run_all_mmtsfm.sh
#   ABLATIONS=$'grassmann_interleaved|model=vision_chronos2_grassmann' sbatch scripts/run_all_mmtsfm.sh
#   GPUS=2 MAX_EPOCHS=5 bash scripts/run_all_mmtsfm.sh                   # interactive node
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"     # MMTSFM/
MMTSFM_DIR="$PWD"
REPO_ROOT="$(cd .. && pwd)"

[[ -f .env ]] && source .env
[[ -f "$REPO_ROOT/.env" ]] && source "$REPO_ROOT/.env"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE="${WANDB_MODE:-offline}" HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- config (override via --export / env) ----------------------------------
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME}/hub}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data}"
DATA="${DATA:-${DATA_DIR}/dataset_all.parquet}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints}"
# MMTSFM results land beside the baselines so aggregate_all.py picks them up.
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/baselines/results}"
# committed Smart-Persistence Skill-Score reference (uk_pv), written by the
# baselines suite — run that first for a real SS, else SS is omitted.
SP_REF_UKPV="${SP_REF_UKPV:-${RESULTS_DIR}/smart_persistence_s2_ukpv.json}"

DATASETS="${DATASETS:-uk_pv}"          # space list, e.g. "uk_pv goes_pvdaq"
ENCODER="${ENCODER:-vjepa2}"           # vjepa2 | skip (applied to vision ablations)
PREEXTRACT_VJEPA="${PREEXTRACT_VJEPA:-1}"
EXTRACT_SPLITS="${EXTRACT_SPLITS:-train val test}"
VJEPA_ARCH="${VJEPA_ARCH:-vit_large}"
VJEPA_CACHE_ROOT="${VJEPA_CACHE_ROOT:-${DATA_DIR}/vjepa_cache}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-8}"
EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS:-4}"
EXTRACT_VIDEO_FRAMES="${EXTRACT_VIDEO_FRAMES:-8}"
EXTRACT_IMG_SIZE="${EXTRACT_IMG_SIZE:-224}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"        # per run; × GPUS ≤ --cpus-per-task
SEED="${SEED:-42}"
AGGREGATE="${AGGREGATE:-1}"            # 1 → refresh baselines/results/ALL_RESULTS at the end

# GPUs to saturate. Default = all visible on the node; ≥1 fallback for login/CPU.
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[[ "$GPUS" =~ ^[0-9]+$ && "$GPUS" -ge 1 ]] || GPUS=1

# ---- ablation matrix: "tag|hydra overrides" (one per line) ------------------
# MMTSFM-architecture variants only (NOT baseline-covered Chronos ZS/FT or RAG).
#   grassmann_interleaved : flagship — interleaved fusion + Causal Grassmann mixing
#   selfattn_late         : Variant B diagnostic — late fusion + TimeSelfAttention
#   selfattn_interleaved  : interleaving WITHOUT Grassmann (isolates the mixer)
#   grassmann_no_modbias  : flagship minus modality-pair offset bias (§8.1 ablation)
#   numeric_grassmann     : vision off, Grassmann TS-only (vision-lift lower bound;
#                           distinct from the Chronos-2 baseline — keeps Grassmann)
ABLATIONS_DEFAULT=$'grassmann_interleaved|model=vision_chronos2_grassmann
selfattn_late|model=vision_chronos2_timeselfattn
selfattn_interleaved|model.vision_cfg.fusion_mode=interleaved model.chronos_core_cfg.use_grassmann=false
grassmann_no_modbias|model=vision_chronos2_grassmann model.chronos_core_cfg.grassmann_modality_pair_bias=false
numeric_grassmann|model.vision_cfg.skip_vision_stack=true model.vision_cfg.fusion_mode=interleaved model.chronos_core_cfg.use_grassmann=true'
ABLATIONS="${ABLATIONS:-$ABLATIONS_DEFAULT}"

[[ -f "$DATA" ]] || { echo "FATAL: DATA not found: $DATA (run precache_login.sh)"; exit 1; }
mkdir -p logs/slurm "$CKPT_DIR" "$RESULTS_DIR"

N_ABL="$(grep -c '|' <<< "$ABLATIONS")"
echo "=============================================================="
echo " RUN ALL MMTSFM (parallel ablations)   job=${SLURM_JOB_ID:-local}"
echo " DATASETS=$DATASETS   ENCODER=$ENCODER   epochs=$MAX_EPOCHS"
echo " ABLATIONS=$N_ABL   GPUS=$GPUS   (concurrency $GPUS runs/wave)"
echo " DATA_DIR=$DATA_DIR   RESULTS_DIR=$RESULTS_DIR"
echo "=============================================================="

declare -A STATUS=()

# dataset → hydra data-config group
data_cfg() { case "$1" in uk_pv) echo ukpv;; goes_pvdaq) echo goespvdaq;; *) echo "";; esac; }
# dataset → short tag / Skill-Score reference
short()    { case "$1" in uk_pv) echo ukpv;; goes_pvdaq) echo goes;; *) echo "$1";; esac; }
sp_ref()   { case "$1" in uk_pv) echo "$SP_REF_UKPV";; *) echo "";; esac; }
dataset_horizon() { case "$1" in uk_pv) echo 12;; goes_pvdaq) echo 24;; *) echo "";; esac; }
# encoder → hydra vision override(s)
vis_flags() {
    case "$1" in
        vjepa2) echo "model.vision_cfg.visual_encoder_type=vjepa2" ;;
        skip)   echo "model.vision_cfg.skip_vision_stack=true" ;;
        *)      echo "" ;;
    esac
}

extract_vjepa_dataset() {
    local ds="$1"
    local horizon; horizon="$(dataset_horizon "$ds")"
    [[ -n "$horizon" ]] || { echo "  SKIP V-JEPA extraction for unknown dataset '$ds'"; return 0; }
    local cache_dir="${VJEPA_CACHE_ROOT}/${ds}"
    mkdir -p "$cache_dir"
    echo ""
    echo ">>> pre-extract V-JEPA: dataset=$ds cache=$cache_dir"
    for split in $EXTRACT_SPLITS; do
        local log="logs/slurm/extract_vjepa_${ds}_${split}.log"
        local -a CMD=(
            python scripts/extract_video_embeddings.py
            --encoder vjepa2
            --vjepa-arch "$VJEPA_ARCH"
            --dataset "$ds"
            --split "$split"
            --horizon "$horizon"
            --video-frames "$EXTRACT_VIDEO_FRAMES"
            --img-size "$EXTRACT_IMG_SIZE"
            --imagenet-norm
            --data-dir "$DATA_DIR"
            --batch-size "$EXTRACT_BATCH_SIZE"
            --num-workers "$EXTRACT_NUM_WORKERS"
        )
        echo "    $split → $log"
        echo "    uv run ${CMD[*]}" > "$log"
        CUDA_VISIBLE_DEVICES="${EXTRACT_GPU:-0}" uv run "${CMD[@]}" >> "$log" 2>&1 || {
            echo "FATAL: V-JEPA extraction failed for dataset=$ds split=$split; see $log"
            exit 1
        }
    done
}

if [[ "$PREEXTRACT_VJEPA" == "1" && "$ENCODER" == "vjepa2" ]]; then
    for ds in $DATASETS; do
        extract_vjepa_dataset "$ds"
    done
fi

# ---- build the job list: datasets × ablations -------------------------------
declare -a J_TAG J_DS J_DCFG J_OVR
for ds in $DATASETS; do
    dcfg="$(data_cfg "$ds")"
    [[ -n "$dcfg" ]] || { echo "  SKIP dataset '$ds' — unknown"; continue; }
    s="$(short "$ds")"
    while IFS='|' read -r name ovr; do
        [[ -z "$name" || "$name" == \#* ]] && continue
        J_TAG+=("mmtsfm_${name}_${s}")
        J_DS+=("$ds")
        J_DCFG+=("$dcfg")
        J_OVR+=("$ovr")
    done <<< "$ABLATIONS"
done
NJOBS=${#J_TAG[@]}

# launch_job <index> <gpu> — start one training run in the background on $gpu
launch_job() {
    local i="$1" gpu="$2"
    local tag="${J_TAG[$i]}" ds="${J_DS[$i]}" dcfg="${J_DCFG[$i]}" ovr="${J_OVR[$i]}"
    local ref; ref="$(sp_ref "$ds")"
    local -a CMD=(
        python -m mmtsfm.train
        "data=$dcfg" trainer=slurm trainer.devices=1 "seed=$SEED"
        "trainer.max_epochs=$MAX_EPOCHS"
        "trainer.default_root_dir=${CKPT_DIR}/${tag}"
        "data.data_dir=$DATA_DIR"
        "data.batch_size=$BATCH_SIZE" "data.num_workers=$NUM_WORKERS"
        "model.results_dir=$RESULTS_DIR" "model.results_tag=$tag"
    )
    [[ -n "$ref" && -f "$ref" ]] && CMD+=("model.sp_reference_path=$ref")
    local vf; vf="$(vis_flags "$ENCODER")"
    [[ -n "$vf" ]] && CMD+=($vf)
    if [[ "$ENCODER" == "vjepa2" && -d "${VJEPA_CACHE_ROOT}/${ds}" ]]; then
        CMD+=("data.vjepa_cache_dir=${VJEPA_CACHE_ROOT}/${ds}")
    fi
    # shellcheck disable=SC2206  -- intentional word-split: $ovr is a list of overrides
    CMD+=($ovr)
    local log="logs/slurm/${tag}.log"
    echo ">>> [GPU $gpu] $tag  →  $log"
    echo "    uv run ${CMD[*]}" > "$log"
    CUDA_VISIBLE_DEVICES="$gpu" uv run "${CMD[@]}" >> "$log" 2>&1 &
}

# ---- dispatch in waves of $GPUS, one run pinned per GPU ----------------------
echo ""; echo ">>> dispatching $NJOBS run(s), $GPUS per wave"
i=0
while (( i < NJOBS )); do
    pids=(); ptags=()
    for (( g=0; g<GPUS && i<NJOBS; g++, i++ )); do
        launch_job "$i" "$g"
        pids+=("$!"); ptags+=("${J_TAG[$i]}")
    done
    for k in "${!pids[@]}"; do
        wait "${pids[$k]}"; STATUS["${ptags[$k]}"]=$?
    done
done

# ---- aggregate (best-effort; baselines env) --------------------------------
if [[ "$AGGREGATE" == "1" ]]; then
    echo ""; echo ">>> aggregate → ${RESULTS_DIR}/ALL_RESULTS.{md,json}"
    ( cd "$REPO_ROOT/baselines" && uv run --group tier3 python scripts/aggregate_all.py \
        --results results --md results/ALL_RESULTS.md --json results/ALL_RESULTS.json ) \
        || echo "  (aggregate skipped — run it from baselines/ manually)"
fi

# ---- summary ----------------------------------------------------------------
echo ""; echo "=============================================================="
echo " RUN SUMMARY"
ok=0; fail=0
for n in "${!STATUS[@]}"; do
    rc="${STATUS[$n]}"
    if [[ "$rc" == 0 ]]; then ok=$((ok+1)); tag=OK
    elif [[ "$rc" == skip ]]; then tag=SKIP
    else fail=$((fail+1)); tag="FAIL(rc=$rc)"; fi
    printf "   %-24s %s\n" "$n" "$tag"
done
echo "   ----  ok=$ok fail=$fail"
echo " Results → $RESULTS_DIR/<tag>.json (+ ALL_RESULTS.md)"
echo " Per-run logs → logs/slurm/<tag>.log"
echo "=============================================================="
