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
# Resolve MMTSFM/ as the workdir. Under sbatch the job script runs from a
# SPOOL COPY (/var/spool/slurmd/.../slurm_script), so the script's own path is
# useless there — use $SLURM_SUBMIT_DIR, accepting submission from either
# MMTSFM/ or the repo root. The script-path fallback covers interactive
# `bash scripts/run_all_mmtsfm.sh`. NOTE: the #SBATCH --output path above is
# resolved against the SUBMIT dir before this runs — submit from MMTSFM/ so
# logs/slurm exists (committed .gitkeep), or pre-create it at the repo root.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/run_all_mmtsfm.sh" ]]; then
    cd "$SLURM_SUBMIT_DIR"                                       # sbatch from MMTSFM/
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/MMTSFM/scripts/run_all_mmtsfm.sh" ]]; then
    cd "${SLURM_SUBMIT_DIR}/MMTSFM"                              # sbatch from repo root
else
    cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/.."  # interactive bash
fi
# Fail loudly if resolution went anywhere unexpected (e.g. a future SLURM
# spool-path change) instead of running uv/python in the wrong tree.
[[ -f pyproject.toml && -d src/mmtsfm ]] || {
    echo "FATAL: workdir resolution failed — PWD=$PWD is not MMTSFM/ (submit from MMTSFM/ or the repo root)"
    exit 1
}
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
# Settings-versioned cache subdir: a cache built with a different arch / frame
# count / resolution must never be silently reused (wrong latent dims).
VJEPA_CACHE_VER="${VJEPA_ARCH}_f${EXTRACT_VIDEO_FRAMES}_s${EXTRACT_IMG_SIZE}"
vjepa_cache_dir() { echo "${VJEPA_CACHE_ROOT}/$1/${VJEPA_CACHE_VER}"; }
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"        # per run; × GPUS ≤ --cpus-per-task
SEED="${SEED:-42}"
AGGREGATE="${AGGREGATE:-1}"            # 1 → refresh baselines/results/ALL_RESULTS at the end
RESUME="${RESUME:-1}"                  # 1 → resume each run from <tag>/last.ckpt when present
# Walltime safety valves: stride-1 uk_pv epochs are huge; a job killed mid-fit
# writes NO results (protocol metrics are produced at test time only).
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"   # e.g. 5000 → cap batches/epoch
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-}"     # e.g. 2000 → val + ckpt every N batches

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
selfattn_late|model=vision_chronos2_timeselfattn data.batch_size=8
selfattn_interleaved|model.vision_cfg.fusion_mode=interleaved model.chronos_core_cfg.use_grassmann=false
grassmann_no_modbias|model=vision_chronos2_grassmann model.chronos_core_cfg.grassmann_modality_pair_bias=false
numeric_grassmann|model.vision_cfg.skip_vision_stack=true model.vision_cfg.fusion_mode=interleaved model.chronos_core_cfg.use_grassmann=true +data.emit_vision=false'
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
# dataset → short tag / Skill-Score reference. ALWAYS pass a dataset-specific
# path (even a not-yet-existing one): with a null sp_reference_path the
# evaluator falls back to the committed uk_pv reference, which would score
# goes_pvdaq Skill-Scores against the wrong dataset's Smart Persistence.
short()    { case "$1" in uk_pv) echo ukpv;; goes_pvdaq) echo goes;; *) echo "$1";; esac; }
sp_ref()   { case "$1" in
                 uk_pv)      echo "$SP_REF_UKPV";;
                 goes_pvdaq) echo "${RESULTS_DIR}/smart_persistence_s2_goes.json";;
                 *)          echo "";;
             esac; }
dataset_horizon() { case "$1" in uk_pv) echo 12;; goes_pvdaq) echo 24;; *) echo "";; esac; }
# encoder → hydra vision override(s)
vis_flags() {
    case "$1" in
        vjepa2) echo "" ;;
        skip)   echo "model.vision_cfg.skip_vision_stack=true" ;;
        *)      echo "" ;;
    esac
}

# launch_extract <ds> <split> <gpu> — one extraction in the background on $gpu.
# No directory-level "already populated" skip here: the extractor is idempotent
# per FILE, so a partially-built cache is completed instead of frozen (a frozen
# partial cache silently pushes whole training batches onto the raw V-JEPA
# encode path — the slow/OOM mode).
launch_extract() {
    local ds="$1" split="$2" gpu="$3"
    local horizon; horizon="$(dataset_horizon "$ds")"
    local cache_dir; cache_dir="$(vjepa_cache_dir "$ds")"
    mkdir -p "$cache_dir"
    if [[ -n "$(find "${VJEPA_CACHE_ROOT}/${ds}" -maxdepth 1 -name '*.pt' 2>/dev/null | head -n 1)" ]]; then
        echo "  NOTE: legacy unversioned cache at ${VJEPA_CACHE_ROOT}/${ds} — if it was built"
        echo "        with ${VJEPA_CACHE_VER}, move its *.pt into $cache_dir to reuse it."
    fi
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
        --cache-dir "$cache_dir"
        --batch-size "$EXTRACT_BATCH_SIZE"
        --num-workers "$EXTRACT_NUM_WORKERS"
    )
    echo ">>> [GPU $gpu] extract $ds/$split → $log"
    echo "    uv run ${CMD[*]}" > "$log"
    CUDA_VISIBLE_DEVICES="$gpu" uv run "${CMD[@]}" >> "$log" 2>&1 &
}

if [[ "$PREEXTRACT_VJEPA" == "1" && "$ENCODER" == "vjepa2" ]]; then
    # dataset × split extraction jobs, dispatched across all GPUs in waves.
    declare -a E_DS=() E_SPLIT=()
    for ds in $DATASETS; do
        [[ -n "$(dataset_horizon "$ds")" ]] || { echo "  SKIP V-JEPA extraction for unknown dataset '$ds'"; continue; }
        for split in $EXTRACT_SPLITS; do E_DS+=("$ds"); E_SPLIT+=("$split"); done
    done
    e=0
    while (( e < ${#E_DS[@]} )); do
        epids=(); enames=()
        for (( g=0; g<GPUS && e<${#E_DS[@]}; g++, e++ )); do
            launch_extract "${E_DS[$e]}" "${E_SPLIT[$e]}" "$g"
            epids+=("$!"); enames+=("${E_DS[$e]}/${E_SPLIT[$e]}")
        done
        for k in "${!epids[@]}"; do
            wait "${epids[$k]}" || {
                echo "FATAL: V-JEPA extraction failed for ${enames[$k]}; see logs/slurm/"
                exit 1
            }
        done
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
    # Pass the dataset-specific SP reference even when the file does not exist
    # yet: the evaluator then omits the Skill-Score instead of silently falling
    # back to the committed uk_pv reference (wrong dataset).
    [[ -n "$ref" ]] && CMD+=("model.sp_reference_path=$ref")
    # devices=1 → no DDP process group; skips slurm.yaml's ddp strategy (and the
    # per-run MASTER_PORT juggling it required).
    CMD+=("trainer.strategy=auto")
    # Unique hydra run dir per run: parallel launches in the same second would
    # otherwise share logs/experiments/runs/<timestamp> and clobber .hydra/.
    CMD+=('hydra.run.dir=logs/experiments/runs/${now:%Y-%m-%d_%H-%M-%S}_'"$tag")
    # Resume: a requeued/resubmitted job continues from last.ckpt instead of
    # burning the walltime re-training from scratch.
    local last_ckpt="${CKPT_DIR}/${tag}/last.ckpt"
    [[ "$RESUME" == "1" && -f "$last_ckpt" ]] && CMD+=("ckpt_path=$last_ckpt")
    [[ -n "$LIMIT_TRAIN_BATCHES" ]] && CMD+=("+trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES")
    [[ -n "$VAL_CHECK_INTERVAL" ]] && CMD+=("+trainer.val_check_interval=$VAL_CHECK_INTERVAL")
    local vf; vf="$(vis_flags "$ENCODER")"
    [[ -n "$vf" ]] && CMD+=($vf)
    if [[ "$ENCODER" == "vjepa2" && -d "$(vjepa_cache_dir "$ds")" ]]; then
        CMD+=("data.vjepa_cache_dir=$(vjepa_cache_dir "$ds")")
    fi
    # shellcheck disable=SC2206  -- intentional word-split: $ovr is a list of overrides
    CMD+=($ovr)
    local log="logs/slurm/${tag}.log"
    echo ">>> [GPU $gpu] $tag  →  $log"
    echo "    uv run ${CMD[*]}" > "$log"
    CUDA_VISIBLE_DEVICES="$gpu" uv run "${CMD[@]}" >> "$log" 2>&1 &
}

# ---- dispatch: per-GPU work queue (a freed GPU immediately takes the next
# job — waves would leave GPUs idle for the duration of the slowest run) ------
echo ""; echo ">>> dispatching $NJOBS run(s) across $GPUS GPU(s)"
declare -a GPU_PID=() GPU_TAG=()
for (( g=0; g<GPUS; g++ )); do GPU_PID[$g]=""; done
i=0
while :; do
    running=0
    for (( g=0; g<GPUS; g++ )); do
        pid="${GPU_PID[$g]}"
        if [[ -n "$pid" ]]; then
            if kill -0 "$pid" 2>/dev/null; then
                running=$((running+1)); continue
            fi
            wait "$pid"; STATUS["${GPU_TAG[$g]}"]=$?
            GPU_PID[$g]=""
        fi
        if (( i < NJOBS )); then
            launch_job "$i" "$g"
            GPU_PID[$g]="$!"; GPU_TAG[$g]="${J_TAG[$i]}"
            running=$((running+1)); i=$((i+1))
        fi
    done
    (( running == 0 && i >= NJOBS )) && break
    sleep 10
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
    else fail=$((fail+1)); tag="FAIL(rc=$rc)"; fi
    printf "   %-24s %s\n" "$n" "$tag"
done
echo "   ----  ok=$ok fail=$fail"
echo " Results → $RESULTS_DIR/<tag>.json (+ ALL_RESULTS.md)"
echo " Per-run logs → logs/slurm/<tag>.log"
echo "=============================================================="
# Propagate failures to SLURM: without this the job reports COMPLETED even
# when every run failed.
(( fail > 0 )) && exit 1
exit 0
