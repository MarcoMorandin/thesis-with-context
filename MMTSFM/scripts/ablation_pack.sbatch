#!/bin/bash
#SBATCH --job-name=mmtsfm-pack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# Mail notifications are injected by scripts/ablation_sweep.sh via
# `sbatch --mail-type/--mail-user`. Never poll the controller (`watch squeue`).
# =============================================================================
# ABLATION PACK — run MANY ablations concurrently inside ONE whole-node job.
# =============================================================================
# A Leonardo Booster node is 4x A100 64 GB / 32 cores. Submitting each ablation
# as its own `--gres=gpu:1` job asks for a quarter node and pays a queue wait per
# ablation; the 15 registered ablations x 3 seeds is 45 separate queue waits.
# This runs a work queue over one whole node instead: four runs at a time, and a
# GPU that frees up immediately takes the next job rather than idling until its
# packmates finish.
#
# NOT a budget saving on its own. Leonardo bills allocated resources, and
# 4x(1 GPU + 8 cores) is the same node-hours as 1x(4 GPUs + 32 cores). The win is
# one scheduling decision instead of four, one env/data warm-up instead of four,
# and no hand-typed sbatch per ablation. The loss to watch for is TAIL IDLE: at
# the end of a pack, finished GPUs sit idle while the last run continues. Give a
# pack many more jobs than GPUs (the sweep script does) so the queue drains flat.
#
# Never submit this by hand — scripts/ablation_sweep.sh writes the job file and
# owns the walltime, the pack slicing and the continuation chain.
#
# Required env (exported by the sweep script):
#   JOB_FILE    TSV job list; one run per line (see COLUMNS below)
#   PACK_START  first line index (0-based, inclusive)
#   PACK_END    last line index (exclusive)
# =============================================================================
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -d src/mmtsfm ]] || { echo "FATAL: submit from MMTSFM/"; exit 1; }

[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}" HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# shellcheck source=lib/stage_cmd.sh
source scripts/lib/stage_cmd.sh

: "${JOB_FILE:?JOB_FILE not set}"
[[ -f "$JOB_FILE" ]] || { echo "FATAL: JOB_FILE not found: ${JOB_FILE}"; exit 1; }
: "${DATA_DIR:?DATA_DIR not set}" "${CKPT_DIR:?CKPT_DIR not set}" "${RESULTS_DIR:?RESULTS_DIR not set}"

# GPUs to saturate. Default = all visible on the node; >=1 fallback for a login
# or CPU box (where DRY_RUN is the only sensible mode anyway).
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[[ "$GPUS" =~ ^[0-9]+$ && "$GPUS" -ge 1 ]] || GPUS=1
# Dataloader workers PER SLOT. 4 slots x 8 would be 32 workers plus 4 main
# processes on 32 cores; 6 leaves headroom so the trainers are not descheduled
# by their own dataloaders.
NUM_WORKERS="${NUM_WORKERS:-6}"
# A results JSON that already exists means the run finished. The continuation
# chain re-submits the SAME job file, so without this every pack in the chain
# would retrain everything the previous pack completed.
SKIP_DONE="${SKIP_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
# Seconds between sweeps of the slot table. 10 s is nothing against a run measured
# in hours; the tests turn it down so the queue drains at test speed.
POLL="${POLL:-10}"

# ---- slice this pack's share of the job list --------------------------------
mapfile -t ALL_JOBS < <(grep -v -e '^\s*#' -e '^\s*$' "$JOB_FILE")
PACK_START="${PACK_START:-0}"
PACK_END="${PACK_END:-${#ALL_JOBS[@]}}"
declare -a JOBS=()
for (( n=PACK_START; n<PACK_END && n<${#ALL_JOBS[@]}; n++ )); do JOBS+=("${ALL_JOBS[$n]}"); done
NJOBS=${#JOBS[@]}
(( NJOBS > 0 )) || { echo "FATAL: pack slice [${PACK_START},${PACK_END}) of ${JOB_FILE} is empty"; exit 1; }

mkdir -p logs/slurm "$CKPT_DIR" "$RESULTS_DIR"
echo "=============================================================="
echo " ABLATION PACK   job=${SLURM_JOB_ID:-local}"
echo " JOB_FILE=${JOB_FILE}  slice=[${PACK_START},${PACK_END})  jobs=${NJOBS}"
echo " GPUS=${GPUS}  NUM_WORKERS=${NUM_WORKERS}/slot  SKIP_DONE=${SKIP_DONE}"
echo "=============================================================="

declare -A STATUS=()

# COLUMNS of one job line (`|`-separated, written by ablation_sweep.sh):
#   1 tag  2 mode  3 stage  4 model_cfg  5 seed  6 ablation  7 init_ckpt  8 score_ckpt
#   9 dcfg 10 ds  11 max_epochs  12 batch  13 accum
# Not tab-separated: tab is an IFS whitespace character, so `read` would collapse
# the empty init_ckpt/score_ckpt column and shift every later field left by one.
#
# launch_job <index> <gpu> — start one run in the background pinned to $gpu.
launch_job() {
  local i="$1" gpu="$2" line="${JOBS[$1]}"
  local tag mode stage model_cfg seed abl init_ckpt score_ckpt dcfg ds epochs batch accum
  IFS='|' read -r tag mode stage model_cfg seed abl init_ckpt score_ckpt \
                  dcfg ds epochs batch accum <<< "$line"

  local log="logs/slurm/${tag}.log"
  # Each slot builds its command in a SUBSHELL: build_stage_cmd reads and writes
  # globals, and four concurrent slots sharing one shell would interleave TAG /
  # STAGE_DIR / CMD and launch four copies of whichever assignment won.
  (
    export TAG="$tag" MODE="$mode" STAGE="$stage" MODEL_CFG="$model_cfg" SEED="$seed"
    export DCFG="$dcfg" DS="$ds" MAX_EPOCHS="$epochs" BATCH_SIZE="$batch" ACCUM="$accum"
    export NUM_WORKERS PREV_CKPT="$init_ckpt" SCORE_CKPT="$score_ckpt"
    # The ablation ID is part of the checkpoint dir as well as the tag: A17 and
    # A22 share a stage, a model config and a seed, so without it they would
    # warm-start into one directory and overwrite each other's best.ckpt.
    export STAGE_DIR="${CKPT_DIR}/${ds}_${stage}$(arm_suffix "$model_cfg" "$seed")_${abl}"
    export EXTRA_OVERRIDES="+ablation=${abl}${EXTRA_OVERRIDES:+ ${EXTRA_OVERRIDES}}"
    # MANDATORY for every ablation: without it the vision-on/off decomposition is
    # absent from the JSON and the ablation cannot be read at all.
    export MARGINAL_GAIN=1
    declare -a CMD=()
    build_stage_cmd || exit 1
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'TAG=%s\nSTAGE=%s\nSTAGE_DIR=%s\nGPU=%s\nMODE=%s\nABLATION=%s\nCMD=%s\n' \
        "$tag" "$stage" "$STAGE_DIR" "$gpu" "$mode" "$abl" "${CMD[*]}"
      exit 0
    fi
    printf 'uv run %s\n' "${CMD[*]}" > "$log"
    CUDA_VISIBLE_DEVICES="$gpu" uv run "${CMD[@]}" >> "$log" 2>&1
  ) &
  echo ">>> [GPU ${gpu}] ${tag}  (${mode}, +ablation=${abl})  →  ${log}"
}

# ---- dispatch: per-GPU work queue -------------------------------------------
# A freed GPU immediately takes the next job. Fixed waves would leave GPUs idle
# for the duration of the slowest run in each wave, which on a 24 h walltime is
# hours of allocated-but-unused node.
declare -a GPU_PID=() GPU_TAG=()
for (( g=0; g<GPUS; g++ )); do GPU_PID[$g]=""; done
i=0
while :; do
  running=0
  for (( g=0; g<GPUS; g++ )); do
    pid="${GPU_PID[$g]}"
    if [[ -n "$pid" ]]; then
      if kill -0 "$pid" 2>/dev/null; then running=$((running+1)); continue; fi
      wait "$pid"; STATUS["${GPU_TAG[$g]}"]=$?
      GPU_PID[$g]=""
    fi
    # Skip past anything already finished before occupying a GPU with it.
    while (( i < NJOBS )); do
      _tag="$(cut -d'|' -f1 <<< "${JOBS[$i]}")"
      if [[ "$SKIP_DONE" == "1" && -f "${RESULTS_DIR}/${_tag}.json" ]]; then
        echo "    skip ${_tag} — ${RESULTS_DIR}/${_tag}.json already exists"
        STATUS["$_tag"]=0; i=$((i+1)); continue
      fi
      break
    done
    if (( i < NJOBS )); then
      launch_job "$i" "$g"
      GPU_PID[$g]="$!"; GPU_TAG[$g]="$(cut -d'|' -f1 <<< "${JOBS[$i]}")"
      running=$((running+1)); i=$((i+1))
    fi
  done
  (( running == 0 && i >= NJOBS )) && break
  sleep "$POLL"
done

# ---- summary ----------------------------------------------------------------
echo ""; echo "=============================================================="
echo " PACK SUMMARY"
ok=0; fail=0
for n in "${!STATUS[@]}"; do
  rc="${STATUS[$n]}"
  if [[ "$rc" == 0 ]]; then ok=$((ok+1)); mark=OK
  else fail=$((fail+1)); mark="FAIL(rc=${rc})"; fi
  printf "   %-40s %s\n" "$n" "$mark"
done
echo "   ----  ok=${ok} fail=${fail}"
echo " Results → ${RESULTS_DIR}/<tag>.json     Logs → logs/slurm/<tag>.log"
echo "=============================================================="
# One failed ablation must not fail the pack for its three packmates, but the
# job itself has to report non-zero or SLURM calls a pack of failures COMPLETED.
(( fail > 0 )) && exit 1
exit 0
