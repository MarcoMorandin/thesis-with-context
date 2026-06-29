#!/bin/bash
#SBATCH --job-name=run-all-mmtsfm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# =============================================================================
# MMTSFM OFFLINE ORCHESTRATOR — run on a GPU node (no internet).
# =============================================================================
# Trains + tests the protocol-aligned MMTSFM model (BASELINE_PROTOCOL.md) on each
# requested dataset of record and writes NMAE/NRMSE/Skill-Score into
# baselines/results (the baselines results schema), so aggregate_all.py lists
# MMTSFM next to every baseline.
#
# Prereq: scripts/precache_login.sh has run on the login node (uv env, V-JEPA 2.1
# + Chronos-2 weights, data staged to $DATA_DIR).
#
#   sbatch scripts/run_all_mmtsfm.sh
#   DATASETS="uk_pv goes_pvdaq" sbatch scripts/run_all_mmtsfm.sh
#   ENCODER=skip RUN_NUMERIC_SANITY=0 sbatch scripts/run_all_mmtsfm.sh   # numeric+cov only
#   MAX_EPOCHS=5 bash scripts/run_all_mmtsfm.sh                          # interactive node
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
ENCODER="${ENCODER:-vjepa2}"           # vjepa2 | vidtok | skip
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
RUN_NUMERIC_SANITY="${RUN_NUMERIC_SANITY:-1}"   # 1 → quick numeric+cov pre-run (encoder off)
SANITY_EPOCHS="${SANITY_EPOCHS:-3}"
AGGREGATE="${AGGREGATE:-1}"            # 1 → refresh baselines/results/ALL_RESULTS at the end

[[ -f "$DATA" ]] || { echo "FATAL: DATA not found: $DATA (run precache_login.sh)"; exit 1; }
mkdir -p logs/slurm "$CKPT_DIR" "$RESULTS_DIR"

echo "=============================================================="
echo " RUN ALL MMTSFM   job=${SLURM_JOB_ID:-local}"
echo " DATASETS=$DATASETS   ENCODER=$ENCODER   epochs=$MAX_EPOCHS"
echo " DATA_DIR=$DATA_DIR   RESULTS_DIR=$RESULTS_DIR"
echo "=============================================================="

declare -A STATUS=()

# dataset → hydra data-config group
data_cfg() { case "$1" in uk_pv) echo ukpv;; goes_pvdaq) echo goespvdaq;; *) echo "";; esac; }
# dataset → short tag / Skill-Score reference
short()    { case "$1" in uk_pv) echo ukpv;; goes_pvdaq) echo goes;; *) echo "$1";; esac; }
sp_ref()   { case "$1" in uk_pv) echo "$SP_REF_UKPV";; *) echo "";; esac; }
# encoder → hydra vision override(s)
vis_flags() {
    case "$1" in
        vjepa2) echo "model.vision_cfg.visual_encoder_type=vjepa2" ;;
        vidtok) echo "model.vision_cfg.visual_encoder_type=vidtok" ;;
        skip)   echo "model.vision_cfg.skip_vision_stack=true" ;;
        *)      echo "" ;;
    esac
}

run_one() {  # run_one <dataset> <encoder> <epochs> <tag>
    local ds="$1" enc="$2" ep="$3" tag="$4"
    local dcfg; dcfg="$(data_cfg "$ds")"
    [[ -n "$dcfg" ]] || { echo "  SKIP $tag — unknown dataset '$ds'"; STATUS[$tag]=skip; return; }
    local ref; ref="$(sp_ref "$ds")"
    echo ""; echo ">>> [$tag] dataset=$ds encoder=$enc epochs=$ep"
    local -a CMD=(
        python -m mmtsfm.train
        "data=$dcfg" trainer=slurm "seed=$SEED"
        "trainer.max_epochs=$ep"
        "trainer.default_root_dir=${CKPT_DIR}/mmtsfm_${tag}"
        "data.data_dir=$DATA_DIR"
        "data.batch_size=$BATCH_SIZE" "data.num_workers=$NUM_WORKERS"
        "model.results_dir=$RESULTS_DIR" "model.results_tag=$tag"
    )
    [[ -n "$ref" && -f "$ref" ]] && CMD+=("model.sp_reference_path=$ref")
    local vf; vf="$(vis_flags "$enc")"
    [[ -n "$vf" ]] && CMD+=($vf)
    echo "    uv run ${CMD[*]}"
    uv run "${CMD[@]}"
    STATUS[$tag]=$?
}

# ---- run each dataset -------------------------------------------------------
for ds in $DATASETS; do
    s="$(short "$ds")"
    if [[ "$RUN_NUMERIC_SANITY" == "1" ]]; then
        run_one "$ds" skip "$SANITY_EPOCHS" "mmtsfm_numeric_s2_${s}"
    fi
    run_one "$ds" "$ENCODER" "$MAX_EPOCHS" "mmtsfm_s2_${s}"
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
echo "=============================================================="
