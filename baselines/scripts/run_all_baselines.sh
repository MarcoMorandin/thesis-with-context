#!/bin/bash
#SBATCH --job-name=run-all-baselines
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=1-00:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# =============================================================================
# MASTER OFFLINE ORCHESTRATOR — run ONCE on a GPU node (no internet).
# =============================================================================
# Runs EVERY baseline (Tiers 0-6) on the dataset of record, parallelizing
# independent train/eval jobs across the node's GPUs (one job per GPU), then
# aggregates all results/*.json into ONE file: results/ALL_RESULTS.{md,json}.
#
# Prereq: scripts/precache_login.sh has run on the login node (weights, conda
# envs, uk_pv export) and the data is staged to $DATA / $IMAGES_H5.
#
#   sbatch scripts/run_all_baselines.sh
#   sbatch --export=ALL,RUN_LOPO=1 scripts/run_all_baselines.sh   # + goes LOPO
#   NUM_GPUS=4 bash scripts/run_all_baselines.sh                  # interactive node
#
# Each baseline is GATED on its env/weights: anything not prepared is SKIPPED
# (logged), so the single run produces as complete a table as the cache allows.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

[[ -f .env ]] && source .env
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1            # compute node has no network
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- config (override via --export / env) ----------------------------------
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${TEAM_SCRATCH}/conda_pkgs}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-${TEAM_SCRATCH}/conda_envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TEAM_SCRATCH}/pip_cache}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data}"
DATA="${DATA:-${DATA_DIR}/dataset_all.parquet}"
IMAGES_H5="${IMAGES_H5:-${DATA_DIR}/images_all.h5}"
UKPV_CSV_DIR="${UKPV_CSV_DIR:-${DATA_DIR}/ukpv_rag}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${TEAM_SCRATCH}/weights}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints}"
SEEDS="${SEEDS:-42 43 44}"
GROUP="${GROUP:-tier3}"
RUN_LOPO="${RUN_LOPO:-0}"

# vendored-model uv envs
ENV_TIMEVLM="${ENV_TIMEVLM:-timevlm}"; ENV_VISIONTS="${ENV_VISIONTS:-visionts}"
ENV_UNICAST="${ENV_UNICAST:-unicast}"; ENV_AURORA="${ENV_AURORA:-aurora}"
ENV_CROSSVIVIT="${ENV_CROSSVIVIT:-crossvivit}"; ENV_SUNSET="${ENV_SUNSET:-sunset}"
ENV_TSRAG="${ENV_TSRAG:-tsrag}"; ENV_CROSSRAG="${ENV_CROSSRAG:-crossrag}"
# gated backbone weights (defaults match precache_login.sh)
MAE_CKPT="${MAE_CKPT:-}"                                   # set to the VisionTS++ .ckpt file
VISION_MODEL_PATH="${VISION_MODEL_PATH:-${WEIGHTS_DIR}/clip-vit-base-patch32}"
CHRONOS_PATH="${CHRONOS_PATH:-${WEIGHTS_DIR}/chronos-bolt-base}"
AURORA_CKPT="${AURORA_CKPT:-}"
RAG_BASE_CKPT="${RAG_BASE_CKPT:-${WEIGHTS_DIR}/chronos-bolt-base}"
RAG_MIXER_CKPT="${RAG_MIXER_CKPT:-}"                       # released ARM/cross-attn ckpt
SOLARVLM_DIR="${SOLARVLM_DIR:-}"

# ---- GPU count -------------------------------------------------------------
if [[ -n "${NUM_GPUS:-}" ]]; then :;
elif command -v nvidia-smi >/dev/null 2>&1; then NUM_GPUS=$(nvidia-smi -L | wc -l);
else NUM_GPUS=8; fi
(( NUM_GPUS > 0 )) || NUM_GPUS=1

LOGDIR="logs/orchestrator/${SLURM_JOB_ID:-local}"
mkdir -p "$LOGDIR" results logs/slurm
[[ -f "$DATA" ]] || { echo "FATAL: DATA not found: $DATA"; exit 1; }

echo "=============================================================="
echo " RUN ALL BASELINES   job=${SLURM_JOB_ID:-local}  GPUs=$NUM_GPUS"
echo " DATA=$DATA"
echo " IMAGES_H5=$IMAGES_H5   logs=$LOGDIR"
echo "=============================================================="

# ---- uv envs ------------------------------------------------------------------
export UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"
env_has() { [[ -d "$UV_ENVS_DIR/$1" ]]; }

# ---- task table + gating ----------------------------------------------------
declare -a T_NAME=() T_CMD=()
declare -A SKIP=()
declare -A STATUS=()
add()  { T_NAME+=("$1"); T_CMD+=("$2"); }
skip() { SKIP["$1"]="$2"; echo "  SKIP $1 — $2"; }

echo ">>> planning tasks"

# Tier 3 zero-shot (uv venv, 1 GPU each, unique tag avoids reference races)
for m in chronos2_zs timesfm_zs tirex_zs ttm_zs; do
    add "$m" "uv run --group $GROUP python run_eval.py --model $m --data '$DATA' --tag s2_$m"
done
# Tier 3/4 trained (3 seeds inside one invocation)
for m in chronos2_ft ttm_ft cora; do
    add "$m" "uv run --group $GROUP python run_eval.py --model $m --data '$DATA' --tag s2_$m --seeds $SEEDS"
done
# Optional goes leave-one-plant-out (heavy; §4.1)
if [[ "$RUN_LOPO" == 1 ]]; then
    add "lopo_goes" "uv run --group $GROUP python run_eval.py --model chronos2_zs timesfm_zs tirex_zs ttm_zs chronos2_ft ttm_ft cora --data '$DATA' --lopo-dataset goes_pvdaq --tag lopo --seeds $SEEDS"
fi

# Tier 4 RAG originals (own uv env + gated ckpts)
if env_has "$ENV_TSRAG" && [[ -d "$UKPV_CSV_DIR" && -d "$RAG_BASE_CKPT" && -f "$RAG_MIXER_CKPT" ]]; then
    add "ts_rag" "METHOD=ts_rag REGIME=orig VENV_NAME=$ENV_TSRAG UKPV_CSV_DIR='$UKPV_CSV_DIR' BASE_CKPT='$RAG_BASE_CKPT' MIXER_CKPT='$RAG_MIXER_CKPT' bash scripts/slurm_rag_original.sh"
else skip "ts_rag" "needs uv env:$ENV_TSRAG + UKPV_CSV_DIR + RAG_BASE_CKPT + RAG_MIXER_CKPT"; fi
if env_has "$ENV_CROSSRAG" && [[ -d "$UKPV_CSV_DIR" && -d "$RAG_BASE_CKPT" && -f "$RAG_MIXER_CKPT" ]]; then
    add "cross_rag" "METHOD=cross_rag REGIME=orig VENV_NAME=$ENV_CROSSRAG UKPV_CSV_DIR='$UKPV_CSV_DIR' BASE_CKPT='$RAG_BASE_CKPT' MIXER_CKPT='$RAG_MIXER_CKPT' bash scripts/slurm_rag_original.sh"
else skip "cross_rag" "needs uv env:$ENV_CROSSRAG + UKPV_CSV_DIR + RAG_BASE_CKPT + RAG_MIXER_CKPT"; fi

# Tier 5 (own uv env; private UKPV_DIR per task avoids export races)
if env_has "$ENV_TIMEVLM"; then
    add "time_vlm" "VENV_NAME=$ENV_TIMEVLM DATA='$DATA' UKPV_DIR='${UKPV_CSV_DIR}_tvlm' bash scripts/slurm_time_vlm.sh"
else skip "time_vlm" "needs uv env:$ENV_TIMEVLM"; fi
if env_has "$ENV_VISIONTS" && [[ -f "$MAE_CKPT" ]]; then
    add "visionts_pp" "VENV_NAME=$ENV_VISIONTS MAE_CKPT='$MAE_CKPT' DATA='$DATA' UKPV_DIR='${UKPV_CSV_DIR}_vts' bash scripts/slurm_visionts_pp.sh"
else skip "visionts_pp" "needs uv env:$ENV_VISIONTS + MAE_CKPT (set to VisionTS++ .ckpt)"; fi
if env_has "$ENV_UNICAST" && [[ -d "$VISION_MODEL_PATH" && -d "$CHRONOS_PATH" && -f "$IMAGES_H5" ]]; then
    add "unicast" "VENV_NAME=$ENV_UNICAST VISION_MODEL=CLIP VISION_MODEL_PATH='$VISION_MODEL_PATH' CHRONOS_PATH='$CHRONOS_PATH' DATA='$DATA' IMAGES_H5='$IMAGES_H5' bash scripts/slurm_unicast.sh"
else skip "unicast" "needs uv env:$ENV_UNICAST + VISION_MODEL_PATH + CHRONOS_PATH + IMAGES_H5"; fi
if env_has "$ENV_AURORA" && [[ -d "$AURORA_CKPT" ]]; then
    add "aurora" "VENV_NAME=$ENV_AURORA AURORA_CKPT='$AURORA_CKPT' DATA='$DATA' IMAGES_H5='$IMAGES_H5' MODE=finetune bash scripts/slurm_aurora.sh"
else skip "aurora" "needs uv env:$ENV_AURORA + AURORA_CKPT dir"; fi

# Tier 6 (own uv env)
if env_has "$ENV_CROSSVIVIT" && [[ -f "$IMAGES_H5" ]]; then
    add "crossvivit" "VENV_NAME=$ENV_CROSSVIVIT DATA='$DATA' IMAGES_H5='$IMAGES_H5' bash scripts/slurm_crossvivit.sh"
else skip "crossvivit" "needs uv env:$ENV_CROSSVIVIT + IMAGES_H5"; fi
if env_has "$ENV_SUNSET" && [[ -f "$IMAGES_H5" ]]; then
    add "sunset" "VENV_NAME=$ENV_SUNSET DATA='$DATA' IMAGES_H5='$IMAGES_H5' bash scripts/slurm_sunset.sh"
else skip "sunset" "needs uv env:$ENV_SUNSET + IMAGES_H5"; fi
if [[ -n "$SOLARVLM_DIR" && -d "$SOLARVLM_DIR" ]]; then
    add "solar_vlm" "SOLARVLM_DIR='$SOLARVLM_DIR' bash scripts/slurm_solar_vlm.sh"
else skip "solar_vlm" "set SOLARVLM_DIR to the Solar-VLM repo"; fi

echo ">>> ${#T_NAME[@]} GPU tasks queued, ${#SKIP[@]} skipped"

# ---- Phase A: splits + Tier 0-2 reference (CPU, sequential, FIRST) ----------
# run_eval always (re)writes smart_persistence as the SS reference; doing the
# whole reference + Tier-0/1/2 set once up front gives the canonical
# smart_persistence_s2.json that the Tier-5/6 importers point at.
echo ""; echo ">>> Phase A: splits + Tier 0-2 reference (CPU)"
uv run --group "$GROUP" python -m common.splits --data "$DATA" || true
( CUDA_VISIBLE_DEVICES="" uv run --group "$GROUP" python run_eval.py \
    --model persistence smart_persistence climatology_hourly seasonal_naive \
            lightgbm mlp dlinear patchtst itransformer tft \
    --data "$DATA" --tag s2 --seeds $SEEDS ) > "$LOGDIR/tier0_2_cpu.log" 2>&1 &
CPU_PID=$!
# uk_pv CSV export the RAG originals read (single canonical copy)
uv run python tier4/vendor/export_ukpv.py --data "$DATA" --out "$UKPV_CSV_DIR" \
    > "$LOGDIR/export_ukpv.log" 2>&1 || true
wait "$CPU_PID" && STATUS["tier0_2_cpu"]=0 || STATUS["tier0_2_cpu"]=$?
# Tier-5/6 importers hard-reference results/smart_persistence_s2_ukpv.json
[[ -f results/smart_persistence_s2.json ]] && cp -f results/smart_persistence_s2.json results/smart_persistence_s2_ukpv.json
echo "    Phase A done (rc=${STATUS[tier0_2_cpu]})"

# ---- Phase B: GPU pool ------------------------------------------------------
echo ""; echo ">>> Phase B: GPU pool ($NUM_GPUS slots)"
declare -a FREE=(); for ((g=0; g<NUM_GPUS; g++)); do FREE+=("$g"); done
declare -A PID2GPU=() PID2NAME=()

launch() {  # launch <name> <cmd>
    local name="$1" cmd="$2" gpu="${FREE[0]}"
    FREE=("${FREE[@]:1}")
    ( export CUDA_VISIBLE_DEVICES="$gpu"
      echo "[$(date +%T)] START $name on GPU $gpu"
      eval "$cmd"
      echo "[$(date +%T)] END $name rc=$?" ) > "$LOGDIR/${name}.log" 2>&1 &
    local pid=$!; PID2GPU[$pid]="$gpu"; PID2NAME[$pid]="$name"
    echo "  launch $name → GPU $gpu (pid $pid)"
}
reap() {
    for pid in "${!PID2GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; STATUS["${PID2NAME[$pid]}"]=$?
            FREE+=("${PID2GPU[$pid]}")
            echo "  done ${PID2NAME[$pid]} (GPU ${PID2GPU[$pid]}, rc=${STATUS[${PID2NAME[$pid]}]})"
            unset 'PID2GPU[$pid]' 'PID2NAME[$pid]'
        fi
    done
}

i=0
while (( i < ${#T_NAME[@]} )) || (( ${#PID2GPU[@]} > 0 )); do
    while (( ${#FREE[@]} > 0 )) && (( i < ${#T_NAME[@]} )); do
        launch "${T_NAME[$i]}" "${T_CMD[$i]}"; ((i++))
    done
    if (( ${#PID2GPU[@]} > 0 )); then
        wait -n 2>/dev/null || true
        reap
    fi
done
echo "    Phase B done"

# ---- Phase C: single results file ------------------------------------------
echo ""; echo ">>> Phase C: aggregate → results/ALL_RESULTS.{md,json}"
uv run --group "$GROUP" python scripts/aggregate_all.py \
    --results results --md results/ALL_RESULTS.md --json results/ALL_RESULTS.json || true
# also refresh the per-track tables (best-effort)
uv run --group "$GROUP" python scripts/summarize_ukpv.py 2>/dev/null || true
uv run --group "$GROUP" python scripts/make_tables.py --results results --out results/tables_s2.md 2>/dev/null || true

# ---- run summary ------------------------------------------------------------
echo ""; echo "=============================================================="
echo " RUN SUMMARY"
ok=0; fail=0
for n in "tier0_2_cpu" "${T_NAME[@]}"; do
    rc="${STATUS[$n]:-?}"; [[ "$rc" == 0 ]] && { ok=$((ok+1)); tag=OK; } || { fail=$((fail+1)); tag="FAIL(rc=$rc)"; }
    printf "   %-14s %s\n" "$n" "$tag"
done
for n in "${!SKIP[@]}"; do printf "   %-14s SKIP (%s)\n" "$n" "${SKIP[$n]}"; done
echo "   ----  ok=$ok fail=$fail skip=${#SKIP[@]}"
echo " Single results file → results/ALL_RESULTS.md (+ .json)"
echo " Per-task logs       → $LOGDIR/<task>.log"
echo "=============================================================="
