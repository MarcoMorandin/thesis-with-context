#!/bin/bash
#SBATCH --job-name=pv-baselines
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --account=IscrC_MTSFM
# Leonardo QOS: 'normal' = up to 24 h. For a quick <30 min smoke override at
# submit time: sbatch --qos=boost_qos_dbg --time=00:30:00 scripts/slurm_baselines.sh
# (boost_qos_dbg caps at 30 min / 2 nodes — never use it with --time>00:30:00).
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Train + test the GPU-bound baselines (Tier 3 TSFMs, Tier 4 adaptation) on the
# numerical track via baselines/run_eval.py. Tiers 0-2 run on a laptop and are
# excluded by default. run_eval does fit+eval in one pass, so "training" and
# "testing" happen in the same invocation for the trained models
# (chronos2_ft, ttm_ft, cora); the ZS models skip fit. TS-RAG / Cross-RAG run
# separately from vendored original code (scripts/slurm_rag_original.sh).
#
# Usage (submit from the baselines/ directory):
#   sbatch scripts/slurm_baselines.sh                       # default plan below
#   sbatch --export=ALL,STAGE=zs   scripts/slurm_baselines.sh
#   sbatch --export=ALL,STAGE=trained scripts/slurm_baselines.sh
#   sbatch --export=ALL,STAGE=lopo scripts/slurm_baselines.sh   # goes_pvdaq LOPO
#   sbatch --export=ALL,MODELS="chronos2_zs cora",SCENARIO=s2 scripts/slurm_baselines.sh
#
# Overrides (env / --export):
#   STAGE     zs | trained | lopo | all          (default: all)
#   MODELS    explicit model list (overrides STAGE's model set)
#   SCENARIO  s2 | s3 | s4 | s1                   (default: s2 cross-plant)
#   SEEDS     trained-model seeds                 (default: "42 43 44")
#   DATA      path to all_curated.parquet
#   GROUP     uv dependency group                 (default: tier3)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

[[ -f .env ]] && source .env    # WANDB_API_KEY, HF token, cluster vars (optional)

# ---- offline mode (compute nodes have no internet) -------------------------
export WANDB_MODE=offline
export UV_OFFLINE=1
export UV_NO_SYNC=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HYDRA_FULL_ERROR=1

# ---- caches (pre-populated on the login node, see login_node_setup) --------
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"

# ---- knobs -----------------------------------------------------------------
STAGE="${STAGE:-all}"
SCENARIO="${SCENARIO:-s2}"
SEEDS="${SEEDS:-42 43 44}"
GROUP="${GROUP:-tier3}"
DATA="${DATA:-${TEAM_SCRATCH}/data/numerical/all_curated.parquet}"

ZS_MODELS="chronos2_zs timesfm_zs tirex_zs ttm_zs"
# ts_rag / cross_rag are NOT here — they run from vendored original code via
# scripts/slurm_rag_original.sh, not through run_eval.
TRAINED_MODELS="chronos2_ft ttm_ft cora"

mkdir -p logs/slurm results

echo "============================================================"
echo " Job ID    : ${SLURM_JOB_ID:-local}"
echo " Stage     : $STAGE        Scenario: $SCENARIO"
echo " Data      : $DATA"
echo " uv group  : $GROUP        Seeds: $SEEDS"
echo "============================================================"
[[ -f "$DATA" ]] || { echo "ERROR: data not found: $DATA"; exit 1; }

# Ensure the plant split exists (idempotent; committed in configs/splits.json).
uv run --group "$GROUP" python -m common.splits --data "$DATA" || true

# scenario id -> extra run_eval flags
scenario_flags() {
    case "$1" in
        s1) echo "--in-domain" ;;
        s2) echo "" ;;                                   # default cross-plant
        s3) echo "--train-datasets uk_pv --eval-datasets goes_pvdaq" ;;
        s4) echo "--horizon 48 --eval-stride 48" ;;
        *)  echo "" ;;
    esac
}

run() {   # run <tag> <seeds-flag> <models...>
    local tag="$1"; shift
    local seedflag="$1"; shift
    echo ""; echo ">>> run_eval [$tag]: $*"
    uv run --group "$GROUP" python run_eval.py \
        --model $* \
        --data "$DATA" \
        $(scenario_flags "$SCENARIO") \
        --tag "$tag" $seedflag
}

# ---- model selection -------------------------------------------------------
if [[ -n "${MODELS:-}" ]]; then
    run "$SCENARIO" "--seeds $SEEDS" $MODELS
    echo "✓ done (explicit MODELS)"; exit 0
fi

case "$STAGE" in
    zs)
        run "$SCENARIO" "" $ZS_MODELS ;;                 # ZS: single deterministic run
    trained)
        run "$SCENARIO" "--seeds $SEEDS" $TRAINED_MODELS ;;
    lopo)
        # goes_pvdaq leave-one-plant-out (mandatory, §4.1): all tiers
        uv run --group "$GROUP" python run_eval.py \
            --model $ZS_MODELS $TRAINED_MODELS \
            --data "$DATA" --lopo-dataset goes_pvdaq --tag s2 --seeds $SEEDS ;;
    all)
        run "$SCENARIO" "" $ZS_MODELS
        run "$SCENARIO" "--seeds $SEEDS" $TRAINED_MODELS ;;
    *)
        echo "unknown STAGE: $STAGE"; exit 1 ;;
esac

# ---- render tables ---------------------------------------------------------
echo ""; echo ">>> make_tables"
uv run --group "$GROUP" python scripts/make_tables.py \
    --results results --out results/tables_${SCENARIO}.md || true

echo ""; echo "✓ baselines stage '$STAGE' done → results/"
