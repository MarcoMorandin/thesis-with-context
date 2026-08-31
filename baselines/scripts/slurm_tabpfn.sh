#!/bin/bash
#SBATCH --job-name=t1-tabpfn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Tier-1 TabPFN-3 (Prior Labs, ModelVersion.V3) — tabular FM counterpoint to the
# TSFMs, on the same flattened (Y, X_cov) feature table as LightGBM.
#
# Why this needs its own script instead of riding along with the other tiers:
#   * optional dep group `tabpfn` (not in the base env) — needs `--group tabpfn`
#   * gated weights: the V3 ckpt is downloaded ONCE on the login node by
#     scripts/precache_login.sh into TABPFN_MODEL_CACHE_DIR. Compute nodes are
#     offline, so a cache miss here is fatal, not a slow path — we fail fast.
#   * GPU-bound at the context sizes we use: TabPFN re-encodes the whole
#     in-context training table on every predict() call, and refuses large
#     datasets on CPU unless TABPFN_ALLOW_CPU_LARGE_DATASET=1.
#
# Usage (submit from the baselines/ directory):
#   sbatch scripts/slurm_tabpfn.sh
#   sbatch --export=ALL,SCENARIO=s1 scripts/slurm_tabpfn.sh
#   sbatch --export=ALL,MAX_CONTEXT_ROWS=50000 scripts/slurm_tabpfn.sh
#   sbatch --export=ALL,STAGE=lopo scripts/slurm_tabpfn.sh
#
# Overrides (env / --export):
#   SCENARIO          s1 | s2 | s3 | s4                (default: s2 cross-plant)
#   STAGE             main | lopo                      (default: main)
#   SEEDS             seed list                        (default: 42)
#   MAX_CONTEXT_ROWS  in-context rows after subsample  (default: 100000)
#   DATA              path to dataset_all.parquet
#   BATCH_SIZE        run_eval eval batch size         (default: 256)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

[[ -f .env ]] && source .env    # TABPFN_TOKEN, cluster vars (optional)

# ---- offline mode (compute nodes have no internet) -------------------------
export WANDB_MODE=offline
export UV_OFFLINE=1
export UV_NO_SYNC=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---- caches (pre-populated on the login node by precache_login.sh) ---------
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TEAM_SCRATCH}/pip_cache}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${TEAM_SCRATCH}/weights}"
# Must match precache_login.sh exactly: tabpfn resolves its ckpt relative to this
# (pydantic-settings field `model_cache_dir`, env prefix TABPFN_). A mismatch
# sends it down the download path, which cannot succeed offline.
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${WEIGHTS_DIR}/tabpfn}"

# ---- knobs -----------------------------------------------------------------
SCENARIO="${SCENARIO:-s2}"
STAGE="${STAGE:-main}"
SEEDS="${SEEDS:-42}"
MAX_CONTEXT_ROWS="${MAX_CONTEXT_ROWS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
# NOTE: do NOT name this GROUPS — that is a bash special variable holding the
# caller's numeric group IDs, and assignments to it are silently ignored, so the
# expansion becomes the user's GIDs and `uv` tries to exec them.
GROUP_FLAGS=(--group tier3 --group tabpfn)

mkdir -p logs/slurm results

echo "============================================================"
echo " Job ID     : ${SLURM_JOB_ID:-local}"
echo " Stage      : $STAGE        Scenario: $SCENARIO"
echo " Data       : $DATA"
echo " Context    : $MAX_CONTEXT_ROWS rows      Seeds: $SEEDS"
echo " TabPFN cache: $TABPFN_MODEL_CACHE_DIR"
echo "============================================================"

[[ -f "$DATA" ]] || { echo "ERROR: data not found: $DATA"; exit 1; }

# ---- fail fast on a cold weight cache --------------------------------------
# Offline node: if the V3 regressor ckpt is not already here, tabpfn will try to
# hit the HF gate and die deep inside .fit() after we have already paid for the
# window build. Check it up front instead.
if ! find "$TABPFN_MODEL_CACHE_DIR" -iname '*tabpfn-v3*regressor*.ckpt' 2>/dev/null | grep -q .; then
    echo "ERROR: no TabPFN-3 regressor ckpt under $TABPFN_MODEL_CACHE_DIR"
    echo "       Run on the LOGIN node first (needs internet + TABPFN_TOKEN):"
    echo "         STAGE=weights bash scripts/precache_login.sh"
    echo "       License gate: https://ux.priorlabs.ai"
    exit 1
fi

# Ensure the plant split exists (idempotent; committed in configs/splits.json).
uv run "${GROUP_FLAGS[@]}" python -m common.splits --data "$DATA" || true

# scenario id -> extra run_eval flags (same mapping as the other tier scripts)
scenario_flags() {
    case "$1" in
        s1) echo "--in-domain" ;;
        s2) echo "" ;;                                   # default cross-plant
        s3) echo "--train-datasets uk_pv --eval-datasets goes_pvdaq" ;;
        s4) echo "--horizon 48 --eval-stride 48" ;;
        *)  echo "" ;;
    esac
}

MODEL_KWARGS="{\"max_context_rows\": ${MAX_CONTEXT_ROWS}}"

case "$STAGE" in
    main)
        echo ""; echo ">>> run_eval [tabpfn / $SCENARIO]"
        uv run "${GROUP_FLAGS[@]}" python run_eval.py \
            --model tabpfn \
            --data "$DATA" \
            $(scenario_flags "$SCENARIO") \
            --batch-size "$BATCH_SIZE" \
            --model-kwargs "$MODEL_KWARGS" \
            --tag "$SCENARIO" --seeds $SEEDS
        ;;
    lopo)
        # goes_pvdaq leave-one-plant-out (mandatory, §4.1)
        echo ""; echo ">>> run_eval [tabpfn / goes_pvdaq LOPO]"
        uv run "${GROUP_FLAGS[@]}" python run_eval.py \
            --model tabpfn \
            --data "$DATA" \
            --lopo-dataset goes_pvdaq \
            --batch-size "$BATCH_SIZE" \
            --model-kwargs "$MODEL_KWARGS" \
            --tag "$SCENARIO" --seeds $SEEDS
        ;;
    *)
        echo "unknown STAGE: $STAGE"; exit 1 ;;
esac

echo ""; echo "✓ tabpfn stage '$STAGE' done → results/"
