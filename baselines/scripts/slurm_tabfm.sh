#!/bin/bash
#SBATCH --job-name=t2-tabfm
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

# TabFM v1.0.0 (Google Research) — the SECOND zero-shot tabular FM, run on the
# same flattened (Y, X_cov) table as LightGBM and TabPFN-3. Two arms, mirroring
# Google's own TabArena reporting:
#   CONFIG=plain -> --model tabfm       (default TabFMRegressor)
#   CONFIG=ens   -> --model tabfm_ens   (TabFMRegressor.ensemble(): sqrt feature
#                                        crosses + sqrt SVD feats + NNLS blend)
#
# Filed under tier 2 by request; it is NOT a supervised deep-TS model like the
# rest of that tier. Report it as a tabular FM beside tabpfn.
#
# Differences from scripts/slurm_tabpfn.sh:
#   * dep group `tabfm` is a GIT dependency (TabFM is not on PyPI) — it resolves
#     on the login node only; compute nodes run UV_NO_SYNC=1 like every other arm
#   * weights are UNGATED (plain snapshot_download, no token analogue of
#     TABPFN_TOKEN) but are licensed tabfm-non-commercial-v1.0: research use
#     only, no commercial or production use
#   * they live in the normal HF cache under $HF_HOME, not a bespoke cache dir
#   * POINT PREDICTIONS ONLY — upstream has no regression quantile head, so
#     CRPS / coverage / ECE come out absent for this arm, by design
#
# Usage (submit from the baselines/ directory):
#   sbatch scripts/slurm_tabfm.sh
#   sbatch --export=ALL,CONFIG=ens scripts/slurm_tabfm.sh
#   sbatch --export=ALL,SCENARIO=s1 scripts/slurm_tabfm.sh
#   sbatch --export=ALL,STAGE=lopo scripts/slurm_tabfm.sh
#
# Overrides (env / --export):
#   CONFIG            plain | ens                      (default: plain)
#   SCENARIO          s1 | s2 | s3 | s4                (default: s2 cross-plant)
#   STAGE             main | lopo                      (default: main)
#   SEEDS             seed list                        (default: 42)
#   MAX_CONTEXT_ROWS  in-context rows after subsample  (default: 10000)
#   N_ESTIMATORS      TabFM ensemble members           (default: 32)
#   DATA              path to dataset_all.parquet
#   BATCH_SIZE        run_eval eval batch size         (default: 256)
#   DATASETS          dataset filter for s1/s2/s4      (default: uk_pv)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# `|| true` because `set -e` would otherwise abort the job when .env is absent
# (a bare `[[ ]] && cmd` that tests false is a failing list).
[[ -f .env ]] && source .env || true    # cluster vars (optional; TabFM needs no token)

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

# ---- knobs -----------------------------------------------------------------
CONFIG="${CONFIG:-plain}"
SCENARIO="${SCENARIO:-s2}"
STAGE="${STAGE:-main}"
SEEDS="${SEEDS:-42}"
# 10k, not TabPFN's 100k: TabFM reads the context through N_ESTIMATORS
# separately-transformed views. Both knobs are REPORTED protocol parameters —
# if a job overruns, lower them and say so next to the number; do not retune
# silently.
MAX_CONTEXT_ROWS="${MAX_CONTEXT_ROWS:-10000}"
N_ESTIMATORS="${N_ESTIMATORS:-32}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DATA="${DATA:-${TEAM_SCRATCH}/data_v2/dataset_all.parquet}"
DATASETS="${DATASETS:-uk_pv}"   # cadence must be uniform — see scenario_flags()
# NOTE: do NOT name this GROUPS — that is a bash special variable holding the
# caller's numeric group IDs, and assignments to it are silently ignored, so the
# expansion becomes the user's GIDs and `uv` tries to exec them.
GROUP_FLAGS=(--group tabfm)

case "$CONFIG" in
    plain) MODEL=tabfm ;;
    ens)   MODEL=tabfm_ens ;;
    *)     echo "unknown CONFIG: $CONFIG (expected plain|ens)"; exit 1 ;;
esac

mkdir -p logs/slurm results

echo "============================================================"
echo " Job ID     : ${SLURM_JOB_ID:-local}"
echo " Model      : $MODEL          Config: $CONFIG"
echo " Stage      : $STAGE          Scenario: $SCENARIO"
echo " Data       : $DATA"
echo " Datasets   : $DATASETS"
echo " Context    : $MAX_CONTEXT_ROWS rows   Members: $N_ESTIMATORS   Seeds: $SEEDS"
echo " HF cache   : $HF_HOME"
echo "============================================================"

[[ -f "$DATA" ]] || { echo "ERROR: data not found: $DATA"; exit 1; }

# ---- fail fast on a cold weight cache --------------------------------------
# Offline node: without the snapshot, tabfm_v1_0_0.load() dies inside .fit()
# after we have already paid for the window build. Check it up front.
TABFM_SNAPSHOT="${HF_HOME}/hub/models--google--tabfm-1.0.0-pytorch"
if [[ ! -d "$TABFM_SNAPSHOT" ]]; then
    echo "ERROR: no TabFM snapshot at $TABFM_SNAPSHOT"
    echo "       Run on the LOGIN node first (needs internet):"
    echo "         STAGE=weights bash scripts/precache_login.sh"
    echo "       Weights are ungated but non-commercial (tabfm-non-commercial-v1.0)."
    exit 1
fi

# Ensure the plant split exists (idempotent; committed in configs/splits.json).
uv run "${GROUP_FLAGS[@]}" python -m common.splits --data "$DATA" || true

# scenario id -> extra run_eval flags.
# The dataset filter is NOT optional: run_eval defaults to physical-time windows
# (history_days=14, horizon_hours=6), and WindowDataset refuses a mixed cadence
# (uk_pv 48 steps/day vs goes_pvdaq 96). Every run_eval result in results/ is
# uk_pv-only, including the lightgbm_s2 and tabpfn_s2 rows this arm is compared
# against — so s1/s2/s4 pin $DATASETS on both sides. s3 is cross-dataset by
# definition and is uniform on each side already.
scenario_flags() {
    case "$1" in
        s1) echo "--in-domain --train-datasets $DATASETS --eval-datasets $DATASETS" ;;
        s2) echo "--train-datasets $DATASETS --eval-datasets $DATASETS" ;;
        s3) echo "--train-datasets uk_pv --eval-datasets goes_pvdaq" ;;
        s4) echo "--horizon 48 --eval-stride 48 --train-datasets $DATASETS --eval-datasets $DATASETS" ;;
        *)  echo "--train-datasets $DATASETS --eval-datasets $DATASETS" ;;
    esac
}

MODEL_KWARGS="{\"max_context_rows\": ${MAX_CONTEXT_ROWS}, \"n_estimators\": ${N_ESTIMATORS}}"

case "$STAGE" in
    main)
        echo ""; echo ">>> run_eval [$MODEL / $SCENARIO]"
        uv run "${GROUP_FLAGS[@]}" python run_eval.py \
            --model "$MODEL" \
            --data "$DATA" \
            $(scenario_flags "$SCENARIO") \
            --batch-size "$BATCH_SIZE" \
            --model-kwargs "$MODEL_KWARGS" \
            --tag "$SCENARIO" --seeds $SEEDS
        ;;
    lopo)
        # goes_pvdaq leave-one-plant-out (mandatory, §4.1)
        echo ""; echo ">>> run_eval [$MODEL / goes_pvdaq LOPO]"
        uv run "${GROUP_FLAGS[@]}" python run_eval.py \
            --model "$MODEL" \
            --data "$DATA" \
            --lopo-dataset goes_pvdaq \
            --batch-size "$BATCH_SIZE" \
            --model-kwargs "$MODEL_KWARGS" \
            --tag "${TAG:-lopo}" --seeds $SEEDS
        ;;
    *)
        echo "unknown STAGE: $STAGE"; exit 1 ;;
esac

echo ""; echo "✓ $MODEL stage '$STAGE' done → results/"
