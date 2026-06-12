#!/bin/bash
#SBATCH --job-name=eval-compare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Compare three models on the SKIPPD held-out test set:
#
#   stage2a  — visual alignment (late fusion, freeze_chronos=true)
#              checkpoint: vision_chronos2_timeselfattn trained with fusion_mode=late
#
#   stage2b  — cross-modal alignment (interleaved fusion, freeze_chronos=true)
#              checkpoint: resumed from stage2a, fusion_mode=interleaved
#
#   base     — pretrained amazon/chronos-2 zero-shot (no fine-tuning, no vision)
#
# SKIPPD test split is never seen during training (temporal hold-out).
#
# Usage:
#   sbatch scripts/slurm_eval_compare.sh
#
# Required overrides (no automatic checkpoint discovery — set these):
#   STAGE2A_CKPT   absolute path to the best stage-2a .ckpt
#   STAGE2B_CKPT   absolute path to the best stage-2b .ckpt
#
# Optional overrides (via --export or environment):
#   RUN_STAGES     space-separated subset of: 2a 2b base  (default: "2a 2b base")
#   DATASET        space-separated dataset names (default: skippd)
#   EVAL_OUT       base output dir  (sub-dirs stage2a/, stage2b/, base_chronos2/)
#   SPLIT          test (default) | val
#   HORIZON        forecast horizon (default 12)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source .env                      # WANDB_API_KEY and cluster-specific vars

# ---- offline mode (compute nodes have no internet) -------------------------
export WANDB_MODE=offline
export UV_OFFLINE=1
export UV_NO_SYNC=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ---- paths -----------------------------------------------------------------
# slurm_train_vjepa.sh writes checkpoints to ${TEAM_SCRATCH}/checkpoints/vjepa_proposal.
# Stage 2a and 2b use the same CKPT_DIR by default, so you must supply distinct
# paths for STAGE2A_CKPT and STAGE2B_CKPT — there is no automatic naming.
TEAM_SCRATCH="/leonardo_scratch/fast/IscrC_MTSFM"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data}"
CKPT_BASE="${CKPT_BASE:-${TEAM_SCRATCH}/checkpoints/vjepa_proposal}"

# Must point to the best checkpoint saved by each training run.
# Example: $(ls -t ${CKPT_BASE}/epoch=*.ckpt | head -1) if you only have one ckpt dir.
STAGE2A_CKPT="${STAGE2A_CKPT:-${CKPT_BASE}/stage2a/best.ckpt}"
STAGE2B_CKPT="${STAGE2B_CKPT:-${CKPT_BASE}/stage2b/best.ckpt}"
BASE_CHRONOS_MODEL="${BASE_CHRONOS_MODEL:-amazon/chronos-2}"

EVAL_OUT="${EVAL_OUT:-${CKPT_BASE}/eval_compare}"
RUN_STAGES="${RUN_STAGES:-2a 2b base}"   # subset: "2a", "2b", "base", or any combo
# Evaluate on SKIPPD test split only (held-out, never seen during training).
DATASET="${DATASET:-skippd}"
SPLIT="${SPLIT:-test}"
HORIZON="${HORIZON:-12}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_ENTITIES="${NUM_ENTITIES:-10}"
HIST_STEPS="${HIST_STEPS:-24}"
N_SAMPLES="${N_SAMPLES:-12}"

# ---- torch / HF cache dirs (pre-populated by login_node_setup.sh) ----------
# V-JEPA 2.1 weights are loaded via torch.hub — cached under TORCH_HOME/hub/
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME}/hub}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
export HYDRA_FULL_ERROR=1

# ---- validate assets (only for stages being run) ---------------------------
[[ "$RUN_STAGES" == *"2a"* ]] && { [[ -f "$STAGE2A_CKPT" ]] || { echo "ERROR: STAGE2A_CKPT not found: $STAGE2A_CKPT"; exit 1; }; }
[[ "$RUN_STAGES" == *"2b"* ]] && { [[ -f "$STAGE2B_CKPT" ]] || { echo "ERROR: STAGE2B_CKPT not found: $STAGE2B_CKPT"; exit 1; }; }

mkdir -p logs/slurm "$EVAL_OUT"

echo "============================================================"
echo " Job ID     : $SLURM_JOB_ID"
echo " Stage 2a   : $STAGE2A_CKPT"
echo " Stage 2b   : $STAGE2B_CKPT"
echo " Base model : $BASE_CHRONOS_MODEL"
echo " Dataset    : $DATASET  (split=$SPLIT, horizon=$HORIZON)"
echo " Base outdir: $EVAL_OUT"
echo "============================================================"

# ---- helper: run evaluate.py for one (run_tag, ckpt_args...) x dataset loop
run_eval() {
    local tag="$1"; shift   # e.g. "stage2a"
    local -a ckpt_args=()
    # consume key=value pairs until "--" sentinel
    while [[ "$1" != "--" ]]; do
        ckpt_args+=("$1"); shift
    done
    shift  # drop "--"

    for d in $DATASET; do
        local ds_out="${EVAL_OUT}/${tag}/${d}"
        mkdir -p "$ds_out"
        echo "  dataset: $d  →  $ds_out"
        uv run python scripts/evaluate.py \
            "${ckpt_args[@]}"            \
            --data-dir     "$DATA_DIR"   \
            --dataset      "$d"          \
            --split        "$SPLIT"      \
            --horizon      "$HORIZON"    \
            --batch-size   "$BATCH_SIZE" \
            --num-workers  "$NUM_WORKERS" \
            --num-entities "$NUM_ENTITIES" \
            --hist-steps   "$HIST_STEPS" \
            --n-samples    "$N_SAMPLES"  \
            --output-dir   "$ds_out"
    done
}

# ---- 1 / 3 — Stage 2a: visual alignment (late fusion) ----------------------
if [[ "$RUN_STAGES" == *"2a"* ]]; then
    echo ""
    echo ">>> [2a] Stage 2a — late fusion (freeze_chronos=true)"
    run_eval "stage2a" --ckpt "$STAGE2A_CKPT" --
fi

# ---- 2 / 3 — Stage 2b: cross-modal alignment (interleaved fusion) ----------
if [[ "$RUN_STAGES" == *"2b"* ]]; then
    echo ""
    echo ">>> [2b] Stage 2b — interleaved fusion (freeze_chronos=true)"
    run_eval "stage2b" --ckpt "$STAGE2B_CKPT" --
fi

# ---- 3 / 3 — Base Chronos-2 (pretrained, zero-shot, no vision) -------------
if [[ "$RUN_STAGES" == *"base"* ]]; then
    echo ""
    echo ">>> [base] Base Chronos-2 — pretrained zero-shot ($BASE_CHRONOS_MODEL)"
    run_eval "base_chronos2" \
        --base-chronos2 --pretrained-model "$BASE_CHRONOS_MODEL" --
fi

# ---- Comparison table -------------------------------------------------------
echo ""
echo ">>> Comparison table"
export EVAL_OUT
uv run python - <<'PYEOF'
import json, os, math
from pathlib import Path

eval_out = Path(os.environ["EVAL_OUT"])

RUNS = [
    ("stage2a  (late fusion)",       eval_out / "stage2a"),
    ("stage2b  (interleaved fusion)", eval_out / "stage2b"),
    ("base_chronos2 (zero-shot)",    eval_out / "base_chronos2"),
]
METRICS = ["mse", "rmse", "mae", "mase", "smape", "crps"]
W = 74

datasets = sorted(
    p.name for p in (eval_out / "stage2a").iterdir() if p.is_dir()
)

for ds in datasets:
    print(f"\n{'='*W}")
    print(f"  Dataset : {ds}")
    print(f"{'─'*W}")
    hdr = f"  {'Model':<33}" + "".join(f"  {m.upper():>7}" for m in METRICS)
    print(hdr)
    print(f"{'─'*W}")
    for run_name, run_dir in RUNS:
        mj = run_dir / ds / "metrics.json"
        if not mj.exists():
            print(f"  {run_name:<33}  (missing)")
            continue
        m = json.loads(mj.read_text())
        row = f"  {run_name:<33}" + "".join(
            f"  {m.get(k, float('nan')):>7.4f}" for k in METRICS
        )
        print(row)

print(f"\n{'='*W}")
PYEOF

echo ""
echo "✓ All outputs written to: $EVAL_OUT"

# Example usage:
#
# All three (default):
#   sbatch scripts/slurm_eval_compare.sh
#
# Base Chronos-2 only (no checkpoint needed):
#   sbatch --export=ALL,RUN_STAGES=base scripts/slurm_eval_compare.sh
#
# Stage 2a only:
#   sbatch --export=ALL,RUN_STAGES=2a,STAGE2A_CKPT=/path/to/2a.ckpt \
#          scripts/slurm_eval_compare.sh
#
# Stages 2a + 2b (skip base):
#   sbatch --export=ALL,RUN_STAGES="2a 2b",STAGE2A_CKPT=...,STAGE2B_CKPT=... \
#          scripts/slurm_eval_compare.sh
