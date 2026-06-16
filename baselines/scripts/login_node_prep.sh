#!/bin/bash
# RUN ON THE LOGIN NODE (has internet). Prepares every off-repo artifact so the
# compute-node SLURM jobs (slurm_baselines.sh, slurm_rag_original.sh) run FULLY
# OFFLINE — compute nodes on this cluster have no network access.
#
# After this finishes, the compute jobs only read local caches (HF_HOME,
# TORCH_HOME), the exported uk_pv CSVs, and downloaded checkpoints.
#
#   bash scripts/login_node_prep.sh            # prep tier3/4 (run_eval) + RAG originals
#   STAGE=rag  bash scripts/login_node_prep.sh # only the vendored RAG artifacts
#   STAGE=tsfm bash scripts/login_node_prep.sh # only Tier-3/4 run_eval HF models
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-all}"
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${TEAM_SCRATCH}/conda_pkgs}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-${TEAM_SCRATCH}/conda_envs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TEAM_SCRATCH}/pip_cache}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"
DATA="${DATA:-${TEAM_SCRATCH}/data/dataset_all.parquet}"
UKPV_CSV_DIR="${UKPV_CSV_DIR:-${TEAM_SCRATCH}/data/ukpv_rag}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$UKPV_CSV_DIR" "$CKPT_DIR"

echo "HF_HOME=$HF_HOME  UKPV_CSV_DIR=$UKPV_CSV_DIR"

# ---- Tier-3/4 run_eval models (uv venv, group tier3) -----------------------
if [[ "$STAGE" == "all" || "$STAGE" == "tsfm" ]]; then
    echo ">>> sync uv tier3 group (downloads wheels)"
    uv sync --group tier3
    echo ">>> pre-cache the Tier-3 TSFM weights into HF_HOME"
    uv run --group tier3 python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("amazon/chronos-2",
             "google/timesfm-2.5-200m-pytorch",
             "NX-AI/TiRex",
             "ibm-research/ttm-r3"):
    try:
        snapshot_download(repo); print("cached", repo)
    except Exception as e:
        print("WARN could not cache", repo, "->", e)
PY
fi

# ---- vendored TS-RAG / Cross-RAG (separate conda env) ----------------------
if [[ "$STAGE" == "all" || "$STAGE" == "rag" ]]; then
    echo ">>> export uk_pv to the upstream CSV format (baselines venv, pandas only)"
    uv run python tier4/vendor/export_ukpv.py --data "$DATA" --out "$UKPV_CSV_DIR"
    echo ">>> input-contract preflight (offline)"
    uv run python tier4/vendor/contract_check.py --inputs "$UKPV_CSV_DIR"

    echo ">>> pre-cache the Chronos backbones the upstream code loads"
    # zeroshot.py hardcodes amazon/chronos-t5-base for retrieval embeddings;
    # ChronosBolt base weights are passed as a local dir (--pretrained_model_path).
    export UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"
    VENV_NAME="${VENV_NAME:-tsrag}"
    if [[ -d "$UV_ENVS_DIR/$VENV_NAME" ]]; then
        source "$UV_ENVS_DIR/$VENV_NAME/bin/activate"
    else
        echo "WARN: uv env '$VENV_NAME' not found in $UV_ENVS_DIR"
    fi
    python - <<PY || echo "WARN: HF cache step needs the upstream env (huggingface_hub)"
from huggingface_hub import snapshot_download
for repo in ("amazon/chronos-t5-base", "amazon/chronos-bolt-base"):
    try:
        d = snapshot_download(repo); print("cached", repo, "->", d)
    except Exception as e:
        print("WARN could not cache", repo, "->", e)
PY
    echo ""
    echo "NOTE: download the released ARM / cross-attention checkpoints (Google"
    echo "Drive / HF nkh/TS-RAG-Data) into $CKPT_DIR by hand — see VENDOR_NOTICE.md."
    echo "The per-dataset retrieval DB builds itself on first compute-node run"
    echo "(chronos-t5-base is now cached, so do_retrieve works with HF_HUB_OFFLINE=1)."
fi

echo ""
echo "✓ login-node prep done. Compute jobs can now run with HF_HUB_OFFLINE=1."
echo "  sbatch scripts/slurm_baselines.sh"
echo "  sbatch --export=ALL,METHOD=ts_rag,REGIME=orig,VENV_NAME=$VENV_NAME,\\"
echo "         UKPV_CSV_DIR=$UKPV_CSV_DIR,BASE_CKPT=<chronos-bolt-dir>,MIXER_CKPT=<arm.pth> \\"
echo "         scripts/slurm_rag_original.sh"
