#!/bin/bash
# =============================================================================
# MMTSFM LOGIN-NODE PRECACHE  —  run ONCE on the Leonardo login node (internet).
# =============================================================================
# Prepares every off-repo artifact the offline GPU run (scripts/run_all_mmtsfm.sh)
# needs, so compute nodes run FULLY OFFLINE:
#   1. uv sync   (torch / lightning / transformers / vjepa2 / h5py / pandas;
#                 + the optional `vidtok` group when WITH_VIDTOK=1)
#   2. V-JEPA 2.1 hub weights + Chronos-2 backbone  (delegates to login_node_setup.sh)
#   3. verify the MMTSFM <-> baselines bridge: h5py/pandas, common importable,
#      splits.json present, Chronos-2 config loads, PVRecordDataset imports
#   4. data-staging checks (dataset_all.parquet + images_all.h5)
#
#   bash scripts/precache_login.sh
#   WITH_VIDTOK=1 bash scripts/precache_login.sh   # also install the VidTok deps
#   STAGE=verify  bash scripts/precache_login.sh   # skip sync/weights, only checks
#
# After it finishes, allocate a GPU node and run scripts/run_all_mmtsfm.sh.
set -uo pipefail
cd "$(dirname "$0")/.."          # MMTSFM/
MMTSFM_DIR="$PWD"
REPO_ROOT="$(cd .. && pwd)"

[[ -f .env ]] && { set -a; source .env; set +a; }
[[ -f "$REPO_ROOT/.env" ]] && { set -a; source "$REPO_ROOT/.env"; set +a; }

STAGE="${STAGE:-all}"            # all | weights | verify | data
WITH_VIDTOK="${WITH_VIDTOK:-0}"
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME}/hub}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data}"
DATA="${DATA:-${DATA_DIR}/dataset_all.parquet}"
IMAGES_H5="${IMAGES_H5:-${DATA_DIR}/images_all.h5}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints}"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCH_HUB_DIR" "$DATA_DIR" "$CKPT_DIR" logs/slurm
info() { echo "[mmtsfm-precache] $*"; }
warn() { echo "[mmtsfm-precache][WARN] $*" >&2; }

export PYTHONPATH="${MMTSFM_DIR}/src:${REPO_ROOT}/baselines:${PYTHONPATH:-}"

echo "=============================================================="
echo " MMTSFM PRECACHE   stage=$STAGE  with_vidtok=$WITH_VIDTOK"
echo " HF_HOME=$HF_HOME   TORCH_HUB_DIR=$TORCH_HUB_DIR"
echo " DATA_DIR=$DATA_DIR"
echo "=============================================================="

# --- 1+2: env + weights (uv sync, V-JEPA 2.1, Chronos-2) ---------------------
if [[ "$STAGE" == "all" || "$STAGE" == "weights" ]]; then
    info "uv sync (main deps incl. vjepa2 / h5py / pandas)"
    uv sync || warn "uv sync failed — fix before submitting the GPU job"
    if [[ "$WITH_VIDTOK" == "1" ]]; then
        info "uv sync --group vidtok (decord / av — Linux only)"
        uv sync --group vidtok || warn "vidtok group sync failed (only needed for the VidTok encoder)"
    fi
    info ">>> V-JEPA 2.1 hub weights + Chronos-2 backbone (login_node_setup.sh)"
    bash scripts/login_node_setup.sh || warn "login_node_setup.sh had warnings"
fi

# --- 3: verify the MMTSFM <-> baselines protocol bridge ----------------------
if [[ "$STAGE" == "all" || "$STAGE" == "verify" ]]; then
    info ">>> verifying offline imports + protocol bridge"
    uv run python - <<'PY' || warn "import/bridge verification FAILED — the GPU run will not be offline-clean"
import importlib, sys
for m in ("torch", "lightning", "h5py", "pandas", "pyarrow", "transformers"):
    importlib.import_module(m)
print("core imports OK")

# baselines/common is on PYTHONPATH (pv_record bootstraps it at runtime too)
from common import config
from common.splits import load_splits
splits = load_splits()
assert "uk_pv" in splits, "uk_pv missing from baselines/configs/splits.json"
print(f"baselines bridge OK — {len(config.COV_COLS)} covariates, "
      f"uk_pv splits {{k: len(v) for k,v in splits['uk_pv'].items()}}")

# PVRecordDataset imports (its module-level bootstrap must resolve `common`)
from mmtsfm.data.pv_record import PVRecordDataset  # noqa: F401
print("pv_record import OK")
PY

    info ">>> verifying Chronos-2 backbone loads from cache"
    uv run python - <<'PY' || warn "Chronos-2 backbone load failed — check amazon/chronos-2 in HF cache"
from mmtsfm.models.chronos2.config import Chronos2CoreConfig
Chronos2CoreConfig.from_pretrained("amazon/chronos-2")
print("Chronos-2 config OK")
PY
fi

# --- 4: data staging checks --------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "data" ]]; then
    info ">>> data staging"
    [[ -f "$DATA" ]]      && info "OK dataset_all.parquet : $DATA"     || warn "MISSING $DATA — copy thesis-dataset/dataset_all.parquet here"
    [[ -f "$IMAGES_H5" ]] && info "OK images_all.h5       : $IMAGES_H5" || warn "MISSING $IMAGES_H5 — copy thesis-dataset/images_all.h5 here"
fi

echo ""
echo "=============================================================="
echo " MMTSFM PRECACHE DONE. run_all_mmtsfm.sh auto-resolves:"
echo "   DATA_DIR    = $DATA_DIR"
echo "   HF_HOME     = $HF_HOME   (amazon/chronos-2)"
echo "   TORCH_HUB_DIR = $TORCH_HUB_DIR   (V-JEPA 2.1)"
echo "   CKPT_DIR    = $CKPT_DIR"
echo ""
echo " Then on a GPU node:  sbatch scripts/run_all_mmtsfm.sh"
echo "=============================================================="
