#!/bin/bash
# =============================================================================
# MASTER LOGIN-NODE PRECACHE  —  run ONCE on the Leonardo login node (internet).
# =============================================================================
# Prepares every off-repo artifact the offline GPU orchestrator
# (scripts/run_all_baselines.sh) needs, so compute nodes run FULLY OFFLINE:
#   1. uv env (+ tier3 group) and all Tier-3/4 HF weights
#   2. RAG Chronos backbones + uk_pv CSV export
#   3. Tier-5/6 backbone weights (CLIP / VisionTS++ MAE / Chronos-Bolt / Aurora)
#   4. one uv env per vendored model (Tier 5/6 + RAG)
#   5. Solar-VLM repo setup (optional)
#   6. data-staging checks (dataset_all.parquet + images_all.h5)
#
#   bash scripts/precache_login.sh
#   MAKE_ENVS=0 bash scripts/precache_login.sh           # skip uv env creation
#   STAGE=weights bash scripts/precache_login.sh         # only HF/torch weights
#
# After it finishes, allocate a GPU node and run scripts/run_all_baselines.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
BASELINES_DIR="$PWD"

STAGE="${STAGE:-all}"          # all | weights | envs | data
MAKE_ENVS="${MAKE_ENVS:-1}"
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
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${TEAM_SCRATCH}/weights}"   # tier5/6 backbone dirs
SOLARVLM_DIR="${SOLARVLM_DIR:-}"
PY_VER="${PY_VER:-3.10}"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$DATA_DIR" "$UKPV_CSV_DIR" "$CKPT_DIR" \
         "$WEIGHTS_DIR" logs/slurm
info() { echo "[precache] $*"; }
warn() { echo "[precache][WARN] $*" >&2; }

# --- uv env helper -------------------------------------------------------
export UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"
mkdir -p "$UV_ENVS_DIR"

make_env() {   # make_env <name> <install-command...>
    local name="$1"; shift
    local venv_path="$UV_ENVS_DIR/$name"
    [[ "$MAKE_ENVS" == "1" ]] || { info "skip env $name (MAKE_ENVS=0)"; return; }
    if [[ -d "$venv_path" ]]; then
        info "uv env '$name' already exists — skipping create"
    else
        info "creating uv env '$name' (python=$PY_VER)"
        uv venv --python "$PY_VER" "$venv_path" >/dev/null || { warn "create $name failed"; return; }
    fi
    info "installing deps into '$name'"
    VIRTUAL_ENV="$venv_path" uv "$@" || warn "dep install in '$name' failed (inspect log)"
}
hf_pull() {    # hf_pull <repo> [local_dir]
    local repo="$1" dest="${2:-}"
    uv run --group tier3 python - "$repo" "$dest" <<'PY' || warn "HF cache failed: $1"
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], (sys.argv[2] or None)
p = snapshot_download(repo_id=repo, local_dir=dest) if dest else snapshot_download(repo_id=repo)
print("cached", repo, "->", p)
PY
}

echo "=============================================================="
echo " MASTER PRECACHE   stage=$STAGE  make_envs=$MAKE_ENVS"
echo " HF_HOME=$HF_HOME"
echo " weights=$WEIGHTS_DIR   data=$DATA_DIR"
echo "=============================================================="

# --- 1/6 + 2/6: Tier-3/4 HF weights + RAG export (reuse login_node_prep) ----
if [[ "$STAGE" == "all" || "$STAGE" == "weights" ]]; then
    info "uv sync (base + tier3)"
    uv sync --group tier3 || warn "uv sync failed"
    info ">>> Tier-3/4 + RAG via login_node_prep.sh"
    DATA="$DATA" UKPV_CSV_DIR="$UKPV_CSV_DIR" bash scripts/login_node_prep.sh || warn "login_node_prep had warnings"

    # --- 3/6: Tier-5/6 backbone weights ------------------------------------
    info ">>> Tier-5/6 backbones"
    hf_pull "openai/clip-vit-base-patch32" "${WEIGHTS_DIR}/clip-vit-base-patch32"   # Time-VLM + UniCast vision
    hf_pull "Lefei/VisionTSpp"             "${WEIGHTS_DIR}/visiontspp"              # VisionTS++ MAE ckpt
    hf_pull "amazon/chronos-bolt-base"     "${WEIGHTS_DIR}/chronos-bolt-base"       # UniCast backbone + RAG BASE_CKPT

    # Resolve the VisionTS++ checkpoint to a stable path run_all_baselines.sh
    # globs for (the repo ships the MAE weights under an arbitrary file name).
    vts_ckpt="$(find "${WEIGHTS_DIR}/visiontspp" -maxdepth 2 \
                  \( -name '*.ckpt' -o -name '*.pth' -o -name '*.safetensors' \) \
                  2>/dev/null | head -1)"
    if [[ -n "$vts_ckpt" ]]; then
        ln -sf "$vts_ckpt" "${WEIGHTS_DIR}/visiontspp/visiontspp.ckpt"
        info "VisionTS++ ckpt → ${WEIGHTS_DIR}/visiontspp/visiontspp.ckpt ($vts_ckpt)"
    else
        warn "no VisionTS++ .ckpt/.pth/.safetensors under ${WEIGHTS_DIR}/visiontspp — visionts_pp will skip"
    fi
fi

# --- 4/6: one uv env per vendored model ----------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "envs" ]]; then
    info ">>> vendored model conda envs"
    make_env timevlm   pip install -r "$BASELINES_DIR/tier5/vendor/time_vlm/requirements.txt"
    make_env visionts  pip install -e "$BASELINES_DIR/tier5/vendor/visionts_pp"
    # UniCast ships its deps under requirements/ (chronos backbone variant), not a
    # top-level requirements.txt — install that one so the `unicast` env is built.
    make_env unicast pip install -r "$BASELINES_DIR/tier5/vendor/unicast/requirements/chronos_requirements.txt"
    # Aurora has no requirements file; it imports ViT/BERT/chronos-style backbones.
    # Pin the minimal set its runner/dataset/connector actually import.
    make_env aurora  pip install torch torchvision transformers huggingface_hub \
                                 einops numpy pandas scikit-learn tqdm matplotlib
    make_env crossvivit pip install -r "$BASELINES_DIR/tier6/vendor/crossvivit/requirements.txt"
    make_env sunset    pip install tensorflow h5py pyarrow pandas numpy
    # RAG originals pin numpy==1.25 + chronos-forecasting + faiss-gpu (TIER4_RAG_INTEGRATION §1)
    [[ -f tier4/vendor/ts_rag/requirements.txt    ]] && make_env tsrag   pip install -r "$BASELINES_DIR/tier4/vendor/ts_rag/requirements.txt"
    [[ -f tier4/vendor/cross_rag/requirements.txt ]] && make_env crossrag pip install -r "$BASELINES_DIR/tier4/vendor/cross_rag/requirements.txt"

    # Aurora "checkpoint": the orchestrator runs MODE=finetune, i.e. runner.py
    # builds AuroraForPrediction from AuroraConfig (config.json) and fine-tunes
    # from scratch on uk_pv, then evals — so model_path only needs the config dir
    # the repo already ships (aurora/config.json + vit_config + bert_config, all
    # random-init/local, no external ViT/BERT weights). The vendored
    # utils/download_ckpt.py is NOT used: it hardcodes a fake HF token, the
    # hf-mirror endpoint and /home/Aurora, and pulls ViT/BERT weights the code
    # never loads via from_pretrained.
    AURORA_CFG_DIR="$BASELINES_DIR/tier5/vendor/aurora/aurora"
    if [[ -f "$AURORA_CFG_DIR/config.json" ]]; then
        info "Aurora config dir OK → $AURORA_CFG_DIR (use as AURORA_CKPT, MODE=finetune)"
    else
        warn "Aurora config.json missing under $AURORA_CFG_DIR — aurora will skip"
    fi
fi

# --- 5/6: Solar-VLM repo (optional, its own setup) --------------------------
if [[ -n "$SOLARVLM_DIR" && -d "$SOLARVLM_DIR" ]]; then
    info ">>> Solar-VLM setup_env.sh"
    ( cd "$SOLARVLM_DIR" && bash setup_env.sh ) || warn "Solar-VLM setup_env had warnings"
fi

# --- 6/6: data-staging checks -----------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "data" ]]; then
    info ">>> data staging"
    [[ -f "$DATA" ]]       && info "OK dataset_all.parquet  : $DATA" || warn "MISSING $DATA — copy thesis-dataset/dataset_all.parquet here"
    [[ -f "$IMAGES_H5" ]]  && info "OK images_all.h5        : $IMAGES_H5" || warn "MISSING $IMAGES_H5 — copy thesis-dataset/images_all.h5 here"
    if [[ -f "$DATA" ]]; then
        uv run python tier4/vendor/export_ukpv.py --data "$DATA" --out "$UKPV_CSV_DIR" || warn "uk_pv export failed"
    fi
fi

echo ""
echo "=============================================================="
echo " PRECACHE DONE. run_all_baselines.sh auto-resolves these defaults:"
echo "   DATA            = $DATA"
echo "   IMAGES_H5       = $IMAGES_H5"
echo "   UKPV_CSV_DIR    = $UKPV_CSV_DIR"
echo "   MAE_CKPT        = ${WEIGHTS_DIR}/visiontspp/visiontspp.ckpt  (symlink resolved above)"
echo "   VISION_MODEL_PATH = ${WEIGHTS_DIR}/clip-vit-base-patch32"
echo "   CHRONOS_PATH / RAG_BASE_CKPT = ${WEIGHTS_DIR}/chronos-bolt-base"
echo "   AURORA_CKPT     = $BASELINES_DIR/tier5/vendor/aurora/aurora  (config dir, MODE=finetune)"
echo ""
echo " STILL NEED A MANUAL ARTIFACT (run_all skips these until present):"
echo "   RAG_MIXER_CKPT  = ${CKPT_DIR}/arm.pth   ← drop the released ARM/cross-attn ckpt here"
echo "                       (enables ts_rag + cross_rag)"
echo "   SOLARVLM_DIR    = ${TEAM_SCRATCH}/Solar-VLM   ← git clone the Solar-VLM repo here,"
echo "                       then re-run this precache with SOLARVLM_DIR set (runs its setup_env.sh)"
echo ""
echo " Then on a GPU node:  sbatch scripts/run_all_baselines.sh"
echo "=============================================================="
