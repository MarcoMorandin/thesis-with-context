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

# Local secrets / overrides (e.g. TABPFN_TOKEN for TabPFN's one-time license +
# weight download). `set -a` auto-exports everything sourced, so `uv run`
# subprocesses (the TabPFN warm-up below) inherit it. Mirrors run_all_baselines.sh.
if [[ -f .env ]]; then set -a; source .env; set +a; fi

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
# TabPFN (tier1) pulls its model checkpoint from HF on first .fit; pin a stable
# cache dir so this warm-up and the offline compute run resolve the same files.
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${WEIGHTS_DIR}/tabpfn}"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$DATA_DIR" "$UKPV_CSV_DIR" "$CKPT_DIR" \
         "$WEIGHTS_DIR" "$TABPFN_MODEL_CACHE_DIR" logs/slurm
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
    # Standalone Chronos-2 baselines (chronos2_zs / chronos2_ft / chronos2_oracle)
    # use the official chronos-forecasting package — a CORE dep, so the uv sync
    # above installs it — plus the amazon/chronos-2 snapshot cached by
    # login_node_prep.sh below. Verify the import now, while the login node has
    # internet, so a broken install fails loud here instead of silently offline.
    uv run python -c "from chronos import Chronos2Pipeline" \
        && info "chronos-forecasting OK → chronos2_zs/ft/oracle ready" \
        || warn "official 'chronos' import failed — chronos2 baselines will not run (check chronos-forecasting in baselines/pyproject.toml)"
    info ">>> Tier-3/4 + RAG via login_node_prep.sh"
    DATA="$DATA" UKPV_CSV_DIR="$UKPV_CSV_DIR" bash scripts/login_node_prep.sh || warn "login_node_prep had warnings"

    # --- TS-RAG / Cross-RAG released mixer checkpoint (Google Drive) -------
    # Release folder holds checkpoints/{base, chronos-bolt/best.pth} (+ datasets/
    # & retrieval_database/ we don't use — we retrieve over our own uk_pv CSVs).
    # Both RAG repos run zeroshot.py --model ChronosBoltRetrieve with
    #   --pretrained_model_path <chronos-bolt-base dir>   (already cached from HF;
    #        the Drive base/config.json is _name_or_path autogluon/chronos-bolt-base)
    #   --checkpoint_model_path <best.pth>                (the trained retrieval mixer)
    # so the single best.pth serves BOTH ts_rag and cross_rag. The file is shared
    # anyone-with-link, so gdown fetches it (handling the >100 MB virus-scan confirm
    # token) with NO gws / Google account on the compute side — only `uv`. Override
    # RAG_MIXER_DRIVE_ID if the release moves.
    RAG_MIXER_DRIVE_ID="${RAG_MIXER_DRIVE_ID:-1O17Dl_x_YPSOAORsyW7PADREMHaze2w5}"  # checkpoints/chronos-bolt/best.pth
    if [[ -s "${CKPT_DIR}/arm.pth" ]]; then
        info "RAG mixer already present at ${CKPT_DIR}/arm.pth"
    else
        info "fetching RAG mixer (Drive ${RAG_MIXER_DRIVE_ID}) → ${CKPT_DIR}/arm.pth"
        uv run --with gdown python -m gdown "$RAG_MIXER_DRIVE_ID" \
            -O "${CKPT_DIR}/arm.pth" || warn "gdown RAG mixer download failed"
        sz="$(stat -c%s "${CKPT_DIR}/arm.pth" 2>/dev/null || stat -f%z "${CKPT_DIR}/arm.pth" 2>/dev/null || echo 0)"
        if [[ "$sz" -gt 1000000 ]]; then
            info "RAG mixer ($(du -h "${CKPT_DIR}/arm.pth" | cut -f1)) → enables ts_rag + cross_rag"
        else
            rm -f "${CKPT_DIR}/arm.pth"
            warn "RAG mixer download invalid/too small — stage ${CKPT_DIR}/arm.pth by hand"
        fi
    fi

    # --- Cross-RAG native pretrain data (HF nkh/TS-RAG-Data) ----------------
    # Cross-RAG has no released mixer (the Drive best.pth is TS-RAG's ARM), so
    # cross_rag pretrains its cross-attention mixer at 512/64 on the 50m pretrain
    # pairs. Stage the retrieval DB + pair chunks (~tens of GB) here; the offline
    # GPU run trains on them. Skip with CROSSRAG_PRETRAIN=0.
    CROSSRAG_PRETRAIN_DIR="${CROSSRAG_PRETRAIN_DIR:-${TEAM_SCRATCH}/crossrag_pretrain}"
    if [[ "${CROSSRAG_PRETRAIN:-1}" == "1" ]]; then
        info ">>> Cross-RAG pretrain data → $CROSSRAG_PRETRAIN_DIR (nkh/TS-RAG-Data)"
        mkdir -p "$CROSSRAG_PRETRAIN_DIR"
        uv run --group tier3 python - "$CROSSRAG_PRETRAIN_DIR" <<'PY' || warn "Cross-RAG pretrain-data download failed — cross_rag will skip"
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
p = snapshot_download("nkh/TS-RAG-Data", repo_type="dataset", local_dir=dest,
                      allow_patterns=["retrieval_database_512.parquet",
                                      "pretrain_pairs_ctx512/*.parquet"])
print("cross-rag pretrain data ->", p)
PY
    fi

    # --- 3/6: Tier-5/6 backbone weights ------------------------------------
    info ">>> Tier-5/6 backbones"
    hf_pull "openai/clip-vit-base-patch32" "${WEIGHTS_DIR}/clip-vit-base-patch32"   # Time-VLM + UniCast vision
    hf_pull "Lefei/VisionTSpp"             "${WEIGHTS_DIR}/visiontspp"              # VisionTS++ MAE ckpt
    hf_pull "amazon/chronos-bolt-base"     "${WEIGHTS_DIR}/chronos-bolt-base"       # UniCast backbone + RAG BASE_CKPT
    hf_pull "DecisionIntelligence/Aurora"  "${WEIGHTS_DIR}/aurora-tsfm"             # Aurora zero-shot TS FM ckpt

    # Resolve the VisionTS++ checkpoint to a stable path run_all_baselines.sh
    # globs for (the repo ships the MAE weights under several file names).
    # -type f skips the visiontspp.ckpt symlink we create below, so a re-run
    # cannot resolve to (and then re-link) the symlink onto itself. Prefer the
    # leakage-safe large ckpt; fall back to the first real checkpoint otherwise.
    _vts_all="$(find "${WEIGHTS_DIR}/visiontspp" -maxdepth 2 -type f \
                  \( -name '*.ckpt' -o -name '*.pth' -o -name '*.safetensors' \) \
                  2>/dev/null | sort)"
    vts_ckpt="$(printf '%s\n' "$_vts_all" | grep -m1 'visiontspp_large_gifteval_no_leakage' || true)"
    [[ -z "$vts_ckpt" ]] && vts_ckpt="$(printf '%s\n' "$_vts_all" | head -1)"
    if [[ -n "$vts_ckpt" ]]; then
        ln -sf "$vts_ckpt" "${WEIGHTS_DIR}/visiontspp/visiontspp.ckpt"
        info "VisionTS++ ckpt → ${WEIGHTS_DIR}/visiontspp/visiontspp.ckpt ($vts_ckpt)"
    else
        warn "no VisionTS++ .ckpt/.pth/.safetensors under ${WEIGHTS_DIR}/visiontspp — visionts_pp will skip"
    fi

    # Solar-VLM offline vision backbone (Qwen3-VL-Embedding-2B). Override
    # QWEN_REPO if the exact HF id differs in your account.
    QWEN_REPO="${QWEN_REPO:-Qwen/Qwen3-VL-Embedding-2B}"
    hf_pull "$QWEN_REPO" "${WEIGHTS_DIR}/qwen3-vl-embedding-2b" \
        || warn "Qwen3-VL pull failed ($QWEN_REPO) — set QWEN_REPO/QWEN_PATH; solar_vlm will skip"

    # --- TabPFN (Tier-1 tabular FM, optional `tabpfn` group) ------------------
    # TabPFN lives in the MAIN project env (not a vendored uv env). Sync the
    # optional `tabpfn` group into the project env LAST so the offline compute
    # node (UV_NO_SYNC) has base + tier3 + tabpfn, then warm its model checkpoint
    # (pulled from HF on first .fit) into TABPFN_MODEL_CACHE_DIR while the login
    # node has internet. Skip with TABPFN=0.
    if [[ "${TABPFN:-1}" == "1" ]]; then
        info ">>> TabPFN-3 (tier1 tabular FM) → group sync + V3 weight download"
        # TabPFN-3 (ModelVersion.V3) gates its local-inference weights behind a
        # one-time license keyed by TABPFN_TOKEN (https://ux.priorlabs.ai). Put
        # TABPFN_TOKEN=<key> in baselines/.env (sourced at top); the offline
        # compute run reuses it via run_all_baselines.sh's own .env source.
        [[ -n "${TABPFN_TOKEN:-}" ]] \
            || warn "TABPFN_TOKEN not set (baselines/.env) — V3 weight download will hit the license gate"
        uv sync --group tier3 --group tabpfn || warn "uv sync (tier3+tabpfn) failed"
        # DOWNLOAD-ONLY: the login node has no GPU and cgroup-kills the heavy CPU
        # forward pass, so we DON'T require fit() to finish — the download (the
        # only thing we need cached) completes first. The real fit runs on the
        # GPU compute node. Success = the V3 ckpt landing in the shared cache.
        OMP_NUM_THREADS=1 uv run --group tier3 --group tabpfn python - <<'PY' || true
import numpy as np
from tabpfn import TabPFNRegressor, ModelVersion
m = TabPFNRegressor.create_default_for_version(ModelVersion.V3)
try:
    m.fit(np.random.rand(16, 3), np.random.rand(16))  # triggers the V3 download
    print("tabpfn-3 fit OK on login node")
except Exception as e:  # no GPU / cgroup kill — download already done
    print("note: login-node fit did not finish (expected, no GPU):", type(e).__name__)
PY
        if find "$TABPFN_MODEL_CACHE_DIR" -iname '*tabpfn-v3*regressor*.ckpt' 2>/dev/null | grep -q .; then
            info "TabPFN-3 ckpt cached → $TABPFN_MODEL_CACHE_DIR (offline GPU fit will load it)"
        else
            warn "TabPFN-3 ckpt NOT under $TABPFN_MODEL_CACHE_DIR — tabpfn will skip offline (set TABPFN_TOKEN, accept license at https://ux.priorlabs.ai)"
        fi
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
    # chronos_requirements pins torch 2.6 (cu124, needs driver 12.4+); Leonardo's
    # driver is 12.2 → CPU fallback. Override with the cu121 build (works on 12.2).
    [[ "$MAKE_ENVS" == "1" && -d "$UV_ENVS_DIR/unicast" ]] && \
        VIRTUAL_ENV="$UV_ENVS_DIR/unicast" uv pip install torch==2.4.1 torchvision==0.19.1 \
        || true
    # Aurora (decisionintelligence/Aurora) zero-shot: it needs transformers>=4.50
    # but its modeling code uses the pre-5.0 tied-weights API (_tied_weights_keys);
    # transformers 5.x's from_pretrained calls all_tied_weights_keys and crashes,
    # so cap it <5. Plus the deps its generate()/model actually import.
    # torch 2.4.1 is the cu121 build (CUDA 12.1) — Leonardo's driver is 12.2, so
    # the default latest torch (cu124, needs driver 12.4+) silently falls back to
    # CPU. Pin cu121 so Aurora's generate() runs on GPU. (Aurora needs torch>=2.4.)
    make_env aurora  pip install torch==2.4.1 torchvision==0.19.1 'transformers>=4.50,<5' \
                                 huggingface_hub einops numpy pandas scikit-learn tqdm matplotlib
    make_env crossvivit pip install -r "$BASELINES_DIR/tier6/vendor/crossvivit/requirements.txt"
    make_env sunset    pip install tensorflow h5py pyarrow pandas numpy
    # Solar-VLM (in-tree, Tier-6): its own requirements + h5py/pyarrow for the
    # uk_pv loader and the Qwen3-VL vision precompute.
    make_env solar_vlm pip install --index-strategy unsafe-best-match -r "$BASELINES_DIR/tier6/vendor/solar_vlm/requirements.txt" h5py pyarrow
    # RAG originals pin numpy==1.25 + chronos-forecasting + faiss-gpu (TIER4_RAG_INTEGRATION §1)
    [[ -f tier4/vendor/ts_rag/requirements.txt    ]] && make_env tsrag   pip install -r "$BASELINES_DIR/tier4/vendor/ts_rag/requirements.txt"
    [[ -f tier4/vendor/cross_rag/requirements.txt ]] && make_env crossrag pip install -r "$BASELINES_DIR/tier4/vendor/cross_rag/requirements.txt"

    # Aurora is ZERO-SHOT (decisionintelligence/Aurora): run_ukpv.py loads the
    # released DecisionIntelligence/Aurora checkpoint (cached in the weights stage
    # → $WEIGHTS_DIR/aurora-tsfm) and calls model.generate(). No training, no
    # config-dir / download_ckpt.py (that vendored downloader is broken & unused).
    if [[ -d "${WEIGHTS_DIR}/aurora-tsfm" ]]; then
        info "Aurora ckpt OK → ${WEIGHTS_DIR}/aurora-tsfm (use as AURORA_CKPT, zero-shot)"
    else
        warn "Aurora ckpt missing (${WEIGHTS_DIR}/aurora-tsfm) — run STAGE=weights; aurora will skip"
    fi
fi

# --- 5/6: Solar-VLM (in-tree at tier6/vendor/solar_vlm) ----------------------
# No external repo / setup_env.sh anymore — its uv env is built in the 4/6 block
# (make_env solar_vlm) and the Qwen3-VL backbone is cached in the 3/6 block.
if [[ -d "$UV_ENVS_DIR/solar_vlm" && -d "${WEIGHTS_DIR}/qwen3-vl-embedding-2b" ]]; then
    info "Solar-VLM ready (uv env + Qwen3-VL weights). QWEN_PATH=${WEIGHTS_DIR}/qwen3-vl-embedding-2b"
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
echo "   AURORA_CKPT     = ${WEIGHTS_DIR}/aurora-tsfm  (DecisionIntelligence/Aurora, zero-shot)"
echo "   QWEN_PATH       = ${WEIGHTS_DIR}/qwen3-vl-embedding-2b  (Solar-VLM vision)"
echo "   RAG_MIXER_CKPT  = ${CKPT_DIR}/arm.pth  (best.pth via gdown → enables ts_rag + cross_rag)"
echo ""
echo " Then on a GPU node:  sbatch scripts/run_all_baselines.sh"
echo "=============================================================="
