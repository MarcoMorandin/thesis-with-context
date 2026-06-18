#!/usr/bin/env bash
# ============================================================================
# Run this on the LOGIN NODE (has internet access).
#
# Does:
#   1. Create uv virtualenv and install dependencies
#   2. Download SKIPPD dataset from HuggingFace
#
# Usage:
#   bash setup_env.sh [--skip-download] [--hf-token <token>]
#
# Run once before submitting the SLURM job:
#   bash setup_env.sh
#   sbatch train_skippd.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

SCRATCH="/leonardo_scratch/fast/IscrC_MTSFM/SolarVLM"
export SOLARVLM_SCRATCH="$SCRATCH"

SKIP_DOWNLOAD=0
HF_TOKEN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo "Project dir : ${PROJECT_DIR}"
echo "Scratch dir : ${SCRATCH}"
echo "Date        : $(date)"
echo "======================================================================"

# ── Step 1: Python environment ────────────────────────────────────────────
# echo ""
# echo "==> Step 1: Create uv virtualenv and install dependencies"

# if ! command -v uv &>/dev/null; then
#     echo "uv not found — installing via pip"
#     pip install --user uv
# fi


# uv pip install -r requirements.txt \
#     --extra-index-url https://download.pytorch.org/whl/cu126 \
#     --index-strategy unsafe-best-match

# echo "Virtualenv ready at ${PROJECT_DIR}/.venv"

# # ── Step 2: Download SKIPPD ───────────────────────────────────────────────
# if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
#     echo ""
#     echo "==> Step 2: Download / verify SKIPPD dataset"
#     mkdir -p "${SCRATCH}/dataset/skippd"

#     DOWNLOAD_CMD="uv run --no-sync scripts/download_skippd.py \
#         --output_dir ${SCRATCH}/dataset/skippd \
#         --image_dir  ${SCRATCH}/dataset/skippd/images"
#     if [[ -n "$HF_TOKEN" ]]; then
#         DOWNLOAD_CMD+=" --hf_token ${HF_TOKEN}"
#     fi
#     eval "$DOWNLOAD_CMD"

#     # Download ERA5 and labels configs for covariates
#     echo ""
#     echo "==> Step 2b: Download ERA5 + labels configs"
#     uv run --no-sync - <<PYEOF
# import datasets as hf_datasets
# from pathlib import Path
# cache = Path("${SCRATCH}/dataset/skippd/skippd_hf_cache")
# cache.mkdir(parents=True, exist_ok=True)
# for cfg in ['ERA5', 'labels']:
#     for split in ['train', 'test']:
#         out = cache / f"{cfg}_{split}.parquet"
#         if out.exists():
#             print(f"  [{cfg}/{split}] already cached, skipping")
#             continue
#         print(f"  Downloading [{cfg}/{split}] ...")
#         ds = hf_datasets.load_dataset('solarbench/SKIPPD', cfg,
#                                       split=split,
#                                       cache_dir=str(cache / 'hf_raw'))
#         ds.to_pandas().to_parquet(out, index=False)
#         print(f"  -> {out.name}")
# # Remove stale merged cache so loader rebuilds with ERA5
# for p in cache.glob("merged_*.parquet"):
#     p.unlink()
#     print(f"  Removed stale {p.name}")
# PYEOF
# else
#     echo ""
#     echo "==> Step 2: Skipping dataset download (--skip-download)"
# fi

# ── Step 3: Download Qwen3-VL-Embedding-2B model weights ─────────────────
echo ""
echo "==> Step 3: Download Qwen3-VL-Embedding-2B weights"
MODEL_DIR="${SCRATCH}/QwenQwen3-VL-Embedding-2B"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
    echo "Model already present at ${MODEL_DIR} — skipping"
else
    HF_TOKEN_ARG=""
    if [[ -n "$HF_TOKEN" ]]; then
        HF_TOKEN_ARG="token='${HF_TOKEN}',"
    fi
    uv run --no-sync - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-VL-Embedding-2B",
    local_dir="${MODEL_DIR}",
    ${HF_TOKEN_ARG}
)
print("Downloaded to ${MODEL_DIR}")
PYEOF
fi

echo ""
echo "======================================================================"
echo "Setup complete: $(date)"
echo ""
echo "Next: sbatch train_skippd.sh"
echo "======================================================================"
