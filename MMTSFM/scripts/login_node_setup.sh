#!/bin/bash
# Run this script ONCE from the login node (internet access required).
# Sets up everything needed for a V-JEPA 2.1 / Chronos-2 training run:
#   1. uv sync (installs vjepa2 and other dependencies)
#   2a. Download vjepa2_1 weights to torch hub checkpoints cache
#   2b. Clone vjepa2 hub repo and verify loading (reads weights from cache)
#   3. Prefetch Chronos-2 backbone
#   4. Create runtime directories
#
# Usage:
#   bash scripts/login_node_setup.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${TEAM_SCRATCH}/torch_cache}"
export TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME}/hub}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[setup] $*"; }
check() { command -v "$1" &>/dev/null || { echo "ERROR: '$1' not found on PATH"; exit 1; }; }

# ---------------------------------------------------------------------------
# 0. Prerequisite checks
# ---------------------------------------------------------------------------
check uv
check git
check python3

info "Team scratch  : $TEAM_SCRATCH"
info "HF cache      : $HF_HOME"
info "Torch cache   : $TORCH_HOME"
info "Hub dir       : $TORCH_HUB_DIR"
echo

# ---------------------------------------------------------------------------
# 1. Python environment
# ---------------------------------------------------------------------------
info "Syncing uv environment ..."
uv sync
info "uv sync done."

# ---------------------------------------------------------------------------
# 2a. Download vjepa2_1 weights to the torch hub checkpoints cache.
#     torch.hub.load_state_dict_from_url checks this directory before any network
#     download, so pre-placing the file bypasses the localhost:8300 testing URL
#     that the current vjepa2 main branch ships with.
# ---------------------------------------------------------------------------
CKPT_CACHE="${TORCH_HUB_DIR}/checkpoints"
CKPT_DEST="${CKPT_CACHE}/vjepa2_1_vitl_dist_vitG_384.pt"
mkdir -p "$CKPT_CACHE"

if [[ -f "$CKPT_DEST" ]]; then
    info "vjepa2_1 weights already cached: $CKPT_DEST"
else
    info "Downloading vjepa2_1_vitl_dist_vitG_384.pt from dl.fbaipublicfiles.com ..."
    wget -q --show-progress \
        -O "$CKPT_DEST" \
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
fi

# ---------------------------------------------------------------------------
# 2b. Clone vjepa2 hub repo and verify it loads (weights already in cache above).
# ---------------------------------------------------------------------------
info "Pre-caching V-JEPA 2.1 hub repo and verifying load ..."
mkdir -p "$TORCH_HUB_DIR"
uv run python - <<'EOF'
import os, sys, types, torch

hub_dir = os.environ["TORCH_HUB_DIR"]
os.makedirs(hub_dir, exist_ok=True)
torch.hub.set_dir(hub_dir)

repo_dir = os.path.join(hub_dir, "facebookresearch_vjepa2_main")

# First attempt: let torch.hub download the repo (will fail due to src conflict).
if not os.path.isdir(repo_dir):
    try:
        torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_large_384", trust_repo=True)
    except Exception:
        pass

if not os.path.isdir(repo_dir):
    raise RuntimeError(f"Hub repo not found at {repo_dir} after download attempt.")

# Patch the testing leftover in backbones.py: the current main branch ships with
# VJEPA_BASE_URL = "http://localhost:8300" (a debug override). Fix it to the real CDN.
backbones = os.path.join(repo_dir, "src", "hub", "backbones.py")
with open(backbones) as f:
    src = f.read()
patched = src.replace(
    'VJEPA_BASE_URL = "http://localhost:8300"',
    'VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"',
)
if patched != src:
    with open(backbones, "w") as f:
        f.write(patched)
    print("Patched VJEPA_BASE_URL in backbones.py")

# Pin sys.modules['src'] to the hub repo's src/ so imports resolve from there,
# not from the installed vjepa2 wheel's partial src/ (which lacks utils/tensors).
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
for key in list(sys.modules.keys()):
    if key in ("src", "app") or key.startswith(("src.", "app.")):
        del sys.modules[key]
src_stub = types.ModuleType("src")
src_stub.__path__ = [os.path.join(repo_dir, "src")]
src_stub.__package__ = "src"
sys.modules["src"] = src_stub

torch.hub.load(repo_dir, "vjepa2_1_vit_large_384", source="local", trust_repo=True)
print("V-JEPA 2.1 cached.")
EOF

# ---------------------------------------------------------------------------
# 3. Chronos-2 Base Model Checkpoint
# ---------------------------------------------------------------------------
info "Pre-downloading Chronos-2 backbone (amazon/chronos-2) to HF cache..."
uv run python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="amazon/chronos-2")
EOF

# ---------------------------------------------------------------------------
# 4. Runtime directories
# ---------------------------------------------------------------------------
mkdir -p \
    "${TEAM_SCRATCH}/checkpoints/vjepa_proposal" \
    "${TEAM_SCRATCH}/data" \
    logs/slurm

info "Runtime directories created."

echo
echo "===== Setup complete ====="
echo "  V-JEPA 2.1    : weights → $CKPT_DEST"
echo "                  hub repo → $TORCH_HUB_DIR"
echo "  Chronos-2     : amazon/chronos-2 (HF cache)"
echo
echo "Submit training with:"
echo "  sbatch scripts/slurm_train_vjepa.sh"
echo
