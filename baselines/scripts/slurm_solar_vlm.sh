#!/bin/bash
#SBATCH --job-name=t6-solarvlm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=4-00:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Tier-6 Solar-VLM (P0, domain SOTA). Unlike the other vendored baselines,
# Solar-VLM lives in its OWN repository (PROJECT_DIR / SOLARVLM_DIR) with its own
# uv venv and Qwen3-VL embedding backbone — it is not in the baselines/ tree.
# This wrapper just drives that repo's training+eval on the cluster so the single
# orchestrator can launch it like every other baseline.
#
#   sbatch --export=ALL,SOLARVLM_DIR=/leonardo/home/userexternal/<user>/Solar-VLM \
#          scripts/slurm_solar_vlm.sh
#
# Required: SOLARVLM_DIR (the Solar-VLM repo checkout, prepared on the login node
#           via its setup_env.sh — Qwen3-VL weights + dataset under SCRATCH).
# Optional: SOLARVLM_RUN  (script inside the repo to run; default train_skippd.sh)
#           SOLARVLM_SCRATCH (default /leonardo_scratch/fast/IscrC_MTSFM/SolarVLM)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
BASELINES_DIR="$PWD"

: "${SOLARVLM_DIR:?set SOLARVLM_DIR to the Solar-VLM repo checkout}"
[[ -d "$SOLARVLM_DIR" ]] || { echo "ERROR: SOLARVLM_DIR not found: $SOLARVLM_DIR"; exit 1; }
SOLARVLM_RUN="${SOLARVLM_RUN:-train_skippd.sh}"
export SOLARVLM_SCRATCH="${SOLARVLM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM/SolarVLM}"

# offline (compute node has no internet; weights cached on login node)
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-${TEAM_SCRATCH}/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TEAM_SCRATCH}/pip_cache}"
export UV_ENVS_DIR="${UV_ENVS_DIR:-${TEAM_SCRATCH}/uv_envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${TEAM_SCRATCH}/conda_pkgs}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-${TEAM_SCRATCH}/conda_envs}"

echo ">>> Solar-VLM ($SOLARVLM_RUN) in $SOLARVLM_DIR"
( cd "$SOLARVLM_DIR" && bash "$SOLARVLM_RUN" )

# The Solar-VLM repo writes its metrics under $SOLARVLM_SCRATCH/test_results.
# If it emits a metrics.json, surface a result row the aggregator can read.
mkdir -p "$BASELINES_DIR/results"
SV_METRICS="$(ls -t "${SOLARVLM_SCRATCH}"/test_results/*metrics*.json 2>/dev/null | head -1 || true)"
if [[ -n "$SV_METRICS" ]]; then
    cp "$SV_METRICS" "$BASELINES_DIR/results/solar_vlm_s2_ukpv_mm.json"
    echo "✓ Solar-VLM metrics → results/solar_vlm_s2_ukpv_mm.json"
else
    echo "NOTE: no Solar-VLM metrics.json found under ${SOLARVLM_SCRATCH}/test_results."
    echo "      Solar-VLM runs in its own repo on its own dataset; import its"
    echo "      predictions into results/ manually if the schema differs"
    echo "      (it does not share the import_predictions npz contract)."
fi
