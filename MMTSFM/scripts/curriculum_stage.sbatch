#!/bin/bash
#SBATCH --job-name=mmtsfm-stage
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# Mail notifications are injected by the submitter (slurm_curriculum.sh) via
# `sbatch --mail-type/--mail-user` when MAIL_USER is set — the CINECA-recommended
# alternative to polling squeue. Never run `watch -n N squeue` against the controller.
# =============================================================================
# ONE curriculum stage on ONE GPU (a quarter of a Leonardo Booster node).
# Driven entirely by environment (exported by the submitter
# scripts/slurm_curriculum.sh via `sbatch --export`). Never run this directly —
# walltime/QoS and the STAGE/DS/PREV_CKPT wiring come from the submitter. Writes
# a stable best.ckpt (train.py) so the next stage can resume, and a tagged
# protocol JSON (mmtsfm_<STAGE>_<DS>) into RESULTS_DIR.
#
# For a BATCH of ablations, use scripts/ablation_sweep.sh instead: it packs four
# runs onto one whole node, which is the same node-hours but one scheduling
# decision and one queue wait instead of four.
# =============================================================================
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -d src/mmtsfm ]] || { echo "FATAL: submit from MMTSFM/"; exit 1; }

[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}" HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The Hydra command is built by the shared library so this launcher and the
# node-packing one (ablation_pack.sbatch) cannot drift apart.
# shellcheck source=lib/stage_cmd.sh
source scripts/lib/stage_cmd.sh

declare -a CMD=()
build_stage_cmd || exit 1

echo ">>> [$TAG] stage=$STAGE ds=$DS prev=${PREV_CKPT:-<none>}"
echo "    uv run ${CMD[*]}"
# Inspect the composed Hydra command without training (wave pre-flight + tests).
[[ "${DRY_RUN:-0}" == "1" ]] && exit 0
exec uv run "${CMD[@]}"
