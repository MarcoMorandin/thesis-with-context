#!/bin/bash
#SBATCH --job-name=t2-itransformer-nf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# Mail notifications: pass `sbatch --mail-type=... --mail-user=...` at submit
# time (CINECA-recommended). Never run `watch -n N squeue` against the
# controller.
# =============================================================================
# Tier-2 iTransformer (neuralforecast) — TRAIN + TEST on MMTSFM's protocol.
# =============================================================================
# The control arm for the MMTSFM curriculum: same windows, same recipe, same
# scorer (tier2/train_itransformer_nf.py documents the parity list). Submit
# from baselines/, one job per seed:
#
#   for s in 42 43 44; do
#     sbatch --export=ALL,SEED=$s scripts/slurm_itransformer.sh
#   done
#
# Prereq (login node, has internet):  uv sync --group nf
# Writes results/itransformer_nf_s2_ukpv_seed<SEED>.json in the baselines
# schema, matching the other tiers.
#
# Knobs (all optional): DATA_DIR RESULTS_DIR CKPT_DIR SP_REF DS SEED
#                       EPOCHS BATCH_SIZE ACCUM TRAIN_STRIDE FUTURE_COV LOSS
#                       NUM_WORKERS TAG EXTRA_ARGS
# =============================================================================
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f pyproject.toml && -d common ]] || { echo "FATAL: submit from baselines/"; exit 1; }
[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
mkdir -p logs/slurm

# ---- offline: compute nodes have no internet -------------------------------
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TEAM_SCRATCH}/uv_cache}"
# Dataset of record, the same copy MMTSFM/scripts/slurm_curriculum.sh reads
# (DATA_DIR -> <dir>/dataset_all.parquet, resolved inside PVRecordDataset).
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data_v2}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints/itransformer_nf}"

DS="${DS:-uk_pv}"
SEED="${SEED:-42}"
# Recipe defaults = the MMTSFM curriculum's (knowledge/protocol.md + MMTSFM
# configs/trainer/default.yaml). Change one only to ablate it deliberately —
# every change here breaks the like-for-like claim against mmtsfm_s2b_ukpv.
EPOCHS="${EPOCHS:-40}"            # S1's from-scratch budget (iTransformer has no pretraining)
BATCH_SIZE="${BATCH_SIZE:-16}"    # effective batch 16, as MMTSFM (4 x accum 4)
ACCUM="${ACCUM:-1}"
TRAIN_STRIDE="${TRAIN_STRIDE:-12}"
FUTURE_COV="${FUTURE_COV:-all}"   # future weather known, as PVRecordDataset(future_cov="all")
LOSS="${LOSS:-mse}"               # matches thuml/iTransformer's own nn.MSELoss(); mae also valid
NUM_WORKERS="${NUM_WORKERS:-8}"

dcfg() { case "$1" in uk_pv) echo ukpv;; goes_pvdaq) echo goespvdaq;; *) echo "$1";; esac; }
TAG="${TAG:-itransformer_nf_s2_$(dcfg "$DS")_seed${SEED}}"
# Skill-Score reference: the SAME Smart-Persistence run MMTSFM was scored
# against, or the evaluator's default if the tagged file is absent.
SP_REF="${SP_REF:-${RESULTS_DIR}/smart_persistence_s2_$(dcfg "$DS").json}"

[[ -f "${DATA_DIR}/dataset_all.parquet" ]] || {
  echo "FATAL: no dataset_all.parquet under ${DATA_DIR} — stage the dataset of record first"; exit 1; }

declare -a CMD=(
  python tier2/train_itransformer_nf.py
  --data-dir "$DATA_DIR" --dataset "$DS" --seed "$SEED"
  --train-stride "$TRAIN_STRIDE" --future-cov "$FUTURE_COV" --loss "$LOSS"
  --batch-size "$BATCH_SIZE" --accumulate "$ACCUM" --max-epochs "$EPOCHS"
  --num-workers "$NUM_WORKERS"
  --out "$RESULTS_DIR" --tag "$TAG" --ckpt-dir "$CKPT_DIR"
)
[[ -f "$SP_REF" ]] && CMD+=(--sp-reference "$SP_REF")
[[ -n "${EXTRA_ARGS:-}" ]] && CMD+=(${EXTRA_ARGS})

echo ">>> [$TAG] ds=$DS seed=$SEED stride=$TRAIN_STRIDE epochs=$EPOCHS loss=$LOSS"
echo "    uv run --group nf ${CMD[*]}"
[[ "${DRY_RUN:-0}" == "1" ]] && exit 0
exec uv run --group nf "${CMD[@]}"
