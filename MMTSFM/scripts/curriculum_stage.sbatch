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
# =============================================================================
# ONE curriculum stage. Driven entirely by environment (exported by the
# submitter scripts/slurm_curriculum.sh via `sbatch --export`). Never run this
# directly — walltime/QoS and the STAGE/DS/PREV_CKPT wiring come from the
# submitter. Writes a stable best.ckpt (train.py) so the next stage can resume,
# and a tagged protocol JSON (mmtsfm_<STAGE>_<DS>) into RESULTS_DIR.
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

: "${STAGE:?STAGE not set}" "${DS:?DS not set}" "${DCFG:?DCFG not set}" "${TAG:?TAG not set}"
: "${MODEL_CFG:=vision_chronos2_grassmann}"
: "${DATA_DIR:?DATA_DIR not set}" "${CKPT_DIR:?CKPT_DIR not set}" "${RESULTS_DIR:?RESULTS_DIR not set}"
: "${MAX_EPOCHS:=50}" "${BATCH_SIZE:=16}" "${NUM_WORKERS:=8}" "${SEED:=42}"

STAGE_DIR="${CKPT_DIR}/${DS}_${STAGE}"
mkdir -p "$STAGE_DIR" logs/slurm

declare -a CMD=(
  python -m mmtsfm.train
  "+stage=${STAGE}" "model=${MODEL_CFG}" "data=${DCFG}"
  trainer=slurm trainer.devices=1 trainer.strategy=auto
  "seed=${SEED}" "trainer.max_epochs=${MAX_EPOCHS}"
  "trainer.default_root_dir=${STAGE_DIR}"
  "data.data_dir=${DATA_DIR}" "data.batch_size=${BATCH_SIZE}" "data.num_workers=${NUM_WORKERS}"
  "model.results_dir=${RESULTS_DIR}" "model.results_tag=${TAG}"
  'hydra.run.dir=logs/experiments/runs/${now:%Y-%m-%d_%H-%M-%S}_'"${TAG}"
)

# Cross-stage: WEIGHTS-ONLY warm start from the previous stage's best.ckpt (fresh
# optimizer + epoch 0 — the freezing/param-groups differ per stage). S1 has none.
[[ -n "${PREV_CKPT:-}" && -f "${PREV_CKPT}" ]] && CMD+=("init_ckpt=${PREV_CKPT}")
# Requeue-safety: FULL-STATE resume of THIS same stage from its own last.ckpt.
# Takes precedence — a mid-stage requeue continues where it left off.
[[ "${RESUME:-1}" == "1" && -f "${STAGE_DIR}/last.ckpt" ]] && CMD+=("ckpt_path=${STAGE_DIR}/last.ckpt")
# Pre-extracted V-JEPA latents (skipped for S1 which sets emit_vision=false).
[[ -n "${VJEPA_CACHE:-}" && -d "${VJEPA_CACHE}" ]] && CMD+=("data.vjepa_cache_dir=${VJEPA_CACHE}")
[[ -n "${SP_REF:-}" ]] && CMD+=("model.sp_reference_path=${SP_REF}")
[[ -n "${TRAIN_STRIDE:-}" ]] && CMD+=("+data.train_stride=${TRAIN_STRIDE}")
[[ -n "${LIMIT_TRAIN_BATCHES:-}" ]] && CMD+=("+trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}")
[[ -n "${EXTRA_OVERRIDES:-}" ]] && CMD+=(${EXTRA_OVERRIDES})

echo ">>> [$TAG] stage=$STAGE ds=$DS prev=${PREV_CKPT:-<none>}"
echo "    uv run ${CMD[*]}"
exec uv run "${CMD[@]}"
