#!/bin/bash
#SBATCH --job-name=A46b_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MTSFM
#SBATCH --no-requeue
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# =============================================================================
# A46b seed 42 — score the trained checkpoint. ONE GPU, deliberately.
# =============================================================================
# scripts/a46b_ddp.sbatch trains with `test=false`, so a finished DDP run leaves
# a checkpoint and NO results JSON. This is the other half.
#
# THE SCORING PASS CANNOT RUN UNDER DDP. DistributedSampler pads the last batch
# by repeating samples so every rank sees the same count; those duplicates land
# in ProtocolEvaluator's accumulator and are indistinguishable from real windows
# by the time the metrics are reduced. The output is a well-formed JSON with
# quietly wrong cross-plant numbers — the worst failure mode available, because
# nothing downstream can detect it. devices=1 strategy=auto is the fix and the
# reason this is a separate job rather than a flag on the training one.
#
# `train=false` is set HERE, not in the config. stage_cmd.sh:82 notes that the
# eval-only arms (A09/A10) carry `train: false` in their own ablation YAML so
# the config stays the single statement of what the run does. A46b is a TRAIN
# arm — configs/ablation/A46b.yaml:`train: true` — so scoring it means
# overriding that on the command line, once, here.
#
# EVERY OTHER OVERRIDE MIRRORS THE TRAINING JOB. The checkpoint was fit under
# +ablation=A46b at n_visual_context_steps=5 against the 8-frame s2d cache; a
# scoring pass that resolves a different visual payload would load the weights
# without complaint and report numbers for a model that was never trained.
#
#   sbatch scripts/a46b_eval.sbatch                     # score best.ckpt
#   SCORE_CKPT=.../last.ckpt sbatch scripts/a46b_eval.sbatch
#
# AFTER THIS RUNS, THE NUMBER IS STILL NOT COMPARABLE. A46b drops every window
# whose 5-anchor ladder is incomplete (2.77% of train groups, 1.70% of val), so
# its test set is a strict SUBSET of S2D's and S2E's. A46b.yaml says it plainly:
# re-score S2D and S2E on the same surviving windows before quoting a delta, or
# the count effect is entangled with a ~2% change of evaluation set. That is a
# re-score, not a re-train.
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
export PYTHONUNBUFFERED=1

# Paths: identical to a46b_ddp.sbatch. Kept as plain assignments rather than
# sourced from it, because that script runs training as a side effect.
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data_v2}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints/curriculum}"
RESULTS_DIR="${RESULTS_DIR:-$(cd .. && pwd)/baselines/results}"
VJEPA_CACHE="${VJEPA_CACHE:-/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224_nonhrv_sp45}"
STAGE_DIR="${STAGE_DIR:-${CKPT_DIR}/uk_pv_s2e_s2e_s42_A46b}"
SCORE_CKPT="${SCORE_CKPT:-${STAGE_DIR}/best.ckpt}"
SP_REF="${SP_REF:-${RESULTS_DIR}/smart_persistence_s2_ukpv.json}"

echo "=============================================================="
echo " A46b EVAL   job=${SLURM_JOB_ID:-local}"
echo " SCORE_CKPT=${SCORE_CKPT}"
echo " DATA_DIR=${DATA_DIR}"
echo " VJEPA_CACHE=${VJEPA_CACHE}"
echo " RESULTS_DIR=${RESULTS_DIR}"
echo "=============================================================="

# Same check as the training script, same reason: a malformed path does not
# fail on Lustre, it parks in uninterruptible sleep with an empty log.
for _v in SCORE_CKPT DATA_DIR VJEPA_CACHE; do
  _p="${!_v}"
  case "$_p" in
    *" "*)
      echo "FATAL: ${_v} contains a space: '${_p}'" >&2
      echo "       Check .env and the submitting shell; sbatch exports both." >&2
      exit 1 ;;
    /leonardo_scratch/*|/leonardo_work/*) ;;
    *)
      echo "FATAL: ${_v} is not under /leonardo_scratch or /leonardo_work: '${_p}'" >&2
      exit 1 ;;
  esac
done

# A missing checkpoint here would score a randomly-initialised model and write a
# results JSON nothing downstream could tell from a real one (stage_cmd.sh:114).
timeout 120 test -f "$SCORE_CKPT" || {
  echo "FATAL: checkpoint not found (or scratch is hung): ${SCORE_CKPT}" >&2
  exit 1
}
timeout 120 mkdir -p logs/slurm || {
  echo "FATAL: mkdir did not return within 120 s; scratch may be degraded." >&2
  exit 1
}

CMD=(
  python -m mmtsfm.train
  +stage=s2e model=vision_chronos2_s2e data=ukpv
  trainer=slurm
  # The scoring pass is single-GPU by necessity — see the header.
  trainer.devices=1
  trainer.strategy=auto
  trainer.num_nodes=1
  seed=42
  "trainer.default_root_dir=${STAGE_DIR}"
  data.batch_size=2
  data.num_workers=6
  "model.results_dir=${RESULTS_DIR}" model.results_tag=mmtsfm_A46b_s2e_ukpv_s42
  "hydra.run.dir=logs/experiments/runs/${SLURM_JOB_ID:-local}_A46b_eval"
  "data.data_dir=${DATA_DIR}"
  "data.vjepa_cache_dir=${VJEPA_CACHE}"
  model.compute_marginal_gain=true
  model.vision_cfg.n_visual_context_steps=5
  "ckpt_path=${SCORE_CKPT}"
  # Appended last so +ablation=A46b beats the model and stage configs, then
  # train=false beats the ablation's own `train: true`.
  +ablation=A46b
  train=false
  test=true
)
[[ -f "$SP_REF" ]] && CMD+=("model.sp_reference_path=${SP_REF}")

printf 'srun uv run %s\n' "${CMD[*]}"
srun uv run "${CMD[@]}"
