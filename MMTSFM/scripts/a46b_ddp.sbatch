#!/bin/bash
#SBATCH --job-name=A46b_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# =============================================================================
# A46b seed 42 — the SAME run as the pack launches, on 4 GPUs instead of 1.
# =============================================================================
# Why this file exists at all. The normal route is
#
#   MANIFEST=configs/ablation/A46b.manifest ... bash scripts/ablation_sweep.sh
#
# which allocates a whole 4-GPU node and then runs ONE ablation per GPU. With
# SEEDS=42 that is one run on one GPU and three A100s allocated, billed and
# idle for the whole 24 h. Measured on the live job: GPU 0 at 96-99 % util and
# 63070 of 64942 MiB, GPUs 1-3 untouched. The run is compute-bound, so more
# dataloader workers buy nothing; the only lever left is the three idle GPUs.
#
# This is not a new configuration. `trainer=slurm` already ships
#   devices: auto
#   strategy: ddp_find_unused_parameters_true   # frozen V-JEPA + modality dropout
# and lightning_module.py:80 explains that the LR gate exists *because* flipping
# requires_grad mid-run re-buckets DDP gradients. The DDP path is designed for.
# It is scripts/lib/stage_cmd.sh:102 that pins `trainer.devices=1
# trainer.strategy=auto` for every stage and arm, and ablation_pack.sbatch that
# pins CUDA_VISIBLE_DEVICES per slot. This script bypasses both for one run
# rather than editing build_stage_cmd, which every other arm depends on.
#
# THE EFFECTIVE BATCH IS UNCHANGED, and that is the whole point — A46b is only
# readable against S2D and S2E if the optimizer sees the same thing:
#   1 GPU : batch 2 x accum 8            = 16
#   4 GPU : batch 2 x 4 ranks x accum 2  = 16
# Epoch arithmetic matches too: 11786 train batches / 4 ranks / accum 2 = 1473
# optimizer steps, exactly the 11786 / accum 8 the single-GPU run performs.
# Expect ~3-3.4x wall-clock (find_unused_parameters costs 5-10 %).
#
# --ntasks-per-node MUST EQUAL trainer.devices. Lightning's SLURMEnvironment
# activates whenever SLURM variables are present and then does NOT spawn its own
# workers — it assumes srun already started one task per device and reads
# SLURM_LOCALID. With --ntasks-per-node=1 and devices=4 the job hangs at
# rendezvous. For the same reason do NOT add --gpus-per-task=1: every task must
# see all four GPUs so Lightning can bind rank -> device itself.
#
# test=false IS DELIBERATE. DistributedSampler pads the final batch by repeating
# samples; those duplicates land in ProtocolEvaluator's accumulator and produce
# a well-formed results JSON with quietly wrong cross-plant metrics. Train here,
# then score the checkpoint on ONE GPU:
#
#   MODE=eval SCORE_CKPT=${STAGE_DIR}/best.ckpt   # stage_cmd.sh:113-118
#
# IF A RANK OOMs AT STEP 0 the DDP reducer's gradient buckets did not fit in the
# ~1.8 GiB the single-GPU run leaves free. Drop to data.batch_size=1 with
# trainer.accumulate_grad_batches=4 — effective batch is still 16. Do NOT cut
# visual_evs_keep to fit: that reintroduces the payload confound A46b exists to
# remove (see configs/ablation/A46b.yaml).
#
# THE RANK COUNT IS NOT FIXED AT 4, because on a busy queue it cannot be. A
# 4-GPU / 32-core request needs an ENTIRELY EMPTY node, and whole-node gaps are
# rare; partial-node gaps open constantly. Measured on boost_usr_prod with this
# job pending: PRIORITY=135864 against a field of 205k-255k (256-node jobs on
# the high QOS, which a 1-node job cannot join) and 2553 eligible 1-4 node jobs
# ahead. sprio showed QOS=120000, FAIRSHARE=15603 (factor 0.624), AGE=253,
# JOBSIZE=8 — so the ceiling on `normal` is ~165k and the gap NEVER closes by
# waiting. This job runs by backfill or not at all, and backfill is won by
# asking for less: fewer GPUs and a shorter --time.
#
# So the rank count follows the allocation instead of being hardcoded, and the
# accumulation is recomputed to hold the effective batch at 16:
#   1 GPU : 2 x 1 x 8 = 16      2 GPU : 2 x 2 x 4 = 16      4 GPU : 2 x 4 x 2 = 16
# Override --ntasks-per-node and --gres together; they must agree or the run
# aborts before srun (the guard below), because a mismatch means Lightning binds
# ranks to devices that were never allocated.
#
#   sbatch scripts/a46b_ddp.sbatch                        # 4 GPUs, resumes last.ckpt
#   sbatch --ntasks-per-node=2 --gres=gpu:2 --time=04:00:00 \
#          scripts/a46b_ddp.sbatch                        # 2 GPUs, better backfill odds
#   RESUME=0 sbatch scripts/a46b_ddp.sbatch               # ignore last.ckpt, start clean
#
# ONE JOB AT A TIME. Every rank count writes the same STAGE_DIR; two jobs live
# at once would interleave writes to last.ckpt and corrupt both.
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
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Rank count = tasks srun will start. --ntasks-per-node on the sbatch command
# line overrides the #SBATCH directive above, so this follows whatever was asked
# for. SLURM_GPUS_ON_NODE is the allocated GPU count; if the two disagree the
# user changed one of --ntasks-per-node / --gres without the other, and
# Lightning would bind ranks to devices that do not exist.
NGPU="${SLURM_NTASKS:-4}"
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "${NGPU}" ]]; then
  echo "FATAL: ${NGPU} tasks but ${SLURM_GPUS_ON_NODE} GPUs allocated." >&2
  echo "       --ntasks-per-node and --gres=gpu:N must match." >&2
  exit 1
fi
# Effective batch is pinned at 16 — the only thing that makes A46b readable
# against S2D and S2E. batch_size stays 2 (the measured OOM ceiling at 533
# tokens), so accumulation absorbs the rank count.
BATCH=2
if (( 16 % (BATCH * NGPU) != 0 )); then
  echo "FATAL: ${NGPU} ranks x batch ${BATCH} does not divide 16." >&2
  echo "       Use 1, 2, 4 or 8 GPUs, or the effective batch shifts." >&2
  exit 1
fi
ACCUM=$(( 16 / (BATCH * NGPU) ))

# Paths: the defaults of scripts/ablation_sweep.sh, resolved for this one row.
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data_v2}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints/curriculum}"
RESULTS_DIR="${RESULTS_DIR:-$(cd .. && pwd)/baselines/results}"
# NOTE the ${DS} segment — ablation_sweep.sh:255 exports
# VJEPA_CACHE=${VJEPA_CACHE_ROOT}/${DS}/${VJEPA_CACHE_VER}. This is the 8-frame
# s2d cache that A46b RE-READS at five origins; pointing it at the 20-frame a24h
# cache would load 20-latent ladders where the config expects 5 x 4, silently.
VJEPA_CACHE="${VJEPA_CACHE:-/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224_nonhrv_sp45}"
# ablation_pack.sbatch: ${CKPT_DIR}/${ds}_${stage}$(arm_suffix "$model_cfg" "$seed")_${abl}
# arm_suffix(vision_chronos2_s2e, 42) = _s2e_s42, so the doubled s2e is correct.
STAGE_DIR="${STAGE_DIR:-${CKPT_DIR}/uk_pv_s2e_s2e_s42_A46b}"
PREV_CKPT="${PREV_CKPT:-${CKPT_DIR}/uk_pv_s1_selfattn_s42/best.ckpt}"
SP_REF="${SP_REF:-${RESULTS_DIR}/smart_persistence_s2_ukpv.json}"

# The banner prints BEFORE the first Lustre touch, deliberately. A malformed
# path does not fail at `mkdir`, it PARKS: job 57984364 spent 49 minutes of a
# 4 h allocation asleep in `mkdir -p` on a STAGE_DIR carrying a space where a
# slash belonged (/leonardo_scratch/fast<space>IscrC_MTSFM/...), because
# TEAM_SCRATCH came in from the submitting shell -- sbatch defaults to
# --export=ALL -- and wrote not one byte to stdout the whole time. Resolving
# the paths into the log first makes that state readable in the first second
# instead of requiring `ps` on the compute node.
echo "=============================================================="
echo " A46b DDP   job=${SLURM_JOB_ID:-local}  ranks=${SLURM_NTASKS:-?}"
echo " STAGE_DIR=${STAGE_DIR}"
echo " DATA_DIR=${DATA_DIR}"
echo " VJEPA_CACHE=${VJEPA_CACHE}"
echo " PREV_CKPT=${PREV_CKPT}"
echo " effective batch = ${BATCH} x ${NGPU} ranks x ${ACCUM} accum = 16"
echo "=============================================================="

# Every path that lands on Lustre, checked before anything touches Lustre. The
# space is the failure actually seen; the prefix test catches the wider class
# (empty variable, relative path, pointed at the wrong filesystem).
for _v in STAGE_DIR DATA_DIR VJEPA_CACHE CKPT_DIR PREV_CKPT; do
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

# `timeout` so a genuinely degraded filesystem also fails loud instead of
# eating the allocation in uninterruptible sleep.
timeout 120 mkdir -p "$STAGE_DIR" logs/slurm || {
  echo "FATAL: mkdir -p '${STAGE_DIR}' did not return within 120 s." >&2
  echo "       Scratch may be degraded: ls /leonardo_scratch/fast/IscrC_MTSFM/" >&2
  exit 1
}

CMD=(
  python -m mmtsfm.train
  +stage=s2e model=vision_chronos2_s2e data=ukpv
  trainer=slurm
  # The three lines stage_cmd.sh:102 would otherwise pin to a single GPU.
  "trainer.devices=${NGPU}"
  trainer.strategy=ddp_find_unused_parameters_true
  trainer.num_nodes=1
  seed=42 trainer.max_epochs=20
  "trainer.default_root_dir=${STAGE_DIR}"
  # BATCH x NGPU ranks x ACCUM = 16, the effective batch of the 2 x 8 run.
  "data.batch_size=${BATCH}"
  "trainer.accumulate_grad_batches=${ACCUM}"
  data.num_workers=6
  "model.results_dir=${RESULTS_DIR}" model.results_tag=mmtsfm_A46b_s2e_ukpv_s42
  # Pinned, NOT ${now:...}: four ranks resolve `now` independently and would
  # otherwise scatter the run across up to four hydra output directories.
  "hydra.run.dir=logs/experiments/runs/${SLURM_JOB_ID:-local}_A46b_ddp"
  "data.data_dir=${DATA_DIR}"
  "data.vjepa_cache_dir=${VJEPA_CACHE}"
  model.compute_marginal_gain=true
  model.vision_cfg.n_visual_context_steps=5
  +data.train_stride=12
  # Appended last so +ablation=A46b beats the model and stage configs, then
  # test=false beats the ablation's own `test: true`.
  +ablation=A46b
  test=false
)

if [[ -f "$PREV_CKPT" ]]; then
  CMD+=("init_ckpt=${PREV_CKPT}")
else
  # A train row that silently starts from random init writes a well-formed
  # results JSON that no downstream check can tell from a real one.
  echo "FATAL: warm-start checkpoint not found: ${PREV_CKPT}" >&2
  exit 1
fi
[[ -f "$SP_REF" ]] && CMD+=("model.sp_reference_path=${SP_REF}")
[[ "${RESUME:-1}" == "1" && -f "${STAGE_DIR}/last.ckpt" ]] && CMD+=("ckpt_path=${STAGE_DIR}/last.ckpt")

printf 'srun uv run %s\n' "${CMD[*]}"
srun uv run "${CMD[@]}"
