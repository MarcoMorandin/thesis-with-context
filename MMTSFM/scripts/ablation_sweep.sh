#!/bin/bash
# =============================================================================
# MMTSFM ABLATION SWEEP SUBMITTER  —  run on a Leonardo LOGIN node.
# =============================================================================
# Turns configs/ablation/sweep.manifest into a handful of whole-node SLURM jobs
# that each run several ablations concurrently, instead of ~45 single-GPU jobs
# submitted by hand.
#
#   bash scripts/ablation_sweep.sh                       # submit the whole manifest
#   DRY_RUN=1 bash scripts/ablation_sweep.sh             # print the plan, submit nothing
#   ONLY="A09 A10" bash scripts/ablation_sweep.sh        # just the eval controls
#   ONLY=A17 SEEDS=42 NPACKS=1 bash scripts/ablation_sweep.sh
#   CHAIN=3 bash scripts/ablation_sweep.sh               # 3 linked packs: survive walltime
#
# What it does, and what it does NOT buy you
# ------------------------------------------
# Each pack is `--nodes=1 --gres=gpu:4 --cpus-per-task=32` and runs a work queue:
# four ablations at a time, a freed GPU takes the next. Against four separate
# `--gres=gpu:1` jobs that is the SAME node-hours — Leonardo bills allocated
# resources and four quarter-nodes equal one node — so this is not a way to
# stretch the IscrC_MTSFM budget. What it buys:
#
#   - one queue wait per pack instead of one per ablation;
#   - one env + data warm-up amortised over every run in the pack;
#   - no hand-typed sbatch line (and so no hand-typed wrong PREV_CKPT);
#   - a continuation chain that resumes whatever the 24 h cap cut off.
#
# What it can COST, if used carelessly: tail idle. When a pack runs out of queued
# jobs, finished GPUs sit allocated and idle until the last run ends. Keep
# jobs-per-pack well above 4 (the default packing does) and put runs of similar
# length together; a single 24 h straggler alongside three finished slots wastes
# three GPU-days. With fewer than ~8 jobs, submit them as single-GPU curriculum
# stages instead — see knowledge/running-ablations.md §1.
#
# Prereq: scripts/precache_login.sh has staged the uv env, weights, dataset and
# the V-JEPA latent cache. Submit from MMTSFM/.
# =============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/.."
[[ -f pyproject.toml && -d src/mmtsfm ]] || { echo "FATAL: run from MMTSFM/"; exit 1; }
REPO_ROOT="$(cd .. && pwd)"

# shellcheck source=lib/stage_cmd.sh
source scripts/lib/stage_cmd.sh

# ---- config (override via env) ----------------------------------------------
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"
DATA_DIR="${DATA_DIR:-${TEAM_SCRATCH}/data_v2}"
CKPT_DIR="${CKPT_DIR:-${TEAM_SCRATCH}/checkpoints/curriculum}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/baselines/results}"
VJEPA_CACHE_ROOT="${VJEPA_CACHE_ROOT:-/leonardo_work/IscrC_MTSFM/vjepa_cache}"
VJEPA_ARCH="${VJEPA_ARCH:-vit_large}"
# Cache of record: v2 non-HRV frames, 8 frames over 6 h => 45-min spacing. The
# bare `${VJEPA_ARCH}_f8_s224` name still exists on /leonardo_work and holds the
# OBSOLETE v1 HRV latents, so defaulting to it would silently train on
# grayscale daylight-only imagery.
VJEPA_CACHE_VER="${VJEPA_CACHE_VER:-${VJEPA_ARCH}_f8_s224_nonhrv_sp45}"
MANIFEST="${MANIFEST:-configs/ablation/sweep.manifest}"
DS="${DS:-uk_pv}"
TRAIN_STRIDE="${TRAIN_STRIDE:-12}"
ACCOUNT="${ACCOUNT:-IscrC_MTSFM}"
PARTITION="${PARTITION:-boost_usr_prod}"
SWEEP_TIME="${SWEEP_TIME:-24:00:00}"     # boost_usr_prod cap
GPUS="${GPUS:-4}"                        # GPUs per pack = Booster node width
NPACKS="${NPACKS:-4}"                    # nodes to spread the sweep over
CHAIN="${CHAIN:-1}"                      # linked resubmissions per pack
# Per-stage micro-batch + accumulation, matching slurm_curriculum.sh so an
# ablation is comparable to the arm it ablates.
SWEEP_EPOCHS="${SWEEP_EPOCHS:-20}"
SWEEP_BATCH="${SWEEP_BATCH:-4}"
SWEEP_ACCUM="${SWEEP_ACCUM:-4}"
# Job status by mail (CINECA-recommended). Never poll with `watch -n N squeue`.
MAIL_USER="${MAIL_USER:-}"
MAIL_TYPE="${MAIL_TYPE:-END,FAIL}"
ONLY="${ONLY:-}"                         # space list of IDs; empty = whole manifest
SEEDS_OVERRIDE="${SEEDS:-}"              # override every row's seed list
DRY_RUN="${DRY_RUN:-0}"

dcfg_for() { case "$1" in uk_pv) echo ukpv;; goes_pvdaq) echo goespvdaq;; *) echo "$1";; esac; }
# n_visual_context_steps per dataset for patch=16: uk_pv (30-min) = ceil(12/16)=1,
# goes_pvdaq (15-min) = ceil(24/16)=2.
nvis_for() { [[ -n "${N_VIS:-}" ]] && { echo "$N_VIS"; return; }; case "$1" in goes_pvdaq) echo 2;; *) echo 1;; esac; }
sp_ref_for() { local f="${RESULTS_DIR}/smart_persistence_s2_$(dcfg_for "$1").json"; [[ -f "$f" ]] && echo "$f" || echo ""; }

DCFG="$(dcfg_for "$DS")"
[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest not found: ${MANIFEST}"; exit 1; }

# ---- expand the manifest into one job per (ablation, seed) ------------------
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${SWEEP_DIR:-logs/sweeps/${SWEEP_ID}}"
mkdir -p "$SWEEP_DIR" logs/slurm

declare -a J_LINE=() J_TAG=()
missing=0
while IFS='|' read -r id mode stage model_cfg seeds base; do
  id="${id// /}"; mode="${mode// /}"; stage="${stage// /}"
  model_cfg="${model_cfg// /}"; seeds="${seeds// /}"; base="${base// /}"
  [[ -z "$id" || "$id" == \#* ]] && continue
  if [[ -n "$ONLY" ]] && ! grep -qw -- "$id" <<< "$ONLY"; then continue; fi
  [[ -n "$SEEDS_OVERRIDE" ]] && seeds="${SEEDS_OVERRIDE// /,}"

  for seed in ${seeds//,/ }; do
    ckpt="${CKPT_DIR}/${base//\{seed\}/$seed}/best.ckpt"
    # Fatal, not a warning. A train row that silently starts from random init, or
    # an eval row that silently scores an untrained model, both write a
    # well-formed results JSON that no downstream check can tell from a real one.
    if [[ ! -f "$ckpt" && "$DRY_RUN" != "1" ]]; then
      echo "FATAL: ${id} seed ${seed} — checkpoint not found: ${ckpt}"
      missing=$((missing+1)); continue
    fi
    tag="mmtsfm_${id}_${stage}_$(dcfg_for "$DS")_s${seed}"
    init_ckpt=""; score_ckpt=""
    if [[ "$mode" == "eval" ]]; then score_ckpt="$ckpt"; else init_ckpt="$ckpt"; fi
    J_TAG+=("$tag")
    # `|`-separated, NOT tab: tab is an IFS whitespace character, so `read` with
    # IFS=$'\t' collapses runs of them and an empty init_ckpt would shift every
    # later column left by one — silently handing the batch size to `seed`.
    J_LINE+=("$(printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' \
      "$tag" "$mode" "$stage" "$model_cfg" "$seed" "$id" "$init_ckpt" "$score_ckpt" \
      "$DCFG" "$DS" "$SWEEP_EPOCHS" "$SWEEP_BATCH" "$SWEEP_ACCUM")")
  done
done < "$MANIFEST"

(( missing == 0 )) || { echo "FATAL: ${missing} missing checkpoint(s) — fix BASE in ${MANIFEST}, or run the arms they warm-start from first."; exit 1; }
NJOBS=${#J_TAG[@]}
(( NJOBS > 0 )) || { echo "FATAL: manifest selected no jobs (ONLY='${ONLY}')"; exit 1; }

# A tag collision means two runs write one results JSON and one checkpoint dir,
# clobbering each other mid-run with no error raised. Same failure mode the
# wave-safety tests exist to prevent, so it is checked before anything is queued.
dupes="$(printf '%s\n' "${J_TAG[@]}" | sort | uniq -d)"
[[ -z "$dupes" ]] || { echo "FATAL: duplicate run tags in the sweep:"; echo "$dupes"; exit 1; }

# ---- split round-robin across packs -----------------------------------------
# Round-robin, not contiguous slices: the manifest is ordered cheapest-first, so
# contiguous packs would put every minutes-long eval row in pack 0 and every 24 h
# training row in the last pack — one node finishing in an hour while another
# runs out of walltime.
(( NPACKS < 1 )) && NPACKS=1
(( NPACKS > NJOBS )) && NPACKS=$NJOBS
for (( p=0; p<NPACKS; p++ )); do : > "${SWEEP_DIR}/pack${p}.jobs"; done
for (( k=0; k<NJOBS; k++ )); do
  printf '%s\n' "${J_LINE[$k]}" >> "${SWEEP_DIR}/pack$((k % NPACKS)).jobs"
done

echo "=============================================================="
echo " MMTSFM ABLATION SWEEP   ${SWEEP_ID}"
echo " manifest=${MANIFEST}   ds=${DS}   jobs=${NJOBS}"
echo " packs=${NPACKS} x ${GPUS} GPU   chain=${CHAIN}   walltime=${SWEEP_TIME}"
echo " job files → ${SWEEP_DIR}/pack*.jobs"
echo "=============================================================="

[[ "$DRY_RUN" == "1" ]] || command -v sbatch >/dev/null || {
  echo "FATAL: sbatch not found (run on a Leonardo login node, or DRY_RUN=1)"; exit 1; }

declare -a MAIL=()
[[ -n "$MAIL_USER" ]] && MAIL=(--mail-type="$MAIL_TYPE" --mail-user="$MAIL_USER")

# ---- submit ------------------------------------------------------------------
for (( p=0; p<NPACKS; p++ )); do
  job_file="${SWEEP_DIR}/pack${p}.jobs"
  n_in_pack="$(wc -l < "$job_file" | tr -d ' ')"
  exports="ALL,JOB_FILE=${job_file},DATA_DIR=${DATA_DIR},CKPT_DIR=${CKPT_DIR}"
  exports+=",RESULTS_DIR=${RESULTS_DIR},GPUS=${GPUS},TRAIN_STRIDE=${TRAIN_STRIDE}"
  exports+=",VJEPA_CACHE=${VJEPA_CACHE_ROOT}/${DS}/${VJEPA_CACHE_VER}"
  exports+=",N_VIS=$(nvis_for "$DS")"
  sp="$(sp_ref_for "$DS")"; [[ -n "$sp" ]] && exports+=",SP_REF=${sp}"

  prev_jid=""
  # CHAIN links identical resubmissions with `afterany`, NOT `afterok`: the point
  # is to pick up after a pack that hit the 24 h cap, which SLURM reports as
  # TIMEOUT (a failure). SKIP_DONE in the worker makes the repeat idempotent —
  # a run whose results JSON exists is skipped, everything else resumes from
  # its own last.ckpt.
  for (( c=0; c<CHAIN; c++ )); do
    name="sweep${SWEEP_ID}_p${p}$( ((c>0)) && echo "_c${c}" )"
    declare -a DEP=()
    [[ -n "$prev_jid" ]] && DEP=(--dependency="afterany:${prev_jid}")
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "--- would submit ---"
      echo "  JOB_NAME=${name}"
      echo "  PACK=${p}"
      echo "  JOB_FILE=${job_file}"
      echo "  NJOBS_IN_PACK=${n_in_pack}"
      echo "  GPUS=${GPUS}"
      echo "  TIME=${SWEEP_TIME}"
      echo "  DEPENDENCY=${DEP[*]:-<none>}"
      while IFS='|' read -r t _; do echo "  TAG=${t}"; done < "$job_file"
      prev_jid="dry$((p * 100 + c))"
      continue
    fi
    jid="$(sbatch --parsable "${DEP[@]}" "${MAIL[@]}" \
      --job-name="$name" --account="$ACCOUNT" --partition="$PARTITION" \
      --nodes=1 --gres="gpu:${GPUS}" --cpus-per-task=32 \
      --time="$SWEEP_TIME" --export="$exports" \
      scripts/ablation_pack.sbatch)" || { echo "sbatch failed for ${name}"; exit 1; }
    echo "  submitted ${name}  jid=${jid}  (${n_in_pack} runs)${prev_jid:+  afterany:${prev_jid}}"
    prev_jid="$jid"
  done
done

echo ""
echo "Results  → ${RESULTS_DIR}/mmtsfm_<ID>_<stage>_<ds>_s<seed>.json"
echo "Per-run  → logs/slurm/<tag>.log        Pack     → logs/slurm/<jobid>_sweep*.out"
[[ -n "$MAIL_USER" ]] || echo "NOTE: MAIL_USER unset — set it for END/FAIL mail instead of polling squeue."
