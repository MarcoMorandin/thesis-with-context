#!/bin/bash
#SBATCH --job-name=t6-sunset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --account=IscrC_MTSFM
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Tier-6 SUNSET (P0, domain SOTA, MULTIMODAL track) — TRAIN + EVAL.
# Nie et al. (Stanford) — canonical sky-image CNN: a stack of past sky images +
# PV history → 15-min-ahead PV. Runs the authors' ORIGINAL TF2/Keras code
# (tier6/vendor/sunset, MIT), adapted to our contract, NOT reimplemented.
# SKIPP'D is SUNSET's NATIVE dataset (the same sky-image data solar_vlm/ uses),
# so this is the least-blocked Tier-6 row: it needs the SKIPP'D HDF5
# (forecast_dataset.hdf5), produced by the multimodal data pipeline.
#
#   sbatch --export=ALL,CONDA_ENV=sunset,SKIPPD_HDF5=<forecast_dataset.hdf5> \
#          scripts/slurm_sunset.sh
#
# Required: CONDA_ENV (TF2.4 env), SKIPPD_HDF5 (SUNSET forecast HDF5).
# Optional: PRED_LEN(12) EPOCHS(20) SEED(42)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -f .env ]] && source .env

export WANDB_MODE=offline TF_CPP_MIN_LOG_LEVEL=2
TEAM_SCRATCH="${TEAM_SCRATCH:-/leonardo_scratch/fast/IscrC_MTSFM}"

: "${CONDA_ENV:?set CONDA_ENV to the SUNSET TF2.4 conda env (TIER6_INTEGRATION.md §1)}"
PRED_LEN="${PRED_LEN:-12}"; EPOCHS="${EPOCHS:-20}"; SEED="${SEED:-42}"
OUT="${OUT:-tier6/vendor/sunset/results_skippd}"

# ---- multimodal-track guard (SKIPP'D sky-image HDF5 required) ---------------
[[ -n "${SKIPPD_HDF5:-}" ]] || { echo "ERROR: SKIPPD_HDF5 unset — SUNSET needs the SKIPP'D
  sky-image HDF5 (forecast_dataset.hdf5: images_log + PV history). Build it with the
  SUNSET data_processing notebooks or reuse the solar_vlm SKIPP'D export, then set
  SKIPPD_HDF5. See docs/experiments/TIER6_INTEGRATION.md."; exit 2; }
[[ -f "$SKIPPD_HDF5" ]] || { echo "ERROR: SKIPPD_HDF5 not found: $SKIPPD_HDF5"; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"

# ---- TRAIN + EVAL (self-contained runner from SUNSET_forecast.ipynb) --------
# run_skippd.py is the adapted, no-notebook runner (added on vendor; see
# tier6/vendor/VENDOR_NOTICE.md "Adaptations") — trains the original SUNSET
# Keras model and dumps sunset_<site>_pred.npz in our baseline-contract format.
echo ">>> TRAIN+EVAL SUNSET (SKIPP'D, multimodal)"
RUNNER="tier6/vendor/sunset/run_skippd.py"
[[ -f "$RUNNER" ]] || { echo "ERROR: $RUNNER not present — convert SUNSET_forecast.ipynb
  to the self-contained run_skippd.py (TIER6_INTEGRATION.md §3) before submitting."; exit 3; }
python "$RUNNER" \
  --hdf5 "$SKIPPD_HDF5" --epochs "$EPOCHS" --seed "$SEED" \
  --pred_len "$PRED_LEN" --out "$OUT"

# ---- baseline-contract check + import → our NMAE/NRMSE/SS results JSON ------
shopt -s nullglob
for npz in "$OUT"/sunset_*_pred.npz; do
    uv run python tier4/vendor/contract_check.py --predictions "$npz" --horizon "$PRED_LEN" || true
done
uv run python scripts/import_predictions.py --model sunset --tag s2_mm \
    --glob "$OUT/sunset_*_pred.npz" \
    --reference results/smart_persistence_s2_ukpv.json
echo "✓ SUNSET done → results/sunset_s2_mm.json (make_tables / summarize_ukpv pick it up)."
