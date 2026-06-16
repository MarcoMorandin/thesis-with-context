# Running all baselines on Leonardo (ISCRA-C)

End-to-end recipe to run **every baseline (Tiers 0–6)** on the dataset of record
and get **one results file**. Two commands: a login-node precache (internet),
then a single offline GPU job that parallelizes across the node's GPUs.

```
login node  →  bash scripts/precache_login.sh        (once, has internet)
GPU node    →  sbatch scripts/run_all_baselines.sh   (once, offline)  →  results/ALL_RESULTS.md
```

All paths/SLURM settings default to the project config: account `IscrC_MTSFM`,
partition `boost_usr_prod`, `TEAM_SCRATCH=/leonardo_scratch/fast/IscrC_MTSFM`.
Submit everything **from the `baselines/` directory**.

---

## 0. What runs

| Tier | Models | Env | GPU |
|---|---|---|---|
| T0 | persistence, smart_persistence, climatology_hourly, seasonal_naive | uv (`baselines`) | no |
| T1 | lightgbm | uv | no |
| T2 | mlp, dlinear, patchtst, itransformer, tft | uv | no |
| T3 | chronos2_zs/ft, timesfm_zs, tirex_zs, ttm_zs/ft | uv (`--group tier3`) | yes |
| T4 | cora (uv); ts_rag, cross_rag (vendored, uv) | uv / uv | yes |
| T5 | time_vlm, visionts_pp, unicast, aurora (vendored) | uv (one env each) | yes |
| T6 | crossvivit, sunset (vendored); solar_vlm (own repo) | uv / own venv | yes |

Tiers 0–2 are CPU and run first (Phase A); Tiers 3–6 run on the GPU pool
(Phase B). Smart Persistence is the skill-score reference and is always produced.

---

## 1. Login node — `scripts/precache_login.sh` (once, internet)

Compute nodes are **offline**, so everything that touches the network is cached
here first:

```bash
cd baselines
bash scripts/precache_login.sh
```

It does:
1. `uv sync --group tier3` and caches the Tier-3 weights (chronos-2, timesfm-2.5,
   TiRex, ttm-r3) + RAG Chronos backbones (chronos-t5-base, chronos-bolt-base)
   into `$HF_HOME`.
2. Caches the Tier-5/6 backbones into `$WEIGHTS_DIR`
   (`clip-vit-base-patch32`, `Lefei/VisionTSpp` MAE, `chronos-bolt-base`).
3. Creates **one uv env per vendored model**: `timevlm`, `visionts`,
   `unicast`, `aurora`, `crossvivit`, `sunset`, `tsrag`, `crossrag`
   (skip with `MAKE_ENVS=0`; skips envs that already exist).
4. Downloads the Aurora checkpoint (in the `aurora` env) and, if `SOLARVLM_DIR`
   is set, runs the Solar-VLM repo's `setup_env.sh` (Qwen3-VL weights).
5. Exports the uk_pv CSVs the RAG originals read and checks the staged data.

Useful overrides:

```bash
MAKE_ENVS=0 bash scripts/precache_login.sh          # weights only, no uv envs
STAGE=weights bash scripts/precache_login.sh         # only HF/torch weights
SOLARVLM_DIR=/leonardo/home/userexternal/<you>/Solar-VLM bash scripts/precache_login.sh
```

### 1a. Stage the data

Copy the **dataset of record** to `$TEAM_SCRATCH/data/` (the precache checks it):

```bash
cp /path/to/thesis-dataset/dataset_all.parquet "$TEAM_SCRATCH/data/"
cp /path/to/thesis-dataset/images_all.h5        "$TEAM_SCRATCH/data/"
```

### 1b. Note the checkpoint paths it prints

A few artifacts can't be auto-resolved — copy the exact paths the script prints
and pass them to the GPU job (next step):

| Variable | What | Default |
|---|---|---|
| `MAE_CKPT` | VisionTS++ MAE `.ckpt` file | `$WEIGHTS_DIR/visiontspp/<file>.ckpt` |
| `VISION_MODEL_PATH` | CLIP weights dir (UniCast/Time-VLM) | `$WEIGHTS_DIR/clip-vit-base-patch32` |
| `CHRONOS_PATH` / `RAG_BASE_CKPT` | Chronos-Bolt dir | `$WEIGHTS_DIR/chronos-bolt-base` |
| `AURORA_CKPT` | Aurora checkpoint dir | `tier5/vendor/aurora/<ckpt>` |
| `RAG_MIXER_CKPT` | released ARM / cross-attn ckpt (download by hand) | — |
| `SOLARVLM_DIR` | Solar-VLM repo checkout | — |

---

## 2. GPU node — `scripts/run_all_baselines.sh` (once, offline)

One sbatch job runs the whole suite and writes the single results file.

```bash
cd baselines
sbatch --export=ALL,\
MAE_CKPT=$TEAM_SCRATCH/weights/visiontspp/visiontspp.ckpt,\
AURORA_CKPT=$TEAM_SCRATCH/checkpoints/aurora,\
RAG_MIXER_CKPT=$TEAM_SCRATCH/checkpoints/arm.pth,\
SOLARVLM_DIR=/leonardo/home/userexternal/<you>/Solar-VLM \
  scripts/run_all_baselines.sh
```

(Anything you omit just means that baseline is **skipped**, not an error — see §4.)

Phases:
- **A (CPU, first):** plant splits + Tiers 0–2 → canonical `smart_persistence_s2.json`
  reference + uk_pv CSV export.
- **B (GPU pool):** up to `NUM_GPUS` jobs at once (auto-detected via `nvidia-smi`,
  default 8), each pinned to one GPU. Heterogeneous envs (uv / uv) are handled
  per task. Per-task logs in `logs/orchestrator/<jobid>/<task>.log`.
- **C (aggregate):** `aggregate_all.py` → **`results/ALL_RESULTS.md` + `.json`**,
  then a run summary listing each task as OK / FAIL / SKIP.

Knobs (`--export=ALL,KEY=VAL`):

| Knob | Default | Meaning |
|---|---|---|
| `NUM_GPUS` | auto (8) | pool size |
| `SEEDS` | `42 43 44` | seeds for trained models |
| `RUN_LOPO` | `0` | also run goes_pvdaq leave-one-plant-out (heavy, §4.1) |
| `DATA` / `IMAGES_H5` | `$TEAM_SCRATCH/data/...` | dataset of record |
| `ENV_TIMEVLM`, `ENV_CROSSVIVIT`, … | model name | uv env names |

### SLURM sizing

The header requests `--gres=gpu:8 --cpus-per-task=32 --qos=boost_qos_lprod
--time=1-00:00:00` on a single node. Leonardo Booster nodes are **4×A100** —
if you get 4 GPUs, the pool auto-adapts (or set `NUM_GPUS=4`). For a true 8-GPU
run on 2 nodes you'd need a multi-node variant (the current pool is single-node).

---

## 3. Output

```
results/ALL_RESULTS.md     # headline table (all tiers) + §4.4 aggregation vs Smart Persistence
results/ALL_RESULTS.json   # same, machine-readable
results/<model>_<tag>.json # per-baseline raw metrics (NMAE/NRMSE/SS/CRPS/ramp, per plant)
logs/orchestrator/<jobid>/ # per-task stdout/stderr
logs/slurm/<jobid>_*.{out,err}
```

`ALL_RESULTS.md` has one row per (model, scenario tag); SS = 1 − NRMSE /
NRMSE(Smart Persistence) within the same scenario.

---

## 4. Gating — why a baseline is skipped

`run_all_baselines.sh` never aborts the whole run; it skips any baseline whose
env/weights aren't ready and logs the reason in the final summary. Checklist:

| Baseline | Needs |
|---|---|
| T3 / cora | nothing extra (uv `tier3` group cached on login node) |
| ts_rag / cross_rag | uv env + `UKPV_CSV_DIR` + `RAG_BASE_CKPT` + `RAG_MIXER_CKPT` |
| time_vlm | uv env `timevlm` |
| visionts_pp | uv env `visionts` + `MAE_CKPT` |
| unicast | uv env `unicast` + `VISION_MODEL_PATH` + `CHRONOS_PATH` + `IMAGES_H5` |
| aurora | uv env `aurora` + `AURORA_CKPT` |
| crossvivit / sunset | uv env + `IMAGES_H5` |
| solar_vlm | `SOLARVLM_DIR` (its own repo + venv) |

---

## 5. Re-running a single baseline

Each baseline keeps its own script — submit it directly (1 GPU):

```bash
sbatch --export=ALL,VENV_NAME=crossvivit,IMAGES_H5=$TEAM_SCRATCH/data/images_all.h5 \
       scripts/slurm_crossvivit.sh
sbatch --export=ALL,MODELS="chronos2_ft cora",SCENARIO=s2 scripts/slurm_baselines.sh
```

Then re-aggregate:

```bash
uv run python scripts/aggregate_all.py --results results \
    --md results/ALL_RESULTS.md --json results/ALL_RESULTS.json
```

---

## 6. Troubleshooting

- **`HF_HOME missing` / offline download error** → the login precache didn't run
  or `$TEAM_SCRATCH` differs; re-run `precache_login.sh` on the login node.
- **A Tier-5/6 task fails immediately** → its uv env or a checkpoint is
  missing; check `logs/orchestrator/<jobid>/<task>.log` and §4.
- **`uv run` tries to hit the network on the compute node** → ensure
  `UV_OFFLINE=1 UV_NO_SYNC=1` (the orchestrator sets them; if you call a
  `slurm_*.sh` by hand, it sets them too).
- **Different ISCRA-C account** → submit with `sbatch --account=<your_account> …`.
- **Quick smoke** → `sbatch --qos=boost_qos_dbg --time=00:30:00 …` (30-min cap).

See `docs/experiments/TIER{4,5,6}_INTEGRATION.md` for per-tier detail and
`docs/experiments/BASELINE_PROTOCOL.md` for the fairness contract.
