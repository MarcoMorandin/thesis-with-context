# Tier-5 — running the *original* multimodal-TS baselines

Tier 5 (BASELINE_COMPARISON.md §1) = generic vision/text-augmented forecasters. We run the
authors' **original code**, vendored under `baselines/tier5/vendor/` (`VENDOR_NOTICE.md` =
SHAs + licensing; 3 of 4 carry **no license** — research-repro only). Adapted to our
contract/dataset, not reimplemented. **Cluster-only** (heavy VLM/MAE/Chronos stacks, GPU);
**not runnable on the laptop** and each needs its **own env** (deps conflict with `baselines/`).

| Model | Track | Inputs | Runnable now? | Prio |
|---|---|---|---|---|
| **Time-VLM** | numerical (uk_pv) | `Y` → pseudo-image (+auto text) | ✅ yes | P0 |
| **VisionTS++** | numerical (uk_pv) | `Y` → image (MAE) | ✅ yes | P2 |
| **UniCast** | multimodal | `Y` + real frames + text | ⛔ needs image track | P1 |
| **Aurora** | multimodal | `Y` + real frames + text | ⛔ needs image track | P2 |

Two of four render the series itself as an image and need **no satellite frames** → they run
on the numerical uk_pv track today and match our `Y → ŷ` contract. UniCast/Aurora consume
**real** image+text → blocked on the multimodal track (skippd / goes16_nsrdb, downloading).

---

## 0. Shared: uk_pv → upstream inputs

- **Time-VLM** uses the Informer/Time-Series-Library harness (`run.py`, `--data custom`,
  `Dataset_Custom`): an Informer CSV `date,<cols>,OT`. **Reuse the Tier-4 bridge** —
  `tier4/vendor/export_ukpv.py` already emits exactly this (`uk_pv_test_<site>.csv`,
  `uk_pv_train.csv`). No new exporter needed.
- **VisionTS++** uses `uni2ts`/GluonTS datasets → export uk_pv as a GluonTS `FileDataset`
  (one series per plant; reuse `common.windows.build_site_series` for the native grid).
- **UniCast / Aurora** need the per-window **image + text** tensors of the multimodal track
  (DATASET_CONTRACT `V` frames + generated weather text) — produced by the multimodal data
  pipeline, not the numerical parquet. Defer until that data lands.

Capacity de-normalisation + the baseline-contract check on outputs reuse
`tier4/vendor/contract_check.py --predictions <npz>` (shape (N,H[,1]), finite, [0,1]).

## 1. Environments (one per model; never share the `baselines/` venv)

```bash
# Time-VLM (CLIP/BLIP2 VLM + TSLib)
conda create -n timevlm python=3.10 && conda activate timevlm
pip install -r baselines/tier5/vendor/time_vlm/requirements.txt   # torch, transformers, einops, CLIP

# VisionTS++ (uni2ts + vision MAE)
conda create -n visionts python=3.10 && conda activate visionts
pip install -e baselines/tier5/vendor/visionts_pp                  # uni2ts, gluonts, lightning, timm

# UniCast (Chronos/Timer + vision/text encoders)   — multimodal track
conda create -n unicast python=3.10 && pip install -r .../unicast/requirements.txt
# Aurora (HF generative MTSFM)                      — multimodal track
conda create -n aurora python=3.10 && pip install -r .../aurora/requirements.txt
```

## 2. Login-node prep (compute nodes are offline)

Cache every backbone on the login node (see `scripts/login_node_prep.sh` pattern):
- Time-VLM: the VLM weights (`openai/clip-vit-base-patch32` or the `--vlm_type` choice).
- VisionTS++: the continual-pretrained MAE checkpoint (HF `Lefei/VisionTSpp`).
- UniCast: Chronos/Timer backbone + vision/text encoder weights.
- Aurora: the Aurora checkpoint (`utils/download_ckpt.py`).
Then compute jobs run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## 3. Run recipes — one dedicated SLURM script per model (train + eval)

Each model has its **own** offline-guarded script that does export → train → eval →
contract-check end-to-end. Submit from `baselines/`; everything is set up in the script
(no edits to vendored code at run time — the needed edits were made on push, see
`tier5/vendor/VENDOR_NOTICE.md` "Adaptations").

| Model | Script | Train | Eval |
|---|---|---|---|
| Time-VLM | `scripts/slurm_time_vlm.sh` | on `uk_pv_train_stacked.csv` | each test plant (reuses the checkpoint) |
| VisionTS++ | `scripts/slurm_visionts_pp.sh` | — (zero-shot MAE) | `run_ukpv.py` over test plants |
| UniCast | `scripts/slurm_unicast.sh` | multimodal (gated) | multimodal (gated) |
| Aurora | `scripts/slurm_aurora.sh` | fine-tune (gated) | multimodal (gated) |

```bash
sbatch --export=ALL,CONDA_ENV=timevlm,DATA=<parquet> scripts/slurm_time_vlm.sh
sbatch --export=ALL,CONDA_ENV=visionts,MAE_CKPT=<ckpt>,DATA=<parquet> scripts/slurm_visionts_pp.sh
# UniCast / Aurora fail loud until the multimodal dataset (frames+text) exists:
sbatch --export=ALL,CONDA_ENV=unicast,MM_TRAIN=…,MM_TEST=…,MM_TEXT=…,CHRONOS_PATH=…,VISION_MODEL=…,TEXT_MODEL=… scripts/slurm_unicast.sh
sbatch --export=ALL,CONDA_ENV=aurora,AURORA_CKPT=…,MM_DATASET=… scripts/slurm_aurora.sh
```
Time-VLM / VisionTS++ run at seq_len=24/pred_len=12 (our protocol) on uk_pv now.

## 4. Metrics back into our pipeline

Same as Tier 4 (TIER4_RAG_INTEGRATION.md §6): dump per-window `pred`/`true` to `.npz`, invert
each model's normalisation back to `norm_power`, feed `(pred, true, mask·daylight)` to
`common/runner.py`'s metric core, register results under `time_vlm` / `visionts_pp` /
`unicast` / `aurora` in `results/` so `make_tables.py` (Tier-5 rows) picks them up.

## 5. Status

- [x] Original code vendored (`tier5/vendor/{time_vlm,visionts_pp,unicast,aurora}`) + provenance.
- [x] **Dedicated per-model SLURM scripts** (train+eval): `slurm_{time_vlm,visionts_pp,unicast,aurora}.sh`.
- [x] Time-VLM reuses `export_ukpv.py` (+ `uk_pv_train_stacked.csv` for univariate training);
      prediction-dump patch landed (`exp_long_term_forecasting.py`).
- [x] VisionTS++ `run_ukpv.py` zero-shot runner over the uk_pv CSVs (no GluonTS export needed).
- [x] Prediction-contract check wired into every script (`tier4/vendor/contract_check.py`).
- [ ] Metric import (npz → NMAE/SS via `common/runner.py`) + `make_tables.py` rows.
- [ ] First **cluster validation** run (none of this is laptop-runnable — verify on Leonardo).
- [ ] UniCast / Aurora: blocked on the multimodal track (image+text data).

Tier-5 is **not** an in-process registry baseline (unlike Tiers 0-4): the upstream stacks
are too heavy and conflict with our venv, so they run from their own code/env like the
Tier-4 RAG originals. `make_tables.py` ingests their results by file stem.
