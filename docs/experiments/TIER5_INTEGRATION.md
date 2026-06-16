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
| **UniCast** | uk_pv multimodal (images) | `Y` + real CLIP/BLIP frames + text | ✅ via `tier5/uk_export.py` | P1 |
| **Aurora** | uk_pv (TS + **text**) | `Y` + BERT text (**no image branch**) | ✅ via `tier5/uk_export.py` | P2 |

Time-VLM / VisionTS++ render the series itself as an image and need **no satellite
frames** → they run on the numerical uk_pv track and match our `Y → ŷ` contract.
UniCast needs **real frames** — available in `images_all.h5` (pointer `image_h5_index`). Aurora is
**TS + text** (its `Aurora_Single_Dataset` reads a CSV + JSON text, no image input), so
uk images do not apply; it was blocked on the per-window text. `tier5/uk_export.py`
emits each model's native on-disk format from the shared `tier6.uk_multimodal` bridge,
so both now run on uk_pv (gated only on their backbone weights — CLIP/BLIP + Chronos-Bolt
for UniCast, the Aurora checkpoint).

---

## 0. Shared: uk_pv → upstream inputs

- **Time-VLM** uses the Informer/Time-Series-Library harness (`run.py`, `--data custom`,
  `Dataset_Custom`): an Informer CSV `date,<cols>,OT`. **Reuse the Tier-4 bridge** —
  `tier4/vendor/export_ukpv.py` already emits exactly this (`uk_pv_test_<site>.csv`,
  `uk_pv_train.csv`). No new exporter needed.
- **VisionTS++** uses `uni2ts`/GluonTS datasets → export uk_pv as a GluonTS `FileDataset`
  (one series per plant; reuse `common.windows.build_site_series` for the native grid).
- **UniCast** (real images) and **Aurora** (TS + text) consume the uk_pv multimodal
  windows via `tier5/uk_export.py`, which reuses the shared `tier6.uk_multimodal` bridge
  (curated `Y` + `images_all.h5` frames + covariate-templated text) and writes each
  model's native on-disk format — UniCast: `inputs.pt`/`targets_<H>.pt`/`img/`; Aurora:
  per-series CSV + JSON text. No separate multimodal pipeline needed.

Capacity de-normalisation + the baseline-contract check on outputs reuse
`tier4/vendor/contract_check.py --predictions <npz>` (shape (N,H[,1]), finite, [0,1]).

> **Dataset of record** (DATASET_CONTRACT.md §1.0): `thesis-dataset/dataset_all.parquet`
> + `images_all.h5`, canonical frame pointer `image_h5_index` (both `uk_pv` 128px
> gray and `goes_pvdaq` 256px RGB). **Code repoint needed:** the shared
> `tier6.uk_multimodal` bridge (`DEFAULT_H5`, `FRAME_IDX_COL`) still hardcodes the
> **now-removed** files — point it at `images_all.h5` / `image_h5_index` to run on
> the dataset of record and to add a `goes_pvdaq` multimodal run.

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

## 4. Metrics back into our pipeline (wired)

The per-model SLURM scripts already call **`scripts/import_predictions.py`**, which reduces
the dumped `*_pred.npz` to the same result JSON the in-repo baselines write
(`PerPlantAccumulator` → macro NMAE/NRMSE/SS/CRPS, per plant), so
`scripts/summarize_ukpv.py` and `make_tables.py` render the Tier-5 rows:

```bash
uv run python scripts/import_predictions.py --model time_vlm --tag s2_ukpv \
    --glob 'tier5/vendor/time_vlm/results/*/uk_pv_test_*_pred.npz' \
    --reference results/smart_persistence_s2_ukpv.json
# → results/time_vlm_s2_ukpv.json
```

Two caveats are written into each result manifest and **must** be respected when reading
the table (they also apply to the Tier-4 RAG originals):
- **Daylight mask = `true > 0`** (a proxy; night `norm_power` is exactly 0), not the exact
  clear-sky daylight mask of Tiers 0-4 — a few daytime near-zero overcast steps may drop.
- **Native eval windows**: each harness uses its own windowing (Time-VLM = TSLib test
  split; VisionTS++ `run_ukpv.py` = our non-overlapping windows), so these are **not
  bit-aligned** with Tiers 0-4. There is no per-window loss sidecar ⇒ no DM/bootstrap vs
  Smart Persistence; compare via **SS / win-rate / rank** (§4.4), not pooled raw NMAE.

## 5. Status

- [x] Original code vendored (`tier5/vendor/{time_vlm,visionts_pp,unicast,aurora}`) + provenance.
- [x] **Dedicated per-model SLURM scripts** (train+eval): `slurm_{time_vlm,visionts_pp,unicast,aurora}.sh`.
- [x] Time-VLM reuses `export_ukpv.py` (+ `uk_pv_train_stacked.csv` for univariate training);
      prediction-dump patch landed (`exp_long_term_forecasting.py`).
- [x] VisionTS++ `run_ukpv.py` zero-shot runner over the uk_pv CSVs (no GluonTS export needed).
- [x] Prediction-contract check wired into every script (`tier4/vendor/contract_check.py`).
- [x] Metric import wired: `scripts/import_predictions.py` (npz → results JSON) called by
      each SLURM script; `summarize_ukpv.py` + `make_tables.py` carry the Tier-5 rows.
- [x] UniCast on uk_pv: `tier5/uk_export.py --model unicast` builds its image layout
      from `images_all.h5`; `slurm_unicast.sh` exports → trains → per-plant test
      (`--dump_npz`, added) → import (tag `s2_ukpv_mm`). Export verified on real data;
      model run gated only on the CLIP/BLIP + Chronos-Bolt weights.
- [x] Aurora on uk_pv: it is **TS + text** (no image branch); `tier5/uk_export.py
      --model aurora` emits per-series CSV + weather text; `slurm_aurora.sh` consumes it.
      Export verified; eval-output → npz dump is the remaining cluster step.
- [ ] First **cluster validation** run (none of this is laptop-runnable — verify on Leonardo).

Tier-5 is **not** an in-process registry baseline (unlike Tiers 0-4): the upstream stacks
are too heavy and conflict with our venv, so they run from their own code/env like the
Tier-4 RAG originals. `make_tables.py` ingests their results by file stem.
