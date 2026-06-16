# Tier-6 — running the *original* PV-specialized multimodal baselines (domain SOTA)

Tier 6 (BASELINE_COMPARISON.md §1) = PV-domain multimodal forecasters — the
strongest "did you compare to the people who actually do solar?" row. We run the
authors' **original code**, vendored under `baselines/tier6/vendor/`
(`VENDOR_NOTICE.md` = SHAs + licensing; both **MIT**). Adapted to our
contract/dataset, not reimplemented. **Cluster-only** (vision CNN / cross-attention
stacks, GPU); each needs its **own env** (deps conflict with `baselines/`).

| Model | Track | Inputs | Runnable now? | Prio |
|---|---|---|---|---|
| **Solar-VLM** | multimodal (SKIPP'D) | `Y, X_cov, V` + text | ✅ ported — see `baselines/solar_vlm/` | P0 |
| **SUNSET** | multimodal (SKIPP'D-native) | `V` sky images + `Y` PV history | ⛔ needs SKIPP'D HDF5 | P0 |
| **CrossViViT** | multimodal (satellite) | `V` satellite + `Y` irradiance | ⛔ needs multimodal frames | P0 |

Both vendored models consume **real frames** (unlike Tier-5 Time-VLM / VisionTS++
which render the series as a pseudo-image) → they live on the **multimodal track**
(skippd / goes16_nsrdb per DATASET_CONTRACT), the same treatment as Tier-5's
UniCast / Aurora: vendored + scaffolded now, gated on that data landing.

Solar-VLM, the third P0 domain-SOTA model, is already ported (`baselines/solar_vlm/`,
runs on SKIPP'D with offline precomputed vision features) and is **not** re-vendored
here. SPIRIT (P1) and PV-VLM / M3S-Net / MDCTL-MCI (P2) are cite-only for now.

---

## 0. Shared: our data → upstream inputs

- **SUNSET** reads a SKIPP'D-style HDF5 (`forecast_dataset.hdf5`) with
  `trainval/images_log` of shape `(N, 16, 64, 64, 3)` (a 16-frame sky-image log
  stack, reshaped to `(64, 64, 24)`) plus the PV-output history and 15-min-ahead
  target. SKIPP'D is SUNSET's **native** dataset and the same sky-image data
  `solar_vlm/` already consumes — reuse that export or the SUNSET
  `data_processing/` notebooks to build the HDF5.
- **CrossViViT** reads per-window **satellite context frames + irradiance `Y`**
  via its `tscontext_datamodule` (native = the DeepLake `hub://crossvivit/SunLake`,
  EUMETSAT satellite + ground stations). For our protocol, point the datamodule at
  the multimodal-track frames (goes16_nsrdb / skippd) + `Y`.

Capacity de-normalisation + the baseline-contract check on outputs reuse
`tier4/vendor/contract_check.py --predictions <npz>` (shape (N,H[,1]), finite, [0,1]),
exactly as Tiers 4–5.

## 1. Environments (one per model; never share the `baselines/` venv)

```bash
# CrossViViT (PyTorch 2.0 + Lightning + Hydra)
conda create -n crossvivit python=3.10 && conda activate crossvivit
pip install -r baselines/tier6/vendor/crossvivit/requirements.txt   # torch, lightning, hydra, einops

# SUNSET (TensorFlow 2.4 / Keras)
conda create -n sunset python=3.9 && conda activate sunset
pip install -r baselines/tier6/vendor/sunset/requirements.txt       # tensorflow~=2.4, h5py, numpy
```

## 2. Login-node prep (compute nodes are offline)

- CrossViViT: cache any released checkpoint + the multimodal frames locally (the
  DeepLake `SunLake` copy, or our goes16_nsrdb/skippd export). Run with
  `HF_HUB_OFFLINE=1`.
- SUNSET: stage the SKIPP'D `forecast_dataset.hdf5` to `$TEAM_SCRATCH`. No HF
  weights (model trains from scratch on SKIPP'D).

## 3. Run recipes — one dedicated SLURM script per model (train + eval)

Each model has its **own** offline-guarded script (export → train → eval →
contract-check → metric-import), submitted from `baselines/`. Both **fail loud**
until the multimodal data exists (same as Tier-5 UniCast / Aurora).

| Model | Script | Train | Eval |
|---|---|---|---|
| CrossViViT | `scripts/slurm_crossvivit.sh` | `main.py experiment=cross_vivit train=True` | `test=True` → `*_pred.npz` |
| SUNSET | `scripts/slurm_sunset.sh` | `run_skippd.py` on SKIPP'D HDF5 | held-out plants → `sunset_*_pred.npz` |

```bash
sbatch --export=ALL,CONDA_ENV=crossvivit,MM_DATA=<frames+Y>,EXPERIMENT=cross_vivit \
       scripts/slurm_crossvivit.sh
sbatch --export=ALL,CONDA_ENV=sunset,SKIPPD_HDF5=<forecast_dataset.hdf5> \
       scripts/slurm_sunset.sh
```

**Adaptations still owed before a real run** (tracked in `VENDOR_NOTICE.md`):

- `crossvivit/` — a datamodule config pointing at our multimodal frames + `Y`, and
  a prediction-dump hook in the Lightning `test_step` writing
  `out_ukpv/<site>_pred.npz` (`pred`,`true` (N,H[,1])).
- `sunset/` — `run_skippd.py`, a self-contained runner converted from
  `SUNSET_forecast.ipynb` (no notebook execution at run time) that trains the
  original Keras model and dumps `sunset_<site>_pred.npz`. The SLURM script
  fails with a clear message until this file exists.

These are deliberately deferred until the multimodal `V` frames are wired — the
vendored upstream code stays pristine until then (diff vs the pinned SHA = the
adaptation).

## 4. Metrics back into our pipeline (wired)

Both per-model scripts call **`scripts/import_predictions.py`**, which reduces the
dumped `*_<site>_pred.npz` to the same result JSON the in-repo baselines write
(`PerPlantAccumulator` → macro NMAE/NRMSE/SS/CRPS, per plant), tagged `s2_mm`
(multimodal track), so `scripts/summarize_ukpv.py` and `make_tables.py` render the
Tier-6 rows:

```bash
uv run python scripts/import_predictions.py --model sunset --tag s2_mm \
    --glob 'tier6/vendor/sunset/results_skippd/sunset_*_pred.npz' \
    --reference results/smart_persistence_s2_ukpv.json
# → results/sunset_s2_mm.json
```

The same two caveats as Tiers 4–5 apply and are written into each result manifest:
- **Daylight mask = `true > 0`** (proxy; night PV is ~0), not the exact clear-sky
  daylight mask of Tiers 0-4.
- **Native eval windows**: each harness uses its own windowing, so these are **not
  bit-aligned** with Tiers 0-4 and carry no per-window loss sidecar ⇒ no
  DM/bootstrap vs Smart Persistence; compare via **SS / win-rate / rank** (§4.4),
  not pooled raw NMAE. The multimodal track (`_mm`) is also a different test set
  from the numerical `uk_pv` track — compare within-track.

## 5. Status

- [x] Original code vendored (`tier6/vendor/{crossvivit,sunset}`) + provenance/SHAs/MIT.
- [x] **Dedicated per-model SLURM scripts** (train+eval): `slurm_{crossvivit,sunset}.sh`.
- [x] Metric import wired (`scripts/import_predictions.py`, tag `s2_mm`); contract
      check (`tier4/vendor/contract_check.py`) called per dumped npz.
- [x] Solar-VLM (3rd P0 domain-SOTA) already ported — `baselines/solar_vlm/`.
- [ ] Wire the multimodal `V` loaders + prediction dump (CrossViViT `test_step`;
      SUNSET `run_skippd.py`) — gated on the multimodal track (skippd/goes16) data.
- [ ] First **cluster validation** run (not laptop-runnable — verify on Leonardo).

Tier-6 is **not** an in-process registry baseline (like Tiers 5 / the Tier-4 RAG
originals): the upstream stacks are heavy and conflict with our venv, so they run
from their own code/env. `make_tables.py` ingests their results by file stem.
