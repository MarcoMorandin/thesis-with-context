# Tier-6 — running the *original* PV-specialized multimodal baselines (domain SOTA)

Tier 6 (BASELINE_COMPARISON.md §1) = PV-domain multimodal forecasters — the
strongest "did you compare to the people who actually do solar?" row. We run the
authors' **original code**, vendored under `baselines/tier6/vendor/`
(`VENDOR_NOTICE.md` = SHAs + licensing; both **MIT**). Adapted to our
contract/dataset via a thin per-model `run_ukpv.py` (the upstream model code is
unmodified), not reimplemented. **Cluster-only** for full runs (vision CNN /
cross-attention over GPUs); each needs its **own env** (deps conflict with
`baselines/`).

| Model | Inputs | Runnable on uk_pv? | Prio |
|---|---|---|---|
| **Solar-VLM** | `Y, X_cov, V` + text | ✅ ported — see `baselines/solar_vlm/` | P0 |
| **SUNSET** | `V` sky/satellite + `Y` PV history | ✅ `run_ukpv.py` | P0 |
| **CrossViViT** | `V` satellite + `Y`/cov cross-attention | ✅ `run_ukpv.py` (with approximations) | P0 |

Both vendored models consume **real frames** (unlike Tier-5 Time-VLM / VisionTS++
which render the series as a pseudo-image). The dataset of record carries them in
`images_all.h5` (per-site groups `<dataset>_<site>`; `uk_pv` 128px gray, `goes_pvdaq`
256px RGB), aligned to the curated `Y` by the canonical `image_h5_index` pointer
(DATASET_CONTRACT.md §1.0). Solar-VLM, the third P0 domain-SOTA model,
is already ported (`baselines/solar_vlm/`) and is **not** re-vendored here.
SPIRIT (P1) and PV-VLM / M3S-Net / MDCTL-MCI (P2) are cite-only for now.

---

## 0. Shared bridge: uk_pv → (Y, V) windows

`tier6/uk_multimodal.py` (`UKMultimodalDataset`) is the single in-repo,
laptop-importable feed for both runners. It reuses `common.windows` for *all*
numerical/fairness logic (disjoint plant splits, NaN handling, deterministic
future covariates, seasonal reference) and adds, per window:

- `V` (T, 1, S, S) in [0,1] — the satellite frames over the history window,
  read straight from `images_all.h5[<dataset>_<site>]["images"]` by
  `image_h5_index`, average-pooled to `img_size` (default 64);
- `mask_visual` (T,) — 1 where a frame exists on that 30-min step (daylight);
- `latlon` — plant coordinates (for CrossViViT's station/grid coords).

No ETL: frames come directly from the dataset-of-record HDF5 (`images_all.h5`). The disjoint plant
split + capacity-normalised `norm_power` target are shared with Tiers 0-5, so
the fairness contract is identical.

## 1. Environments (one per model; never share the `baselines/` venv)

```bash
# CrossViViT (PyTorch + einops; original RoCrossViViT model)
conda create -n crossvivit python=3.10 && conda activate crossvivit
pip install -r baselines/tier6/vendor/crossvivit/requirements.txt   # torch, einops, h5py, pyarrow, pandas
# (run_ukpv.py drives the model directly — it does not need lightning/hydra)

# SUNSET (TensorFlow 2 / Keras)
conda create -n sunset python=3.10 && conda activate sunset
pip install tensorflow h5py pyarrow pandas numpy                    # TF2 + data deps
```

Both runners import `tier6.uk_multimodal` + `common.*` from the repo, so submit
from `baselines/` (the scripts put it on `PYTHONPATH` via the file layout).

## 2. Login-node prep (compute nodes are offline)

No pretrained weights — both models train from scratch on uk_pv. Stage the data
to `$TEAM_SCRATCH`. **Dataset of record** (DATASET_CONTRACT.md §1.0):
- `thesis-dataset/dataset_all.parquet` (the `Y` + canonical `image_h5_index` pointer),
- `thesis-dataset/images_all.h5` (the satellite frames; `uk_pv` 128px gray + `goes_pvdaq` 256px RGB).

> **Code repoint needed:** the in-repo bridge `tier6/uk_multimodal.py`
> (`DEFAULT_H5`, `FRAME_IDX_COL`) and the SLURM `DATA`/`IMAGES_H5` defaults still
> hardcode the **now-removed** files. Point them at
> `thesis-dataset/dataset_all.parquet` + `images_all.h5` with frame pointer
> `image_h5_index` (works for both datasets — enables a `goes_pvdaq` multimodal
> run too).

## 3. Run recipes — one dedicated SLURM script per model (train + eval)

Each model has its **own** script (train → eval per held-out plant →
contract-check → metric-import), submitted from `baselines/`:

| Model | Script | Runner | Output |
|---|---|---|---|
| CrossViViT | `scripts/slurm_crossvivit.sh` | `tier6/vendor/crossvivit/run_ukpv.py` | `crossvivit_<site>_pred.npz` |
| SUNSET | `scripts/slurm_sunset.sh` | `tier6/vendor/sunset/run_ukpv.py` | `sunset_<site>_pred.npz` |

```bash
sbatch --export=ALL,CONDA_ENV=crossvivit,DATA=<dataset_all.parquet>,\
       IMAGES_H5=<images_all.h5> scripts/slurm_crossvivit.sh
sbatch --export=ALL,CONDA_ENV=sunset,DATA=<dataset_all.parquet>,\
       IMAGES_H5=<images_all.h5> scripts/slurm_sunset.sh
```

Both default to seq_len=24 / pred_len=12 (our protocol), img_size=64, stride 3.

### How each runner uses the original model

- **SUNSET** (`run_ukpv.py`) transcribes the exact `SUNSET_forecast.ipynb` Keras
  graph and feeds it the sky-image stack `V` + PV history `y_hist`. Only change:
  the final Dense head is widened 1 → H (the original forecasts a single
  15-min-ahead step; our protocol forecasts H), trained with masked MSE.
- **CrossViViT** (`run_ukpv.py`) imports the original
  `src.models.cross_vivit.RoCrossViViT` **unchanged**. The last `pred_len` steps
  of each history window form CrossViViT's shared context window (satellite `V` +
  PV/covariate `ts`), and the model forecasts the next `pred_len` PV steps.
  **Approximations** (uk_pv ≠ the authors' georeferenced DeepLake SunLake):
  single-channel 128px→S crops, no optical-flow/elevation, synthetic per-pixel
  `ctx_coords` around the plant. These weaken CrossViViT's spatial grounding —
  report the row with this caveat (also in `tier6/vendor/VENDOR_NOTICE.md`).

## 4. Metrics back into our pipeline (wired)

Both per-model scripts call **`scripts/import_predictions.py`**, reducing the
dumped `<model>_<site>_pred.npz` to the same result JSON the in-repo baselines
write (`PerPlantAccumulator` → macro NMAE/NRMSE/SS/CRPS, per plant), tagged
`s2_ukpv_mm` (uk_pv multimodal), so `scripts/summarize_ukpv.py` and
`make_tables.py` render the Tier-6 rows:

```bash
uv run python scripts/import_predictions.py --model sunset --tag s2_ukpv_mm \
    --glob 'tier6/vendor/sunset/results_ukpv/sunset_*_pred.npz' \
    --reference results/smart_persistence_s2_ukpv.json
# → results/sunset_s2_ukpv_mm.json
```

The same two caveats as Tiers 4–5 apply and are written into each result manifest:
- **Daylight mask = `true > 0`** (proxy; night PV is ~0), not the exact clear-sky
  daylight mask of Tiers 0-4.
- **Native eval windows**: each runner uses `UKMultimodalDataset`'s windows
  (stride 3, daylight-valid futures), so these are **not bit-aligned** with the
  Tiers 0-4 sidecars ⇒ no DM/bootstrap vs Smart Persistence; compare via
  **SS / win-rate / rank** (§4.4), not pooled raw NMAE.

## 5. Status

- [x] Original code vendored (`tier6/vendor/{crossvivit,sunset}`) + provenance/SHAs/MIT.
- [x] Shared uk_pv multimodal bridge `tier6/uk_multimodal.py` (Y + frames from `images_all.h5`),
      reusing `common.windows`; verified on real data.
- [x] Per-model `run_ukpv.py` runners driving the original models on uk_pv;
      verified end-to-end on real data (CPU smoke test): SUNSET (TF2, 13.7M
      params) and CrossViViT (RoCrossViViT, 3.8M params) both train + predict
      (N,12) finite.
- [x] **Dedicated per-model SLURM scripts** (train+eval) on uk_pv:
      `slurm_{crossvivit,sunset}.sh`; contract check + import_predictions wired
      (tag `s2_ukpv_mm`).
- [x] Solar-VLM (3rd P0 domain-SOTA) already ported — `baselines/solar_vlm/`.
- [ ] First **full cluster sweep** (laptop runs only the CPU smoke test — run the
      real multi-epoch training on Leonardo).

Tier-6 is **not** an in-process registry baseline (like Tiers 5 / the Tier-4 RAG
originals): the upstream stacks are heavy and conflict with our venv, so they run
from their own code/env. `make_tables.py` ingests their results by file stem.
