# Dataset — contract of record

**Canonical for**: schema, physical paths, plant splits, the batch dict. Anything else
needing these facts must link here, not restate them.
Companions: [protocol.md](protocol.md) (how the splits are *used*) ·
[../report/REPORT.md](../report/REPORT.md) (exploratory statistics, provenance, construction).

This document defines the schema, file structures, expected modalities, and tensor formatting for the standardized dataset used in PV power forecasting. All models (including the main foundation model and all baseline implementations) must consume the dataset according to this contract.

No ETL or raw data processing code should be present in the model or baseline codebases. They must read directly from the standardized paths defined here.

---

## 1. Physical Location and Directory Structure

### 1.0 Experiment dataset of record — `/leonardo_scratch/fast/IscrC_MTSFM/data_v2/`

All experiments run against the **consolidated, experiment-ready dataset**: one
flat numerical table plus one packed image archive covering **both** numerical-track
datasets (`uk_pv` and `goes_pvdaq` — `goes_pvdaq` is now fully present, see §1.0a):

Verified on the cluster 2026-08-25. The `_v2` is the **directory** name; the files
inside keep the canonical names:

```
/leonardo_scratch/fast/IscrC_MTSFM/data_v2/
├── dataset_all.parquet     #  92,099,550 B — 1,337,654 rows × 35 cols; see §1bis
└── images_all.h5           # 105,237,477,184 B — v2 non-HRV, 110 groups; see §2
```

So `pv_record.py`'s hard-coded `dataset_all.parquet` / `images_all.h5` are
correct as written — only `DATA_DIR` moves. The local sync at
`/Volumes/dataset/dataset/` is the same data with `_v2` appended to the
*filenames*; sizes match byte-for-byte, and rebuilding the test windows from it
reproduces `n_steps = 165,295` exactly.

**V-JEPA latent cache of record**:
`/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224_nonhrv_sp45`,
built 2026-08-21 from `data_v2` (`_extract_meta.txt`: `arch=vit_large|frames=8|
img=224|window_h=6.0|spacing_min=auto|train_stride=12`; auto spacing resolves to
45 min). This is the cache the s2b run of record consumed, so every vision number
describes v2 imagery.

> ⚠ The obsolete **v1 HRV** cache still exists at `.../uk_pv/vit_large_f8_s224`.
> `slurm_curriculum.sh` used to default to that bare name; it now defaults to the
> `_nonhrv_sp45` cache, and `curriculum_stage.sbatch` treats an absent cache — or
> one with no `_extract_meta.txt` — as fatal rather than silently falling back to
> live encoding or to unknown imagery.

`images_all.h5` packs every frame referenced by the table. Each per-site group
`<dataset>_<site>` holds `images` + `timestamps` (`|S20` ISO-8601, e.g.
`2019-01-01T08:00:00Z`). **v2 stores PNG-encoded bytes** (variable-length
`object` dtype), not raw arrays — `pv_record._decode_frame` detects this via
`_frame_maps_from_h5` and decodes with PIL. Frames are aligned to table rows by
the canonical **`image_h5_index`** pointer — a *local-to-group* index into
`images_all.h5[<dataset>_<site>]["images"]`, timestamp-exact, valid for **both**
datasets (verified row-by-row). The frame grid is denser than the power grid, so
`PVRecordDataset` builds its lookup from the H5's own `timestamps`, not from the
parquet pointer (§2).

#### 1.0a Per-dataset specs

| Dataset | Sites | Rows | Valid power steps | Cadence | Span (UTC) | Frame tensor | Capacity | Region |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| `uk_pv` | 100 | 1,232,862 | 1,217,399 | 30-min (power) / **15-min (frames)** | 2019-01-01 → 2020-12-31 | `(128,128,3)` uint8 RGB, PNG | 1.5–4.0 kW (residential rooftop) | UK (lat 50.7–57.8, lon −5.6–0.5) |
| `goes_pvdaq` | 10 | 104,792 | 103,451 | 15-min | 2019-01-01 → 2019-09-30 | `(256,256,3)` uint8 RGB, PNG | 1.8–408 kW (residential→utility) | US (lat 36.0–39.9, lon −115.2…−75.0) |

Quality flags (`dataset_all.parquet`): `bad_site_flag` on **`uk_pv` 7239, 8587**
and **`goes_pvdaq` 1283, 51**; `outage_flag` 15,486; `stuck_flag` 1,318;
`night_clamped` 1,535. (The committed `goes_pvdaq` split in
`baselines/configs/splits.json` predates these bad-site flags and still lists
`1283`/`51` — reconcile before running `goes_pvdaq`.)

`/leonardo_scratch/fast/IscrC_MTSFM/data_v2/` is the **only** dataset volume — `dataset_all.parquet`
(numerical) + `images_all.h5` (frames), covering both `uk_pv` and `goes_pvdaq`. There
is no separate source/ETL volume.

> **Code note:** any code with a hardcoded data path
> (`baselines/common/config.py::DEFAULT_DATA_PATH`,
> `tier6/uk_multimodal.py::DEFAULT_H5`, the per-model `run_ukpv.py` `--h5` defaults)
> must point inside `/leonardo_scratch/fast/IscrC_MTSFM/data_v2/`, with frame pointer
> `image_h5_index`. `MMTSFM` needs no such edit — `pv_record` resolves both filenames
> from `data.data_dir`, so only `DATA_DIR` changes.

### 1bis. Numerical table (`dataset_all.parquet`)

One flat table, datasets `uk_pv` (100 plants, 30-min cadence) and `goes_pvdaq` (10 plants, 15-min cadence), native grids preserved, no gap interpolation. Key columns:

| Column group | Columns |
| :--- | :--- |
| Identity | `dataset`, `site_id`, `station_id`, `camera_id`, `timestamp_utc`, `latitude`, `longitude` |
| Target | `power_w`, `norm_power` (= power / audited `installed_power_w`, in [0,1]; NaN on outage/stuck rows) |
| Weather covariates | `temperature_2m`, `shortwave_radiation`, `direct_radiation`, `diffuse_radiation`, `direct_normal_irradiance`, `cloudcover`, `windspeed_10m`, `precipitation` |
| Solar geometry / clear-sky | `solar_zenith`, `solar_azimuth`, `clearsky_ghi` (Haurwitz), `kt`, `csi` (NaN below 50 W/m² clear-sky), `doy_sin`, `doy_cos`, `solar_time` |
| Quality flags | `capacity_fixed`, `outage_flag`, `stuck_flag`, `night_clamped`, `bad_site_flag` |
| Frame pointers | `image_h5_index` (**canonical** — local-to-group index into `images_all.h5[<dataset>_<site>]`, both datasets), `image_index` (≈ `image_h5_index`), `image_path` (relative frame path). (`image_uk128_index` is a dead column — it pointed into a removed file; use `image_h5_index`.) |

Splits for this track are generated once (seed 42) and committed to `baselines/configs/splits.json`; `baselines/common/splits.py` asserts train/val/test plant disjointness at every load. Baseline code consumes this table through the windowing adapter in `baselines/common/windows.py`, which emits the numerical subset of the canonical dict in §4.

---

## 2. Frames (`images_all.h5`) — **v2, non-HRV**

Frames live in `images_all.h5` as per-site HDF5 groups `<dataset>_<site>`, each with
`images` (PNG bytes, `object` dtype) + `timestamps` (`|S20` ISO-8601), addressed by
`image_h5_index` (§1.0). **110 groups, 4,103,892 frames, 98 GB.**

| | `uk_pv` | `goes_pvdaq` |
|---|---|---|
| Decoded frame | `(128, 128, 3)` uint8 RGB | `(256, 256, 3)` uint8 RGB |
| Encoding | PNG, ~20 KB/frame | PNG, ~79 KB/frame |
| Cadence | **15-min** | 15-min |
| Frames/site | 39,991 (uniform, all 100 sites) | 7,537–12,655 |
| Diurnal coverage (UTC) | **02:00–16:00**, every day | **10:00–23:00** |
| Span | 2019-01-01 → 2020-12-31 | 2019-01-01 → 2019-09-30 |

### 2.1 Non-HRV: the three channels are real, and they work at night

v2 replaced the earlier single-channel HRV crops. **This supersedes any doc,
comment, or memory describing `uk_pv` frames as grayscale or daylight-only.**
Measured on `uk_pv_10793` (2026-08-25):

* **Three genuinely distinct bands.** Inter-channel correlation R–G 0.91, R–B 0.67,
  G–B 0.83; mean `|R−G|` 16.1 DN, `|G−B|` 24.5 DN. A grayscale-replicated frame
  would give correlation 1.0 and difference 0. `pv_record._prep_frame` therefore
  takes the native 3-channel path — **no gray→RGB replication**, chroma features are
  live.
* **Night frames carry structure.** December pre-dawn (02:00–05:00 UTC, well before
  UK sunrise) frames have mean 134 / std 37, versus midday mean 123 / std 38. These
  are IR bands, not visible, so cloud fields are observable through the night. This
  is exactly what the non-HRV re-extraction was for.
* **Advection decorrelates in ~2 h.** Frame-to-frame mean absolute difference is
  8.3 DN at Δt = 15 min and 35.3 DN at Δt = 2 h, against a within-frame std of
  38.6 DN. Cloud-motion information is effectively exhausted by ~2 h ahead — i.e.
  horizon steps `h ≲ 4` on the `uk_pv` 30-min grid. Treat any claimed visual gain at
  `h > 4` as suspect.

### 2.2 Visual coverage of the scored windows

Measured by rebuilding the protocol test windows from `dataset_all_v2.parquet`
(14 test plants, `T=672`, `H=12`, `stride=H`) and running
`PVRecordDataset._load_vision`'s slot-selection rule (8 frames, 45-min spacing,
±22.5-min tolerance) against the H5 grid. Reproduces `n_steps = 165,295` exactly:

| Origin (UTC) | Windows | Mean frames filled | Windows with 0 frames | Share of scored steps |
|---|---:|---:|---:|---:|
| 07:30 | 10,004 | 7.68 / 8 | 3.9 % | **70.7 %** |
| 13:30 |  9,960 | 7.67 / 8 | 3.8 % | 29.3 % |

`stride = H` admits only two origin times; the 01:30 and 19:30 origins are dropped
by `min_future_valid` (no daylight in their horizon). **96.1 % of scored windows
carry all 8 frames**; only 3.7 % of scored steps have no visual input at all.

> Retired claim: an earlier note held that "all 07:30 origins are a blank-frame
> embedding", diluting every vision number ~2×. That was true of the v1 HRV
> daylight-only archive. It is **false for v2** — the 07:30 window (02:15–07:30) is
> fully covered by IR frames.

Normalize to `[0, 1]` on load (÷255). `pv_record` then applies ImageNet mean/std
(`data.imagenet_norm: true`) because V-JEPA 2's own transform does — note the
`visual_encoder.VisualEncoder.forward` docstring still says "normalized to [0, 1]",
which is stale; the code is correct.

---

## 4. Tensor Output Format (The "Canonical Dict")

The dataset PyTorch adapter (`PVTSFMDataset`) must output a dictionary containing the following keys and tensor formats. 

Here, **\(N\)** is the number of entities in a batch, **\(T\)** is the history window size, **\(H\)** is the forecasting horizon, **\(T_v\)** is the number of video frames, and **\(C\)** indicates channel dimensions.

| Tensor Key | Shape | Type | Range / Content |
| :--- | :--- | :--- | :--- |
| `Y` | `(N, T, C_target)` | `float32` | Normalized historical targets. |
| `Y_future` | `(N, H, C_target)` | `float32` | Target values to predict (ground truth). |
| `X_cov` | `(N, T+H, C_cov)` | `float32` | Historical + future covariates. |
| `V` | `(N, T_v, C_img, H_img, W_img)` | `float32` | Visual frames normalized to `[0, 1]`. |
| `timestamps` | `(T+H,)` | `int64` | Unix epoch timestamps for the entire window. |
| `entity_ids` | `(N,)` | `int64` | Unique IDs for the plants in the batch. |
| `timestamps_v` | `(T_v,)` | `int64` | Unix epoch timestamps for the visual frames. |
| `mask_target` | `(N, T, C_target)` | `float32` | Mask for historical targets. |
| `mask_future` | `(N, H, C_target)` | `float32` | Mask for future targets (1.0 for evaluation). |
| `mask_visual` | `(N, T_v)` | `float32` | Mask indicating validity of each visual frame. |
| `mask_modality_dropout`| `(N, 2)` | `float32` | `[numeric, visual]` dropout masks. |
| `adj_matrix` | `(N, N)` | `float32` | Precomputed spatial adjacency matrix. |

---

## 5. Splits & Cross-Plant Generalization Protocol

Instead of performing few-shot context matching on held-out plants, we utilize a **disjoint cross-plant generalization split**:

1. **Disjoint Entities**: Test plants are completely held out. The model does not see their timeseries or satellite history during training.
2. **Evaluation Scenario**: At inference time, the model is presented with:
   * A short history of target values and covariates `(Y, X_cov)` up to time step `T` for a *held-out* plant.
   * A short history of visual frames `V` up to time step `T`.
3. **Generalization Task**: The model must forecast `Y_future` for the held-out plant based entirely on the spatial/temporal mapping learned from other plants.
4. **Data Splits**: 
   * **Train Split**: Trained on the training set of plants.
   * **Val Split**: Evaluated on disjoint validation plants to monitor convergence and avoid overfitting.
   * **Test Split**: Disjoint test plants, representing the final generalization metric.
