# Dataset Contract

This document defines the schema, file structures, expected modalities, and tensor formatting for the standardized dataset used in PV power forecasting. All models (including the main foundation model and all baseline implementations) must consume the dataset according to this contract.

No ETL or raw data processing code should be present in the model or baseline codebases. They must read directly from the standardized paths defined here.

---

## 1. Physical Location and Directory Structure

### 1.0 Experiment dataset of record — `/leonardo_scratch/fast/IscrC_MTSFM/data/`

All experiments run against the **consolidated, experiment-ready dataset**: one
flat numerical table plus one packed image archive covering **both** numerical-track
datasets (`uk_pv` and `goes_pvdaq` — `goes_pvdaq` is now fully present, see §1.0a):

```
/leonardo_scratch/fast/IscrC_MTSFM/data/
├── dataset_all.parquet     # 1,337,654 rows × 35 cols (uk_pv + goes_pvdaq); see §1bis
└── images_all.h5           # 27 GB, 110 per-site HDF5 groups <dataset>_<site>
```

`images_all.h5` packs every frame referenced by the table. Each per-site group
`<dataset>_<site>` holds `images` + `timestamps` (`|S20` ISO-8601, e.g.
`2019-01-01T08:00:00Z`). Frames are aligned to table rows by the canonical
**`image_h5_index`** pointer — a *local-to-group* index into
`images_all.h5[<dataset>_<site>]["images"]`, timestamp-exact, valid for **both**
datasets (verified row-by-row).

#### 1.0a Per-dataset specs

| Dataset | Sites | Rows | Valid power steps | Cadence | Span (UTC) | Frame tensor | Capacity | Region |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| `uk_pv` | 100 | 1,232,862 | 1,217,399 | 30-min | 2019-01-01 → 2020-12-31 | `(N,128,128)` uint8 grayscale | 1.5–4.0 kW (residential rooftop) | UK (lat 50.7–57.8, lon −5.6–0.5) |
| `goes_pvdaq` | 10 | 104,792 | 103,451 | 15-min | 2019-01-01 → 2019-09-30 | `(N,256,256,3)` uint8 RGB | 1.8–408 kW (residential→utility) | US (lat 36.0–39.9, lon −115.2…−75.0) |

Quality flags (`dataset_all.parquet`): `bad_site_flag` on **`uk_pv` 7239, 8587**
and **`goes_pvdaq` 1283, 51**; `outage_flag` 15,486; `stuck_flag` 1,318;
`night_clamped` 1,535. (The committed `goes_pvdaq` split in
`baselines/configs/splits.json` predates these bad-site flags and still lists
`1283`/`51` — reconcile before running `goes_pvdaq`.)

`/leonardo_scratch/fast/IscrC_MTSFM/data/` is the **only** dataset volume — `dataset_all.parquet`
(numerical) + `images_all.h5` (frames), covering both `uk_pv` and `goes_pvdaq`. There
is no separate source/ETL volume.

> **Code note:** any code with a hardcoded data path
> (`baselines/common/config.py::DEFAULT_DATA_PATH`,
> `tier6/uk_multimodal.py::DEFAULT_H5`, the per-model `run_ukpv.py` `--h5` defaults)
> must point at `thesis-dataset/dataset_all.parquet` + `images_all.h5` with frame
> pointer `image_h5_index`.

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

## 2. Frames (`images_all.h5`)

Frames live in `images_all.h5` as per-site HDF5 groups `<dataset>_<site>`, each with
`images` (`uint8`) + `timestamps` (`|S20` ISO-8601), addressed by `image_h5_index`
(§1.0):

* `uk_pv` — `(N, 128, 128)` single-channel satellite crops, 30-min daylight cadence.
* `goes_pvdaq` — `(N, 256, 256, 3)` RGB satellite frames, 15-min daylight cadence.

Normalize to `[0, 1]` on load (÷255); average-pool to a smaller side when a model
needs one. A frame is valid (`mask_visual = 1`) on a step iff that step has a row with
a matching `image_h5_index`/timestamp (daylight); night/outage steps have no frame.

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
