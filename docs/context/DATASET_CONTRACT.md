# Dataset Contract

This document defines the schema, file structures, expected modalities, and tensor formatting for the standardized dataset used in PV power forecasting. All models (including the main foundation model and all baseline implementations) must consume the dataset according to this contract.

No ETL or raw data processing code should be present in the model or baseline codebases. They must read directly from the standardized paths defined here.

---

## 1. Physical Location and Directory Structure

The dataset is mounted/stored as a read-only volume at:
`/Volumes/SSD/standardized-dataset/`

Two tracks live under this root:

```
/Volumes/SSD/standardized-dataset/
├── numerical/                  # curated numerical track (tiers 0-4)
│   ├── all.parquet             # unified raw extraction
│   └── all_curated.parquet     # curated table (see §1bis)
├── images.h5                   # packed frames for the numerical track
├── images_uk128.h5             # 128px uk_pv frame variant
├── solar/                      # multimodal track (tiers 5-6, PVTSFM)
│   ├── skippd/
│   ├── solarnet/
│   └── goes16_nsrdb/
└── meteorology/
    ├── earthnet2021/
    ├── era5_eu/
    └── meteonet/
```

### 1bis. Numerical curated track (`numerical/all_curated.parquet`)

Produced by `dataset_exploration/curate_dataset.py` from `all.parquet`. One flat table, datasets `uk_pv` (100 plants, 30-min cadence) and `goes_pvdaq` (10 plants, 15-min cadence), native grids preserved, no gap interpolation. Key columns:

| Column group | Columns |
| :--- | :--- |
| Identity | `dataset`, `site_id`, `station_id`, `camera_id`, `timestamp_utc`, `latitude`, `longitude` |
| Target | `power_w`, `norm_power` (= power / audited `installed_power_w`, in [0,1]; NaN on outage/stuck rows) |
| Weather covariates | `temperature_2m`, `shortwave_radiation`, `direct_radiation`, `diffuse_radiation`, `direct_normal_irradiance`, `cloudcover`, `windspeed_10m`, `precipitation` |
| Solar geometry / clear-sky | `solar_zenith`, `solar_azimuth`, `clearsky_ghi` (Haurwitz), `kt`, `csi` (NaN below 50 W/m² clear-sky), `doy_sin`, `doy_cos`, `solar_time` |
| Quality flags | `capacity_fixed`, `outage_flag`, `stuck_flag`, `night_clamped`, `bad_site_flag` |
| Frame pointers | `image_path`, `image_h5_index`, `image_uk128_index` (into `images.h5` / `images_uk128.h5`) |

Splits for this track are **not** in `metadata.json`: they are generated once (seed 42) and committed to `baselines/configs/splits.json`; `baselines/common/splits.py` asserts train/val/test plant disjointness at every load. Baseline code consumes this table through the windowing adapter in `baselines/common/windows.py`, which emits the numerical subset of the canonical dict in §4.

### Files Present Per Dataset Folder
Each dataset directory contains a standard set of files:
1. `timeseries.parquet`: Tabular time series including targets, covariates, and target masks.
2. `frame_index.parquet`: Index mapping timestamps and entity IDs to visual frames.
3. `frames/`: Directory containing visual frames (either JPEGs or float16 NPZ arrays).
4. `graph.json`: Spatial graph representation of the entities, including a precalculated adjacency matrix.
5. `metadata.json`: Dataset-wide configurations, coordinate mapping, and normalization factors.

---

## 2. Table Schemas

### `timeseries.parquet`
This table holds the primary temporal records. Columns are divided into targets and covariates, including normalized versions of each.

| Column | Type | Description |
| :--- | :--- | :--- |
| `entity_id` | `int32` / `int64` | Identifier for the plant / location / grid cell. |
| `timestamp_unix` | `int64` | Epoch timestamp in seconds. |
| `{target_cols}` | `float32` | Raw target variable (e.g., PV power output, GHI, NDVI). |
| `{target_cols}_norm` | `float32` | Min-max or z-score normalized target (main model target). |
| `{cov_cols}` | `float32` | Raw covariate variables (e.g., temperature, clear sky GHI, cloud cover). |
| `{cov_cols}_norm` | `float32` | Normalized covariates. |
| `mask_target` | `float32` (0 or 1) | Binary mask indicating target availability (1 = valid, 0 = missing/night). |

### `frame_index.parquet`
This table matches temporal intervals to visual files.

| Column | Type | Description |
| :--- | :--- | :--- |
| `entity_id` | `int32` / `int64` | Identifier for the plant / location / grid cell. |
| `timestamp_unix` | `int64` | Epoch timestamp in seconds. |
| `rel_path` | `string` | Relative path to the frame file in `frames/`. |
| `mask_visual` | `float32` (0 or 1) | Binary mask indicating frame validity / availability. |

---

## 3. Visual Modalities and Frames

Visual files in the `frames/` directory differ by dataset:

* **JPEG/PNG (Sky Cameras)**: Used in ground-station solar datasets (e.g., `skippd`, `solarnet`; the numerical-track `image_path` files are PNG). Stored as standard `RGB` images. Loaded and normalized to range `[0, 1]`.
* **NPZs (Satellite Frames)**: Used in regional/satellite datasets (e.g., `goes16_nsrdb`, `earthnet2021`). Stored as `float16` numpy zip archives containing spatial channels (e.g., multispectral bands or pre-extracted variables).
  * Load parameters must specify the `npz_key` (default is `"frame"`).
  * Values are normalized to `[0, 1]` per band during loading.

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
