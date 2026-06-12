# Dataset Contract

This document defines the schema, file structures, expected modalities, and tensor formatting for the standardized dataset used in PV power forecasting. All models (including the main foundation model and all baseline implementations) must consume the dataset according to this contract.

No ETL or raw data processing code should be present in the model or baseline codebases. They must read directly from the standardized paths defined here.

---

## 1. Physical Location and Directory Structure

The dataset is mounted/stored as a read-only volume at:
`/Volumes/SSD/standardized-dataset/`

Inside this root, data is organized by modality group and dataset name:

```
/Volumes/SSD/standardized-dataset/
├── solar/
│   ├── skippd/
│   ├── solarnet/
│   └── goes16_nsrdb/
└── meteorology/
    ├── earthnet2021/
    ├── era5_eu/
    └── meteonet/
```

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

* **JPEGs (Sky Cameras)**: Used in ground-station solar datasets (e.g., `skippd`, `solarnet`). Stored as standard `RGB` images. Loaded and normalized to range `[0, 1]`.
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
