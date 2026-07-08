## 1. Dataset Overview

- **Total records:** 1,297,015
- **Columns:** 35
- **Sites:** 106

| dataset    |    rows |   sites | sampling        | first_ts                  | last_ts                   |
|:-----------|--------:|--------:|:----------------|:--------------------------|:--------------------------|
| goes_pvdaq |   88806 |       8 | 15 minutes | 2019-01-01 00:00:00+00:00 | 2019-09-30 23:45:00+00:00 |
| uk_pv      | 1208209 |      98 | 30 minutes | 2019-01-01 08:00:00+00:00 | 2020-12-31 16:00:00+00:00 |

### 1.1 Organization

Two source datasets (`uk_pv` satellite HRV crops, `goes_pvdaq` GOES-16 crops) merged into a single table `dataset_all.parquet` — one row per `(site_id, timestamp_utc)`. Each row pairs a PV power reading and its weather/solar covariates with a co-registered image frame; pixels live in `images_all.h5`, referenced by the integer `image_*_index` pointers. `norm_power` (power ÷ installed capacity) is the model target; splits are cross-plant on `site_id`.

### 1.2 Columns

| Column | Meaning | Role |
|---|---|---|
| `dataset` | Source dataset: `uk_pv` or `goes_pvdaq` | id — cross-region split key |
| `site_id` | PV plant identifier | id — cross-plant split key |
| `timestamp_utc` | UTC sample time | index — windowing |
| `power_w` | Measured PV power output (W) | raw target; not fed directly |
| `installed_power_w` | Nameplate capacity (W) | normalizer |
| `norm_power` | power_w ÷ installed_power_w | **target** |
| `temperature_2m` | 2 m air temperature (°C) | covariate — observed |
| `shortwave_radiation` | Surface downwelling shortwave (W/m²) | covariate — observed |
| `direct_radiation` | Direct-beam component (W/m²) | covariate — observed |
| `diffuse_radiation` | Diffuse component (W/m²) | covariate — observed |
| `direct_normal_irradiance` | DNI (W/m²) | covariate — observed |
| `cloudcover` | Total cloud cover (%) | covariate — observed |
| `windspeed_10m` | 10 m wind speed (m/s) | covariate — observed |
| `precipitation` | Precipitation (mm) | covariate — observed |
| `solar_zenith` | Solar zenith angle (deg) | covariate — deterministic |
| `solar_azimuth` | Solar azimuth angle (deg) | covariate — deterministic |
| `clearsky_ghi` | Clear-sky GHI model (W/m²) | covariate — deterministic |
| `doy_sin`, `doy_cos` | Cyclic day-of-year encoding | covariate — deterministic |
| `solar_time` | Local apparent solar time (hours) | covariate — deterministic |
| `image_h5_index` | Row index into `images_all.h5` | vision input pointer |
| `latitude`, `longitude` | Site coordinates | metadata — unused (can be used for graph building) |
| `station_id` | Weather station tag | metadata — unused |
| `camera_id` | Imagery channel (HRV/GOES) | metadata — unused |
| `outage_flag` | Flag: outage / zero-output anomaly | curation only, unused |
| `night_clamped` | Flag: nighttime power clamped to 0 | curation only, unused |

### 1.3 Construction & sources

The dataset is assembled fusing four public sources into one standardized `(site_id, timestamp_utc)` table plus co-registered image frames:

- **PV power — UK** ([`openclimatefix/uk_pv`](https://huggingface.co/datasets/openclimatefix/uk_pv), HF): 30-minutely generation (2019–2020) and per-site metadata (rounded lat/lon, `kWp` capacity). `generation_Wh` over 30 min is converted to watts (×2); short gaps (≤ 3 steps) are linearly interpolated, unfixable ones dropped.
- **UK satellite imagery** — EUMETSAT **SEVIRI RSS HRV** from the Open Climate Fix public GCS bucket (`gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/<year>_hrv.zarr`), ~1 km/px. Frames are reprojected (pyproj) to each site and cropped to **128×128 px (~128 km)** for cloud-advection context.
- **PV power + imagery — US** (`goes_pvdaq`): NREL **PVDAQ** systems with site metadata (lat/lon, nameplate power) from the [OEDI data lake](https://oedi-data-lake.s3.amazonaws.com/pvdaq/), paired with **GOES-16** satellite crops (10 sites).
- **Weather covariates** — [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api): 8 hourly variables (2 m temperature, shortwave/direct/diffuse radiation, DNI, cloud cover, 10 m wind speed, precipitation), fetched per site and joined to each row by nearest-timestamp (`merge_asof`).

**Operations.** Per frame: NaN → 0, clip to [0,1], keep only non-empty crops; images are written as per-site PNGs / a single `uint8` HDF5 (millions of small files are impractical). Numeric rows from both sources are concatenated, enriched with `installed_power_w` + coordinates, then joined with the weather track and saved.

### 1.4 Other public dataset — ClimateHackAI 2023

The **ClimateHackAI 2023** competition dataset ([DOXA](https://doxaai.com/competition/climatehackai-2023/overview) · [HF](https://huggingface.co/datasets/climatehackai/climatehackai-2023), ~600 GB) is a multimodal PV-nowcasting corpus over **Great Britain**. Per site it provides **PV generation** as a fraction of installed capacity at **5-min** resolution, with metadata (latitude, longitude, orientation, tilt, installed capacity); two **EUMETSAT** satellite streams at 5-min cadence — **HRV** `[12,128,128]` and **non-HRV** 11-channel `[12,128,128,11]` crops centred on each site; **DWD ICON-EU** numerical weather forecasts (38 variables, T−1h…T+4h, 128×128 grids); and **ECMWF CAMS** air-quality forecasts (13 variables × 8 altitudes). The competition task: from **1 h of history** (PV + satellite) plus the hourly weather/air-quality forecast steps, predict the **next 4 h** of PV output — 48 values at 5-min intervals — scored by mean absolute error (target < 0.15).

It is structured differently from the dataset of record: built for **nowcasting**, it is organised as short fixed windows (1 h in → 4 h out) at 5-min resolution with per-window satellite/NWP/air-quality tensors, rather than a single long tall table of `(site, timestamp)` rows spanning history and horizon. That short-window, high-cadence, forecast-tensor layout targets minutes-to-hours nowcasting and would need re-windowing and re-sampling before it could serve the longer-history, cross-plant protocol used here.

Note: this is a **competition dataset with no accompanying published paper**; it is documented only through the competition overview and Hugging Face repository, so there is no peer-reviewed reference to cite for its construction or provenance.

### 1.5 Other public dataset — MMSP (FusionSF)

The **MMSP** (Multi-Modal Solar Power) dataset is released with **FusionSF** ([arXiv:2402.05823](https://arxiv.org/abs/2402.05823) · [code](https://github.com/MAZiqing/FusionSF) · [data](https://drive.google.com/drive/folders/1qGVOw-hAVQlO3n-1d4ZNHvL42L9PkdBK)), a multimodal PV corpus over **88 geographically dispersed solar plants across a Chinese province (~157,100 km²)**, spanning **Jan 2021 – Jun 2022** at **hourly** resolution (downsampled from 10-min). Each record fuses three modalities: **historical PV power**; **Himawari-8/9 satellite imagery** cropped to **64×64 px** (from 640×640) — 1 channel in the small variant **MMSP(S)** (10 plants), 4 visible/near-infrared channels in the full **MMSP(L)** (88 plants); and **ECMWF NWP** (17 weather features: radiation, temperature, pressure, cloud cover, wind). The task is **day-ahead** forecasting: 24 h of history → next 24 h. (The paper reports the method deployed on 300+ plants / >15 GW in production; MMSP is the released research subset.)

Like ClimateHackAI, it differs structurally from the dataset of record — hourly cadence, fixed day-ahead 24-in → 24-out windows over China with Himawari imagery and per-window NWP tensors — so it would need re-windowing/re-sampling before serving the longer-history, cross-plant protocol used here.

## 2. Data Splits

Cross-plant, seeded per-dataset partition over the 106 sites. Counts combine both datasets.

|  | Plants | Rows |
| :--- | :---: | :---: |
| **Train** | 75 | 918 158 |
| **Validation** | 16 | 193 553 |
| **Test** | 15 | 185 304 |

Per-dataset plants — uk_pv: 69/15/14 (train/val/test); goes_pvdaq: 6/1/1.

## 3. Configurations

| Dimension | Specs|
| :--- | :---: |
| **History** | 14 days (steps: 672 uk and 1344 pvdaq) |
| **Horizon** | 6 hours (steps: 12 uk and 24 pvdaq) |
| **Frames** | 8 frames @ 60-min cadence (last 8 h ending at forecast time) |

Models should be tested also at intra-hour frame cadence. On `uk_pv` 30 min and on `goes_pvdaq` 15 min can be used (longer training).

## 2. Leaderboard

**Rank** is the global position by skill score (SS ↑) across all models; each subtable is sorted best-first. Metrics: SS ↑, R² ↑, CRPS ↓. Reference floor is `smart_persistence` (SS ≡ 0).

**Metric intuition.** All point metrics run on the capacity-normalized target (`norm_power` ∈ [0,1]) over daylight, valid steps, so values read as fraction-of-nameplate error and compare across plants.

- **R²** ↑ — squared Pearson correlation between predicted and true power, r² = corr(true, pred)², pooled across test plants on daylight, valid steps. Measures how much of the *variance* the forecast tracks (shape/timing), independent of bias or scale. 1 = perfectly correlated, 0 = no linear relation.
- **SS (skill score)** ↑ — how much a model beats the `smart_persistence` baseline on root-mean-square error: 0 = no better than baseline, 1 = perfect, negative = worse.
- **CRPS** ↓ — scores the *whole predictive distribution* (via quantile pinball), not just the mean; rewards forecasts both accurate *and* honestly calibrated about uncertainty (probabilistic models only).

**Cov flag.** Each table carries a **Cov** column: ✓ = the model ingests the weather/solar covariate track

### 2.1 Reference / statistical

| Rank | Model | **SS ↑** | R² ↑ | CRPS ↓ | Cov | Description |
| --- | --- | --- | --- | --- | --- | --- |
| 23 | climatology_hourly | 0.234 | 0.314 | — | ✗ | Per-hour-of-day mean power from the training set; a fixed seasonal climatology. |
| 28 | seasonal_naive | 0.107 | 0.276 | — | ✗ | Copies power from the same clock time on the previous day. |
| 30 | persistence | 0.014 | 0.171 | — | ✗ | Carries the last observed power flat across the horizon. |
| — | smart_persistence | 0.000 | 0.210 | — | ✗ | Persistence rescaled by the clear-sky ratio — the sun-adjusted reference floor |

### 2.2 Classical ML

| Rank | Model | **SS ↑** | R² ↑ | CRPS ↓ | Cov | Description |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | LightGBM | 0.384 | 0.555 | 0.0768 | ✓ | Gradient-boosted decision trees over lagged power + covariate features. |

### 2.3 Tabular foundation models

| Rank | Model | **SS ↑** | R² ↑ | CRPS ↓ | Cov | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | TabPFN 3 | 0.339 | 0.499 | 0.0815 | ✓ | [arXiv:2605.13986](https://arxiv.org/abs/2605.13986) |
| — | TabFM 1.0 | TBD | — | TBD | ✓ | [Google Research](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/) |

### 2.4 Supervised DL
| Rank | Model | **SS ↑** | R² ↑ | CRPS ↓ | Cov | Description |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **iTransformer** | **0.552** | 0.765 | — | ✓ | [arXiv:2310.06625](https://arxiv.org/pdf/2310.06625) |
| 7 | PatchTST | 0.458 | 0.651 | — | ✓ | Splits each series into patches for a channel-independent transformer. |
| 9 | TFT | 0.423 | 0.626 | 0.0689 | ✓ | Temporal Fusion Transformer; gated attention with quantile outputs. |
| 10 | MLP | 0.413 | 0.600 | — | ✓ | Plain multilayer perceptron over the flattened history window. |
| 20 | DLinear | 0.325 | 0.462 | — | ✗ | Single linear layer on trend/seasonal decomposition — target-only simplicity check. |

### 2.5 Zero-shot TSFM
| Rank | Model | **SS ↑** | R² ↑ | CRPS ↓ | Cov | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | tirex_zs | 0.287 | 0.440 | 0.0892 | ✗ | [arXiv:2505.23719](https://arxiv.org/abs/2505.23719) |
| 22 | timesfm_zs | 0.271 | 0.429 | 0.0923 | ✗ | [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |
| 26 | chronos2_zs | 0.187 | 0.277 | 0.1072 | ✓ | [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) |
| ✗ | ttm_zs | **−0.081** | 0.155 | — | ✓ | [TTM-R2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) |

### 2.6 Fine-tuned TSFM
| Rank | Model | **SS ↑** | R² ↑ | CRPS ↓ | Cov | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | chronos2_oracle_ft | 0.504 | 0.706 | 0.0630 | ✓ | [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) |
| 6 | chronos2_oracle | 0.474 | 0.679 | 0.0635 | ✓ | [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) |
| 13 | ttm_ft | 0.364 | 0.520 | — | ✓ | [TTM-R2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) |
| 19 | chronos2_ft | 0.331 | 0.475 | 0.0855 | ✓ | [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) |

> **`chronos2_*` vs `chronos2_oracle_*`** — same Chronos-2 model; the only difference is *which covariates are exposed over the forecast horizon*. The plain variants (`chronos2_zs`, `chronos2_ft`) uses only deterministic future covariates (solar geometry + calendar) that are known ahead. The `oracle` variants feed the model the **future observed weather** (temperature, cloud cover, irradiance, …) over the horizon.

### 2.7 Retrieval / frozen-FM adaptation
| Rank | Model | **SS ↑** | R² ↑ | CRPS ↓ | Cov | Regime | Paper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | ts_rag | 0.478 | 0.331 | — | ✗ | Zero-shot (frozen FM + retrieval) | [arXiv:2503.07649](https://arxiv.org/abs/2503.07649) |
| 5 | cross_rag | 0.477 | 0.375 | — | ✗ | Zero-shot (frozen FM + retrieval) | [arXiv:2603.14709](https://arxiv.org/abs/2603.14709) |
| 12 | cora | 0.374 | 0.550 | 0.0816 | ✓ | Fine-tuned (adapter, frozen FM) | [arXiv:2510.12681](https://arxiv.org/abs/2510.12681) |

### 2.8 Endogenous multimodal (second modality synthesized from the numeric track — no external sensor)

Pseudo-images rendered from the series (time_vlm, visionts_pp) or weather-text templated from the covariates (aurora). No satellite/sky frames consumed.

| Rank | Model | **SS ↑** | R² ↑ | CRPS ↓ | Cov | Regime | Paper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | **time_vlm** | **0.540** | 0.487 | — | ✗ | Fine-tuned (frozen VLM backbones) | [arXiv:2502.04395](https://arxiv.org/abs/2502.04395) |
| 24 | aurora | 0.232 | 0.440 | — | ✓ | Zero-shot | [arXiv:2509.22295](https://arxiv.org/abs/2509.22295) |
| 29 | visionts_pp | 0.017 | 0.397 | — | ✗ | Zero-shot (frozen image MAE) | [arXiv:2508.04379](https://arxiv.org/abs/2508.04379) |

### 2.9 Exogenous multimodal (external satellite imagery)

| Rank | Model | **SS ↑** | R² ↑ | CRPS ↓ | Cov | Regime | Paper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | solar_vlm | 0.443 | 0.660 | — | ✓ | Fine-tuned (pretrained vision) | [arXiv:2604.04145](https://arxiv.org/abs/2604.04145) |
| 14 | crossvivit | 0.349 | 0.565 | — | ✓ | From scratch | [arXiv:2306.01112](https://arxiv.org/abs/2306.01112) |
| 16 | **mmtsfm_grassmann_interleaved** | **0.3429** | — | 0.0805 | ✓ | Fine-tuned (frozen FM + vision) | ours (this work) |
| 25 | sunset | 0.216 | 0.359 | — | ✗ | From scratch | [github: YuchiSun/SUNSET](https://github.com/YuchiSun/SUNSET) |
| 27 | unicast | 0.121 | 0.371 | — | ✗ | Fine-tuned (soft-prompt, frozen FM) | [arXiv:2508.11954](https://arxiv.org/abs/2508.11954) |
| — | FusionSF | TBD | — | TBD | ✓ | TBD | [arXiv:2402.05823](https://arxiv.org/abs/2402.05823) |
| — | PIPE | TBD | — | TBD | ✓ | TBD | [arXiv:2506.14786](https://arxiv.org/abs/2506.14786) |
| — | MATE | TBD | — | TBD | ✓ | TBD | [OpenReview](https://openreview.net/forum?id=jn7GJdyuVn) |

## 4. Conclusions

Drawn from the cross-plant leaderboard. All numbers are zero-shot on **disjoint test plants** — no test-plant data in training.

**1. A well-tuned supervised transformer is the model to beat.** iTransformer tops every metric (SS 0.552, R² 0.765), ahead of every foundation model, retrieval scheme, and multimodal system. On this cross-plant PV task, architecture + in-domain supervised training still outperforms large pretrained backbones.

**2. Zero-shot time-series foundation models underdeliver on PV.** The zero-shot TSFMs (chronos2_zs SS 0.187, timesfm_zs 0.271, tirex_zs 0.287, ttm_zs **−0.081** — worse than the naive floor) trail even classical baselines. PV's sharp, weather-driven dynamics are out-of-distribution for generic TS pretraining; the models need adaptation to be useful.

**3. Adaptation closes the gap, and retrieval is the best frozen-backbone lever.** Every adaptation step helps: Chronos-2 jumps zero-shot → fine-tuned, and TTM flips from below-baseline to SS 0.364 once fine-tuned. Notably, **retrieval over a frozen backbone** (ts_rag SS 0.478, cross_rag 0.477) rivals fine-tuning **without updating any weights**, making it the most effective — and cheapest — frozen-FM strategy tested.

**4. Multimodality has not yet paid off** The one multimodal model at the top (time_vlm, SS 0.540, rank 2) uses *endogenous pseudo-images synthesized from the numeric series* and no covariates. Every model that ingests genuine **exogenous satellite/sky imagery** ranks mid-pack or worse.

**7. R² and SS disagree for biased forecasters — read them together.** Several models track the day's *shape* well yet sit off the diagonal: e.g. visionts_pp has R² 0.397 but SS 0.017, and solar_vlm / sunset / crossvivit show systematic under-prediction. High variance-tracking (R²) with low skill (SS) flags a **bias/scale** problem, not a timing problem — fixable by calibration rather than a better temporal model.