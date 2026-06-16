# Deep Exploratory Data Analysis Report

> ⚠ **Superseded snapshot.** This EDA predates the consolidated **dataset of
> record** `/Volumes/SSD/thesis-dataset/` (`dataset_all.parquet` + `images_all.h5`;
> DATASET_CONTRACT.md §1.0). Two numbers below changed after the full `goes_pvdaq`
> download — rerun `run_eda.py` against `dataset_all.parquet` to regenerate the
> figures/tables. Corrected top-line facts:
>
> | dataset | rows | sites | cadence | span (UTC) | capacity | frames |
> |---|---|---|---|---|---|---|
> | `uk_pv` | 1,232,862 | 100 | 30-min | 2019-01-01 → 2020-12-31 | 1.5–4.0 kW | `(N,128,128)` uint8 gray |
> | `goes_pvdaq` | **104,792** | 10 | 15-min | **2019-01-01 → 2019-09-30** | 1.8–408 kW | `(N,256,256,3)` uint8 RGB |
>
> Total rows **1,337,654**; 35 columns; canonical frame pointer `image_h5_index`
> (local-to-group into `images_all.h5[<dataset>_<site>]`). `bad_site_flag`:
> `uk_pv` 7239/8587, `goes_pvdaq` 1283/51. The `goes_pvdaq` block in this report
> (≈14.9k rows, June-2019-only) reflects a partial early download.

Cross-site PV power dataset with satellite/sky image augmentation. This report covers data integrity, physical coherence of production, temporal structure, feature importance, cross-site similarity (zero-shot transfer relevance) and image quality.

## 1. Dataset Overview

- **Total records:** 1,247,760
- **Columns:** 23 (raw: 18 + derived norm_power/hour/month/date/solar_hour)
- **Time range:** 2019-01-01 08:00:00+00:00 → 2020-12-31 16:00:00+00:00
- **Unique sites:** 110

| dataset    |    rows |   sites | sampling        | first_ts                  | last_ts                   |   median_rows_per_site | capacity_range_W   |
|:-----------|--------:|--------:|:----------------|:--------------------------|:--------------------------|-----------------------:|:-------------------|
| goes_pvdaq |   14898 |      10 | 0 days 00:15:00 | 2019-06-01 00:00:00+00:00 | 2019-06-30 23:45:00+00:00 |                   1543 | 1,820 – 408,240    |
| uk_pv      | 1232862 |     100 | 0 days 00:30:00 | 2019-01-01 08:00:00+00:00 | 2020-12-31 16:00:00+00:00 |                  12325 | 1,500 – 4,000      |

### Missing values

No missing values in any column.

### Descriptive statistics (numeric columns)

|                          |       count |     mean |       std |      min |     25% |     50% |      75% |        max |
|:-------------------------|------------:|---------:|----------:|---------:|--------:|--------:|---------:|-----------:|
| power_w                  | 1.24776e+06 | 1004.73  |  7202.55  | -4739.82 |  100    |  400    |  980.66  | 430697     |
| installed_power_w        | 1.24776e+06 | 4227.64  | 19535.1   |  1500    | 2000    | 2500    | 3900     | 408240     |
| norm_power               | 1.24776e+06 |    0.223 |     0.216 |    -2.37 |    0.05 |    0.15 |    0.354 |      2.998 |
| temperature_2m           | 1.24776e+06 |   11.587 |     5.682 |    -8.8  |    7.4  |   11.2  |   15.4   |     41.3   |
| shortwave_radiation      | 1.24776e+06 |  276.276 |   218.543 |     0    |   88    |  229    |  434     |   1015     |
| direct_radiation         | 1.24776e+06 |  156.331 |   180.607 |     0    |    8    |   83    |  254     |    903     |
| diffuse_radiation        | 1.24776e+06 |  119.946 |    82.009 |     0    |   58    |  104    |  172     |    439     |
| direct_normal_irradiance | 1.24776e+06 |  292.051 |   272.844 |     0    |   28.8  |  224.4  |  508.3   |    950.5   |
| cloudcover               | 1.24776e+06 |   72.081 |    33.857 |     0    |   48    |   91    |  100     |    100     |
| windspeed_10m            | 1.24776e+06 |   17.418 |     8.623 |     0    |   11    |   16.6  |   22.7   |     72.6   |
| precipitation            | 1.24776e+06 |    0.155 |     0.431 |     0    |    0    |    0    |    0.1   |     18.4   |

## 2. Data Integrity & Physical Coherence Checks

| check                                                              |   count |
|:-------------------------------------------------------------------|--------:|
| Duplicate (site, timestamp) rows                                   |       0 |
| Negative power values                                              |       1 |
| Power exceeding installed capacity                                 |      34 |
| Power exceeding capacity by >5%                                    |       7 |
| Negative radiation values                                          |       0 |
| NaN power values                                                   |       0 |
| Zero power rows (13.6% of data)                                    |  169988 |
| Zero power with shortwave_radiation > 200 W/m² (suspected outages) |   14229 |
| Power > 1% capacity with zero irradiance (night production)        |     912 |
| Sites with non-constant installed capacity                         |       0 |
| Sites with non-constant coordinates                                |       0 |

**Stuck-sensor heuristic** (identical non-zero power for ≥6 consecutive daytime steps): 1,297 rows across 70 sites.

### Temporal coverage per site

Data is daytime-only by design (no night rows), so coverage is measured as days-with-data over the site's active span, plus missing steps *within* days.

| dataset    |   ('rows', 'median') |   ('rows', 'min') |   ('rows', 'max') |   ('days_with_data', 'median') |   ('days_with_data', 'min') |   ('days_with_data', 'max') |   ('day_coverage_%', 'median') |   ('day_coverage_%', 'min') |   ('day_coverage_%', 'max') |   ('intra_day_gaps', 'median') |   ('intra_day_gaps', 'min') |   ('intra_day_gaps', 'max') |
|:-----------|---------------------:|------------------:|------------------:|-------------------------------:|----------------------------:|----------------------------:|-------------------------------:|----------------------------:|----------------------------:|-------------------------------:|----------------------------:|----------------------------:|
| goes_pvdaq |                 1543 |               985 |              1575 |                             30 |                          21 |                          30 |                          100   |                        70   |                         100 |                           48.5 |                          28 |                          57 |
| uk_pv      |                12325 |             12309 |             12361 |                            730 |                         726 |                         731 |                           99.9 |                        99.3 |                         100 |                            5   |                           3 |                          18 |

Worst 10 sites by day coverage:

| dataset    |   site_id |   rows |   days_with_data |   span_days |   day_coverage_% |   intra_day_gaps |
|:-----------|----------:|-------:|-----------------:|------------:|-----------------:|-----------------:|
| goes_pvdaq |      1277 |    985 |               21 |          30 |             70   |               43 |
| uk_pv      |      7498 |  12316 |              726 |         731 |             99.3 |                3 |
| uk_pv      |     11042 |  12316 |              727 |         731 |             99.5 |                3 |
| uk_pv      |     12495 |  12316 |              727 |         731 |             99.5 |                3 |
| uk_pv      |      6481 |  12316 |              727 |         731 |             99.5 |                3 |
| uk_pv      |      6493 |  12316 |              727 |         731 |             99.5 |                3 |
| uk_pv      |      6838 |  12316 |              727 |         731 |             99.5 |                3 |
| uk_pv      |      7547 |  12316 |              727 |         731 |             99.5 |                3 |
| uk_pv      |      8587 |  12316 |              727 |         731 |             99.5 |                3 |
| uk_pv      |     10048 |  12320 |              728 |         731 |             99.6 |                4 |

![coverage_per_site.png](plots/coverage_per_site.png)

![site_availability_timeline.png](plots/site_availability_timeline.png)

## 3. Site-Level Analysis

![site_map.png](plots/site_map.png)

![capacity_distribution.png](plots/capacity_distribution.png)

Capacity spans multiple orders of magnitude across datasets — power must be normalized by installed capacity for any cross-site model.

![site_capacity_factor_and_corr.png](plots/site_capacity_factor_and_corr.png)

### Production coherence per site

Per-site correlation between normalized power and shortwave radiation is the primary coherence check: a healthy PV site should have r ≳ 0.7. Low values indicate metering faults, wrong capacity, tracker issues or bad weather joins.

**2 sites with corr < 0.6:**

|   site_id | dataset   |   cf_mean |   cf_p95 |   pw_rad_corr |
|----------:|:----------|----------:|---------:|--------------:|
|      7239 | uk_pv     |     0.057 |    0.429 |         0.28  |
|      8587 | uk_pv     |     0.142 |    0.64  |         0.428 |

![site_norm_power_boxplot.png](plots/site_norm_power_boxplot.png)

## 4. Target Variable: PV Power

![power_distributions.png](plots/power_distributions.png)

- Zero-power rows: 169,988 (13.6%) — mostly dawn/dusk edges of the daytime window; verify they are real zeros, not sentinel fills.
- Normalized power > 1: 34 rows (max = 2.998).

![diurnal_profile.png](plots/diurnal_profile.png)

![month_hour_heatmap.png](plots/month_hour_heatmap.png)

Diurnal and seasonal structure should look like a clean solar geometry surface; horizontal stripes or holes indicate data problems for specific months.

![monthly_production.png](plots/monthly_production.png)

![ramp_rates.png](plots/ramp_rates.png)

Ramp-rate tails capture cloud-driven variability — the signal satellite imagery should help predict. Heavy tails = more value from the vision modality.

![week_goes_pvdaq.png](plots/week_goes_pvdaq.png)

![week_uk_pv.png](plots/week_uk_pv.png)

### Intraday persistence (autocorrelation)

![power_acf.png](plots/power_acf.png)

High short-lag autocorrelation sets the persistence baseline any forecast model must beat.

## 5. Weather Features vs Production

![correlation_matrices.png](plots/correlation_matrices.png)

Correlation with normalized power (daytime rows):

|                          |   pearson |   spearman |
|:-------------------------|----------:|-----------:|
| shortwave_radiation      |     0.726 |      0.757 |
| direct_radiation         |     0.68  |      0.695 |
| direct_normal_irradiance |     0.584 |      0.584 |
| temperature_2m           |     0.401 |      0.436 |
| diffuse_radiation        |     0.399 |      0.528 |
| windspeed_10m            |    -0.11  |     -0.101 |
| precipitation            |    -0.198 |     -0.264 |
| cloudcover               |    -0.33  |     -0.361 |

![feature_vs_power_hexbin.png](plots/feature_vs_power_hexbin.png)

The power–shortwave relation should be tightly linear with a flat saturation near the inverter limit; wide vertical spread at high irradiance means curtailment, soiling, shading or capacity errors.

![cloudcover_residual_effect.png](plots/cloudcover_residual_effect.png)

If cloud cover still separates power *within* an irradiance bin, images carry information beyond the reanalysis weather — the core hypothesis of the project.

### Feature importance for predicting normalized power

Random-forest tabular baseline on weather + calendar + location features: **R² = 0.654** on held-out 25% (random split — optimistic vs a true site-held-out split, but a useful upper reference for tabular-only signal).

|                          |   rf_gini |   permutation |   mutual_info |
|:-------------------------|----------:|--------------:|--------------:|
| shortwave_radiation      |    0.7224 |        0.8414 |        0.4449 |
| longitude                |    0.0489 |        0.1033 |        0.5599 |
| latitude                 |    0.0577 |        0.0968 |        0.5541 |
| cloudcover               |    0.0237 |        0.0491 |        0.0812 |
| solar_hour               |    0.0341 |        0.0398 |        0.3198 |
| precipitation            |    0.011  |        0.0165 |        0.0499 |
| direct_radiation         |    0.0229 |        0.0162 |        0.3412 |
| direct_normal_irradiance |    0.0191 |        0.0154 |        0.2394 |
| diffuse_radiation        |    0.0146 |        0.0141 |        0.2391 |
| temperature_2m           |    0.0202 |        0.0105 |        0.1175 |
| month                    |    0.0078 |        0.0063 |        0.197  |
| windspeed_10m            |    0.0177 |        0.0039 |        0.0046 |

![feature_importance.png](plots/feature_importance.png)

**Cross-dataset zero-shot probe** (RF trained on uk_pv weather features, evaluated on goes_pvdaq): R² = 0.324, MAE = 0.170 normalized power. This is the tabular transfer floor your foundation-model + vision approach should beat.

## 6. Cross-Site Structure (Zero-Shot Transfer Relevance)

![site_similarity_clustermap.png](plots/site_similarity_clustermap.png)

- Median intra-uk_pv site correlation: **0.709**
- Median intra-goes_pvdaq site correlation: **0.005**
- Median cross-dataset site correlation: **-0.014**

High intra-dataset correlation means weather regimes are shared and leave-site-out splits within a region are *not* independent; cross-dataset (UK ↔ US) evaluation is the honest zero-shot test.

### Distribution shift between datasets (KS statistic, daytime rows)

| feature                  |   KS_stat |
|:-------------------------|----------:|
| temperature_2m           |     0.726 |
| direct_radiation         |     0.365 |
| shortwave_radiation      |     0.362 |
| windspeed_10m            |     0.361 |
| direct_normal_irradiance |     0.344 |
| cloudcover               |     0.269 |
| precipitation            |     0.213 |
| diffuse_radiation        |     0.165 |
| norm_power               |     0.109 |

![dataset_shift_kde.png](plots/dataset_shift_kde.png)

## 7. Image Modality

- Existence/readability check on 3,000 random rows: **0 missing**, **0 corrupt**.
- Filename↔timestamp alignment check on 500 rows: **0 mismatches**.

### Image formats

| dataset    | sizes             | modes        |
|:-----------|:------------------|:-------------|
| goes_pvdaq | {(256, 256): 400} | {'RGB': 400} |
| uk_pv      | {(32, 32): 400}   | {'L': 400}   |

Heterogeneous image sizes/modes across datasets — the video encoder pipeline needs an explicit resize + channel policy per source.

![image_brightness_contrast.png](plots/image_brightness_contrast.png)

### Do images carry production-relevant signal?

| dataset    |   corr(brightness, norm_power) |   corr(brightness, shortwave_rad) |   corr(brightness, cloudcover) |   corr(contrast, norm_power) |   n |
|:-----------|-------------------------------:|----------------------------------:|-------------------------------:|-----------------------------:|----:|
| goes_pvdaq |                         -0.31  |                            -0.439 |                          0.467 |                       -0.186 | 400 |
| uk_pv      |                          0.066 |                             0.267 |                          0.324 |                        0.337 | 400 |

Even a trivial brightness statistic correlating with power/cloud confirms the imagery is informative; the video encoder should extract far more (cloud morphology, motion).

![brightness_vs_power.png](plots/brightness_vs_power.png)

![image_samples_goes_pvdaq.png](plots/image_samples_goes_pvdaq.png)

![image_samples_uk_pv.png](plots/image_samples_uk_pv.png)

![image_sequence_goes_pvdaq.png](plots/image_sequence_goes_pvdaq.png)

![image_sequence_uk_pv.png](plots/image_sequence_uk_pv.png)

## 8. Auto-Detected Issues & Recommendations

- 🔴 **ERROR**: Power exceeding capacity by >5%: 7 rows
- 🔴 **ERROR**: 2 sites have power–radiation corr < 0.6: ['7239', '8587']
- 🟡 **WARN**: `goes_pvdaq` covers only 29 days (2019-06-01 → 2019-06-30) — no seasonal diversity; cross-dataset evaluation on it tests a single season only.
- 🟡 **WARN**: Negative power values: 1 rows
- 🟡 **WARN**: Power exceeding installed capacity: 34 rows
- 🟡 **WARN**: Zero power with shortwave_radiation > 200 W/m² (suspected outages): 14,229 rows
- 🟡 **WARN**: Power > 1% capacity with zero irradiance (night production): 912 rows

### Recommendations for the modeling phase

1. **Normalize power by installed capacity** everywhere; clip to [0, 1] after verifying >1 rows are inverter-rating artifacts and not capacity errors.
2. **Mask or drop suspected-outage rows** (zero power under strong irradiance) and stuck-sensor runs — they corrupt both training and evaluation.
3. **Exclude or down-weight sites** with power–radiation correlation < 0.6 until their metering is explained.
4. Use **cross-dataset (UK↔US) splits** as the primary zero-shot benchmark; intra-region leave-site-out is contaminated by shared weather.
5. Handle the **sampling-rate mismatch** (15 vs 30 min) and **image heterogeneity** (32×32 L vs 256×256 RGB) explicitly in the data pipeline.
6. The tabular RF baselines above (random-split R² and UK→US transfer R²) are the numbers the foundation-model + vision approach must beat.