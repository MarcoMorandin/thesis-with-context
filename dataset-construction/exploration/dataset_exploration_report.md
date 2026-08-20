# Deep Exploratory Data Analysis Report

Cross-site PV power dataset with satellite/sky image augmentation. This report covers data integrity, physical coherence of production, temporal structure, feature importance, cross-site similarity (zero-shot transfer relevance) and image quality.

## 1. Dataset Overview

- **Total records:** 1,337,654
- **Columns:** 39 (raw: 18 + derived norm_power/hour/month/date/solar_hour)
- **Time range:** 2019-01-01 00:00:00+00:00 → 2020-12-31 16:00:00+00:00
- **Unique sites:** 110

| dataset    |    rows |   sites | sampling        | first_ts                  | last_ts                   |   median_rows_per_site | capacity_range_W   |
|:-----------|--------:|--------:|:----------------|:--------------------------|:--------------------------|-----------------------:|:-------------------|
| goes_pvdaq |  104792 |      10 | 0 days 00:15:00 | 2019-01-01 00:00:00+00:00 | 2019-09-30 23:45:00+00:00 |                11175.5 | 1,820 – 408,240    |
| uk_pv      | 1232862 |     100 | 0 days 00:30:00 | 2019-01-01 08:00:00+00:00 | 2020-12-31 16:00:00+00:00 |                12325   | 1,500 – 4,000      |

### Missing values

|            |   missing |     % | cause                                                             |
|:-----------|----------:|------:|:------------------------------------------------------------------|
| power_w    |     16804 | 1.256 | intentional — masked outage/stuck rows (curation)                 |
| norm_power |     16804 | 1.256 | intentional — masked outage/stuck rows (curation)                 |
| kt         |     86139 | 6.44  | intentional — clear-sky GHI < 50 W/m² (undefined)                 |
| csi        |    102929 | 7.695 | intentional — clear-sky GHI < 50 W/m² or masked power (undefined) |

All NaNs above are introduced on purpose by `curate_dataset.py` (target masking on outage/stuck rows; kt/csi undefined below the clear-sky floor) and verified row-for-row against their cause — none are unexplained data loss.

### Descriptive statistics (numeric columns)

|                          |       count |      mean |       std |    min |     25% |      50% |      75% |      max |
|:-------------------------|------------:|----------:|----------:|-------:|--------:|---------:|---------:|---------:|
| power_w                  | 1.32085e+06 |  2697.64  | 15207.9   |    0   |  121.24 |  452.66  | 1100     | 408240   |
| installed_power_w        | 1.33765e+06 | 11544.6   | 45702.8   | 1500   | 2000    | 2550     | 3920     | 408240   |
| norm_power               | 1.32085e+06 |     0.231 |     0.22  |    0   |    0.05 |    0.159 |    0.367 |      1   |
| temperature_2m           | 1.33765e+06 |    12.13  |     6.595 |  -19.9 |    7.5  |   11.5   |   15.9   |     44.3 |
| shortwave_radiation      | 1.33765e+06 |   288.378 |   227.355 |    0   |   93    |  240     |  451     |   1050   |
| direct_radiation         | 1.33765e+06 |   167.685 |   190.845 |    0   |   10    |   92     |  273     |    906   |
| diffuse_radiation        | 1.33765e+06 |   120.692 |    81.81  |    0   |   59    |  105     |  172     |    591   |
| direct_normal_irradiance | 1.33765e+06 |   307.404 |   280.36  |    0   |   33.7  |  245.6   |  533.6   |    989   |
| cloudcover               | 1.33765e+06 |    70.349 |    34.909 |    0   |   44    |   89     |  100     |    100   |
| windspeed_10m            | 1.33765e+06 |    17.021 |     8.698 |    0   |   10.5  |   16.2   |   22.5   |     72.6 |
| precipitation            | 1.33765e+06 |     0.149 |     0.426 |    0   |    0    |    0     |    0.1   |     18.4 |

## 2. Data Integrity & Physical Coherence Checks

| check                                                              |   count |
|:-------------------------------------------------------------------|--------:|
| Duplicate (site, timestamp) rows                                   |       0 |
| Negative power values                                              |       0 |
| Power exceeding installed capacity                                 |       0 |
| Power exceeding capacity by >5%                                    |       0 |
| Negative radiation values                                          |       0 |
| NaN power values (16,804 are intended outage/stuck masks)          |       0 |
| Zero power rows (12.0% of data)                                    |  160432 |
| Zero power with shortwave_radiation > 200 W/m² (suspected outages) |       0 |
| Power > 1% capacity with zero irradiance (night production)        |       0 |
| Sites with non-constant installed capacity                         |       0 |
| Sites with non-constant coordinates                                |       0 |

**Stuck-sensor heuristic** (identical non-zero power for ≥6 consecutive daytime steps): 0 rows across 0 sites.

### Temporal coverage per site

Data is daytime-only by design (no night rows), so coverage is measured as days-with-data over the site's active span, plus missing steps *within* days.

| dataset    |   ('rows', 'median') |   ('rows', 'min') |   ('rows', 'max') |   ('days_with_data', 'median') |   ('days_with_data', 'min') |   ('days_with_data', 'max') |   ('day_coverage_%', 'median') |   ('day_coverage_%', 'min') |   ('day_coverage_%', 'max') |   ('intra_day_gaps', 'median') |   ('intra_day_gaps', 'min') |   ('intra_day_gaps', 'max') |
|:-----------|---------------------:|------------------:|------------------:|-------------------------------:|----------------------------:|----------------------------:|-------------------------------:|----------------------------:|----------------------------:|-------------------------------:|----------------------------:|----------------------------:|
| goes_pvdaq |              11175.5 |              7537 |             12655 |                            251 |                         166 |                         273 |                           98.4 |                        71.8 |                         100 |                          299.5 |                          62 |                         514 |
| uk_pv      |              12325   |             12309 |             12361 |                            730 |                         726 |                         731 |                           99.9 |                        99.3 |                         100 |                            5   |                           3 |                          18 |

Worst 10 sites by day coverage:

| dataset    |   site_id |   rows |   days_with_data |   span_days |   day_coverage_% |   intra_day_gaps |
|:-----------|----------:|-------:|-----------------:|------------:|-----------------:|-----------------:|
| goes_pvdaq |      1283 |   8275 |              176 |         245 |             71.8 |              217 |
| goes_pvdaq |        36 |   7537 |              166 |         174 |             95.4 |              213 |
| goes_pvdaq |      1277 |  11972 |              264 |         273 |             96.7 |              305 |
| goes_pvdaq |      1289 |  10379 |              238 |         245 |             97.1 |              294 |
| goes_pvdaq |      1203 |  12345 |              266 |         273 |             97.4 |               62 |
| uk_pv      |      7498 |  12316 |              726 |         731 |             99.3 |                3 |
| goes_pvdaq |        51 |   7711 |              197 |         198 |             99.5 |              514 |
| uk_pv      |     11042 |  12316 |              727 |         731 |             99.5 |                3 |
| uk_pv      |     12495 |  12316 |              727 |         731 |             99.5 |                3 |
| uk_pv      |      6481 |  12316 |              727 |         731 |             99.5 |                3 |

![coverage_per_site.png](plots/coverage_per_site.png)

![site_availability_timeline.png](plots/site_availability_timeline.png)

## 3. Site-Level Analysis

![site_map.png](plots/site_map.png)

![capacity_distribution.png](plots/capacity_distribution.png)

Capacity spans multiple orders of magnitude across datasets — power must be normalized by installed capacity for any cross-site model.

![site_capacity_factor_and_corr.png](plots/site_capacity_factor_and_corr.png)

### Production coherence per site

Per-site correlation between normalized power and shortwave radiation is the primary coherence check: a healthy PV site should have r ≳ 0.7. Low values indicate metering faults, wrong capacity, tracker issues or bad weather joins.

**2 sites with corr < 0.6** (2 already carry `bad_site_flag` from curation):

|   site_id | dataset    |   cf_mean |   cf_p95 |   pw_rad_corr | bad_site_flag   |
|----------:|:-----------|----------:|---------:|--------------:|:----------------|
|        51 | goes_pvdaq |     0.395 |    0.849 |         0.504 | True            |
|      1283 | goes_pvdaq |     0.301 |    0.781 |         0.516 | True            |

All low-correlation sites are already flagged `bad_site_flag` by `curate_dataset.py` — exclude or down-weight them downstream. No unhandled coherence failures.

![site_norm_power_boxplot.png](plots/site_norm_power_boxplot.png)

## 4. Target Variable: PV Power

![power_distributions.png](plots/power_distributions.png)

- Zero-power rows: 160,432 (12.0%) — mostly dawn/dusk edges of the daytime window; verify they are real zeros, not sentinel fills.
- Normalized power > 1: 0 rows (max = 1.000).

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
| shortwave_radiation      |     0.733 |      0.775 |
| direct_radiation         |     0.69  |      0.716 |
| direct_normal_irradiance |     0.599 |      0.605 |
| diffuse_radiation        |     0.393 |      0.531 |
| temperature_2m           |     0.38  |      0.431 |
| windspeed_10m            |    -0.108 |     -0.098 |
| precipitation            |    -0.201 |     -0.275 |
| cloudcover               |    -0.339 |     -0.371 |

![feature_vs_power_hexbin.png](plots/feature_vs_power_hexbin.png)

The power–shortwave relation should be tightly linear with a flat saturation near the inverter limit; wide vertical spread at high irradiance means curtailment, soiling, shading or capacity errors.

![cloudcover_residual_effect.png](plots/cloudcover_residual_effect.png)

If cloud cover still separates power *within* an irradiance bin, images carry information beyond the reanalysis weather — the core hypothesis of the project.

### Feature importance for predicting normalized power

Random-forest tabular baseline on weather + calendar + location features: **R² = 0.677** on held-out 25% (random split — optimistic vs a true site-held-out split, but a useful upper reference for tabular-only signal).

|                          |   rf_gini |   permutation |   mutual_info |
|:-------------------------|----------:|--------------:|--------------:|
| shortwave_radiation      |    0.7154 |        0.8182 |        0.4683 |
| latitude                 |    0.0697 |        0.1075 |        0.5418 |
| longitude                |    0.0449 |        0.0775 |        0.5352 |
| solar_hour               |    0.0356 |        0.0498 |        0.3304 |
| cloudcover               |    0.022  |        0.0478 |        0.0884 |
| direct_radiation         |    0.0271 |        0.027  |        0.3495 |
| direct_normal_irradiance |    0.0177 |        0.0237 |        0.2411 |
| precipitation            |    0.0092 |        0.0174 |        0.0506 |
| diffuse_radiation        |    0.0136 |        0.0134 |        0.2474 |
| temperature_2m           |    0.0196 |        0.0097 |        0.1174 |
| month                    |    0.0083 |        0.0078 |        0.1906 |
| windspeed_10m            |    0.017  |        0.005  |        0.0095 |

![feature_importance.png](plots/feature_importance.png)

**Cross-dataset zero-shot probe** (RF trained on uk_pv weather features, evaluated on goes_pvdaq): R² = 0.392, MAE = 0.163 normalized power. This is the tabular transfer floor your foundation-model + vision approach should beat.

## 6. Cross-Site Structure (Zero-Shot Transfer Relevance)

![site_similarity_clustermap.png](plots/site_similarity_clustermap.png)

- Median intra-uk_pv site correlation: **0.714**
- Median intra-goes_pvdaq site correlation: **0.156**
- Median cross-dataset site correlation: **0.212**

High intra-dataset correlation means weather regimes are shared and leave-site-out splits within a region are *not* independent; cross-dataset (UK ↔ US) evaluation is the honest zero-shot test.

### Distribution shift between datasets (KS statistic, daytime rows)

| feature                  |   KS_stat |
|:-------------------------|----------:|
| temperature_2m           |     0.498 |
| windspeed_10m            |     0.34  |
| direct_normal_irradiance |     0.324 |
| direct_radiation         |     0.313 |
| shortwave_radiation      |     0.289 |
| cloudcover               |     0.276 |
| precipitation            |     0.223 |
| diffuse_radiation        |     0.108 |
| norm_power               |   nan     |

![dataset_shift_kde.png](plots/dataset_shift_kde.png)

## 7. Image Modality

- Index-resolvability/readability check on 3,000 random rows: **0 unresolvable**, **0 corrupt**.
- h5-timestamp↔row alignment check on 500 rows: **0 mismatches**.

### Image formats

| dataset    | sizes             | modes        |
|:-----------|:------------------|:-------------|
| goes_pvdaq | {(256, 256): 400} | {'RGB': 400} |
| uk_pv      | {(128, 128): 400} | {'L': 400}   |

Heterogeneous image sizes/modes across datasets — the video encoder pipeline needs an explicit resize + channel policy per source.

![image_brightness_contrast.png](plots/image_brightness_contrast.png)

### Do images carry production-relevant signal?

| dataset    |   corr(brightness, norm_power) |   corr(brightness, shortwave_rad) |   corr(brightness, cloudcover) |   corr(contrast, norm_power) |   n |
|:-----------|-------------------------------:|----------------------------------:|-------------------------------:|-----------------------------:|----:|
| goes_pvdaq |                         -0.377 |                            -0.495 |                          0.448 |                       -0.233 | 400 |
| uk_pv      |                          0.105 |                             0.297 |                          0.321 |                        0.401 | 400 |

Even a trivial brightness statistic correlating with power/cloud confirms the imagery is informative; the video encoder should extract far more (cloud morphology, motion).

![brightness_vs_power.png](plots/brightness_vs_power.png)

![image_samples_goes_pvdaq.png](plots/image_samples_goes_pvdaq.png)

![image_samples_uk_pv.png](plots/image_samples_uk_pv.png)

![image_sequence_goes_pvdaq.png](plots/image_sequence_goes_pvdaq.png)

![image_sequence_uk_pv.png](plots/image_sequence_uk_pv.png)

## 8. Auto-Detected Issues & Recommendations

No blocking issues detected. ✓

### Recommendations for the modeling phase

1. **Normalize power by installed capacity** everywhere; clip to [0, 1] after verifying >1 rows are inverter-rating artifacts and not capacity errors.
2. **Mask or drop suspected-outage rows** (zero power under strong irradiance) and stuck-sensor runs — they corrupt both training and evaluation.
3. **Exclude or down-weight sites** with power–radiation correlation < 0.6 until their metering is explained.
4. Use **cross-dataset (UK↔US) splits** as the primary zero-shot benchmark; intra-region leave-site-out is contaminated by shared weather.
5. Handle the **sampling-rate mismatch** (15 vs 30 min) and **image heterogeneity** (32×32 L vs 256×256 RGB) explicitly in the data pipeline.
6. The tabular RF baselines above (random-split R² and UK→US transfer R²) are the numbers the foundation-model + vision approach must beat.