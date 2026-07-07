## 1. Dataset Overview

- **Total records:** 1,247,760
- **Columns:** 23 (raw: 18 + derived norm_power/hour/month/date/solar_hour)

| dataset    |    rows |   sites | sampling        | first_ts                  | last_ts                   |
|:-----------|--------:|--------:|:----------------|:--------------------------|:--------------------------|
| goes_pvdaq |   14898 |      10 | 15 minutes | 2019-06-01 00:00:00+00:00 | 2019-06-30 23:45:00+00:00 |
| uk_pv      | 1232862 |     100 | 30 minutes | 2019-01-01 08:00:00+00:00 | 2020-12-31 16:00:00+00:00 |

## 2. Data Splits

|  | Plants | Rows |
| :--- | :---: | :---: |
| **Train** | 69 | 850 654 |
| **Validation** | 15 | 184 899 |
| **Test** | 14 | 172 656 |

## 3. Configurations

| Dimension | Specs|
| :--- | :---: |
| **History** | 14 days (steps: 672 uk and 1344 pvdaq) |
| **Horizon** | 6 hours (steps: 12 uk and 24 pvdaw) |
| **Frame Cadence** | 8 frame |

## 2. Leaderboard


| Rank | Tier | Model | NMAE ↓ | NRMSE ↓ | **SS ↑** | CRPS ↓ | Ramp NRMSE ↓ | Note |
|---:|---|---|---|---|---|---|---|---|
| 1 | T2 | **iTransformer** | 0.0699 | 0.1032 | **0.552** | — | 0.1769 | best overall, supervised |
| 2 | **T5** | **time_vlm** | 0.0692 | 0.1061 | **0.540** | — | **0.1720** | **best multimodal; best ramp** |
| 3 | T3 | chronos2_oracle_ft | 0.0808 | 0.1142 | 0.504 | 0.0630 | 0.1824 | oracle |
| 4 | T4 | ts_rag | 0.0705 | 0.1203 | 0.478 | — | 0.2004 | retrieval-augmented |
| 5 | T4 | cross_rag | 0.0726 | 0.1206 | 0.477 | — | 0.1969 | retrieval-augmented |
| 6 | T3 | chronos2_oracle | 0.0817 | 0.1213 | 0.474 | 0.0635 | 0.1915 | oracle |
| 7 | T2 | PatchTST | 0.0886 | 0.1249 | 0.458 | — | 0.1888 | supervised |
| 8 | T6 | solar_vlm | 0.0955 | 0.1283 | 0.443 | — | 0.1849 | multimodal (PV-specialized) |
| 9 | T2 | TFT | 0.0889 | 0.1330 | 0.423 | 0.0689 | 0.2001 | quantile |
| 10 | T2 | MLP | 0.0958 | 0.1352 | 0.413 | — | 0.1981 | supervised |
| 11 | T1 | LightGBM | 0.1000 | 0.1419 | 0.384 | 0.0768 | 0.2024 | classical ML |
| 12 | T4 | cora | 0.1025 | 0.1444 | 0.374 | 0.0816 | 0.2021 | frozen-TSFM adapt |
| 13 | T3 | ttm_ft | 0.1029 | 0.1465 | 0.364 | — | 0.2062 | tiny TSFM, fine-tuned |
| 14 | T6 | crossvivit | 0.1112 | 0.1500 | 0.349 | — | 0.2100 | multimodal |
| 16 | **MM** | **mmtsfm_grassmann_interleaved** | 0.1059 | 0.1514 | **0.3429** | 0.0805 | — |  |
| 17 | T1 | TabPFN | 0.1076 | 0.1524 | 0.339 | 0.0815 | 0.2050 | tabular FM |
| 19 | T3 | chronos2_ft | 0.1120 | 0.1543 | 0.331 | 0.0855 | 0.2115 |  FT |
| 20 | T2 | DLinear | 0.1131 | 0.1556 | 0.325 | — | 0.2092 | linear check |
| 21 | T3 | tirex_zs | 0.1145 | 0.1642 | 0.287 | 0.0892 | 0.2233 | zero-shot |
| 22 | T3 | timesfm_zs | 0.1172 | 0.1680 | 0.271 | 0.0923 | 0.2319 | zero-shot |
| 23 | T0 | climatology_hourly | 0.1353 | 0.1766 | 0.234 | — | 0.2037 | reference |
| 24 | T5 | aurora | 0.1280 | 0.1769 | 0.232 | — | 0.2516 | multimodal |
| 25 | T6 | sunset | 0.1384 | 0.1806 | 0.216 | — | 0.2177 | multimodal |
| 26 | T3 | chronos2_zs | 0.1376 | 0.1873 | 0.187 | 0.1072 | 0.2335 |  ZS |
| 27 | T5 | unicast | 0.1433 | 0.2025 | 0.121 | — | 0.2814 | weak |
| 28 | T0 | seasonal_naive | 0.1419 | 0.2058 | 0.107 | — | 0.2575 | reference |
| 29 | T5 | visionts_pp | 0.1690 | 0.2266 | 0.017 | — | 0.2515 | ≈ persistence |
| 30 | T0 | persistence | 0.1643 | 0.2272 | 0.014 | — | 0.2990 | floor |
| — | T0 | smart_persistence | 0.1593 | 0.2304 | 0.000 | — | 0.2806 | **reference (SS≡0)** |
| ✗ | T3 | ttm_zs | 0.1704 | 0.2490 | **−0.081** | — | 0.3414 | worse than SP |
