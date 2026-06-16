# Baseline Comparison Protocol

This document establishes the experimental protocol for evaluating and comparing all baseline models against the PVTSFM foundation model. Adherence to this protocol is mandatory to ensure scientific rigor and fair comparisons in our PV power forecasting research.

---

## 1. Core Principles

To guarantee that performance improvements are due to architectural and modeling advances rather than data leakage or domain engineering advantages, all experiments must follow these three rules:
1. **Same Horizon & Granularity**: Every baseline must forecast the exact same future window using the same history cadence.
2. **Disjoint Test Plants**: No baseline may train on, or leak statistics from, the held-out test plants.
3. **No Domain Physics Heuristics**: Baselines must not utilize hardcoded energy-domain formulas (such as Clear Sky Index conversions or physical irradiance coordinates) unless explicitly evaluating an ablation designed to test them.

---

## 2. Evaluation Scenario & Data Splits

We use a **Disjoint Cross-Plant Protocol** rather than a few-shot context-matching paradigm. 

### Plant Split Architecture
* **Train Plants**: Used for training baseline parameters (or fine-tuning foundation backbones).
* **Val Plants**: Disjoint from Train; used for hyperparameter tuning and early stopping.
* **Test Plants**: Disjoint from both Train and Val; used strictly for final reporting.

Numerical track data of record: `/Volumes/SSD/thesis-dataset/dataset_all.parquet` (+ frames `images_all.h5`, pointer `image_h5_index`) — both `uk_pv` and `goes_pvdaq` are now fully present (DATASET_CONTRACT.md §1.0). For the numerical track the split is generated once (seed 42, per-dataset 70/15/15, `bad_site_flag` sites excluded) and committed to `baselines/configs/splits.json`; disjointness is asserted at every load (`baselines/common/splits.py`). **`goes_pvdaq` (10 plants) must additionally be evaluated leave-one-plant-out** — its 15 % test share is 1-2 plants and per-plant variance would dominate a fixed split (see BASELINE_COMPARISON §4.1).

### Committed plant assignment — `uk_pv` (numerical track)

`uk_pv` is a fleet of **100 residential rooftop systems** (1.5–4.0 kW capacity, 30-minute cadence, 2019-01-01 → 2020-12-31 UTC). Two sites (`7239`, `8587`) carry `bad_site_flag` and are dropped, leaving 98 sites partitioned by the seed-42 shuffle into disjoint plant sets:

| Role | Plants | Rows | Valid power steps | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 69 | 850 654 | 846 633 | Fit ML params / fine-tune backbones |
| **Validation** | 15 | 184 899 | 182 809 | Early stopping, hyperparameter / α tuning |
| **Test** | 14 | 172 656 | 171 543 | Final reporting only — never seen in fit |
| _excluded_ | 2 (bad) | — | — | `7239`, `8587` (`bad_site_flag`) |

Exact `site_id` membership (source of truth: `baselines/configs/splits.json`):

* **Train (69)**: `10048 10367 10512 10589 10630 10702 10840 10843 11042 11287 12495 12642 13309 13311 13390 13773 14394 14467 14531 14649 14859 14924 16216 16921 18161 26811 26831 26844 26846 26848 26869 26879 26904 26919 27054 3149 3175 3333 3770 3872 4090 6427 6493 6618 6669 6675 6827 6838 6892 6966 6975 7017 7051 7088 7338 7359 7378 7401 7412 7498 7521 7533 7547 7608 7651 7674 7834 7836 9153`
* **Validation (15)**: `6075 6481 6498 6732 7019 7356 7648 10973 12826 13057 16474 16769 18249 26901 26970`
* **Test (14)**: `3432 6648 7315 7756 8066 9191 10793 11176 13388 13817 18989 26854 26933 27020`

The `goes_pvdaq` companion split (used only when its dataset is in scope) is 7 train / 2 val / 1 test (sites: train `1202 1277 1278 1283 1289 1367 1420`, val `1203 51`, test `36`), and is additionally rotated leave-one-plant-out per §4.1 of BASELINE_COMPARISON. **`goes_pvdaq` is now fully downloaded** (10 plants, 104,792 rows, 2019-01-01 → 2019-09-30 UTC, 15-min, `(256,256,3)` RGB frames, 1.8–408 kW capacities). ⚠ The new dataset flags `goes_pvdaq` sites `1283` and `51` with `bad_site_flag`, yet the committed split still lists them (train/val) — reconcile (regenerate the split excluding the 2 bad sites → 8 usable) before running `goes_pvdaq`. The runs documented in `docs/experiments/BASELINE_RESULTS_UKPV.md` are restricted to `uk_pv` (`--train-datasets uk_pv --eval-datasets uk_pv`) — they predate the `goes_pvdaq` download and the consolidated `thesis-dataset`.

### Inference Setup
During evaluation on a test plant:
* The model is given a history window of target power values and covariates: \(Y \in \mathbb{R}^{1 \times T \times 1}\) and \(X_{\text{cov}} \in \mathbb{R}^{1 \times T \times C_{\text{cov}}}\).
* The model is given historical visual frames (satellite PNGs): \(V \in \mathbb{R}^{1 \times T_v \times C_{\text{img}} \times H_{\text{img}} \times W_{\text{img}}}\).
* The model must predict future target power values: \(\hat{Y}_{\text{future}} \in \mathbb{R}^{1 \times H \times 1}\).

---

## 3. Horizon, Cadence, and Window Sizes

Unless overridden by a specific dataset-level configuration, the default temporal configurations are:

* **Granularity**: Intra-hour, native per dataset — no resampling. Numerical track: `uk_pv` 30-minute, `goes_pvdaq` 15-minute steps.
* **History Window (\(T\))**: 24 steps (12 h on `uk_pv`, 6 h on `goes_pvdaq`).
* **Forecasting Horizon (\(H\))**: 12 steps (6 h on `uk_pv`, 3 h on `goes_pvdaq`) up to a long horizon of 48 steps to verify long-term decay.
* **Visual Frame Cadence (\(T_v\))**: 8 frames, sampled at a decoupled resolution matching the physical satellite/sky cam intervals.
* **Cadence rule**: windows are defined in *steps*, so the physical lead time differs across datasets. Report the physical lead time next to every per-dataset table and aggregate across datasets only with scale-free statistics (win rate / geometric-mean skill / rank — BASELINE_COMPARISON §4.4); never pool raw step-horizon metrics across cadences. See BASELINE_COMPARISON §4.1.1.

---

## 4. Evaluated Baselines

We categorize and evaluate baselines across multiple levels of complexity:

| Baseline Category | Model | Inputs Used |
| :--- | :--- | :--- |
| **Statistical / Reference** | Smart Persistence (clearness-index persistence) | Target \(Y\) |
| **Classical Machine Learning** | LightGBM / XGBoost / TabPFN | Flattened \(Y, X_{\text{cov}}\) |
| **Deep Learning** | MLP / Temporal Fusion Transformer (TFT) | \(Y, X_{\text{cov}}\) |
| **Time-Series Foundation** | Chronos-2 (Zero-Shot & Fine-Tuned) | Target \(Y\) |
| **Multimodal Foundation** | Solar-VLM / MMTSFM (Prior state) | \(Y, X_{\text{cov}}, V\) |
| **Retrieval-Augmented** | Chronos-2 + TS-RAG (or Cross-RAG) | \(Y\) + retrieved historical windows |

---

## 5. Standardized Evaluation Metrics

All models must output forecasts in the normalized range. Metrics must be computed on the original physical scale (un-normalized using the dataset's plant-specific capacity) or reported as normalized metrics:

1. **Normalized Mean Absolute Error (NMAE)**:
   \[
   \text{NMAE} = \frac{1}{M \cdot H} \sum_{i=1}^{M} \sum_{h=1}^{H} \frac{|\hat{y}_{i,h} - y_{i,h}|}{C_i}
   \]
   *where \(C_i\) is the capacity of plant \(i\), and \(M\) is the total number of evaluation samples.*

2. **Normalized Root Mean Squared Error (NRMSE)**:
   \[
   \text{NRMSE} = \sqrt{\frac{1}{M \cdot H} \sum_{i=1}^{M} \sum_{h=1}^{H} \left( \frac{\hat{y}_{i,h} - y_{i,h}}{C_i} \right)^2}
   \]

3. **Forecast Skill Score (SS)**:
   Relative improvement over the Smart Persistence baseline. **The headline SS is NRMSE-based** (matches BASELINE_COMPARISON §4.2 and the `baselines/` implementation); an NMAE-based SS may be reported as a secondary column but must be labeled as such:
   \[
   \text{Skill Score} = 1 - \frac{\text{NRMSE}_{\text{Model}}}{\text{NRMSE}_{\text{Smart Persistence}}}
   \]

---

## 6. Execution & Configuration Rules

To prevent baseline results from diverging due to environment inconsistencies:
* **Self-Contained Configuration**: The configuration for each baseline (hyperparameters, batch size, target columns) must live entirely within its own folder / codebase inside this repository.
* **Seed Reproductibility**: All baselines must be run with a fixed global seed (`42` by default) for data loaders, weight initialization, and dropouts.
* **SLURM Execution**: GPU-intensive training jobs must be run via the SLURM cluster using the configured scripts in `scripts/`. Standard console logs and error output must be redirected to `logs/slurm/`.
