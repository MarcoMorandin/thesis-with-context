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

### Inference Setup
During evaluation on a test plant:
* The model is given a history window of target power values and covariates: \(Y \in \mathbb{R}^{1 \times T \times 1}\) and \(X_{\text{cov}} \in \mathbb{R}^{1 \times T \times C_{\text{cov}}}\).
* The model is given historical visual frames (satellite PNGs): \(V \in \mathbb{R}^{1 \times T_v \times C_{\text{img}} \times H_{\text{img}} \times W_{\text{img}}}\).
* The model must predict future target power values: \(\hat{Y}_{\text{future}} \in \mathbb{R}^{1 \times H \times 1}\).

---

## 3. Horizon, Cadence, and Window Sizes

Unless overridden by a specific dataset-level configuration, the default temporal configurations are:

* **Granularity**: Intra-hour (typically 5-minute or 15-minute steps, depending on the dataset).
* **History Window (\(T\))**: 24 steps (e.g., 6 hours at 15-min granularity).
* **Forecasting Horizon (\(H\))**: 12 steps (e.g., 3 hours) up to a long horizon of 48 steps (12 hours) to verify long-term decay.
* **Visual Frame Cadence (\(T_v\))**: 8 frames, sampled at a decoupled resolution matching the physical satellite/sky cam intervals.

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
   Relative improvement over the Smart Persistence baseline:
   \[
   \text{Skill Score} = 1 - \frac{\text{Metric}_{\text{Model}}}{\text{Metric}_{\text{Smart Persistence}}}
   \]

---

## 6. Execution & Configuration Rules

To prevent baseline results from diverging due to environment inconsistencies:
* **Self-Contained Configuration**: The configuration for each baseline (hyperparameters, batch size, target columns) must live entirely within its own folder / codebase inside this repository.
* **Seed Reproductibility**: All baselines must be run with a fixed global seed (`42` by default) for data loaders, weight initialization, and dropouts.
* **SLURM Execution**: GPU-intensive training jobs must be run via the SLURM cluster using the configured scripts in `scripts/`. Standard console logs and error output must be redirected to `logs/slurm/`.
