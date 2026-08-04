# MMTSFM: Architecture Deep Dive & Design Rationale

**Multimodal Multiscale Temporal Spatiotemporal Foundation Model**

This document provides an in-depth explanation of the MMTSFM architecture and the technical justifications behind its core design decisions.

***

## 1. The Core Challenge: Multimodal Scaling

Standard time-series models often struggle with "late fusion," where video or imagery is processed independently and only combined with numeric data at the final prediction head. This prevents the model from learning how physical dynamics (like cloud movement or traffic congestion) directly correlate with the evolution of numeric states.

MMTSFM solves this by performing **deep token-level alignment** within the temporal backbone itself.

***

## 2. Decoupled Resolution Architecture

**Decision:** Explicitly separate the temporal regimes of numeric and visual data.

- **Macro-Numeric Baseline:** Processes long-range context (e.g., 1 year of hourly data). This is essential for capturing seasonality, long-term trends, and regime shifts.
- **Micro-Visual Refinement:** Processes a short recent window (e.g., the last 6 hours) at high cadence (e.g., 15-minute frames).
- **Rationale:** Resampling 1 year of video to match hourly TS is computationally impossible; resampling TS to match video frequency dilutes the macro-context. By decoupling, we keep both modalities at their native, most informative resolutions.

***

## 3. Causal Grassmann Mixing (The Temporal Backbone)

**Decision:** Replace $O(L^2)$ Self-Attention with $O(L)$ Grassmann Manifold Mixing.

- **Plücker Embedding:** Instead of computing dot-product similarities, the model encodes the *transition* between consecutive hidden states as a 2D subspace on a Grassmann manifold.
- **Geometry over Magnitude:** Grassmann mixing captures the **direction** of state evolution. In physical systems (meteorology, traffic), the direction of change (e.g., "rapidly cooling" vs. "stable") is often more predictive than the raw value.
- **Efficiency:** Because it is $O(L)$, we can process context lengths of $T=1000+$ (essential for yearly cycles) on a single GPU, whereas standard attention would hit VRAM limits.
- **Multi-Scale:** The model aggregates these geometric transitions across multiple offsets (e.g., $\delta \in \{1, 2, 4, 8, 12, 16\}$), allowing it to track dynamics at different temporal frequencies simultaneously.

***

## 4. Selective Temporal Interleaving

**Decision:** Weave visual tokens into the TS sequence **only** in the refinement window.

- **Mechanism:** In the macro region, the sequence is pure TS. In the refinement region, tokens alternate: `[ts_k, vis_k, ts_{k+1}, vis_{k+1}]`.
- **Deep Fusion:** By interleaving, the Grassmann Flow layer computes **cross-modal Plücker pairs**. It literally measures the "geometric angle" between a numeric state and a visual observation.
- **Minimal Overhead:** For a 1000-token TS context with 24 visual refinement steps, the sequence length only grows to 1024. This is a ~2% overhead, providing deep fusion at a fraction of the cost of full multimodal attention.

***

## 5. Visual Stack: V-JEPA 2.1 & Sensor Projection

**Decision:** Use a frozen, high-capacity spatiotemporal encoder with learned domain adapters.

- **V-JEPA 2.1 vs. VidTok:** While the initial design considered **VidTok** (a VQ-VAE based latent video tokenizer), we pivoted to **V-JEPA 2.1** for several critical reasons:
  - **Predictive Semantics:** VidTok focuses on reconstruction (pixel-level accuracy), which often wastes capacity on noise. V-JEPA 2.1 uses predictive self-supervision, learning features that are "spatially structured and semantically coherent"—ideal for detecting physically relevant patterns like cloud formation or vehicle flow.
  - **Native Temporal Modeling:** V-JEPA processes video clips holistically, encoding motion patterns within its transformer layers. This simplifies the downstream `LatentSummarizer`, which can focus on spatial compression while the backbone handles the temporal dynamics.
  - **Frozen Performance:** V-JEPA is explicitly designed for frozen-backbone use cases. Its features are more robust for linear probing or adapter-based fine-tuning, which is the cornerstone of our "Safe Path" training strategy.
- **Sensor Projection:** To handle the "sensor zoo" (SAR, Multispectral, IR, Radar), we implement a $1 \times 1$ convolution that maps native channels (e.g., 13 bands for Sentinel-2) to the 3 RGB channels expected by V-JEPA. This allows a single pretrained backbone to generalize across heterogeneous physical sensors.
- **Latent Summarizer:** A Perceiver-style cross-attention module that compresses $P$ spatial patches per frame into a single "visual summary token" aligned with the TS cadence.
  - Core implementation: **`VisionChronos2Model`** (`MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py`), **`VidTokEncoder`** (`MMTSFM/src/mmtsfm/models/vision/vidtok_encoder.py`), **`LatentSummarizer`** (`MMTSFM/src/mmtsfm/models/vision/latent_summarizer.py`), and **`CrossModalAdapter`** (`MMTSFM/src/mmtsfm/models/vision/cross_modal_adapter.py`).

***

## 6. Training Curriculum: Pretrained Weight Recycling

**Decision:** A 4-stage curriculum to safely merge Chronos-2 (numeric) and V-JEPA (visual).

1. **Stage 1 (Grassmann Warmup):** Trains the new Grassmann parameters on TS-only data. This prevents the randomly initialized Grassmann layer from corrupting the pretrained Chronos-2 residual stream.
2. **Stage 2a (Visual Alignment):** Learns the mapping from V-JEPA space to Chronos space using a stable late-fusion target.
3. **Stage 2b (Cross-Modal Alignment):** Switches to Interleaved mode, training the Grassmann layer to understand cross-modal (TS-Visual) geometries.
4. **Stage 3 (Joint Fine-tuning):** Full end-to-end optimization across the diverse multimodal corpus.

***

## 7. Multi-Token Probabilistic Forecasting

**Decision:** Non-autoregressive quantile prediction.

- **Non-Autoregressive:** The model predicts the entire horizon $H$ in a single forward pass. This eliminates "error accumulation" where a wrong prediction at $t+1$ ruins the forecast for $t+10$.
- **Quantiles:** Instead of a single "best guess," the model outputs multiple quantiles (e.g., 0.1 to 0.9). This is critical for high-stakes decisions in energy and traffic where understanding uncertainty is as important as the mean value.

***

## 8. Software Engineering: Lightning & Hydra

The project is built on a modern research stack to ensure reproducibility and scalability.

### PyTorch Lightning: The Lifecycle Manager

The training logic is decoupled from the model architecture using **PyTorch Lightning**.

- **VisionChronos2LightningModule:** Encapsulates the forward pass, loss computation, optimizer configuration, and learning rate scheduling.
- **MMTSFMDataModule:** Manages data loading, train/val/test splits, and multi-worker orchestration.
- **Boilerplate-Free:** Lightning handles DDP (Distributed Data Parallel) on the Leonardo cluster, mixed-precision (bf16), and automatic checkpointing, allowing the research code to focus on the architecture.

### Hydra: Hierarchical Configuration

We use **Hydra** to manage the complexity of a multimodal foundation model.

- **Composition:** The configuration is composed from a hierarchy in `MMTSFM/configs/`:
  - `model/`: Hyperparameters for Chronos and V-JEPA (e.g., `d_model`, `num_layers`, `fusion_mode`).
  - `data/`: Dataset settings (e.g., `skippd`, `batch_size`, `hist_steps`).
  - `trainer/`: Hardware-specific settings (e.g., `max_epochs`, `precision`, `slurm` config).
