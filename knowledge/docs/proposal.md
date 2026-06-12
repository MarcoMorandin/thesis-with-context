# MMTSFM — Technical Proposal

**Multimodal Multiscale Temporal Spatiotemporal Foundation Model**

**Status:** Active development — `feat/grassmann-flow-integration`\
\
**References:** [Notion bibliography](https://www.notion.so/c43ccbe3627344e4ba201e1f8262a57e?pvs=21)

***

## Abstract

This proposal introduces a native spatiotemporal foundation model designed to enhance the predictive capabilities of time-series foundation models through early-fusion multimodal reasoning. The core objective is to enable deep cross-modal temporal reasoning for forecasting numeric time-series by extracting physically meaningful dynamics from video and imagery across diverse physical-world sensing domains (solar, meteorology, traffic, hydrology, agriculture, wildfire, ecology), while grounding visual anomalies in structured telemetry and known covariates.

Unlike late-fusion architectures — which encode modalities independently and merge them only at the decision layer — MMTSFM performs deep token-level alignment across time, covariates, and physical entities, enabling multimodal interaction throughout the forecasting backbone.

The architecture has three primary contributions:

**1. Decoupled Resolution Architecture.** The model explicitly separates two temporal regimes: a long macro-numeric context (e.g., 1-year of hourly data) for modeling seasonality, trends, and regime structure; and a short micro-visual refinement window (e.g., last 6 hours at 15-minute visual sampling) for sub-interval dynamics. These regimes are processed at their native resolution without resampling or temporal pooling.

**2. Selective Temporal Interleaving.** Rather than injecting visual tokens only at the group attention layer (late fusion), the model interleaves visual summary tokens with TS tokens *exclusively in the visual refinement window*, feeding the joint sequence directly into the temporal mixing layer. The macro-history processes as pure TS, preserving its temporal geometry. This enables the Grassmann flow layer to track cross-modal state transitions in the recent window while maintaining O(L) complexity over the full long-context sequence. The cost increase is proportional only to $n_{\text{vis}}$ (~2% of $T_{\text{ctx}}$), not to the full context length.

**3. Sensor-Agnostic Visual Encoding.** A pretrained V-JEPA 2.1 spatiotemporal encoder (frozen, then progressively fine-tuned) acts as the visual backbone across all RGB-compatible datasets. A learned per-sensor `SensorProjection` module maps native multi-channel sensor observations (SAR, multispectral, IR, radar) to 3-channel RGB before encoding, enabling a single backbone to serve heterogeneous physical-world sensors without architecture changes.

Additional components: Chronos-2 time-series tokenization (arcsinh normalization, patch segmentation), Perceiver-style latent summarization, a two-level attention backbone (**Time Grassmann Flow → Group**), and non-autoregressive multi-token probabilistic forecasting.

***

## Problem Setup

Given a set of $N$ physical entities, the model receives:

- **Historical target values** $Y \in \mathbb{R}^{N \times T \times C_y}$ — the numeric quantities to forecast
- **Historical and future-known covariates** $X \in \mathbb{R}^{N \times (T+H) \times C_x}$ — structured contextual features known over the forecast horizon
- **Recent visual observations** $V \in \mathbb{R}^{N \times T_v \times C_{\text{sensor}} \times H_{\text{px}} \times W_{\text{px}}}$ — sparsely sampled frames from the last $n_{\text{vis}}$ context steps, at higher temporal frequency than the TS cadence; $C_{\text{sensor}}$ is sensor-specific (3 for RGB cameras, 13 for Sentinel-2, 2 for SAR, etc.)

The model outputs probabilistic forecasts (quantiles) $\hat{Y} \in \mathbb{R}^{N \times H \times Q}$ for each target entity over the forecast horizon $H$.

### Batch data schema

Each training batch produced by `MMTSFMDataset` contains:

```python
{
  "Y":          (num_entities, T, C_target),        # target time series
  "X_cov":      (num_entities, T+H, C_cov),         # covariates (incl. forecast horizon)
  "V":          (num_entities, T_v, C_sensor, H, W), # raw sensor frames (pre-projection)
  "timestamps": (T+H,),                             # unix timestamps
  "entity_ids": (num_entities,),
  "masks":      per-modality visibility masks,
  "sensor_type": str,                               # routes to correct SensorProjection
}
```

***

## Model Architecture

### Overview: data flow

```text
Raw sensor frames [B, T_v, C_sensor, H, W]
        │
        ▼ SensorProjection (learned C_sensor → 3)
        │
        ▼ VisualEncoder: V-JEPA 2.1 (high-cadence)
        │  → [B, T_lat, P, D_v=1024]  spatial patch tokens per frame
        │
        ▼ LatentSummarizer (Perceiver cross-attention, causal)
        │  → [B, n_vis, d_model]  one visual summary token per TS refinement step
        │
        ┌─────────────────────────────────────────────────────────────┐
        │ [B, T_ctx + n_vis + T_fut, d_model]  (interleaved mode)     │
        │   or                                                         │
        │ [B + B*N_soft, T_ctx + T_fut, d_model]  (late-fusion mode)  │
        └─────────────────────────────────────────────────────────────┘
        │
        ▼ Chronos2Encoder  (repeated × num_layers)
        │
        │  1. CausalGrassmannMixing  (or TimeSelfAttention)
        │     → temporal axis — tracks state evolution across time
        │     → sees interleaved [ts, vis] in refinement window
        │
        │  2. GroupSelfAttention
        │     → batch axis — cross-entity + cross-modal fusion
        │
        │  3. FeedForward
        │
        ▼ Output head: last T_fut hidden states → quantile projections
```

***

### 1. Decoupled Resolution Architecture

The core design principle is that the two modalities operate at different temporal resolutions covering different time horizons. Rather than resampling one to match the other, the architecture explicitly maintains both:

| Stream                      | Coverage                                | Granularity                       | Role                                            |
| --------------------------- | --------------------------------------- | --------------------------------- | ----------------------------------------------- |
| **Macro-Numeric Baseline**  | Full historical lookback (e.g., 1 year) | Forecasting cadence (e.g., 1h)    | Macro-seasonality, trends, regime structure     |
| **Micro-Visual Refinement** | Recent window only (e.g., last 6h)      | Sub-cadence (e.g., 15-min frames) | Sub-interval dynamics, visual anomaly detection |

The `MMTSFMDataLoader` generates aligned sliding windows across both streams, maintaining shared chronological anchors. The macro TS context covers $T_{\text{ctx}}$ patches; the visual context covers $n_{\text{vis}}$ patches (the last $n_{\text{vis}}$ positions of the TS sequence), with $T_v$ raw frames sampled within that window at a finer resolution.

***

### 2. Sensor Projection

Physical-world sensors produce observations in heterogeneous spectral configurations: RGB cameras (3 channels), Sentinel-2 (13 bands), Sentinel-1 SAR (VV + VH), GOES-16 (16 channels), NEXRAD radar (reflectivity + velocity + spectrum width), thermal cameras (1 channel). The `VisualEncoder` backbone (V-JEPA 2.1) expects 3-channel RGB input.

A lightweight `SensorProjection` module maps each sensor's native channels to 3:

```text
SensorProjection(in_channels: int)
  → nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
```

One `SensorProjection` per sensor type, applied per frame before the visual backbone. Initialization: identity mapping for the first 3 channels (or channel replication for $C_{\text{sensor}} < 3$), so training starts from a meaningful RGB approximation.

The learned projection is not a fixed pseudocolor mapping — it is trained jointly in Stage 2a and learns the **optimal 3-channel compression for discriminative feature extraction** in each domain. For example, for Sentinel-2 the network can learn to emphasize SWIR and NIR bands over raw RGB if those carry more forecasting-relevant signal (e.g., NDVI, soil moisture).

This approach is strictly superior to fixed pseudocolor mappings and avoids the need for a sensor-specific backbone architecture.

***

### 3. Visual Encoding

#### V-JEPA 2.1

**V-JEPA 2.1** (Meta AI, March 2026 — arXiv:2603.14482) is the visual backbone for all datasets.

Key properties relevant to MMTSFM:

- **Dense predictive self-supervision**: the 2.1 variant introduces per-token spatial supervision and deep self-supervision across encoder layers, producing "spatially structured, semantically coherent, and temporally consistent" patch representations — exactly what `LatentSummarizer`'s cross-attention needs as KV input.
- **Native temporal modeling**: V-JEPA processes video clips holistically, encoding temporal motion patterns internally. This means `LatentSummarizer` compresses primarily spatially (not temporally), simplifying its role.
- **Frozen backbone compatibility**: explicitly designed for frozen use; evaluation protocol uses frozen encoder + lightweight head, which matches Stage 2a.
- **Full code**: `github.com/facebookresearch/vjepa2` — training scripts, SLURM support, PyTorch Hub.

Output shape: `[B, T_lat, P, D_v=1024]` (ViT-L) — directly compatible with `LatentSummarizer`'s KV input.

***

### 4. Time-Series Tokenization (Chronos-2)

1. **Instance normalization** via arcsinh scaling: $\tilde{y}_t = \sinh^{-1}\!\left(\frac{y_t - \mu}{\sigma}\right)$ — stabilizes heavy-tailed and sparse physical distributions without clipping.
2. **Non-overlapping patch segmentation**: contiguous windows of `input_patch_size` timesteps flattened into patch vectors.
3. **Patch embedding**: each patch $(p, m, \tau)$ — values, mask, time encoding — projected via a residual MLP to $\mathbb{R}^d$.

***

### 5. Embedding Layer

All tokens are projected into a shared geometric space via a unified embedding schema applied additively:

| Embedding                           | Dimension          | Purpose                                              |
| ----------------------------------- | ------------------ | ---------------------------------------------------- |
| **Patch embedding** (ResidualBlock) | $d_{\text{model}}$ | Learned projection from raw patch features           |
| **Modality type**                   | $d_{\text{model}}$ | 0 = numeric, 1 = visual                              |
| **Segment type**                    | $d_{\text{model}}$ | 0 = context (past), 1 = future horizon               |
| **Token type**                      | $d_{\text{model}}$ | 0 = target, 1 = covariate, 2 = visual soft token     |
| **Temporal positional encoding**    | via RoPE           | Relative position along the time axis                |

All embeddings use `std=0.02` initialization to avoid saturating the pretrained Chronos-2 residual stream.

***

### 6. Latent Summarization

`LatentSummarizer` bridges the visual backbone output and the TS forecasting cadence via Perceiver-style causal cross-attention. It acts as a **spatial compressor** — collapsing $P$ spatial patches per refinement step into a single summary token, since V-JEPA already encodes temporal dynamics across the clip.

**Architecture:**

- **Queries**: $n_{\text{vis}}$ learned latent tokens $Q \in \mathbb{R}^{n_{\text{vis}} \times d}$, one per TS refinement step.
- **Keys/Values**: flattened visual backbone tokens projected to $d_{\text{model}}$ via `kv_proj: Linear(D_v=1024, d_model)`.
- **Causal mask**: query at step $k$ attends only to frames in $[0, \lceil (k+1)T_{\text{lat}} / n_{\text{vis}} \rceil - 1]$. Built once per forward call, broadcast across batch and heads.
- **Frame availability mask**: `key_padding_mask` blocks corrupted or missing frames.

**Output**: `[B, n_vis, d_model]` — one compact Visual Summary Token per entity per TS refinement step.

**Null visual token**: For macro positions ($T_M = T_{\text{ctx}} - n_{\text{vis}}$ steps outside the visual window), a learned null token $\mathbf{e}_{\text{null}} \in \mathbb{R}^d$ (initialized $\mathcal{N}(0, d^{-1/2})$) fills the position. This prevents degenerate Plücker subspaces at the macro/refinement boundary.

***

### 7. Selective Temporal Interleaving

> Full documentation: `docs/temporal-interleaving.md`

Visual summary tokens are interleaved with TS tokens **exclusively in the visual refinement window** before the temporal mixing layer — not across the full context.

#### Sequence construction

Let $T_M = T_{\text{ctx}} - n_{\text{vis}}$. The encoder input is:

$$
\mathbf{S} = \underbrace{[\text{ts}_0,\; \ldots,\; \text{ts}_{T_M - 1}]}_{\text{macro: } T_M \text{ pure-TS tokens}} \;\|\; \underbrace{[\text{ts}_{T_M}, v_{T_M}, \ldots, \text{ts}_{T_{\text{ctx}}-1}, v_{T_{\text{ctx}}-1}]}_{\text{refinement: } 2n_{\text{vis}} \text{ interleaved tokens}} \;\|\; \underbrace{[\text{fut}_0,\; \ldots]}_{\text{future}}
$$

Total context length: $T_{\text{ctx}} + n_{\text{vis}}$ — only $n_{\text{vis}}$ extra tokens. Token-count overhead is $n_{\text{vis}} / T_{\text{ctx}}$; for the reference setup (1-year hourly history, 6h × 15-min refinement, `input_patch_size = 16` ⇒ $T_{\text{ctx}} \approx 547$ patches, $n_{\text{vis}} = 6$) this is **≈1.1%**. The figure scales with refinement-window length, not horizon: longer macro context lowers the ratio further. Quoted as "≈2%" elsewhere as a loose upper bound for typical configurations ($n_{\text{vis}} \le 0.02\,T_{\text{ctx}}$).

#### Causal access pattern

- $\text{ts}_{T_M+k}$ (position $T_M+2k$): attends to full macro history + all prior refinement pairs. Cannot attend to $v_{T_M+k}$ at $T_M+2k+1$.
- $v_{T_M+k}$ (position $T_M+2k+1$): attends to everything above, plus $\text{ts}_{T_M+k}$.

#### Two fusion variants

|                      | Variant A: Grassmann                                   | Variant B: TimeSelfAttention      |
| -------------------- | ------------------------------------------------------ | --------------------------------- |
| **Temporal layer**   | `CausalGrassmannMixing`                                | `TimeSelfAttention` + RoPE        |
| **Macro pairs**      | $(ts_{t-1}, ts_t)$ — pure TS, unaffected               | Full causal attention, pure TS    |
| **Refinement pairs** | $(ts_k, v_k)$, $(v_k, ts_{k+1})$ — cross-modal Plücker | Full causal cross-modal attention |
| **Cost increase**    | $\approx +2\%$ — O(L) preserved                        | $\approx +4.4\%$ — O($L^2$)       |
| **Role**             | Primary contribution                                   | Diagnostic ablation               |

#### Fusion mode configuration

`VisionChronos2Config.fusion_mode`:

- `"late"` — visual tokens injected at GroupSelfAttention via batch-dim concatenation (`N_soft` rows per entity). `CrossModalAdapter` active.
- `"interleaved"` — selective temporal interleaving. `CrossModalAdapter` bypassed.

`Chronos2CoreConfig.use_grassmann` independently selects Variant A or B.

> **Training-time switching:** start with `fusion_mode="late"` during Stage 2a (alignment), switch to `"interleaved"` in Stage 2b (Grassmann alignment). The late-fusion path provides a stable alignment target before cross-modal Plücker pairs are introduced.

***

### 8. Attention Backbone

Each `Chronos2EncoderBlock` applies three operations in sequence:

```text
Time Grassmann Flow  →  Group Self-Attention  →  FeedForward
```

#### 8.1 Time Grassmann Flow (`CausalGrassmannMixing`)

Replaces O($L^2$) temporal self-attention with an O($L$) attention-free layer. For each position $i$ and offset $\delta \in \{1, 2, 4, 8, 12, 16\}$:

1. **Reduction**: $\mathbf{z}_i = W_{\text{red}} \mathbf{h}_i \in \mathbb{R}^r$ (must have even $r$ for RoPE).
2. **RoPE phase injection**: rotary embeddings applied to $\mathbf{z}$ for temporal position awareness.
3. **Plücker encoding**: @@TOLARIA_MATH_BLOCK:%5Cmathbf%7Bp%7D_%7Bi%2C%5Cdelta%7D%20%3D%20%5Cfrac%7B%5Cmathbf%7Bz%7D_%7Bi-%5Cdelta%7D%20%5Cwedge%20%5Cmathbf%7Bz%7D_i%7D%7B%5C%7C%5Cmathbf%7Bz%7D_%7Bi-%5Cdelta%7D%20%5Cwedge%20%5Cmathbf%7Bz%7D_i%5C%7C%20%2B%20%5Cvarepsilon%7D%20%5Cin%20G(2%2C%20r)%2C%20%5Cquad%20%5Cdim(%5Cmathbf%7Bp%7D)%20%3D%20%5Cbinom%7Br%7D%7B2%7D@@
4. **Projection**: $\mathbf{g}_{i,\delta} = W_{\text{plu}} \mathbf{p}_{i,\delta} \in \mathbb{R}^d$.
5. **Multi-scale aggregation**: softmax-weighted sum over valid offsets.
6. **Gated fusion**: @@TOLARIA_MATH_INLINE:%5Cmathbf%7Bh%7D'*i%20%3D%20%5Calpha_i%20%5Codot%20%5Cmathbf%7Bh%7D_i%20%2B%20(1-%5Calpha_i)%20%5Codot%20%5Cmathbf%7Bg%7D_i@@, @@TOLARIA_MATH_INLINE:%5Calpha_i%20%3D%20%5Csigma(W*%7B%5Ctext%7Bgate%7D%7D%5B%5Cmathbf%7Bh%7D_i%20%5C%7C%20%5Cmathbf%7Bg%7D_i%5D)@@.

Key properties: causal (pairs only with past), scale-invariant (Plücker normalization), O(L), multi-scale. With selective interleaving, refinement pairs are cross-modal (TS, visual); macro pairs are pure TS — both handled transparently by the same layer.

##### Modality semantics across offsets

In the interleaved refinement window the modality of pair $(i-\delta, i)$ depends on parity of $\delta$ relative to the local stride. With pattern `[ts_k, v_k, ts_{k+1}, v_{k+1}, ...]`:

| $\delta$  | Pair at TS query $\text{ts}_{T_M+k}$ (pos $T_M{+}2k$)                     | Pair at visual query $v_{T_M+k}$ (pos $T_M{+}2k{+}1$) | Semantics                    |
| --------- | ------------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------- |
| 1         | $(v_{T_M+k-1}, \text{ts}_{T_M+k})$                                        | $(\text{ts}_{T_M+k}, v_{T_M+k})$                      | cross-modal                  |
| 2         | $(\text{ts}_{T_M+k-1}, \text{ts}_{T_M+k})$                                | $(v_{T_M+k-1}, v_{T_M+k})$                            | unimodal (TS-TS / VV)        |
| 4         | TS-TS (stride 2)                                                          | V-V (stride 2)                                        | unimodal                     |
| 8, 12, 16 | unimodal (skipping back into pure-macro region for large enough $\delta$) | unimodal                                              | unimodal / boundary-crossing |

Offset 1 is the *only* offset producing genuinely cross-modal Plücker subspaces inside refinement; even offsets produce TS-TS or V-V pairs whose Plücker geometries live on different statistical manifolds and should not share a single $W_{\text{plu}}$ projection. Boundary offsets (e.g., $\delta = 8$ from a refinement query reaching back into macro) additionally mix interleaved- and pure-cadence pairs.

**Mitigation: modality-aware offset gating.** The aggregation softmax over offsets is augmented with a per-pair modality-pair embedding $\mathbf{m}_{\delta,i} \in \{TT, TV, VT, VV\}$, added to the offset logit before softmax: $\ell_{i,\delta} = \langle \mathbf{q}_i, \mathbf{k}_\delta \rangle + b_{\mathbf{m}_{\delta,i}}$. Four learned scalar biases — one per modality-pair class — let the model down-weight offsets that produce semantically incoherent pairs in the refinement window without disturbing pure-TS macro behavior. Cost: 4 extra parameters; no asymptotic change.

**Config** (`Chronos2CoreConfig`): `grassmann_reduced_dim` (default 32, must be even), `grassmann_window_offsets` (default `[1,2,4,8,12,16]`), `grassmann_plucker_eps` (default 1e-8), `use_grassmann` (bool), `grassmann_modality_pair_bias` (bool, default `true` when `fusion_mode="interleaved"`).

#### 8.2 Group Self-Attention

Computes self-attention *across the batch axis* at each sequence position over all tokens sharing the same `group_id`. Fuses: target TS tokens from different entities, covariate tokens, visual soft tokens (late-fusion mode). RoPE is not applied (no natural ordering along the batch/entity axis). The `group_time_mask` is the outer product of the group identity mask and the temporal padding mask.

With selective interleaving, group attention at refinement positions $T_M + 2k$ and $T_M + 2k + 1$ fuses visual tokens cross-entity.

#### 8.3 FeedForward

Position-wise MLP with residual: $\mathbf{h}' = \mathbf{h} + \text{Dropout}(W_2 \cdot \text{act}(W_1 \cdot \text{LayerNorm}(\mathbf{h})))$. Default activation: ReLU.

***

### 9. Multi-Token Prediction

All $H$ future timesteps are predicted in a **single forward pass** (non-autoregressive). The encoder receives full context plus future-covariate patches; the last $T_{\text{fut}}$ output embeddings are decoded in parallel:

$$
\hat{Y}_{b, h, q} = \big[W_{\text{out}}\, \mathbf{h}_{b,\, T_{\text{ctx}} + k}^{(L)}\big]_{q,\, j}, \quad h = k \cdot p_{\text{out}} + j,\; k \in [0, T_{\text{fut}}),\; j \in [0, p_{\text{out}})
$$

Benefits: no temporal error accumulation, reduced inference latency, naturally probabilistic ($Q$ quantiles per output patch). Inspired by Moirai 2.0.

***

### 10. Output Head

1. **Quantile projection** (ResidualBlock): $d_{\text{model}} \to Q \times p_{\text{out}}$, where $p_{\text{out}}$ is the output patch size (timesteps decoded per output embedding). With $T_{\text{fut}}$ output embeddings, total horizon coverage is $H = T_{\text{fut}} \cdot p_{\text{out}}$.
2. **Instance norm inversion**: rescale from normalized space to operational units using stored $(loc, scale)$.

Default quantiles: $[0.1, 0.2, \ldots, 0.9]$. Training loss: pinball (quantile) loss, masked on future-known covariate positions.

***

## Training Strategy

### Modality Dropout (Asymmetric Bernoulli Masking)

Independent stochastic masking per sample during training:

- **Visual stream**: dropped with $p_v = 0.5$ — zeroes all visual slots for that sample.
- **Numeric stream**: dropped with $p_n = 0.1$, *only* when visual stream is active — prevents both streams zeroed simultaneously.

Effective numeric drop rate: $p_n \cdot (1 - p_v) = 0.05$.

In interleaved mode, visual dropout zeroes only the $2n_{\text{vis}}$ interleaved visual slots, not the macro TS region.

***

### Pretrained Model Recycling Strategy

MMTSFM is built on two pretrained models — Chronos-2 and V-JEPA 2.1 — whose embedding spaces are **independent and initially incompatible**. Understanding exactly what transfers and what does not is essential for designing the training curriculum.

#### Chronos-2

| Component                                      | Recycle?             | Reason                                                                   |
| ---------------------------------------------- | -------------------- | ------------------------------------------------------------------------ |
| arcsinh normalization + patch segmentation     | ✅ fully              | Pure preprocessing; domain-agnostic; universally correct for physical TS |
| Input patch embedding (ResidualBlock)          | ✅ fully              | Encodes universal TS patch representations across hundreds of domains    |
| Group self-attention weights                   | ✅ fully              | Multivariate mixing; domain-agnostic                                     |
| Feed-forward weights                           | ✅ fully              | Feature transformation; domain-agnostic                                  |
| Output quantile projection head                | ✅ fully              | Calibrated to normalized TS values                                       |
| **TimeSelfAttention weights**                  | ❌ irrelevant         | Replaced by `CausalGrassmannMixing` from scratch                         |
| `CausalGrassmannMixing` (W_red, W_plu, W_gate) | ❌ train from scratch | New component; does not exist in pretrained checkpoint                   |

**~80% of Chronos-2 parameters transfer directly.** The Grassmann layer is always new. Retraining Chronos-2 from scratch would be prohibitively expensive and unnecessary — its TS breadth (pretrained on hundreds of millions of diverse TS) is irreplaceable.

**Critical risk at Stage 1:** Randomly initialized Grassmann parameters produce arbitrary output. The gated fusion $\alpha = \sigma(W_{\text{gate}}[\mathbf{h} \| \mathbf{g}])$ could corrupt the pretrained Chronos-2 residual stream at initialization if the gate passes random Grassmann output. **Mitigation:** use a dedicated Grassmann warmup in Stage 1 — apply a 0.1× LR multiplier to all Grassmann parameters for the first 2,000 steps, allowing the pretrained residual stream to anchor the Grassmann layer before it learns to contribute.

#### V-JEPA 2.1

| Component                         | Recycle?                     | Reason                                                                                                  |
| --------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| V-JEPA 2.1 spatiotemporal encoder | ✅ fully (frozen → fine-tune) | General motion + structure useful across domains; domain semantics learned via LatentSummarizer queries |
| VidTok decoder / KL-4ch latents   | ❌ not used                   | Replaced entirely by V-JEPA 2.1                                                                         |

**Critical constraint:** V-JEPA 2.1 was pretrained on general internet video (humans, objects, indoor/outdoor). Its features encode general spatiotemporal patterns but **not** domain-specific physical semantics (cloud optical depth, storm propagation, crop stress signals). All domain adaptation happens via two mechanisms:

1. `SensorProjection` — learns to emphasize domain-relevant channels before encoding
2. `LatentSummarizer` learned queries — act as domain-specific feature extractors on top of general V-JEPA features

This is analogous to CLIP's frozen visual encoder repurposed via learned adapters: the backbone provides strong generic features; the adapter specializes them.

#### New components (always trained from scratch)

| Component                                                  | Parameters (approx.) | Training starts  |
| ---------------------------------------------------------- | -------------------- | ---------------- |
| `SensorProjection` (per sensor)                            | ~few thousand        | Stage 2a         |
| `LatentSummarizer` (kv_proj + cross-attn + null token)     | ~5M                  | Stage 2a         |
| `CrossModalAdapter` (late-fusion path only)                | ~2M                  | Stage 2a         |
| `MultimodalEmbedding` (modality/segment/token-type)        | ~4M                  | Stage 2a         |
| `CausalGrassmannMixing` (W_red + W_plu + W_gate)           | ~3M                  | Stage 1 (warmup) |

New components are small relative to the full model (Chronos-2 ~200M + V-JEPA 2.1 ~300M). Training them efficiently is achievable without full-model compute budget.

***

### Multi-Stage Training Curriculum

Training proceeds in four stages designed around the pretrained weight recycling constraints above.

#### Stage 1 — TS Pretraining and Grassmann Initialization

| Frozen / not instantiated                                  | Trainable                                                                               | Data                                         | Purpose                                                                                                     |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| All vision modules **skipped at construction** (see below) | Chronos-2 encoder (group attn + FF + embeddings) + `CausalGrassmannMixing` (at 0.1× LR) | Diverse TS-only datasets (visual mask = 1.0) | Establish strong numeric temporal geometry; initialize Grassmann compatibly with pretrained residual stream |

Load Chronos-2 pretrained weights for all components except `CausalGrassmannMixing`. Apply 0.1× LR multiplier to Grassmann parameters for the first 2,000 warmup steps, then restore normal LR. Train on diverse TS data — include all TS-only datasets plus the TS streams of multimodal datasets with 100% visual masking. This produces a Grassmann layer that has learned to encode meaningful TS temporal subspaces before encountering visual tokens.

> **Vision-module skip:** with `data.visual_mask_prob = 1.0` no visual tensor reaches the encoder, so `VisualEncoder`, `SensorProjection`, `LatentSummarizer`, and `CrossModalAdapter` should not be instantiated in Stage 1 — guard them behind a `model.skip_vision_stack=true` flag rather than loading and freezing ~300M of V-JEPA weights into GPU memory unused. Stage 2a constructs the full multimodal stack from scratch (or from a Stage 0 vision warmup checkpoint) and resumes the Chronos-2 weights from the Stage 1 checkpoint.

#### Stage 2a — Visual Alignment (Late Fusion)

| Frozen                             | Trainable                                                                                                                           | Data                             | Purpose                                                                                                      |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Full Chronos-2 encoder + Grassmann | V-JEPA 2.1 (partial unfreeze last 4 layers) + `SensorProjection` + `LatentSummarizer` + `CrossModalAdapter` + `MultimodalEmbedding` | Multimodal datasets, $p_v = 0.7$ | Align visual embedding space to Chronos-2 numeric space; train sensor projections; keep `fusion_mode="late"` |

Partial V-JEPA 2.1 unfreeze (last 4 transformer layers) allows domain adaptation without disrupting the full pretrained backbone. `SensorProjection` learns optimal spectral compression per sensor type. `LatentSummarizer` queries learn to extract domain-relevant features from V-JEPA's general representations. `fusion_mode="late"` is used throughout — the Grassmann layer must not encounter cross-modal pairs until it is explicitly trained for them.

#### Stage 2b — Grassmann Cross-Modal Alignment (Interleaved)

| Frozen                                                                                | Trainable                             | Data                             | Purpose                                                                                                                                       |
| ------------------------------------------------------------------------------------- | ------------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Chronos-2 encoder except Grassmann {W_red, W_plu, W_gate}; V-JEPA 2.1 fully re-frozen | Grassmann params + `LatentSummarizer` | Multimodal datasets, $p_v = 0.5$ | Co-adapt visual projection and Grassmann reduction to produce meaningful cross-modal Plücker subspaces; switch to `fusion_mode="interleaved"` |

Switch `fusion_mode` to `"interleaved"`. Now the Grassmann layer sees cross-modal pairs (ts, vis) in the refinement window for the first time. Training the Grassmann reduction $W_{\text{red}}$ jointly with `LatentSummarizer`'s `kv_proj` allows the two to co-adapt: visual tokens learn to lie in a region of $\mathbb{R}^{512}$ where $W_{\text{red}}$ projects them into geometrically coherent Plücker subspaces alongside TS tokens.

> **Note on shared-weight stability:** Grassmann parameters $\{W_{\text{red}}, W_{\text{plu}}, W_{\text{gate}}\}$ are position-shared — the same weights apply to macro pairs (pure TS) and refinement pairs (cross-modal). Updates therefore affect both regimes simultaneously; there is no "frozen macro Grassmann" to anchor against catastrophic forgetting. Stability instead relies on: (a) the macro region still consisting overwhelmingly of pure-TS pairs ($T_M \gg n_{\text{vis}}$), so the gradient signal remains TS-dominated; (b) the gated fusion $\alpha = \sigma(W_{\text{gate}}[\mathbf{h} \| \mathbf{g}])$, which can locally suppress the Grassmann path if early cross-modal updates degrade pure-TS subspaces; (c) a low LR multiplier (recommended 0.3×) on Grassmann params during the first 1,000 steps of Stage 2b, mirroring the Stage 1 warmup.

#### Stage 3 — Full Joint Fine-Tuning

| Frozen  | Trainable      | Data                           | Purpose                                                         |
| ------- | -------------- | ------------------------------ | --------------------------------------------------------------- |
| Nothing | All components | Full diverse multimodal corpus | End-to-end cross-modal temporal optimization across all domains |

Progressive V-JEPA 2.1 unfreezing (4 more layers per epoch) prevents early-stage feature corruption. Full asymmetric Bernoulli modality masking enforces modality-robust representations. Training on the **full diverse dataset corpus** — not a single dataset — is what distinguishes a foundation model from a fine-tuned specialist: the model must generalize zero-shot across solar, traffic, meteorology, hydrology, agriculture.

***

## Implementation

### Key source files

| File                                              | Role                                                                                                               |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `MMTSFM/src/mmtsfm/models/chronos2/config.py`     | `Chronos2CoreConfig` — backbone hyperparameters; `d_model`, `use_grassmann`, `grassmann_*`                         |
| `MMTSFM/src/mmtsfm/models/chronos2/grassmann.py`  | `CausalGrassmannMixing` — O(L) Plücker temporal mixing                                                             |
| `MMTSFM/src/mmtsfm/models/chronos2/layers.py`     | `TimeSelfAttention`, `GroupSelfAttention`, `MHA`, `RoPE`, `FeedForward`                                            |
| `MMTSFM/src/mmtsfm/models/chronos2/model.py`      | `Chronos2Encoder`, `Chronos2Model` — full encoder stack                                                            |
| `MMTSFM/src/mmtsfm/models/chronos2/chronos_bolt.py` | Patch segmentation, arcsinh instance normalization                                                                 |
| `MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py` | `VisionChronos2Model` — multimodal assembly; `interleave_sequences`; `fusion_mode` routing; `VisionChronos2Config` |
| `MMTSFM/src/mmtsfm/models/chronos2/lightning_module.py` | `VisionChronos2LightningModule` — training loop, LR schedule, optimizer                                            |
| `MMTSFM/src/mmtsfm/models/vision/latent_summarizer.py` | `LatentSummarizer` — Perceiver compressor with causal mask + null token                                            |
| `MMTSFM/src/mmtsfm/models/vision/cross_modal_adapter.py` | `CrossModalAdapter` — N_soft soft tokens *(late-fusion path only)*                                                 |
| `MMTSFM/src/mmtsfm/models/vision/vidtok_encoder.py` | `VidTokEncoder` → **to be replaced by** `VisualEncoder` wrapping V-JEPA 2.1                                        |
| `MMTSFM/src/mmtsfm/models/vision/sensor_projection.py` | `SensorProjection` — per-sensor learned C_sensor → 3 projection *(to be implemented)*                              |
| `MMTSFM/src/mmtsfm/data/dataset.py`               | `MMTSFMDataset` — synthetic and SKIPPD loaders                                                                     |
| `MMTSFM/src/mmtsfm/train.py`                      | Hydra entry point                                                                                                  |

### Key config changes vs. current implementation

| Config field                          | Current             | Updated                                              |
| ------------------------------------- | ------------------- | ---------------------------------------------------- |
| `vision_cfg.d_video_latent`           | `4` (VidTok KL-4ch) | `1024` (V-JEPA 2.1 ViT-L) or `768` (ViT-B)           |
| `vision_cfg.vidtok_cfg_path`          | VidTok YAML         | *(removed)*                                          |
| `vision_cfg.vidtok_ckpt_path`         | VidTok checkpoint   | *(removed)*                                          |
| `vision_cfg.visual_encoder_type`      | *(missing)*         | `"vjepa2"`                                           |
| `vision_cfg.visual_encoder_ckpt_path` | *(missing)*         | Path to V-JEPA 2.1 checkpoint                        |
| `vision_cfg.sensor_type`              | *(missing)*         | `"rgb"`                                              |
| `vision_cfg.freeze_visual_encoder`    | *(missing)*         | `true` (Stage 1-2a), `"partial"` (2b), `false` (3)   |
| `vision_cfg.fusion_mode`              | `"late"`            | `"late"` (Stage 1-2a) → `"interleaved"` (Stage 2b-3) |

### Fusion mode routing

```text
fusion_mode = "late"
  → LatentSummarizer → CrossModalAdapter → [B*N_soft, T_ctx, d] batch rows
  → stacked with TS rows → encoder (GroupSelfAttention fuses at each step)

fusion_mode = "interleaved"
  → LatentSummarizer → selective_interleave([B, T_M, d], [B, 2*n_vis, d])
  → [B, T_ctx + n_vis + T_fut, d] single sequence per entity
  → encoder (CausalGrassmannMixing sees cross-modal pairs in refinement window)
```

### Running

All running commands should be executed from within the `MMTSFM/` directory:

```bash
cd MMTSFM

# Stage 1 (TS pretraining, Grassmann warmup)
uv run python -m mmtsfm.train \
  model.vision_cfg.fusion_mode=late \
  model.vision_cfg.freeze_visual_encoder=true \
  data.visual_mask_prob=1.0

# Stage 2a (visual alignment, late fusion)
uv run python -m mmtsfm.train \
  model.vision_cfg.fusion_mode=late \
  model.vision_cfg.freeze_visual_encoder=partial \
  model.freeze_chronos=true

# Stage 2b (Grassmann cross-modal alignment, interleaved)
uv run python -m mmtsfm.train \
  model.vision_cfg.fusion_mode=interleaved \
  model.chronos_core_cfg.use_grassmann=true \
  model.freeze_chronos=true  # only Grassmann params trainable

# Stage 3 (full joint training)
uv run python -m mmtsfm.train \
  model.vision_cfg.fusion_mode=interleaved \
  model.chronos_core_cfg.use_grassmann=true \
  model.freeze_chronos=false

# Cluster (SLURM)
sbatch MMTSFM/scripts/slurm_train.sh \
  model.chronos_core_cfg.d_model=512 \
  model.vision_cfg.d_video_latent=1024
```

***

## Datasets

> All datasets require download, exploration, and standardization. TS-only datasets are used in Stage 1 (100% visual masking). Non-RGB sensors use `SensorProjection` to map to 3-channel RGB before the visual backbone.

### Mobility

| Dataset                     | Sensor type | SensorProjection  | Modalities                                              | Notes                                                                |
| --------------------------- | ----------- | ----------------- | ------------------------------------------------------- | -------------------------------------------------------------------- |
| **I-24 MOTION**             | RGB camera  | None (native RGB) | Visual: highway cameras · TS: vehicle trajectories      | Natively complete multimodal.                                        |
| **I24-WaveX**               | RGB camera  | None              | Visual: I-24 camera · TS: radar detector TS (30s)       | Multi-rate (radar + camera). Ideal for macro/micro split validation. |
| **MARVEL**                  | RGB CCTV    | None              | Visual: CCTV · TS: pedestrian/vehicle counts            |                                                                      |
| **City-scale Trajectories** | RGB camera  | None              | Visual: traffic images · TS: 5M trajectories            |                                                                      |

### Meteorology & Earth Observation

| Dataset                    | Sensor type          | SensorProjection     | Modalities                                                         | Notes                             |
| -------------------------- | -------------------- | -------------------- | ------------------------------------------------------------------ | --------------------------------- |
| **MeteoNet**               | Radar + satellite    | Radar: 1→3; Sat: 3→3 | Visual: radar (5min) + satellite · TS: 500 stations                | Strong multi-rate structure.      |
| **MP-Bench**               | Meteorological grids | N→3                  | Visual: 4D gridded fields · TS: multi-year + weather events        |                                   |
| **EarthNet2021**           | Sentinel-2           | 13→3                 | Visual: Sentinel-2 · TS: E-OBS weather                             |                                   |
| **SEVIR / SVRIMG**         | GOES-16 + NEXRAD     | IR→3; Radar→3        | Visual: satellite + radar · TS: lightning + storm metadata         |                                   |
| **NOAA NEXRAD + GOES ABI** | Radar + satellite    | N→3                  | Visual: satellite + 3D radar · TS: integrate ASOS/AWOS             | Heavy preprocessing required.     |
| **ERA5 (ECMWF)**           | Atmospheric grids    | N→3                  | Visual: 2D/3D gridded fields · TS: 40+ years hourly               | Stage 1 TS pretraining candidate. |
| **RainBench**              | Satellite + radar    | N→3                  | Visual: precipitation maps · TS: ERA5-derived                      |                                   |

### Earth Observation & Wildfire

| Dataset        | Sensor type             | SensorProjection   | Modalities                                                  | Notes        |
| -------------- | ----------------------- | ------------------ | ----------------------------------------------------------- | ------------ |
| **FireSentry** | RGB + thermal UAV       | Thermal: 1→3       | Visual: UAV video · TS: wind, temp, humidity                |              |
| **TS-SatFire** | VIIRS multispectral     | N→3                | Visual: multi-spectral imagery · TS: GRIDMET/GFS            |              |
| **TerraMesh**  | Sentinel-1 + Sentinel-2 | SAR: 2→3; S2: 13→3 | Visual: SAR + optical · TS: NDVI, land cover                | CVPRW 2025.  |
| **MillionST**  | Satellite               | N→3                | Visual: 1M patches · TS: 10 temporal phases / 5yr           |              |

### Photovoltaic Energy Production

| Dataset                      | Sensor type             | SensorProjection | Modalities                                                     | Notes                                                |
| ---------------------------- | ----------------------- | ---------------- | -------------------------------------------------------------- | ---------------------------------------------------- |
| **SKIPP'D**                  | Fisheye RGB             | None             | Visual: sky camera · TS: PV power (1 min)                      | **Primary dev dataset** — read directly from `/Volumes/SSD/standardized-dataset/solar/skippd/`. |
| **Girasol**                  | IR + visible fisheye    | IR: 1→3          | Visual: IR + visible · TS: GSI + sun position                  |                                                      |
| **SolarBench (SkyImageNet)** | RGB sky camera          | None             | Visual: harmonized sky cameras · TS: irradiance + power        | ICLR 2024 Climate Change AI.                         |
| **SolarNet**                 | RGB sky camera          | None             | Visual: cloud images · TS: pyranometer irradiance              |                                                      |
| **SIRTA & DEWA**             | RGB sky camera          | None             | Visual: sky images (France + UAE) · TS: local irradiance       | Distinct climate zones for generalization.           |
| **GOES-16/18 ABI + NSRDB**   | Geostationary satellite | 16→3             | Visual: 5-15 min satellite · TS: GHI/DNI + weather             | Large-scale multi-entity.                            |

### Agriculture

| Dataset                   | Sensor type                  | SensorProjection | Modalities                                                          | Notes |
| ------------------------- | ---------------------------- | ---------------- | ------------------------------------------------------------------- | ----- |
| **CYCleSS**               | Satellite RS                 | N→3              | Visual: RS data (5-day) · TS: daily weather + yearly yield          |       |
| **CropClimateX**          | Sentinel-1/2, Landsat, MODIS | Multi→3          | Visual: multi-sensor imagery · TS: Daymet + drought maps            |       |
| **California Crop Yield** | Landsat multispectral        | N→3              | Visual: Landsat sequences · TS: daily climate + evapotranspiration  |       |

### Hydrology

| Dataset                             | Sensor type         | SensorProjection | Modalities                                                   | Notes |
| ----------------------------------- | ------------------- | ---------------- | ------------------------------------------------------------ | ----- |
| **Planet SkySat River Video**       | RGB satellite video | None             | Visual: sub-daily satellite · TS: river discharge + ADCP     |       |
| **Xiaomai Island Wave Dataset**     | RGB shore camera    | None             | Visual: wave monitoring video · TS: buoy height + SWAN       |       |
| **IceNet**                          | Satellite sea ice   | N→3              | Visual: sea ice concentration maps · TS: climate variables   |       |
| **Sentinel-1 Global Flood Archive** | SAR                 | 2→3              | Visual: SAR imagery · TS: flood extent + streamflow          |       |

### Air Quality & Pollution

| Dataset                | Sensor type | SensorProjection | Modalities                                                          | Notes |
| ---------------------- | ----------- | ---------------- | ------------------------------------------------------------------- | ----- |
| **OpenAQ + Satellite** | GOES-16 ABI | N→3              | Visual: aerosol optical depth (10-15min) · TS: PM2.5, NO2, ozone   |       |

### Healthcare *(data access may be restricted)*

| Dataset                      | Sensor type   | SensorProjection | Modalities                                                     | Notes                         |
| ---------------------------- | ------------- | ---------------- | -------------------------------------------------------------- | ----------------------------- |
| **REVIT Dataset**            | RGB + thermal | Thermal: 1→3     | Visual: patient monitoring video · TS: heart rate, respiration | Single-patient.               |
| **ICU Video-Vitals Dataset** | RGB camera    | None             | Visual: ICU feeds · TS: ECG, PPG, SpO2                         | PhysioNet / MIMIC extensions. |

### Ecology

| Dataset             | Sensor type              | SensorProjection | Modalities                                                         | Notes                                     |
| ------------------- | ------------------------ | ---------------- | ------------------------------------------------------------------ | ----------------------------------------- |
| **EarthNet2021**    | Sentinel-2               | 13→3             | Visual: Sentinel-2 TS · TS: E-OBS weather                          | Dual-listed with Meteorology.             |
| **GreenEarthNet**   | Sentinel-2               | 13→3             | Visual: vegetation indices · TS: meteorological                    |                                           |
| **Digital Typhoon** | Geostationary IR/visible | IR→3             | Visual: decades of typhoon imagery · TS: pressure, wind speed      | National Institute of Informatics, Japan. |

### Port Monitoring

| Dataset                        | Sensor type | SensorProjection | Modalities                                                   | Notes                       |
| ------------------------------ | ----------- | ---------------- | ------------------------------------------------------------ | --------------------------- |
| **Sentinel-1 SAR Maritime**    | SAR         | 2→3              | Visual: SAR imagery · TS: AIS coordinates                    | Cloud/darkness penetrating. |
| **Global Fishing Watch (GFW)** | VIIRS + SAR | N→3              | Visual: nighttime lights + SAR · TS: AIS + fishing activity  |                             |
