# Architecture — MMTSFM as implemented, and why

Single source of truth for **what the model is** and **why each piece is there**.

- How to *run* it → [runbook.md](runbook.md)
- The long-form scientific argument → [proposal.md](proposal.md)
- Newer AI-first reframing (v5, not implemented) → [specs/2026-07-15-statecast-v5-design.md](specs/2026-07-15-statecast-v5-design.md)
- Code navigation → GitNexus (`query`, `context`, `impact`), not this file

---

## 1. What is actually built

Backbone = **Chronos-2** at its native checkpoint size (`amazon/chronos-2`): **d_model 768,
12 layers, 12 heads (d_kv 64), d_ff 3072, patch 16, 9 protocol quantiles, arcsinh norm**.

> ⚠️ The YAML `d_model: 512 / num_layers: 6 / d_ff: 2048` are **silently ignored** —
> `from_pretrained` keeps the checkpoint's architecture. Only the grassmann fields and
> `chronos_config` propagate. Any doc or config claiming 512/6 is wrong.

Vision = **V-JEPA 2.1 ViT-L/16**, spatiotemporal, frozen → progressively unfrozen.

Reference shapes (`uk_pv`, batch `B`, one entity per row):
`T_ctx = ceil(672/16) = 42` · `n_vis = 1` (goes_pvdaq: 2) · `T_fut = ceil(12/16) = 1`.

```text
                          INPUTS  (per entity row)
  Y hist [B,672]      X_cov [B,672+12,14]      V frames [B,3,8,224,224]
  PV power            14 protocol covariates    8 RGB sky/sat frames, 6h window
       │                    │  (future 12 steps)          │
       │ arcsinh + patch16  │ (each channel → its own     │ whole clip → V-JEPA 2.1
       ▼                    │  covariate token-row)       ▼ (tubelet, temporal stride 2)
  input_patch_embedding ◄───┘                      [B, 4, 196, 1024]   (T_lat,P,D_v)
  (PRETRAINED — transfers)                                │  ViT-L/16: 4 time × 196 patches
       │                                                  ▼
  context [B,42,768]                              LatentSummarizer  (Perceiver, causal,
  future  [B, 1,768]                              spatial compress, null token)
  14 cov rows [B,1,768] each ──┐                         │
                               │                  visual summary [B,1,768]  (n_vis, uk_pv)
                               │                         │
                               ▼                         ▼
        MultimodalEmbedding (additive): modality{num=0,vis=1} · segment{ctx,fut}
        · token-type{target,cov,vis} · entity · RoPE positions
                               │
              ┌────────────────┴───────────────────────────────┐
              │  FUSION  (VisionChronos2Config.fusion_mode)     │
              ├──────────────────────┬──────────────────────────┤
              │  late (S2a)          │  interleaved (S2b/S3)     │
              │  vis → CrossModal-   │  weave vis into refinement│
              │  Adapter → N_soft=1  │  window only:             │
              │  batch rows          │  [ts … ts, ts, v, fut]    │
              │                      │  seq = 42+n_vis+1 (uk_pv  │
              │                      │  n_vis=1 → 44)            │
              └──────────┬───────────┴─────────────┬─────────────┘
                         │                          │
   sequence per row:  [B, T_ctx(+n_vis) + T_fut, 768]
   + covariate rows + (late) visual rows on the BATCH axis, sharing group_id
                         │
                         ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Chronos2Encoder  ×12 blocks                                          │
   │                                                                       │
   │   1. TEMPORAL MIX  (choose one — this is the variant switch)          │
   │      ├─ CausalGrassmannMixing   [vision_chronos2_grassmann]           │
   │      │    O(L) Plücker: reduce→RoPE→wedge p=z_{i-δ}∧z_i on G(2,32),    │
   │      │    offsets {1,2,4,8,12,16}, softmax multi-scale, gated α,       │
   │      │    +4 modality-pair biases. FROM SCRATCH.                      │
   │      └─ TimeSelfAttention + RoPE [vision_chronos2_timeselfattn]       │
   │           O(L²) causal self-attention over the time axis.             │
   │                                                                       │
   │   2. GroupSelfAttention  — BATCH axis; fuses target + 14 covariate    │
   │      rows (+ late visual rows) across rows sharing group_id. PRETRAINED│
   │                                                                       │
   │   3. FeedForward  (LN → 768→3072 → ReLU → 3072→768, residual). PRETR. │
   └─────────────────────────────────────────────────────────────────────┘
                         │  last T_fut hidden states (target rows only)
                         ▼
   output_patch_embedding → quantiles×patch  →  reshape → arcsinh⁻¹
                         │
                         ▼
              Ŷ  [B, 12, 9]   (horizon × 9 quantiles)
                         │
                         ▼
              Masked pinball loss   →   NMAE / NRMSE / Skill-Score
```

### Component map (file → role)

| Component | File (under `MMTSFM/src/mmtsfm/`) | Notes |
|-----------|------|-------|
| Assembly, fusion routing, interleave | `models/chronos2/vision_chronos2.py` | `VisionChronos2Model` |
| Encoder stack, GroupSelfAttention, FFN | `models/chronos2/model.py` | native Chronos-2, 12 blocks |
| Grassmann mixer | `models/chronos2/grassmann.py` | `CausalGrassmannMixing`, O(L) |
| TimeSelfAttention + RoPE | `models/chronos2/layers.py` | O(L²) diagnostic mixer |
| V-JEPA 2.1 wrapper | `models/vision/visual_encoder.py` | ViT-L/16, torch.hub |
| Spatial summarizer | `models/vision/latent_summarizer.py` | Perceiver, causal, null token |
| Late-fusion soft tokens | `models/vision/cross_modal_adapter.py` | `N_soft=1` batch rows |
| Training loop, freeze/warmup/unfreeze | `models/chronos2/lightning_module.py` | per-stage policy |
| Hydra entry, warm-start / resume | `train.py` | `init_ckpt` vs `ckpt_path` |

---

## 2. Design rationale

Why each choice was made. Claims here are design *intent*; measured outcomes live in
[../report/BASELINE_TEST_REPORT.md](../report/BASELINE_TEST_REPORT.md).

### 2.1 Decoupled resolution

Numeric and visual streams keep **separate temporal regimes**: a long macro-numeric context
(seasonality, trend, regime structure) and a short micro-visual refinement window at high
cadence. Resampling a year of frames to the TS grid is computationally impossible;
resampling TS up to frame cadence dilutes macro-context. Decoupling keeps both modalities
at their native, most informative resolution.

### 2.2 Causal Grassmann mixing — O(L) instead of O(L²)

Instead of dot-product similarity, the mixer encodes the **transition** between consecutive
hidden states as a 2-D subspace on a Grassmann manifold (Plücker embedding).

- **Geometry over magnitude** — captures the *direction* of state evolution. In physical
  systems the direction of change ("rapidly clouding" vs "stable") is often more predictive
  than the raw level.
- **Efficiency** — O(L) admits `T = 1000+` context on one GPU; standard attention hits VRAM.
- **Multi-scale** — aggregates transitions across offsets `δ ∈ {1,2,4,8,12,16}`, tracking
  several temporal frequencies at once.

`TimeSelfAttention` is retained as the matched diagnostic twin: identical stage schedule,
only the mixer differs, so any delta is attributable to the inductive bias.

### 2.3 Selective temporal interleaving

Visual tokens are woven into the TS sequence **only inside the refinement window**. The
macro region stays pure TS, preserving its temporal geometry; in the refinement region
tokens alternate `[ts_k, vis_k, ts_{k+1}, …]` so the mixer computes **cross-modal Plücker
pairs** — literally the geometric angle between a numeric state and a visual observation.

Overhead is proportional to `n_vis`, not to context length: 42 + 1 + 1 = 44 tokens for
`uk_pv`, ~2 % — deep fusion at a fraction of full multimodal attention.

### 2.4 Frozen V-JEPA 2.1 + learned summarizer

V-JEPA over VidTok (the earlier candidate):

- **Predictive, not reconstructive** — VidTok's VQ-VAE objective spends capacity on
  pixel-level noise; V-JEPA's predictive self-supervision yields spatially structured,
  semantically coherent features, which is what cloud-formation detection needs.
- **Native temporal modelling** — V-JEPA consumes the whole 8-frame clip and encodes motion
  internally (temporal stride 2 → 4 latent steps), so the `LatentSummarizer` only has to do
  spatial compression.
- **Built for frozen use** — robust under linear probing / adapter tuning, which is the
  cornerstone of the staged curriculum.

Domain specialization is carried entirely by the learned `LatentSummarizer` queries
(CLIP-style adapter recycling). **No per-sensor channel projection** — the `.h5` loader
already delivers uniform 3-channel frames for every plant and source. (An earlier revision
described a learned multi-sensor `1×1` projection for a "sensor zoo"; that framing is
retired along with the general multi-domain scope.)

### 2.5 Non-autoregressive quantile head

The whole horizon `H` is predicted in one forward pass — no error accumulation where a wrong
step at `t+1` poisons `t+10`. Output is 9 quantiles, not a point estimate, because
uncertainty is a first-class requirement of the evaluation protocol
([protocol.md §5](protocol.md)).

### 2.6 Training curriculum — pretrained-weight recycling

Four stages merge Chronos-2 (numeric) and V-JEPA (visual) without letting randomly
initialized modules corrupt pretrained residual streams.

| Stage | Fusion | Vision | Chronos | Purpose |
|-------|--------|--------|---------|---------|
| **S1** | — (vision skipped) | off | trainable + Grassmann warmup (2000 steps) | TS pretraining; anchor the mixer before it can corrupt the residual stream |
| **S2a** | late | V-JEPA last-4 unfrozen | frozen | learn the V-JEPA→Chronos mapping against a stable late-fusion target |
| **S2b** | interleaved | V-JEPA re-frozen | frozen except mixer | teach the mixer cross-modal (TS↔visual) geometry |
| **S3** | interleaved | progressive unfreeze | all trainable | full joint fine-tuning |

Stage overrides live in `MMTSFM/configs/stage/{s1,s2a,s2b,s3}.yaml`. Each stage warm-starts
(weights only, via `init_ckpt`) from the previous stage's `best.ckpt`.

### 2.7 Stack: Lightning + Hydra

- **PyTorch Lightning** decouples the training lifecycle from the architecture and handles
  DDP on Leonardo, bf16, and checkpointing (`VisionChronos2LightningModule`,
  `MMTSFMDataModule`).
- **Hydra** composes `model/` · `data/` · `trainer/` · `stage/` config groups. All
  hyperparameters live in `MMTSFM/configs/` — never hardcoded in model code.

---

## 3. Verified behaviour

Facts confirmed by test or run log, not by design intent:

- **Pretrained tokenizer transfers** — `input_patch_size=16` matches the checkpoint, so
  `input_patch_embedding` loads pretrained; only the 9-quantile output head reinitializes.
- **Known future covariates are live** — the 14 covariate channels enter as batch-axis token
  rows (mask=1 so values survive) and reach the encoder in **both** fusion modes; perturbing
  a covariate changes the forecast (regression-tested).
- **Variant switch is clean** — `MODEL_CFG` alone selects Grassmann vs TimeSelfAttention; the
  stage schedule is identical.
- **V-JEPA is a video encoder**, not a per-frame image encoder.

## 4. Open issues

- **Checkpoint integrity** — checkpoints do not reproduce their in-process scores when
  re-scored fresh. In-process numbers are the record; there is no post-hoc re-scoring path.
- **Measured lift** — on `uk_pv`, both the vision and the Grassmann arms show ≈ 0 lift over
  the TS-only arm. See [../report/BASELINE_TEST_REPORT.md](../report/BASELINE_TEST_REPORT.md).
