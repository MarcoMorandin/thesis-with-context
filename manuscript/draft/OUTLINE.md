# Thesis outline — Multimodal Foundation Models for Cross-Plant PV Power Forecasting

**Author**: Marco Morandin · **Supervisor**: Elisa Ricci · **Co-supervisor**: Francesco Gentile
**Programme**: MSc Artificial Intelligence Systems, DISI — University of Trento · AY 2025/2026
**Status**: draft index. Not thesis prose. Each section lists *what goes in* and *where the
content of record lives*.

---

## Rules this outline obeys

1. **Every number traces to `baselines/results/ALL_RESULTS.md`** (generated) or
   `report/BASELINE_TEST_REPORT.md` (interpretation). Nothing is retyped from memory
   (`AGENTS.md` §5).
2. **In-process MMTSFM numbers are the record.** Checkpoints are known not to reproduce
   their scores post-hoc — no re-scored number may enter the manuscript
   (`report/BASELINE_TEST_REPORT.md` §3.5).
3. **AI framing, not PV engineering.** The contribution is multimodal foundation-model
   fusion + cross-plant generalization; PV is the testbed (`knowledge/scope.md`).
4. **Negative results are reported as results.** Vision lift ≈ 0 and Grassmann ≈
   TimeSelfAttention on `uk_pv` are findings, not failures — the thesis is stronger for
   stating them with the controls that establish them.

**Page budget** — the template allows ~70 pages (ToC + abstract + chapters; title page,
acknowledgements and attachments excluded).

| Ch | Title | Pages |
|---|---|---|
| — | Abstract | 1 |
| 1 | Introduction | 7 |
| 2 | Background and Related Work | 11 |
| 3 | Dataset Construction | 10 |
| 4 | Evaluation Protocol | 6 |
| 5 | Baseline Suite and Leaderboard | 12 |
| 6 | MMTSFM — Method | 12 |
| 7 | Results and Ablations | 9 |
| 8 | Conclusions | 3 |
| — | Appendices (excluded from count) | — |

---

## Abstract

One paragraph, in this order: cross-plant PV forecasting as a zero-shot generalization
problem → a 28-baseline benchmark on 98 UK plants under one fairness contract → MMTSFM
(frozen Chronos-2 + frozen V-JEPA 2.1, selective temporal interleaving, O(L) Grassmann
mixing) → headline numbers (MMTSFM SS 0.343 vs Chronos-2 FT 0.331; ceiling set by
iTransformer 0.552 and Time-VLM 0.540) → the two negative findings and what they imply for
multimodal foundation-model design.

`manuscript/abstract.tex` — currently lorem ipsum.

---

## Chapter 1 — Introduction

*Goal: the reader knows what the problem is, why foundation models are the natural attack,
what was actually done, and what came out — in seven pages.*

### 1.1 Motivation — forecasting a fleet you have never seen
- PV power forecasting framed as an AI problem: short-horizon (6 h) prediction of a
  weather-driven, non-stationary signal, with a hard deployment constraint — a *new* plant
  arrives with two weeks of history and no training data of its own.
- Why per-plant supervised models do not answer this: they require a training set per
  plant. Zero-shot cross-plant transfer is the deployment-realistic setting.
- Source: `knowledge/scope.md`, `knowledge/proposal.md` §Problem Setup.

### 1.2 Why foundation models, and why multimodal
- Time-series FMs (Chronos-2, TimesFM 2.5, TiRex) promise zero-shot transfer; vision FMs
  (V-JEPA 2.1) see cloud advection, the physical driver of ramps. The claim under test is
  that fusing them **deeply, at token level** beats late fusion and beats domain-specific
  PV architectures.
- State the falsifiable claim verbatim from `knowledge/baselines.md`:
  > A frozen multimodal foundation model stack (Chronos-2 + V-JEPA 2.1) with deep
  > token-level fusion achieves cross-plant PV power forecasting on disjoint test plants,
  > beating late fusion, unimodal FMs, and domain-specific architectures.

### 1.3 Research question and hypothesis ladder
- RQ verbatim from `knowledge/scope.md`.
- H0–H4 ladder (FM baseline → late fusion → interleaving → cross-plant zero-shot → RAG),
  each mapped forward to the chapter that tests it (table).

### 1.4 Contributions
1. **A dataset of record for cross-plant multimodal PV forecasting** — 1.34 M rows, 110
   sites, two regions (UK/US), power + 14 covariates + co-registered satellite frames in a
   single contract (Ch. 3).
2. **A 28-model benchmark under one fairness contract** — Tiers 0–6, identical windows,
   identical splits, identical metrics, disjoint test plants (Ch. 4–5). This is the largest
   single-protocol comparison of TSFMs, retrieval adaptation, and multimodal forecasters on
   PV that this work is aware of.
3. **MMTSFM**, a frozen-backbone multimodal architecture with two novel components:
   *selective temporal interleaving* (visual tokens woven only into the refinement window,
   ~2 % token overhead) and *causal Grassmann mixing* (O(L) transition geometry replacing
   O(L²) attention) (Ch. 6).
4. **Two falsifying negative results** with the controls that establish them: on `uk_pv`,
   V-JEPA fusion adds ≈ 0 skill over the identical vision-free arm, and Grassmann mixing
   ties plain TimeSelfAttention (Ch. 7).
5. **A diagnosis of where the headroom actually is** — the oracle-covariate gap (0.173 SS)
   and the retrieval-over-fine-tuning result (0.478 vs 0.331) (Ch. 7–8).

### 1.5 Scope and non-goals
Explicit exclusions, so a reader does not look for them: no irradiance/CSI physics inside
models, no few-shot in-context adaptation curves (design decision A06), no grid-operations
or market analysis, no pre-2025 methods as primary comparators. Source:
`knowledge/scope.md` "Out of scope", `knowledge/baselines.md` §2 "Deliberately excluded".

### 1.6 Thesis structure
One paragraph per chapter.

---

## Chapter 2 — Background and Related Work

*Goal: position the work in 2025–2026 literature. Corpus lives in `knowledge/papers/`
(`baselines/`, `related/`) — query via Graphify, do not read the PDFs.*

### 2.1 Problem formalization
Notation fixed once and used everywhere after: entities \(i\), history \(Y \in
\mathbb{R}^{N\times T\times 1}\), covariates \(X_{cov} \in \mathbb{R}^{N\times(T+H)\times
C}\), frames \(V \in \mathbb{R}^{N\times T_v\times C\times H_{img}\times W_{img}}\), target
\(\hat Y_{fut}\in\mathbb{R}^{N\times H\times Q}\). Source: `knowledge/dataset.md` §4,
`knowledge/proposal.md` §Problem Setup.

### 2.2 Time-series foundation models
Chronos-2 (arXiv:2510.15821) as the backbone of record; TimesFM 2.5, TiRex
(arXiv:2505.23719), TTM-R2 as independent families. Patch tokenization, in-context
covariates, quantile heads. Why "≥3 distinct TSFM families zero-shot" is the review bar.

### 2.3 Adapting frozen backbones
- **Retrieval**: TS-RAG (arXiv:2503.07649), Cross-RAG (arXiv:2603.14709).
- **Covariate adapters**: CoRA (arXiv:2510.12681) — the closest published competitor; if
  MMTSFM ≤ CoRA-with-image-features the token-fusion claim dies.
- **Memory**: MEMTS / TS-Memory (positioning only, not run).

### 2.4 Vision foundation models for spatiotemporal signal
V-JEPA 2.1 ViT-L/16: predictive (not reconstructive) self-supervision, native temporal
modelling, robustness under frozen use. Contrast with VidTok/VQ-VAE reconstruction
objectives — argument in `knowledge/architecture.md` §2.4.

### 2.5 Multimodal time-series forecasting
Split the literature the way the leaderboard does, because the distinction turns out to
matter empirically (Ch. 7):
- **Endogenous multimodal** — the second modality is synthesized from the series itself:
  Time-VLM (arXiv:2502.04395), VisionTS++ (arXiv:2508.04379), Aurora (arXiv:2509.22295).
- **Exogenous multimodal** — genuine external sensors: UniCast (arXiv:2508.11954),
  CrossViVit (arXiv:2306.01112), Solar-VLM (arXiv:2604.04145), SUNSET.

### 2.6 PV-specific forecasting and its benchmarks
Smart persistence and the solar skill score as community convention; SUNSET, Solar-VLM,
PV-VLM, FusionSF/M3S-Net. What each measures and why their protocols are not comparable
across papers — the motivation for Ch. 4.

### 2.7 Related multimodal PV datasets, and why a new one
Comparative table: dataset of record vs **ClimateHackAI 2023** (5-min nowcasting, 1 h→4 h
windows, ~600 GB, no published paper) vs **MMSP/FusionSF** (88 Chinese plants, hourly,
24 h→24 h day-ahead, Himawari-8/9). Both are short-window / fixed-task corpora that would
need re-windowing before serving a 14-day-history cross-plant protocol. Source:
`report/REPORT.md` §1.4–1.5.

### 2.8 Gap statement
No existing benchmark evaluates TSFMs, frozen-backbone adaptation, and multimodal
forecasters on *one* cross-plant protocol with *real* satellite frames. That gap is what
Ch. 3–5 build.

---

## Chapter 3 — Dataset Construction

*Goal: a reader could rebuild it. Canonical: `knowledge/dataset.md` (contract) +
`report/REPORT.md` (construction and EDA).*

### 3.1 Design requirements
One row per `(site_id, timestamp_utc)`; power, weather covariates and a co-registered
satellite frame on the same clock; native cadence preserved (no resampling, no gap
interpolation); plant identity as the split key; frames addressable in O(1) without
millions of small files.

### 3.2 Sources fused
| Track | Source | What it gives |
|---|---|---|
| PV power, UK | `openclimatefix/uk_pv` (HF) | 30-min generation 2019–2020, per-site kWp + rounded lat/lon |
| Imagery, UK | EUMETSAT SEVIRI RSS **HRV** (OCF public GCS bucket) | ~1 km/px; reprojected per site, cropped 128×128 (~128 km) |
| PV power + imagery, US | NREL **PVDAQ** (OEDI data lake) + **GOES-16** crops | 10 sites, 15-min, 256×256 RGB |
| Weather | **Open-Meteo** Archive API | 8 hourly variables joined by `merge_asof` |

### 3.3 Harmonization pipeline
`generation_Wh`→W (×2); gap policy (≤3 steps linearly interpolated, else dropped);
capacity audit → `installed_power_w`; target `norm_power = power / installed_power_w ∈
[0,1]`; solar-geometry and clear-sky derivations (Haurwitz `clearsky_ghi`, `kt`, `csi`,
`doy_sin/cos`, `solar_time`); frame ops (NaN→0, clip [0,1], drop empty crops).

### 3.4 Storage layout and the frame pointer
`dataset_all.parquet` (1,337,654 × 35) + `images_all.h5` (27 GB, 110 per-site groups
`<dataset>_<site>` with `images` + `timestamps`). The canonical `image_h5_index` is
*local-to-group* and timestamp-exact for both datasets — worth its own paragraph, since the
dead `image_uk128_index` column is a live trap for anyone re-using the data.

### 3.5 Quality flags and curation
`bad_site_flag` (`uk_pv` 7239, 8587; `goes_pvdaq` 1283, 51), `outage_flag` (15,486),
`stuck_flag` (1,318), `night_clamped` (1,535). State the known inconsistency honestly: the
committed `goes_pvdaq` split predates the bad-site flags and must be regenerated before that
track is run.

### 3.6 Dataset statistics
| | Sites | Rows | Cadence | Span (UTC) | Frames | Capacity |
|---|---|---|---|---|---|---|
| `uk_pv` | 100 | 1,232,862 | 30 min | 2019-01-01 → 2020-12-31 | (N,128,128) uint8 gray | 1.5–4.0 kW residential |
| `goes_pvdaq` | 10 | 104,792 | 15 min | 2019-01-01 → 2019-09-30 | (N,256,256,3) uint8 RGB | 1.8–408 kW residential→utility |

Figures to lift from `report/REPORT.pdf`: diurnal/seasonal power profiles, capacity
distribution, missingness heatmap, sample satellite crops under clear vs ramp conditions.

### 3.7 Splits
Cross-plant, seed 42, per-dataset 70/15/15, `bad_site_flag` sites excluded, committed to
`baselines/configs/splits.json`, disjointness asserted at every load
(`baselines/common/splits.py`). `uk_pv`: 69 train / 15 val / 14 test plants (850,654 /
184,899 / 172,656 rows). Full `site_id` membership → Appendix A.
**Argue the choice**: random plant assignment can leak spatial information between nearby
plants; a distance/region-based split is the stronger variant and is named as future work.

### 3.8 The canonical batch dict
The `PVTSFMDataset` output contract (`Y`, `Y_future`, `X_cov`, `V`, masks,
`mask_modality_dropout`, `adj_matrix`, timestamps) with shapes — reproduce
`knowledge/dataset.md` §4 verbatim. Every model in Ch. 5–6 consumes exactly this dict; that
is what makes the comparison fair.

---

## Chapter 4 — Evaluation Protocol

*Short, dense, and load-bearing: the credibility of Ch. 5 and 7 rests here. Canonical:
`knowledge/protocol.md`; scenario design `knowledge/baselines.md` §3–4.*

### 4.1 Three fairness principles
Same horizon and granularity for all; disjoint test plants with no statistic leakage
(including normalizers); no domain-physics heuristics inside models (Smart Persistence is
exempt — it *is* the physics reference).

### 4.2 Windows in physical time, not steps
History \(T\) = **14 days** (672 steps `uk_pv` / 1344 `goes_pvdaq`); horizon \(H\) = **6 h**
(12 / 24 steps); visual \(T_v\) = **8 frames** over a short recent window, decoupled from the
TS context. Why physical time: cadences differ, so step-horizons are not the same task —
hence the rule that cross-dataset aggregation uses only scale-free statistics.
Why decoupled visual resolution: vision carries signal only over the cloud-advection
horizon, so widening it adds cost, not signal.

### 4.3 Metrics
Capacity-normalized NMAE and NRMSE, computed per plant then macro-averaged; **Skill Score
SS = 1 − NRMSE_model / NRMSE_smart_persistence** as the headline; CRPS / pinball for
quantile-native models; R² as a variance-tracking diagnostic; ramp-subset NMAE/NRMSE
(top-decile |ΔY|). Explain *why SS and R² must be read together* — high R² with low SS is a
bias/scale problem, not a timing problem (the visionts_pp / solar_vlm / sunset pattern).

### 4.4 Scale-free aggregation
Win rate, geometric-mean skill, average rank — with the rule that raw step-horizon metrics
are never pooled across cadences.

### 4.5 Scenario battery
S1 in-domain · **S2 cross-plant (primary)** · S3 cross-dataset · S4 long-horizon
(skill-decay at 1/6/24 h) · S5 data efficiency · S6 ramp subset · S7 seasonal transfer.
State plainly which were executed: **everything reported in this thesis is S2 on `uk_pv`,
plus the S6 ramp subset**; S3/S5 are the named limitation of Ch. 8.

### 4.6 Input-parity matrix
The tier × input table (`Y` / `X_cov` / `V` / retrieval / text) and the two rules reviewers
attack first: the **retrieval datastore rule** (train plants only; transductive retrieval is
a separately labelled condition) and the **oracle-covariate rule** (`chronos2_oracle*` sees
future observed weather → upper bound, not a competitor).

### 4.7 Reproducibility
Seed 42 everywhere; Hydra-only configuration; self-contained per-baseline configs; SLURM
execution on Leonardo with logs retained; `uv` dependency management.

---

## Chapter 5 — Baseline Suite and Leaderboard

*Goal: establish the bar MMTSFM must clear, and show that every rebuttal cell is covered.
Canonical: `knowledge/baselines.md` (design) · `baselines/results/ALL_RESULTS.md` (numbers)
· `report/BASELINE_TEST_REPORT.md` (reading).*

### 5.1 Suite design — one baseline per falsification
The tier table, each row annotated with *the reviewer question it answers*: simpler model?
(DLinear) · just covariates? (CoRA) · just retrieval? (TS-RAG) · backbone-specific?
(TimesFM/TiRex) · vision actually used? (shuffled frames) · domain SOTA?
(Solar-VLM/CrossViVit) · fusion depth? (late vs interleaved).

| Tier | Models |
|---|---|
| T0 reference | persistence, **smart persistence**, hourly climatology, seasonal naive |
| T1 classical | LightGBM (per-quantile), TabPFN-3 |
| T2 supervised DL | MLP, DLinear, PatchTST, iTransformer, TFT |
| T3 TSFM | Chronos-2 ZS/FT (+oracle), TimesFM 2.5 ZS, TiRex ZS, TTM-R2 ZS/FT |
| T4 frozen-FM adaptation | TS-RAG, Cross-RAG, CoRA |
| T5 endogenous multimodal | Time-VLM, Aurora, VisionTS++, UniCast |
| T6 exogenous multimodal (PV) | Solar-VLM, CrossViVit, SUNSET |

### 5.2 Implementation and integration notes
Vendored third-party code, the deviations that matter (TFT-lite; CrossViVit
single-channel/synthetic-coordinate approximation; SUNSET port; TTM-R2 not R3 and *why* —
the R3 checkpoint's keys do not load into `tsfm_public` 0.3.2 and all weights end up
random). These notes are what make the table honest.

### 5.3 Headline leaderboard (S2, `uk_pv`, 14 disjoint test plants)
Full 28-row table cited from `baselines/results/ALL_RESULTS.md`. Anchors:

| Model | SS ↑ | R² ↑ | CRPS ↓ |
|---|---|---|---|
| iTransformer | **0.552** | 0.765 | — |
| Time-VLM | 0.540 | 0.487 | — |
| chronos2_oracle_ft *(upper bound)* | 0.504 | 0.706 | 0.0630 |
| TS-RAG | 0.478 | 0.331 | — |
| Cross-RAG | 0.477 | 0.375 | — |
| Solar-VLM | 0.443 | 0.660 | — |
| LightGBM | 0.384 | 0.555 | 0.0768 |
| CoRA | 0.374 | 0.550 | 0.0816 |
| CrossViVit | 0.349 | 0.565 | — |
| **MMTSFM (ours, flagship)** | **0.343** | — | 0.0805 |
| chronos2_ft | 0.331 | 0.475 | 0.0855 |
| chronos2_zs | 0.187 | 0.277 | 0.1072 |
| smart persistence *(reference)* | 0.000 | 0.210 | — |
| ttm_zs | **−0.081** | 0.155 | — |

### 5.4 Four findings from the baseline suite
1. **A well-tuned supervised transformer is the model to beat** — iTransformer tops every
   metric, ahead of every FM, retrieval scheme and multimodal system.
2. **Zero-shot TSFMs underdeliver on PV** — all four ZS rows trail classical ML; TTM-R2 ZS
   is below the naive floor. PV's sharp weather-driven dynamics are out of distribution for
   generic TS pretraining.
3. **Retrieval beats fine-tuning on a frozen backbone** — 0.478 vs 0.331 for the same
   Chronos-2, with no weight updates. The cheapest adaptation is also the best.
4. **Genuine exogenous imagery has not yet paid off** — the best multimodal model
   (Time-VLM 0.540) synthesizes its images *from the series*; every model consuming real
   satellite frames sits mid-pack. This is the finding that frames Ch. 7.

### 5.5 Qualitative analysis
Forecast traces and actual-vs-predicted scatter on test site 10793
(`baselines/plots/`), with the caveat that per-site numbers in plot titles are not the
pooled 14-plant numbers. The systematic under-prediction of solar_vlm / sunset / crossvivit
is a calibration failure, and calibration is cheaper to fix than a temporal model.

---

## Chapter 6 — MMTSFM: Method

*Canonical: `knowledge/architecture.md` (as built) · `knowledge/proposal.md` (the argument).
Note in the text that the v5 "StateCast" redesign
(`knowledge/specs/2026-07-15-statecast-v5-design.md`) is future work, not what was measured.*

### 6.1 Overview
The full data-flow figure from `knowledge/architecture.md` §1, redrawn: inputs → tokenizers
→ embedding → fusion switch → 12 encoder blocks → quantile head. Reference shapes for
`uk_pv`: `T_ctx = ⌈672/16⌉ = 42`, `n_vis = 1`, `T_fut = ⌈12/16⌉ = 1`, output `[B, 12, 9]`.

### 6.2 Design principle: decoupled resolution
Long macro-numeric context (seasonality, regime structure) + short micro-visual refinement
window at high cadence. Resampling a year of frames to the TS grid is computationally
impossible; resampling TS up to frame cadence destroys macro-context.

### 6.3 Numeric branch — Chronos-2
Native checkpoint `amazon/chronos-2`: d_model 768, 12 layers, 12 heads, d_ff 3072, patch 16,
9 quantiles, arcsinh normalization. **Report the trap**: the YAML `d_model: 512 /
num_layers: 6` values are silently ignored because `from_pretrained` keeps the checkpoint
architecture — a reproducibility footnote worth writing down. `input_patch_size=16` matches
the checkpoint, so the pretrained tokenizer transfers; only the 9-quantile head reinitializes.

### 6.4 Visual branch — V-JEPA 2.1 + latent summarizer
Frozen ViT-L/16, whole 8-frame clip, tubelet with temporal stride 2 → `[B, 4, 196, 1024]`.
A Perceiver-style causal `LatentSummarizer` with a null token does the spatial compression
to `[B, 1, 768]`. Domain specialization lives entirely in the learned summarizer queries
(CLIP-style adapter recycling); there is no per-sensor channel projection.

### 6.5 Multimodal embedding
Additive: modality {num, vis} · segment {ctx, fut} · token-type {target, cov, vis} · entity
· RoPE positions. The 14 covariate channels enter as batch-axis token rows with mask = 1 —
verified live in both fusion modes by a covariate-perturbation regression test.

### 6.6 Fusion — the contribution
- **Late fusion (S2a)**: vision → `CrossModalAdapter` → `N_soft = 1` batch rows.
- **Selective temporal interleaving (S2b/S3)**: visual tokens woven into the TS sequence
  *only inside the refinement window*, so the macro region keeps its pure temporal geometry
  and the mixer computes cross-modal Plücker pairs. Sequence 42 + 1 + 1 = 44 tokens for
  `uk_pv` — **~2 % overhead**, proportional to `n_vis`, not to context length.
- Position against UniCast-style soft prompting and CoRA-style adapters: same design space,
  shallower coupling.

### 6.7 Temporal mixing — causal Grassmann flow
Plücker embedding of the transition between consecutive hidden states as a 2-D subspace on
G(2,32): reduce → RoPE → wedge \(p = z_{i-\delta} \wedge z_i\), offsets δ ∈ {1,2,4,8,12,16},
softmax multi-scale aggregation, gated α, plus 4 modality-pair biases. Three arguments:
geometry over magnitude (direction of state evolution — "rapidly clouding" vs "stable"),
O(L) instead of O(L²), and multi-scale by construction. Trained from scratch.
`TimeSelfAttention` + RoPE is the **matched diagnostic twin**: identical stage schedule,
only the mixer differs, so any delta is attributable to the inductive bias.

### 6.8 Encoder block, group attention, output head
Per block: temporal mix → `GroupSelfAttention` over the batch axis (fuses target + 14
covariate rows + late visual rows sharing a `group_id`; pretrained) → FeedForward.
Non-autoregressive quantile head: the whole horizon in one pass, no error accumulation,
9 quantiles because uncertainty is a first-class protocol requirement.

### 6.9 Training curriculum
Four-stage pretrained-weight recycling, each stage warm-starting from the previous
`best.ckpt` (weights only, `init_ckpt`):

| Stage | Fusion | Vision | Chronos | Purpose |
|---|---|---|---|---|
| S1 | — | off | trainable + 2000-step Grassmann warmup | anchor the from-scratch mixer before it can corrupt the residual stream |
| S2a | late | last-4 unfrozen | frozen | learn the V-JEPA→Chronos mapping against a stable target |
| S2b | interleaved | re-frozen | frozen except mixer | teach the mixer cross-modal geometry |
| S3 | interleaved | progressive unfreeze | all trainable | joint fine-tuning |

Plus: masked pinball loss, asymmetric Bernoulli modality dropout, Lightning + Hydra, DDP +
bf16 on Leonardo.

### 6.10 Engineering realities worth reporting
The vision-free arm OOMs where the fusion arm does not — with the vision stack off, samples
take the standard multivariate path and every covariate channel becomes its own series row
(15 rows/sample → `[30, 86, 768]` at batch 2 vs `[2, 89, 768]` interleaved), a 15× batch-dim
blow-up into group attention; fixed with `batch_size=2` + `accumulate_grad_batches=8`. A
short subsection on this is more useful to a follow-up student than another paragraph of
motivation.

---

## Chapter 7 — Results and Ablations

*Every number in-process and cross-plant (S2, `uk_pv`, 14 disjoint test plants, 165,295
scored steps). Canonical: `report/BASELINE_TEST_REPORT.md` §1, §3.5.*

### 7.1 Main result
| Arm | Vision | Mixer | Fusion | NMAE ↓ | NRMSE ↓ | **SS ↑** | CRPS ↓ | cov@80 |
|---|---|---|---|---|---|---|---|---|
| `mmtsfm_numeric_grassmann` | off | Grassmann | — | 0.1069 | 0.1510 | **0.3446** | 0.0810 | 0.804 |
| `mmtsfm_selfattn_late` | on | SelfAttn | late | 0.1076 | 0.1513 | **0.3432** | 0.0810 | — |
| `mmtsfm_grassmann_interleaved` *(flagship)* | on | Grassmann | interleaved | 0.1059 | 0.1514 | **0.3429** | 0.0805 | 0.793 |
| `mmtsfm_grassmann_no_modbias` | on | Grassmann | interleaved | 0.1066 | 0.1527 | **0.337** | 0.0812 | 0.756 |

Reading: all four arms beat `chronos2_ft` (0.331) with the same backbone, so the training
recipe transfers ≈ 0.013 SS. All four trail the RAG pair (≈0.477) and the 0.55 bar.

### 7.2 Ablation A — does vision contribute? *(H1)*
`numeric_grassmann` is the vision-lift **lower bound** and it *matches or beats* every
fusion variant. On `uk_pv` the V-JEPA stream contributes ≈ zero measurable skill. Support
with the per-horizon NMAE profile: identical across arms — strong ≤2 h (0.067–0.092), jump
from step 6 (0.127–0.155). Whatever vision is supposed to buy, it is not arriving at the
horizons where the error lives.

### 7.3 Ablation B — does Grassmann mixing beat attention? *(A03)*
`grassmann_interleaved` 0.3429 vs `selfattn_late` 0.3432 — a statistical tie under a matched
stage schedule. The geometric inductive bias buys nothing over plain self-attention *at this
context length*; report the O(L) VRAM/latency advantage separately, since that is where the
argument survives (a scaling curve over T ∈ {128, 512, 1024, 2048} is the right figure).

### 7.4 Ablation C — modality-pair bias
Removing it costs ~0.006 SS (0.3429 → 0.337) and, more informatively, **calibration**:
coverage@80 drops 0.793 → 0.756. The bias helps the predictive distribution more than the
point forecast.

### 7.5 Fusion depth — late vs interleaved *(H2)*
A01 vs A02, same backbones, same data: no separation. State the honest conclusion —
H2 is **not supported on `uk_pv`**.

### 7.6 Where the headroom actually is
- **Oracle gap**: `chronos2_oracle_ft` 0.504 − `chronos2_ft` 0.331 = **0.173 SS** available
  from better conditioning/context alone. This is larger than anything fusion has produced,
  and it points the next iteration at conditioning rather than at architecture.
- **Retrieval gap**: RAG ≈ 0.477 with a frozen backbone vs 0.331 fine-tuned.
- **Endogenous vs exogenous imagery**: Time-VLM 0.540 with pseudo-images and the best ramp
  NRMSE in the suite (0.172) proves visual *inductive bias* converts; genuine satellite
  frames, as currently encoded, do not. Formulate the hypothesis this leaves open: the
  bottleneck is the V-JEPA→token interface, not the availability of visual information.

### 7.7 Threats to validity
1. **Single dataset** — everything is `uk_pv`; cross-dataset generalization to
   `goes_pvdaq` (S3) is *unproven*, not disproven.
2. **Checkpoint integrity (open)** — saved MMTSFM checkpoints do not reproduce their
   training-time scores in a fresh process (interleaved epoch-23 claims val 2.8435 → fresh
   val 4.08 / SS 0.102; affects both architectures). Weight restore was verified
   tensor-faithful across 236 keys, configs bit-identical, data deterministic; the cause is
   unresolved. **Consequence stated in the thesis**: in-process numbers are the record and
   post-hoc re-scoring is blocked. Document the eliminations — a reader must be able to see
   this was investigated, not ignored.
3. **Two harness-limited baseline rows** — Time-VLM runs on non-aligned eval windows;
   UniCast's weakness is amplitude under-prediction (`pred ≈ 0.61·true + 0.063`) plus a
   late-day window skew.
4. **Ramp subsets are not bit-aligned across tiers** (T4–T6 re-scored from saved
   predictions) → read ramp within-tier or by rank.
5. **Oracle rows are upper bounds**, never competitors.

### 7.8 Controls that were designed but not run
Be explicit rather than silent: shuffled-frames (A09), mismatched-plant frames (A10),
vision-only upper bound (A11), the 4-way modality grid (A12), visual-token budget (A13),
frozen vs partial unfreeze (A14), RAG datastore sweep (A15). Say what each would have
established and why the vision-lift-≈0 result makes A09/A12 the highest-value next runs.

---

## Chapter 8 — Conclusions

### 8.1 Summary of contributions
Restate the five contributions of §1.4 with the evidence attached to each.

### 8.2 Answering the research question
Answer it *as measured*, without softening: on cross-plant `uk_pv`, deep token-level fusion
of frozen Chronos-2 + V-JEPA 2.1 **does not** outperform late fusion, and does not beat
either supervised in-domain transformers or retrieval-augmented frozen backbones. H1 and H2
are unsupported at the measured scale; H0 and H3 are established (FMs need adaptation;
cross-plant zero-shot evaluation is feasible and discriminative); H4 is supported —
retrieval is the strongest frozen-backbone lever tested.

### 8.3 What the negative results teach
The value of the vision-free control: without it, a 0.343 flagship reads as a success over
`chronos2_ft`. Time-VLM's result separates *visual inductive bias* from *visual
information* — the former converts on this task, the latter does not through the current
encoder interface. A general lesson for multimodal FM work: report the modality-off arm
under the identical recipe, or the ablation is not an ablation.

### 8.4 Limitations
Single dataset · single seed per arm · no significance testing (Diebold–Mariano) · open
checkpoint-integrity bug · half the control battery unexecuted · `goes_pvdaq` split needs
regeneration against the bad-site flags.

### 8.5 Future work
1. Run the missing controls first — A09 shuffled frames and A12 modality grid decide whether
   the vision path is dead or merely mis-interfaced.
2. Attack conditioning, not architecture: the 0.173 SS oracle gap is the biggest measured
   headroom.
3. Combine retrieval with fusion (RAG over frozen Chronos-2 + visual tokens) — the two best
   levers are currently untested together.
4. Cross-dataset S3 (`uk_pv` → `goes_pvdaq`) and region-based rather than random plant
   splits.
5. The v5 StateCast redesign as the structured alternative to token interleaving.
6. Close the checkpoint-integrity bug via activation-fingerprint bisection (dump per-module
   outputs on a fixed batch at save time vs after reload).

---

## Appendices *(excluded from the 70-page count)*

- **A** — Exact plant membership of the `uk_pv` train/val/test splits (`splits.json`).
- **B** — Full 28-row leaderboard with all metrics, plus per-tag scale-free aggregation
  tables (win rate / SS_geo / avg rank).
- **C** — Hydra configuration of record per stage (`s1`, `s2a`, `s2b`, `s3`) and the
  headline model config.
- **D** — Baseline integration notes and deviations from published implementations.
- **E** — Ablation registry (`knowledge/ablations.md`) with status and job IDs.
- **F** — Compute budget: SLURM jobs, GPU-hours, Leonardo account `IscrC_MTSFM`.

---

## Mapping to the LaTeX skeleton

`manuscript/` currently has `chapter1–3.tex` (lorem ipsum). Target layout:

| File | Chapter |
|---|---|
| `abstract.tex` | Abstract |
| `chapter1.tex` | 1 Introduction |
| `chapter2.tex` | 2 Background and Related Work |
| `chapter3.tex` | 3 Dataset Construction |
| `chapter4.tex` *(new)* | 4 Evaluation Protocol |
| `chapter5.tex` *(new)* | 5 Baseline Suite and Leaderboard |
| `chapter6.tex` *(new)* | 6 MMTSFM — Method |
| `chapter7.tex` *(new)* | 7 Results and Ablations |
| `chapter8.tex` *(new)* | 8 Conclusions |
| `attachments.tex` | Appendices A–F |

Also to update: `front.tex` title/subtitle (still `Title` / `Sub-title (optional)`), and
`biblio.bib` — currently the template's placeholder entries, needs the ~30 arXiv keys cited
above.

## Open decisions

1. **Chapter 2 vs Chapter 5 split** — related work can either live entirely in Ch. 2 or be
   distributed as per-baseline descriptions in Ch. 5. Recommendation: keep positioning in
   Ch. 2, keep implementation/deviation notes in Ch. 5, and do not repeat a paper's
   description in both.
2. **Framing of the flagship result** — "MMTSFM beats `chronos2_ft`" (true, +0.013 SS) vs
   "fusion adds nothing, and here is the control that proves it" (also true, more
   defensible). Recommendation: the second; the first invites exactly the question the
   vision-free arm already answers.
3. **Whether to run A09/A12 before submission** — they are cheap relative to their value and
   would convert §7.2 from an observation into a demonstration.
