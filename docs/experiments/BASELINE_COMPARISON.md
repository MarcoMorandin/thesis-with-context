# Baseline Comparison, Tests & Evaluation Protocols — PVTSFM

**Status**: Living document. Companion to [BASELINE_PROTOCOL.md](BASELINE_PROTOCOL.md) (fairness rules) and [ABLATION_REGISTRY.md](ABLATION_REGISTRY.md) (run tracking).
**Goal**: Define the complete baseline suite, the comparison matrix, the test battery, and the evaluation protocols required to position PVTSFM as a top-tier contribution (NeurIPS / ICLR / ICML grade).

**Research claim under test** (from [RESEARCH_SCOPE.md](../context/RESEARCH_SCOPE.md)):
> A frozen multimodal foundation model stack (Chronos-2 + V-JEPA 2.1) with deep token-level fusion achieves cross-plant PV power forecasting on disjoint test plants, beating late fusion, unimodal FMs, and domain-specific architectures.

Every baseline below exists to falsify one part of that claim. A reviewer must not be able to say "the gain could come from X and you never tested X".

---

## 1. Baseline Suite (tiered)

Legend — **Priority**: `P0` = mandatory for submission; `P1` = strongly expected by reviewers; `P2` = nice-to-have / rebuttal ammunition. **Status**: ✅ in repo, 🔧 planned in protocol, ➕ **new addition proposed by this document**.

### Tier 0 — Reference / statistical (P0, near-zero cost)

| Model | Inputs | Why it must be there | Status |
|---|---|---|---|
| Persistence (naive last-value) | `Y` | Absolute floor; intra-hour solar persistence is hard to beat at h=1 | 🔧 |
| **Smart Persistence** (clearness-index persistence) | `Y` (+ clear-sky GHI cov) | The denominator of the Skill Score (§4.3); solar-community standard | 🔧 |
| Hourly climatology | `Y` (train plants) | Detects leakage / trivial seasonality wins | ➕ P1 |
| Seasonal-naive (same time yesterday) | `Y` | GIFT-Eval / fev-bench standard reference | ➕ P1 |

### Tier 1 — Classical ML (P0)

| Model | Inputs | Notes | Status |
|---|---|---|---|
| LightGBM / XGBoost | flattened `Y, X_cov` | Strong tabular baseline; quantile objective for CRPS | 🔧 |
| TabPFN-3 (time-series mode) | `Y, X_cov` | Tabular FM counterpoint to TSFMs | 🔧 |

### Tier 2 — Supervised deep TS (trained on train plants)

| Model | Inputs | Why | Status |
|---|---|---|---|
| MLP | `Y, X_cov` | Simplest learned baseline | 🔧 |
| **DLinear** | `Y` | The "embarrassingly simple linear" check — reviewers *always* ask | ➕ **P0** |
| **PatchTST** | `Y, X_cov` | Strongest supervised patch transformer | ➕ **P0** |
| **iTransformer** | `Y, X_cov` | Channel-attention SOTA, covariate-friendly | ➕ P1 |
| TFT | `Y, X_cov` | Quantile-native; gives CRPS comparison for free | 🔧 |

All Tier-2 models come essentially free via the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (PatchTST, iTransformer, DLinear share one trainer). One `baselines/tslib/` port covers the whole tier.

### Tier 3 — TS foundation models, zero-shot & fine-tuned

| Model | Inputs | Why | Status |
|---|---|---|---|
| Chronos-2 (ZS + FT) | `Y` (+`X_cov` group attention) | Our own backbone; ZS = H0 anchor | 🔧 (A00) |
| **TimesFM 2.5** (ZS) | `Y` | Independent TSFM family — shows H3 results are not Chronos-specific | ➕ **P0** |
| **TiRex** (ZS) | `Y` | xLSTM family, top GIFT-Eval ZS performer; third architecture family | ➕ P1 |
| TTM-R3 (ZS/FT) | `Y, X_cov` | Tiny-model counterpoint (does scale matter?) | ➕ P2 |
| Toto-2 / Moirai-2 (ZS) | `Y` | Optional breadth; only if leaderboard framing needed | ➕ P2 |

> **Top-tier rule of thumb**: ≥3 distinct TSFM families evaluated zero-shot on the same protocol. With Chronos-2 + TimesFM 2.5 + TiRex this box is checked.

### Tier 4 — Frozen-TSFM adaptation (direct competitors to our fusion thesis)

| Model | Inputs | Why it competes with us | Status |
|---|---|---|---|
| TS-RAG on frozen Chronos-2 | `Y` + retrieved windows | "Retrieval, not vision, closes the gap" rebuttal (H4) | 🔧 (A07) |
| Cross-RAG | `Y` + retrieved windows | Stronger RAG fusion (A08) | 🔧 |
| **CoRA** (covariate-aware TSFM adaptation, [arXiv:2510.12681](https://arxiv.org/abs/2510.12681)) | `Y, X_cov` (+any-modality exogenous) | **Closest published competitor**: injects exogenous covariates into frozen TSFM backbones. If PVTSFM ≤ CoRA-with-image-features, the token-fusion claim dies. Must compare. | ➕ **P0** |
| MEMTS / TS-Memory | `Y` + memory adapter | Parametric-memory alternative to retrieval | ➕ P2 |

### Tier 5 — Generic multimodal TS (vision/text-augmented forecasters)

| Model | Real images? | Why | Status |
|---|---|---|---|
| **Time-VLM** | ❌ (renders TS as image) | Must show that *real* satellite frames beat TS-rendered pseudo-images | 🔧 noted, ➕ **P0** to actually run |
| **UniCast** ([arXiv:2508.11954](https://arxiv.org/abs/2508.11954)) | ✅ soft-prompt vision+text into TSFM | Frozen-FM multimodal prompting — same design space as ours, weaker fusion. Ideal contrast for H2 (deep fusion > prompting). | ➕ P1 |
| **Aurora** ([arXiv:2509.22295](https://arxiv.org/abs/2509.22295)) | ✅ multimodal TSFM, ZS probabilistic | Generative multimodal TSFM; covers the "why not just use a multimodal TSFM" question | ➕ P2 |
| VisionTS++ | ❌ (TS→image) | Cite + position; run only if reviewers demand | ➕ P2 |

### Tier 6 — PV-specialized multimodal (domain SOTA)

| Model | Inputs | Why | Status |
|---|---|---|---|
| **Solar-VLM** | `Y, X_cov, V` (satellite) + text | Primary domain SOTA; already ported | ✅ `baselines/solar_vlm/` |
| SUNSET | `Y, V` | Canonical CNN solar baseline, used by every related work | ➕ **P0** |
| **CrossViVit** (Boussif et al., NeurIPS 2023) | `Y, X_cov, V` (satellite) | The reference *deep* satellite+TS cross-attention model; the strongest non-FM multimodal competitor. Code public. | ➕ **P0** |
| SPIRIT | `Y, V` (ViT features) | Zero-shot vision-FM transfer — direct comparison for cross-plant ZS claims (H3) | ➕ P1 |
| PV-VLM ([arXiv:2504.13624](https://arxiv.org/abs/2504.13624)) | `Y, X_cov, V` + text | Second PV-domain VLM; rebuttal breadth beyond Solar-VLM | ➕ P2 |
| M3S-Net ([arXiv:2602.19832](https://arxiv.org/abs/2602.19832)) / FusionSF | `Y, X_cov, V` | Non-foundation PV multimodal fusion — "do FMs beat specialized PV fusion?" | ➕ P2 |
| MDCTL-MCI | `Y, X_cov` | Missing-data transfer learning; only for robustness section | ➕ P2 |

### Tier 7 — Internal ablations & controls (P0, ours)

| Control | What it proves | Registry |
|---|---|---|
| Chronos-2 frozen, TS-only | Vision adds signal at all (H1) | A00 |
| Late fusion (V-JEPA adapter) | Fusion depth matters (H1 vs H2) | A01 |
| Interleaved token fusion | Main contribution (H2) | A02 |
| **Shuffled-frames control** (random temporal permutation of `V`) | Model truly *reads* the frames; gain is not regularization | ➕ **P0** — register as A09 |
| **Mismatched-plant frames control** (frames from a different plant) | Spatial grounding, not generic cloud prior | ➕ P1 — register as A10 |
| Vision-only (no `Y` history beyond 1 step) | Upper bound on visual signal | ➕ P1 — A11 |
| **Modality-contribution grid**: TS / TS+cov / TS+vis / TS+cov+vis | Clean 4-way decomposition of where the gain comes from (subsumes single zero-out ablations) | ➕ **P0** — A12 |
| Modality dropout sweep | Robustness to missing frames (§5.3) | 🔧 (mask exists in contract) |
| Grassmann vs self-attention mixer (**param-matched**) | Architecture choice; report VRAM + latency vs context length T ∈ {128, 512, 1024, 2048}, not just accuracy | A03 |
| Visual window 3/6/12 h | Sensitivity | A04 |
| Visual token budget (n° tokens after latent summarizer) | Compression-quality trade-off of the vision branch | ➕ P1 — A13 |
| Frozen vs partial-unfreeze backbone | Separates fusion architecture from adaptation capacity (complements Chronos-2 FT row) | ➕ P1 — A14 |
| Retrieval datastore size / top-k sweep (RAG baselines) | Fairness: RAG baselines tuned, not strawmanned | ➕ P1 — A15 |

---

## 2. Minimum bar for "top-tier grade"

The P0 set, in one line each:

1. Smart Persistence + Persistence (skill-score anchors).
2. LightGBM (tabular), DLinear (linear), PatchTST (supervised SOTA).
3. Chronos-2 ZS/FT + **TimesFM 2.5** ZS (second TSFM family; TiRex as third, P1).
4. **CoRA** (covariate adaptation of frozen TSFMs — closest published method).
5. TS-RAG (retrieval alternative).
6. Time-VLM (pseudo-image multimodal).
7. Solar-VLM + SUNSET + **CrossViVit** (domain SOTA, shallow→deep multimodal).
8. Shuffled-frames control + late-fusion ablation.

What this buys at review time: every cell of the rebuttal matrix is covered — *simpler model?* (DLinear), *just covariates?* (CoRA), *just retrieval?* (TS-RAG), *backbone-specific?* (TimesFM/TiRex), *vision actually used?* (shuffled frames), *domain SOTA?* (Solar-VLM/CrossViVit), *fusion depth?* (late vs interleaved).

### Claims → required evidence map

| Paper claim | Evidence that proves it |
|---|---|
| Improves cross-plant generalization | Beats Chronos-2 ZS/FT, Solar-VLM, Cross-RAG, CoRA on disjoint test plants (S2) with DM-significance |
| Visual tokens add value | Beats TS-only and no-vision ablations, **especially on the ramp subset (S6)**; shuffled-frames control degrades |
| Deep fusion matters | Interleaved beats late fusion, same backbones, same data (A01 vs A02); beats UniCast-style prompting |
| Not just better adaptation | Beats or matches Chronos-2 FT, partial-unfreeze (A14), retrieval (TS-RAG/Cross-RAG) and memory (MEMTS) wrappers |
| Practical robustness | Holds under missing/stale frames, low-history, per-plant variance; efficiency table competitive |

### Deliberately excluded (state it in the paper, pre-empt the reviewer)

**Few-shot in-context adaptation curves** (support set of K historical days per test plant, MAE-vs-K): excluded by design decision [A06 in ABLATION_REGISTRY.md] — the protocol is disjoint cross-plant zero-shot with short inference history, not few-shot context matching. If a reviewer requests it, the harness supports it as an appendix experiment: K ∈ {0, 1, 3, 7} days as RAG datastore (retrieval baselines) or linear-probe update (adapter models); keep it out of headline tables.

---

## 3. Input-parity matrix (fairness contract)

Every model consumes only the canonical dict from [DATASET_CONTRACT.md](../context/DATASET_CONTRACT.md). No baseline may receive inputs the others cannot access in their tier.

| Tier | `Y` | `X_cov` | `V` | Retrieval | Text |
|---|---|---|---|---|---|
| T0 Reference | ✅ | clear-sky only | — | — | — |
| T1/T2 Supervised | ✅ | ✅ | — | — | — |
| T3 TSFM | ✅ | ✅ where native | — | — | — |
| T4 Adaptation | ✅ | ✅ (CoRA) | — | ✅ (RAG) | — |
| T5/T6 Multimodal | ✅ | ✅ | ✅ | model-specific | model-specific |
| PVTSFM (ours) | ✅ | ✅ | ✅ | optional (H4) | — |

Rules (inherited from BASELINE_PROTOCOL.md, restated as hard constraints):
- Same `T`(=24), `H`(=12 primary, 48 long-horizon), `T_v`(=8), cadence, and normalization for everyone.
- Disjoint train/val/test **plants**; no per-test-plant statistics anywhere (including normalizers — use train-plant or capacity normalization only).
- No clear-sky-index physics inside models (Smart Persistence exempt, it *is* the physics reference).
- Multimodal models that natively want text (Solar-VLM) may generate weather text only from covariates available to all — no external weather APIs.
- **Retrieval datastore rule**: RAG/memory baselines (TS-RAG, Cross-RAG, MEMTS) may populate their datastore/memory with **train-plant data only**. Transductive retrieval (test-plant history in the datastore) is a separate, explicitly-labeled condition — never mixed into the headline table.

---

## 4. Evaluation protocol

### 4.1 Scenarios

| Scenario | Split | Question answered |
|---|---|---|
| **S1 In-domain** | train plants, held-out time range | Sanity / upper bound |
| **S2 Cross-plant (primary)** | disjoint test plants, SKIPP'D + goes16_nsrdb | Headline result (H3) |
| **S3 Cross-dataset** | train on SKIPP'D → test solarnet (and reverse) | Distribution-shift generalization |
| **S4 Long-horizon** | S2 with H ∈ {12, 24, 48} | Skill decay curves (short / mid / long) |
| **S5 Data efficiency** | S2 with 10/25/50/100 % train plants | FM sample-efficiency claim |
| **S6 Ramp subset** | S2 restricted to high-variability windows (top-decile \|ΔY\| cloud-transition periods) | Where vision *should* win — the sharpest test of H1/H2 |
| **S7 Seasonal transfer** (P2) | Train on subset of months, test unseen season | Temporal distribution shift |

Plant-split variants for S2: if test-plant count is small, use **leave-one-plant-out** rotation (mean ± std over folds); if geographic metadata permits, prefer a **distance/region-based split** over random plant assignment — random splits of nearby plants leak spatial information.

S3, S5 and S6 are what separate a good paper from an accepted paper: zero-shot/few-data curves are the standard FM evidence, and the ramp subset is where the multimodal claim lives or dies (clear-sky periods are won by persistence; vision earns its tokens during cloud transitions).

### 4.2 Point metrics

Computed per plant, then macro-averaged over plants (prevents large plants dominating):

- **NMAE**, **NRMSE** — capacity-normalized, as defined in BASELINE_PROTOCOL.md §5.
- **Skill Score** `SS = 1 − NRMSE_model / NRMSE_smartpersistence` — solar-community headline number.
- **Ramp-event NMAE/NRMSE** — same metrics restricted to the S6 ramp subset (ramp ≔ top-decile \|ΔY\| within plant). Report alongside overall metrics in the headline table; this is the direct evidence that cloud-motion information is captured.
- **Per-horizon breakdown** — report NMAE(h) for h ∈ {1, …, H}; plot decay curves for S4.
- **TEMPLATE transferability scores** (P1, per RESEARCH_SCOPE): DLS / PLS / TAS on frozen representations — ranks backbones and fusion variants without fine-tuning.
- Daylight-only masking: all metrics computed where `mask_future · daylight = 1`; report the mask convention once, use everywhere.

### 4.3 Probabilistic metrics (P0 — the scope marks CRPS as primary)

- **CRPS** approximated by mean weighted quantile loss over quantiles {0.1, …, 0.9} (Chronos/GIFT-Eval convention).
- **Pinball loss** at q ∈ {0.1, 0.5, 0.9}.
- **Coverage / calibration**: empirical coverage of the 80 % interval (target 0.80); reliability diagram in appendix; summarize as quantile **ECE** = mean over q ∈ Q of \|q − q̃\| where q̃ is the empirical proportion of ground truth below the predicted q-quantile.
- Deterministic-only baselines (DLinear, SUNSET…): report point metrics only, mark CRPS as `—`. Quantile-capable baselines (TFT, LightGBM-quantile, Chronos-2, TimesFM, PVTSFM) fill the full table.

### 4.4 Aggregation across datasets (fev-bench convention)

When reporting across multiple datasets/scenarios:
- **Win rate** vs Smart Persistence and vs Chronos-2-ZS.
- **Geometric-mean skill score** across datasets.
- **Average rank** with critical-difference, never raw metric averaging across datasets with different scales.

### 4.5 Statistical rigor (non-negotiable for top venues)

| Requirement | Procedure |
|---|---|
| Seeds | ≥3 seeds (42, 43, 44) for every *trained* model; report mean ± std. ZS models: single deterministic run, but 3 seeds over data-order where sampling is stochastic. |
| Significance | **Diebold–Mariano test** on per-sample loss differentials (PVTSFM vs each P0 baseline), plus **paired block bootstrap** (block = day, 1000 resamples) → 95 % CI on ΔNMAE. Bold table entries only when CI excludes 0. |
| Multiple comparisons | Holm–Bonferroni over the baseline set per scenario. |
| Variance disclosure | Per-plant variance in appendix (cross-plant results can hide one bad plant). |

### 4.6 Efficiency reporting (expected at FM venues)

For every model: trainable params / total params, GPU-hours to train (or "0, zero-shot"), single-window inference latency (ms, A100 + M-series CPU), peak VRAM. One table; PVTSFM's frozen-backbone story is a selling point here — make it visible.

---

## 5. Robustness & controls battery

| Test | Protocol | Pass criterion |
|---|---|---|
| **Shuffled frames** (A09) | Permute `V` along `T_v` at eval | PVTSFM degrades toward TS-only — proves vision is read |
| **Mismatched plant frames** (A10) | Swap in frames from another plant | Degradation > shuffled — proves spatial grounding |
| **Missing modality** | Drop frames at rate p ∈ {0, .25, .5, 1.0} via `mask_visual`; plus **stale-frames** variant (repeat last valid frame) | Graceful decay; at p=1.0 matches TS-only, not worse; stale ≥ missing |
| **Low-history regime** | Shrink history `T` to {4, 8, 12, 24} steps at eval on test plants | Supports "deployable on new plant" claim; FM in-context strength |
| **Night / masked targets** | Verify metrics identical with and without masked-step leakage check | Exact match |
| **Covariate ablation** | Zero out `X_cov` | Quantifies covariate vs vision contribution separately |
| **Horizon stress** | S4 decay curves vs Smart Persistence crossover point | Report the h where skill → 0 |

---

## 6. Test suite (pytest, runs in CI before any result is logged)

Extend `tests/` with a per-baseline contract test, parametrized over all registered baselines:

```text
tests/
├── test_metrics.py              # exists — extend with CRPS, pinball, coverage, DM-test unit tests
├── test_baseline_contract.py    # NEW — every baseline: consumes canonical dict → ŷ (N,H,1) finite, in range
├── test_eval_protocol.py        # NEW — split disjointness, no normalizer leakage, daylight masking
└── test_controls.py             # NEW — shuffled-frame harness produces permuted-but-aligned batches
```

Mandatory assertions:

1. **Contract**: `forecast(batch)` returns `(N, H, 1)` float32, no NaN where `mask_future=1`, for every baseline adapter (smoke, synthetic data, CPU, <60 s each).
2. **Split disjointness**: `set(train_plants) ∩ set(test_plants) == ∅` asserted from `metadata.json` at loader construction — fail loud, not in the paper rebuttal.
3. **No leakage**: normalization constants used at test time derive only from capacity or train-plant stats (unit test on the normalizer object).
4. **Metric correctness**: NMAE/NRMSE/CRPS validated against hand-computed values and against `gluonts`/`properscoring` references.
5. **Skill-score sanity**: Smart Persistence has SS = 0 by construction; persistence at h=1 has SS ≈ 0.
6. **Determinism**: same seed → identical forecasts (hash test) for trained baselines.
7. **Reproducibility manifest**: every eval run writes `{git_sha, config_hash, seed, dataset_version}` next to results (`scripts/verify_reproducibility.py` already exists — wire it in).

---

## 7. Results table templates

### 7.1 Headline (S2, cross-plant, H=12)

| Tier | Model | NMAE ↓ | NRMSE ↓ | SS ↑ | CRPS ↓ | Params (train.) | ZS? |
|---|---|---|---|---|---|---|---|
| T0 | Persistence | | | ~0 | — | 0 | ✅ |
| T0 | Smart Persistence | | | 0 | — | 0 | ✅ |
| T1 | LightGBM | | | | | | ❌ |
| T2 | DLinear | | | | — | | ❌ |
| T2 | PatchTST | | | | — | | ❌ |
| T2 | TFT | | | | | | ❌ |
| T3 | Chronos-2 ZS | | | | | 0 | ✅ |
| T3 | Chronos-2 FT | | | | | | ❌ |
| T3 | TimesFM 2.5 ZS | | | | | 0 | ✅ |
| T3 | TiRex ZS | | | | | 0 | ✅ |
| T4 | Chronos-2 + TS-RAG | | | | | 0 | ✅ |
| T4 | CoRA (Chronos-2) | | | | | adapter | ❌ |
| T5 | Time-VLM | | | | | | ❌ |
| T6 | SUNSET | | | | — | | ❌ |
| T6 | CrossViVit | | | | | | ❌ |
| T6 | Solar-VLM | | | | | | ❌ |
| — | PVTSFM late fusion (A01) | | | | | adapter | ❌ |
| — | **PVTSFM interleaved (A02)** | | | | | adapter | ❌ |

± std over 3 seeds; **bold** only when DM-test + bootstrap CI confirm significance vs best baseline.

### 7.2 Secondary tables
- S6 ramp-subset table (same rows as 7.1, ramp-event NMAE/NRMSE — expected to be the most decisive table).
- S3 cross-dataset transfer matrix (train→test dataset grid).
- S4 per-horizon NMAE curves (figure) + crossover table.
- S5 data-efficiency curves (figure).
- §5 robustness battery table.
- §4.6 efficiency table.

---

## 8. Execution order (effort-ranked)

1. **Week 0** — T0 references + metrics/test suite (§6). Everything downstream depends on it.
2. **T2 via one Time-Series-Library port** (`baselines/tslib/`): DLinear, PatchTST, iTransformer, (TFT). One adapter, four baselines.
3. **T3 zero-shot sweeps** (no training): Chronos-2, TimesFM 2.5, TiRex — one inference harness over the canonical dict.
4. **A00/A01/A02 + controls A09/A10** (our model variants — already in MMTSFM codebase).
5. **TS-RAG, then CoRA** (both wrap frozen Chronos-2; reuse harness from step 3).
6. **SUNSET, CrossViVit ports** (public code; CrossViVit is the heaviest port).
7. **Solar-VLM runs** (port exists; SLURM).
8. P1/P2 stragglers (SPIRIT, UniCast, TTM-R3, Aurora) only if time or reviewers demand.

Register every run in [ABLATION_REGISTRY.md](ABLATION_REGISTRY.md) before launch; new IDs A09–A15 defined in §1 Tier 7.

---

## 9. References

| Model / framework | Link |
|---|---|
| Chronos-2 | https://arxiv.org/abs/2510.15821 |
| TimesFM 2.5 | https://huggingface.co/google/timesfm-2.5-200m-pytorch |
| TiRex | https://arxiv.org/abs/2505.23719 |
| TTM-R3 | https://huggingface.co/ibm-research/ttm-r3 |
| CoRA | https://arxiv.org/abs/2510.12681 |
| TS-RAG | https://arxiv.org/abs/2503.07649 |
| Cross-RAG | https://arxiv.org/abs/2603.14709 |
| MEMTS / TS-Memory | https://arxiv.org/abs/2602.13783 / https://arxiv.org/abs/2602.11550 |
| Time-VLM | https://arxiv.org/abs/2502.04395 |
| UniCast | https://arxiv.org/abs/2508.11954 |
| Aurora | https://arxiv.org/abs/2509.22295 |
| VisionTS++ | https://arxiv.org/abs/2508.04379 |
| Solar-VLM | https://arxiv.org/abs/2604.04145 |
| PV-VLM | https://arxiv.org/abs/2504.13624 |
| M3S-Net | https://arxiv.org/abs/2602.19832 |
| SUNSET | https://github.com/YuchiSun/SUNSET |
| CrossViVit | https://arxiv.org/abs/2306.01112 (NeurIPS 2023) |
| SPIRIT | https://arxiv.org/abs/2502.10307 |
| PatchTST / iTransformer / DLinear | https://github.com/thuml/Time-Series-Library |
| TabPFN-3 | https://arxiv.org/abs/2605.13986 |
| GIFT-Eval (aggregation conventions) | https://arxiv.org/abs/2410.10393 |
| fev-bench (win rate / bootstrap CI conventions) | https://arxiv.org/abs/2509.26468 |
| TEMPLATE (transferability scores, P1 metric) | NeurIPS 2025 |
