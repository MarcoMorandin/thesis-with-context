# Visual Fusion — Differential Diagnosis and Matched Intervention (Design Spec)

**Date:** 2026-08-19
**Status:** Approved design, pending spec review
**Branch:** `main`
**Scope:** A gated methodology for determining *why* the visual modality contributes almost
nothing in MMTSFM, and fixing whichever cause the evidence identifies. Testbed `uk_pv`;
`goes_pvdaq` held as validation. Deliverable is the procedure plus the measurement
instrument, not a single tuned model.

---

## 1. Problem statement

On `uk_pv`, with the full curriculum complete and both checkpoint-integrity bugs fixed, the
visual pathway measurably contributes — but barely. Forced vision-off ablation on s2b's own
weights (job 52698586, deterministic, exact):

    dNMAE  = 0.00029119   (0.39% relative)
    dNRMSE = 0.00102141   (0.92% relative)

Decomposed against the vision-free s1 baseline:

| metric | s2b gap vs s1 | from vision | from training recipe |
|---|---|---|---|
| NMAE  | 0.00149646 | 19% | 81% |
| NRMSE | 0.00231685 | 44% | 56% |

Of the +0.0101 SS that s2b holds over s1, roughly **+0.004 is the visual signal and +0.006
is the recipe**. Late fusion (s2a, SS 0.5076) is indistinguishable from having no vision at
all (s1, SS 0.5087) — `scope.md` H1 is falsified. Interleaved fusion (s2b, SS 0.5188) beats
late fusion — H2 is confirmed.

Three structural facts constrain any explanation:

1. **The visual pathway is one token.** `architecture.md` §2.3: "42 + 1 + 1 = 44 tokens for
   `uk_pv`, ~2 %". `n_vis = ceil(12 steps / patch 16) = 1`. A `[4, 196, 1024]` V-JEPA latent
   — 802,816 values — is compressed into a single 768-d token, ~500:1. Vision then delivers
   0.4–0.9 %, roughly proportional to its share of the sequence.
2. **The representation has a double domain gap.** `dataset.md` §2: `uk_pv` frames are
   `(N, 128, 128)` **single-channel grayscale** satellite crops. The pipeline upsamples to
   224×224 and replicates to 3 channels for a V-JEPA ViT-L/16 pretrained on natural RGB
   video. The upsample adds no information; cloud fields are far out of distribution.
3. **The numeric channel may already carry the signal — but not via `csi`/`kt`.**
   Correction to an earlier draft: `baselines/common/config.py` defines `COV_COLS` as the 14
   keys of `COV_SCALES`, and **`csi` and `kt` are not among them** — the model never sees
   them. What it does see is `cloudcover` plus four radiation fields
   (`shortwave_radiation`, `direct_radiation`, `diffuse_radiation`,
   `direct_normal_irradiance`), and per `DETERMINISTIC_COVS` those are **history-only**; for
   future steps the model gets solar geometry and `clearsky_ghi` alone. So the redundancy
   question is sharper than first stated: past observed irradiance and cloud cover give a
   cloud-persistence baseline, and vision must beat that by supplying cloud *motion*. This is
   also exactly the gap where vision should help, since future cloud is genuinely unknown to
   the numeric channel.

## 2. What this is not

Not a search for a better fusion architecture. Not a leaderboard push — s2b at 0.5188 already
sits above `chronos2_oracle_ft` (0.5042) and below iTransformer (0.5546/0.5521/0.5509), and
that ordering is not what is being contested here.

The contribution is that **"the second modality does not help" decomposes into five
distinguishable causes, four of which are cheaply testable before any architecture changes.**
`uk_pv` is the testbed; the procedure is meant to transfer to any frozen-FM modality pair.

## 3. Candidate causes

| # | Cause | Would look like |
|---|---|---|
| C1 | **Representation** — V-JEPA features do not encode cloud dynamics for this imagery | no extractable signal even with an unconstrained probe |
| C2 | **Redundancy** — history irradiance/`cloudcover` already imply near-future cloud | signal extractable from vision alone, but nothing conditional on covariates |
| C3 | **Capacity** — 500:1 compression into one token | marginal gain scales with visual token count |
| C4 | **Optimization** — modality laziness / gradient starvation | vision-path gradients orders below numeric; fusion gate closed |
| C5 | **Horizon** — 6 h ahead is long for cloud advection | gain concentrated at short horizons, diluted in the aggregate |

## 4. Architecture — three gates

Each gate can terminate the study with a defensible answer, so the expensive gate runs only
when the cheap ones justify it.

```
G0  CEILING      what is extractable at all?     free (CPU)   -- may end here
     |           cached latents, no model forward
     v
G1  LOCALIZE     which of C1-C5?                 free (CPU)   -- may end here
     |           test-only probes + existing logs
     v
G2  INTERVENE    does the matched fix move it?   1 node-day (4x A100)
                 2x2 factorial + seed control
```

### 4.1 G0 — ceiling probe

Operates directly on the cached V-JEPA latents
(`/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224`, 139,611 files). No V-JEPA
forward pass, no Chronos, no GPU. Implementation verifies the `[4, 196, 1024]` fp16 shape
before building the probe geometry.

**Target.** Power at each horizon step, `P_{t+h}` — the same quantity `protocol.md` scores,
under the same daylight mask.

An earlier draft used clear-sky index `k_{t+h}` to strip the diurnal cycle. Power is the
better choice for two reasons. First, the diurnal concern does not apply to the quantity this
study actually turns on: predictor set (b) already contains solar geometry and
`clearsky_ghi`, so it absorbs the deterministic diurnal component, and `(c) − (b)` is by
construction the non-diurnal residual regardless of target. Second, `csi` is NaN below
50 W/m² clear-sky (`dataset.md` §1bis), so a `k`-target silently drops or imputes exactly the
low-light steps, whereas power under the protocol's daylight mask keeps the sample definition
identical to every other number in the project.

The cost is that predictor set (a) alone becomes uninterpretable in isolation — a visual-only
probe can score on time-of-day inferred from scene brightness. That is acceptable: (a) was
only ever a sanity check, and `(c) − (b)` was always the operative quantity.

**Three predictor sets:**

| set | inputs | question |
|---|---|---|
| (a) | visual latents only | is there any cloud signal in the representation? |
| (b) | exactly what the model gets: `norm_power` lags + history `COV_COLS` (incl. `cloudcover`, 4 radiation fields) + future `DETERMINISTIC_COVS` | what does the numeric channel already know? |
| (c) | both, concatenated | does vision add anything on top? |

**`(c) − (b)` is the quantity the study turns on.** It is the information vision carries
*conditional on* covariates already computed analytically, and it upper-bounds what any
fusion architecture could extract. Reported per horizon step `h = 1..12`, never aggregated.

**Feature variants.** F2 = spatial mean-pool preserving time, `[4, 1024]` = 4096 dims; ridge
over all ~98 k stride-12 train samples, closed form. F3 = 196 patches pooled to a 4×4 grid,
`[4, 16, 1024]`; small MLP on a ~20 k subsample. F3 exists because spatial mean-pooling
destroys *where* cloud sits relative to the plant, which is the advection signal.

**Reference lines.** The covariates-only probe (b) is the reference that matters; a smart
persistence line is deliberately NOT computed.

Reasoning (decided 2026-08-19 during implementation): the gate quantity is a *difference*,
`skill(c) − skill(b)`, in which any shared reference cancels — so smart persistence would
change the number's scale without changing its sign, its zero, or its comparison against the
CV spread. Building it would additionally require `build_arrays` to return `norm_power` at the
window origin (smart persistence is `P(t)·clearsky(t+h)/clearsky(t)`), widening the data layer
for no gain to the decision.

What is reported instead:
- **absolute per-horizon NMAE** for each predictor set. This compares *directly* against the
  model's own NMAE (s2b: 0.07539) with no baseline in between — a tighter comparison than a
  skill score, since it needs no shared reference to be meaningful.
- **`conditional_rel`** = `(NMAE_b − NMAE_c) / NMAE_b` per horizon: the fraction of the
  covariates-only error that vision removes. Reference-free by construction, and the quantity
  the success gate is evaluated on.
- per-horizon `n_test_valid` counts, so "no data at this horizon" can never be misread as
  "no signal at this horizon".

**Regularization — the probe must not be able to invent a negative result.** Added
2026-08-19 after the first real run; the earlier fixed `alpha=1.0` on raw features is
superseded. Set (c) bolts ~4 k visual dimensions onto ~100 strongly predictive covariates, so
three things are load-bearing:

1. **Standardize** on train statistics. Ridge penalizes every coefficient equally, so on raw
   features the V-JEPA activation scale — an arbitrary property of the encoder — decides how
   hard the visual block is regularized relative to covariates already scaled into `[0,1]`.
2. **Separate penalty per block**, selected by the same plant-disjoint `GroupKFold` that
   produces `cv_spread`. Under one shared penalty there is no setting that both keeps the
   covariate fit and suppresses an uninformative visual block, so (c) is *forced* below (b)
   and the probe reports "vision hurts" when it only overfit.
3. **The visual grid reaches 1e10**, far above the standardized eigenvalues of order `n`, so
   the block can be switched off. With (c)'s covariate penalty pinned to the one (b) selected,
   (c) can always reproduce (b) exactly — `conditional_rel` is therefore bounded below by ≈ 0
   *by construction*. That is the correct null for a ceiling probe: it may report that vision
   helps or that it does nothing, never that it harms.

`alpha_selected` is written to the report. A visual penalty pinned at the grid maximum means
"vision off won"; a covariate penalty at either edge means the grid was too narrow and the
run is not trustworthy.

**Run log.**

| run | probe | result | status |
|---|---|---|---|
| 1 | fixed `alpha=1.0`, raw features | `conditional_rel` −0.027…−0.016 at h≤5, exactly 0.00000 at h≥6 | **superseded** — the reliable negative is the overfitting signature above, not evidence about cloud information |

Run 1 did establish two things that carry forward: `n_skipped` = 0 on both splits (the
`build_site_series` grid join is sound), and `n_test_valid` halves at h=6 (19,831 → 9,899)
and then stays flat — beyond 3 h ahead only windows whose targets stay in daylight survive,
so h≤5 and h≥6 are measured on different, time-of-day-restricted populations and are not
comparable across that boundary. Read each region on its own.

### 4.2 G1 — localization probes

| probe | tests | source |
|---|---|---|
| horizon decomposition | C5 | re-score existing `results/predictions/*_pred.npz` |
| condition stratification | C5 dilution | stratify by within-window `csi` variance |
| gradient flow | C4 | `_GRAD_GROUPS` already logs `train/grad_norm/vision_adapter`, `latent_summarizer` |
| gate/bias inspection | C4 | `W_gate` α statistics, `modality_pair_bias` magnitude |

The gate probe has a known-good contrast available: `modality_pair_bias` was **exactly 0.0**
in all 12 blocks through s1 and s2a, so the logs already show what "this pathway receives no
gradient" looks like in this codebase.

### 4.3 G2 — intervention, 2×2 factorial plus control

Runs only if G0 shows `(c) − (b) > 0` while the model's marginal gain stays ≈ 0 — i.e. the
signal exists and the model is not using it.

| GPU | arm | purpose |
|---|---|---|
| 0 | s2b config, **seed 43** | seed-noise floor |
| 1 | **capacity**: `n_visual_tokens_per_step` 1 → 8 | widen the bottleneck 8×, same temporal extent |
| 2 | **forcing**: auxiliary loss, visual tokens → `k_{t+h}` | breaks modality laziness |
| 3 | **capacity + forcing** | do the two compose, or is one subsumed? |

s2b is the fourth cell of the 2×2 and is already run, so four GPUs complete the factorial
plus the control.

**Capacity arm — why not simply raise `n_vis`.** `validate_n_visual_context_steps`
(`vision_chronos2.py:51`) defines `n_visual_context_steps` as "how many of the most-recent
context PATCHES are given visual tokens". With `input_patch_size=16`, `n_vis=8` therefore
asserts visual coverage over 8 × 16 = 128 TS steps = **64 hours**, while the underlying clip
spans 6. The extra tokens would be placed at TS positions the video never observed, and RoPE
would encode them as 64 hours apart. That inflates token count while corrupting temporal
semantics — it would not be a clean capacity manipulation.

The clean version keeps `n_vis=1` and adds `n_visual_tokens_per_step` (default 1, preserving
current behaviour): `LatentSummarizer` emits k tokens for the single refinement position, and
`interleave_sequences` places all k after that TS token, sharing its position id — consistent
with the existing design where a TS token and its visual partner are already co-temporal. The
bottleneck widens from 768 to k × 768, taking the compression from ~500:1 to ~62:1 at k=8,
with the temporal story unchanged. Requires a change to `LatentSummarizer` and
`interleave_sequences`; the alternative capacity levers are worse (widening
`visual_window_hours` changes the information rather than the capacity and needs a full cache
re-extraction; shrinking `input_patch_size` reinitialises the pretrained Chronos-2 tokenizer,
a mistake `2026-07-20-mmtsfm-curriculum.md` §1 already documents).

**Forcing arm.** New code: a head on the visual summary tokens predicting **clear-sky index**
`k_{t+h}`. Since `csi` is not a model input, the target is derived inside the batch as
`Y_future / max(clearsky_ghi_future, eps)` using `COV_COLS` index 13 (`clearsky_ghi`, scaled
by 1000 in `COV_SCALES`) — no dataloader change needed — and masked where clear-sky is below
the protocol's daylight threshold. Loss weight defaults to 0 so current behaviour
is bit-preserved when off. Note this target deliberately differs from G0's: the probe measures
what is extractable and must stay commensurable with the protocol, whereas the auxiliary loss
must force the visual tokens to encode *cloud* specifically — a power target would be
satisfiable by encoding time-of-day.

## 5. The seed-noise floor

Every MMTSFM number to date is n=1. `+0.010 SS` cannot currently be distinguished from seed
variance; iTransformer's 3-seed spread is 0.0037, but that is a different model. One G2 GPU
therefore re-runs the exact s2b config at seed 43.

Without it, a successful intervention is unfalsifiable. With it, every claim downstream reads
"moved by more than seed noise" rather than "moved".

## 6. Execution and allocation

**G0/G1 belong on CPU.** Ridge fits, MLP probes on ~20 k subsamples, and re-scoring cached
`.npz` predictions do not need A100s. Run on `lrd_all_serial`; this costs nothing from the GPU
budget. They parallelize four ways trivially (3 predictor sets × 2 feature variants, horizons
vectorized as multi-output regression).

**G2 saturates one node**, reusing the pattern already in `scripts/run_all_mmtsfm.sh`:
`--gres=gpu:4`, four background `uv run` invocations each pinned with `CUDA_VISIBLE_DEVICES`
and `trainer.devices=1`, `wait` on the PIDs, per-arm logs. ~15 h wall clock against ~60 h
serial.

Total GPU cost for the whole study: **one 4-GPU node-day**, plus free CPU time.

Walltime follows `slurm_curriculum.sh`'s current policy — reserve to the 24 h partition cap,
because `train.py` exports `best.ckpt` and runs the test pass only after `fit()` returns, so a
TIMEOUT loses the artifacts entirely.

## 7. Decision procedure

| Evidence | Diagnosis | Action |
|---|---|---|
| `(c)−(b)` ≈ 0 at every h | C1 or C2 | Stop fusion work. Disambiguate by re-extracting features with a different encoder from `images_all.h5` (NOT free — the cache holds V-JEPA latents, not frames, so this is a fresh extraction pass) and re-running G0 only. Still ≈ 0 means C2 (redundancy), and that is the finding. Resolution-limited → `goes_pvdaq`. |
| `(c)−(b)` > 0 only at h ≤ 3 | C5 | Horizon-aware fusion; visual tokens gate only near-horizon output patches. Short-horizon gain becomes the primary metric. |
| `(c)−(b)` > 0 across h, model gain ≈ 0 | C3 or C4 | Proceed to G1 to discriminate, then G2. |
| G1: gain scales with `n_vis` | C3 | Capacity arm. |
| G1: vision grad ≪ numeric, gate α saturated | C4 | Forcing arm. |
| G1: gain concentrated in variable-sky strata | C5 dilution | Condition-stratified evaluation as primary. |

## 8. Success gates

- **G0 passes** if `conditional_rel` > 0 at any horizon by more than that horizon's
  cross-validation spread (`cv_spread_rel`), defined as the standard deviation of the same
  reference-free quantity across 5-fold GroupKFold on the train split (folds grouped by plant,
  never random), evaluated on the held-out test plants. Both sides of this comparison are on
  the same normalization basis — an earlier draft compared a globally-normalized conditional
  against a fold-locally-normalized spread, which is not a valid test.
- **G1 passes** if it identifies a cause by a discriminating signal, not a plausible story.
- **G2 succeeds** if an intervention moves dNMAE (currently 0.0002912, 0.39 % relative) beyond
  **2× the measured seed-noise band** — measured at G2, not assumed.

## 9. Testing

- Probe code gets unit tests on synthetic data with a **known** injected signal: a probe that
  cannot recover a planted correlation is not evidence of absence. This is the guard against
  the failure mode where G0 returns ≈ 0 for an implementation reason.
- The `(b)` covariates-only probe doubles as a sanity check: it must beat csi-persistence, or
  the feature assembly is wrong.
- Aux-loss arm: a test asserting `visual_aux_loss_weight=0` reproduces current s2b behaviour
  exactly, so the factorial's control cell is genuinely a control.
- All arms re-verify checkpoint integrity via `repair_vjepa_checkpoint.py --inspect` before
  any number is trusted, — a stripped V-JEPA encoder silently substitutes the pretrained baseline and makes a
  checkpoint score differently than the run that produced it (fixed in cc88f41).

## 10. Anticipated outcome and its consequence

`csi` and `kt` are analytically-derived cloud-attenuation indices already in the covariate
channel. There is a substantial chance `(c) − (b)` returns ≈ 0 because vision duplicates them.

That ends the study at G0 in roughly two days with a defensible finding about **modality
redundancy in frozen-FM fusion** — that a strong analytic covariate can fully account for a
second modality's apparent contribution — but not a "we made fusion work" result. The gate
structure exists to reach that verdict cheaply rather than after a month of architecture work.

## 11. Related

- [architecture.md](../architecture.md) §2.3 selective temporal interleaving, §2.4 frozen
  V-JEPA + summarizer
- [scope.md](../scope.md) hypothesis ladder: H1 falsified, H2 confirmed by the current
  curriculum
- [protocol.md](../protocol.md) metric and split definitions
- [ablations.md](../ablations.md) registry — G2 arms register here before they run
