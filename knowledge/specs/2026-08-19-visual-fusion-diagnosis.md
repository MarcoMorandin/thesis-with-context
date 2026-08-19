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
3. **The numeric channel may already carry the signal.** `csi`, `kt` and `clearsky_ghi` are
   covariates. These are analytically-derived cloud-attenuation indices. Vision may be
   duplicating them rather than adding anything.

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
| C2 | **Redundancy** — `csi`/`kt` already carry it | signal extractable from vision alone, but nothing conditional on covariates |
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

**Target.** Clear-sky index at each horizon step, `k_{t+h} = P_{t+h} / P_clearsky_{t+h}`,
from the existing `clearsky_ghi` column. This removes deterministic solar geometry and
isolates the cloud-driven component. Predicting raw power would let a probe score well on the
diurnal cycle alone and answer nothing.

**Three predictor sets:**

| set | inputs | question |
|---|---|---|
| (a) | visual latents only | is there any cloud signal in the representation? |
| (b) | `csi`, `kt`, `clearsky_ghi`, solar geometry, power lags | what does the numeric channel already know? |
| (c) | both, concatenated | does vision add anything on top? |

**`(c) − (b)` is the quantity the study turns on.** It is the information vision carries
*conditional on* covariates already computed analytically, and it upper-bounds what any
fusion architecture could extract. Reported per horizon step `h = 1..12`, never aggregated.

**Feature variants.** F2 = spatial mean-pool preserving time, `[4, 1024]` = 4096 dims; ridge
over all ~98 k stride-12 train samples, closed form. F3 = 196 patches pooled to a 4×4 grid,
`[4, 16, 1024]`; small MLP on a ~20 k subsample. F3 exists because spatial mean-pooling
destroys *where* cloud sits relative to the plant, which is the advection signal.

**Reference lines.** Smart persistence (the `protocol.md` reference), csi-persistence, and
the covariates-only probe. Reported as skill per horizon so it is commensurable with existing
SS numbers.

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
| 1 | **capacity**: `n_vis` 1 → 8 | 8× visual tokens over the same 6 h window |
| 2 | **forcing**: auxiliary loss, visual tokens → `k_{t+h}` | breaks modality laziness |
| 3 | **capacity + forcing** | do the two compose, or is one subsumed? |

s2b is the fourth cell of the 2×2 and is already run, so four GPUs complete the factorial
plus the control.

The capacity arm is close to free: `LatentSummarizer` already emits `n_vis` tokens with a
causal sub-interval mask per token, and `interleave_sequences` already handles arbitrary
`n_vis`. It is a config change plus relaxing `validate_n_visual_context_steps`. The forcing
arm is new code: a head on the visual summary predicting clear-sky index at horizon, with a
loss weight defaulting to 0 so current behaviour is bit-preserved when off.

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

- **G0 passes** if `(c) − (b)` skill > 0 at any horizon by more than the probe's
  cross-validation spread, defined as the standard deviation of per-fold skill across 5-fold
  CV on the train split, evaluated on the held-out test plants.
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
