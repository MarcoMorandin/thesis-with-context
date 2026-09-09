# s2d paper-readiness audit — 2026-09-07

Scope: what the repo already holds toward a top-tier (ICLR/NeurIPS-grade) paper whose
proposed architecture is **s2d** (`fusion_mode=interleaved_raw`), and what is missing.
s2b, s2b_wide and s2c are excluded from the paper by the user's instruction; consequences of
that exclusion are in §3. Analysis only — nothing was run. Sources: `knowledge/ablations.md`,
`knowledge/baselines.md`, `knowledge/protocol.md`, `report/report.typ`, `baselines/results/`,
`MMTSFM/configs/`, GitNexus, and the closed tickets of this map.

## 1. What already exists (paper-grade)

| Asset | State | Where |
|---|---|---|
| s2d headline, n=3 | SS 0.5510 ± 0.0020, ramp NMAE 0.1440 ± 0.0008, best on disk on both P0 metrics | `mmtsfm_s2d_ukpv_s2d_s4{2,3,4}.json`, `ablations.md` §1 |
| Marginal gain (vision-off pass, shared ramp masks) | Δ NMAE 0.0047 ± 0.0001, Δ ramp 0.0063 ± 0.0006, all seeds clear floor | same JSONs, `eval/protocol_eval.py::_ramp_masks` |
| Content-grounding controls | A30-b stale-sky and A30-c swap-plant: +0.016–0.019 ramp NMAE, marginal gain flips negative, all 3 seeds | tickets 22, 25 |
| Mechanism controls | A30-a frame-shuffle (near-null), A30-d EVS sweep (U-shaped, confounded), A30-e 7×7 probe (unsupportive) | tickets 21, 24, 23 |
| Gate call + H1/H2 verdicts | ramp recovery real; mechanism story not established; H1 split, H2 provisional | tickets 26, 10 |
| Calibration cost measured | coverage_80 0.768→0.731, ECE 0.0280→0.0359, flat CRPS | `ablations.md` §1 |
| Prior art verified | SolCAD-Net (Energy 2026) real, pre-empts s2c framing not s2d's; PVNet, pySTEPS, PV-VLM, KNMI benchmark positioned | `knowledge/specs/2026-09-08-prior-art-verification.md` |
| Code path + tests | `VisualPatchProjector`, `interleaved_raw` branch, `test_s2d_interleaved_raw.py`; eval controls with falsifiability guard | GitNexus |
| Baseline suite | 27 baselines across tiers 0–6, 76 JSONs at n_plants=14; per-horizon and coverage blocks present for tiers 0–3 + MMTSFM | `baselines/results/` |
| Pre-registered floors | ramp NMAE 0.0011, SS 0.0037 | `protocol.md` §5.1 |

The thesis-grade bar (map destination as of 2026-09-08) is met. The paper-grade bar is not,
for the reasons below.

## 2. Gaps

### 2.1 Ablations the paper cannot ship without

**G1 — H2 is not isolated (the contribution itself).** Standing decision 2 says the fusion
mechanism is the contribution. The only late-fusion arm on disk (s2a) uses the old
resampler-pooled path, so s2d-vs-s2a moves fusion mode *and* resampler *and* resolution at
once (ticket 10). The thesis accepted this as a Limitation; a top-tier reviewer will not.
Required: a **late-fusion, resampler-free arm** — the same `VisualPatchProjector` payload
(pixel-shuffle r=2, EVS keep=98) appended after the series instead of spliced at fractional
positions inside the last context patch. `vision_chronos2.py:1390` shows `"late"` is hard-wired
to `CrossModalAdapter`; a `late_raw` branch is new code. n=3, same s1 warm start. This is the
single ablation that converts "s2d works" into "*where* vision enters decides whether it is
read". Ticket 28.

**G2 — zero trained component ablations on s2d.** All five A30 controls are eval-only or
probes. Every training-time component ablation in the registry (A17–A29, A18–A21, A24–A28)
is written against an s2c base (`sweep.manifest`) and is now out of scope. A reviewer's
"ablation study" means a table of s2d's own design choices, none of which has been varied at
training time:

| Variable | Why it must be trained, not eval-swept | Fixes |
|---|---|---|
| EVS keep-count q ∈ {0, 0.3, 0.7} retrained | A30-d's U-shape is a train/test length mismatch, not evidence | A30-d confound |
| Pixel-shuffle r=1 (14×14 native) vs r=2 (7×7) | A30-e's probe mean-pools; s2d concatenates — architecture-level test unrun | A30-e caveat |
| Single shared position vs fractional positions | A30-a's null is eval-only; the design choice itself is untested | A30-a interpretability |
| Stage-0 projector warmup on/off | Adopted from Nemotron, never ablated | — |
| Cell embedding, token-type / modality / segment embeddings (A24–A26 rebased) | never ablated on any arm | component table |

First three at n=3, rest n=1. Ticket 32.

**G3 — A23 (recipe-matched s2a).** s2a trains at `visual_dropout_prob` 0.3, s2d at 0.5. A01's
"late fusion is inert" null — which the paper leans on for H1 — is confounded with recipe
until A23 runs. Config ready, base s1, n=3, s2a cost. Ticket 33.

### 2.2 Baseline rows that are wrong or incomplete (no new models needed)

The suite's *coverage* is sufficient for the rebuttal matrix in `baselines.md` §2 (simpler
model, covariates-only, retrieval, other TSFM families, domain SOTA, pseudo-image
multimodal, tabular FMs are all present). What is not sufficient is the *state* of the rows:

| Row | Problem | Fix |
|---|---|---|
| CrossViViT, SUNSET, UniCast | aggregated over 15 plants incl. goes_pvdaq **train** site 1202; no protocol-aligned ramp column | re-score via `import_predictions.py --ukpv_dir --data` on the 14 plants |
| TS-RAG | aggregated over 19 plants (3 val + 2 train) | same |
| Time-VLM | **rank 2 in the current table**, SS carried from vendor eval, never re-scored into common format, no ramp | same — its 14 `time_vlm_*_pred.npz` exist |
| Aurora, Cross-RAG, VisionTS++ | no ramp column | same |
| PatchTST (P0) | seed 42 only | seeds 43, 44 |
| Chronos-2 FT | seeds 42, 43 only | seed 44 |
| Chronos-2 ZS | registry A00 says **missing**; report table shows a "zero-shot 0.4737" row — likely the oracle-covariate run mislabelled | verify provenance or run A00 |
| iTransformer_nf vs s2d ramp | 0.1445 vs 0.1440 — inside the 0.0011 floor, a **tie**, not the win the report's framing implies | state as tie |

Ramp is P0 and the paper's claim is "vision earns its tokens on ramps"; the multimodal
competitors are the rows where that claim is contested, and they are the rows without a
ramp number. Tickets 29, 30.

### 2.3 Rigor infrastructure the protocol promises and the repo lacks

| Requirement (`baselines.md` §4.5–4.6) | State |
|---|---|
| Diebold–Mariano on per-window losses + paired block bootstrap (day blocks, 1000×) + Holm–Bonferroni | **not implemented.** `runner.py` writes `_losses.npz` sidecars for tiers 0–3 only; MMTSFM writes per-plant `_pred.npz` and no loss sidecar; no DM/bootstrap script exists. The paper currently bolds on the seed-floor rule alone. |
| Efficiency table: trainable/total params, GPU-h, latency, peak VRAM | **not instrumented.** Design doc: "no measured wall-clock exists"; only `pipeline.py` counts params. s2d is 3.3× s2b's tokens — a reviewer will ask. |
| Missing-modality sweep p ∈ {0, .25, .5, 1} + low-history regime | not run on any arm; eval-only, `mask_visual` exists |
| Calibration | measured, not addressed. Sharpness bought the ramp gain; a post-hoc quantile recalibration on val plants is eval-only and would show whether coverage is recoverable at no ramp cost |

Tickets 31, 35, 37, 36.

### 2.4 The scope decisions only the user can make

**D1 — single dataset.** goes_pvdaq is ruled out of the thesis (no LOPO harness for MMTSFM,
no V-JEPA cache for it, 1 TB quota). "One dataset, 14 test plants" is the most common reject
reason at the target venues for a generalization claim. Either build it (real work: MMTSFM
LOPO loop + cache extraction + all P0 baselines under `--lopo-dataset goes_pvdaq`) or write the
paper as a single-testbed *mechanism* study with the limitation stated in the abstract. Ticket 34.

**D2 — the resampler foil.** Excluding s2b removes the only resampler-*interleaved* arm. With
it gone, "resampler-free" has no in-paper foil on the interleaved axis; the contrast that
remains is interleaved-raw (s2d) vs late-pooled (s2a) plus the new late-raw arm (G1). The
paper's framing must follow: the contribution is *placement* (where vision enters), and the
resampler removal is a design choice justified by the latent probe, not a measured ablation.
s2b_wide (n=3, on disk) could be readmitted as that foil without new compute. Ticket 27.

**D3 — encoder generality.** The claim is about "frozen vision-FM tokens"; only V-JEPA has
been tried. A second frozen encoder (Prithvi-EO-2.0 or DINOv2) through the same projector
would answer "is this V-JEPA-specific". Needs a new latent cache (quota). Left in fog.

## 3. Consequences of excluding s2b and s2c

- The A30 registry hypothesis ("recovers the ramp signal that **s2b's** pooled arm destroys")
  must be rewritten against s2a or against the late-raw arm.
- `report/report.typ` contains **no mention of s2d**; its Method, Results, and Conclusion are
  built around S2c as the flagship, its "open ablations" table lists A29/A17/A22, and its
  related work has no SolCAD-Net, PVNet or Nemotron positioning. Method + Results + Conclusion
  are a rewrite, not an edit. Ticket 38.
- D15 (s2c's horizon-attention diagnostic) was s2c's only mechanism story. s2d has none that
  survived. The paper claims the recovery, not the mechanism (ticket 26), and G2 is what could
  supply one.
- Every "attribution" ablation in the registry (A13, A17, A22, A29) is now provenance only.

## 4. Priority order (compute-cheapest decision first)

1. Ticket 29 — re-score multimodal rows on protocol (eval-only, minutes). Fixes contaminated
   and missing-ramp rows that the leaderboard table is built on.
2. Ticket 30 — seed-complete PatchTST / Chronos-2 FT, settle Chronos-2 ZS (cheap trains).
3. Ticket 31 — DM + bootstrap script (CPU, no GPU).
4. Ticket 33 — A23 (config ready, s2a cost, n=3).
5. Ticket 28 — late-raw arm (new code + 3 seeds at s2d cost). **Highest scientific value.**
6. Ticket 32 — s2d component ablations (largest compute block; EVS/resolution/position n=3
   first, rest n=1).
7. Tickets 36, 37, 35 — calibration, robustness, efficiency (eval-only or instrumentation).
8. Ticket 34 — second dataset: decide before 5–6 launch, because the answer changes what
   they are run on.
9. Ticket 38 — rewrite, blocked on the numbers above.

## 5. What is explicitly *not* needed

- No new baseline *models*. The tier structure already covers every rebuttal cell in
  `baselines.md` §2. SolCAD-Net (closest prior art) is closed-access with no code found —
  position, do not baseline (ticket 39 confirms).
- A03 Grassmann, A14 unfreeze, A07/A08/A15 retrieval — stay out (map Out of scope).
- S4 long-horizon and S5 data-efficiency scenarios — not load-bearing for a fusion-mechanism
  claim; state as future work.

## 6. Architecture-derived ablation audit (2026-09-07, independent of §2)

Re-derived from `configs/model/vision_chronos2_s2d.yaml`, `configs/stage/s2d.yaml`,
`patch_projector.py` and the `interleaved_raw` branch, without consulting §2 or the tickets.
Every design choice s2d makes, and whether anything on disk tests it:

| Design choice | Tested? | Ticket / ID |
|---|---|---|
| Future weather covariates known (`future_cov="all"`, incl. next-6h cloudcover + irradiance) | **no run at history-only on any MMTSFM arm** | 40 / A36 |
| Sequence-axis placement (vs late, same payload) | no | 28 |
| Pixel-shuffle concat vs average at 7×7 | probe only, model-free | 41 / A37 |
| EVS novelty rule vs random keep-98 | no | 42 / A38 |
| EVS keep-count, retrained | eval-only sweep, length-confounded (A30-d) | 32 / A31 |
| Fractional positions vs one shared position | eval-only shuffle (A30-a) | 32 / A33 |
| 3 of 6 encoder blocks trainable; s1 warm start is fully fine-tuned | no; **"frozen TSFM" wording is wrong** | 43 / A39 |
| r=1 (14×14) vs r=2 | probe only | 32 / A32 |
| Stage-0 warmup, cell embedding, projector depth, warm-start source, visual window, embeddings, numeric dropout | no | 32 (n=1 rows) |
| Vision-off pass preserves sequence length | unverified | 44 |
| Content grounding (stale sky, swap plant), frame order, marginal gain | **yes, n=3** | done (A30-a/b/c) |

Cannot be ablated, state as limitations: between-token interleaving (Chronos-2 patch 16 =
8 h, the whole visual window sits inside one patch); V-JEPA unfreeze (latent cache bypasses
the encoder); a resampler-interleaved foil (s2b excluded).

**Launch order** (cheapest decision first; each is a `task` ticket with its full spec):

1. Ticket 44 — sequence-length check (minutes, one seed). Gate for every Δ quoted below.
2. Ticket 42 A38-eval — random-98 on existing checkpoints (minutes). Decides whether A38-train runs.
3. Ticket 43 A39 — freeze 0 blocks, pure config, n=3.
4. Ticket 40 A36 — history-only covariates, s1 + s2d, n=3 each. Small datamodule plumb. **Highest risk to the headline.**
5. Ticket 28 — late-raw placement arm, n=3. New branch.
6. Ticket 41 A37 — avg-pool payload, n=3. One projector switch.
7. Ticket 32 — A31/A32/A33 at n=3, then the n=1 rows.

Must-have total ≈ 6 trained arms × 3 seeds ≈ 18 s2d-cost runs + 3 s1 runs; the two eval-only
items and the freeze arm need no new code.
