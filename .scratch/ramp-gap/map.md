# Map: Closing the ramp gap

Label: `wayfinder:map`

## Destination

A defensible verdict on H1 and H2 backed by seeded runs — and, if the evidence permits,
MMTSFM ahead of iTransformer on ramp NMAE (0.1481 → <0.1429) on the aligned uk_pv
cross-plant protocol. The deliverable is thesis-grade evidence: rewritten Ch7–Ch9 whose
claims survive a viva. A leaderboard win is a stretch that rides on the same runs, never
a reason to reorder the map.

The way is clear when the mixer question is settled, H1/H2 carry supported/falsified
verdicts against a measured noise floor, and nothing remains to *decide* before writing.

## Notes

**Domain.** Cross-plant PV power forecasting, disjoint test plants. Read
[`knowledge/INDEX.md`](../../knowledge/INDEX.md) first; it routes to `scope.md` (research
question, hypothesis ladder), `protocol.md` (windows, metrics, splits), `architecture.md`
(what the model is), `dataset.md` (v2 non-HRV imagery facts). Vocabulary comes from those
files — `origin`, `plant`, `skill score`, and `ramp` in the top-decile-|Δy| sense of
`protocol.md` only.

**Skills every session should consult.** `grilling` and `domain-modeling` by default.
Tracker conventions: [`knowledge/agents/issue-tracker.md`](../../knowledge/agents/issue-tracker.md).

**This map plans; it does not build.** Exception: `task` tickets do real work, because a
decision here is blocked until a run finishes or a measurement exists.

### Standing decisions (settled while charting, 2026-08-25)

1. **Two P0 metrics**: generalization skill score *and* ramp NMAE. Neither is subordinate.
2. **The contribution is the fusion mechanism** — selective temporal interleaving over late
   fusion (H2). It is independent of which operator sits in `layer[0]`, so it survives a
   mixer swap intact. The Grassmann operator is *negotiable*.
3. **The A03 gate rule**, fixed before the data exists:
   > Swap the mixer **iff** self-attention's ramp NMAE beats Grassmann's by more than the
   > seed floor, **and** its skill score does not regress by more than the seed floor.
   > Ramp win + skill-score win → swap. Ramp win + flat skill score → swap.
   > Ramp win bought by a skill-score collapse → do not swap.
   >
   > **seed floor = max(** the MMTSFM 3-seed sd for that config, **2 ×** iTransformer's
   > 3-seed sd **)**, the latter being 0.0011 on ramp NMAE and 0.0037 on skill score.
   > The borrowed lower bound stops a tight sample from making a null look decisive.
4. **All arms terminate at s2b.** s3 regressed on the flagship (SS 0.5257 vs 0.5284, ramp
   0.1509 vs 0.1481, ECE 0.0325 vs 0.0278) and is reported as a finding about frozen-backbone
   adaptation, not re-collected per arm.
5. **Rigor bar is seeds**, n=3 per config. Controls are cheap and taken anyway, not traded
   against seeds.
6. **The manuscript is output, not a constraint.** Ch9's published follow-up list carries no
   authority here; it will be rewritten to match whatever this map finds.
7. **Deadline is soft.** 10 Sep is a supervisor checkpoint. The binding limit is the 6,545
   local-h monthly cap on `IscrC_MTSFM` (45,739 h balance, account ends 2026-12-02).

### Wave 1 composition

**Six** curriculum chains, each s1 → s2a → s2b (`END_STAGE=s2b`, `MARGINAL_GAIN=1`):
`grassmann@{42,43,44}` + `selfattn@{42,43,44}`. Six, not five: the interleaved
attention-mask fix means the existing `mmtsfm_s2b_ukpv.json` was trained with the bug and
`grassmann@42` must be rebuilt. Move the pre-fix file aside first — the rebuilt run claims
the same bare tag. Chains are independent single-GPU sbatch sequences wired with `afterok`;
parallel width is a queue question, not a node question.

## Decisions so far

<!-- one line per closed ticket: gist + link. -->

- [Store the vision-off pass so ramp can be decomposed](issues/02-store-vision-off-pass.md):
  ramp is now computed for both visual passes against a **single** set of per-site
  thresholds derived from the on pass; `delta_nmae_ramp` / `delta_nrmse_ramp` emitted at
  overall and per-plant, `pred_off` npz dumped for `localize`. Default path unchanged.

- [Make the curriculum runner wave-safe](issues/11-make-curriculum-runner-wave-safe.md):
  arm identity (variant + seed) now reaches tag, checkpoint dir and warm-start path;
  `END_STAGE` stops the chain at s2b; `MARGINAL_GAIN=1` plumbed; `DRY_RUN=1` on both
  scripts. Canonical `grassmann@42` keeps `mmtsfm_s2b_ukpv`.

- [Was the V-JEPA cache built from v1 HRV or v2 non-HRV frames?](issues/12-vjepa-cache-imagery-provenance.md):
  **v2, confirmed** — cache postdates the imagery, records `data_v2`, and the s2b run
  consumed it. All existing vision numbers stand. The stale v1 cache still exists at the
  old default path, so the runner now defaults to the non-HRV cache and treats an absent
  or provenance-less cache as fatal.

- [Promote ramp to P0 and record the gate rule](issues/04-promote-ramp-to-p0.md): ramp is P0
  in `scope.md` beside generalization; the subset is defined in `protocol.md` §5 as
  implemented; the A03 gate rule is pre-registered in new §5.1. Stale H1 verdict and the
  v1 `data/` path corrected across five files.

- [Fix the interleaved mask override before wave 1, or carry it?](issues/03-mask-override-before-or-after.md):
  **fixed before** — the interleaved path now interleaves the context attention mask instead
  of sending all-ones. `grassmann@42` must be rebuilt, so wave 1 is six chains. An
  output-level test cannot catch this bug (the mask is also a patch feature); the guard is a
  spy on the encoder boundary.

- **s2c design settled** (grilling, 2026-08-28) — external review rejected the summarizer
  rewrite: N Perceiver queries give N *content summaries*, not a *coordinate system*, and
  advection is a claim about coordinates. The arm instead retains a 4x4x4 V-JEPA spatial
  field as 64 KV tokens and lets **3 future positions** (`output_patch_size` 16 -> 4), each
  carrying a learned lead-time embedding, cross-attend it via the already-present but unused
  `TimeCrossAttention`, residual gated to future positions only. Freeze policy mirrors s2b
  exactly. Full spec and rationale in
  [Build the s2c arm](issues/14-build-the-s2c-arm.md); gate pre-registered in
  [Call the pre-registered s2c gate](issues/17-call-the-s2c-gate.md).

- [Widen the visual bottleneck](issues/13-widen-the-visual-bottleneck.md): the cause of the
  ramp/aggregate split is **pooling**, measured model-free. Three rivals falsified first: the
  ramp subset is ~90 % cloud-driven (metric is right), 30-min ramps have R² ≤ 0.015 against
  themselves at any lag (frame spacing is the wrong dial), and the crop already spans ~100 km.
  A pooled 1×1 feature is neutral-to-**negative** on ramps while spatial resolution helps
  monotonically. `n_soft_tokens` turned out to be **inert in interleaved fusion** — the
  adapter was built only for late fusion, so every ramp number on this map was produced at an
  effective N=1 no config could change. Now wired, N=1 bit-identical, plus a dormant
  late-path flatten bug fixed. `vision_chronos2_wide.yaml` is ready to run.

- [Does wave 1 fit the monthly cap?](issues/01-leonardo-billing-conversion.md): yes — ~370
  GPU-h for six chains against 5,202 h left in August plus a fresh 6,545 on 1 September.
  Compute is not the binding constraint; the local-h conversion stays unmeasured until a
  wave is large enough for it to matter.

## Not yet specified

- **Retarget the model to clear-sky index.** Every external reviewer's top recommendation,
  and deliberately parked: `P = P_clear * CSI` assumes cloud is the only thing between
  irradiance and power, when soiling, clipping, curtailment, outages and shading all live in
  that gap. Revisit as a *multitask* auxiliary (`L_power + lambda * L_CSI`) rather than a
  target swap, after s2c reports.
- **Explicit cloud-motion vectors as a model input** (not merely as the measuring instrument
  of ticket 18). The literature-standard answer, and the one that would convert this thesis
  from fusion-mechanism research into feature engineering. Kept in scope but deliberately
  behind s2c.
- **Neighbour-plant csi as covariates.** Our own hypothesis-3 measurement found neighbour
  csi anomalies explain R^2 0.18-0.28 of the future 30-min csi change — 4-6x the best vision
  probe. Caveat nobody raised: using concurrent neighbour power changes the task from
  single-plant to networked forecasting, so it cannot be dropped in as "one more covariate"
  without a protocol note.
- **Encoder domain gap.** Whether V-JEPA (natural video) should be swapped for an EO
  foundation model or LoRA-tuned. The latent probe partially exonerates it — the structure
  signal is present when spatial layout is retained — so this sits below the fusion fix.
- **Weakened claim to repair in the writeup**: the 45-min frame-spacing falsification does
  not hold as stated. csi *level* autocorrelation of 0.78 says nothing about whether a fast
  *edge* is adequately sampled for motion estimation. Downgrade from "falsified" to
  "untested"; the right test is optical-flow endpoint error at 15/30/45/60-min separations.
- **Threshold-based ramp definition** (swinging-door or threshold-duration) reported
  alongside the top-decile subset, for comparability with the published literature.
- **12 future query positions** (`output_patch_size=1`) and **14x14 spatial resolution**,
  both follow-ups gated on s2c at 3 positions / 4x4 showing signal.

- **Wave 2 composition.** Hangs entirely on the gate outcome and on G0. If the gate swaps the
  mixer, wave 2 is a re-baselined curriculum; if it holds, wave 2 is the objective work.
- **The ramp-weighted objective.** Loss is uniform pinball in `arcsinh` space (sublinear, so
  it compresses exactly the large deviations ramp scores), selection is on `val/loss`. A
  ramp-weighted loss and/or ramp-based checkpoint selection is in scope for wave 2. Carries an
  unresolved fairness question: tier-2 baselines optimise plain pinball, so a ramp-weighted
  MMTSFM beating them needs either a ramp-weighted baseline or an explicit note in
  `protocol.md`.
- **Visual-branch interventions.** Widening the bottleneck is no longer gated on G0 and no
  longer unspecified — ticket 13 measured the mechanism from the data side and shipped the
  code; what remains is the run (N ∈ {1,16}, three seeds, self-attention mixer). Still open
  and still gated: an auxiliary clear-sky-index loss on the visual tokens, and injecting
  visual representations at the *future* positions rather than the context. Note ticket 13
  retires the stated rationale for the latter — frames do decorrelate in ~2 h, but the ramp
  itself has no temporal persistence to exploit at ANY lag, so a future-position argument has
  to rest on spatial advection, not on frame recency.
- **Whether a mixer swap needs a fresh curriculum** or can warm-start from existing weights.
- **How H1/H2 verdicts get presented** once measured — which chapter carries which claim.
- **Whether V-JEPA should ever be unfrozen.** The latent cache bypasses the encoder, so the
  curriculum's unfreeze is dead code and V-JEPA has never been adapted to satellite imagery.
  Live-encoding is expensive; unknown whether it is worth a wave.

## Out of scope

- **`goes_pvdaq` cross-dataset validation.** Ruled out 2026-08-25. The dataset is downloaded
  and a cross-dataset H2/H3 result would be the single biggest upgrade to the claim, but
  `protocol.md` §2 requires leave-one-plant-out for it and no LOPO harness exists. Single
  dataset, stated as a Limitation. Returns only as a fresh effort.
- **Retrieval on top of the fusion model.** Proposed in the current Ch9. A new contribution,
  not a ramp-gap fix; stays as future work.
- **Tuning s3 / full joint fine-tuning.** The curriculum terminates at s2b; s3 stands as a
  reported regression, not a stage to improve.
