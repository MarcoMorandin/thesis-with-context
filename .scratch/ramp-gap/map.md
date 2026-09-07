# Map: Closing the ramp gap

Label: `wayfinder:map`

## Destination

**s2d (interleaved, resampler-free vision fusion) is the publication's proposed
architecture.** s2b, s2b_wide, and s2c are discarded — kept in the codebase and in the
registry (`knowledge/ablations.md`) for provenance, removed from this map's forward path.
The Grassmann-vs-selfattn mixer question (A03) is cut outright: selfattn only, reported as
a stated limitation, decided 2026-09-08 without running the pre-registered gate.

The way is clear when s2d's own claim is thesis-grade evidence — Ch7–9 rewritten around
it, H1/H2 answered specifically for s2d, and nothing remains to *decide* before writing.
MMTSFM ahead of iTransformer on ramp NMAE (0.1481 → <0.1429) stays the stretch goal it
always was — a leaderboard win that rides on the same evidence, never a reason to reorder
the map.

## Notes

**Domain.** Cross-plant PV power forecasting, disjoint test plants. Read
[`knowledge/INDEX.md`](../../knowledge/INDEX.md) first; it routes to `scope.md` (research
question, hypothesis ladder), `protocol.md` (windows, metrics, splits), `architecture.md`
(what the model is), `dataset.md` (v2 non-HRV imagery facts), and
[`specs/2026-09-05-A30-s2d-design.md`](../../knowledge/specs/2026-09-05-A30-s2d-design.md)
(s2d's own design and open risks). Vocabulary comes from those files — `origin`, `plant`,
`skill score`, and `ramp` in the top-decile-|Δy| sense of `protocol.md` only.

**Skills every session should consult.** `grilling` and `domain-modeling` by default.
Tracker conventions: [`knowledge/agents/issue-tracker.md`](../../knowledge/agents/issue-tracker.md).

**This map plans; it does not build.** Exception: `task` tickets do real work, because a
decision here is blocked until a run finishes or a measurement exists.

### Standing decisions

1. **Two P0 metrics**: generalization skill score *and* ramp NMAE. Neither is subordinate.
   (settled 2026-08-25)
2. **The contribution is the fusion mechanism**, now specifically s2d's resampler-free
   interleaved path — selective spatial/temporal fusion over late fusion (H2, provisionally
   supported per ticket 10, not cleanly isolated from the resampler-removal confound).
   **Grassmann is cut**, not merely negotiable: selfattn only, everything on disk is
   selfattn, reported as a limitation. (settled 2026-08-25; Grassmann cut 2026-09-08 — see
   closed tickets 08, 09)
3. **Rigor bar is seeds**, n=3 per config. Controls are cheap and taken anyway, not traded
   against seeds. All five of s2d's planned controls (A30-a through A30-e) are on disk at
   n=3. (settled 2026-08-25)
4. **The manuscript is output, not a constraint.** Ch9's published follow-up list carries no
   authority here; it will be rewritten to match whatever this map finds. (settled
   2026-08-25)
5. **Deadline is soft.** The binding limit is the monthly local-h cap on `IscrC_MTSFM`, not
   any calendar date. (settled 2026-08-25)
6. **s2d is the flagship arm.** Superseded 2026-08-25's "all arms terminate at s2b" — s2d
   is a later, better-performing design (ramp NMAE beats s2c by −0.0020 paired, all 3
   seeds) that this map didn't originate but now adopts as the destination. (settled
   2026-09-08)

## Decisions so far

<!-- one line per closed ticket: gist + link. s2b/s2b_wide/s2c-specific entries moved to
     Out of scope below, 2026-09-08 — general infra and s2d-relevant decisions stay. -->

- [Store the vision-off pass so ramp can be decomposed](issues/02-store-vision-off-pass.md):
  ramp is now computed for both visual passes against a **single** set of per-site
  thresholds derived from the on pass; `delta_nmae_ramp` / `delta_nrmse_ramp` emitted at
  overall and per-plant. General infra — every s2d run uses this.

- [Was the V-JEPA cache built from v1 HRV or v2 non-HRV frames?](issues/12-vjepa-cache-imagery-provenance.md):
  **v2, confirmed** — cache postdates the imagery, records `data_v2`. All vision numbers on
  this map, including s2d's, stand on this cache.

- [Promote ramp to P0 and record the gate rule](issues/04-promote-ramp-to-p0.md): ramp is P0
  in `scope.md` beside generalization; the subset is defined in `protocol.md` §5. (The A03
  gate rule this ticket also pre-registered is now moot — see Standing decision 2.)

- [Fix the interleaved mask override before wave 1, or carry it?](issues/03-mask-override-before-or-after.md):
  **fixed before** — the interleaved path interleaves the context attention mask instead of
  sending all-ones. General fix to the interleaved-fusion family; s2d's `interleaved_raw`
  mode inherits the fixed behavior.

- [Widen the visual bottleneck](issues/13-widen-the-visual-bottleneck.md): the cause of the
  ramp/aggregate split is **pooling**, measured model-free — a pooled 1×1 feature is
  neutral-to-negative on ramps while spatial resolution helps monotonically, and
  `n_soft_tokens` was inert in interleaved fusion (the adapter fans out one pooled vector,
  adding no information). **This is s2d's origin story**: the resampler-free pixel-shuffle
  design exists because this ticket showed pooling, not architecture, was the ceiling.

- [Build and run the s2d arm (interleaved, no resampler)](issues/20-build-run-s2d-arm.md):
  built and run n=3 (registered retroactively in `knowledge/ablations.md` as **A30**,
  2026-09-07). Best arm on both P0 metrics: ramp NMAE beats s2c by −0.0020 all 3 seeds
  (paired, clears floor), skill score ties. Costs coverage_80 and ECE.

- [A09 frame-shuffle control on s2d](issues/21-a09-frame-shuffle-s2d.md): **unexpected
  near-null**, all 3 seeds — shuffling frame order costs s2d ~0 ramp NMAE, despite
  fractional RoPE positions making the control structurally live for the first time. s2d's
  gain does not come from reading temporal order.

- [A10b stale-sky control on s2d](issues/22-a10b-stale-sky-s2d.md): **strong non-null, all 3
  seeds** — staling the sky costs +0.016 to +0.018 ramp NMAE and flips the vision marginal
  gain negative. Kills "any recent sky would do".

- [A30-e: pooling probe re-run at GRID=7](issues/23-a30e-probe-grid7.md): **non-monotonic,
  4×4 beats 7×7 on ramp R² at every horizon** — s2d's shipped resolution shows no more
  mean-pooled ramp signal than 1×1. Caveat: the probe mean-pools, s2d's pixel-shuffle
  concatenates instead — doesn't falsify the architecture, but removes "resolution match"
  as a supporting story.

- [A30-d: EVS q-sweep on s2d](issues/24-a30d-evs-qsweep.md): **U-shaped, centered on the
  trained keep=98, not the monotonic trend predicted** — every deviation from the trained
  token count hurts, in both directions. Reads as train/test sequence-length mismatch, not
  motion-selectivity evidence.

- [A30-c: swap-plant control on s2d](issues/25-a30c-swap-plant-s2d.md): **strong non-null,
  all 3 seeds, bigger than A30-b** — wrong plant's sky costs +0.016–0.019 ramp NMAE and
  ~−0.18 skill score, marginal gain flips hard negative. Kills "the grid isn't spatially
  grounded" cleanly.

- [Call s2d's gate](issues/26-call-s2d-gate.md): all 5 controls in. **Content grounding**
  (A30-b, A30-c) strongly supported — s2d needs the right plant's current sky.
  **Architectural mechanism** (A30-a frame order, A30-e resolution, A30-d
  motion-selectivity) has no supporting evidence. Manuscript should claim the ramp-NMAE
  recovery as real without claiming the design doc's mechanistic story as its explanation.
  Its closing line ("present both s2d and s2c") is superseded by the 2026-09-08 destination
  redraw below — s2c is no longer being presented at all.

- **Destination narrowed to s2d** (2026-09-08) — the publication's architecture is s2d,
  full stop, not the outcome of a further s2b/s2b_wide/s2c comparison. Grassmann/A03 cut in
  the same pass (Standing decision 2). Tickets 06, 07, 08, 09, 14, 15, 16, 17, 18 and the
  "Wave 1 composition" plan closed or moved to Out of scope; ticket 10 (H1/H2 verdicts)
  rescoped to s2d's own evidence rather than closed, since it's still the map's core
  question.

- [What verdicts do H1 and H2 get?](issues/10-h1-h2-verdicts.md): called via `/grilling`,
  2026-09-08. **H1 split**: falsified for late fusion (s2a's vision pathway is inert, A01),
  supported and content-grounded for s2d (A30-b/c strong, A30-a null — depends on which
  plant and how current, not frame order). **H2 provisionally supported, not cleanly
  isolated**: s2d beats s2a by a wide margin (+0.0252 SS, −0.0047 ramp, both far past
  floor) but the comparison is confounded with the resampler removal; accepted as a stated
  Limitation rather than building an isolating control arm. The map's core question is
  closed — remaining work is writeup, not more ablations.

- [Verify the prior art the reviewers cited](issues/19-verify-prior-art.md): resolved via
  research subagent. **SolCAD-Net is real** (*Energy* 361 (2026), DOI
  `10.1016/j.energy.2026.141988`, verified against 3 independent sources) and pre-empts
  s2c's old cross-attention framing directly — but not s2d's actual mechanism. Recommended
  foil for related work: SolCAD-Net hard-codes advection architecturally, s2d's ablations
  show implicit content-grounded interleaving captures the ramp signal without that prior.
  Everything else (PVNet, Cloudcasting, pySTEPS, Prithvi, PV-VLM, KNMI benchmark) real but
  non-threatening, or not found (SolarSTEPS — cite Carpentieri et al. 2023 instead). Full
  writeup: [`knowledge/specs/2026-09-08-prior-art-verification.md`](../../knowledge/specs/2026-09-08-prior-art-verification.md).

- [How much visual headroom is left? (G0 ceiling probe)](issues/05-g0-ceiling-probe.md):
  **contradicts the expected decay** — aggregate `conditional_rel` rises 4.1%→12.0% over
  h1–h5 instead of decaying. Sharper finding: ramp-severity tiers diverge hard — vision
  actively **hurts** the mild-ramp tier (−16%→−33%), is null on mid, and only reliably
  helps the extreme-ramp tier (+2.2%→+7.0%, rising), which is the regime the flagship P0
  ramp metric already targets. Inconclusive-but-suggestive on headroom: still rising at h5,
  not flat-zero, but the h≥6 population break (known 13:30-origin dropout) blocks
  confirming further, and no matched-aggregation comparison to s2d's own realized gain
  exists yet.

## Not yet specified

- **Retarget the model to clear-sky index / auxiliary CSI loss.** Every external reviewer's
  top recommendation, and deliberately parked: `P = P_clear * CSI` assumes cloud is the
  only thing between irradiance and power, when soiling, clipping, curtailment, outages and
  shading all live in that gap. Revisit as a *multitask* auxiliary (`L_power + lambda *
  L_CSI`) on s2d's visual tokens rather than a target swap.
- **Explicit cloud-motion vectors as a model input.** The literature-standard answer, and
  the one that would convert this thesis from fusion-mechanism research into feature
  engineering. Kept in scope but deliberately behind s2d's own writeup.
- **Neighbour-plant csi as covariates.** Hypothesis-3 measurement found neighbour csi
  anomalies explain R² 0.18–0.28 of the future 30-min csi change — 4–6x the best vision
  probe. Caveat: using concurrent neighbour power changes the task from single-plant to
  networked forecasting, needs a protocol note, not a drop-in covariate.
- **Encoder domain gap.** Whether V-JEPA (natural video) should be swapped for an EO
  foundation model or LoRA-tuned. The latent probe partially exonerates it — structure
  signal is present when spatial layout is retained.
- **Weakened claim to repair in the writeup**: the 45-min frame-spacing falsification does
  not hold as stated. csi *level* autocorrelation of 0.78 says nothing about whether a fast
  *edge* is adequately sampled for motion estimation. Downgrade from "falsified" to
  "untested"; the right test is optical-flow endpoint error at 15/30/45/60-min separations.
- **Threshold-based ramp definition** (swinging-door or threshold-duration) reported
  alongside the top-decile subset, for comparability with the published literature.
- **EVS vs. actual cloud motion** — does the pruned-in token set (ticket 23's fixed probe
  showed 4×4 beats 7×7 on plain resolution, but EVS's own token *selection* is untested
  against ground-truth motion). Surfaced while closing ticket 18: repoint its optical-flow
  method at "does EVS keep the cells that moved" instead of a per-tau attention centroid
  that doesn't exist in s2d. Not yet a ticket — needs the method sharpened first.
- **Late-fusion, resampler-free control arm.** Surfaced by ticket 10's H2 answer: the
  +0.0252 SS / −0.0047 ramp margin over s2a is confounded with the resampler removal, and
  nothing on disk isolates fusion-mode from that. Would need a new curriculum arm (s2d's
  pixel-shuffle projector wired for late fusion instead of interleaved), ≥3 seeds — real new
  engineering, explicitly not attempted for this thesis (ticket 10's Limitations). Follow-up
  work, not currently a ticket.
- **How H1/H2 verdicts get presented.** Content is decided (ticket 10) — what remains is
  which chapter carries which claim, and how the H2 Limitations paragraph reads next to
  Ch9's existing follow-up list.
- **Whether V-JEPA should ever be unfrozen.** The latent cache bypasses the encoder, so
  unfreeze is dead code and V-JEPA has never been adapted to satellite imagery.
  Live-encoding is expensive; unknown whether it is worth a wave.
- **Re-run G0 with the h≥6 population handled cleanly, matched against s2d's realized
  gain.** Surfaced by ticket 05: the extreme-ramp ceiling is still rising at h5 with no
  matched comparison yet to what s2d actually extracts (Δ ramp NMAE 0.0063). Would need the
  13:30-origin population break resolved (separate fit per population, or drop the mixed
  regime) before a real headroom number exists. Not yet a ticket — needs the aggregation
  method specified first.
- **s2d follow-on wave**, if any — the ramp-weighted objective (loss is uniform pinball in
  `arcsinh` space; a ramp-weighted loss and/or ramp-based checkpoint selection carries an
  unresolved fairness question against tier-2 baselines on plain pinball), 12 future-query
  positions and 14×14 resolution as follow-ups (both were gated on s2c showing signal at 3
  positions / 4×4 — s2c is gone, so these need re-justifying against s2d's own mechanism
  before they're worth a ticket).

## Out of scope

- **s2b, s2b_wide, s2c, and the A03 Grassmann-vs-selfattn gate.** Ruled out 2026-09-08 —
  destination narrowed to s2d as the sole publication architecture. Kept in the codebase
  and in `knowledge/ablations.md` for provenance; removed from this map's forward path.
  Closed tickets: [08](issues/08-launch-wave-1.md) (wave-1 six chains, never launched),
  [09](issues/09-call-the-a03-gate.md) (A03 gate, decided without running it),
  [06](issues/06-shuffled-frames-control.md) (s2b frame-shuffle, subsumed by A30-a on s2d),
  [07](issues/07-rescore-s2b-ramp-marginal.md) (s2b ramp marginal, subsumed by s2d's native
  `compute_marginal_gain`), [14](issues/14-build-the-s2c-arm.md) (build s2c — real
  completed work, just off the forward path now), [15](issues/15-horizon-attention-diagnostics.md)
  (s2c's cross-attention diagnostic — mechanism doesn't exist in s2d),
  [16](issues/16-run-s2c-three-seeds.md) (s2c 3 seeds — completed, status-corrected, now
  moot), [17](issues/17-call-the-s2c-gate.md) (call s2c's gate — never called, moot),
  [18](issues/18-advection-alignment-of-attention.md) (per-tau attention vs cloud motion —
  s2d has no per-tau queries; a repointed version might return as a new ticket, see fog).
- **`goes_pvdaq` cross-dataset validation.** Ruled out 2026-08-25. The dataset is downloaded
  and a cross-dataset H2/H3 result would be the single biggest upgrade to the claim, but
  `protocol.md` §2 requires leave-one-plant-out for it and no LOPO harness exists. Single
  dataset, stated as a Limitation. Returns only as a fresh effort.
- **Retrieval on top of the fusion model.** Proposed in the current Ch9. A new contribution,
  not a ramp-gap fix; stays as future work.
- **Tuning s3 / full joint fine-tuning.** No curriculum stage goes past s2b/s2d-equivalent;
  s3 stands as a reported regression, not a stage to improve.
