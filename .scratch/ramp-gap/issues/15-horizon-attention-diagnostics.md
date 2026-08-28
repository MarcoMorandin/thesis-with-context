# 15 — Prove the horizon queries actually behave differently

Type: task
Status: done
Blocked by: 14
Blocks: 16

## Question

Instrument s2c so that a result — positive or null — is *mechanistically* interpretable
rather than just a number. This is not debug logging; it is the evidence that turns an
architecture experiment into a mechanistic finding.

## Why this is not optional

The ideal outcome of s2c is not "lower ramp NMAE". It is:

> lower short-horizon ramp NMAE, **and** the +30 / +120 / +240 min queries attend to
> different spatial regions consistent with an evolving cloud field.

Without the second half, a win is unexplained and a null is unattributable. Three queries
that collapse to near-duplicates would produce the same null as a genuinely failed
hypothesis, and we would not be able to tell which happened.

## Diagnostic hierarchy

**Primary — attention distributions over the 4x4x4 field, quantified.** Not "the heatmaps
look different". Report a divergence between the per-tau attention distributions:
`D_KL(attn_30 || attn_120)`, `D_KL(attn_120 || attn_240)`, or a normalised L1 / cosine
distance. This is the behaviour that matters, because it is what the model *does*.

**Secondary — pairwise distance between the learned tau embeddings.** Weaker evidence in
both directions: large distances do not prove the model uses them differently, and similar
embeddings do not prove the attention behaviour is identical. Report it, do not lead with it.

## Done when

- [x] per-tau attention maps over the 64 KV tokens are captured at eval time
- [x] a quantified divergence between the three distributions is emitted to the results JSON
- [x] pairwise tau-embedding distances are emitted alongside
- [x] a degenerate case (three identical queries) is distinguishable from a genuine null in
      the recorded output

## Resolution

`results/*.json` for any arm with cross-attention blocks now carries a
`horizon_attention` object. Four verdict strings, and the one that matters is
`degenerate_queries_collapsed` — it is the outcome the ticket exists to make sayable.

**The deciding statistic is a ratio against a measured floor, not a constant.** Comparing
the per-tau attention distributions is easy; knowing whether the difference is *bigger than
sampling noise* is the hard part, and a hand-picked divergence threshold would have been a
number chosen to be passed. Instead the same L1 is computed WITHIN one tau, between two
seeded random half-splits of its own eval rows. That is the noise floor, measured on the
same data at the same sample size, and `separation_ratio = between / within` needs no magic
constant: 1.0 is indistinguishable from noise.

Second gate, `MIN_BETWEEN_L1 = 0.05`. The ratio alone is not enough, because a large enough
epoch shrinks the floor until a shift of a fraction of one KV token passes it. That failure
mode is not hypothetical — `test_a_tiny_shift_is_not_promoted_by_a_large_sample` builds
exactly it (centres 32 / 32.1 / 32.2 over 4000 rows), confirms the ratio *does* pass, and
shows the L1 floor is what vetoes. Both thresholds are pre-registered here, before any s2c
checkpoint exists, for the same reason 0.0011 was in ticket 14.

Half-splits are assigned **per row**, seeded, not per batch: the test loader is not
shuffled, so a batch-parity split would tie the two halves to site order and understate the
floor. `test_the_half_split_ignores_batch_boundaries` pins that by streaming identical rows
at chunk 1 and chunk 137.

Plumbing, three pieces:

- `Chronos2EncoderBlock.capture_visual_attn` / `.last_visual_attn` — plain attributes, not
  buffers, so an s2c checkpoint stays loadable by an arm that never runs the diagnostic.
  Armed in `on_test_start` only: capture forces the eager attention path, which has no
  business running during training.
- `_capture_horizon_attention` runs after the **vision-ON** pass only, and takes target rows
  with active vision at future positions only. Covariate rows live past index B and vision-off
  rows gate no residual; folding either in would pollute the distribution with attention that
  never reached the forecast. Maps are cleared after each harvest so a missed call fails as
  `None` rather than silently double-counting.
- `ProtocolEvaluator.extra`, merged **inside** `finalize()` with `setdefault`. `write()` calls
  `finalize()` a second time internally, so anything added to the dict `finalize()` returned
  never reaches disk — the trap `test_extra_diagnostics_survive_the_double_finalize` exists to
  hold shut. `setdefault` also means a diagnostic can never overwrite a protocol metric.

Absent vs. silent is a real distinction and both are recorded: an arm with no cross-attention
blocks emits **no key at all** (s1/s2a/s2b JSONs stay byte-identical), while an arm that has
the machinery and captured nothing emits `verdict: not_measured` — a bug report, not a null.

16 tests. Full suite green.

**Read the s2c result in two dimensions, not one.** Ramp NMAE alone cannot falsify the
hypothesis: a flat metric with `degenerate_queries_collapsed` means the parameterisation
collapsed and the experiment did not test advection at all. Only a flat metric with
`queries_differ` is evidence against horizon-specific spatial attention.
