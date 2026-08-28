# 15 — Prove the horizon queries actually behave differently

Type: task
Status: blocked
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

- [ ] per-tau attention maps over the 64 KV tokens are captured at eval time
- [ ] a quantified divergence between the three distributions is emitted to the results JSON
- [ ] pairwise tau-embedding distances are emitted alongside
- [ ] a degenerate case (three identical queries) is distinguishable from a genuine null in
      the recorded output
