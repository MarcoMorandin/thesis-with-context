# 10 — What verdicts do H1 and H2 get?

Type: grilling
Status: resolved

## Rescoped 2026-09-08

Destination narrowed to s2d as the publication's architecture. Its blockers (06, 07, 08)
are closed — out of scope, all three were s2b-specific and s2b is discarded. This ticket is
unblocked, but the evidence underneath it changed and the text below still reads for s2b.
Before this is called, replace the s2b evidence with s2d's:

- **H1** now has *better* evidence than the shuffled-frames control it was waiting on:
  `ablations.md` A30-b (stale sky, ticket 22) and A30-c (swap plant, ticket 25) are both
  clean, strong, all-3-seed positives — s2d needs the right plant's *current* sky, not
  merely "some vision helps in aggregate". A30-a (frame shuffle, ticket 21) came back a
  near-null, which narrows H1's wording: content-grounded, not order-sensitive.
- **H2** ("selective interleaving beats late fusion") is *not* cleanly answered for s2d.
  s2a is the only late-fusion arm on disk, and it used the old resampler-pooled
  architecture — comparing it against s2d confounds fusion-mode with the resampler
  removal, the same "moves 4 variables at once" problem the design doc flagged (design
  §5.3). No late-fusion, resampler-free control exists to isolate that. Decide: accept the
  confound and state it as a limitation, or scope a new control arm.
- s2a's own status is undecided by this redraw — the user asked to discard s2b/s2b_wide/s2c
  specifically, not s2a. It's still the late-fusion baseline H2 needs, confound and all.

## Question (original, s2b-framed — needs a rewrite pass before calling)

`knowledge/scope.md`'s ladder and the drafted Ch9 both leave H1 and H2 **Open**, on the
stated grounds that the model's arms had not completed. They have. This ticket closes both,
on evidence, at n=3.

**H1 — "a visual stream improves over an identical time-series-only model."** The honest
verdict is split and should be written as such:

- *Falsified for late fusion.* s2a (0.5086) is indistinguishable from s1 (0.5087).
- *Supported inside interleaved fusion.* The forced vision-off pass on s2b's own weights
  gives +2.7% NMAE / +2.1% NRMSE, positive on 14/14 plants.

What is not yet settled, and what 06 and 07 supply: whether that gain survives the
shuffled-frames control, whether it lands on **ramp** at all, and whether it is
horizon-local as the ~2 h decorrelation predicts. Decide how much of H1 each of those
licenses, and what the claim reduces to if the control comes back flat.

**H2 — "selective interleaving improves over late fusion."** s2b (0.5284) over s2a (0.5086)
at n=1 each; wave 1 supplies the seed floor that makes it a verdict rather than an
observation. Note the confound from 03: s2b changed the mask handling at the same time it
changed the fusion mode, so decide explicitly whether the H2 claim rests on the seeded delta
alone or needs the mask-clean rebuild.

Output is the sentence each hypothesis gets in the rewritten conclusions, plus the table row
in the verdict table, plus what goes in Limitations for whichever part the evidence does not
reach. Standing decision 2 says the fusion mechanism is the contribution, so H2's wording
carries the most weight in the thesis and deserves the most care here.

## Answer

Called via `/grilling`, 2026-09-08. Two structural decisions, both confirmed by the user:
keep H1 as a split verdict (late fusion vs s2d) rather than a single s2d-only statement,
and accept H2's confound as a stated Limitation rather than scoping a new control arm.

**H1 — split by fusion mode, not settled uniformly.**

*Falsified for late fusion:* s2a's skill-score gain over s1 survives forcing vision off
(Δ NMAE 0.0008 ± 0.0005, Δ ramp 0.0000 ± 0.0015 — both at the seed floor). The gain is a
recipe effect, not a visual one (A01).

*Supported, content-grounded, for s2d:* forcing vision off costs 0.0047 ± 0.0001 NMAE and
0.0063 ± 0.0006 ramp NMAE, both clearing the seed floor in all 3 seeds. Two controls
establish genuine content dependence, not an artifact — swapping in another plant's sky
(A30-c) or staling the current plant's sky by one horizon (A30-b) each cost 0.016–0.019
ramp NMAE and flip the marginal gain negative, all 3 seeds. A third control narrows what
"content" means: shuffling frame order (A30-a) costs ~0 ramp NMAE — s2d's gain depends on
*which* plant and *how current* the sky is, not on reading temporal order across the 4
frames.

**H2 — provisionally supported, not cleanly isolated.**

s2d beats s2a by a wide margin on both P0 metrics (SS 0.5510 ± 0.0020 vs 0.5258 ± 0.0043,
+0.0252, ~7× floor; ramp NMAE 0.1440 ± 0.0008 vs 0.1487 ± 0.0010, −0.0047, ~4× floor).
Confounded — s2a is the old resampler-pooled architecture, s2d also removes the resampler
and changes spatial resolution, so the margin can't be attributed to fusion mode alone.

What the confound doesn't touch: the arms differ *structurally*, not just numerically. Late
fusion's own vision pathway is inert (A01) — it ignores the visual stream regardless of
content. s2d's fusion mode is demonstrably content-grounded (A30-b, A30-c). One arm
structurally blind to vision, the other structurally dependent on the right plant's current
sky — that contrast isn't confounded, even though the numeric margin is.

**Limitations:** the +0.0252/−0.0047 margin over s2a is corroborating, not decisive — it
rests on a comparison across two architectures differing in more than fusion mode. No
late-fusion, resampler-free control was built to separate the effects; noted as a direction
for follow-up work, not attempted here.

