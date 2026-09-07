# 26 — Call s2d's gate: what all 5 controls add up to

Type: task
Status: resolved
Blocked by: 20, 21, 22, 23, 24, 25

## Question

All five planned A30 controls now exist. What does the full picture say about s2d's
ramp-NMAE win over s2c, and is it defensible enough to carry a paper claim?

## Answer

**Two axes, and they split cleanly.**

**Content grounding — strongly supported, no caveats.**
- A30-b (ticket 22, stale sky): costs +0.016–0.018 ramp NMAE, all 3 seeds, marginal gain
  flips negative.
- A30-c (ticket 25, swap plant): costs +0.016–0.019 ramp NMAE, all 3 seeds, −0.18 skill
  score, marginal gain flips harder negative than A30-b.

s2d needs the *right* plant's *current* sky. Both controls are clean, large, unanimous
across seeds. This part of the claim is as solid as anything on this map.

**Architectural mechanism — none of the three isolable variables show supporting
evidence.** Design §5.3 named summarizer removal + resolution + fractional positions +
EVS as the four things s2d moves at once relative to s2b, with the summarizer removal
itself not separately isolable and the other three meant to be checked here:
- A30-a (ticket 21, frame order / positions): near-null. Shuffling frame order costs ~0
  ramp NMAE, despite fractional RoPE positions making this a live test for the first time.
- A30-e (ticket 23, resolution): 4x4 beats s2d's shipped 7x7 on ramp R² at every horizon,
  mean-pooled. Caveated — the probe mean-pools, s2d concatenates — but it does not supply
  the supporting evidence hoped for.
- A30-d (ticket 24, EVS / motion-selectivity): confounded by train/test sequence-length
  mismatch, inconclusive rather than negative.

**Reading: s2d's ramp gain is real and grounded in genuine visual content (which plant,
how current), but the *specific architectural story* for why s2d beats s2c — finer native
resolution, frame-order-aware fractional positions, motion-selective pruning — has no
positive evidence on this map. Two mechanistic tests came back negative/unsupportive
(A30-a, A30-e) and one is uninterpretable as run (A30-d).**

**For the manuscript**: report the ramp-NMAE win (ticket 20) as SUPPORTED with the content
controls (A30-b, A30-c) as its evidence. Do NOT claim the design rationale (resolution,
frame order, motion selectivity) as the explanation — state plainly that the mechanism
by which s2d extracts more from the same sky than s2c is not established, only that it
does. This is a narrower, more honest claim than the design doc's original hypothesis
(§1: "recovers the ramp signal... removing the LatentSummarizer and feeding pixel-shuffled
patches... recovers it"). The recovery is real; the "how" is open.

**Costs still stand** (ticket 20): coverage_80 0.768→0.731, ECE 0.0280→0.0359, all 3
seeds, flat CRPS.

**s2d vs s2c in the writeup**: report both. s2d wins ramp NMAE (paired, floor-clearing,
all 3 seeds) at a calibration cost s2c doesn't pay, with weaker mechanistic support than
s2c's own headline (s2c has ticket-15's horizon-attention diagnostic as a genuine
mechanism story; s2d has none surviving). Neither supersedes the other outright — present
both with their respective evidence and let the ramp-NMAE number and the calibration cost
speak for themselves.
