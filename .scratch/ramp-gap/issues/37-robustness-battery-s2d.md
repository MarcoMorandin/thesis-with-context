# 37 — Robustness battery on s2d (missing frames, low history)

Type: task
Status: open

## Question

`baselines.md` §5 lists a missing-modality sweep (drop frames at p ∈ {0, .25, .5, 1.0} via
`mask_visual`, plus a stale-frames variant) and a low-history regime (T ∈ {4, 8, 12, 24}
steps). Neither has run on any arm. Both are eval-only on the three s2d checkpoints. s2d
trains with `visual_dropout_prob=0.5`, so graceful decay is expected; the p=1.0 point must
land on s1's numbers, not below (that is a falsifiable check on the marginal-gain
instrument). A30-b already supplies the stale-frames point.

Resolved when the two curves (SS and ramp NMAE vs p; vs T) are recorded here for 3 seeds and
the p=1.0 ≈ s1 check is stated pass/fail.

Context: `paper-readiness-audit.md` §2.3.
