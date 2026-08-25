# 06 — Does the model read cloud motion, or time of day?

Type: task
Status: needs-triage
Blocked by: 02

## Question

Vision measurably helps (+2.7% NMAE, 14/14 plants). Nothing yet shows it helps *for the
right reason*. A satellite crop encodes scene brightness, which encodes solar elevation,
which encodes time of day — and a model that learned only that would produce exactly the
result observed.

The control: take the s2b weights, apply a random temporal permutation to the visual window
**at evaluation only**, re-score. No retraining, one evaluation pass.

- Score drops toward the vision-off number → the model reads temporal structure across
  frames. H1 becomes a claim about cloud dynamics.
- Score unchanged → the visual token carries a static, order-free summary. Every wave-2
  visual intervention changes meaning, and the honest description of the visual branch
  becomes "a per-window scene descriptor", not "cloud advection".

Run the mismatched-plant variant too if it is cheap — swapping in another plant's frames
tests spatial grounding the same way permutation tests temporal grounding. Both are
registered as A09/A10 in `knowledge/ablations.md` and both are still TODO.

This is not traded against seeds; it is one eval pass and it changes what the seeded numbers
*mean*. Blocked only by 02, because the comparison worth reporting is the shuffled-frames
delta on **ramp**, which does not exist until both passes are stored.
