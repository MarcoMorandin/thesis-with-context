# 21 — A09 frame-shuffle control on s2d

Type: task
Status: resolved
Blocked by: 20

## Question

s2d is the design's claimed first arm where A09 is structurally live: visual tokens sit at
fractional RoPE positions, so a frame permutation changes the input (inert on s2b — no
positional encoding on the KV; proven no-op on s2c). Does shuffling frame order actually
cost s2d anything?

## Answer

Run n=3 (seeds 42/43/44), 2026-09-07 — `mmtsfm_A09_s2d_ukpv_s{42,43,44}.json`
(`eval_control: shuffle_frames`), compared per seed against the s2d baseline
(`mmtsfm_s2d_ukpv_s2d_s{42,43,44}.json`):

| seed | ramp NMAE base → shuffled | Δ ramp | SS base → shuffled | Δ SS |
|---|---|--:|---|--:|
| 42 | 0.14325 → 0.14324 | −0.00001 | 0.5500 → 0.5484 | −0.0016 |
| 43 | 0.14482 → 0.14490 | +0.00008 | 0.5497 → 0.5437 | −0.0060 |
| 44 | 0.14407 → 0.14378 | −0.00030 | 0.5532 → 0.5506 | −0.0027 |

**Unexpected: near-inert.** Δ ramp NMAE is ~0 in all 3 seeds, an order of magnitude under
the 0.0011 floor — the opposite of the design's prediction that s2d would be the first arm
where this control bites. Δ SS is small and inconsistent (s43 alone exceeds the 0.0037
floor; s42/s44 don't) — no seed-consistent signal either.

Structurally the control *is* live here (fractional positions do encode frame identity,
confirmed by inspection at design time) — this is a measured null, not a repeat of the
s2b/s2c structural no-op. Reading: s2d's ramp gain does not depend on frame **order**
specifically, even though it depends on **recent** frames existing at all (ticket 22). Downgrades
the "coordinate system" story in the design doc — the model may be using the spatial grid
as a static snapshot rather than reading temporal structure across the 4 latents.

**Does not by itself kill A30** — A30's claim is spatial resolution + summarizer removal,
not frame-order sensitivity — but it removes one piece of mechanistic evidence the design
doc expected to have in hand, and is worth a line in the manuscript rather than silence.
