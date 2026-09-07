# 22 — A10b stale-sky control on s2d

Type: task
Status: resolved
Blocked by: 20

## Question

s2c's strongest control: staling the sky by one horizon costs it +0.0201 NMAE and flips its
marginal gain negative. Does the same hold for s2d — direct comparison against s2c's
result?

## Answer

Run n=3 (seeds 42/43/44), 2026-09-07 — `mmtsfm_A10b_s2d_ukpv_s{42,43,44}.json`
(`eval_control: stale_sky`), compared per seed against the s2d baseline:

| seed | ramp NMAE base → staled | Δ ramp | SS base → staled | Δ SS | Δ NMAE (vision on−off) |
|---|---|--:|---|--:|--:|
| 42 | 0.14325 → 0.16108 | +0.0178 | 0.5500 → 0.4135 | −0.1365 | −0.0179 (flips negative) |
| 43 | 0.14482 → 0.16042 | +0.0156 | 0.5497 → 0.4140 | −0.1357 | −0.0177 (flips negative) |
| 44 | 0.14407 → 0.16228 | +0.0182 | 0.5532 → 0.4180 | −0.1352 | −0.0154 (flips negative) |

**Strong non-null, all 3 seeds, both P0 metrics blow past the floor by more than an order
of magnitude.** Same direction and similar magnitude to s2c's A10b (+0.0201 overall NMAE).
Marginal gain (vision-on minus vision-off) flips from positive to clearly negative in every
seed — under a stale sky, using the visual stream at all actively hurts, same pattern as
s2c.

**Kills the "any recent sky would do" rival explanation (A30-b) — SUPPORTED.** s2d's ramp
gain depends on the sky being genuinely current, not just present. This is the first
positive control on disk for the A30 claim (registry §2.2.3 previously listed zero).
