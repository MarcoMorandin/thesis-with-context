# 25 — A30-c: swap-plant control on s2d

Type: task
Status: resolved
Blocked by: 20

## Question

Design §3.5 / A10: is the 4x4/7x7 grid spatially grounded in the plant's own sky, or would
any plant's sky do? Direct test — score with `data.shuffle_test=true` so each sample sees a
different plant's frames, matched row-to-row.

## Answer

Run 2026-09-0x, n=3 (`mmtsfm_A10_s2d_ukpv_s{42,43,44}.json`), against the s2d baseline:

| seed | ramp NMAE base → swapped | Δ ramp | SS base → swapped | Δ SS | marginal gain (on−off) |
|---|---|--:|---|--:|--:|
| 42 | 0.14325 → 0.16110 | +0.0178 | 0.5500 → 0.3717 | −0.1783 | −0.0274 (flips negative) |
| 43 | 0.14482 → 0.16103 | +0.0163 | 0.5497 → 0.3660 | −0.1837 | −0.0282 (flips negative) |
| 44 | 0.14407 → 0.16303 | +0.0189 | 0.5532 → 0.3653 | −0.1879 | −0.0282 (flips negative) |

**Strong non-null, all 3 seeds, every metric — bigger than A30-b (stale sky) on skill
score.** Feeding the wrong plant's sky costs almost 0.18 skill score and flips the
vision marginal gain hard negative: with a mismatched sky, turning vision *off* beats
turning it *on* by ~0.028 NMAE, worse than the corresponding stale-sky flip in ticket 22.

**Kills "the grid isn't spatially grounded" (A30-c) — SUPPORTED, cleanly.** Combined with
A30-b, s2d demonstrably needs both the *right* plant's sky (this ticket) and a *current*
one (ticket 22). Two of five planned controls are now clean, strong positives.
