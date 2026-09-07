# 24 — A30-d: EVS q-sweep on s2d

Type: task
Status: resolved
Blocked by: 20

## Question

Design §3.5: does ramp NMAE improve as EVS pruning gets more aggressive (higher q, fewer
kept tokens)? If so, the signal is concentrated in the tokens that changed — moving cloud
edges — free motion-selectivity evidence.

## Answer

Run 2026-09-0x, 9 evals (`mmtsfm_A30d_q{0,59,137}_s2d_ukpv_s{42,43,44}.json`), against the
s2d baseline (q=0.5, keep=98 — the trained config, not re-run):

| keep (q) | ramp NMAE mean | Δ vs baseline | SS mean | Δ vs baseline |
|---|--:|--:|--:|--:|
| 196 (q=0)  | 0.1534 | +0.0102 | 0.4993 | −0.0470 |
| 137 (q=0.3)| 0.1462 | +0.0022 | 0.5375 | −0.0088 |
| **98 (q=0.5, trained)** | **0.1441** | — | **0.5510** | — |
| 59 (q=0.7) | 0.1512 | +0.0069 | 0.5279 | −0.0231 |

All 3 seeds agree on direction at every keep value. **U-shaped, centered on the trained
config — not the monotonic trend the design predicted.** q=0 (no pruning at all, sequence
239 tokens vs the trained 141) is the worst point by a wide margin; q=0.7 (keep=59, more
aggressive than trained) is also worse than baseline, not better.

**Read: this is a train/test sequence-length mismatch, not motion-selectivity evidence.**
None of these configs were retrained at their new keep count — EVS reorders which
(frame, cell) tokens survive and at what fractional-RoPE positions, so any keep value
other than 98 hands the model an input distribution it never saw in training. Degradation
scaling with distance from keep=98 in *both* directions is exactly what that predicts,
independent of whether "the tokens that changed" carry more signal. **A30-d does not
answer its registered question.** It does show the model is sharply specialized to its
trained token layout, which is at least evidence it isn't ignoring the visual tokens'
specific identity — weak, indirect support, not the clean test intended.

To answer the original question properly would need retraining at each keep value, out of
scope here.
