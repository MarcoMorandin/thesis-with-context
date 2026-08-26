# 13 — Widen the visual bottleneck (`n_soft_tokens` 1 → N)

Type: task
Status: ready-for-human

## Question

Vision helps aggregate NMAE (13/14 plants, t=7.1, ~1.9 % rel) and does nothing on the ramp
subset (10/14, t=0.9, +0.39 % rel) — the one thing it was added for. This ticket asks
whether the cause is the compression ratio, and if so removes it.

## What the evidence says

Three competing explanations were measured first and **all three are falsified**. None of
them should be re-opened without new data.

- *The ramp subset is really sunrise/sunset, so vision cannot help there.* No. Decomposing
  `Δy = csi·Δcs + cs·Δcsi` over the exact subset `protocol_eval._ramp_masks` selects,
  geometry dominates 0.9 % of steps and carries 10.1 % of |Δy|; only 12.7 % sit above 70°
  zenith and the solar-time lift is mid-day-weighted (1.29× at 8–10 h, 0.35× at 14–16 h).
  The subset is ~90 % cloud-driven. **The metric is correctly aimed — do not change it.**
- *45-min frame spacing aliases the 30-min ramp.* No. csi level autocorrelation is ~0.78 at
  45 min. And the 30-min csi *change* has R² ≤ 0.015 against itself at **any** lag, slightly
  negative — ramps do not persist, so denser temporal sampling of the plant's own history
  has nothing to recover. `visual_frame_spacing_min` is the wrong dial.
- *The crop is too small to hold the incoming cloud.* No. Frame correlation vs plant
  separation is 0.85 at 20–40 km, 0.56 at 80–150 km → footprint ≈ 100 km, which spans the
  radius where the signal lives (neighbour-plant csi anomalies at the origin explain
  R² 0.18–0.28 of the future 30-min csi change).

**Pooling is what remains, and it reproduces the symptom.** Out-of-sample ridge probe on raw
128 px crops (t and t−45 min, matching the deployed `sp45` cache), predicting csi at t+h,
123 402 samples, fit on each plant's first 70 % of time and scored on the last 30 %:

| visual features | dim | ALL steps | RAMP steps | Δ ramp |
|---|---|---|---|---|
| none (csi persistence) | 4 | 0.3945 | 0.2833 | — |
| **1×1 pooled — the one-token arm** | 6 | 0.3994 | **0.2787** | **−0.0046** |
| 2×2 | 14 | 0.3996 | 0.2782 | −0.0052 |
| 4×4 | 38 | 0.4055 | 0.2897 | +0.0063 |
| 8×8 | 134 | 0.4069 | 0.2920 | +0.0087 |
| **16×16** | 518 | 0.4070 | **0.2983** | **+0.0150** |

(h = 60 min; same shape at 30 and 120 min.) On the aggregate a single pooled number captures
nearly all of the benefit. On ramps it is neutral-to-**negative**, and only spatial
resolution flips the sign — monotonically. That negative sign explains the 4/14 plants where
vision made ramps *worse*: a global brightness average says "cloudy" and pulls the forecast
toward the mean, which is exactly wrong during a transition, and its weight was learned on
the 90 % of steps that are not ramps.

Reproduce: `scripts/probes/pooling_bottleneck.py`.

## What is actually wired

The map's "Not yet specified" entry assumed the knob exists. It does not, in the path that
matters:

- `vision_chronos2.py:341-346` builds `CrossModalAdapter` **only** when
  `fusion_mode == "late"`. For `interleaved` it is `None`.
- The interleaved branch calls `latent_summarizer(..., T_ts=n_vis)` → `[B, n_vis, d_model]`,
  one token per visual step, and interleaves that directly. `n_soft_tokens` is never read.

So `n_soft_tokens` is **inert in s2b** — every ramp number on the map was produced at an
effective N=1 that no config could have changed. It is live only in late fusion (s2a).

A second, dormant bug sits in the late path: `vision_chronos2.py:556` flattens
`[B, T_ctx, N, d]` with `soft.reshape(B*N, T_ctx, d)`, which matches the `b*N+n` layout its
consumers assume (`repeat_interleave(N_soft)` at 1063/1094/1099) only when `N == 1`. At
N > 1 it scrambles tokens across time. Never fired, because N has always been 1.

## Decision

Wire `n_soft_tokens` through the interleaved path and fix the late-path flatten, so the
ablation the map already wants — `{1, 4, 16, 64}` per the `VisionConfig` docstring — becomes
runnable. Keep N=1 bit-identical so every existing checkpoint and result stands.

## Done when

- [x] `n_soft_tokens > 1` produces `n_vis · N` visual tokens per step in interleaved fusion
- [x] N=1 is bit-identical to the current code, and existing checkpoints load unchanged
- [x] late-path flatten ordering fixed and pinned by a test
- [x] attention mask, position IDs and modality mask all correct at N > 1
- [x] config for the ablation arm

## Resolution

`CrossModalAdapter` is now built whenever it does something — late fusion, or any
`n_soft_tokens > 1` — and the interleaved branch expands the summary through it, giving each
refined step a block of `[ts, v¹..v^N]`. `interleave_sequences`,
`build_interleaved_position_ids`, the interleaved attention mask, the modality mask and the
covariate rows all carry `N_soft`. A step and all of its visual tokens share one position ID,
so RoPE sees the block as co-temporal and the N tokens are order-free within it. The output
side is untouched — it slices `[-T_fut:]`, which does not depend on refinement length.

At N=1 the adapter is not constructed, so the parameter set and the emitted sequence are
unchanged and every curriculum checkpoint still loads. That is asserted, not assumed:
`tests/test_visual_bottleneck_width.py` writes the pre-change interleave out literally and
requires equality, and checks the N=1 `state_dict` has no adapter keys while N=16's is a
superset.

The late-path flatten bug is fixed with a `permute(0, 2, 1, 3)` before the reshape.

Cost: `n_vis · (N−1)` extra encoder positions. At the uk_pv geometry (`n_vis=1`, ~44-token
sequence) N=16 adds 15 tokens, ~34 %. Cheap; the visual branch is not the throughput limit.

**Not yet run.** `configs/model/vision_chronos2_wide.yaml` is `timeselfattn` plus
`n_soft_tokens: 16` and nothing else, so the pair reads as a clean A/B. Wave 2 should carry
N ∈ {1, 16} at three seeds, on the self-attention mixer — grassmann trails it by 0.014 SS on
every seed (ticket A03), so pairing the widening with the losing mixer would confound.

Magnitude expectation, stated before the run so it can be wrong: the probe's linear arm
recovers ~1 % error of the 4 % ramp gap to iTransformer. It bounds the *mechanism*, not the
architecture — a learned N-token adapter over V-JEPA patches has more to work with than
ridge over 16×16 block means — but nothing here promises the gap closes.
