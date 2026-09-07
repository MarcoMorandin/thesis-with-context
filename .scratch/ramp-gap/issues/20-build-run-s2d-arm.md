# 20 — Build and run the s2d arm (interleaved, no resampler)

Type: task
Status: resolved
Blocks: 21, 22

## Question

Does dropping `LatentSummarizer` and feeding pixel-shuffled V-JEPA patches through an MLP
projector at fractional temporal positions recover the ramp signal s2b's pooled arm loses?
Full design: [`knowledge/specs/2026-09-05-A30-s2d-design.md`](../../../knowledge/specs/2026-09-05-A30-s2d-design.md).
Registry row: `knowledge/ablations.md` §2.1, ID **A30**.

## Answer

Built and run n=3 (seeds 42/43/44), 2026-09-07 — `mmtsfm_s2d_ukpv_s2d_s{42,43,44}.json` in
`baselines/results/`. ⚠ Registered retroactively the same day (ran before the row existed —
AGENTS.md §4 violation, noted in the registry itself).

**Best arm on both P0 metrics**: SS 0.5510 ± 0.0020, ramp NMAE 0.1440 ± 0.0008 — best of
anything on disk, ahead of s2c (SS 0.5470 ± 0.0060, ramp 0.1461 ± 0.0020).

Paired against s2c per seed: ramp NMAE **−0.0020, all 3 seeds, clears the 0.0011 floor** —
**SUPPORTED**. Skill score **+0.0040 flips sign on s42** at a 0.0037 floor — tie, not
supported. Δ vision-off NMAE **drops** 0.0071→0.0047 across all 3 seeds: reliance
concentrates onto ramps, exactly as predicted.

**Cost**: coverage_80 degrades 0.768→0.731 (nominal 0.80), quantile ECE 0.0280→0.0359, both
in all 3 seeds, at flat CRPS. Sharpness bought the ramp gain — not yet stated anywhere else
in the paper trail.

**Not yet defensible**: zero positive controls existed at build time (design §5.3, registry
§2.2.3). See tickets 21/22 for the first two.
