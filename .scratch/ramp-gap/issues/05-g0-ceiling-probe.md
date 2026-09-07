# 05 — How much visual headroom is left? (G0 ceiling probe)

Type: task
Status: resolved

## Question

`MMTSFM/scripts/probes/fit_ceiling.py` and `g0_ceiling.sbatch` exist and have never been run
to completion. The probe fits three predictor sets on cached V-JEPA latents — (a) visual
only, (b) exactly what the model already gets, (c) both — and reports `conditional_rel`,
the fraction of the covariates-only error that vision removes, **per horizon step**.

It answers the question no training run can: not "does vision help" (settled — +2.7% NMAE on
14/14 plants, 4–5σ) but **how much of the available signal the current architecture is
leaving on the table**. That number decides whether the wave-2 visual interventions
(widening the 1000:1 bottleneck, auxiliary loss, future-position injection) are worth a
curriculum, or whether the model is already near the ceiling and the remaining gap is
information-theoretic.

Free: CPU only, `lrd_all_serial`, no GPU budget, no model forward. Runs in parallel with
everything else on this map.

Two things to read carefully in the output. First, `alpha_selected` — a visual penalty
pinned at the grid maximum means "vision off won", and a covariate penalty at either edge
means the grid was too narrow and the run is not trustworthy. Second, the horizon boundary:
`n_test_valid` halves at h=6 because the 13:30 origin contributes zero scored steps there,
so h≤5 and h≥6 are measured on different populations and must be read separately, never as
one curve.

Expected shape, from the ~2 h frame decorrelation: a positive `conditional_rel` at h≤4
decaying to zero. If it is flat-zero everywhere, the visual branch is done and wave 2 should
be objective work only.

## Answer

Run `baselines/results/probes/g0_ceiling_ukpv.json`, `n_train`=98,272, `n_test`=19,964, no
skipped rows. Sanity checks pass: no `alpha_selected` pinned at either grid edge for any of
the three predictor sets (visual-only, covariate-only, both) — not a mis-specified grid or a
degenerate vision-only fit.

**Contradicts the expected shape.** Aggregate `conditional_rel` over the trustworthy
population (h1–h5, `n_test_valid` ≈19,800 throughout) **rises**, not decays: 4.1% → 5.4% →
7.3% → 8.7% → 12.0%. h6–h12 jump further (24–30%), but that's a different, smaller
population — `n_test_valid` exactly halves at h6 (the known 13:30-origin dropout) with a
simultaneous 10× drop in the selected ridge penalty for both the covariate-only and combined
models. Read as one population break, not a continuous curve, exactly as this ticket warned.

**Sharper finding — ramp severity tiers diverge hard** (the `ramp` block's 3 rows are
severity tiers, mild → extreme |Δy|, not predictor sets):

| tier | conditional_rel, h1→h5 |
|---|---|
| mild | −16% → −33% (vision actively **hurts**, worsening with horizon) |
| mid | ~0%, noisy sign flips — null |
| **extreme** | **+2.2% → +7.0%**, monotonically rising |

Vision only reliably helps the most extreme ramp tier — which is the top-decile-|Δy| regime
the flagship P0 ramp metric already targets. Good consistency check that the metric points
at the right regime, but it means the aggregate `conditional_rel` number above is diluted by
a tier (mild) where vision is counterproductive; don't quote the aggregate as "vision's
ceiling" — quote the extreme-tier one.

**Answers the ticket's actual question inconclusively-but-suggestively.** The extreme-ramp
ceiling is still rising at h5 with no sign of plateauing — argues there's headroom left for
further visual-branch work. But the h≥6 population break blocks confirming this past 2.5h
without re-running the probe with that population handled cleanly, and a direct comparison
to what s2d already extracts (Δ ramp NMAE 0.0063, all horizons pooled) needs matched
aggregation before it's a real headroom number rather than an eyeballed one. Not
flat-zero — wave 2 visual-branch work is not obviously done, but the case isn't airtight
either.
