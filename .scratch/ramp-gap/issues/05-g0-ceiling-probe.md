# 05 — How much visual headroom is left? (G0 ceiling probe)

Type: task
Status: ready-for-agent

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
