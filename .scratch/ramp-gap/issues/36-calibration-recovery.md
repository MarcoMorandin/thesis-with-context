# 36 — Can post-hoc recalibration restore s2d's coverage without giving back the ramp gain?

Type: task
Status: open

## Question

s2d pays for its ramp win in calibration: coverage_80 0.768 → 0.731 (nominal 0.80), quantile
ECE 0.0280 → 0.0359, in all three seeds, at flat CRPS. Reported as-is, a reviewer reads
"sharper, not better". Eval-only fix to test: fit a per-quantile conformal / isotonic
recalibration on the **validation plants only** (never test), apply to s2d's test quantiles,
and re-score.

Resolved when, for all 3 seeds, the recalibrated coverage_80, ECE, CRPS, NMAE and ramp NMAE
are recorded here beside the raw numbers. Coverage restored at unchanged ramp NMAE → report
both rows. Ramp NMAE moves → the trade-off is real and the paper says so.

Context: `paper-readiness-audit.md` §2.3; `ablations.md` §1 calibration paragraph.
