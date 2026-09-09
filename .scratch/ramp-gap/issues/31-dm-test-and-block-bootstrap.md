# 31 — Implement the significance machinery the protocol promises

Type: task
Status: open

## Question

`baselines.md` §4.5 commits the paper to a Diebold–Mariano test on per-sample loss
differentials (s2d vs each P0 baseline), a paired block bootstrap (block = day, 1000
resamples) giving a 95 % CI on ΔNMAE, and Holm–Bonferroni across the baseline set — with
table entries bolded only when the CI excludes 0. None of it exists: `runner.py` writes
`_losses.npz` sidecars for tiers 0–3 only, MMTSFM writes per-plant `_pred.npz` with no loss
sidecar, and `import_predictions.py` explicitly notes "no DM/bootstrap sidecar". The paper
currently decides wins by the seed-floor rule alone.

Deliver: (1) a per-window loss sidecar for MMTSFM arms and imported tier 4–6 rows, derived
from the existing `_pred.npz` files; (2) one script under `baselines/scripts/` that computes
DM statistic, block-bootstrap CI and Holm-adjusted p-values for s2d against every P0 row and
against s2a / s1 / the ticket-28 arm, on both NMAE and ramp NMAE; (3) the resulting table
linked here. CPU only.

Resolved when the table exists and states, for each comparison, whether the CI excludes 0.

Context: `paper-readiness-audit.md` §2.3.
