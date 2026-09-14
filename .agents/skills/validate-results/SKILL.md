---
name: validate-results
description: Validate baseline result JSONs before results are reported or aggregated. Invoke as `$validate-results`.
disable-model-invocation: true
---

Audit result artifacts; do not repair them.

1. Read `knowledge/protocol.md` for expected models, splits, horizons, and metrics.
2. Inspect one result JSON to establish the actual schema, then inspect all files under `baselines/results/` and any aggregate.
3. Require finite NMAE, NRMSE, and NRMSE Skill Score; `cross_plant` coverage; a Smart Persistence reference; and consistent horizon/granularity.
4. Verify expected dataset coverage, including required `goes_pvdaq` leave-one-plant-out and bad-site reconciliation when in scope.
5. List missing, stale, invalid, and complete files.

Output aggregate status, per-file findings, and the exact next corrective action. End `Safe to aggregate/report.` only when every check passes.
