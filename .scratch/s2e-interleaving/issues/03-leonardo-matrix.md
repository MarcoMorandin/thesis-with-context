# Choose the minimal controlled Leonardo training matrix

Type: grilling
Label: wayfinder:grilling
Status: open
Assignee: unassigned
Parent: ../map.md
Blocked by: 01, 02, 04

## Question

Which sequential runs distinguish the selector failure, per-anchor capacity and
the effect of interleaving while preserving the recent-vision control?

## Comments

Results of record are the in-process JSONs, not checkpoint rescoring:
[S2D seed 42](../../../baselines/results/mmtsfm_s2d_ukpv_s2d_s42.json),
[S2D seed 44](../../../baselines/results/mmtsfm_s2d_ukpv_s2d_s44.json),
[S2E seed 42](../../../baselines/results/mmtsfm_s2e_ukpv_s2e_s42.json),
[S2E seed 44](../../../baselines/results/mmtsfm_s2e_ukpv_s2e_s44.json).
Compare paired seeds first. S2E seed 43 is absent. Dataset manifests and scored
plant/step counts match, but JSONs do not establish equal realized training steps,
epochs or checkpoint selection. Establish these from retained training metadata.

Recommended sequence, pending the blocking decisions:

1. A short validation-only GPU pilot verifies selection counts, masks, memory and
   useful visual gradients; it is not a scored forecasting result.
2. Compare shipped S2E against a repaired-selector S2E at the same 100-token budget,
   same cache, daily anchors, initialization and training schedule. Existing S2D
   remains the control. Use paired seeds 42/43/44 for the final comparison; seed 43
   completes the existing S2E arm, not an additional architecture sweep.
3. If the repaired 100-token arm still loses, use existing S2E cached latents with
   keep=490 (98 per anchor, no pruning) to test the budget bottleneck. This needs no
   cache extraction but increases sequence length from 143 to 533. The quadratic
   attention-score work scales about 13.9x; actual memory and step time must be
   measured. Reduce microbatch and preserve effective batch/update budget.
4. Only then consider greater latent depth or a design that preserves S2D's full
   recent visual payload and adds historical evidence separately. Unequal anchor
   budgets need an explicit implementation; the current reshape requires equality.

Operational contract: seed-matched S1 warm-starts; keep optimizer, freeze policy,
selection metric and optimizer-update budget matched. Use the shared curriculum
launcher and cache offline on GPU nodes. New arms need registration and experiment
review before launch. Follow [runbook](../../../knowledge/runbook.md) and
[launch procedure](../../../knowledge/running-ablations.md). Do not reuse a cache
directory for a different ladder or change the dataset of record.

No new runs or source changes were made during charting.
