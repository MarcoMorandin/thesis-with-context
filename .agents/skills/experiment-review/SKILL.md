---
name: experiment-review
description: Review an A-series ablation before launch. Invoke as `$experiment-review <id>`.
disable-model-invocation: true
---

Review the named planned experiment; report only.

1. Check `knowledge/ablations.md` for a testable hypothesis, real config path, non-main branch, and non-TODO launch status.
2. Inspect `MMTSFM/configs/ablation/<id>.yaml`: it must be a Hydra delta, seed 42, and free of hard-coded data paths and unapproved physics features.
3. Check the branch is `exp/<id>-<short-name>` and exists locally.
4. Check `knowledge/protocol.md`: `cross_plant`, a standard comparator, NMAE, NRMSE, and NRMSE Skill Score; use the required dataset-specific split rules.
5. Confirm GPU launches use SLURM and that data, checkpoints, and logs are not committed.

For each item output `✓`, `✗` plus one precise fix, or `?` if inspection cannot prove it. If any item fails, end with `Blocked`; otherwise end with `Ready to run.`
