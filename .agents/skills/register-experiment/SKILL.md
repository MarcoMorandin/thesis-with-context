---
name: register-experiment
description: Register an A-series ablation before it runs. Invoke as `$register-experiment <id> "<hypothesis>"`.
disable-model-invocation: true
---

Register the proposed ablation before a run.

1. Read `knowledge/ablations.md` and `knowledge/protocol.md`; confirm the ID is free or has a TODO row.
2. On an `exp/<id>-<short-name>` branch, add or complete one registry row: a declarative hypothesis, `MMTSFM/configs/ablation/<id>.yaml`, branch, `IN PROGRESS`, and `-` result.
3. Create that Hydra config as a delta only. Use seed 42, the `cross_plant` split, and no hard-coded data paths or energy-physics feature.
4. State the standard comparator and required NMAE, NRMSE, and Skill Score metrics. GPU work launches through `sbatch`.
5. Report the row, config, branch, and exact launch command. Direct the user to `$experiment-review` before launch.

Ask once only if the ID or hypothesis is missing. Do not start the run.
