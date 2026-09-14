---
name: new-baseline
description: Scaffold a protocol-compliant baseline. Invoke as `$new-baseline <tier> <name>`.
disable-model-invocation: true
---

Create a new baseline only after reading `knowledge/protocol.md` and a peer baseline.

1. Create `baselines/tier<TIER>/<NAME>/`; put vendored code under `vendor/<NAME>/` with a `VENDOR_NOTICE.md`.
2. Add a short SLURM script following an existing peer's header, offline cache exports, required environment guards, prediction export, contract check, and result import.
3. Add one self-contained Hydra config. Use environment-provided data paths, seed 42, and no energy-physics heuristic.
4. Load the committed split through `baselines/common/splits.py`; assert disjoint plants and apply required `goes_pvdaq` reconciliation.
5. Register the baseline or its associated ablation, and wire it into the master runner only if it belongs there.

Run the focused checks, show created files, and give the exact `sbatch --export=ALL,...` command. Do not commit artifacts.
