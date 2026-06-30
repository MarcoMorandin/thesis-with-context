---
name: new-baseline
description: Scaffold a new tier baseline — directory, Hydra/CLI config, SLURM submit script from the project template, and an ABLATION_REGISTRY stub. Use when adding a new baseline model under baselines/tierN/. Invoke with /new-baseline <tier> <name>.
disable-model-invocation: true
---

You scaffold a new baseline for the PV forecasting thesis. The user invokes `/new-baseline <tier> <name>` (e.g. `/new-baseline 4 my_rag`). If tier or name missing, ask once, then proceed.

Read `AGENTS.md`, `docs/experiments/BASELINE_PROTOCOL.md`, and an existing peer script (e.g. `baselines/scripts/slurm_solar_vlm.sh`) to match current conventions before writing — do not hardcode values that may have drifted.

## Steps

1. **Directory**: create `baselines/tier<TIER>/<NAME>/` (vendored upstream goes under `baselines/tier<TIER>/vendor/<NAME>/` with a `VENDOR_NOTICE.md` — never hand-edit vendored code).

2. **SLURM script** `baselines/scripts/slurm_<NAME>.sh` — copy the header block from an existing `slurm_*.sh` verbatim (same `#SBATCH` account `IscrC_MTSFM`, partition `boost_usr_prod`, qos, `logs/slurm/%j_%x.{out,err}`). Keep the offline env exports (`TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE`, `WANDB_MODE=offline`, `TEAM_SCRATCH`, `UV_*`/`HF_HOME` cache dirs). Use `: "${VAR:?...}"` guards for required inputs. End by emitting predictions, running `tier4/vendor/contract_check.py`, then `scripts/import_predictions.py` to produce `results/<NAME>_<tag>.json`.

3. **Config**: one self-contained config in the baseline folder (Hydra `@dataclass`/yaml, or CLI flags matching the codebase pattern). No hardcoded dataset paths — read from `DATA`/`IMAGES_H5` env or `data.data_dir`. Seed 42. No energy-domain physics heuristics.

4. **Split discipline**: load the committed seed-42 split from `baselines/configs/splits.json` via `baselines/common/splits.py`; assert disjoint train/val/test plants. `goes_pvdaq` needs leave-one-plant-out + bad-site reconciliation (sites `1283`, `51`).

5. **Registry stub**: add a row to `docs/experiments/ABLATION_REGISTRY.md` (or note it's a baseline, not an A-series ablation) and wire it into `baselines/scripts/run_all_baselines.sh` if it should run in the master orchestrator.

## Rules

- Keep files short (target <150 lines, one capability per file — AGENTS.md §1).
- `uv` only; never emit `pip`/`python` calls — use `uv run`.
- Do not commit data, checkpoints, or logs.
- After scaffolding, print the exact `sbatch --export=ALL,...` line to launch it.
- Tell the user to run `/register-experiment` if this baseline backs an ablation hypothesis.

Be terse. Show the files created and the launch command.
