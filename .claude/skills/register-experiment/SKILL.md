---
name: register-experiment
description: Register a new ablation per the project protocol — add the ABLATION_REGISTRY row, create the configs/ablation config diff, and cut the exp/<id>-<name> branch. Use before running any A-series ablation. Invoke with /register-experiment <id> "<hypothesis>".
disable-model-invocation: true
---

You enforce the experiment protocol (AGENTS.md §4, `docs/experiments/BASELINE_PROTOCOL.md`). The user invokes `/register-experiment <id> "<hypothesis>"` (e.g. `/register-experiment A07 "TS-RAG on frozen Chronos-2 beats Chronos-2 ZS cross-plant"`). If id or hypothesis missing, ask once.

Read `docs/experiments/ABLATION_REGISTRY.md` first to see the existing format and confirm the ID is free / matches an existing TODO row.

## Steps

1. **Registry row** (`docs/experiments/ABLATION_REGISTRY.md`):
   - If a TODO row for `<id>` exists, fill it; else add a new row in ID order.
   - Columns: `ID | Hypothesis | Config | Branch | Status | Result`.
   - Hypothesis = one declarative testable sentence (no vague wording).
   - Config = real path/key (e.g. `configs/ablation/<id>.yaml` or an override key).
   - Branch = `exp/<id>-<short-name>`. Status = `IN PROGRESS`. Result = `-`.

2. **Config diff** (`configs/ablation/<id>.yaml` — confirm the actual ablation config dir from the registry "How to register" section; create the dir if absent):
   - Contains ONLY the delta from the base config, not a full copy.
   - No hardcoded dataset paths (override `data.data_dir` / use env).
   - Seed 42. No energy-domain physics heuristics (CSI, irradiance) unless the hypothesis explicitly ablates them.

3. **Branch**: create `exp/<id>-<short-name>` (`git checkout -b`). If the working tree is dirty, warn before branching.

4. **Protocol assertions** (state these in the config/notes, do not silently assume):
   - Eval on `cross_plant` (disjoint test plants), seed-42 split from `baselines/configs/splits.json`.
   - Compare against ≥1 standard baseline (Smart Persistence, Chronos-2 ZS, Solar-VLM, MMTSFM) per BASELINE_PROTOCOL §4.
   - Metrics: NMAE + NRMSE + Skill Score per §5.
   - GPU runs via `sbatch`. No committed data/checkpoints/logs.

## Output

Print: registry row added, config path created, branch name. End with the exact run command and a reminder to invoke the `experiment-reviewer` agent before launching, and to flip Status→DONE with the W&B run ID + key metric after the run.

Be terse.
