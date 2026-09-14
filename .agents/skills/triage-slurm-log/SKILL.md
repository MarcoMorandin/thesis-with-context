---
name: triage-slurm-log
description: Diagnose a failed Leonardo SLURM job from its log pair. Invoke as `$triage-slurm-log <job-id|path>`.
disable-model-invocation: true
---

Inspect the named log pair, or the newest pair in `logs/slurm/` if none is named.

1. Read `.err`, then the end of `.out`, and identify the first fatal error rather than a cascade symptom.
2. Inspect the matching SLURM script only as needed to identify its command and exports.
3. Classify exactly one cause: OOM, module/environment, CUDA/driver, data path, walltime, split/contract, or code.
4. Give one or two concrete corrective steps and a corrected `sbatch --export=ALL,...` command when applicable.

Output `Job`, `Verdict`, quoted `Root cause`, `Fix`, and optional `Resubmit`. If ambiguous, name the one next check.
