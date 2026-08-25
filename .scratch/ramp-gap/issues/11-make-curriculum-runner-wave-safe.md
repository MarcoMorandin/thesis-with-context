# 11 — Make the curriculum runner wave-safe

Type: task
Status: resolved

## Question

`slurm_curriculum.sh` cannot launch wave 1. Four defects, the first fatal:

**1. Tags and checkpoint dirs are not unique per arm.** Lines 165-166:

```bash
tag="mmtsfm_${st}_${dcfg}"           # -> mmtsfm_s2b_ukpv
stage_dir="${CKPT_DIR}/${ds}_${st}"  # -> $CKPT_DIR/uk_pv_s2b
```

Neither carries the seed or `MODEL_CFG`. Wave 1's five chains would write one results
JSON and **share one checkpoint directory**, clobbering each other's `best.ckpt` and
`last.ckpt` mid-run while `afterok` chains stages onto whichever file won the race. Five
chains of compute, one corrupted answer, no error raised. It would also overwrite the
committed `mmtsfm_s2b_ukpv.json` — the `grassmann@42` baseline the gate compares against.

**2. No `END_STAGE`.** `STAGES=(s1 s2a s2b s3)` and `RUN_STAGES` runs from `START_STAGE`
to the end. Map standing decision 4 terminates every arm at s2b; the runner cannot express
it, so each chain pays ~24 h for a stage already known to regress.

**3. `compute_marginal_gain` is not plumbed.** `curriculum_stage.sbatch` has knobs for
`MODEL_CFG`, `SEED`, `N_VIS`, `SP_REF`, `TRAIN_STRIDE` and none for this. Reachable via
`EXTRA_OVERRIDES`, but wave 1 needs it on every chain and an escape hatch you must remember
is one you forget. Without it the arms emit no on/off decomposition.

**4. No dry run.** Nothing lets you see the planned submissions before committing five
`afterok` chains to the queue.

Resolve all four. (4) is also the testable seam for (1): a `DRY_RUN=1` mode that prints
what it would submit makes tag uniqueness assertable from pytest instead of from a cluster.

Backward compatibility matters: `mmtsfm_s2b_ukpv.json` is referenced by the map's gate,
`ALL_RESULTS`, and the manuscript. The canonical `(grassmann, seed 42)` combination must
keep its historical tag; every other combination must be distinct.

## Answer

All four resolved. 245 tests green (7 new in `test_curriculum_runner_wave_safety.py`).

**Arm identity.** `ARM_SUFFIX` = `_<variant>_s<seed>`, empty only for the canonical
`(grassmann, 42)` pair, applied to `tag`, `stage_dir` and the warm-start `prev_ckpt`.
Verified by dry-running the real wave: **18 tags, 18 unique**, and
`mmtsfm_s2b_ukpv` still belongs to `grassmann@42`.

**A second half of the bug that the ticket did not name.** `curriculum_stage.sbatch`
recomputed `STAGE_DIR="${CKPT_DIR}/${DS}_${STAGE}"` itself, so the suffix never reached the
checkpoint directory and every chain of a wave would still have shared one — defeating the
tag fix while the tags looked correct. The submitter now exports the resolved `STAGE_DIR`
and the stage script only falls back when run by hand. Pinned by
`test_submitter_exports_the_resolved_stage_dir` and `test_stage_honours_the_exported_stage_dir`.

**`END_STAGE`.** Defaults to `s3` (unchanged behaviour); `END_STAGE=s2b` yields exactly
`s1, s2a, s2b`. Validated against `STAGES` so a typo fails loudly instead of silently
running the whole chain.

**`MARGINAL_GAIN=1`.** Submitter exports it; `curriculum_stage.sbatch` translates it to
`model.compute_marginal_gain=true`. Two seams, two tests — the first version tested the
wrong one and asserted the Hydra override on the submitter, which cannot know it.

**`DRY_RUN=1`.** Both scripts. The submitter prints STAGE / TAG / STAGE_DIR / MODEL_CFG /
SEED / EXPORTS per planned job; the stage script prints the composed Hydra command and
exits. The `sbatch`-not-found guard is skipped under it, so a wave can be inspected from a
laptop. This is also the seam the tests drive.

### Wave 1 launch, verified against the dry run

```bash
for arm in "vision_chronos2_grassmann 43" "vision_chronos2_grassmann 44" \
           "vision_chronos2_timeselfattn 42" "vision_chronos2_timeselfattn 43" \
           "vision_chronos2_timeselfattn 44"; do
  set -- $arm
  MODEL_CFG="$1" SEED="$2" END_STAGE=s2b MARGINAL_GAIN=1 \
    bash scripts/slurm_curriculum.sh
done
```

Prepend `DRY_RUN=1` first and read the plan. **In `bash`, not `zsh`** — zsh does not
word-split unquoted `$arm`, which silently collapses `MODEL_CFG` and `SEED` into one
argument and produces tags like `mmtsfm_s2b_ukpv_grassmann 42_s42`. That happened while
verifying this ticket and the dry run is what caught it.

**Facts later tickets depend on:**

- Wave-1 result files are `mmtsfm_s2b_ukpv_{grassmann,selfattn}_s{43,44,42,43,44}.json`,
  plus the existing `mmtsfm_s2b_ukpv.json` for grassmann@42. The gate reads six files.
- `aggregate_all.py` globs `mmtsfm_*.json`, so the new tags will appear in ALL_RESULTS as
  separate rows — intended, but the leaderboard will grow six MMTSFM lines.
- Nothing here is pushed yet; Leonardo pulls from git.
