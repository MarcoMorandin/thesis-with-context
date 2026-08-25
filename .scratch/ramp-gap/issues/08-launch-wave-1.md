# 08 — Launch wave 1: six seeded curriculum chains

Type: task
Status: ready-for-human

## Question

Launch the six chains that make the mixer gate callable. Each is `s1 → s2a → s2b`, s3
dropped per standing decision 4, submitted through `slurm_curriculum.sh`:

| chain | `MODEL_CFG` | seed | tag |
|---|---|---|---|
| grassmann | `vision_chronos2_grassmann` | 42 | `mmtsfm_s2b_ukpv` |
| grassmann | `vision_chronos2_grassmann` | 43 | `mmtsfm_s2b_ukpv_grassmann_s43` |
| grassmann | `vision_chronos2_grassmann` | 44 | `mmtsfm_s2b_ukpv_grassmann_s44` |
| selfattn | `vision_chronos2_timeselfattn` | 42 | `mmtsfm_s2b_ukpv_selfattn_s42` |
| selfattn | `vision_chronos2_timeselfattn` | 43 | `mmtsfm_s2b_ukpv_selfattn_s43` |
| selfattn | `vision_chronos2_timeselfattn` | 44 | `mmtsfm_s2b_ukpv_selfattn_s44` |

`grassmann@42` is rebuilt, not reused: 03 landed the interleaved attention-mask fix, so the
existing `mmtsfm_s2b_ukpv.json` describes pre-fix code. **Move that file aside before
launching** — the rebuilt run claims the same bare tag and would otherwise overwrite the
only record of the pre-fix run.

Run in `bash`, not `zsh`: zsh does not word-split unquoted `$arm`, which collapses
`MODEL_CFG` and `SEED` into one argument and yields tags like `..._grassmann 42_s42`.
`DRY_RUN=1` first and read the plan.

The work is operational, and the failure modes are known and expensive:

- `slurm_curriculum.sh` chains stages with `afterok`. A stage that hits walltime writes no
  `best.ckpt`, no results JSON, and takes every downstream stage with it
  (`DependencyNeverSatisfied`). Reserve to the 24 h partition cap; unused walltime is free.
- Confirm `MODEL_CFG` actually propagates. `use_grassmann` is set by the model config and the
  stage YAMLs deliberately do not force it — verify the selfattn chains really build
  `TimeSelfAttention` in `layer[0]` rather than silently inheriting the hub config's
  `use_grassmann=True`, which `lightning_module` guards against but which has bitten before.
- Keep `TRAIN_STRIDE` aligned with the V-JEPA cache keys, or training silently falls onto the
  live-encode path.
- `S3_EPOCHS` is irrelevant now, but confirm the chain actually stops after s2b rather than
  submitting an s3 job that nothing reads.

Set `model.compute_marginal_gain=true` on every chain so each arm emits its own vision
on/off decomposition natively (needs 02).

The answer records: the five result JSON paths, wall-clock and billed cost per chain against
the estimate in 01, and any stage that needed a resume.
