# 16 — Run s2c, 3 seeds, against s2b

Type: task
Status: in-progress
Blocked by: 14, 15
Blocks: 17, 18

## Question

Run the arm and collect the comparison.

## Protocol

Same s1 checkpoints as s2b (`uk_pv_s1_selfattn_s{42,43,44}`, borrowed via `INIT_CKPT`),
same data split, same aligned protocol (165,295 scored steps, 14 disjoint plants), 3 seeds,
same ramp and aggregate metrics, `MARGINAL_GAIN=1` so the forced vision-off pass is scored.

Control is the existing wave-1 `mmtsfm_s2b_ukpv_selfattn_s{42,43,44}`. Note s42 and s43
were killed mid-run and are currently unresumed — the control is n=1 (s44, ramp 0.1487)
until they are re-run. Resume them before or alongside this.

~10 GPU-h per seed. Compute is not the binding constraint.

## Pre-flight (verified 2026-08-28, code side only)

Code is on `main` as of `7772d9f`; Leonardo can pull. Launch from the login node, one
submission per seed:

```bash
export MAIL_USER="marco.morandin@studenti.unitn.it"
export DATA_DIR=/leonardo_scratch/fast/IscrC_MTSFM/data
CKPT=/leonardo_scratch/fast/IscrC_MTSFM/checkpoints/curriculum
for S in 42 43 44; do
  START_STAGE=s2c END_STAGE=s2c SEED=$S \
    MODEL_CFG=vision_chronos2_s2c \
    INIT_CKPT=$CKPT/uk_pv_s1_selfattn_s$S/best.ckpt \
    MARGINAL_GAIN=1 \
    bash scripts/slurm_curriculum.sh
done
```

Checked against the runner source, not assumed:

- `MARGINAL_GAIN=1` reaches the job (`slurm_curriculum.sh:241`) and becomes
  `model.compute_marginal_gain=true` (`curriculum_stage.sbatch:113`). The `false` in
  `vision_chronos2_s2c.yaml` is the same default every other arm carries — leave it.
- `INIT_CKPT` overrides the derived warm-start path and is **fatal if missing**
  (`:216-218`). It has to be set: `STAGES=(s1 s2a s2b s3 s2c)` puts s2c after s3, so the
  derived predecessor would be an s3 checkpoint this arm never produced.
- Tags come out `mmtsfm_s2c_ukpv_s2c_s{42,43,44}`, checkpoint dirs `uk_pv_s2c_s2c_s$S`.
  Redundant-looking but distinct per seed, and consistent with the s2b control's
  `mmtsfm_s2b_ukpv_selfattn_s44`.
- **No `CKPT_DIR` override.** The runner already defaults to
  `${TEAM_SCRATCH}/checkpoints/curriculum` (`:30`), and that is where the s1 warm-start
  checkpoints actually live — confirmed on the cluster 2026-08-28. runbook.md:116 shows a
  `curriculum_tsa` tree for the timeselfattn variant; that separate-directory recipe is
  not what the wave-1 runs did. The `_selfattn` arm suffix is what separates the variants
  inside the one `curriculum/` tree, so s2c's `uk_pv_s2c_s2c_s$S` cannot collide with the
  s2b control sitting beside it.

One bug fixed on the way: the runner's own usage comment told you to pass
`ARM_SUFFIX=_s2c`, but `ARM_SUFFIX` is unconditionally recomputed at `:132-135` from
MODEL_CFG + SEED, so the environment value is discarded. Harmless as it stands — the
derived suffix is the correct one — but the documented value carries no seed, so had the
variable been honoured a 3-seed wave would have written all three runs into one tag and
one checkpoint dir. Comment corrected.

**Not verified**: `uk_pv_s1_selfattn_s44/best.ckpt` was confirmed present by the user
(2026-08-28); s42 and s43 have not been checked, and a missing one is a hard FATAL at
submit time. Queue state and whether the killed s2b s42/s43 are resumable also remain
unchecked — `ssh` is blocked by the local shell allowlist, so all cluster-side steps run
from the user's own login-node session.

## Done when

- [ ] 3 seeds completed with results JSONs, no walltime kills
- [ ] ramp NMAE, ramp NRMSE, NMAE, SS, and the vision marginal recorded per seed
- [ ] attention divergence and tau distances (ticket 15) present in each JSON
- [ ] per-seed values reported individually, plus mean +/- sd — not only the mean

## Launch log

Two submissions died before a single step ran. Both were s2c-only, both caused by
`output_patch_size: 16 -> 4`, and neither was reachable by any test that existed when
ticket 14 closed — the sixteen s2c tests all build a `Chronos2EncoderBlock` or a
`SimpleNamespace`, never a `Chronos2Model`, and none of them loads a checkpoint.

- **54833985** — `AssertionError: input_patch_size and output_patch_size sizes must be
  equal, but found 16 and 4`, at construction. `encode` embeds context patches and future
  patches with the same `input_patch_embedding`, and a future patch is
  `[time_enc, covariates, mask]` each *output*_patch_size wide. Fixed in `e94d234`: the
  future path gets its own `ResidualBlock`, built only when the sizes differ, so every
  other arm's parameter set and checkpoint keys stay byte-identical.
- **54837551** — `size mismatch for model.chronos.output_patch_embedding.output_layer.weight:
  ... torch.Size([144, 3072]) from checkpoint, ... [36, 3072]`, at the s1 warm start. The
  HF backbone load had already succeeded. `load_state_dict(strict=False)` skips ABSENT
  keys but raises on keys present at the wrong shape, and `output_patch_embedding` is
  `num_quantiles * output_patch_size` wide: 9*16=144 in the donor, 9*4=36 here. Fixed in
  `cfebe08`: `drop_reshaped_tensors` filters exactly those four tensors, logs each with
  both shapes, and refuses a donor that mismatches on more than max(8, 5%) — a wrong
  `INIT_CKPT` must not become a from-scratch run that logs as a chained stage.

- **54838774** — `RuntimeError: mat1 and mat2 shapes cannot be multiplied (12x12 and
  48x3072)`, in Lightning's sanity check at `vision_chronos2.py:741`. 12 is
  `output_patch_size * 3`, 48 is `input_patch_size * 3`. The previous fix taught
  `Chronos2Model.encode` to pick the right embedding, and the arm that needs it never
  calls `encode`: `VisionChronos2Model` overrides `forward` and inlines its own copy of
  the encode path, so the one arm that sets unequal patch sizes was still handing future
  patches to the context embedding. Fixed in `5033fa8`: the dispatch moves into
  `Chronos2Model.embed_future` and both files call that, so a third copy of the encode
  path would have to bypass it deliberately.
  - Found in the same pass, before it shipped: `vision_chronos2.py:785` embedded the
    **covariate** rows with `input_patch_embedding` too. They come from a second
    `_prepare_patched_future` call, so they are `output_patch_size` wide as well, and
    uk_pv passes weather covariates — that was failure #4, one `grep -n` earlier than it
    needed to be.

The pattern across all three: each fix was verified against `Chronos2Model` while the arm
that actually runs is `VisionChronos2Model`. The new tests build the vision arm from the
shipped YAML across video on/off x covariates on/off, and were verified adversarially —
reverting only `vision_chronos2.py` fails 5 of them.

Suite 342 passed; `main` at `9ea5dc5`. Third relaunch pending.
