# MMTSFM — Final-Model Curriculum on Leonardo (Design Spec)

**Date:** 2026-07-20
**Status:** Approved design, pending spec review
**Branch:** `main` (user opted to stay on current branch, overriding the CLAUDE.md task-branch rule)
**Scope:** Make `MMTSFM/` run the full 4-stage curriculum (S1→2a→2b→3) with all proposal features, dependency-linked on the Leonardo SLURM cluster, for both datasets of record (`uk_pv`, `goes_pvdaq`). Files prepared here; the user submits on the cluster.

---

## 1. Problem statement

The reported `results/mmtsfm/` runs were produced by a **single `trainer.fit()`** per ablation with all new modules trained jointly from step 0 — not the designed curriculum. Concrete gaps:

1. **No curriculum runner.** `scripts/run_all_mmtsfm.sh` runs one fit per ablation; no `slurm_curriculum.sh` chains S1→2a→2b→3.
2. **Pretrained tokenizer discarded.** `chronos_config` forces `input_patch_size=8` + 9 quantiles, mismatching the Chronos-2 checkpoint (native patch 16, 21 quantiles) → `input_patch_embedding`, `output_patch_embedding`, `shared` reinitialize (run log lines 165-171). The domain-critical value↔latent map is random.
3. **Grassmann warmup never runs.** `grassmann_warmup_steps` defaults 0, never set by any config.
4. **No freeze schedule as-run.** All reported runs use `freeze_chronos=false`.
5. **Misleading config.** YAML `d_model:512 / num_layers:6 / d_ff:2048` are silently ignored — `from_pretrained` keeps the checkpoint's native architecture. Only grassmann fields + `chronos_config` propagate (`lightning_module.py:100-132`).

Result: NMAE ≈ 0.106 (below a plain MLP), and vision + Grassmann show ≈ 0 lift because the whole stack is cold-started at once.

## 2. Goals / non-goals

**Goals**
- A dependency-linked `slurm_curriculum.sh` that runs the full curriculum for `uk_pv` and `goes_pvdaq`, threading checkpoints between stages.
- Per-stage Hydra configs encoding the freeze/fusion/warmup schedule from `proposal.md` §"Multi-Stage Training Curriculum".
- Fix Chronos-2 weight transfer so the pretrained TS tokenizer loads (patch 16, keep 9 protocol quantiles → only the small head reinitializes).
- Wire and verify Grassmann warmup.
- Audit that known future covariates actually influence the forecast; fix if inert.
- Cluster hardening: resume safety, per-stage walltime, results written in the baselines schema; each stage writes a tagged protocol JSON.
- Local CPU smoke test exercising the full chain before cluster submit.

**Non-goals**
- No new modeling components beyond what the proposal describes.
- No flip of the `future_covariates_mask` loss semantics (that mechanism marks *known future targets*, not weather covariates — flipping it collapses the loss).
- No cluster submission from here (no Leonardo access); local verification only.

## 3. Design

### 3.1 Weight-transfer fix (config-only)
In every model config's `chronos_core_cfg.chronos_config`:
- `input_patch_size: 16`, `input_patch_stride: 16`, `output_patch_size: 16` → matches the checkpoint's patch dim so `input_patch_embedding` (shared by context/future/covariate patches) **transfers** instead of reinitializing.
- Keep `quantiles: [0.1..0.9]` (9) → protocol-aligned. The output head still reinitializes (9≠21), which is small and learned during S1.
- Remove or comment the dead `d_model/num_layers/num_heads/d_ff` lines, with a note that the pretrained architecture wins. (Do **not** attempt to force a smaller model — that would break weight loading.)
- Consequence on shapes (uk_pv): `T_ctx = ceil(672/16) = 42` context patches; `num_output_patches = ceil(12/16) = 1`. Assertions using `n_visual_context_steps <= T_ctx` still hold (3 ≤ 42). `validate_n_visual_context_steps` re-checked.

**Verification:** parse the from_pretrained load report in a test / smoke run and assert `input_patch_embedding` is **not** in the MISMATCH set.

### 3.2 Per-stage configs — `configs/stage/` (new Hydra group)
Thin overrides composed on top of a model + data config. Selected via `+stage=s1` etc.

| File | fusion_mode | use_grassmann | freeze_chronos | skip_vision_stack | visual_mask_prob | V-JEPA | grassmann_warmup_steps | notes |
|---|---|---|---|---|---|---|---|---|
| `s1.yaml` | late | true | false | **true** | 1.0 | not built | **2000** | TS + Grassmann warmup, vision skipped |
| `s2a.yaml` | late | true | **true** | false | 0.3 (p_v=0.7) | partial unfreeze last 4 | 0 | visual alignment |
| `s2b.yaml` | **interleaved** | true | **true** | false | 0.5 | re-frozen | 0 (0.3× first 1000 via warmup) | Grassmann cross-modal |
| `s3.yaml` | interleaved | true | false | false | 0.5 | progressive unfreeze | 0 | full joint |

- `freeze_chronos=true` keeps only Grassmann {W_red,W_plu,W_gate,offset_weights,modality_pair_bias} + reinit'd embeddings/head + last N encoder blocks trainable (existing logic `lightning_module.py:160-188`).
- S2a partial-V-JEPA unfreeze and S3 progressive unfreeze use the existing `VisualEncoder.partial_unfreeze` / `set_freeze` hooks; add a small Lightning callback or config flag to drive per-epoch progressive unfreeze in S3 (`n_unfreeze_encoder_blocks` analog for vision).
- Modality-dropout knobs (`visual_dropout_prob`, `numeric_dropout_prob`) set per stage as in the table.

### 3.3 Grassmann warmup
Already plumbed: `grassmann_warmup_steps` → `configure_optimizers` LR lambda applies a reduced multiplier to the Grassmann param groups for the first N steps. Tasks: (a) set it in `s1.yaml`/`s2b.yaml`; (b) add `test_grassmann_warmup` asserting the Grassmann group LR is scaled during warmup and restored after.

### 3.4 Curriculum runner — `scripts/slurm_curriculum.sh`
- For each dataset in `DATASETS`, submit 4 jobs chained with `sbatch --dependency=afterok:<jobid>`.
- Each stage: `python -m mmtsfm.train +stage=<s> data=<dcfg> +ckpt_path=<prev_stage_best>` … writing to `${CKPT_DIR}/<dataset>_<stage>/`.
- Checkpoint threading: a helper resolves the previous stage's best finite checkpoint (reuse `_best_finite_checkpoint_path` logic or glob the stage's ckpt dir). S1 starts from the HF pretrained (no `+ckpt_path`).
- Each stage runs its own test pass → tagged JSON `mmtsfm_<stage>_<dataset>` in `${RESULTS_DIR}` (= `baselines/results/`), so `aggregate_all.py` shows per-stage progression.
- Per-stage SBATCH walltime overridable via env (S1/S3 longer; S2a/S2b shorter). Partition/account defaults: `boost_usr_prod`, `IscrC_MTSFM`.
- Resume: each stage resumes from its own `last.ckpt` if present (requeue safety), independent of the cross-stage `+ckpt_path`.
- `--smoke` / `SMOKE=1`: run the whole chain locally with a tiny synthetic dataset, 1-2 steps/stage, CPU, no sbatch — for pre-submit validation.
- Reuse `precache_login.sh` (exists) for weights + V-JEPA latent pre-extraction; document the login-node prereq in the script header.

### 3.5 Covariate correctness audit
- Add `test_future_covariates_influence`: build a batch, run forward, perturb one future covariate channel, assert the forecast changes beyond tolerance. If it does not, trace the covariate-row path (`vision_chronos2.py:683-725`) — likely fixes: ensure cov rows carry non-zero future embeddings, correct group_id sharing, and are not dropped by masking.
- Document (in code + `MMTSFM_AS_IMPLEMENTED.md`) that `future_covariates_mask=0` is the correct loss setting and covariates enter via GroupSelfAttention rows, not the loss arg.

### 3.6 Config honesty
- `MMTSFM_AS_IMPLEMENTED.md` and the schema artifact currently state d_model 512 / 6 layers; correct them to "pretrained Chronos-2 native size (YAML values ignored)". Update after confirming the actual size at implementation time.

## 4. Components & interfaces

| Unit | Purpose | Depends on |
|---|---|---|
| `configs/stage/{s1,s2a,s2b,s3}.yaml` | per-stage override deltas | model + data configs |
| `scripts/slurm_curriculum.sh` | chain 4 stages × N datasets, thread ckpts, write per-stage JSON | `train.py`, `precache_login.sh`, SLURM |
| config edits (patch 16) | pretrained tokenizer transfer | Chronos-2 ckpt |
| progressive-unfreeze callback (S3) | per-epoch vision unfreeze | `VisualEncoder` hooks |
| tests: warmup, covariate-influence, weight-transfer, chain-smoke | verify each fix | pytest, synthetic dataset |

## 5. Testing / verification

- `uv run pytest` — new unit tests (warmup LR, covariate influence, patch-16 transfer report).
- Local chain smoke: `SMOKE=1 bash scripts/slurm_curriculum.sh` runs S1→S3 on synthetic CPU data, asserting each stage loads the prior checkpoint and produces a finite test loss + a JSON.
- Confirm `input_patch_embedding` absent from the from_pretrained MISMATCH report.
- Do **not** claim cluster success — only local verification is possible here.

## 6. Risks

- **Patch 16 changes context/output granularity** → downstream shape asserts and the interleave (`n_vis`, `T_M`) must be re-validated; covered by the smoke test.
- **Progressive vision unfreeze** may need a new callback; keep it minimal and config-gated.
- **Covariate audit** may surface a deeper wiring bug; if the fix balloons beyond config, flag before expanding scope.
- **Working on `main`** per user choice — mitigate with micro-commits so any step is revertible.

## 7. Rollout

Micro-commits per sub-task (patch-16 config, stage configs, runner, warmup test, covariate audit, doc fixes). Local smoke green before handing off for cluster submission.
