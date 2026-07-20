# MMTSFM — Architecture As Actually Implemented

> This document describes what the **code** in `MMTSFM/` does, derived from the
> source and from the run logs that produced `results/mmtsfm/`. It is the
> engineering ground-truth counterpart to `knowledge/docs/proposal.md` and
> `knowledge/docs/mmtsfm-architecture-deep-dive-design-rationale.md`. Where the
> two disagree, this file wins for "what ran".

---

## ⚠ Update 2026-07-20 — curriculum + fixes landed

The gaps diagnosed below (single-fit runs, discarded tokenizer, inert covariates)
have since been fixed. Sections 3–7 describe the **old `results/mmtsfm/` runs**;
the current code now supports the full curriculum. What changed:

- **Weight transfer** — `input_patch_size`/`output_patch_size` set to **16** to
  match the `amazon/chronos-2` checkpoint (native d_model **768**, **12** layers,
  patch 16, 21 quantiles — the YAML `d_model:512/num_layers:6` were always
  ignored). The pretrained `input_patch_embedding` now transfers; only the small
  output head reinits (9 protocol quantiles).
- **Curriculum runner** — `scripts/slurm_curriculum.sh` +
  `scripts/curriculum_stage.sbatch` chain S1→2a→2b→3 as dependency-linked SLURM
  jobs per dataset, threading a stable `best.ckpt` (exported by `train.py`).
  Per-stage configs in `configs/stage/{s1,s2a,s2b,s3}.yaml`.
- **Grassmann warmup** — wired via `grassmann_warmup_steps` (S1=2000, S2b=1000).
- **Vision unfreeze** — `freeze_visual_encoder="partial"` + progressive unfreeze
  now drive `VisualEncoder.partial_unfreeze`/`set_freeze` per stage.
- **Covariates** — known future weather now actually influences the forecast:
  the covariate rows kept a mask=0 that zeroed their values pre-embedding, and
  the interleaved branch dropped the rows entirely. Both fixed + regression-tested
  (`tests/test_curriculum_features.py`).

Design spec: `docs/superpowers/specs/2026-07-20-mmtsfm-curriculum-final-design.md`.
The diagnosis below is retained as the record of *why* the old numbers were poor.

---

## 0. TL;DR

- The **model architecture** (Chronos-2 backbone + Grassmann mixing + V-JEPA
  vision + LatentSummarizer + late/interleaved fusion) is implemented and
  matches the proposal's *structure*.
- The **4-stage training curriculum is NOT run**. Every result in
  `results/mmtsfm/` comes from a **single `trainer.fit()`** with all new modules
  trained jointly from step 0. No Stage 1 → 2a → 2b → 3 chaining, no Grassmann
  warmup, no freeze schedule. The proposal itself admits `slurm_curriculum.sh`
  "does not yet exist".
- The reported runs land at **NMAE ≈ 0.106 / NRMSE ≈ 0.151 / SS ≈ 0.34**, worse
  than a plain MLP (0.096) and far worse than iTransformer (0.070) or Solar-VLM
  (0.096). All four MMTSFM variants score within ~1% of each other → **vision
  and Grassmann add ≈ 0 lift**; the multimodal machinery is inert.
- Root causes (detail in §6): (a) no curriculum, (b) Chronos-2's input tokenizer
  and output head are **reinitialized from scratch** (patch-size mismatch), so
  the most domain-critical pretrained weights are thrown away, (c) known future
  covariates are masked as *unknown*.

---

## 1. Entry point & training loop (`train.py`)

`MMTSFM/src/mmtsfm/train.py` is a plain Hydra + Lightning single-shot:

```
instantiate(data) → instantiate(model) → instantiate(trainer)
if cfg.train:  trainer.fit(model, datamodule, ckpt_path=cfg.ckpt_path)
if cfg.test:   trainer.test(model, datamodule, ckpt_path=best_or_given)
```

- **No stage logic exists in code.** The only multi-stage hook is
  `+ckpt_path=<file>` (resume weights) plus the per-stage flags
  (`fusion_mode`, `freeze_chronos`, `use_grassmann`, `skip_vision_stack`). Chaining
  them into a curriculum is left to the operator; nothing does it automatically.
- `train=false +ckpt_path=<file>` = test-only scoring of an existing checkpoint.

## 2. How the reported runs were launched (`scripts/run_all_mmtsfm.sh`)

The results were produced by this orchestrator, **one single fit per ablation**,
`MAX_EPOCHS=50`, `TRAIN_STRIDE=12`, V-JEPA latents pre-extracted to a cache dir.
Ablation matrix (`ABLATIONS_DEFAULT`):

| Tag (result file) | Override | Fusion | Grassmann | Vision |
|---|---|---|---|---|
| `grassmann_interleaved` | `model=vision_chronos2_grassmann` | interleaved | on | on |
| `selfattn_late` | `model=vision_chronos2_timeselfattn` | late | off | on |
| `selfattn_interleaved` | `fusion_mode=interleaved use_grassmann=false` | interleaved | off | on |
| `grassmann_no_modbias` | `…grassmann_modality_pair_bias=false` | interleaved | on | on |
| `numeric_grassmann` | `skip_vision_stack=true …emit_vision=false` | (numeric) | on | off |

Each is a fresh `python -m mmtsfm.train …` (see log line 1 of any
`results/mmtsfm/*.log`). No stage passes checkpoints to the next.
`RESUME=1` only resumes the **same** tag's `last.ckpt` for requeue safety.

## 3. Data (`data/pv_record.py`, `data/datamodule.py`, `configs/data/ukpv.yaml`)

- Dataset of record: `dataset_all.parquet` + `images_all.h5`, `uk_pv` track.
- Physical-time window: **14-day history / 6-hour horizon** → uk_pv 30-min =
  **672 / 12 steps**.
- `num_entities=4` at train (cross-plant mixing via GroupSelfAttention); forced
  to 1 at val/test for clean per-plant protocol metrics.
- Covariates: `covariate_dim=14` (protocol `COV_COLS`, incl. future weather).
- Vision: `video_frames=8` over a 6-h recency window, 224², ImageNet-normed.
  V-JEPA latents cached offline (`vjepa_cache_dir`), or encoded on cache miss.

Batch → `_unpack_batch` (lightning_module.py:320) flattens `[BS, N] → [BS*N]`
and emits `context, future_target, covariate_channels (per-channel tokens),
video / video_latents, visual_mask, …`.

## 4. Model (`models/chronos2/vision_chronos2.py` → `VisionChronos2Model`)

### 4.1 Time-series backbone (Chronos-2)
- `Chronos2Model.from_pretrained("amazon/chronos-2", …, ignore_mismatched_sizes=True)`.
- Config forces `d_model=512, num_layers=6, num_heads=8`,
  `input_patch_size=8, output_patch_size=8, context_length=2048, use_arcsinh=true`,
  9 quantiles, `max_output_patches=4`.
- **⚠ Reinitialized due to shape mismatch** (from run log, lines 165-171):
  `input_patch_embedding`, `output_patch_embedding`, and `shared` special-token
  embedding — ckpt patch dim (16 → feature 48, output 336) vs model (8 → 24 / 72).
  These trained-from-scratch pieces are exactly the value↔latent and
  latent↔quantile projections. The encoder transformer blocks *do* load.

### 4.2 Temporal mixer (`grassmann.py` → `CausalGrassmannMixing`)
- O(L) Plücker manifold mixing, offsets `[1,2,4,8,12,16]`, reduced dim 32, RoPE
  phase, gated fusion `α = σ(W_gate[h‖g])`, optional 4 modality-pair biases.
- Selected by `use_grassmann`; alternative is `TimeSelfAttention` (`layers.py`).
- Always **trained from scratch** (not in the pretrained checkpoint).

### 4.3 Vision stack
- `visual_encoder.py` → `VisualEncoder` loads **V-JEPA 2.1** via torch.hub
  (`vjepa2_1_vit_large_384`, `d_v=1024`). *(The old `vidtok_encoder.py` is gone —
  the deep-dive doc's VidTok / "1×1 sensor-projection conv" text is stale.)*
- `latent_summarizer.py` → `LatentSummarizer`: Perceiver causal cross-attention,
  compresses `[B, T_lat, P, D_v] → [B, n_vis, d_model]`; learned null token for
  macro positions. `n_visual_context_steps=3`.
- `cross_modal_adapter.py` → `CrossModalAdapter`: **late-fusion only**, emits
  `N_soft=1` soft token rows on the batch axis.

### 4.4 Fusion routing (`VisionChronos2Model.forward`, line 569)
- `fusion_mode="late"` → summary tokens → adapter → extra batch rows fused by
  `GroupSelfAttention` at each step.
- `fusion_mode="interleaved"` → `interleave_sequences()` (line 76) weaves visual
  tokens into the **refinement window only** → single `[B, T_ctx+n_vis+T_fut, d]`
  sequence; Grassmann sees cross-modal pairs.
- `skip_vision_stack=true` → vision modules not constructed (numeric-only).
- `MultimodalEmbedding` adds modality / segment / token-type embeddings.

### 4.5 Output
- Non-autoregressive multi-token quantile head; last `T_fut` hiddens → 9
  quantiles × output patch; arcsinh instance-norm inverted to physical units.
- Loss: masked pinball. **⚠ `future_covariates_mask` is set to all-zeros**
  (lightning_module.py:359-362, "C1 fix") → known future weather is treated as
  *unknown* to avoid a degenerate zero-loss; the model does not exploit it the
  way `chronos2_oracle` baselines do.

## 5. Optimizer / schedule (`lightning_module.configure_optimizers`, line 848)

- AdamW, param groups: backbone (`lr × backbone_lr_ratio=0.1`), new modules
  (`lr=1e-4`), separate Grassmann group.
- Linear warmup (`warmup_steps=500`) → cosine decay to `min_lr_ratio=0.1`.
- `grassmann_warmup_steps` exists but **defaults to 0 and is never set** by any
  config → the proposal's "0.1× LR Grassmann warmup for 2000 steps" does not run.
- `freeze_chronos` path (lines 160-188) exists for a Stage-2a-style vision-only
  pilot, but the reported runs use `freeze_chronos=false` (everything trainable).

## 6. Why the results are poor (evidence-backed)

From `results/mmtsfm/*.json` (overall):

| variant | NMAE | NRMSE | SS |
|---|---|---|---|
| grassmann_interleaved | 0.1059 | 0.1514 | 0.343 |
| numeric_grassmann | 0.1069 | 0.1510 | 0.345 |
| selfattn_late | 0.1076 | 0.1513 | 0.343 |
| grassmann_no_modbias | 0.1066 | 0.1527 | 0.337 |

Reference baselines (`baselines/results/ALL_RESULTS.md`): mlp 0.0958, patchtst
0.0886, tft 0.0889, itransformer 0.0699, solar_vlm 0.0955, ts_rag 0.0705,
chronos2_oracle_ft 0.0808. MMTSFM sits near lightgbm/tabpfn/chronos2_ft — the
weak end.

1. **No curriculum → cold-start interference.** Random Grassmann + random
   LatentSummarizer + random MultimodalEmbedding + **random reinit'd patch
   embeddings & output head** are all optimized jointly against the pretrained
   6-layer encoder in one 50-epoch run. The proposal's entire staging exists to
   prevent exactly this; it was skipped.

2. **Pretrained TS knowledge largely discarded.** The reinitialized
   input/output projections (§4.1) are the domain-critical mappings. The
   proposal's "~80% of Chronos-2 transfers" does not hold as-run — the trunk
   loads but the tokenizer/head are scratch. Effectively a small from-scratch TS
   model on a pretrained trunk.

3. **Vision & Grassmann are inert.** `numeric_grassmann` (vision off) ≈
   `grassmann_interleaved` (vision on) ≈ `selfattn_late` (Grassmann off). The two
   headline contributions produce no measurable lift on uk_pv. Consistent with
   prior finding "vision lift ≈ 0, grassmann lift ≈ 0".

4. **Known future covariates unused** (§4.5) — a systematic disadvantage vs
   covariate-aware baselines.

5. **Early stopping at a mediocre optimum** — interleaved run stopped epoch 23/30
   with `val/loss=2.844`, never improving past patience.

## 7. Deviations from the proposal / deep-dive (quick map)

| Proposal claim | Code reality |
|---|---|
| 4-stage curriculum (S1→2a→2b→3) | Not run; single fit. No chaining script. |
| Stage-1 Grassmann warmup (0.1× LR, 2000 steps) | `grassmann_warmup_steps=0`, never set. |
| Stage-wise freezing (freeze Chronos, partial V-JEPA unfreeze) | Supported by flags; reported runs = all trainable. |
| ~80% Chronos-2 params transfer | Trunk loads; input tokenizer + output head + `shared` reinitialized (patch-size mismatch). |
| Known future weather covariates used | `future_covariates_mask=0` → treated as unknown. |
| V-JEPA 2.1 replaces VidTok | ✅ done (`visual_encoder.py`); `vidtok_encoder.py` removed. |
| 1×1 sensor-projection conv (deep-dive §5) | Not present; frames are uniform 3-channel, no projection. |
| Deep fusion beats late fusion | No measurable difference between fusion modes on uk_pv. |

## 8. To actually test the design

1. Add `slurm_curriculum.sh` chaining S1→2a→2b→3 with `+ckpt_path` and the
   documented per-stage flags; set `grassmann_warmup_steps` in the S1 config.
2. Match `input_patch_size`/`output_patch_size` to the pretrained Chronos-2
   checkpoint so the tokenizer + head transfer instead of reinitializing.
3. Stop masking future covariates as unknown (or ablate it explicitly).
4. Re-run the fusion ablations *after* a real curriculum before concluding
   vision/Grassmann add nothing.
