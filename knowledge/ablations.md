# Ablation registry

**Canonical for**: which experiments exist, what they are for, and their status. Every run is
registered here **before** it launches (`/register-experiment`). Hypothesis ladder →
[scope.md](scope.md) · fairness rules → [protocol.md](protocol.md) · known architecture
defects → [architecture.md](architecture.md) · live plan of record → `.scratch/ramp-gap/map.md`. Launch procedure → [running-ablations.md](running-ablations.md).

**Measured numbers live in `baselines/results/*.json`.** The aggregate view
`baselines/results/ALL_RESULTS.md` **does not currently exist** — regenerate with
`uv run python baselines/scripts/aggregate_all.py` before quoting anything downstream.

## 0. What is being defended

Three fusion arms are candidates for publication. All share the same Chronos-2 numeric
backbone and the same frozen V-JEPA 2.1 ViT-L/16 visual features; they differ only in *where
and how* visual information enters the sequence.

| Arm | Config | Mechanism |
|---|---|---|
| **s2a** late fusion | `vision_cfg.fusion_mode="late"` | visual latents → `LatentSummarizer` → `CrossModalAdapter` → N soft tokens appended after the series |
| **s2b** deep token fusion | `fusion_mode=interleaved` | pooled visual tokens woven into the refinement window of the TS sequence |
| **s2c** cross-attention fusion | `fusion_mode=future_query` | V-JEPA field block-pooled to 4×4 grid × 4 temporal slices = 64 KV tokens, cross-attended by 3 *future* decoder positions in the last 4 encoder blocks |

Two P0 metrics: generalization skill score and **ramp NMAE** (top-decile \|Δy\|, protocol.md §5).
Pre-registered seed floors: **ramp NMAE 0.0011**, **skill score 0.0037**. A difference smaller
than its floor is a null, not a win.

**Marginal gain** (`model.compute_marginal_gain=true`) is the load-bearing statistic: a second
forward pass over the *same trained weights* with vision forced off. `delta_nmae` /
`delta_nmae_ramp` separate *model reliance on the images* from *training-recipe side effects*.
An arm that scores better but has Δ ≈ 0 did not learn to see.

## 1. Measured state (derived snapshot — regenerate, do not trust indefinitely)

uk_pv, 14 disjoint test plants, 165,295 scored steps, `selfattn` temporal mixer throughout.
Seeds 42/43/44 unless noted. Reproduce from `baselines/results/mmtsfm_*.json`.

| Arm | n | Skill score | ramp NMAE | Δ NMAE (vision-off) | Δ ramp |
|---|--:|---|---|---|---|
| s1 TS-only | 3 | 0.5230 ± 0.0041 | 0.1506 ± 0.0010 | — | — |
| s2a late | 3 | 0.5258 ± 0.0043 | 0.1487 ± 0.0010 | 0.0008 ± 0.0005 | 0.0000 ± 0.0015 |
| s2b interleaved N=1 | **1** | 0.5322 | 0.1487 | 0.0014 | 0.0006 |
| s2b_wide N=16 | 3 | 0.5352 ± 0.0026 | 0.1484 ± 0.0010 | 0.0022 ± 0.0005 | 0.0002 ± 0.0016 |
| **s2c future-query** | 3 | **0.5470 ± 0.0060** | **0.1461 ± 0.0020** | **0.0071 ± 0.0006** | **0.0056 ± 0.0006** |

Reading: only s2c has a vision-off delta that clears the ramp floor by a wide margin. s2a's
score gain survives deleting the images — it is a recipe effect, not a visual one.

**No Grassmann result exists anywhere on disk.** Every number above is `selfattn`.

## 2. Registry

Status legend: `DONE` = ran, n=3, verdict callable · `PARTIAL` = ran, under-seeded ·
`BLOCKING` = paper cannot ship without it · `DEFER` = out of scope for this paper ·
`FOLDED` / `DEPRECATED` = kept for ID continuity only.

### 2.1 Fusion arms

| ID | Hypothesis | Config | Status | Verdict |
|----|------------|--------|--------|---------|
| A01 | Late fusion (s2a) reads the images | `model.vision_cfg.fusion_mode="late"` | DONE n=3 | **H1 falsified for late fusion.** Δramp 0.0000 ± 0.0015 — the ramp gain over s1 survives forcing vision off. Report as a negative result, not as a win. |
| A02 | Interleaved fusion (s2b) reads the images | `model.fusion_mode=interleaved` | **PARTIAL n=1 — BLOCKING** | s42/s43 killed mid-run, never resumed. Only `..._selfattn_s44` on disk. s2c's headline is measured *against this control*; at n=1 the comparison is not defensible. Resume the two seeds. |
| A16 | Future-query cross-attention (s2c) reads the images | `model=vision_chronos2_s2c`, `fusion_mode=future_query` | DONE n=3 | **SUPPORTED** on the ticket-17 gate against both controls: vs s2b(n=1) Δramp +0.0026, vs s2b_wide(n=3) Δramp +0.0024; all 3 seeds improve in both. Below the STRONG tier (0.00275). Δ vision-off 0.0056 — the only arm that visibly uses the images. |

### 2.2 Controls that isolate the s2c claim

s2c changes **five** things at once relative to s2b. Only #1 is the intended variable:

1. `fusion_mode` interleaved → future_query
2. `output_patch_size` 16 → 4, i.e. **1 decoder position → 3** (nothing to do with vision)
3. new `future_patch_embedding`, fresh trainable weights
4. bypasses `LatentSummarizer` + `CrossModalAdapter`; 4×4 spatial grid instead of a 1×1 pooled blob
5. warm-starts from **s1**, not s2a

| ID | Rival explanation it kills | Config | Status | Verdict |
|----|---------------------------|--------|--------|---------|
| A13 | *"s2c only wins because it gets more visual tokens."* | `model=vision_chronos2_wide` — differs from `vision_chronos2_timeselfattn` by exactly one line, `n_soft_tokens: 1 → 16` | **DONE n=3** | **Killed — publishable null.** 16× more *pooled* tokens moved ramp 0.0003 (inside the 0.0011 floor); 64 *spatially arranged* tokens moved it 0.0024. Token count is not the mechanism; spatial arrangement is. Was previously logged as TODO; it is done. |
| A17 | *"s2c only wins because of the 3-slot forecast head (#2/#3), not the grid."* | `+ablation=A17` — s2c with `vision_cfg.visual_grid=1` | **CONFIG READY — highest value open item** | Holds the 3-position decoder fixed, varies only grid-vs-blob. This is the **one** run that separates the architecture claim from a decoder-granularity artefact. Run at n=3. |
| A09 | *"the model isn't reading the frames at all, it's exploiting a correlate."* | `+ablation=A09` — `model.eval_control=shuffle_frames`, eval-only | **CONFIG READY — MANDATORY**, eval-only, cheap | Temporal shuffle destroys motion but preserves marginals. Expected: s2c degrades, s2a does not. |
| A10 | *"the grid isn't spatially grounded."* | `+ablation=A10` — `model.eval_control=swap_plant_frames`, eval-only | **CONFIG READY — MANDATORY for s2c** | Mismatched-plant frames. The 4×4 grid claim requires this; for s2a/s2b it is optional. |
| A22 | *"s2c only wins because of the grid, not the 3-slot head."* — the mirror of A17 | `+ablation=A22` — s2c with `output_patch_size=16`, `max_output_patches=4` (1 decoder position, grid held at 4×4) | **CONFIG READY** | A17 and A22 are only interpretable as a pair: A17 kills the grid, A22 kills the decoder, and which one the gain follows is the attribution. Run at n=3. |
| A23 | *"A01's null is a dropout artefact, not a fusion-mode result."* | `+ablation=A23` — s2a with `visual_dropout_prob=0.5` | **CONFIG READY** | s2a trains at 0.3, s2b/s2c at 0.5, so A01 currently confounds fusion mode with recipe. If Δ stays ~0 the negative result is clean; if it moves, A01 as written must be withdrawn. n=3. |

### 2.3 s2c mechanism sweeps (do after A17, for the analysis section)

| ID | Question | Config | Status |
|----|----------|--------|--------|
| A18 | How much cross-attention depth is needed? | `+ablation=A18a` / `A18b` — `chronos_core_cfg.visual_cross_attn_blocks` 1 / 2; k=4 is the existing s2c number, so only two new runs | **CONFIG READY** — low priority, but D15 shows blocks 1–2 near-flat, so k=1 is live and would be a cheaper published architecture |
| A19 | Does the learned lead-time embedding τ matter? | `+ablation=A19` — `vision_cfg.use_lead_time_embed=false`; the parameter is then not constructed and `forward` falls through its `is not None` guard | **CONFIG READY** — load-bearing for D15: says whether τ or sequence position separates the 3 future queries |
| A20 | Finer decoder granularity | `+ablation=A20` — `output_patch_size=1`, `max_output_patches=12` → 12 future positions | **CONFIG READY** — low priority |
| A21 | Finer spatial grid | `+ablation=A21` — `vision_cfg.visual_grid=14` (native ViT patch grid) | **CONFIG READY** — follow-up only if A17 is positive. ⚠ ~784 KV tokens/sample against s2c's 64; reduce `BATCH_SIZE` |

### 2.4 Diagnostics

| ID | Question | Where | Status |
|----|----------|-------|--------|
| D15 | Do the 3 future queries attend to *different* visual regions, or is the parameterisation degenerate? | `horizon_attention` block in each s2c result JSON; ticket `.scratch/ramp-gap/issues/15-*` | **AMBIGUOUS — report both numbers, claim neither** |

D15 detail, because the one-word verdict is misleading. All 3 seeds print
`degenerate_queries_collapsed`, but that verdict is driven by the absolute gate, not the
self-calibrating one:

- `separation_ratio` = 10.56 / 10.31 / 11.89 against threshold **2.0** → queries differ ~10×
  above their own noise floor. This gate **passes comfortably**.
- `min_between_l1 = 0.05` applied to the **4-block mean** = 0.0394 / 0.0466 / 0.0469 → **fails**.
- Block 0 alone clears it easily: s42 `[0.0631, 0.0886, 0.0372]`, s43 `[0.1122, 0.1563, 0.0494]`,
  s44 `[0.1246, 0.1512, 0.0570]`. Blocks 1–2 are near-flat and drag the mean under.

Publication-safe wording: the queries differentiate in the first cross-attention block and
flatten in later ones; do not claim learned advection, do not concede degeneracy.

### 2.5 Component ablations

| ID | Hypothesis | Config | Status | Verdict |
|----|------------|--------|--------|---------|
| A03 | Grassmann mixer vs self-attention | `model.chronos_core_cfg.use_grassmann=true` | **TODO — DECISION REQUIRED** | Zero Grassmann results exist. Either it runs at n=3 on the winning arm, or Grassmann is cut from the paper's claims. Standing decision 4 in `.scratch/ramp-gap/map.md` says the *fusion mechanism* is the contribution and Grassmann is negotiable — cutting it is the cheaper honest option. |
| A04 / W5 | Visual window 3h vs 6h vs 12h | `data.visual_window_hours=...` | TODO — medium | Larger effect expected on s2c (its 4 temporal slices tile this window directly) than on s2a/s2b. |
| A11 | Vision-only upper bound | no config key exists — needs a `numeric_dropout_prob=1.0` variant or new code | TODO — low | Nice-to-have context for the marginal-gain numbers. |
| A12 | Modality grid TS / TS+cov / TS+vis / full | no `model.inputs` key exists; the covariate arm needs `data`-side work | Half-obsolete | The vision-off arm is already covered by `compute_marginal_gain`. Only the covariate arm adds information; drop the other three. |
| A14 | Frozen vs partially-unfrozen visual backbone | `train_strategy.n_visual_unfreeze_layers` / `progressive_vision_unfreeze` | **RECLASSIFIED — Limitation, not an ablation** | Not runnable as written. With `VJEPA_CACHE` set, `_unpack_batch` fills `video_latents` and leaves `video=None`; both fusion branches consume the cached latents and never call the encoder. `_apply_vision_unfreeze_policy` flips `requires_grad` on modules that are outside the autograd graph. Fine-tuning the backbone requires re-running V-JEPA in-loop. State as a limitation. See architecture.md. |
| W4 | Cross-plant group batching (`num_entities>1`) improves cross-plant skill | `data.num_entities=4` (train only) | CODE DONE, no numbers — see **A28** | Needs one number or it drops out of the paper. CPU test verified group disjointness + cross-entity gradient only. |
| W6 | Marginal gain confirms the visual stream is used | `model.compute_marginal_gain=true` | **DONE** | Present in every result JSON; the `Δ` columns in §1 are this. Registry previously said "deferred". |
| W7 | `n_visual_context_steps` derived from cadence, bounded by T_ctx | derived; asserted at model init | DONE | Internal correctness, not a paper claim. |
| A24 | Does the token-type embedding (target / covariate / visual — the "M1 fix") earn its place? | `+ablation=A24` — `vision_cfg.disable_token_type_embed=true` | **CONFIG READY** | Added to disambiguate the packed sequence and never ablated since. The table is still built when disabled, so the `state_dict` shape is unchanged and the run warm-starts from any existing checkpoint. |
| A25 | Does the modality embedding (numeric / visual) earn its place? | `+ablation=A25` — `vision_cfg.disable_modality_embed=true` | **CONFIG READY** | Most likely null on s2c, where visual tokens never enter the sequence and are reached only through cross-attention. That prediction is itself worth reporting. |
| A26 | Does the segment embedding (context / future) earn its place? | `+ablation=A26` — `vision_cfg.disable_segment_embed=true` | **CONFIG READY** | Third channel of the same leave-one-out. A24–A26 together are the component-effectiveness table. |
| A27 | Is `numeric_dropout_prob` load-bearing, or does it manufacture the marginal gain? | `+ablation=A27` — `model.numeric_dropout_prob=0.0` | **CONFIG READY** | Numeric dropout (p=0.1, effective 0.1 × (1−0.5)) has never been justified by a measurement, and it forces occasional vision-only prediction on the arm whose headline claim is that it uses the images. |
| A28 | Does the entity embedding do anything? (also the missing W4 number) | `+ablation=A28` — `vision_cfg.n_entities=4` + `data.num_entities=4` | **CONFIG READY** | Every model config sets `n_entities: 0` while `data/ukpv.yaml` declares 4, so `entity_embed` is `None` and `add_entity` is a pass-through: the plants in a group batch are currently indistinguishable to the model. |

### 2.6 Baselines, retrieval, and folded IDs

| ID | Item | Status |
|----|------|--------|
| A00 | Chronos-2 zero-shot baseline | **TODO — still missing.** Only `chronos2_oracle` and `chronos2_oracle_ft` exist on disk; both are a different (oracle-covariate) tier. The untuned backbone is the natural floor for every MMTSFM arm — run it. Other zero-shot FM baselines (`timesfm_zs`, `tirex_zs`) are present |
| A05 | Cross-plant held-out | FOLDED into the evaluation protocol (baselines.md §4.1) — not an ablation |
| A06 | Few-shot protocol | DEPRECATED — replaced by disjoint cross-plant test sets |
| A07 | TS-RAG on frozen Chronos-2 | **DEFER** — retrieval is future work per `.scratch/ramp-gap/map.md`. Note: the existing `ts_rag_orig` number is invalid (aggregated over 19 plants including train/val) |
| A08 | Cross-RAG vs TS-RAG | DEFER — same reason |
| A15 | RAG datastore size / top-k sweep | DEFER — same reason |

## 3. Critical path to a defensible paper

Launch lines for every ID below: [running-ablations.md](running-ablations.md).

1. **A09 + A10** — shuffled-frames and swapped-plant-frames controls. Eval-only, minutes
   against checkpoints already on disk, and they decide whether the model reads the sky at
   all. Cheapest thing that can falsify the paper, so it goes first.
2. **A02** — resume s2b seeds 42/43. Without it the headline comparison rests on n=1.
3. **A17 + A22** — the attribution pair, n=3 each. Only interpretable together: A17 kills the
   grid, A22 kills the 3-slot decoder, and which one the gain follows *is* the claim.
4. **A23** — recipe-matched s2a, n=3. A01's null currently confounds fusion mode with a
   `visual_dropout_prob` difference (0.3 vs 0.5); until this runs the negative result is not
   clean enough to publish as one.
5. **A03** — decide: run Grassmann at n=3, or cut it from the claims.
6. **A24–A28, A19, A27** — the component-effectiveness table. Each answers "does this part
   earn its place", which is what a reviewer means by an ablation study.
7. Regenerate `baselines/results/ALL_RESULTS.md`; fix the stale numbers in
   [scope.md](scope.md) (0.5086 / 0.5087 / 0.5284 match nothing on disk) and in
   `report/BASELINE_TEST_REPORT.md` (MMTSFM at SS 0.3432 is pre-RoPE-fix).

**Provenance caveat on everything already on disk.** `run_cfg` used to carry only `seed`,
`model` and `quantile_levels`, and `config_hash` is derived from it — so s1, s2a, s2b and s2c
all hash to `18d5735b73123686` and no existing result JSON identifies its own architecture.
`_run_cfg()` now records the resolved `chronos_core_cfg` / `vision_cfg` / freeze strategy, so
runs from here on are self-identifying; older ones are not, and hashes must not be compared
across the fix. This is why the s2b_wide fusion-mode question in §2.2 cannot be settled from
the results files.

## How to register

1. Add a row above **before** running
2. Create `configs/ablation/<id>.yaml`
3. Branch `exp/<id>-<short-name>`
4. Update Status → DONE with the W&B run ID and the key metric
