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
| A13 | *"s2c only wins because it gets more visual tokens."* | `model=vision_chronos2_wide` — differs from `vision_chronos2_timeselfattn` by exactly one line, `n_soft_tokens: 1 → 16` | **DONE n=3** | **Killed — publishable null**, but a *weaker* null than it was written as. 16× more *pooled* tokens moved ramp 0.0003 (inside the 0.0011 floor); 64 *spatially arranged* tokens moved it 0.0024. ⚠ The 16 tokens were adapter **copies** of one pooled vector (`CrossModalAdapter` is downstream of the summarizer bottleneck), so the null was guaranteed by construction and says nothing about bandwidth; and the 64-token comparison also swaps the query source. "Token count is not the mechanism" stands; "spatial arrangement is" is not yet separated from "the forecast issues the query" — that is **A29** (§2.2.2). |
| A17 | *"s2c only wins because of the 3-slot forecast head (#2/#3), not the grid."* | `+ablation=A17` — s2c with `vision_cfg.visual_grid=1` | **CONFIG READY — highest value open item** | Holds the 3-position decoder fixed, varies only grid-vs-blob. This is the **one** run that separates the architecture claim from a decoder-granularity artefact. Run at n=3. |
| A09 | *"the model isn't reading the frames at all, it's exploiting a correlate."* | `+ablation=A09` — `model.eval_control=shuffle_frames`, eval-only | **NOT RUNNABLE ON s2c — refuses at `on_test_start`** | Temporal shuffle destroys motion but preserves marginals. On s2c it is a **proven no-op**, confirmed both analytically and empirically — see §2.2.1. Runnable and informative only on an arm whose visual path can represent frame order (`fusion_mode != future_query` **and** `n_visual_context_steps > 1`). |
| A09i | *"…"* — the same claim, recorded as an architecture fact instead of a result | `+ablation=A09i` — A09 plus `model.eval_control_allow_inert=true` | **DONE n=3 — architectural null** | The three `mmtsfm_A09_s2c_ukpv_s4*.json` on disk are this run: every reported metric is **bit-identical** to the corresponding plain s2c run (nmae 0.069239 / 0.071379 / 0.069512; Δ 0.007249 / 0.006497 / 0.007630). That is the receipt, not a finding. **Never report it as "shuffling frames does not hurt s2c".** |
| A10 | *"the grid isn't spatially grounded."* | `+ablation=A10` — `model.eval_control=swap_plant_frames` **plus `data.shuffle_test=true`**, eval-only | **CONFIG READY (rewritten 2026-09) — MANDATORY for s2c, NOT YET RUN** | Mismatched-plant frames, donor matched by `batch["site_id"]`. Needs the shuffled test loader: the ordered loader is series-major, so a batch is one plant and a cross-plant donor does not exist in it. Raises rather than substituting silently. |
| A10b | *"the sky helps, but any recent sky would do."* | `+ablation=A10b` — `model.eval_control=stale_sky`, ordered loader | **DONE n=3 — strongest positive control on record** | Same plant, one horizon earlier. **nmae 0.0692→0.0911, 0.0714→0.0875, 0.0695→0.0917** (mean +0.0201, +29%); Δramp +0.0148 / +0.0113 / +0.0154. The marginal gain **flips sign**: +0.0072 → −0.0147 (mean −0.0129), i.e. a one-step-stale sky is *worse than no sky at all*. Vision is being read, and read for its timing. The three `mmtsfm_A10_s2c_ukpv_s4*.json` are these runs, filed under the wrong name — re-file, do not re-run. |
| A22 | *"s2c only wins because of the grid, not the 3-slot head."* — the mirror of A17 | `+ablation=A22` — s2c with `output_patch_size=16`, `max_output_patches=4` (1 decoder position, grid held at 4×4) | **CONFIG READY** | A17 and A22 are only interpretable as a pair: A17 kills the grid, A22 kills the decoder, and which one the gain follows is the attribution. Run at n=3. |
| A23 | *"A01's null is a dropout artefact, not a fusion-mode result."* | `+ablation=A23` — s2a with `visual_dropout_prob=0.5` | **CONFIG READY** | s2a trains at 0.3, s2b/s2c at 0.5, so A01 currently confounds fusion mode with recipe. If Δ stays ~0 the negative result is clean; if it moves, A01 as written must be withdrawn. n=3. |
| A29 | *"s2b's null is a bandwidth artefact, not a fusion-mode result."* — and the mirror: *"s2c wins because its tokens are spatially resolved, not because the forecast issues the query."* | `+ablation=A29` — base `model=vision_chronos2_s2c +stage=s2b`, with `summarizer_time_slices=4`, `summarizer_spatial_grid=4` (64 tokens/step, `n_soft_tokens=1`), `visual_cross_attn_blocks=0` | **CODE + CONFIG READY (2026-09-03)** | The missing cell of the 2×2 in §2.2.2. Widens the `LatentSummarizer` **bottleneck itself** — each of the 64 queries is masked to its own (temporal slice, spatial block), so they cannot collapse onto a shared average the way A13's adapter copies did. Payload is the same 4×4×4 decomposition as s2c's KV set, so the only remaining difference from s2c is *who issues the query*. Run at n=3. |

#### 2.2.1 Why A09 cannot falsify anything on s2c

Established 2026-09-01 by reading `vision_chronos2.py` / `model.py`, then encoded as a
runtime guard (`_assert_eval_control_is_falsifiable`). Two independent routes make a frame
permutation invisible:

1. **`fusion_mode="future_query"` (s2c).** `_build_visual_kv` block-pools `[B, T_lat, P, D_v]`
   to a `g×g` grid and flattens to `[B, T_lat·g·g, D_v]` — 4 slices × 16 cells = 64 keys —
   with **no temporal and no spatial embedding**, under `kv_mask = zeros` (`model.py:102`);
   and `TimeCrossAttention` builds its `MHA(config, use_rope=False)` (`layers.py:450`), so
   there is no positional encoding anywhere on the KV. Softmax over an unordered key set is
   permutation-invariant, so shuffling `T_lat` is *bit-exact* identity. `video_delta_t` is
   unused on this branch, and `visual_mask` enters only as `visual_mask.sum(dim=1) > 0`.
   The s2c path also bypasses `LatentSummarizer` and `CrossModalAdapter` entirely — so
   raising `n_visual_context_steps` alone does **not** make A09 meaningful here.
2. **`LatentSummarizer` with `n_vis_steps == 1`.** No positional encoding on K/V, and the
   single query's causal threshold admits every frame → the summary is a set function.

Consequence: a plain A09 run on s2c writes a `delta` identical to the uncorrupted run, which
reads exactly like the empirical finding *"motion does not matter"* while being a fact about
the wiring. `on_test_start` now raises unless `eval_control_allow_inert=true` (→ A09i). Fixing
this is a **training** change — a temporal embedding on the visual KV — not an eval-config
change; raising `n_visual_context_steps` does **not** help on s2c, because route 1 never
reaches the summarizer.

**Confirmed empirically, 2026-09-01.** The three synced `mmtsfm_A09_s2c_ukpv_s4*.json` agree
with the plain s2c runs to every printed digit — nmae 0.069239 / 0.071379 / 0.069512,
`delta_nmae` 0.007249 / 0.006497 / 0.007630, vision-off 0.076489 / 0.077876 / 0.077142 — while
A10b on the same checkpoints moves nmae by +0.020. The invariance is not a theoretical worry;
it is what the numbers already on disk show.

Provenance note: the controls carry `dataset_version dataset_all.parquet:92099550:…` against
the baselines' `:92166811:…` — the parquet was rewritten between 2026-08-29 and 2026-09-01.
Bit-identical vision-off metrics across the two prove the rewrite did not change the uk_pv
test windows, so control-vs-baseline differences remain valid. Re-check this if the file
changes again.

Related fix, same commit: on the V-JEPA cached-latent path the frame permutation never reached
`visual_mask` (the gate required `video is not None`). Harmless on s2c, where the mask enters
only as `visual_mask.sum(dim=1) > 0`, but it would have silently corrupted availability on any
arm where A09 is actually meaningful.

#### 2.2.2 The 2×2 the registry has only half of

"Spatial arrangement is the mechanism" (A13's verdict) rests on a comparison that moves two
variables at once. Every s2c-vs-s2b contrast changes **both** the payload (1 pooled blob vs 64
resolved tokens) **and** who issues the query (fixed learned latents vs the forecast
positions). Four cells, two of them never run:

|                      | payload pooled to a blob | payload resolved 4×4×4 |
|----------------------|--------------------------|------------------------|
| **fixed latent queries** | s2b / A02 (n=1) ✅        | **A29** ← the missing cell |
| **forecast queries**     | A17 (s2c, grid=1) — config ready | s2c / A16 ✅ |

A13 is *not* the top-right cell. `CrossModalAdapter` sits **downstream** of the summarizer
bottleneck, so `n_soft_tokens: 1 → 16` fans one pooled `d_model` vector into 16 copies — the
null was guaranteed a priori and carries no information about bandwidth. A29 widens the
bottleneck itself (`summarizer_time_slices` × `summarizer_spatial_grid²` queries, each masked
to its own temporal slice and spatial block, so they cannot collapse onto a shared average).

Reads:
- **Δramp ≈ 0 again** → the payload was never the constraint. The sequence axis cannot deliver
  visual signal to this backbone, and s2c's mechanism is the forecast-side query, not the grid.
- **Δramp > 0.0011** → s2b's null is a bandwidth artefact, the "arrangement" attribution in A13
  and in the §2.2 narrative must be rewritten, and A17/A22 become the *secondary* pair.

A29 is also the only arm on the board where **A09 becomes both permitted and informative**,
and it does so *without* raising `n_visual_context_steps`. Route 2 of §2.2.1 — "the single
query's causal threshold admits every frame, so the summary is a set function" — is broken by
`summarizer_time_slices > 1`: the slices partition `T_lat` **inside** the one visual step, so
a frame permutation moves frames between slices and the summary changes. At `n_time_slices=4`,
`n_vis=1` the causal threshold admits everything, so no sub-query takes the spatial-only
fallback and the time partition is exact.



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

1. **A10 (+ A09i, A10b bookkeeping)** — the negative controls. Eval-only, minutes against
   checkpoints already on disk. **A10** is now the only one of the three that can still
   falsify anything: it needs `data.shuffle_test=true` and it is the run that decides whether
   the 4×4 grid is spatially grounded. **A09i** is a one-shot receipt for an architectural
   null (§2.2.1) and **A10b** is already measured — re-file the synced A10 JSONs under that
   name rather than re-running them.
2. **A02** — resume s2b seeds 42/43. Without it the headline comparison rests on n=1.
3. **A17 + A22** — the attribution pair, n=3 each. Only interpretable together: A17 kills the
   grid, A22 kills the 3-slot decoder, and which one the gain follows *is* the claim.
   **A29** (§2.2.2) is the third leg and arguably the cheapest: it is the only run that
   separates "spatially resolved payload" from "forecast-side query", and it decides whether
   s2b's null is a fusion-mode result or a bandwidth artefact. Run it with A17/A22, n=3.
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
