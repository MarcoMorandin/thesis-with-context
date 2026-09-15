# Ablation registry

**Canonical for**: which experiments exist, what they are for, and their status. Every run is
registered here **before** it launches (`/register-experiment`). Hypothesis ladder →
[scope.md](scope.md) · fairness rules → [protocol.md](protocol.md) · known architecture
defects → [architecture.md](architecture.md) · live plan of record → `.scratch/ramp-gap/map.md`. Launch procedure → [running-ablations.md](running-ablations.md).

**Measured numbers live in `baselines/results/*.json`.** The aggregate view
`baselines/results/ALL_RESULTS.md` **does not currently exist** — regenerate with
`uv run python baselines/scripts/aggregate_all.py` before quoting anything downstream.

## 0. What is being defended

Four fusion arms are candidates for publication. All share the same Chronos-2 numeric
backbone and the same frozen V-JEPA 2.1 ViT-L/16 visual features; they differ only in *where
and how* visual information enters the sequence.

| Arm | Config | Mechanism |
|---|---|---|
| **s2a** late fusion | `vision_cfg.fusion_mode="late"` | visual latents → `LatentSummarizer` → `CrossModalAdapter` → N soft tokens appended after the series |
| **s2b** deep token fusion | `fusion_mode=interleaved` | pooled visual tokens woven into the refinement window of the TS sequence |
| **s2c** cross-attention fusion | `fusion_mode=future_query` | V-JEPA field block-pooled to 4×4 grid × 4 temporal slices = 64 KV tokens, cross-attended by 3 *future* decoder positions in the last 4 encoder blocks |
| **s2d** resampler-free interleaved | `fusion_mode=interleaved_raw` | no `LatentSummarizer`, no `CrossModalAdapter` — V-JEPA field pixel-shuffled r=2 (14×14×1024 → 7×7×4096), 2-layer MLP projector, EVS keeps 98 of 196 tokens, injected into the sequence at fractional positions inside the last context patch |

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
| **s2c future-query** | 3 | 0.5470 ± 0.0060 | 0.1461 ± 0.0020 | **0.0071 ± 0.0006** | 0.0056 ± 0.0006 |
| **s2d interleaved_raw** | 3 | **0.5510 ± 0.0020** | **0.1440 ± 0.0008** | 0.0047 ± 0.0001 | **0.0063 ± 0.0006** |

Reading: only s2c and s2d have a vision-off delta that clears the ramp floor. s2a's
score gain survives deleting the images — it is a recipe effect, not a visual one.

s2d is best on both P0 metrics, but only the **ramp NMAE** win over s2c is callable: paired
per seed it is −0.0020 with all three seeds agreeing, against a 0.0011 floor. The skill-score
gain (+0.0040 mean) flips sign on s42 and sits at the 0.0037 floor — a tie. Δ ramp (+0.0007)
is a tie. Δ NMAE **falls** 0.0071 → 0.0047, all three seeds, i.e. s2d relies on the images
*less* across all hours and *more* on ramps (Δramp/Δnmae 1.34 vs s2c's 0.79) — which is what
A30 predicted, and the reason the two arms are not interchangeable in the manuscript.

**Calibration cost, not yet anywhere else in this file**: s2d degrades coverage\_80
0.768 → 0.731 (nominal 0.80) and quantile ECE 0.0280 → 0.0359, in all three seeds, at flat
CRPS (0.0542 → 0.0544). Sharpness bought the ramp gain. Report it or a reviewer will find it.

**No Grassmann result exists anywhere on disk.** Every number above is `selfattn`.

Provenance: s2d carries `dataset_version dataset_all.parquet:92099550:…` against the other
arms' `:92166811:…` — the same 2026-09-01 parquet rewrite covered in §2.2.1, proven there not
to have changed the uk\_pv test windows. `n_plants` 14 and `n_steps` 165,295 match exactly.

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
| A30 | Removing the resampler (s2d) recovers the ramp signal that s2b's pooled arm destroys | `model=vision_chronos2_s2d +stage=s2d`, `fusion_mode=interleaved_raw`; design → [`specs/2026-09-05-A30-s2d-design.md`](specs/2026-09-05-A30-s2d-design.md) | **DONE n=3 — ⚠ REGISTERED RETROACTIVELY 2026-09-07, ran before this row existed (AGENTS.md §4 violation)** | **SUPPORTED on ramp NMAE, NOT on skill score.** Best arm on both P0 metrics (SS 0.5510 ± 0.0020, ramp 0.1440 ± 0.0008) and best of anything on disk. Paired against s2c per seed: ramp NMAE **−0.0020, all 3 seeds, clears the 0.0011 floor**; SS +0.0040 flips sign on s42 at a 0.0037 floor → tie; Δramp +0.0007 → tie. Δ NMAE **drops** 0.0071→0.0047 (all 3 seeds): reliance concentrates onto ramps, as predicted. **Costs**: coverage\_80 0.768→0.731, ECE 0.0280→0.0359, both in all 3 seeds. **Defensible on content, not on mechanism** — all 5 planned controls done (§2.2.3): A30-b and A30-c strongly support content grounding (right plant, current sky); A30-a, A30-d, A30-e supply no supporting evidence for the design's mechanistic story (frame order, motion-selectivity, resolution). Gate call: `.scratch/ramp-gap/issues/26-call-s2d-gate.md` — report the ramp-NMAE recovery as real, not the design doc's "why" as established. |

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

#### 2.2.3 Controls that isolate the s2d claim — all 5 now on disk; gate called

s2d has an n=3 headline and, as of 2026-09-0x, **all five** planned controls on disk
(A30-a through A30-e — see rows above). They split cleanly on two axes. **Content
grounding — strongly supported**: A30-b (stale sky) and A30-c (swap plant) are both
clean, large, unanimous-across-seeds positives; s2d needs the right plant's current sky.
**Architectural mechanism — no supporting evidence**: A30-a (frame order) came back
near-inert, A30-e (resolution) shows 4×4 beating s2d's shipped 7×7 on the (caveated,
mean-pooling) probe, and A30-d (EVS/motion-selectivity) is confounded by a train/test
sequence-length mismatch rather than genuinely negative. None of the three isolable
architectural variables from design §5.3 supply the mechanism the design doc hypothesized.
**Gate call**: `.scratch/ramp-gap/issues/26-call-s2d-gate.md` — claim the ramp-NMAE
recovery as real (grounded in A30-b/A30-c), not the design doc's mechanistic story as its
explanation.

| ID | Rival explanation it kills | Config | Status |
|----|---------------------------|--------|--------|
| A30-a | *"s2d isn't reading frame order either."* | `+ablation=A09` on s2d | **DONE n=3 (2026-09-07).** Structurally live for the first time (fractional RoPE positions), but measured **near-inert**: Δ ramp NMAE ~0 all 3 seeds (s42 −0.00001, s43 +0.00008, s44 −0.00030), an order of magnitude under the 0.0011 floor. Δ SS small and seed-inconsistent (only s43 clears 0.0037). **Contradicts the design prediction** — s2d's ramp gain does not depend on reading frame *order*. Detail: `.scratch/ramp-gap/issues/21-a09-frame-shuffle-s2d.md`. |
| A30-b | *"any recent sky would do."* | `+ablation=A10b` on s2d | **DONE n=3 (2026-09-07) — SUPPORTED, kills the rival explanation.** Staling the sky costs +0.0156 to +0.0182 ramp NMAE, all 3 seeds, an order of magnitude over floor; SS drops ~0.135 in all 3; vision marginal gain flips from positive to negative in every seed. Same direction/magnitude as s2c's A10b. First positive control on disk for A30. Detail: `.scratch/ramp-gap/issues/22-a10b-stale-sky-s2d.md`. |
| A30-c | *"the grid isn't spatially grounded."* | `+ablation=A10` on s2d, needs `data.shuffle_test=true` | **DONE n=3 (2026-09-0x) — SUPPORTED, cleanly, bigger than A30-b.** Wrong plant's sky costs +0.0163 to +0.0189 ramp NMAE, all 3 seeds; SS drops ~0.18 in all 3 (0.55→0.37); marginal gain flips hard negative (−0.027 to −0.028). Kills the rival explanation outright. Detail: `.scratch/ramp-gap/issues/25-a30c-swap-plant-s2d.md`. |
| A30-d | *"EVS is doing nothing / is doing everything."* | `vision_cfg.visual_evs_keep` sweep q ∈ {0, 0.3, 0.5, 0.7}, eval-only on the trained checkpoint | **DONE n=3 per point (2026-09-0x) — INCONCLUSIVE, not the predicted trend.** U-shaped in keep-count, centered exactly on the trained keep=98: q=0 (no pruning, +0.0102 ramp) worst, q=0.3 (+0.0022) and q=0.7 (+0.0069) both worse than baseline, all 3 seeds agree at every point. Reads as train/test sequence-length mismatch — none of these configs were retrained at their new token count — not motion-selectivity evidence either way. Detail: `.scratch/ramp-gap/issues/24-a30d-evs-qsweep.md`. |
| A30-e | *"7×7 is unmeasured."* | `scripts/probes/latent_pooling_bottleneck.py`, `GRID = 4` → `7` | **DONE (2026-09-08) — does NOT support 7×7.** Non-monotonic across all 3 horizons: 4×4 beats every other arm on ramp R² (t+30 0.0512, t+60 0.0815, t+120 0.0516); s2d's shipped 7×7 loses to 4×4 at every horizon and sits at-or-below the 1×1 floor at t+30 (−0.0020). Caveat: the probe mean-pools to each grid size, s2d's pixel-shuffle concatenates instead (space-to-depth, no averaging), so this doesn't falsify s2d's architecture — but it removes "spatial resolution match" as a supporting story for the shipped 7×7 and raises the weight resting on **A30-d**. First re-run of this probe also fixed a real bug (crop-compounding when GRID isn't a power of 2); 1×1 numbers shifted from the historically-cited 0.0060 to 0.0215 (t+30 ramp) as a result — full native-grid basis now, not comparable to the old figure. Detail: `.scratch/ramp-gap/issues/23-a30e-probe-grid7.md`. |

A29 remains the pooled control that isolates *resolution* from *summarizer removal*; the
summarizer removal itself is the residual and is **not** separately isolable (design §5.3).


#### 2.2.4 s2d component ablations — wave 3 (tickets 28, 40-43)

Registered 2026-09-09. §2.2.3 asked *whether s2d reads the imagery* and answered yes on
content, no on mechanism. This wave asks **which of s2d's four design choices earns its
place**, one knob at a time, all four built as mode switches on the shipped code path so
each foil shares s2d's forward pass exactly rather than forking it.

Protocol for every row (per `knowledge/protocol.md` §4-5, asserted here rather than assumed):
eval on `cross_plant` (disjoint test plants, seed-42 split from
`baselines/configs/splits.json`); n=3 seeds 42/43/44; NMAE + NRMSE + Skill Score, plus ramp
NMAE and the forced vision-off marginal gain (`compute_marginal_gain: true`); compared
against s2d (SS 0.5510 ± 0.0020, ramp NMAE 0.1440 ± 0.0008) and, through it, Smart
Persistence and Chronos-2 ZS. Floors: SS 0.0037, ramp NMAE 0.0011. GPU runs via `sbatch`
(`scripts/ablation_sweep.sh`, `configs/ablation/sweep.manifest`); no data, checkpoints or
logs committed.

| ID | Rival explanation it kills | Config | Branch | Status |
|----|---------------------------|--------|--------|--------|
| A38 | *"EVS is just a sequence-length knob — the novelty score itself does no work."* | `configs/ablation/A38.yaml` — `vision_cfg.visual_evs_mode=random`, **eval-only** (`train: false`) on `uk_pv_s2d_s2d_s{seed}/best.ckpt`. Same 98-token budget, subset drawn uniformly at random instead of by temporal cosine dissimilarity. The rule is parameter-free (asserted in `tests/test_s2d_component_ablations.py::TestRandomEVS`), so it swaps onto trained weights with no retrain — which is exactly what A30-d's q-sweep could not do, because varying `q` moved the sequence length at the same time. | `exp/A38-evs-random` | **DONE (2026-09-09) — REFUTES the hypothesis; the novelty score does real work.** n=2 seeds (42, 43); s44 not run, so this is below the section's n=3 protocol and the numbers are paired per-seed rather than averaged over three. Hypothesis was: *EVS's novelty criterion is inert — swapping it for uniform random selection at equal token count moves s2d's ramp NMAE by less than the 0.0011 seed floor.* It moves it by **+0.0039, 3.6× that floor** (A38 0.1480 vs s2d 0.1440; per-seed +0.0029 / +0.0049), and every other metric degrades in the same direction on both seeds: SS **−0.0152, 4.1× its 0.0037 floor** (0.5347 vs 0.5499; −0.0119 / −0.0184), NMAE +0.0026, NRMSE +0.0035, ramp NRMSE +0.0046, CRPS +0.0021, quantile ECE +0.0059, 80% coverage −0.0217 (0.7056 vs 0.7273). 13 of 14 test plants worse on seed 42. The forced vision-off marginal gain localises it to the visual path rather than to noise: s2d's ramp gain is +0.0065 / +0.0069, A38's is +0.0036 / +0.0019 — random selection destroys roughly half to three quarters of what vision contributes, while the vision-**off** number is untouched. Provenance, from the result JSONs: A38 carries `vision_cfg.visual_evs_mode=random`, s2d has the key absent (it predates the parameter, so it ran the `novelty` default); `visual_evs_keep=98`, `visual_n_cells=49`, `visual_shuffle_r=2`, `fusion_mode=interleaved_raw`, `n_unfreeze_encoder_blocks=3` and the dataset fingerprint are identical across all five files. `git_sha` differs (`b5b1ea4` vs `1346540`) — expected, the mode switch is new code — but the eval-only contract is confirmed by a stronger check: `nmae_ramp_vision_off` is identical to four decimals per seed between the two arms (0.1497 on s42, 0.1517 on s43). Vision-off bypasses EVS entirely, so an identical off-number can only come from the same weights. **Consequence: the manifest's conditional fires — ramp NMAE moved by more than the seed floor, so A38t (retrain under `random`) is now live**, and it is the arm that separates "novelty selection picks better tokens for weights trained on novelty-selected tokens" from "novelty selection is better, full stop". Ticket 42 does not close as a null. |
| A39 | *"the writeup's 'frozen TSFM' claim is safe as written."* | `configs/ablation/A39.yaml` — `model.n_unfreeze_encoder_blocks=0`, train from `uk_pv_s1_selfattn_s{seed}/best.ckpt`. Pure config, no code. s2d sets `freeze_chronos: true` but leaves `n_unfreeze_encoder_blocks: 3`, so 3 of 6 encoder blocks keep training at 0.1x LR (`lightning_module.py:245-254`) — the phrase "frozen TSFM" appears twice in the writeup and is **false as written**. The prose fix lands regardless (`knowledge/architecture.md` 2.6 + manuscript: *"Chronos-2 fine-tuned on train plants (s1), then held fixed except its last 3 encoder blocks while the visual path trains"*); this arm supplies the number for what the strong version of the claim would have been. | `exp/A39-frozen-backbone` | **IN PROGRESS** — registered 2026-09-09, not yet launched. Hypothesis: *s2d's lift survives a genuinely frozen Chronos-2 — with `n_unfreeze_encoder_blocks=0` the vision path alone recovers the ramp gain, so the 'frozen TSFM' claim is true as stated.* Decides which wording ships: retain most of s2d's gain and the strong claim is recoverable as the headline; collapse and the 3-block unfreeze is load-bearing, leaving the corrected wording as the only honest option. **DONE (2026-09-11) — NULL on both P0 metrics; the hypothesis survives on accuracy and fails on calibration.** n=3 (42/43/44), `mmtsfm_A39_s2d_ukpv_s{seed}.json`, `git_sha 15673d1`. Manifest diff vs s2d is exactly one key: `train_strategy.n_unfreeze_encoder_blocks` 3 → 0 (`freeze_chronos: true` and `backbone_lr_ratio: 0.1` unchanged, the latter now inert); dataset fingerprint and `n_plants=14` identical. Skill score 0.5529 vs s2d 0.5510 — paired +0.0008 / +0.0015 / +0.0034, mean **+0.0019, inside the 0.0037 floor**. Ramp NMAE 0.1441 vs 0.1441 — paired +0.0025 / −0.0010 / −0.0012, mean **+0.0001, sign-flipping, inside the 0.0011 floor**. So a genuinely frozen Chronos-2 encoder reads the sky as well as one with half its blocks training: the 3-block unfreeze is not where the fusion gain comes from. What it does buy is calibration — removing it costs coverage_80 0.7310 → 0.7095 (−0.0215) and quantile ECE 0.0359 → 0.0444 (+0.0085), the same metric pair s2d already pays for its ramp gain. The vision marginal gain falls 0.0064 → 0.0055 (−0.0008, sign-flipping, inside the floor): not a result, but the one number directionally against the frozen arm, so do not quote the null harder than "within seed noise". **Consequence:** the strong claim is recoverable *with a caveat that never goes away* — the backbone A39 freezes is the **s1 fine-tuned** Chronos-2, not the pretrained one (`stage/s1.yaml` sets `freeze_chronos: false`), so "frozen TSFM" is true of stage 2 only. The 2026-09-09 corrected wording stands for the shipped s2d arm; whether the paper reports A39 instead is a framing call (ticket 27 / 38). Also the numeric answer to `baselines.md`'s A14 row: partial backbone unfreeze is not required for the fusion gain. |
| A37 | *"the gain comes from having 98 visual tokens, not from what is inside them."* | `configs/ablation/A37.yaml` — `vision_cfg.visual_pool_mode=avg`, train from `uk_pv_s1_selfattn_s{seed}/best.ckpt`. Mean-pools each 2x2 patch block to `[B,T,49,1024]` and projects 1024 -> d_model instead of pixel-shuffling to 4096. Token count, token order, fractional positions, EVS rule, cell embedding and the whole interleaved assembly are identical to s2d (asserted in `tests/test_s2d_component_ablations.py::TestAvgPool`); the sole difference is that the four sub-cells are averaged rather than concatenated — the pooling s2d exists to remove, put back in the one place it can be isolated. Cannot warm-start from s2d: `proj[0].in_features` becomes 1024, so the s1 base is mandatory, not a convenience. | `exp/A37-avg-pool` | **IN PROGRESS** — registered 2026-09-09, not yet launched. Hypothesis: *sub-cell spatial detail, not token count, carries s2d's ramp gain — replacing pixel-shuffle with average pooling at identical token count degrades ramp NMAE beyond the 0.0011 seed floor.* This is the direct test of H1: under it A37 loses most of s2d's gain (0.1440 back toward s2a's 0.1503) with sequence length unchanged; if A37 matches s2d instead, the gain is a length/attention-budget effect and the "resampler-free detail" story in the manuscript is dead. **DONE (2026-09-11) — REFUTES the hypothesis in the strongest direction: averaging is not worse, it is BETTER.** n=3 (42/43/44), `mmtsfm_A37_s2d_ukpv_s{seed}.json`, `git_sha 15673d1`. Manifest diff vs s2d is exactly one key: `vision_cfg.visual_pool_mode = avg`; `visual_evs_keep=98`, `fusion_mode=interleaved_raw`, `n_unfreeze_encoder_blocks=3`, dataset fingerprint and `n_plants=14` identical. Ramp NMAE 0.1435 vs s2d 0.1441 — paired −0.0000 / −0.0014 / −0.0001, mean **−0.0005, inside the 0.0011 floor**, so the predicted collapse toward s2a's 0.1487 did not happen at all. Skill score 0.5564 vs 0.5510 — paired **+0.0058 / +0.0052 / +0.0051, every seed past the 0.0037 floor with no sign flip**, mean +0.0054. Calibration moves the same way: coverage_80 0.7310 → 0.7533 (+0.0223, toward nominal) and quantile ECE 0.0359 → 0.0314 (−0.0045); NMAE 0.06971 → 0.06879. The vision marginal gain is if anything larger (Δ ramp 0.0064 → 0.0073, +0.0009, inside the floor). **Consequence:** sub-cell spatial detail carries nothing — the 4096-wide pixel-shuffle payload is not why the arm works, and the r=2 concatenation costs aggregate accuracy and calibration for no ramp return. The founding claim quoted in `configs/model/vision_chronos2_s2d.yaml` ("CONCATENATED, never averaged — averaging is the operation this arm exists to remove"), design doc §1, and the manuscript's "resampler-free detail" motivation are **withdrawn as stated**. What survives is token *geometry*, which A37 keeps in full: 49 spatially distinct cells, cell embedding, novelty-selected keep=98. Read with A43 (placement null) and A39 (unfreeze null), three of the arm's four distinctives are inert and only EVS selection (A38) has a measured effect. Whether the paper ships s2d or A37 as the reported arm is ticket 27's call, noting A37 is the better arm on three of five headline metrics and never worse than the floor on the other two. |
| A43 | *"s2d's margin over s2a is the resampler removal, not the interleaving — placement does nothing."* | `configs/ablation/A43.yaml` — `vision_cfg.fusion_mode=late_raw`, train from `uk_pv_s1_selfattn_s{seed}/best.ckpt`. s2d's visual payload exactly (pixel-shuffle r=2, 2-layer MLP projector, cell embedding, EVS keep=98) with the **position assignment** switched: all 98 tokens share the single integer position `T_M` (s2b's scheme) instead of spreading fractionally across `[T_M, T_M+0.99]`. A literal group-axis late fusion (s2a's mechanism) is unbuildable for a payload with no time axis, and the token *order* is already identical at `n_vis=1`, so position is the only thing that distinguishes late from interleaved here. Completes a payload × position 2×2 (s2b pooled+integer, s2d raw+fractional, A43 raw+integer). `tests/test_s2d_component_ablations.py::TestLateRaw` asserts identical sequence length to s2d and **zero** fractional positions. | `exp/A43-late-raw` | **DONE (2026-09-11) — CONFIRMS the rival; H2 is falsified.** n=3 (42/43/44), `mmtsfm_A43_s2d_ukpv_s{seed}.json`, `git_sha 15673d1`. Manifest diff vs s2d is exactly one key: `vision_cfg.fusion_mode` `interleaved_raw` → `late_raw`; dataset fingerprint and `n_plants=14` identical. Null on every metric and sign-flipping on every one: skill score 0.5508 vs 0.5510 (paired −0.0002 / +0.0010 / −0.0013, mean −0.0002, floor 0.0037), ramp NMAE 0.1444 vs 0.1441 (paired +0.0016 / −0.0009 / +0.0005, mean +0.0004, floor 0.0011), Δ ramp 0.0064 vs 0.0064 (mean +0.0001), NMAE +0.00007, coverage_80 +0.0015. The reference ladder makes it unambiguous — s1 0.5230 / 0.1506, s2a 0.5258 / 0.1487, **A43 0.5508 / 0.1444**, s2d 0.5510 / 0.1441: A43 keeps the entire +0.0252 SS / −0.0047 ramp margin over s2a *while using s2a's position scheme*. **Consequence:** the confound flagged in map ticket 10 resolves against the headline — the resampler removal carries all of the s2a → s2d gain and the interleaving carries none of it. H2 ("interleaving beats late fusion") is no longer a Limitation to state but a negative result to report, and the contribution cannot be named "placement" or "interleaved fusion". Read with A37 and A39, only EVS novelty selection (A38) still has a measured effect. |
| A36s1 | *"the vision gain is an artefact of being handed next-6h NWP irradiance as a known future input"* — this row kills nothing on its own; it is the control base without which A36s2d kills nothing either. | `configs/ablation/A36s1.yaml` — `data.future_cov=deterministic`, plus `configs/model/vision_chronos2_a36.yaml`, which inherits `vision_chronos2_timeselfattn` and changes no model key. The second file exists only to rename the run: `slurm_curriculum.sh` derives both the checkpoint dir and the results tag from `MODEL_CFG` (`variant_slug` -> `ARM_SUFFIX`), so launching this s1 under the ordinary config would write into `uk_pv_s1_selfattn_s{seed}` and over `mmtsfm_s1_ukpv_selfattn_s{seed}.json` — destroying the shipped s1 that A37/A39/A43 warm-start from. With the A36 model config it lands in `uk_pv_s1_a36_s{seed}`, which is A36s2d's BASE. **Not submittable through `ablation_sweep.sh`**: the manifest's BASE column is a mandatory submit-time checkpoint check and an s1 arm starts from pretrained Chronos-2, so this arm goes through `scripts/slurm_curriculum.sh` with `EXTRA_OVERRIDES="+ablation=A36s1"` (appended last by `scripts/lib/stage_cmd.sh:144-146`, so it beats both model and stage configs). No `INIT_CKPT`; no `compute_marginal_gain` — there is no visual path to switch off. | `exp/A36s1-det-covs` | **IN PROGRESS** — registered 2026-09-09, not yet launched. Hypothesis: *vision-free s1 trained on deterministic-only covariates is the correct control base for A36s2d — look-ahead covariates, not vision, may be carrying the ramp signal.* Every recorded MMTSFM number was produced at `future_cov="all"`, i.e. with cloudcover and four irradiance channels exposed over the horizon; `deterministic` keeps only what is knowable in advance (`baselines/common/config.py::DETERMINISTIC_COVS`). Warm-starting A36s2d from the existing all-covariate s1 would carry an all-covariate numeric optimum into a history-only comparison, so this arm is a prerequisite, not an alternative. Read on its own it also gives the vision-free ramp NMAE under history-only covariates, which is the number the A36s2d - A36s1 delta is taken against. Result: - |
| A36s2d | *"the vision gain is an artefact of being handed next-6h NWP irradiance as a known future input — the visual path is re-deriving cloud information the covariates already leak."* | `configs/ablation/A36s2d.yaml` — `data.future_cov=deterministic` + `model.compute_marginal_gain=true`, model `vision_chronos2_s2d`, stage `s2d`, warm start from `uk_pv_s1_a36_s{seed}/best.ckpt` (A36s1's output, **not** the shipped s1). Manifest row `A36s2d`, submittable through `scripts/ablation_sweep.sh` — but only after all three A36s1 seeds land, because BASE is a submit-time file check and will FATAL on the login node otherwise. The base must be the covariate-matched s1: warm-starting from the all-covariate s1 would import an optimum found under exactly the covariates this arm removes, turning the comparison into a transfer experiment. `deterministic` keeps solar_zenith / solar_azimuth / doy_sin / doy_cos / solar_time / clearsky_ghi and zeroes cloudcover and the four irradiance channels over the horizon (`baselines/common/config.py::DETERMINISTIC_COVS`). | `exp/A36s2d-det-covs` | **IN PROGRESS** — registered 2026-09-09, not yet launched. Hypothesis: *s2d's vision lift is not a proxy for look-ahead covariates — with `future_cov=deterministic` the marginal gain over its own covariate-matched s1 survives.* Read as the pair (A36s2d - A36s1), not against the shipped s2d: the absolute numbers move because the covariate budget shrank, so the quantity under test is the **delta**, plus the forced vision-off marginal gain measured inside this arm. Both must clear the floors (SS 0.0037, ramp NMAE 0.0011). If the delta collapses to the floor while the all-covariate delta (+0.028 SS, -0.0066 ramp) stands, the vision path was reading what the NWP irradiance channels already told it and the paper's central claim narrows to "vision substitutes for NWP look-ahead" — still a result, a different paper. Result: - |
| A38t | *"EVS's novelty rule only looks necessary because s2d was trained on it — any fixed 98-token rule would have worked if the model had learned under it."* | `configs/ablation/A38t.yaml` — `vision_cfg.visual_evs_mode=random`, **train** from `uk_pv_s1_selfattn_s{seed}/best.ckpt` (not from s2d: this arm must learn under random selection, not inherit a novelty-trained projector). Identical to A38 in every other key. | `exp/A38t-evs-random-trained` | **LIVE — conditional fired 2026-09-09, not yet launched.** Registered as conditional on A38 and unblocked by it: A38 moved ramp NMAE 3.6× the seed floor, so the manifest row is uncommented. Hypothesis: *if the novelty criterion carries information rather than co-adaptation, a model trained end-to-end under random selection still fails to reach s2d's ramp NMAE.* This exists because A38 confounds two things — s2d's projector and its 3 unfrozen encoder blocks were fit on novelty-selected tokens, so part of A38's degradation is distribution shift at test time rather than the rule being worse. A38t removes the shift. Recovering s2d's number means EVS is one workable token budget among many and the mechanism claim must soften to that; still losing means novelty selection carries real signal and §2.2.4's mechanism story stands. Result: - |


#### 2.2.5 Canonical multi-anchor interleaving — wave 4 (ticket 45)

Registered 2026-09-11. §2.2.4 retired three of s2d's four distinctives, A43 among them, and
the map's Standing decision 2 now reads "interleaving carries none of the s2a→s2d gain".
**A43 did not test interleaving.** At `n_visual_context_steps=1` there is exactly one
(sky, power) pair in the sequence and A43 moved that one pair between two positions; a
mapping cannot be fitted from one point. This wave runs the layout the word actually names,
`N ≥ 2` pairs in context, on the raw path — the first time it exists in this repo.

Same protocol as §2.2.4: `cross_plant`, n=3 seeds 42/43/44, compared against s2d
(SS 0.5510 ± 0.0020, ramp NMAE 0.1440 ± 0.0008), floors SS 0.0037 / ramp NMAE 0.0011,
`compute_marginal_gain: true`. Unlike §2.2.4 this row is **not** a one-key mode switch: it
needs its own stage + model config and its own V-JEPA latent cache, so it cannot go through
`ablation_sweep.sh` and is submitted per-seed via `scripts/curriculum_stage.sbatch`.

**The anchor stride is set by the data, not by taste.** The first attempt laid anchors one
Chronos-2 input patch apart (8 h) because that is the unit the position ids are built in. It
died on job 57357373 and the post-mortem in [`dataset.md` §2.3](dataset.md) says why: uk_pv
satellite frames exist **02:00–16:00 UTC only**, so an 8 h ladder asks for anchors at clock
times that are night for some of them whatever the origin — the −8 h and −32 h anchors need
an origin in [10:00, 16:00], the −16 h anchor needs [02:00, 08:00], and those do not
intersect. Replayed over 24,605 real origins, **all five anchors populated 0.0 % of the
time**. At a 24 h stride each anchor sits at the origin's own clock time and shares its solar
geometry: **94.4 %**. Nothing is given up by the wider stride — advection is exhausted by
~2 h (`dataset.md` §2.1), so the near anchors were never doing nowcasting; they exist to give
the sky→power mapping more than one point, which is the single thing A43 could not test.

| ID | Rival explanation it kills | Config | Branch | Status |
|----|---------------------------|--------|--------|--------|
| A44 | *"interleaving is inert — A43 showed placement does nothing, so the contribution cannot be named interleaving."* | `configs/stage/s2e.yaml` + `configs/model/vision_chronos2_s2e.yaml`, train from `uk_pv_s1_selfattn_s{seed}/best.ckpt` (**not** from s2d — its projector was fit against one anchor at fractional positions). Diff vs s2d is the anchor count, the anchor *stride*, and the data ladder that makes 5 anchors physically available: `vision_cfg.n_visual_context_steps` 1 → 5, the new `vision_cfg.visual_anchor_patch_stride` 1 → 3 (anchors every three TS input patches = 24 h; stride 1 is the contiguous tail s2b/s2d/A43 used and is preserved bit-for-bit by `anchor_patch_indices`), `visual_evs_keep` 98 → 100 (EVS budgets **per anchor** now — `evs_groups` is wired to `n_vis`, so keep must divide by 5; a global top-K would slice a globally-ranked list arbitrarily and group *i* would not hold anchor *i*'s tokens), `data.video_frames` 8 → 20, `visual_window_hours` 6.0 → 100.0, plus the new burst pair `visual_anchor_stride_hours=24.0` / `visual_frames_per_anchor=4`. Frames are drawn as five dense bursts of 4 at the proven 45-min step, each burst starting exactly 24 h (three Chronos-2 patches) before the last, oldest frame at 98.25 h — **not** a stretched uniform ladder, which would decorrelate the two frames inside a single V-JEPA tubelet (csi autocorrelation ~0.78 at 45 min, ~0 at 5 h) and reduce EVS to A38's `random` arm. 24 h and not 8 h because the 8 h ladder is unrealizable on uk_pv (0.0 % of origins have all five anchors populated against 94.4 % at 24 h — see the paragraph above and `dataset.md` §2.3). At `T_ctx=42` the anchors land on context patches **29, 32, 35, 38, 41**, newest co-temporal with the origin. Positions become integer, one shared id per `(TS, V)` block, instead of s2d's fractional sub-patch scheme. Sequence length 143 vs s2d's 141, so compute is effectively identical. Guarded in `vision_chronos2.forward`: `T_lat % n_vis` plus a data-coverage check `(n_vis - 1) * stride * span` that refuses a narrow cache rather than silently reproducing A10b — it is what caught the 8 h ladder (job 57357373: *"visual window spans only 26.2 h"*). The cross-file invariant `visual_anchor_stride_hours == visual_anchor_patch_stride × 8 h` has no runtime reconciler and is asserted in `MMTSFM/tests/test_a44_strided_anchors.py::TestS2eConfigGeometry`; slot-stable burst placement (survivors must **not** left-pack into a neighbouring anchor's latent group) in `MMTSFM/tests/test_a44_burst_ladder.py`. Payload held at s2d's `visual_shuffle_r=2` despite A37 — swapping the payload at the same time as the anchor count would confound the one variable this arm moves. **Requires a separate 525 G latent cache** (`vit_large_f20_s224_nonhrv_sp45_a24h`; keys are `{dataset}_{site}_{origin}` and do not encode the ladder, so only the directory name separates it from the 210 G s2d control — the abandoned `..._a8h` directory from the first attempt is KNOWN-WRONG and must be deleted, not reused). | `exp/a44-daily-anchors` | **IN PROGRESS** — registered 2026-09-11, geometry revised to daily anchors 2026-09-12 after the first launch attempt died in the coverage guard (job 57357373, all three seeds); **blocked on the cache extraction** (~20 h GPU) and on the quota pre-flight, 525 G + 210 G against a 1 TB project quota that has already overflowed once — the known-wrong `..._a8h` extraction has to be removed first. Hypothesis: *interleaving is not inert — it was never tested. With five (sky, power) pairs in context the model can fit an efficiency residual it cannot read off the power history alone, and A44 beats s2d past both floors.* Two mechanisms, neither testable at `n_vis=1`: **physics**, sky/power ratio is instantaneous plant efficiency and absorbs soiling, haze, panel temperature and clipping, so one pair is one noisy estimate of it and five are a trend — and at a 24 h stride the five are sampled at the *same solar geometry* on five consecutive days, which removes zenith angle as the thing that varies between them; **architecture**, Chronos-2 is an in-context forecaster whose `GroupSelfAttention` gives every numeric covariate many in-context examples of its relation to the target, and vision currently gets zero. **Known confound, stated not hidden:** each anchor carries `T_lat=2` against the control's 4, so this is "half-depth s2d × 5", not "s2d × 5"; a follow-up at `frames_per_anchor=8` (1050 G, needs quota relief) separates anchor count from anchor depth, and is the first thing to rule out if A44 loses. **Second confound, new with option A:** the anchors are now a *day* apart rather than a shift apart, so a win is evidence for multi-pair interleaving, not for recent-sky nowcasting — the two are separable only by a stride sweep, which the diurnal frame coverage makes impossible on uk_pv below 24 h. All three outcomes are reportable — a win names the contribution *canonical multi-anchor resampler-free interleaving*; a null leaves H2 falsified at the layout the word actually names, which is far stronger than A43 alone; a loss reads as an A10b-family result with correct labels and bounds the mechanism. Feeds ticket 27, which should not be called until this resolves. Detail: `.scratch/ramp-gap/issues/45-canonical-multi-anchor-interleaving.md`. Result: **LOSS, and CONFOUNDED — do not report as a negative on interleaving.** Seeds 42/44 scored SS 0.53921 / 0.54135 (mean **0.54028**, n=2) against s2d's 0.54999 / 0.54972 / 0.55324 (mean **0.55098**, n=3): **−0.01070**, ~2.9× the 0.0037 seed floor, and negative on both paired seeds (−0.0108, −0.0119). **Seed 43 never ran** — two seeds are not a result and no CI is available until it does. The loss is NOT attributable to the anchor count as it stands: three variables moved together, anchors 1→5, latents per anchor 4→2, and EVS keep per anchor 98→20 (candidates 196→98). s2e is "impoverished s2d × 5", not "s2d × 5". **A45a** isolates the last two at one anchor and splits the −0.01070; until it resolves, this row supports no claim about interleaving in either direction. |


| ID | Rival explanation it kills | Config | Branch | Status |
|----|---------------------------|--------|--------|--------|
| A45a | *"five daily anchors hurt — A44 measured −0.0107, so multi-anchor interleaving is a negative result."* | `+ablation=A45a` on `stage/s2d` — **one** anchor, carrying one s2e anchor's payload. Three things moved between s2d and s2e at once (anchors 1→5, latents per anchor 4→2, EVS keep per anchor 98→20); A45a holds the anchor count at s2d's 1 and applies the other two, so the −0.01070 splits into "per-anchor impoverishment" and "anchor count". Knobs: `data.video_frames` 8 → 4, the new `data.visual_latent_keep_newest` = 2, `vision_cfg.visual_evs_keep` 98 → 20. **Costs no extraction.** `visual_latent_keep_newest` trims the cached `[T_lat=4, P, D_v]` stack to its newest 2 latents, so this arm reuses the **existing 210 G s2d cache** (`vit_large_f20_s224_nonhrv_sp45`, read-only) — no 525 G / 1050 G quota pre-flight, no new extraction, and the `..._a24h` blocker does not apply. Newest-2 is the exact analogue of one s2e burst (4 frames at 45 min = 2 latents spanning the newest 2.25 h). `video_frames == 2 × visual_latent_keep_newest` is required and raises otherwise, because `video_delta_t` is built from the configured ladder while `Z` comes from the cache — a mismatch silently relabels which frames each latent describes. Guarded + pinned in `MMTSFM/tests/test_a45a_latent_depth.py` (slice takes the newest not the oldest; ladder/slice agreement; a cache shallower than the request raises). Sequence 42 TS + 20 visual + 1 future = **63**, the cheapest arm in this registry. Seeds 42/43/44 from `uk_pv_s1_selfattn_s{seed}/best.ckpt`, the same warm start s2d and s2e use. | `exp/a44-daily-anchors` | **CONFIG READY** — registered 2026-09-14, not launched. Decision rule, against the 0.0037 seed floor: **A45a ≈ s2e** → the −0.01070 is per-anchor impoverishment, the anchor count is neutral, and A44 must not be reported as a negative on interleaving; **A45a ≈ s2d** → impoverishment is free and the −0.01070 *is* the anchor count, so A44 stands and ticket 27 can be called; **in between** → report the split and claim neither whole. **What A45a does NOT control:** the anchor *stride*. s2e's anchors are 24 h apart and A45a has one anchor, so daily spacing stays untestable — as it is anywhere on uk_pv below 24 h (0.0 % origin coverage at 8 h, `dataset.md` §2.3). A45a separates depth+density from anchor **count**, never count from spacing. Blocked with A44 on s2e **seed 43**, which has to run for either row to have a comparand at n=3. Result: - |
| A46 | *"S2E underperforms because its 20-token per-anchor EVS budget is exhausted by 49 infinitely pinned first-latent cells, so the intended temporal-novelty payload never reaches Chronos-2."* | `+ablation=A46` on `stage/s2e`, `model=vision_chronos2_s2e`: `visual_evs_mode=paired_novelty`. For each two-latent anchor, rank the 49 spatial trajectories by cosine change and keep both endpoints of the top 10. Holds five daily anchors, cache, 100-token total, sequence length, warm-start, freeze policy and optimizer fixed against A44. Seeds 42/43/44 from matching `uk_pv_s1_selfattn_s{seed}/best.ckpt`. Standard comparators: A44/S2E for the selector effect and S2D for forecasting quality. Report cross-plant NMAE, NRMSE, NRMSE skill score, ramp errors, calibration and same-weight vision-off marginal. | `exp/a46-paired-novelty` | **IN PROGRESS** — registered 2026-09-14, config and selector ready locally; not launched. Primary prediction: A46 restores S2E's visual marginal and improves both P0 metrics relative to A44. A46 remains a five-daily-anchor architecture, so recovery supports the repaired multi-anchor system but does not isolate placement; a matched appended-payload arm is still required to claim that interleaving itself causes the gain. Result: - |

| A46b | *"S2E's −0.0094 against S2D is the anchor count — five interleaved anchors are worse than one."* | `+ablation=A46b` on `stage/s2e`, `model=vision_chronos2_s2e`: **two anchors, each bit-for-bit an S2D anchor.** The arm exists because A46 retired selection (null, 0.5416 ± 0.0031 vs S2E's 0.5409 ± 0.0015) and left the **payload** as the only live explanation: S2E's anchor carries `T_lat=2` and keeps 20 of 98 candidates (20 %), S2D's carries `T_lat=4` and keeps 98 of 196 (50 %), so every S2E anchor is a quarter of an S2D anchor in depth and a fifth in tokens. A46b removes that confound and moves **only the count**. Data: `video_frames` 20 → **16**, `visual_frames_per_anchor` 4 → **8** (two bursts of 8 at the unchanged 45-min step, 5.25 h span → `T_lat=4` per anchor), `visual_window_hours` 100.0 → **30.0** (oldest frame 24 + 5.25 = 29.25 h), `visual_anchor_stride_hours` held at 24.0. Model: `n_visual_context_steps` 5 → **2**, `visual_evs_keep` 100 → **196** (98 per anchor), and `visual_evs_mode` `paired_novelty` → **`novelty`** — forced, not chosen: `paired_novelty` raises *"paired_novelty requires exactly two latent frames per anchor"* at any `T != 2`, and `novelty` is what S2D itself uses, so lineage returns to the control. Verified directly against `evs_select`: at `n_groups=2, keep=196` anchor 0 draws frames [0,1,2,3] (49 pinned + 49 novel) and anchor 1 draws [4,5,6,7] (49 + 49), each leg identical to a standalone S2D call at `keep=98`. Sequence 42 TS + 196 visual + 1 future = **239** vs S2D's 141 and S2E's 143 — launch `BATCH_SIZE=8 ACCUM=2` (`stage_cmd.sh:107`) to hold the effective batch at `ukpv.yaml`'s 16. **Requires a new ~420 G extraction** (`vit_large_f16_s224_nonhrv_sp45_a24h`, ~16 h GPU): the 20-frame `..._a24h` cache cannot be reused, keys are `{dataset}_{site}_{origin}` and do not encode the ladder, so a 16-frame ladder read through it returns wrong tensors silently. Guards all check: 24 h = 3 TS patches × 8 h; `T_lat 8 % n_vis 2 = 0`; `keep 196 % 2 = 0`; `video_frames 16 % 8 = 0`; burst span 5.25 h ≤ 24 h stride. | `exp/a46b-full-payload` | **CONFIG READY** — registered 2026-09-15, revised same day from the keep=490 budget-only design to this depth-matched one; not launched. `configs/ablation/A46b.yaml` + `configs/ablation/A46b.manifest`. Hypothesis: *with the anchor held identical to S2D's, adding a second (sky, power) pair beats S2D — the mapping A43 could not fit from one point becomes fittable at two.* Decision rule against the 0.0037 SS floor: **A46b > S2D** → anchor count helps once the anchor is not impoverished, A44 must not be reported as a negative on interleaving, and the ladder should be widened as far as quota allows; **A46b ≈ S2D** → the multi-anchor hypothesis is dead at its cleanest possible test and S2E's −0.0094 was payload dilution throughout; **A46b < S2D** → interleaved anchors cost something beyond payload, and the cost is in count or stride. **Why two and not five — quota, not preference.** Cache is ~52.5 G per latent frame (525 G at `T_lat=10`, 210 G at `T_lat=4`; same constant from both). Five depth-matched anchors = 40 frames = `T_lat=20` ≈ **1050 G, which exceeds the 1 TB quota standing alone**, before the comparand caches; three ≈ 630 G and only fits by deleting the 525 G `..._a24h` cache that holds S2E and A46, i.e. by destroying the comparands. Two is also *sufficient* for the registered claim — A44's own statement of what interleaving needs is "N ≥ 2 pairs in context, on the raw path", and two is the threshold; five was never the load-bearing number. **Remaining confounds, stated not hidden:** sequence length still moves against S2D (239 vs 141), so a win is "two S2D anchors beat one" and not "interleaving per se"; and the anchor *stride* stays untestable at 24 h, as it is anywhere on uk_pv (0.0 % origin coverage at 8 h, `dataset.md` §2.3). Pre-flight: free the 210 G `..._sp45` s2d cache before extracting (420 + 525 + 210 = 1155 G > 1 TB); do **not** free `..._a24h`. Result: - |

**Launch hold on A45a (2026-09-14):** do not launch it before A46. Its keep=20
configuration inherits A44's first-latent-only selector behavior, so the registered
depth+density interpretation is invalid as written. Revisit it after A46 establishes
the selector contract.

 `+ablation=A18a` / `A18b` — `chronos_core_cfg.visual_cross_attn_blocks` 1 / 2; k=4 is the existing s2c number, so only two new runs | **CONFIG READY** — low priority, but D15 shows blocks 1–2 near-flat, so k=1 is live and would be a cheaper published architecture |
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
1b. **A30's control battery is complete — gate called.** All five controls ran (§2.2.3):
   content grounding (A30-b, A30-c) strongly supported; mechanism (A30-a, A30-d, A30-e)
   unsupported or confounded. `.scratch/ramp-gap/issues/26-call-s2d-gate.md` has the full
   synthesis and the manuscript framing it implies — the ramp-NMAE recovery is real and
   should be reported; the design doc's explanation for *why* should not be claimed as
   established. Nothing left to run on this line; remaining work is writeup, not ablations.
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
