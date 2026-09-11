# Does canonical multi-anchor interleaving rescue the interleaving claim?

Label: `wayfinder:issue`
Type: task
Status: claimed
Blocked by: —

## Question

A43 (ticket [28](28-late-raw-control-arm.md)) falsified H2 at matched payload: moving all 98
raw visual tokens from fractional interleaved positions to one integer position is null on
every metric. The map's Standing decision 2 now reads "interleaving carries none of the
s2a→s2d gain".

**A43 did not test interleaving.** It tested *placement of a single visual anchor*. With
`n_visual_context_steps=1` there is exactly one (sky, power) pair in the sequence, and A43
moved that one pair between two positions. A mapping cannot be fitted from one point. The
canonical interleaved layout — the one the literature means by the word, and the one this
work claims — is

```
TS TS … TS │ TS V │ TS V │ TS V │ TS V │ TS V │ future
└─ T_M=37 ─┘└──────────── n_vis = 5 ─────────────┘
```

with **N ≥ 2 (sky, power) pairs in context**. That layout has never been run on the raw
path. This ticket builds it and runs it.

Constraint carried from the user, binding on the design: **no `LatentSummarizer`.** s2b's
pooled soft tokens are the thing ticket [13](13-widen-the-visual-bottleneck.md) identified as
the ceiling. The arm here is `interleaved_raw` (s2d's `VisualPatchProjector`) with `n_vis > 1`
— raw spatially-resolved cells at multiple anchors, no resampler anywhere.

### Why the existing priors do not settle this

Five results were initially stacked against this idea. On inspection none of them bear on
multi-pair in-context calibration:

| Result | What it measured | Why it does not apply |
|---|---|---|
| **A43** (28) | 98 tokens at position `T_M` vs fractional `[T_M, T_M+0.99]` | One anchor. One pair. No mapping to fit. |
| **A30-a** (21) | frame order shuffled *within* the one 6 h window | Intra-anchor ordering, not inter-anchor spacing. |
| **A30-d** (24) | EVS keep-count swept at eval | Sequence-length mismatch, orthogonal. |
| **A10b** (22) | sky replaced by a **mislabelled** one-horizon-stale frame | Mislabelling test. This design labels old frames *as old*, via position IDs. |
| **A30-c** (25) | wrong plant's sky | Proves vision is content-grounded and time-keyed — the **precondition** for this mechanism, not evidence against it. |

A10b is the one that looks closest and is the most misread: it made the model believe stale
sky was current. Here every anchor carries its own position ID, so nothing is mislabelled.
A30-c is actively supporting: if the visual tokens were inert or plant-agnostic, N of them
would be no better than one — but they are neither.

### The two mechanisms this would test

Neither is in the s2d design doc. Both are testable only with N ≥ 2 anchors.

1. **Physics — efficiency residual.** Sky is irradiance forcing; power is output. Their ratio
   is instantaneous plant efficiency, which absorbs soiling, haze, panel temperature and
   clipping. It is **not derivable from the power history alone** (a low-power hour is
   ambiguous between "cloudy" and "dirty panels"). One (sky, power) pair is one noisy
   estimate of that ratio. Five pairs are a trend.
2. **Architecture — in-context calibration.** Chronos-2 is an in-context forecaster:
   `GroupSelfAttention` fuses the target row with 14 covariate rows on the batch axis
   (`knowledge/architecture.md:71`). Every numeric covariate gets many in-context examples of
   its relationship to the target. **Vision currently gets zero** — one anchor is a single
   observation, not an example set. Multi-anchor is what puts vision on the same footing as
   the covariates the backbone was pretrained to exploit.

### Why the geometry blocked it, and what actually unblocks it

`input_patch_size=16` on 30-minute uk_pv data ⇒ **1 TS token = 8 hours**; `context_length=672`
⇒ `T_ctx=42`. `visual_window_hours=6.0` ⇒ the whole visual window falls inside **one** TS
patch ⇒ `n_vis` can only be 1. Hence the `ValueError` at `vision_chronos2.py:1092`.

That guard has been read as an architectural limit — the map's Out-of-scope entry says
between-token interleaving "would need a re-patched backbone". **It would not.** It is a
*data-coverage* limit. Widen the visual window to `n_vis × 8 h` and the anchors exist. The
sequence-building code is already generic:

- `interleave_sequences` (`:77-117`) — its docstring at `:86` is literally the target layout;
  handles any `n_vis`, any `n_soft`.
- `build_interleaved_position_ids` (`:120-140`) — generic; gives each `(TS, V…)` block one
  shared integer position ID, so RoPE treats the block as co-temporal.
- `vis_summary.reshape(B, n_vis, N_vis_tok, -1)` (`:1259`) — sits **below** the `raw_visual`
  branch and already regroups raw projector tokens per anchor.
- `goes_pvdaq` runs `n_vis=2` today on the pooled path.

Nobody wired the raw path shut. The only genuine piece of work is EVS.

### The one real change: EVS must budget per anchor

`evs_select` ranks novelty **globally** over `T × n_cells` and takes a flat top-K. Reshaping
that flat list into `n_vis` groups slices a globally-ranked list arbitrarily — group *i* would
not contain anchor *i*'s tokens. Fix: give each anchor its own budget of `keep / n_vis`, drawn
from its own frames. `vis_frame_idx` already carries the grouping key. Reusing the same code
path means novelty's pinned-frame-0 anchor becomes per-anchor for free, and A38's `random`
mode keeps working unchanged.

## Design

### Sequence

| | s2d (A30) | this arm |
|---|---|---|
| TS context tokens | 42 | 42 |
| visual anchors `n_vis` | 1 | 5 |
| visual tokens total | 98 | 100 |
| tokens per anchor | 98 | 20 |
| future tokens | 1 | 1 |
| **sequence length** | **141** | **143** |
| position scheme | fractional, `[T_M, T_M+0.99]` | integer, one ID per `(TS,V)` block |
| resampler | none | none |
| pool mode | `shuffle` (r=2) | `shuffle` (r=2) |

Same token budget, same raw representation, same cost. **Only the temporal spread differs.**
This is the controlled experiment A43 could not be.

### Config

```yaml
model.vision.n_visual_context_steps:  1    -> 5
model.vision.visual_evs_keep:         98   -> 100    # divisible by 5 => 20/anchor
data.video_frames:                    8    -> 20     # 5 bursts x 4
data.visual_frame_spacing_min:        45   -> 45     # UNCHANGED - the proven step
data.visual_anchor_stride_hours:      -    -> 8.0    # one Chronos-2 patch
data.visual_frames_per_anchor:        -    -> 4
data.visual_window_hours:             6.0  -> 36.0   # (5-1)*8 + burst span
```

**Frames are drawn as five dense bursts, not as a stretched uniform ladder.** This is the
correction to the first draft of this ticket, which specified "the same 8 frames at 5 h
spacing, same cache size". That was wrong three ways:

1. **5 h spacing destroys motion.** §3.3 of the s2d design doc pins csi autocorrelation at
   **~0.78 at 45 min**; at 5 h it is ~0. V-JEPA encodes tubelets at temporal stride 2
   (`knowledge/architecture.md:32`), so two decorrelated frames per tubelet means the
   *latents themselves* degrade before any of this arm's code runs.
2. **EVS starves with them.** `evs_select` novelty scores cosine dissimilarity to the
   previous frame. Decorrelated frames ⇒ every score ≈ 1.0 ⇒ the ranking is noise. That is
   A38's `random` arm, measured at **+0.0039 ramp (3.6x the seed floor)**.
3. **The config was self-refuting.** 8 frames ⇒ `T_lat = 4`, and this ticket's own new
   `T % n_groups` guard rejects `4 % 5`.

The burst ladder keeps the 45-min step *inside* each anchor and spends the distance on the
gaps, where no frame is drawn at all. An anchor has to be **co-temporal** with its TS patch,
not tile it. The anchor START carries the integer position id, so the stride is 8.0 h exactly
— A10b priced a mislabelled age at **-0.135 SS**.

Night is not a blocker: `knowledge/dataset.md:117-130` — v2 non-HRV channels are **IR bands,
not visible**, so cloud fields are observable through the night. Every anchor carries real
signal.

### Cache and quota — BLOCKING pre-flight

Measured, not estimated: the cache of record
`/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224_nonhrv_sp45` is **210 G** at
`T_lat=4`. Latents are already fp16 (`scripts/extract_video_embeddings.py:161`), so there is
no precision lever. Cost is exactly linear in `T_lat`.

| ladder | spacing | frames/anchor | `Tv`/`T_lat` | anchors | disk |
|---|---|---|---|---|---|
| control (s2d) | 45 min | 8 | 8 / 4 | 1 | 210 G |
| uniform, reaching 5 patches | ~5 h | — | 8 / 4 | 5 | 210 G **but useless** |
| uniform, motion-preserving | 60 min | 8 | 40 / 20 | 5 | **1050 G — over quota** |
| **burst (this arm)** | **45 min** | **4** | **20 / 10** | **5** | **525 G** |

Uniform spacing cannot deliver this: it forces frames to tile the whole span, so
`T_lat >= 4 * n_vis` and the disk cost is `n_vis x 210 G`. Bursts break that coupling.

**525 G (new) + 210 G (control, must be kept) = 735 G against a 1 TB quota that has already
overflowed once (32.84 TB view-save incident).** Before extraction:

- [ ] Prune the obsolete v1 HRV cache at `.../uk_pv/vit_large_f8_s224`
      (`scripts/prune_recompress_vjepa_cache.py`).
- [ ] Account checkpoints/logs against the remaining ~265 G.
- [ ] Extract to a NEW directory — cache keys are `{dataset}_{site}_{origin}` and do **not**
      encode spacing, so only the directory name separates ladders. Suggested:
      `vit_large_f20_s224_nonhrv_sp45_a8h`.

**Known confound, stated rather than hidden:** each anchor carries `T_lat=2`, against the
control's 4. This arm is "half-depth s2d x 5", not "s2d x 5". If it wins, a follow-up at
`frames_per_anchor=8` (1050 G, needs quota relief) separates anchor *count* from anchor
*depth*. If it loses, that confound is the first thing to rule out.

### Code changes

`MMTSFM/src/mmtsfm/models/vision/patch_projector.py`

1. `evs_select` — add `n_groups: int = 1`. When `> 1`: require `T % n_groups == 0` and
   `keep % n_groups == 0`, fold groups into the batch axis, recurse at `keep // n_groups`,
   unfold, offset `frame_idx` per group.
2. `VisualPatchProjector.__init__` — add `evs_groups: int = 1`, store, pass through in
   `forward`.

`MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py`

3. `:526-535` — thread `evs_groups=vision_config.n_visual_context_steps` into the projector.
4. `:1092-1100` — replace the blanket `n_vis != 1` `ValueError` with (i) a divisibility check
   on `visual_evs_keep % n_vis` and (ii) a **data-coverage guard**: `visual_window_hours`
   must cover `n_vis` patches. Without (ii) a 6 h cache at `n_vis=5` would emit five tokens
   describing the *same* recent sky at five different claimed timestamps — exactly A10b's
   condition, which measured **worse than no sky at all**. `validate_n_visual_context_steps`
   (`:52-69`) checks only `n_vis <= T_ctx`, so today this fails silently.
5. `:1155` — `N_vis_tok = vis_summary.shape[1] // n_vis` (was the whole K, valid only at
   `n_vis == 1`).
6. `:1302-1322` — gate `build_subpatch_position_ids` on `n_vis == 1`; for `n_vis > 1` use
   `build_interleaved_position_ids(T_M, n_vis, T_fut, device, n_soft=N_vis_tok)`. Fractional
   sub-patch positions are the single-anchor workaround; with real anchors the canonical
   integer scheme is correct and is what the word "interleaving" means.

Blast radius, all four `impact()` calls **LOW**: `VisualPatchProjector` 3 impacted / 1 direct
/ 0 processes; `build_subpatch_position_ids` 2 impacted / 1 direct / 1 process
(`forward_numeric_only`); `validate_n_visual_context_steps` 1 impacted; `_load_vision` 2
impacted / 1 direct (`__getitem__`) / 1 process.

`MMTSFM/src/mmtsfm/data/pv_record.py`

7. `PVRecordDataset.__init__` — add `visual_anchor_stride_hours` and
   `visual_frames_per_anchor`. Both `None` (default) = today's uniform ladder. Validate as a
   pair: set together, positive, `video_frames % frames_per_anchor == 0`, explicit
   `visual_frame_spacing_min` required, and burst span `<=` anchor stride (overlapping bursts
   would make consecutive anchors share frames).
8. `_load_vision:428-441` — replace `want = t_now - k * spacing_sec` with a decomposed offset
   `(k // f) * anchor_sec + (k % f) * spacing_sec`. `f = Tv`, `anchor_sec = 0` is the uniform
   case and is byte-identical, so **the s2d control cache stays valid**.
9. `datamodule.py` + `scripts/extract_video_embeddings.py` — plumb both fields through and
   into the printed extraction meta, so `_extract_meta.txt` records the ladder.

### Status of the code

All nine changes are **implemented**. Test suite **491 passed** (was 483): 19 for per-anchor
EVS and canonical positions, 8 for the burst ladder — including one asserting the uniform
path is bit-identical when burst params are absent, which is what protects the control arm.
Nothing committed.

### Cost

143 vs 141 tokens ⇒ compute effectively **identical to s2d**, so wall-clock is already known
from this project's own s2d SLURM logs — no new estimate needed. 3 seeds = one wave ≈ **24
node-hours** on `boost_usr_prod` (`IscrC_MTSFM`, `trainer.devices=1`, parallelise across runs
never inside one).

The cache extraction is the real cost, not the training: **2.5x the frames of the sp45 pass**
(20 vs 8 per window), plus the quota prune above. Training cannot start until that lands.

### What each outcome means

- **Beats s2d past the floors** (SS 0.0037, ramp NMAE 0.0011) ⇒ interleaving is rescued, and
  rescued *as interleaving*, not as a placement trick. A43 becomes "one anchor is not enough",
  and the paper's contribution is canonical multi-anchor resampler-free interleaving. This is
  the publication the user wants to write.
- **Null** ⇒ H2 stays falsified, but now at the layout the word actually names, which is a
  far stronger negative result than A43 alone. The paper can then say interleaving was tested
  properly and does not help — defensible under review in a way A43 is not.
- **Worse** ⇒ 4 of the 5 anchors are stale sky the model must learn to discount, and the
  discounting costs more than the calibration buys. Reads as an A10b-family result with
  correct labels, which is itself publishable as the boundary of the mechanism.

Every branch is reportable. That is why this is worth 24 node-hours.

## Relationship to other tickets

Feeds ticket [27](27-resampler-foil-and-paper-framing.md), the map's binding question. 27 asks
what the contribution is named now that placement is null. **27 should not be called until
this resolves** — if multi-anchor works, the contribution is interleaving after all and 27's
premise changes; if it is null, 27 gets a much stronger negative to name around.

Overturns the map's Out-of-scope entry "Between-token interleaving … would need a re-patched
backbone". It needs a wider visual window, not a re-patched backbone.
