# Component coverage: what is ablated, what is not

Primary-source pass over `MMTSFM/src`, `MMTSFM/configs`, `MMTSFM/scripts`, `baselines/results/*.json`,
and `knowledge/{ablations,architecture,protocol,scope}.md`. Every claim carries a `path:line` or a
result-JSON filename. No code or config was modified; no training was run.

---

## 0. The finding that governs the whole table

**No architectural config key has more than one distinct value across the result JSONs — because no
architectural key is recorded in any result JSON at all.**

`MMTSFM/src/mmtsfm/models/chronos2/lightning_module.py:825-829` builds the entire manifest config as:

```python
run_cfg = {
    "seed": getattr(self.hparams, "seed", 42),
    "model": "mmtsfm",
    "quantile_levels": None,
}
```

`baselines/common/runner.py:214-222` then stores that dict verbatim as `manifest["config"]` and sets
`config_hash = sha256(json.dumps(run_config, sort_keys=True))[:16]`. So `config_hash` is a pure
function of the seed.

Confirmed on disk: `config_hash` is `18d5735b73123686` for every seed-42 run, `a42a1773cf4b5d7c` for
every seed-43 run, and `0545205fb14e8022` for every seed-44 run — identical across
`mmtsfm_s1_ukpv_selfattn_s42.json`, `mmtsfm_s2a_ukpv_selfattn_s42.json`,
`mmtsfm_s2b_ukpv_wide_s42.json` and `mmtsfm_s2c_ukpv_s2c_s42.json`, which are four architecturally
different models.

**Consequence for the deliverable's "distinct values seen across result JSONs" column:** the honest
empirical answer for every architectural key is `1 (not recorded)`. Arm identity survives only in the
filename (`results_tag`, set per model YAML) and in `git_sha`. The column below therefore reports
*distinct values reachable from the filename→config-file mapping*, and flags that this is inference,
not measurement.

**Result files on disk** (`baselines/results/`, 59 JSONs total, 13 MMTSFM):

| Filename stem | Seeds present | Model YAML it maps to |
|---|---|---|
| `mmtsfm_s1_ukpv_selfattn_s{42,43,44}` | 42, 43, 44 | `stage/s1.yaml` (`skip_vision_stack: true`) |
| `mmtsfm_s2a_ukpv_selfattn_s{42,43,44}` | 42, 43, 44 | `model/vision_chronos2_timeselfattn.yaml` + `stage/s2a.yaml` |
| `mmtsfm_s2b_ukpv_selfattn_s44` | **44 only** | `model/vision_chronos2_narrow.yaml` + `stage/s2b.yaml` |
| `mmtsfm_s2b_ukpv_wide_s{42,43,44}` | 42, 43, 44 | `model/vision_chronos2_wide.yaml` (`n_soft_tokens: 16`) |
| `mmtsfm_s2c_ukpv_s2c_s{42,43,44}` | 42, 43, 44 | `model/vision_chronos2_s2c.yaml` + `stage/s2c.yaml` |

There is **no `mmtsfm_s2b_ukpv_grassmann_*.json`, no `mmtsfm_s3_*.json`, and no
`mmtsfm_s2b_ukpv_marginal.json`** anywhere under `baselines/results/` or `MMTSFM/baselines/results/`
(the latter holds only duplicate copies of the three s2c files).

---

## 1. Component coverage table

Verdict key: **VALIDATED** = ≥1 run varies it with seeds and the registry has a DONE row ·
**PARTIAL** = varied but n<3, or confounded with another change · **UNVALIDATED** = switchable and
load-bearing but exactly one value ever run · **NOT-SWITCHABLE** = no config key reaches it.

### 1.1 Fusion pathway

| Component | Config key | file:line | Distinct values across runs | Ablation ID | Verdict |
|---|---|---|---|---|---|
| Late fusion (adapter → soft tokens appended) | `model.vision_cfg.fusion_mode="late"` | `vision_chronos2.py:199` (default), branch `vision_chronos2.py:1140` | 1 (not recorded); reachable via s2a files ×3 seeds | A01 | VALIDATED (n=3, H1 falsified) |
| Interleaved fusion (per-step visual partners) | `fusion_mode="interleaved"` | branch `vision_chronos2.py:860`, `n_vis` `:863` | 1 (not recorded); `..._selfattn_s44` only | A02 | **PARTIAL — n=1**, only `mmtsfm_s2b_ukpv_selfattn_s44.json` |
| Future-query fusion (cross-attn from forecast positions) | `fusion_mode="future_query"` | ctor `vision_chronos2.py:367`, branch `:828` | 1 (not recorded); s2c files ×3 seeds | A16 | VALIDATED (n=3) but **confounded**, see §1.6 |
| Visual K/V projection | implicit on `fusion_mode=="future_query"` | `vision_chronos2.py:368` | 1 | NONE | NOT-SWITCHABLE (no independent key) |
| **Lead-time embedding τ** | none — built whenever `fusion_mode=="future_query"` | built `vision_chronos2.py:375-380`, applied `:818-826` | 1 (always on in s2c) | A19 (TODO) | **UNVALIDATED** |
| Cross-modal adapter presence | derived: `fusion_mode=="late" or n_soft_tokens>1` | `vision_chronos2.py:385-387` | 2 (present in s2a/s2b_wide, absent in s2b_narrow) | none isolates it | UNVALIDATED (never varied holding fusion_mode fixed) |
| Adapter type | `model.vision_cfg.adapter_type` | branches `cross_modal_adapter.py:56/59/69`, ValueError `:84` | **1 — `linear` in every YAML** | NONE | **UNVALIDATED** |
| Adapter depth | `adapter_n_layers` | `cross_modal_adapter.py:59-67` | 1 — `2` everywhere | NONE | UNVALIDATED |
| Soft-token width | `n_soft_tokens` | `vision_chronos2.py:179`; `out_dim = d_model*n_soft_tokens` `cross_modal_adapter.py:54` | 2 — `1` (narrow/timeselfattn) vs `16` (`vision_chronos2_wide.yaml`) | A13 | **VALIDATED (n=3, publishable null)** |
| Visual context depth | `n_visual_context_steps` | `vision_chronos2.py:178`; clamp `min(n_visual_context_steps, T_ctx)` `:863`, `:573`, `:1227` | derived per-run by `scripts/run_all_mmtsfm.sh:272-274` / `slurm_curriculum.sh:149`; `1` in `model/vision_chronos2.yaml` for uk_pv | W7 | VALIDATED (registry marks DONE) |

### 1.2 Visual encoding / pooling

| Component | Config key | file:line | Distinct values | Ablation ID | Verdict |
|---|---|---|---|---|---|
| V-JEPA encoder arch | hard-coded `arch="vit_large"` | `vision_chronos2.py:346-349` | 1 | NONE | NOT-SWITCHABLE |
| Encoder freezing | `vision_cfg.freeze_visual_encoder` | `vision_chronos2.py:204`, read `lightning_module.py:233`, policy `:248-255` | `true` / `partial` across stages, but **never takes effect under a V-JEPA cache** (`knowledge/architecture.md` §2.6) | A14 (reclassified) | UNVALIDATED (registry itself downgraded it to a limitation) |
| Progressive vision unfreeze | `progressive_vision_unfreeze` | `lightning_module.py:82`, applied `:262-268` | 1 (`true` only in `stage/s3.yaml`; no s3 result on disk) | NONE | UNVALIDATED |
| **LatentSummarizer bottleneck** (the ~800:1 pool) | none — `latent_queries` shape fixed | `latent_summarizer.py:83`; ctor `:65-111` | 1 | NONE (A13 widened the *adapter*, not this) | **UNVALIDATED / NOT-SWITCHABLE** |
| Summarizer heads | `summarizer_n_heads` | `vision_chronos2.py:188`, passed `:352-358` | 1 — `4` in every YAML | NONE | UNVALIDATED |
| Summarizer causal masking | none | `_build_causal_attn_mask` `latent_summarizer.py:117`, `_build_time_attn_mask` `:150` | 1 | NONE | NOT-SWITCHABLE |
| **Spatial grid retained for s2c** | `vision_cfg.visual_grid` | `vision_chronos2.py:185`; pooling `g = min(int(self.vcfg.visual_grid), g0)` `:636` | **1 — `4`, only in `model/vision_chronos2_s2c.yaml`** | A17 (TODO, registry's highest-value), A21 (`visual_grid=14`, TODO) | **UNVALIDATED** |
| Frame preprocessing | `data.img_size`, `data.imagenet_norm`, `data.video_frames` | `pv_record.py:164-167`, `_prep_frame` `:415` | 1 each | NONE | UNVALIDATED — and **frozen by the cache**: `pv_record.py:469-472` skips frame decode entirely on a latent-cache hit |
| Visual window length | `data.visual_window_hours` | `pv_record.py:170`, window `visual_window_hours*60.0/Tv` `:373` | 1 — `6.0` in `data/ukpv.yaml` | A04 (TODO) | UNVALIDATED (and cache-blocked, same citation) |

### 1.3 Temporal mixer / backbone

| Component | Config key | file:line | Distinct values | Ablation ID | Verdict |
|---|---|---|---|---|---|
| **Grassmann vs self-attention mixer** | `model.chronos_core_cfg.use_grassmann` | switch `model.py:65-70`; op `grassmann.py:73-397`; alt `layers.py:416` | **1 on disk — `false`.** Every result filename maps to a `use_grassmann: false` YAML | A03 (TODO, "DECISION REQUIRED") | **UNVALIDATED — zero Grassmann results exist** |
| Grassmann modality-pair bias | `grassmann_modality_pair_bias` | `config.py:88`; `grassmann._compute_modality_biases:211-233` | 1 (`true`); a `false` arm is defined at `scripts/run_all_mmtsfm.sh:134-138` but produced no result file | NONE | UNVALIDATED |
| Grassmann reduced dim | `grassmann_reduced_dim` | `config.py` default 32 (`:64-84`) | 1 — `32` | NONE | UNVALIDATED |
| Grassmann window offsets | `grassmann_window_offsets` | `config.py:105` (`or [1,2,4,8,12,16]`), `_offset_weights_for grassmann.py:235-288` | 1 — never set, always the default list | NONE | UNVALIDATED |
| Grassmann warmup | `grassmann_warmup_steps` | `lightning_module.py:75`, used `:1158` | 3 (`2000` s1, `1000` s2b, `0` s2a/s2c) — but always confounded with the stage | NONE | UNVALIDATED |
| **Visual cross-attn block count** | `visual_cross_attn_blocks` | `config.py:103`; gate `model.py:53-56`; call `:101-106` | **2 — `0` (all non-s2c) and `4` (s2c only)**, and the 4 arrives bundled with `fusion_mode` and `output_patch_size` | A18 (`{1,2,4}`, TODO) | **PARTIAL/confounded** |
| Backbone freezing | `freeze_chronos` | `lightning_module.py:72`, `:187-189`, keep-list `:191-200` | 2 (`false` s1/s3, `true` s2a/s2b/s2c) — stage-confounded | NONE | UNVALIDATED as an isolated factor |
| Unfrozen encoder blocks | `n_unfreeze_encoder_blocks` | `lightning_module.py:73`, clamp `:223` | 2 (`1` default, `3` in timeselfattn/grassmann/s2c YAMLs) — never varied alone | NONE | UNVALIDATED |
| Backbone LR ratio | `backbone_lr_ratio` | `lightning_module.py:74`, applied `:358`, `:1070` | 1 — `0.1` | NONE | UNVALIDATED |
| RoPE base | `rope_theta` | `config.py:64-84`; `RoPE.__init__(dim, base=10000)` `layers.py:27` | 1 — `10000.0` | NONE | UNVALIDATED |
| Attention implementation | `attn_implementation` | `config.py`, copied `lightning_module.py:149` | 1 — `sdpa` | NONE | NOT-SWITCHABLE (numerics only) |
| `d_model` / `num_layers` / `num_heads` / `d_ff` | present in YAML but **discarded** | only `use_grassmann`/`grassmann_*`/`visual_cross_attn_blocks`/`_attn_implementation`/nested `chronos_config` are copied onto the pretrained config, `lightning_module.py:130-159` | 1 (real values 768/12/3072 come from the checkpoint) | NONE | **NOT-SWITCHABLE — YAML values silently ignored** (matches `knowledge/architecture.md`) |

### 1.4 Embeddings, tokenisation, normalisation

| Component | Config key | file:line | Distinct values | Ablation ID | Verdict |
|---|---|---|---|---|---|
| Multimodal embedding (modality/segment/token-type) | none | `vision_chronos2.py:405-407`; `MultimodalEmbedding.__init__:243` | 1 | NONE | NOT-SWITCHABLE |
| **Entity embedding** | `model.vision_cfg.n_entities` | built only `if n_entities > 0` `vision_chronos2.py:251` | **1 — `0` in every model YAML**, so the entity table is *never instantiated in any run*, despite `data.num_entities: 4` in `data/ukpv.yaml` | W4 ("CODE DONE, no numbers") | **UNVALIDATED — dead in every result on disk** |
| Register token | `use_reg_token` | `config.py:134`; checks `model.py:357/364/877/881`, `vision_chronos2.py:721` | 1 — `false` | NONE | UNVALIDATED |
| arcsinh input transform | `use_arcsinh` | `config.py:135`; `InstanceNorm(use_arcsinh=...)` `model.py:380` | 1 — `true` in all MMTSFM YAMLs (`false` only in the unused `model/chronos2.yaml`) | NONE | UNVALIDATED |
| Time-encoding scale | `time_encoding_scale` | `config.py:137`; defaulted to `context_length` `model.py:338-339`; used `:629`, `:735` | 1 — never set | NONE | UNVALIDATED |
| Input patch size | `input_patch_size` | `model.py:343-353`, embed `:364-366` | 1 — `16` | NONE | UNVALIDATED |
| **Output patch size** | `output_patch_size` | separate `future_patch_embedding` when `input != output` `model.py:343-353`, `:403-406`; out-dim `:394` | **2 — `16` everywhere, `4` in `model/vision_chronos2_s2c.yaml`** — bundled with s2c | A20 (`output_patch_size=1`, TODO) | **PARTIAL/confounded** |
| Context length | `context_length` | `model/vision_chronos2.yaml` | 1 — `2048` | NONE | UNVALIDATED |
| Forecast slots | `max_output_patches` | read for `n_lead` `vision_chronos2.py:375-377` | 1 — `4` in all MMTSFM YAMLs | NONE | UNVALIDATED |

### 1.5 Loss, regularisation, data

| Component | Config key | file:line | Distinct values | Ablation ID | Verdict |
|---|---|---|---|---|---|
| Loss function | none | single call `self.chronos._compute_loss` at `vision_chronos2.py:1106` and `:1347` — **one pinball/quantile term, no auxiliary losses anywhere** | 1 | NONE | NOT-SWITCHABLE |
| Quantile levels | `model.quantile_levels` | `model/vision_chronos2.yaml` (9 levels) | 1 | NONE | NOT-SWITCHABLE in practice |
| Visual modality dropout | `visual_dropout_prob` | `vision_chronos2.py:189`, gate `:460-467`, `:846-858` | 2 (`0.3` s2a/timeselfattn, `0.5` s2b/s2c) — stage-confounded | NONE | UNVALIDATED as isolated factor |
| Numeric modality dropout | `numeric_dropout_prob` | `vision_chronos2.py:190`, same gate | 1 — `0.1` | NONE | UNVALIDATED |
| Vision stack bypass (numeric-only) | `vision_cfg.skip_vision_stack` | `vision_chronos2.py:205`, guards `:333`, `:339` | 2 — `true` (s1) vs `false` (s2*) | W6 / A01 | VALIDATED (this is the s1 arm, n=3) |
| Marginal gain counterfactual | `model.compute_marginal_gain` | `lightning_module.py:96`, wired `:659`, two-pass `test_step:716-736` | on in all s2 runs (`delta_nmae` present in every s2a/s2b/s2c JSON) | W6 | VALIDATED |
| Entity count (data side) | `data.num_entities` | `datamodule.py:41`; `_build_groups pv_record.py:291-330`, `if self.num_entities == 1` `:298`, silent fallback to 1 `:326-330` | 1 — `4` in `data/ukpv.yaml` | W4 | UNVALIDATED (and the model side is 0, §1.4) |
| Train stride | `data.train_stride` | `datamodule.py:63` | overridden by env in `scripts/`; not in any result manifest | NONE | UNVALIDATED |
| Visual cadence multiplier | `data.vis_cadence_multiplier` | `datamodule.py:70` → `:118`; `dataset.py:209/239` | 1 — `1` | NONE | UNVALIDATED |
| History / horizon | `data.history_days`, `data.horizon_hours` | `pv_record.py:160-161`, `self.T`/`self.H` `:238-239` | 1 each (`14.0` / `6.0`) | NONE | NOT-SWITCHABLE (protocol-fixed) |
| **Frame-shuffle control** | `eval.control=shuffle_frames` (registry's name) | **no implementation** — string absent from `MMTSFM/src`, `MMTSFM/scripts`, `baselines/common` | 0 | A09 (TODO, "MANDATORY") | **UNVALIDATED — not implemented** |
| **Plant-swap control** | `eval.control=swap_plant_frames` | **no implementation**, same search | 0 | A10 (TODO, "MANDATORY") | **UNVALIDATED — not implemented** |

### 1.6 The s2c confound, stated plainly

`model/vision_chronos2_s2c.yaml` changes **three** things at once relative to
`model/vision_chronos2_timeselfattn.yaml`: `fusion_mode: future_query`,
`visual_cross_attn_blocks: 4` (from 0), and `output_patch_size: 4` (from 16, input stays 16, which
activates the separate `future_patch_embedding` at `model.py:343-353`). It also adds `visual_grid: 4`,
which only exists on this path.

The headline s2c gain — SS 0.5518 / 0.5402 / 0.5489 versus s2b_wide 0.5370 / 0.5322 / 0.5363
(`mmtsfm_s2c_ukpv_s2c_s{42,43,44}.json` vs `mmtsfm_s2b_ukpv_wide_s{42,43,44}.json`) — is therefore
attributable to any of four simultaneous changes. A16's "SUPPORTED" verdict is a verdict on the
*bundle*, not on cross-attention. This is the single largest reviewer exposure in the table.

---

## 2. Load-bearing in the forward pass, zero ablation coverage

Ranked by how likely a reviewer is to demand it.

1. **`use_grassmann` — the named contribution has no result on disk.** `config.py` defaults it to
   `True` and `grassmann.py:73-397` is 325 lines of bespoke Plücker/Grassmann machinery, yet every
   one of the 13 MMTSFM result files maps to a YAML with `use_grassmann: false`
   (`model/vision_chronos2_timeselfattn.yaml`, `_narrow`, `_wide`, `_s2c`). A reviewer reading the
   architecture section will ask for the mixer comparison first. Registry A03 is TODO and
   `knowledge/protocol.md` §5.1 already pre-registers the decision rule (seed floors 0.0011 ramp
   NMAE / 0.0037 skill score) — the rule exists, the data does not.

2. **The s2c bundle (`visual_cross_attn_blocks` × `output_patch_size` × `fusion_mode` × `visual_grid`).**
   §1.6. Four confounded changes carrying the paper's best number. A18 and A20 are the two open rows
   that would decompose it; neither has run.

3. **`visual_grid` — the claim that s2c uses *spatial* structure rests on one value.** The pooling at
   `vision_chronos2.py:636` reduces to a 4×4 field; `visual_grid=1` (A17) is the single run that
   separates "cross-attention over space" from "cross-attention over a global vector with 3 decoder
   slots". Registry marks A17 highest-value and it is still TODO. If s2c survives `visual_grid=1`
   unchanged, the spatial story is dead.

4. **A09/A10 negative controls — not merely unrun, unimplemented.** `shuffle_frames` and
   `swap_plant_frames` appear nowhere in the source tree. Every vision-attributed gain
   (`delta_nmae` 0.0072/0.0065/0.0076 in the s2c files) currently lacks a control showing the model
   is not exploiting a frame-order or plant-identity shortcut. Registry calls both MANDATORY.

5. **`lead_time_embed`.** Built at `vision_chronos2.py:375-380` and — by deliberate design, per the
   comment at `:818-824` — applied on *every* s2c batch including vision-off ones. It is therefore
   inside the s2c-vs-s2b delta but *outside* the marginal-gain counterfactual. A19 is TODO. A careful
   reviewer will notice the comment and ask exactly this.

6. **`LatentSummarizer.latent_queries` — the actual bottleneck.** `latent_summarizer.py:83`. A13
   widened `n_soft_tokens` 1→16 at the *adapter* (`cross_modal_adapter.py:54`) and found a null; the
   ~800:1 compression is upstream of that and has no config key at all. The published null risks
   being read as "vision does not help" when it may only show "the adapter was not the bottleneck".

7. **Entity embedding: dead code in every run.** `data/ukpv.yaml` sets `num_entities: 4`, but every
   model YAML sets `n_entities: 0`, and `vision_chronos2.py:251` only builds the table when
   `n_entities > 0`. The cross-plant conditioning described as a feature is not present in any number
   on disk. W4's "CODE DONE, no numbers" understates this — the code is done *and disabled*.

8. **`adapter_type`, `adapter_n_layers`, `summarizer_n_heads`, `grassmann_reduced_dim`,
   `grassmann_window_offsets`, `use_arcsinh`, `use_reg_token`, `rope_theta`, `time_encoding_scale`.**
   Nine switchable keys, one value each, no ablation ID between them. Individually low reviewer risk;
   collectively they are the "we did not tune, we inherited" paragraph that the limitations section
   must state explicitly.

9. **`visual_window_hours` / frame preprocessing (A04).** Feasibility caveat worth recording:
   `pv_record.py:469-472` skips raw-frame decode entirely on a V-JEPA latent-cache hit, so
   `video_frames`, `img_size`, `imagenet_norm` and `visual_window_hours` are baked into the cache.
   A04 requires a cache rebuild, not just a config flip.

---

## 3. Disagreements between `knowledge/ablations.md` and disk

Checked row by row. Registry §1 aggregate means and sds **do reproduce** against the raw JSONs, so §1
is trustworthy; the disagreements are in key names, file existence, and coverage claims.

**Files the registry references that do not exist**

1. `mmtsfm_s2b_ukpv_marginal.json` — cited by `knowledge/architecture.md` §4 as the source of
   ΔNMAE 0.00200. Not present in `baselines/results/` or `MMTSFM/baselines/results/`.
2. `ALL_RESULTS.md` — the routing rule in the project brief points numbers at
   `baselines/results/ALL_RESULTS.md`; the registry itself notes it does not exist, and it does not.

**Config keys the registry names that do not exist in code**

3. A01 writes `fusion_mode=late_fusion`. The real value is `"late"` (`vision_chronos2.py:199`,
   branch `:1140`). `late_fusion` would hit no branch.
4. A03 writes `model.temporal_mixer`. No such key. The real switch is
   `model.chronos_core_cfg.use_grassmann` (`model.py:65-70`, copied `lightning_module.py:130`).
5. A11 writes `model.inputs=vision_only`; A12 writes `model.inputs=...`. No `model.inputs` key exists.
   The numeric-only arm is reached via `vision_cfg.skip_vision_stack` (`vision_chronos2.py:205`) plus
   `data.emit_vision=false` (`pv_record.py:174`, `datamodule.py:58`); there is no vision-only path at
   all.
6. A14 writes `model.unfreeze=...`. Real keys are `vision_cfg.freeze_visual_encoder`
   (`vision_chronos2.py:204`), `n_visual_unfreeze_layers` (`lightning_module.py:81`) and
   `progressive_vision_unfreeze` (`:82`).
   — For contrast, A04's `data.visual_window_hours` and W4's `data.num_entities` **are** correctly
   named (`pv_record.py:170`, `datamodule.py:41`).

**Coverage claims that overstate or understate what is on disk**

7. A02 is marked PARTIAL n=1 BLOCKING. Disk confirms it: `mmtsfm_s2b_ukpv_selfattn_s44.json` is the
   only interleaved-arm file, seeds 42 and 43 are absent. Registry correct; recording it because it
   is the one row where the registry's own status is the binding constraint.
8. A16 is marked DONE / SUPPORTED n=3. True on file count, but the comparison is confounded four ways
   (§1.6). The registry does not record the confound.
9. A13 is marked DONE, "publishable null". True for `n_soft_tokens`, but the registry does not note
   that `vision_chronos2_wide.yaml` differs from `_timeselfattn.yaml` by **exactly that one line**
   (verified by diff) — which is good news for the null's internal validity and worth stating.
10. The registry states "No Grassmann result exists anywhere on disk." Confirmed — and stronger than
    stated: the `grassmann_no_modbias` and `numeric_grassmann` arms defined at
    `scripts/run_all_mmtsfm.sh:134-138` also produced nothing.

**Disagreements involving `architecture.md` and `scope.md`**

11. `knowledge/architecture.md`'s fusion switch box lists only `late` and `interleaved`. The entire
    s2c path — `future_query` (`vision_chronos2.py:367`, `:828`), `visual_kv_proj` (`:368`),
    `lead_time_embed` (`:375-380`), `TimeCrossAttention` / `visual_cross_attn_blocks`
    (`model.py:53-56`) — is missing from both the diagram and the component map, despite s2c being
    the arm with the paper's best numbers.
12. `knowledge/scope.md` H1/H2 quote SS 0.5086 / 0.5087 / 0.5284. None of these three values appears
    in any file under `baselines/results/`. The registry already flags this; recording it here because
    the manuscript inherits those numbers.
13. `knowledge/architecture.md` §4 notes the interleaved path overwrites the context attention mask
    with `all_mask = torch.ones(...)`. On disk the mask is genuinely constructed from
    `attention_mask` at `vision_chronos2.py:1010-1028`, with visual partners forced to ones at
    `:1014-1017`. So the note is half-right — the *numeric* mask survives; it is the *visual*
    partners that are unconditionally attended. Worth correcting in `architecture.md`.
14. `knowledge/architecture.md` §2.6 correctly records that V-JEPA unfreeze never happens under
    `VJEPA_CACHE`; `pv_record.py:469-472` is the mechanism (cache hit ⇒ `load_frames=False` ⇒ encoder
    never invoked). No disagreement — noted because it is what makes A14's reclassification correct
    and A04 expensive.

**Meta-disagreement, and the most consequential one**

15. The registry is organised as if result JSONs record the configuration that produced them. They do
    not (§0: `lightning_module.py:825-829`, `runner.py:214-222`). Every provenance claim in
    `ablations.md` rests on filename convention plus `git_sha`, with no in-file config to verify
    against. Two runs with different architectures and the same seed are byte-identical in their
    manifest config block. Any future ablation whose arm is distinguished only by a CLI override —
    e.g. the `run_all_mmtsfm.sh:134-138` arms that pass
    `model.chronos_core_cfg.grassmann_modality_pair_bias=false` — would be **unrecoverable from its
    own result file**. This should be fixed before wave 2 launches, not after.
