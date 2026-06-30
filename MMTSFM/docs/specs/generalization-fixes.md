# Spec — MMTSFM Generalization Fixes

**Goal:** Close the gap between the proposal (`knowledge/docs/proposal.md`) and the
implementation so that the reported model actually exercises the mechanisms that
deliver **cross-site (zero-shot cross-plant)** generalization.

**Audience:** A team of implementation agents. Each workstream below is
self-contained: hypothesis → files → change → acceptance criteria → tests →
dependencies. Work is parallelizable except where `Depends on` is stated.

**Final-model scope (locked):** single visual source — **satellite imagery
(`goes_pvdaq`, RGB)** — via the **V-JEPA 2.1** encoder. No VidTok. No sky-camera
track. No multi-sensor / source-type conditioning. Do **not** add code paths for
those; remove incidental references when you touch a file.

---

## 0. Shared conventions (read first, applies to every workstream)

| Rule | Detail |
|------|--------|
| **Python** | `uv` only. Run code with `uv run …`. Never `python`/`pip`. |
| **Branch** | One branch per workstream: `fix/<id>-<slug>` or `exp/<id>-<slug>`. Never commit to `main`. |
| **Commits** | Micro-commits. Format `fix(<id>): <what+why>` or `exp(<id>): …`. One logical change per commit. |
| **Files** | One class / one capability per file, target < 150 lines. |
| **Config** | Hydra only. No magic numbers in model code — add to `configs/`. |
| **Tests** | Mirror `src/` under `tests/`. Every changed module needs a shape + gradient smoke test. Run `uv run pytest` before claiming done. |
| **Impact** | Run GitNexus `impact({target, direction:"upstream"})` before editing a shared symbol; report HIGH/CRITICAL risk. Run `detect_changes()` before committing. |
| **Data** | `/leonardo_scratch/fast/IscrC_MTSFM/data` is read-only. Do not refactor data ETL. |
| **No physics heuristics** | No clear-sky-index / irradiance conversions unless explicitly ablating. |
| **Registry** | Anything that produces a comparable number gets an `docs/experiments/ABLATION_REGISTRY.md` entry + a config diff under `configs/`. |

**Definition of done (every workstream):** branch merged-ready, `uv run pytest`
green, new tests added, `git diff HEAD` clean of debug prints, registry/config
updated if the change affects reported metrics.

---

## 0b. Development environment — MacBook Air, 16 GB RAM (HARD constraint)

All agent work happens **on a MacBook Air with 16 GB RAM, CPU/MPS only, no GPU,
no cluster, no read-only dataset of record.** The real training run happens later
on the Leonardo cluster, by the human, after this work is merged. Therefore:

- **Never** download / load the real V-JEPA 2.1 weights (~300M) or run a real
  training/eval job during development. They will not fit / will not finish.
- All tests and smoke checks must be **tiny, CPU-only, and fast** (seconds):
  - Stub or mock the V-JEPA backbone with a small `nn.Module` exposing `.d_v`
    and a conformant `forward` returning `[B, T_lat, P, D_v]`. Do not call
    `torch.hub.load`.
  - Use tiny Hydra overrides: `chronos_core_cfg.d_model=32`,
    `chronos_core_cfg.num_layers=1`, `num_heads=2`, tiny patch/context lengths,
    `batch_size=2`, a few synthetic windows.
  - Use the synthetic dataset path / fabricated tensors — **not** the real
    `dataset_all.parquet` / `images_all.h5` (not present on the laptop).
- Allowed commands during dev: `uv run pytest …`, `uv run python -m mmtsfm.train
  … fast_dev_run=true` with the tiny overrides above, Hydra `--cfg job` dry runs
  for config validation.
- Each workstream's **acceptance criteria must be verifiable on this laptop**.
  Cluster-scale verification (full V-JEPA, full splits, real metrics) is the
  human's follow-up step — mark such checks explicitly as **[cluster, deferred]**.
- If a change can only be validated at cluster scale, the agent still must add a
  CPU smoke test that exercises the code path with stubs and document the
  deferred cluster check.

---

## Workstream dependency graph

```
W1 (d_v wiring bug)  ─┐
W2 (hub-repo race)   ─┤→ do first (unblocks V-JEPA path)
                      │
W3 (headline config) ─┴─ depends on W1, W2
W4 (cross-plant batching / group attn)   ─ independent
W5 (visual recency window + frame Δt)    ─ independent
W6 (visual-ablation metric)              ─ independent
W7 (config drift: n_visual_context)      ─ depends on W5 (shares window def)
```

Suggested parallel assignment: **Agent A** → W1+W2+W3 (chain),
**Agent B** → W4, **Agent C** → W5+W7, **Agent D** → W6.

---

## W1 — Fix V-JEPA latent-dim wiring bug

**Severity:** Critical (silent footgun blocking the headline encoder).

**Problem:** `vision_chronos2.py:297` sets `_d_v = vision_config.d_video_latent`
on the `vjepa2` path. Default `d_video_latent: 4` (`vision_chronos2.yaml:47`).
Switching `visual_encoder_type=vjepa2` without also hand-editing `d_video_latent`
builds `LatentSummarizer(d_v=4)` while the encoder emits 1024 → shape mismatch.

**Change:** On the `vjepa2` path, derive the KV dim from the encoder itself:
`_d_v = self.video_encoder.d_v`. `VisualEncoder` already exposes `.d_v`
(`visual_encoder.py:127`). Drop reliance on `vision_config.d_video_latent` for the
V-JEPA path. (VidTok is out of scope — see §0; if removing the VidTok branch is
trivial in the file you touch, do it, otherwise leave it untouched and unused.)

**Files:** `MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py`.

**Acceptance criteria (laptop):**
- With a stubbed `VisualEncoder` exposing `.d_v=1024`, constructing
  `VisionChronos2Model` (vjepa2 path) builds a `LatentSummarizer` whose
  `kv_proj.in_features == 1024`.
- A forward pass with a synthetic `[B, C, T_v, H, W]` tensor runs without shape
  error on CPU with tiny dims.

**Tests:** `tests/models/test_vision_chronos2.py` — case mocking `VisualEncoder`
(`.d_v=1024`, conformant forward) asserting summarizer KV dim + a forward smoke
pass.

**Dependencies:** none.

---

## W2 — Make VisualEncoder load race-safe

**Severity:** Critical for parallel cluster runs.

**Problem:** `visual_encoder.py:79-88` rewrites `backbones.py` inside the shared
torch-hub cache on every load. Cluster ablations run one-per-GPU in parallel →
concurrent processes write the same file → race / corruption on the shared FS.

**Change:** Make the patch idempotent and concurrency-safe. Prefer (1):
1. Apply the URL replacement **in memory** at import time — no disk write at
   runtime, or
2. Guard the write with a file lock + sentinel marker so only one process writes
   once.

Move the canonical one-time patch into `scripts/login_node_setup.sh`; keep the
runtime path read-only / assert-only.

**Files:** `MMTSFM/src/mmtsfm/models/vision/visual_encoder.py`,
optionally `MMTSFM/scripts/login_node_setup.sh`.

**Acceptance criteria (laptop):**
- Unit test simulating two concurrent `_load` calls (threads, `torch.hub.load`
  mocked — no network, no real weights) does not corrupt the patch target and
  both succeed.
- Repeated patching is idempotent (stable content; no duplicate edits).
- **[cluster, deferred]** real hub load + real parallel jobs.

**Tests:** test the patch function in isolation + a mocked concurrent-load test.
Gate any hub-touching path behind a marker (skipped on laptop).

**Dependencies:** none. If adding `filelock`, `uv add filelock`.

---

## W3 — Promote the proposed architecture to the headline config

**Severity:** Critical (otherwise reported numbers are off-method).

**Problem:** Default `vision_chronos2.yaml` runs late fusion +
`use_grassmann:false`. The proposal's V-JEPA 2.1 + interleaved + Grassmann path is
reachable only via manual overrides.

**Change:** Create the **headline** model config = the proposed architecture:
- New `configs/model/vision_chronos2_headline.yaml` (or repurpose
  `vision_chronos2_grassmann.yaml`) with `visual_encoder_type: vjepa2`,
  `fusion_mode: interleaved`, `use_grassmann: true`, correct
  `visual_encoder_ckpt_path` slot.
- Update `knowledge/docs/proposal.md` "Key config changes vs. current
  implementation" table so Current == headline config (no longer stale). Drop the
  VidTok rows.

**Files:** `MMTSFM/configs/model/*.yaml`, `knowledge/docs/proposal.md`.

**Acceptance criteria (laptop):**
- `uv run python -m mmtsfm.train model=vision_chronos2_headline --cfg job`
  (dry run) shows the three switches set; a tiny-dims construction smoke test
  (V-JEPA stubbed) builds the interleaved + Grassmann stack.
- Proposal table reflects reality.
- **[cluster, deferred]** full V-JEPA load + a real training step.

**Tests:** config-instantiation test (Hydra compose → `instantiate`, V-JEPA
stubbed) asserting the three switches and that the model builds.

**Dependencies:** W1, W2.

---

## W4 — Real cross-plant batching (activate group attention)

**Severity:** Major (the cross-entity mechanism is currently dead).

**Problem:** `goespvdaq.yaml` / `ukpv.yaml` set `num_entities: 1`.
`GroupSelfAttention` fuses across the batch axis within a `group_id`; with one
plant per window it fuses nothing. The proposal's cross-plant token alignment
never fires.

**Decision required (record in registry):**
- **(a)** Multiple *distinct* plants per group per step (true cross-plant mixing),
  vs **(b)** keep `N=1` and drop the cross-entity claim from the proposal.

Default to **(a)** unless batching disjoint plants is judged to violate the
zero-shot protocol. (It does not, as long as test plants are never in a train
group.)

**Change (if a):** Extend `PVRecordDataset` / `datamodule` to assemble groups of
`num_entities>1` plants from the *same split* sharing a time window, emit
`group_ids`, and verify `GroupSelfAttention`'s `group_time_mask` is correct for
N>1. `_unpack_batch` already builds `entity_ids` for N>1
(`lightning_module.py:301-306`).

**Files:** `MMTSFM/src/mmtsfm/data/pv_record.py`,
`MMTSFM/src/mmtsfm/data/datamodule.py`, `configs/data/*.yaml`.

**Acceptance criteria (laptop):**
- With synthetic multi-plant windows and `num_entities=4`, group attention is
  correctly shaped + masked; no test plant appears in any train group.
- `GroupSelfAttention` receives non-zero gradient for N>1 (assert in smoke test).
- **[cluster, deferred]** correctness on the real splits.

**Tests:** dataset test asserting per-group plant disjointness vs splits
(synthetic split fixture); model test asserting group-attention gradient is
non-trivial for N>1 (tiny dims, CPU).

**Dependencies:** none.

---

## W5 — Bound the visual window + pass frame timestamps

**Severity:** Major (breaks the decoupled-resolution premise).

**Problem:** `pv_record.py:199-204` selects the last `Tv` frame-bearing steps from
the **entire 14-day history** with no recency cap; sparse frames can span days,
not the "recent cloud-advection window." Frame Δt is never passed to
`LatentSummarizer`, which assumes uniform spacing.

**Change:**
1. Add a `visual_window_hours` knob (default ≈ 6h per proposal §1). In
   `_load_vision`, restrict candidate frames to
   `[t_now - visual_window_hours, t_now]` before taking the most-recent `Tv`. Emit
   a mask when fewer than `Tv` frames exist.
2. Emit per-frame timestamps (or normalized Δt relative to forecast origin) in the
   batch dict and thread them into `LatentSummarizer` so its causal chunking
   reflects true spacing, not a uniform assumption.

**Files:** `MMTSFM/src/mmtsfm/data/pv_record.py`,
`MMTSFM/src/mmtsfm/models/vision/latent_summarizer.py`,
`configs/data/*.yaml`.

**Acceptance criteria (laptop):**
- On a synthetic sparse-frame fixture, selected frames all fall within
  `visual_window_hours` of the forecast origin; correct mask when frames missing.
- Frame Δt reaches the summarizer (assert shape/plumbing in a test).
- Dense-frame synthetic case shows no regression.

**Tests:** dataset test with synthetic sparse frames (recency bound + mask);
summarizer test asserting Δt is consumed.

**Dependencies:** none.

---

## W6 — Visual-marginal-gain metric (vision on/off)

**Severity:** Moderate (verifies the visual stream is actually used).

**Problem:** With `p_v=0.5` dropout, the model can route around vision — the exact
signal meant to drive sub-hourly cross-site skill. Only `visual_fraction` is
logged (`lightning_module.py:385-390`); no marginal gain.

**Change:** At test time, run each window twice — vision-on and vision-masked —
and report ΔNMAE / ΔNRMSE (visual marginal gain) into the cross-plant results
schema. Add as a protocol-eval option (off by default to keep eval cheap; on for
headline reporting). Force vision-off via the existing modality-dropout zeroing
logic — no separate code path.

**Files:** `MMTSFM/src/eval/protocol_eval.py`,
`MMTSFM/src/mmtsfm/models/chronos2/lightning_module.py`.

**Acceptance criteria (laptop):**
- A single eval pass (tiny dims, synthetic batch, V-JEPA stubbed) reports
  `nmae_vision_on`, `nmae_vision_off`, and the delta.
- Forced vision-off path matches a manually visual-masked forward.
- **[cluster, deferred]** the actual sign/magnitude of the gain on real data.

**Tests:** evaluator test asserting both passes run + delta computed; assert
vision-off equals a manually masked forward.

**Dependencies:** none.

---

## W7 — Remove config drift in `n_visual_context_steps`

**Severity:** Moderate.

**Problem:** `vision_chronos2.yaml:50` hardcodes `n_visual_context_steps: 3` with
a stale comment `ceil(24/8)`. For `goes_pvdaq`, `T_ctx` is much larger — the
visual window is not derived from the data.

**Change:** Derive `n_visual_context_steps` from `visual_window_hours` (W5) and the
patch cadence, or at minimum compute + assert it per dataset config with a correct
comment. Add a derivation/validation in the datamodule that checks the value
against `hist_steps / input_patch_size` and asserts it is `<= T_ctx`.

**Files:** `configs/model/*.yaml`, `configs/data/*.yaml`,
`MMTSFM/src/mmtsfm/data/datamodule.py` (validation/assert).

**Acceptance criteria (laptop):**
- `n_visual_context_steps` is consistent with the dataset cadence; an assert fires
  on an impossible value (> T_ctx).
- Comments reflect the actual dataset.

**Tests:** config test asserting the derived value for `goes_pvdaq`.

**Dependencies:** W5 (shares the window definition); coordinate to avoid conflicts.

---

## Cross-cutting acceptance: end-to-end gate

Before any of this is reported as a result:
1. `uv run pytest` fully green **on the laptop** (CPU, tiny, stubbed V-JEPA).
2. `model=vision_chronos2_headline` passes a `fast_dev_run=true` tiny-dims smoke
   with stubs.
3. `docs/experiments/ABLATION_REGISTRY.md` has a row per generalization-relevant
   change with hypothesis + config diff.
4. **[cluster, deferred — human-run]** full V-JEPA training, real splits, real
   cross-plant metrics, and the visual marginal-gain sign.

## Out of scope
- Sky-camera / multi-sensor source-type conditioning.
- VidTok.
- Intra-site evaluation.
- Data / TS / image augmentation.
- Data ETL / dataset-of-record refactors (read-only).
- Classical ML baselines; domain physics heuristics.
- The end-to-end `scripts/slurm_curriculum.sh` (separate infra task).
