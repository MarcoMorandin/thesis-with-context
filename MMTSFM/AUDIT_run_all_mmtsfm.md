# MMTSFM Audit — `scripts/run_all_mmtsfm.sh` readiness

Date: 2026-07-02 · Scope: everything the orchestrator touches (sbatch → pre-extraction → 5 ablations × uk_pv → protocol test → aggregate).

> **STATUS UPDATE (2026-07-02, branch `fix/mmtsfm-audit`)**: all findings fixed except the two listed under [Remaining / manual](#remaining--manual) below. Fix-status column added to the summary table. Full suite green after fixes (139 passed, incl. 3 new regression tests). See the branch's commit series for per-issue diffs.
Local verification: `uv run pytest` → **136 passed**; Hydra compose of all 5 ablation override sets → **all compose cleanly** (duplicate `data.batch_size` override is last-wins, so the `selfattn_late` batch-halving works as intended).

Severity legend: **P0** = will (or very likely will) block training/eval · **P1** = produces wrong/misleading results · **P2** = inefficiency / waste · **P3** = hygiene.

---

## Summary table

| ID | Sev | Where | Issue | Fixed |
|----|-----|-------|-------|-------|
| B1 | P0 | `scripts/precache_login.sh:66` | Imports nonexistent `mmtsfm.models.chronos2.modeling_chronos2` → Chronos-2 **weights are never cached**; offline compute job then fails at `from_pretrained` | ✅ import from `.model`; warmups now fatal; precache exits nonzero on any failure |
| B2 | P0 | `scripts/run_all_mmtsfm.sh:11` + submit dir | `#SBATCH --output=logs/slurm/...` — dir must exist at submit time; also script breaks if sbatch'd from repo root instead of `MMTSFM/` | ✅ cd anchored on script path; `logs/slurm/.gitkeep` committed (force-added past `logs/` ignore); header documents submit-dir constraint |
| B3 | P0 | `scripts/run_all_mmtsfm.sh:132` | Cache-exists check skips extraction if *any* `.pt` exists → partial cache never completed → mixed batches fall back to on-the-fly V-JEPA encode (slow, OOM risk) | ✅ dir-level skip removed (extractor is per-file idempotent); cache dir now settings-versioned (`<ds>/<arch>_f<frames>_s<size>`) via new `--cache-dir` |
| B4 | P0 | walltime vs stride-1 windows | uk_pv train stride=1 → potentially ~10⁵–10⁶ windows/epoch; 50 epochs in a 24 h job is unrealistic; no resume logic → job killed mid-fit produces **no results JSON at all** | ✅ auto-resume from `<tag>/last.ckpt` (`RESUME=1`); `LIMIT_TRAIN_BATCHES` / `VAL_CHECK_INTERVAL` env knobs (unset by default — set them for 24 h jobs) |
| C1 | P1 | `lightning_module.py:105-123` | `grassmann_modality_pair_bias` **not propagated** to the pretrained config → the `grassmann_no_modbias` ablation silently runs with the bias ON (measures nothing) | ✅ propagated (plus `attn_implementation`) |
| C2 | P1 | `protocol_eval.py:26-33` + `run_all_mmtsfm.sh:116` | goes_pvdaq runs fall back to the **uk_pv Smart-Persistence reference** → overall Skill-Score cross-dataset contaminated | ✅ orchestrator always passes a dataset-specific reference path; missing file → SS omitted instead of borrowed |
| C3 | P1 | `model.py:424-425` | History silently truncated to `context_length=512` steps: uk_pv 672→512 (10.7 d), goes 1344→512 (5.3 d). "14-day history" protocol claim not honored by the model | ✅ `context_length: 2048` in all model configs (hub `time_encoding_scale=8192` preserved through the pretrained merge; baselines use the hub config → full history, so this restores comparability) |
| C4 | P1 | `lightning_module.py:394` + `latent_summarizer.py:246` | `video_delta_t` has length `T_v=8` but summarizer requires length `T_lat=4` → W5 true-frame-spacing feature is **dead in every run** (silent uniform fallback) | ✅ per-frame Δt pooled to latent resolution (per-stride-group max) + regression test |
| E1 | P2 | vjepa cache design | ~3.2 MB/window ( [4,196,1024] fp32 ) × stride-1 train windows → potentially **hundreds of GB–TB** + Lustre inode storm | ✅ fp16 cache (halves size); versioned dir. Frame-hash dedup / per-site shards NOT done (see Remaining) |
| E2 | P2 | `pv_record.py:311` | Raw frames always decoded (H5 read + PIL LANCZOS ×8) even on cache hit / `skip_vision_stack` → dataloader CPU + ~300 MB pinned V tensor per batch wasted | ✅ decode skipped on full-group latent hit; `emit_vision=false` (1×1 placeholder V) for the numeric ablation |
| E3 | P2 | checkpoints | Frozen V-JEPA (~1.2 GB) serialized into every checkpoint; ~15 ckpts/run × 5 runs ≈ >100 GB | ✅ `on_save_checkpoint` strips frozen `model.video_encoder.*` (kept when trainable); `strict_loading=False`; tests both ways |
| E4 | P2 | `lightning_module.py:637-644` | Per-param `.item()` grad-norm loop every step → hundreds of GPU syncs per step | ✅ tensor accumulation, single finite-check |
| E5 | P2 | wave dispatch + serial extraction | Last wave runs 1 job on 4-GPU node; extraction uses only GPU 0 while 3 idle | ✅ per-GPU work queue (verified by simulation); extraction fanned out across GPUs |
| E6 | P2 | `trainer/slurm.yaml:8,12` | DDP wrapper with 1 device; `log_every_n_steps=1` with `sync_dist=True` | ✅ orchestrator passes `trainer.strategy=auto` (MASTER_PORT juggling removed); `log_every_n_steps: 50` |
| H1 | P3 | hydra run dir | Parallel runs launched same second collide on `logs/experiments/runs/${now:...}` | ✅ per-run `hydra.run.dir=…_<tag>` |
| H2 | P3 | script exit code / docs | Script exits 0 even when all runs FAIL; CLAUDE.md still references removed `MMTSFM/scripts/slurm_train.sh` | ✅ exits nonzero on any failed run; dead `skip` branch removed. CLAUDE.md NOT fixed (see Remaining) |

### Remaining / manual

1. **CLAUDE.md:59** still says `sbatch MMTSFM/scripts/slurm_train.sh` (removed script). A repo hook blocks agents from writing CLAUDE.md — replace manually with: `cd MMTSFM && sbatch scripts/run_all_mmtsfm.sh` (+ note: run `scripts/precache_login.sh` first; sbatch from `MMTSFM/`).
2. **E1 (partial)**: cache is fp16 + versioned, but still one file per (site, origin) window. If stride-1 train extraction proves too large on Lustre, next step is content-keyed dedup (hash of the 8 selected frame indices) or per-site shard files.
3. **Cluster migration note**: an existing unversioned fp32 cache at `vjepa_cache/<ds>/` is no longer picked up. If it was built with `vit_large_f8_s224`, move its `*.pt` into `vjepa_cache/<ds>/vit_large_f8_s224/` to reuse (the orchestrator logs this hint too).
4. **C3 provenance**: record the context-length decision (512 → 2048) in `docs/experiments/ABLATION_REGISTRY.md` when registering the runs — earlier 512-context results are not comparable.

---

## P0 — Blockers

### B1. `precache_login.sh` never caches Chronos-2 weights
`scripts/precache_login.sh:66` runs:

```python
from mmtsfm.models.chronos2.modeling_chronos2 import Chronos2Model
```

That module does not exist (the class lives in `src/mmtsfm/models/chronos2/model.py`; the package exports it via `mmtsfm.models.chronos2`). The step fails with `ModuleNotFoundError`, but the script only `warn`s and continues. The verify stage (line 97-101) only loads the **config**, not the weights. Consequence: on a fresh `HF_HOME`, the compute node (which exports `HF_HUB_OFFLINE=1`) fails at `Chronos2Model.from_pretrained("amazon/chronos-2")` in `lightning_module.py:124` — **every ablation dies at model init**.

- If the scratch HF cache is already populated from earlier baseline runs, this is masked. Do not rely on that.
- Fix: change the import to `from mmtsfm.models.chronos2.model import Chronos2Model` (or import from the package) and make precache **fail hard** (`exit 1`) on this step instead of `warn`.
- Same pattern risk: the V-JEPA warmup and the `uv sync` step also only `warn`.

### B2. SLURM output dir + submit-dir sensitivity
- `#SBATCH --output=logs/slurm/%j_%x.out` is resolved relative to the *submission* directory **before** the script's `mkdir -p logs/slurm` runs. If `logs/slurm` doesn't exist there, the job dies at launch with no visible log. It currently exists only if `precache_login.sh` was run from `MMTSFM/` (it does the mkdir at line 36). Locally `MMTSFM/logs/` contains only `experiments/`.
- `run_all_mmtsfm.sh:38` does `cd "${SLURM_SUBMIT_DIR:-...}"`. If you `sbatch MMTSFM/scripts/run_all_mmtsfm.sh` from the repo root (as the top-level docs suggest for other scripts), `SLURM_SUBMIT_DIR` = repo root → the script executes in the wrong directory: `scripts/extract_video_embeddings.py` not found, wrong `uv` project, `logs/slurm` created at repo root.
- Fix: `mkdir -p MMTSFM/logs/slurm` before sbatch (or commit a `.gitkeep`), and derive the workdir from the script path (`cd "$(dirname "$(readlink -f "$0")")/.."`) instead of `SLURM_SUBMIT_DIR`. Document: submit from `MMTSFM/`.

### B3. Extraction skip check freezes a partial cache
`run_all_mmtsfm.sh:132` skips *all* extraction for a dataset if the cache dir contains **one** `.pt` file. A previously interrupted extraction (or one done with different `EXTRACT_SPLITS`) is then never completed. Downstream chain:

1. `PVRecordDataset._attach_latents` (`pv_record.py:385`) attaches `Z` only when *every* entity in the group has a file.
2. `_collate_optional_z` (`datamodule.py:16`) drops `Z` for the **whole batch** if any sample lacks it.
3. `_unpack_batch` then routes raw frames through the full V-JEPA forward on the training GPU — an order-of-magnitude slowdown and the OOM mode that already forced `data.batch_size=8` for `selfattn_late` (commit 8abd792).

The inner extractor is already idempotent (per-file `exists()` check at `extract_video_embeddings.py:119`), so the outer skip is redundant *and* harmful. Fix: drop the directory-level skip and always run the extractor; it will fast-skip complete splits. Also version the cache path by settings, e.g. `${VJEPA_CACHE_ROOT}/${ds}/${VJEPA_ARCH}_f${EXTRACT_VIDEO_FRAMES}_s${EXTRACT_IMG_SIZE}` — today a cache made with a different arch/img-size/frames would be silently reused and either crash on latent-dim mismatch or silently corrupt training.

### B4. 24 h walltime vs. stride-1 epochs, no resume → zero results
`PVRecordDataset` uses `stride=1` for train (`pv_record.py:187`). uk_pv: 69 train plants × (rows − 684) windows each — plausibly 10⁵–10⁶ groups per epoch at `num_entities=4`. With `MAX_EPOCHS=50`, `EarlyStopping(patience=7)` and per-epoch validation only, a 24 h `--time` limit can kill the job mid-fit. Because protocol metrics are written **only in `on_test_epoch_end`** (`lightning_module.py:558`), a walltime kill produces *no* `baselines/results/<tag>.json` — the entire node-day is lost.

Mitigations (pick at least one):
- Add `trainer.limit_train_batches=<N>` (or a train `stride>1`) to the orchestrator defaults so an epoch has a known duration.
- Add `trainer.val_check_interval` so checkpoints/early-stop don't wait a whole epoch.
- Add resume: the ckpt dir per tag already exists (`${CKPT_DIR}/${tag}`); pass `+ckpt_path=.../last.ckpt` when present.
- Consider `#SBATCH --signal=B:USR1@600` + Lightning's SLURM auto-requeue, or at minimum run `trainer.test` from `last.ckpt` in a follow-up job.

---

## P1 — Wrong / misleading results

### C1. `grassmann_no_modbias` ablation is a silent no-op
`lightning_module.py:105-113` copies only `use_grassmann`, `grassmann_reduced_dim`, `grassmann_window_offsets`, `grassmann_plucker_eps` from the YAML config onto the pretrained config. `grassmann_modality_pair_bias` is **not** copied, and the HF hub config doesn't carry it, so `Chronos2CoreConfig.__init__` default (`True`, `config.py:83`) wins. `grassmann.py:55` reads it from the pretrained config → the ablation trains with the pair bias **enabled**. The §8.1 ablation row will be a duplicate of the flagship within noise. Fix: add `pretrained_config.grassmann_modality_pair_bias = core_config.grassmann_modality_pair_bias` (and audit the other non-propagated fields, e.g. `attn_implementation`).

### C2. goes_pvdaq Skill-Score uses the uk_pv reference
`run_all_mmtsfm.sh:116` (`sp_ref()`) returns "" for goes, so `model.sp_reference_path` stays null — but `ProtocolEvaluator._reference_nrmse` then falls back to `default_reference_path()` = `baselines/results/smart_persistence_s2_ukpv.json` (`protocol_eval.py:26-33`), which exists. Overall SS for any goes run is computed against uk_pv Smart Persistence. Per-plant SS silently absent (plant ids don't match). Not triggered by the default `DATASETS=uk_pv`, but armed the moment you run `DATASETS="uk_pv goes_pvdaq"`. Fix: make the evaluator skip SS when the reference's dataset ≠ run dataset (the manifest records data_path), or pass an explicit per-dataset reference / `sp_reference_path=""` sentinel meaning "no SS".

### C3. Model sees 512 of the 672/1344 loaded history steps
`model.py:424-425` truncates context to `chronos_config.context_length=512`. The data pipeline builds 14-day windows (uk_pv 672 steps, goes 1344), so:
- uk_pv trains/evals on the last ~10.7 days; goes on the last ~5.3 days — the "14-day history" in the configs/protocol is not what the model consumes.
- All the extra history still costs dataloader time and RAM.
Decide explicitly: either raise `context_length` (patched context 672/8 = 84 patches fits comfortably) or shrink `history_days` to match 512 steps, and record the choice in the ablation registry — otherwise cross-baseline comparability claims are off.

### C4. W5 recency-aware summarizer window never activates
`_unpack_batch` passes `video_delta_t` with length `T_v=8` (`lightning_module.py:387-394`), but `LatentSummarizer` only uses it when `frame_delta_t.shape[-1] == T_lat` (`latent_summarizer.py:246`), and V-JEPA's temporal stride 2 makes `T_lat=4`. The condition is false in **every** configuration → silent fallback to uniform spacing. Fix: pool the per-frame Δt to latent resolution (e.g. `vdt.reshape(B, T_lat, 2).mean(-1)`) before passing, or change the summarizer check. Add a one-time warning on fallback so this class of silent no-op is visible.

Minor same-family note: `hist_delta_t` is produced by the dataset (`pv_record.py:312`) and never consumed.

---

## P2 — Inefficiencies

### E1. V-JEPA latent cache: size & inode explosion
One cache file = `[T_lat=4, P=196, D_v=1024]` fp32 ≈ **3.2 MB**. Train split is stride-1, so file count ≈ number of train windows (uk_pv: 69 plants × ~16 k windows/plant-year ≈ 10⁶ → ~3 TB and a Lustre metadata storm; even a quarter of that is trouble). Also: windows with zero frames in the visual window still get a full zero-latent file. Options, in order of leverage:
- Cache at **fp16/bf16** (halves size; the model `nan_to_num`s and runs fp32 anyway).
- Dedupe by content key: key = hash of the 8 selected frame indices instead of `(site, origin)`; adjacent windows and frameless (all-zero) windows collapse massively.
- Or extract train at stride>1 and accept on-the-fly encode for the rest (bad — see B3) / consolidate per-site shards instead of one file per window (better for Lustre).
- At minimum: skip writing all-zero latents and teach `_attach_latents` to treat "no-frame window" as a legit Z=0 hit instead of falling back.

### E2. Frames are decoded even when never used
`_build_entity` (`pv_record.py:311`) always calls `_load_vision` — 8 H5 reads + PIL LANCZOS resizes per entity — even when (a) the latent cache hit attaches `Z` and the model ignores `V`, or (b) `skip_vision_stack=true` (the `numeric_grassmann` run pays full vision-I/O for nothing, plus loads 3.2 MB latents per entity that the model discards). The collated `V` tensor is ~300 MB per batch (16×4×8×3×224²×4 B), pinned, ×4 concurrent runs. Fix: pass a `need_frames: bool` / `need_latents: bool` into the dataset (false when cache-complete or vision skipped) and emit a 1-element placeholder `V`.

### E3. Checkpoints embed the frozen V-JEPA encoder
`VisualEncoder` is a registered submodule; no `on_save_checkpoint` filtering exists. Every ckpt ≈ 1.2 GB (V-JEPA ViT-L) + backbone. Per run: `save_top_k=3` + `last` + periodic every 5 epochs `save_top_k=-1` (`trainer/vision_chronos2.yaml`) ≈ 14+ files → ~25 GB/run, >100 GB for the matrix. Fix: drop `model.video_encoder.*` keys in `on_save_checkpoint` (loader already tolerates missing/stale keys), and cut the periodic callback or give it `save_top_k=2`.

### E4. Per-parameter `.item()` in `on_before_optimizer_step`
`lightning_module.py:637-644` does `p.grad.pow(2).sum().item()` per parameter per step — a host-device sync for each of hundreds of tensors, every step, plus `self.log` of 6+ scalars with `sync_dist=True` combined with `log_every_n_steps=1` (slurm trainer). Cheap fix: accumulate squared norms as tensors and call `.item()` once, and/or run the breakdown every N steps only.

### E5. Node utilization
- Extraction runs serially on GPU 0 (train+val+test, per dataset) while GPUs 1-3 idle; splits are independent → run them in parallel across GPUs.
- Wave dispatch (`run_all_mmtsfm.sh:220-229`): 5 jobs on 4 GPUs = wave of 4 + wave of 1; the second wave leaves 3 GPUs idle for the full duration of the longest run. Replace waves with a simple per-GPU work queue (launch next job on whichever GPU frees first).
- `NUM_WORKERS=8` × 4 concurrent runs = 32 loader workers + 4 mains on `--cpus-per-task=32` → oversubscribed once frame decode is active (see E2).

### E6. Single-GPU DDP + logging cadence
`trainer/slurm.yaml` fixes `strategy: ddp_find_unused_parameters_true` even though the orchestrator pins `trainer.devices=1` — needless process-group init (the reason each run needs its own `MASTER_PORT`) and unused-parameter search every backward. Use `strategy=auto` when `devices=1`. `log_every_n_steps: 1` multiplies logger overhead on million-step epochs.

---

## P3 — Hygiene / provenance

- **H1. Hydra run-dir collisions**: `hydra.run.dir` = `logs/experiments/runs/${now:%H-%M-%S}` (`configs/config.yaml:16`); 4 parallel runs launched in the same second share one dir → `.hydra/` configs overwrite each other and wandb `save_dir`s mix. Add the results tag: the orchestrator can pass `hydra.run.dir=logs/experiments/runs/\${now:...}_${tag}`.
- **H2. Exit code & summary**: script always exits 0 (summary prints `fail=N` but doesn't `exit 1`); SLURM/email/monitoring see COMPLETED. Also `STATUS` rc `skip` branch (`run_all_mmtsfm.sh:246`) is dead code — nothing ever sets it.
- **Stale docs**: repo `CLAUDE.md` still says `sbatch MMTSFM/scripts/slurm_train.sh` — that script no longer exists; the entrypoint is `run_all_mmtsfm.sh`. Also update it to warn about the submit-dir requirement (B2).
- **`torch.load` monkeypatch** (`train.py:23-36`) forces `weights_only=False` globally, overriding the explicit `weights_only=True` in `pv_record._attach_latents`. Harmless today, but the patch silently defeats any caller's explicit choice; prefer `torch.serialization.add_safe_globals` for the omegaconf types.
- **`TORCH_HUB_DIR`** exported in both shell scripts is not a variable PyTorch reads (`TORCH_HOME` is). Harmless, but delete to avoid the impression it does something.
- **precache is all-warn**: every stage failure is `warn` + continue; the final banner prints "DONE" regardless. At least propagate a nonzero exit if any stage failed.
- **Local dev note**: dataset of record was consolidated at `/Volumes/SSD/thesis-dataset/` locally; all configs/scripts still point at `/leonardo_scratch/...`. For any local smoke of the pv_record path, override `data.data_dir`.

---

## What checked out clean

- All 5 ablation override strings compose against the configs (verified with Hydra `compose`); duplicate-key overrides are last-wins in Hydra 1.3.3, so `selfattn_late`'s `data.batch_size=8` correctly beats the orchestrator's `data.batch_size=16`.
- `uv run pytest`: 136/136 pass locally (torch 2.12.1, lightning 2.6.5, hydra 1.3.3).
- `lightning 2.6.5` supports `Trainer.test(..., weights_only=False)` used in `train.py:96-101`.
- `MASTER_PORT=29500+i` is unique per job → no DDP port clash between concurrent runs.
- `smart_persistence_s2_ukpv.json` exists in `baselines/results/` → uk_pv Skill-Score path is live.
- Results land in the baselines schema dir with per-tag names; `_best_finite_checkpoint_path` correctly skips the monitor-less periodic checkpoint callback; test is correctly skipped (with a warning) if no finite best ckpt exists.
- Extraction cache keys (`(dataset, site, origin)`) match between producer (`extract_video_embeddings.py`) and consumer (`pv_record._entity_cache_key`), including the `num_entities` asymmetry (extraction at N=1 covers a superset of training's N=4 groups). Batch→key mapping in the extractor is valid because the loader is unshuffled.
- goes_pvdaq train split has 6 plants ≥ `num_entities=4`; val/test are forced to N=1 in the datamodule, and the `_build_groups` fallback prevents empty datasets.
- `max_output_patches=4` covers both horizons (uk 12→2 patches, goes 24→3).

## Suggested fix order (before submitting)

1. B1 precache import + fail-hard (5 min; unblocks everything offline).
2. B2 `mkdir -p logs/slurm` / submit-dir guard (5 min).
3. B3 remove directory-level extraction skip; version cache dir (10 min).
4. C1 propagate `grassmann_modality_pair_bias` (2 lines — otherwise don't bother running that ablation).
5. B4 pick an epoch-budget strategy (`limit_train_batches` or stride) + resume from `last.ckpt`.
6. C2 SS reference guard before any goes run.
7. E1/E2 cache dtype + skip frame decode on cache hit (biggest throughput wins).
