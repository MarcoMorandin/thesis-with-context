# Runbook — how to run everything

Operational single source of truth. **What** the model is and **why** → [architecture.md](architecture.md).

| I want to… | Go to |
|---|---|
| Train MMTSFM on Leonardo (both mixer variants) | §1 below |
| Run the baseline suite | [../baselines/README.md](../baselines/README.md) · [../baselines/RUNNING_ON_LEONARDO.md](../baselines/RUNNING_ON_LEONARDO.md) |
| Register a new ablation | [ablations.md](ablations.md) · `/register-experiment` |
| Scaffold a new tier baseline | `/new-baseline` |
| Refresh the graphs | `node .gitnexus/run.cjs analyze` (code) · `graphify update knowledge/` (prose+papers) |

MMTSFM commands run from `MMTSFM/`. Shapes are the `uk_pv` reference config.

---

## 1. Training MMTSFM on Leonardo

The model trains through a **4-stage curriculum** (S1 → S2a → S2b → S3), submitted as
four dependency-linked SLURM jobs per dataset. Each stage warm-starts (weights only) from
the previous stage's `best.ckpt`. Stage semantics and rationale: [architecture.md §2.6](architecture.md).

| Stage | Fusion | Vision | Chronos | Purpose |
|-------|--------|--------|---------|---------|
| **S1** | — (vision skipped) | off | trainable + Grassmann warmup (2000 steps) | TS pretraining, anchor the mixer |
| **S2a** | late | V-JEPA last-4 unfrozen | frozen | align visual space to numeric |
| **S2b** | interleaved | V-JEPA re-frozen | frozen except mixer | cross-modal Plücker alignment |
| **S3** | interleaved | progressive unfreeze | all trainable | full joint fine-tuning |

### Step 0 — one-time login-node precache

On a **login node** (has internet), stage the environment, weights, dataset, and V-JEPA
latent cache. Do this once:

```bash
cd MMTSFM
sbatch scripts/precache_login.sh        # uv sync + Chronos-2 + V-JEPA weights + data check
```

Then pre-extract the V-JEPA latents on a **GPU node** (avoids re-encoding video every
training step). This is a dedicated SLURM job — it needs a GPU, so it must not run on a
login node:

```bash
sbatch scripts/extract_vjepa.sbatch          # uk_pv, splits train/val/test → cache
```

It writes one `<key>.pt` per plant/window under
`/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224/`, and is **idempotent**
(re-running skips files already present). The **train stride must equal the training
`TRAIN_STRIDE`** (default 12) or the cache keys won't match. Override via env:

```bash
DATASET=uk_pv SPLITS="train val test" TRAIN_STRIDE=12 VIDEO_FRAMES=8 IMG_SIZE=224 \
  sbatch scripts/extract_vjepa.sbatch
```

#### Check the pre-extraction is complete and correct

```bash
CACHE=/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224

# 1. Files present + total size (instant, login node):
find "$CACHE" -name '*.pt' | wc -l        # expect ~113k for uk_pv (train stride 12 + val/test)
du -sh "$CACHE"

# 2. A sample latent has the right shape/dtype ([T_lat, P, D_v] = [4, 196, 1024], fp16):
uv run python -c "import torch,glob; f=sorted(glob.glob('$CACHE/*.pt'))[0]; \
z=torch.load(f, map_location='cpu'); print(f, tuple(z.shape), z.dtype)"

# 3. Authoritative completeness — re-submit; if extraction is done, every split logs
#    'done=0 skip=<N>' (nothing left to encode, no GPU work):
sbatch scripts/extract_vjepa.sbatch
#    then, in logs/slurm/<jobid>_mmtsfm-extract-vjepa.out:
grep -E 'DONE done=|\[extract\] DONE' logs/slurm/*mmtsfm-extract-vjepa.out | tail
```

> If the latent cache is absent the curriculum still runs — the vision stages just encode
> V-JEPA **live** on the GPU (correct, slower). The submitter warns you per dataset.

### Step 1 — validate the chain locally (optional but advised)

On a login node (V-JEPA cached), dry-run the whole chain on synthetic CPU data — no
`sbatch`, ~1 min — to catch config/threading errors before spending GPU hours:

```bash
SMOKE=1 SMOKE_STAGES="s1 s2a s2b s3" bash scripts/slurm_curriculum.sh
```

Expect `SMOKE OK — stages [s1 s2a s2b s3] ran and threaded best.ckpt via init_ckpt`.

### Step 2 — submit the curriculum

Set your notification address so you get **email** on stage completion/failure — do **not**
poll the queue with `watch squeue` (CINECA policy).

```bash
export MAIL_USER="your.address@unitn.it"      # END/FAIL emails per stage
export DATA_DIR=/leonardo_scratch/fast/IscrC_MTSFM/data
```

#### 2a. Grassmann-mixing variant (headline model)

```bash
MODEL_CFG=vision_chronos2_grassmann \
  bash scripts/slurm_curriculum.sh
```

- Temporal mixer = `CausalGrassmannMixing` (O(L) Plücker), `use_grassmann=true`.
- Results → `baselines/results/mmtsfm_{s1,s2a,s2b,s3}_ukpv.json`.

#### 2b. TimeSelfAttention variant (diagnostic ablation)

```bash
MODEL_CFG=vision_chronos2_timeselfattn \
  CKPT_DIR=/leonardo_scratch/fast/IscrC_MTSFM/checkpoints/curriculum_tsa \
  RESULTS_DIR=$PWD/../baselines/results \
  bash scripts/slurm_curriculum.sh
```

- Temporal mixer = `TimeSelfAttention` + RoPE (O(L²)), `use_grassmann=false`.
- **Use a different `CKPT_DIR`** so the two variants don't overwrite each other's stage
  checkpoints. Give the runs distinct result tags too if you want both in `ALL_RESULTS`
  (e.g. append a suffix via `MODEL_CFG` — or move the JSONs after each run).
- The stage schedule (fusion/freeze/warmup) is identical; only the mixer differs. The
  Grassmann warmup steps are harmless here (no Grassmann params to warm up).

> The variant is chosen **only** by `MODEL_CFG`. The stage configs no longer force
> `use_grassmann`, so `vision_chronos2_timeselfattn` correctly runs attention-based mixing.

### Step 3 — knobs (env overrides)

| Var | Default | Meaning |
|-----|---------|---------|
| `DATASETS` | `uk_pv` | primary benchmark. goes_pvdaq excluded — needs LOPO (knowledge/protocol.md §2/§4.1) |
| `START_STAGE` | `s1` | begin the chain at a later stage (e.g. `s2a`); warm-starts from the prior stage's `best.ckpt`, no s1 re-run |
| `MAIL_USER` | *(empty)* | email for `--mail-type=END,FAIL`; empty disables |
| `S1_EPOCHS … S3_EPOCHS` | 40 / 20 / 20 / 50 | per-stage max epochs |
| `S1_TIME … S3_TIME` | 12h / 8h / 8h / 20h | per-stage SLURM walltime |
| `TRAIN_STRIDE` | 12 | train window stride (must match the latent cache) |
| `N_VIS` | per-dataset (uk_pv 1, goes 2) | visual summary tokens per row; ablation knob for the vision-compression axis |
| `BATCH_SIZE` | *(per-stage: s1 8, s2a/b/3 4)* | micro-batch; set to force all stages. Drop to 2 if a stage OOMs |
| `S1_BATCH … S3_BATCH` / `S1_ACCUM … S3_ACCUM` | 8/4/4/4 · 2/4/4/4 | per-stage micro-batch + grad accumulation (effective batch ≈ 16) |
| `CKPT_DIR` | `…/checkpoints/curriculum` | stage checkpoints (separate per variant!) |
| `ACCOUNT` / `PARTITION` | `IscrC_MTSFM` / `boost_usr_prod` | SLURM account/partition |

### Step 4 — monitor & collect

- **Monitoring**: rely on the END/FAIL emails. A single `squeue -u $USER` is fine; never
  loop it.
- **Live logs**: `logs/slurm/<jobid>_mmtsfm_<stage>_<ds>.out`.
- **Checkpoints**: `$CKPT_DIR/<ds>_<stage>/best.ckpt` (threaded to the next stage).
- **Metrics**: each stage writes `baselines/results/mmtsfm_<stage>_<ds>.json` (NMAE / NRMSE /
  Skill-Score). Aggregate next to the baselines:
  ```bash
  python baselines/scripts/aggregate_all.py     # refreshes baselines/results/ALL_RESULTS.md
  ```

### Manual single-stage run (debugging)

```bash
python -m mmtsfm.train +stage=s2b model=vision_chronos2_grassmann data=ukpv \
  trainer=slurm trainer.devices=1 data.data_dir=$DATA_DIR \
  init_ckpt=$CKPT_DIR/uk_pv_s2a/best.ckpt \
  model.results_tag=mmtsfm_s2b_ukpv
```

`init_ckpt` = weights-only warm start (fresh optimizer, epoch 0). `ckpt_path` = full-state
resume (same-stage requeue only).

---

## 2. Local development

```bash
uv run pytest                                    # all smoke tests
uv run pytest MMTSFM/tests/test_vision_chronos2.py

uv run python -m mmtsfm.train                    # synthetic-data smoke train
uv run python -m mmtsfm.train data.dataset_name=uk_pv \
  data.data_dir=/leonardo_scratch/fast/IscrC_MTSFM/data
```

`uv` only — never `pip`, `poetry`, or `conda`. Add deps with `uv add <pkg>` / `uv sync`.
