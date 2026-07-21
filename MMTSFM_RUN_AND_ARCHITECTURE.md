# MMTSFM — Leonardo Run Guide & Architecture

Two parts:
1. [How to train MMTSFM on Leonardo, step by step](#1-training-on-leonardo) — for **both** the
   Grassmann-mixing and TimeSelfAttention variants.
2. [The model architecture as implemented in the code](#2-architecture-as-implemented).

All commands run from `MMTSFM/`. Shapes are the `uk_pv` reference config.

---

## 1. Training on Leonardo

The model trains through a **4-stage curriculum** (S1 → S2a → S2b → S3), submitted as
four dependency-linked SLURM jobs per dataset. Each stage warm-starts (weights only) from
the previous stage's `best.ckpt`.

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

Then pre-extract the V-JEPA latents for both datasets (avoids re-encoding video every
training step — highly recommended). The extractor uses argparse (not Hydra); the cache dir
must be `<root>/<dataset>/<arch>_f<frames>_s<size>` and the **train stride must equal the
training `TRAIN_STRIDE`** or the cache keys won't be found:

```bash
DATA_DIR=/leonardo_scratch/fast/IscrC_MTSFM/data
for DS in uk_pv; do          # uk_pv only — goes_pvdaq is out of scope (needs LOPO, see below)
  for SPLIT in train val test; do
    CACHE=/leonardo_work/IscrC_MTSFM/vjepa_cache/$DS/vit_large_f8_s224
    STRIDE=(); [ "$SPLIT" = train ] && STRIDE=(--stride 12)
    uv run python scripts/extract_video_embeddings.py \
      --encoder vjepa2 --vjepa-arch vit_large --dataset $DS --split $SPLIT \
      --video-frames 8 --img-size 224 --imagenet-norm \
      --data-dir $DATA_DIR --cache-dir $CACHE \
      --batch-size 8 --num-workers 4 "${STRIDE[@]}"
  done
done
```

> Shortcut: `PREEXTRACT_VJEPA=1 DATASETS="uk_pv" sbatch scripts/run_all_mmtsfm.sh`
> pre-extracts the same cache as part of its own run.

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
| `DATASETS` | `uk_pv` | primary benchmark. goes_pvdaq excluded — needs LOPO (BASELINE_PROTOCOL §2/§4.1) |
| `MAIL_USER` | *(empty)* | email for `--mail-type=END,FAIL`; empty disables |
| `S1_EPOCHS … S3_EPOCHS` | 40 / 20 / 20 / 50 | per-stage max epochs |
| `S1_TIME … S3_TIME` | 12h / 8h / 8h / 20h | per-stage SLURM walltime |
| `TRAIN_STRIDE` | 12 | train window stride (must match the latent cache) |
| `N_VIS` | per-dataset (uk_pv 1, goes 2) | visual summary tokens per row; ablation knob for the vision-compression axis |
| `BATCH_SIZE` | 16 | per-GPU batch |
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

## 2. Architecture as implemented

Backbone = **Chronos-2** at its native size (`amazon/chronos-2`): **d_model 768, 12 layers,
12 heads (d_kv 64), d_ff 3072, patch 16, 9 protocol quantiles, arcsinh norm**. The YAML
`d_model:512/num_layers:6` are ignored — `from_pretrained` keeps the checkpoint's size.

Vision = **V-JEPA 2.1 ViT-L/16** (spatiotemporal, frozen→progressively unfrozen).

Reference shapes: `uk_pv`, batch `B`, one entity/row. `T_ctx = ceil(672/16) = 42`,
`n_vis = 1` (uk_pv: the 6h visual window = `ceil(12/16)=1` TS patch; goes_pvdaq = 2),
`T_fut = ceil(12/16) = 1`.

```text
                          INPUTS  (per entity row)
  Y hist [B,672]      X_cov [B,672+12,14]      V frames [B,3,8,224,224]
  PV power            14 protocol covariates    8 RGB sky/sat frames, 6h window
       │                    │  (future 12 steps)          │
       │ arcsinh + patch16  │ (each channel → its own     │ whole clip → V-JEPA 2.1
       ▼                    │  covariate token-row)       ▼ (tubelet, temporal stride 2)
  input_patch_embedding ◄───┘                      [B, 4, 196, 1024]   (T_lat,P,D_v)
  (PRETRAINED — transfers)                                │  ViT-L/16: 4 time × 196 patches
       │                                                  ▼
  context [B,42,768]                              LatentSummarizer  (Perceiver, causal,
  future  [B, 1,768]                              spatial compress, null token)
  14 cov rows [B,1,768] each ──┐                         │
                               │                  visual summary [B,1,768]  (n_vis, uk_pv)
                               │                         │
                               ▼                         ▼
        MultimodalEmbedding (additive): modality{num=0,vis=1} · segment{ctx,fut}
        · token-type{target,cov,vis} · entity · RoPE positions
                               │
              ┌────────────────┴───────────────────────────────┐
              │  FUSION  (VisionChronos2Config.fusion_mode)     │
              ├──────────────────────┬──────────────────────────┤
              │  late (S2a)          │  interleaved (S2b/S3)     │
              │  vis → CrossModal-   │  weave vis into refinement│
              │  Adapter → N_soft=1  │  window only:             │
              │  batch rows          │  [ts … ts, ts, v, fut]    │
              │                      │  seq = 42+n_vis+1 (uk_pv  │
              │                      │  n_vis=1 → 44)            │
              └──────────┬───────────┴─────────────┬─────────────┘
                         │                          │
   sequence per row:  [B, T_ctx(+n_vis) + T_fut, 768]
   + covariate rows + (late) visual rows on the BATCH axis, sharing group_id
                         │
                         ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Chronos2Encoder  ×12 blocks                                          │
   │                                                                       │
   │   1. TEMPORAL MIX  (choose one — this is the variant switch)          │
   │      ├─ CausalGrassmannMixing   [vision_chronos2_grassmann]           │
   │      │    O(L) Plücker: reduce→RoPE→wedge p=z_{i-δ}∧z_i on G(2,32),    │
   │      │    offsets {1,2,4,8,12,16}, softmax multi-scale, gated α,       │
   │      │    +4 modality-pair biases. FROM SCRATCH.                      │
   │      └─ TimeSelfAttention + RoPE [vision_chronos2_timeselfattn]       │
   │           O(L²) causal self-attention over the time axis.             │
   │                                                                       │
   │   2. GroupSelfAttention  — BATCH axis; fuses target + 14 covariate    │
   │      rows (+ late visual rows) across rows sharing group_id. PRETRAINED│
   │                                                                       │
   │   3. FeedForward  (LN → 768→3072 → ReLU → 3072→768, residual). PRETR. │
   └─────────────────────────────────────────────────────────────────────┘
                         │  last T_fut hidden states (target rows only)
                         ▼
   output_patch_embedding → quantiles×patch  →  reshape → arcsinh⁻¹
                         │
                         ▼
              Ŷ  [B, 12, 9]   (horizon × 9 quantiles)
                         │
                         ▼
              Masked pinball loss   →   NMAE / NRMSE / Skill-Score
```

### Component map (file → role)

| Component | File | Notes |
|-----------|------|-------|
| Assembly, fusion routing, interleave | `models/chronos2/vision_chronos2.py` | `VisionChronos2Model` |
| Encoder stack, GroupSelfAttention, FFN | `models/chronos2/model.py` | native Chronos-2, 12 blocks |
| Grassmann mixer | `models/chronos2/grassmann.py` | `CausalGrassmannMixing`, O(L) |
| TimeSelfAttention + RoPE | `models/chronos2/layers.py` | O(L²) diagnostic mixer |
| V-JEPA 2.1 wrapper | `models/vision/visual_encoder.py` | ViT-L/16, torch.hub |
| Spatial summarizer | `models/vision/latent_summarizer.py` | Perceiver, causal, null token |
| Late-fusion soft tokens | `models/vision/cross_modal_adapter.py` | `N_soft=1` batch rows |
| Training loop, freeze/warmup/unfreeze | `models/chronos2/lightning_module.py` | per-stage policy |
| Hydra entry, warm-start / resume | `train.py` | `init_ckpt` vs `ckpt_path` |

### Notes on what actually happens (verified)

- **Pretrained tokenizer transfers** — `input_patch_size=16` matches the checkpoint, so
  `input_patch_embedding` loads pretrained (only the 9-quantile output head reinitializes).
- **Known future covariates are live** — the 14 covariate channels enter as batch-axis
  token-rows (mask=1 so values survive) and reach the encoder in **both** late and
  interleaved fusion; perturbing a covariate changes the forecast (regression-tested).
- **Variant switch is clean** — `MODEL_CFG` alone selects Grassmann vs TimeSelfAttention;
  the stage schedule is identical.
- **V-JEPA is a video encoder**, not a per-frame image encoder — it consumes the whole
  8-frame clip and encodes motion internally (temporal stride 2 → 4 latent steps).
