# Running the component ablations

**Canonical for**: how to launch every ablation registered in
[ablations.md](ablations.md) §2.2–§2.5, in what order, and what each result decides.
The registry says *what exists and why*; this file says *what to type*. Numbers live
in `baselines/results/*.json` — never here.

Config group: `MMTSFM/configs/ablation/<ID>.yaml`, composed with `+ablation=<ID>`.
Each file carries only the delta from its base arm, so `git diff` on it is the ablation.

---

## 0. Before anything: the provenance fix is live

`on_test_epoch_end` used to write a three-key `run_cfg` — `seed`, `model`,
`quantile_levels` — as the results manifest's `config`, and `config_hash` is derived
from it. None of those three keys varies across architectures, so **s1, s2a, s2b and
s2c all carry the identical hash `18d5735b73123686`** on disk. No existing result JSON
can say which model produced it.

`_run_cfg()` now records the resolved `chronos_core_cfg`, `vision_cfg`, freeze/LR
strategy and `eval_control`. Consequences:

- Every ablation below is **self-identifying** from its own results file. This is the
  only reason a 15-run batch is auditable at all.
- New runs get a different `config_hash` from old ones **even at identical settings**.
  That is correct — the old hash was meaningless — but do not compare hashes across
  the fix.
- It does **not** retro-fix the runs already on disk. In particular it cannot resolve
  whether `mmtsfm_s2b_ukpv_wide_*` used `stage/s2a` (late) or `stage/s2b`
  (interleaved): `vision_chronos2_wide.yaml` declares `fusion_mode: "late"` while the
  filename and registry §1 say s2b. Recover it from the SLURM launch line or W&B
  before quoting A13 or the s2c headline. See ablations.md §2.2.

---

## 1. How to launch

### 1.1 The batch — `scripts/ablation_sweep.sh` (default)

The whole manifest goes out as a handful of whole-node jobs, each running four
ablations at once. Submit from a **login node**, from `MMTSFM/`.

```bash
cd MMTSFM
DRY_RUN=1 bash scripts/ablation_sweep.sh          # print the plan, submit nothing
MAIL_USER=you@example.com bash scripts/ablation_sweep.sh
ONLY="A09 A10" bash scripts/ablation_sweep.sh     # just the eval controls
CHAIN=3 bash scripts/ablation_sweep.sh            # 3 linked packs: survive the 24 h cap
```

The manifest is [`MMTSFM/configs/ablation/sweep.manifest`](../MMTSFM/configs/ablation/sweep.manifest)
— one row per ablation (`ID | MODE | STAGE | MODEL_CFG | SEEDS | BASE`). It is the
only place the run list lives; the script expands seeds, resolves each warm-start
or scoring checkpoint from `BASE`, and packs the jobs round-robin onto `NPACKS`
nodes. ⚠ **Verify the `BASE` column against `ls $CKPT_DIR` before the first
submission** — the s2c directory name depends on which `MODEL_CFG` produced it
(`vision_chronos2_s2c` → `uk_pv_s2c_s2c_s42`, `vision_chronos2_timeselfattn` →
`uk_pv_s2c_selfattn_s42`). A missing checkpoint is fatal at submit time, on the
login node, before an allocation is spent.

**What packing does and does not buy.** Each pack is `--nodes=1 --gres=gpu:4
--cpus-per-task=32` running a work queue: four runs at a time, a freed GPU takes
the next job. Against four separate `--gres=gpu:1` jobs that is the **same
node-hours** — Leonardo bills allocated resources and four quarter-nodes equal one
node — so this does not stretch the `IscrC_MTSFM` budget. What it buys: one queue
wait per pack instead of one per ablation (45 → 4), one env + data warm-up
amortised over the pack, no hand-typed `sbatch` (and so no hand-typed wrong
`PREV_CKPT`), and a continuation chain that resumes whatever the walltime cut off.

**What it costs, used carelessly: tail idle.** A pack drains a queue with 4 slots;
once the queue is empty, a slot that finished early sits **allocated and idle**
until the pack's slowest run ends. The sweep therefore bins jobs by cost class
(eval = minutes, train = hours), gives each class its own packs in proportion to
its job count, and round-robins only *within* a class — a minutes-long control
never shares a pack with a 20 h training row. **With fewer than ~8 jobs, use §1.2
instead.**

**Wall-clock is `NPACKS`.** Nothing in the manifest waits for anything else — every
row warm-starts from an arm already on scratch — so the sweep is embarrassingly
parallel and the only limit is how many nodes you ask for. Concurrency is
`NPACKS × GPUS`; a pack of *n* jobs takes `ceil(n/4)` waves of one run-length.

| `NPACKS` | train packs | jobs/pack | waves | wall-clock ≈ |
|---|---|---|---|---|
| 4 (default) | 3 | 8/7/7 | 2 | 2 × run |
| 6 | 4 | 6/6/5/5 | 2 | 2 × run |
| **8** | 6 | 4/4/4/4/3/3 | **1** | **1 × run** |

for the 34 runs the manifest currently expands to (12 eval + 22 train). Node-hours
are identical in all three — only the calendar changes, plus a couple of idle GPUs
in the last wave. Bound by the account's max running jobs, not by the budget:
check with `scontrol show partition boost_usr_prod` and
`sacctmgr show assoc user=$USER format=user,maxjobs,maxsubmitjobs,grpnodes`.

**The axis deliberately not taken:** running one ablation across 4 GPUs
(`trainer.devices=4`) would cut its wall-clock ~4× at the same node-hours, but the
effective batch becomes `4 × BATCH × ACCUM` and the run is no longer comparable to
the single-GPU arm it ablates — the ablation would measure the batch size as much
as the component. Both launchers pin `trainer.devices=1 trainer.strategy=auto` for
that reason. Parallelise across runs, never inside one.

Knobs: `MANIFEST` `DS` `NPACKS`(4) `GPUS`(4) `CHAIN`(1) `SWEEP_TIME`(24:00:00)
`SWEEP_EPOCHS`(20) `SWEEP_BATCH`(4) `SWEEP_ACCUM`(4) `ONLY` `SEEDS` `MAIL_USER`
`DRY_RUN`. Chaining uses `afterany`, not `afterok` — the 24 h cap is reported as
TIMEOUT, i.e. a failure, and resuming from it is the whole point; `SKIP_DONE=1`
(default) makes the repeat idempotent by skipping any run whose results JSON
already exists. Status by mail; never poll with `watch -n N squeue`.

### 1.2 One ablation — the stage launcher

For a single run, a debug pass, or a batch too small to fill a node. The ablation
rides in `EXTRA_OVERRIDES`, which `curriculum_stage.sbatch` appends last, so it wins
over the stage and model configs.

```bash
cd MMTSFM
STAGE=s2c DS=ukpv DCFG=ukpv MODEL_CFG=vision_chronos2_s2c \
TAG=mmtsfm_A17_ukpv_s42 SEED=42 \
DATA_DIR=... CKPT_DIR=... RESULTS_DIR=results \
VJEPA_CACHE=... PREV_CKPT=<uk_pv_s1_selfattn_s42/best.ckpt> \
MARGINAL_GAIN=1 EXTRA_OVERRIDES="+ablation=A17" \
sbatch scripts/curriculum_stage.sbatch
```

Both launchers build the Hydra command from the same
`scripts/lib/stage_cmd.sh`, so a packed ablation and a hand-launched stage run
byte-identical commands. Do not add overrides to one launcher only — that is how an
ablation ends up compared against an arm trained slightly differently with nothing
in the results saying so. `tests/test_ablation_sweep.py` and
`tests/test_curriculum_runner_wave_safety.py` hold both paths to it.

### 1.3 Rules that apply to every row below

- **`MARGINAL_GAIN=1` is mandatory.** Without it the vision-on/off decomposition is
  absent from the JSON and the ablation cannot be read.
- **`TAG` must name the ablation.** The sweep builds it as
  `mmtsfm_<ID>_<stage>_<ds>_s<seed>`; a hand-launched stage must do the same by hand.
  The launcher passes `model.results_tag=${TAG}` on the command line, which beats
  anything a config sets. Two runs sharing a tag share a results JSON and a
  checkpoint directory and clobber each other mid-run with no error raised — the
  sweep refuses to submit if any two tags collide.
- **Seeds 42/43/44** for anything with a verdict attached. n=1 is a look, not a result.
- **Branch `exp/<ID>-<short-name>`** before launching (registry rule).
- `DRY_RUN=1` prints the composed Hydra command and exits — use it once per new ID.

Composition check without SLURM:

```bash
uv run python -m mmtsfm.train +stage=s2c model=vision_chronos2_s2c +ablation=A17 --cfg job
```

---

## 2. Order of work

The batch is deliberately front-loaded: the first three rows can each kill a claim,
and everything after them is analysis that only matters if the claims survive.

| # | Run | Cost | Decides |
|---|---|---|---|
| 1 | **A09 + A10** | eval-only, minutes | whether the model reads the sky at all |
| 2 | **A17** | 3 × full train | grid vs decoder-granularity — the architecture claim |
| 3 | **A22** | 3 × full train | the other half of the same attribution |
| 4 | **A23** | 3 × full train | whether A01's negative result is clean |
| 5 | A19, A27 | 3 × full train each | mechanism honesty (τ, numeric dropout) |
| 6 | A24–A26, A18a/b | 1–3 × train each | component-by-component effectiveness table |
| 7 | A20, A21, A28 | 3 × train each | follow-ups; A21 only if A17 is positive |

---

## 3. Eval-only controls — run these first

No training. Both score an **existing** checkpoint, so they cost one test pass and can
run today against the s2c seeds already on disk.

```bash
# A09 — temporal shuffle
uv run python -m mmtsfm.train +stage=s2c model=vision_chronos2_s2c +ablation=A09 \
  train=false ckpt_path=<uk_pv_s2c_selfattn_s42/best.ckpt> \
  data.data_dir=$DATA_DIR data.vjepa_cache_dir=$VJEPA_CACHE \
  model.results_dir=results model.results_tag=mmtsfm_A09_s2c_ukpv_s42

# A10 — mismatched plant
uv run python -m mmtsfm.train +stage=s2c model=vision_chronos2_s2c +ablation=A10 \
  train=false ckpt_path=<uk_pv_s2c_selfattn_s42/best.ckpt> \
  data.data_dir=$DATA_DIR data.vjepa_cache_dir=$VJEPA_CACHE \
  model.results_dir=results model.results_tag=mmtsfm_A10_s2c_ukpv_s42
```

Run both against **s2a as well** — A09 on s2a is the negative-control's own control. If
s2a degrades as much as s2c under shuffling, the control is measuring something other
than motion.

What the controls do, exactly:

- **A09 `shuffle_frames`** — permutes the temporal axis within each sample. Every frame
  still belongs to the right plant on the right day, so per-plant bias and time-of-day
  marginals survive intact; only cloud **motion** is destroyed. Δt is deliberately left
  unpermuted, so the claimed timestamps no longer describe the frames.
- **A10 `swap_plant_frames`** — rolls the visual tensors one position along the batch:
  frames, mask and Δt move together, so the input is internally consistent but belongs
  to another site. Destroys plant identity and site bias as well as ordering.

Both are pure test-time input transforms — no weights, no training path, no config
touched by the model under test. The permutation is seeded from `seed`, so a rerun
reproduces the same corruption.

**Expected**: s2c degrades on both; s2a moves little on either (its Δ is already ~0).
**If s2c does not degrade**, the 0.0056 Δramp is a correlate — a per-plant or
per-day constant the images happen to encode — and the spatial claim does not survive
review. That is the single cheapest way to find out, which is why it goes first.

⚠ Caveat that limits both controls on `uk_pv`: all 07:30 origins hold a blank-frame
embedding, so vision is only measurable at h≤5. Report the controls on the same
subset as the headline, not on the full test set.

---

## 4. Training ablations

### 4.1 The attribution pair — A17 and A22

s2c changes five things at once relative to s2b (ablations.md §2.2). These two runs
split the two that could plausibly carry the gain:

| Run | Holds fixed | Varies | Base |
|---|---|---|---|
| **A17** | future_query, 3 decoder positions, s1 warm start | 4×4 grid → **1×1 blob** (64 KV tokens → 4) | `model=vision_chronos2_s2c +stage=s2c` |
| **A22** | future_query, 4×4 grid, s1 warm start | 3 decoder positions → **1** (`output_patch_size` 4 → 16) | same |

Read them together:

- gain survives A17, dies in A22 → the win is the **decoder**, not the images.
- gain dies in A17, survives A22 → the win is the **spatial field**. This is the paper.
- gain dies in both → the two interact; the claim needs rewording, not just a number.
- gain survives both → something else in the five is responsible; A19/A23 next.

A17 is registry-flagged as the highest-value open item and is the one run that cannot
be substituted.

### 4.2 A23 — recipe-matched s2a

s2a trains at `visual_dropout_prob=0.3`; s2b and s2c at `0.5`. A01 reports s2a's ramp
gain as a recipe effect (Δramp 0.0000 with vision off) — but s2a is also the only arm
whose recipe differs, so *"late fusion cannot read images"* and *"late fusion got
weaker modality dropout"* are not currently separated by anything on disk. A23 makes
the recipe identical and leaves fusion mode as the only difference.

```bash
STAGE=s2a MODEL_CFG=vision_chronos2 TAG=mmtsfm_A23_ukpv_s42 \
EXTRA_OVERRIDES="+ablation=A23" ...
```

If Δ stays ~0, A01 becomes a clean publishable null. If Δ moves, A01 as written is
wrong and the negative result has to be withdrawn.

### 4.3 A19 — the lead-time embedding τ

Directly load-bearing for **D15**. The three future queries are separated by (a)
sequence position and (b) a learned per-lead-time offset. D15 found they differentiate
in block 0 and flatten afterwards; A19 says which of the two mechanisms does the work.
If the ramp gain is unchanged without τ, the embedding is decoration and the D15
"collapse" reading is the honest one to publish.

With `use_lead_time_embed: false` the parameter is not constructed, so it is absent
from the `state_dict`. Harmless — every load in this repo is `strict=False`.

### 4.4 A27 — numeric dropout

Training drops the visual stream at p=0.5 and the numeric stream at p=0.1 (effective
rate 0.1 × (1−0.5)). The numeric side has never been justified by a measurement, and it
is a regulariser sitting on the arm whose headline claim is *"the model uses the
images"*. A27 sets it to 0. The question a reviewer will ask: does forcing occasional
vision-only prediction manufacture the marginal gain?

### 4.5 A24 / A25 / A26 — embedding leave-one-out

Three additive channels were added to disambiguate the packed sequence — modality
(numeric/visual), segment (context/future), token type (target/covariate/visual, the
"M1 fix") — and **none has ever been ablated**. One run each, on the arm you intend to
publish.

The embedding tables are still constructed when disabled; only the additive
contribution is suppressed, at the single place each channel is applied. So the
`state_dict` shape is unchanged and every one of these warm-starts from an existing
checkpoint. Expect A25 (modality) to be the most likely null on s2c, where visual
tokens never enter the sequence at all and are reached only through cross-attention.

### 4.6 A18a / A18b — cross-attention depth

s2c injects visual KV into the last 4 of 6 encoder blocks; nothing establishes that 4
is needed. `{1, 2, 4}` — the k=4 point is the existing s2c number, so only two new runs.
D15 already shows blocks 1–2 are near-flat, which makes k=1 a live possibility and a
cheaper published architecture.

### 4.7 A20 / A21 / A28 — follow-ups

- **A20** `output_patch_size=1` → 12 future positions, one per horizon step. Costs a
  longer sequence and more fresh `future_patch_embedding` weight.
- **A21** `visual_grid=14`, the native ViT patch grid. **Only if A17 is positive** —
  otherwise there is no spatial claim to refine. Cost warning: KV tokens = `T_lat × g²`,
  so ~784 per sample against s2c's 64. Reduce `BATCH_SIZE`.
- **A28** entity embedding. `data/ukpv.yaml` declares `num_entities: 4` but every model
  config sets `n_entities: 0`, so `entity_embed` is `None` and `add_entity` is a
  pass-through: **the plants in a group batch are currently indistinguishable to the
  model.** This is also the missing number for W4 (cross-plant group batching — code
  done, no result on disk).

---

## 5. After each run

1. Validate the JSON before trusting a number — `result-aggregator` agent, or
   `uv run python baselines/scripts/aggregate_all.py`.
2. Check the ablation actually took effect: the results manifest's `config` block now
   carries the full architecture, so `vision_cfg.visual_grid` etc. can be read straight
   out of the file. A silently-dropped override used to be invisible.
3. Compare against the **pre-registered floors** — ramp NMAE 0.0011, skill score 0.0037.
   A difference smaller than its floor is a null, not a win.
4. Update the ablation's row in [ablations.md](ablations.md) → `DONE` with the W&B run
   ID and the key metric.

---

## 6. What is deliberately not here

- **A03 (Grassmann)** — a decision, not a run. Zero Grassmann results exist anywhere;
  standing decision 4 says the fusion mechanism is the contribution and Grassmann is
  negotiable. Cutting it from the claims is the cheaper honest option.
- **A14 (visual backbone unfreeze)** — not runnable as written. With `VJEPA_CACHE` set,
  `_unpack_batch` fills `video_latents` and leaves `video=None`, so both fusion branches
  consume cached latents and never call the encoder; the unfreeze policy flips
  `requires_grad` on modules outside the autograd graph. State it as a limitation.
- **A07 / A08 / A15 (retrieval)** — deferred, future work.
