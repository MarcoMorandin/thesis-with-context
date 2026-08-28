# 14 — Build the s2c arm: future queries cross-attend a retained spatial field

Type: task
Status: done (2026-08-28, commit d5de785)
Blocks: 15, 16

## Question

Implement the s2c architecture. The design is settled (grilling session 2026-08-28, below);
this ticket is the build, and it does real work rather than deciding anything.

## Why this and not the summarizer rewrite

Ticket 13 established that ~784 V-JEPA patch tokens collapse to ONE vector in
`LatentSummarizer` before fusion, and that widening `n_soft_tokens` downstream of that
collapse moves nothing (ramp NMAE −0.0003 against a 0.0011 floor). The obvious successor —
N queries *inside* the summarizer — was reviewed and rejected: N Perceiver queries give N
**content summaries**, not a **coordinate system**, and advection is a claim about
coordinates. A cloud edge 50 km west and one 50 km east can pool to nearly the same vector
while implying opposite things about the next 30 minutes.

s2c instead keeps the spatial field intact and lets the *forecast* positions query it.

## Design of record

```
Input        same s1 checkpoint as s2b (s1 has no vision params; borrow via INIT_CKPT)

Visual       V-JEPA latents  [T_lat=4, 14, 14, D]
                  -> spatial block-pool to [4, 4, 4, D]     (4x4 grid, all 4 time slices)
                  -> linear D -> d_model
                  -> 64 visual KV tokens

Forecast     output_patch_size 16 -> 4, giving 3 future positions
                  patch 0 = steps 1-4   (30-120 min)
                  patch 1 = steps 5-8   (150-240 min)
                  patch 2 = steps 9-12  (270-360 min)

Query        existing future embedding
                  + LEARNED lead-time embedding tau (one per position, explicit)
                  + existing solar geometry covariates

Fusion       TimeCrossAttention in the LAST 4 encoder blocks
                  residual added to FUTURE positions ONLY

Trainable    exactly s2b's freeze policy, plus the new cross-attention and the
             V-JEPA -> d_model projection
```

**Do not**: inject these visual tokens into the historical context; add plant lat/lon;
run a second simultaneous visual pathway; change the freeze policy.

## Two facts that shaped this

- **`TimeCrossAttention` already exists and is entirely unused** (`layers.py`) — dead code
  from the Chronos-2 port, with RoPE disabled, which is correct for attending a
  non-temporal spatial field. Most of the machinery is already written.
- **There is currently exactly ONE future token.** `num_output_patches = ceil(12/16) = 1`
  (`lightning_module.py:225`), so the whole 6 h horizon is one decoder position emitting 16
  steps at once — nothing for per-horizon queries to be. Dropping `output_patch_size` to 4
  gives 3. This is cheap because the training logs show `output_patch_embedding` is
  **already reinitialised** on load (shape mismatch against the checkpoint), so no
  pretrained weights are lost.

## Design decisions and why

- **Future-only attachment**, not context interleaving (Q1). It is the hypothesis the probe
  supports, it reuses `TimeCrossAttention` as-is, and it leaves s2b's interleaved path
  intact as the control. The residual gating is load-bearing: if the visual residual reaches
  historical positions the arm changes more than one thing and the comparison to s2b dies.
- **3 future positions first, 12 as follow-up if 3 shows signal** (Q2). 12 is most faithful
  to the physics but changes the decoder most aggressively. Patch 0 (30–120 min) covers
  exactly where the latent probe found signal (ramp R² 0.0512 at t+30, 0.0815 at t+60, gone
  by t+120), so a signal should land in patch 0 or nowhere.
- **4x4 grid** (Q3), because that is precisely the arm the probe measured. A null at 4x4 is
  interpretable; a null at an unmeasured resolution is not. 14x14 is the follow-up.
- **tau + existing solar geometry, no lat/lon** (Q4). Lat/lon invites memorising training
  plants, which attacks the cross-plant generalisation claim directly.
- **All 4 temporal slices as KV** (Q5). Motion is not inferable from one instant.
- **Last 4 blocks** (round 2 Q1), matching the existing `n_unfreeze_encoder_blocks`
  convention and bounding new parameters against a frozen backbone.

## The trap to avoid

The learned lead-time embedding must be **explicit and learnable**, not a positional index
the model may or may not use. Without it, `q_30`, `q_120` and `q_240` can be near-identical
and the model simply learns three generic visual summaries — a null that looks like an
architecture failure but is really a degenerate parameterisation. Ticket 15 exists to detect
exactly this.

## Done when

- [x] s2c runs end to end, loss finite, on the fake-encoder test fixture
- [x] visual residual provably reaches future positions and provably does NOT reach context
      positions (asserted by test, not by inspection)
- [x] `output_patch_size=4` produces 3 future positions and 12 scored steps, no leftover
- [x] N=1-equivalent path (s2b) is untouched — existing tests still pass
- [x] arm identity is disjoint from every wave-1 and wave-2 tag and checkpoint dir

## Resolution (2026-08-28, commit `d5de785`)

Built as specified. Every "Done when" box checked: 24 new tests in
`MMTSFM/tests/test_s2c_future_query.py`, full suite 312 passed.

Config surface: `visual_cross_attn_blocks` (core config, default **0** — arms predating s2c
build no cross-attention module at all, so no checkpoint key and no forward-path difference
can creep in), `visual_grid` (vision config), `fusion_mode: future_query`. Arm identity
`mmtsfm_s2c_ukpv`, configs `configs/model/vision_chronos2_s2c.yaml` +
`configs/stage/s2c.yaml`, launched by

```
START_STAGE=s2c END_STAGE=s2c INIT_CKPT=<uk_pv_s1_selfattn_sNN/best.ckpt> \
  MODEL_CFG=vision_chronos2_s2c ARM_SUFFIX=_s2c bash scripts/slurm_curriculum.sh
```

`s2c` is appended last in `STAGES`, so the default `END_STAGE=s3` chain is unchanged.

### Two latent bugs found while building, both silent

1. **`MHA.forward` pinned the KV length to the QUERY length** when shaping K and V. Correct
   for self-attention, where they are equal by construction; it made *any* cross-attention
   with a differently-sized KV impossible — which is the normal case and precisely s2c's (64
   KV tokens against 3 queries). Fixed with a `shape_kv()` that infers the length. This had
   never surfaced because `TimeCrossAttention` was dead code.
2. **`visual_cross_attn_blocks` was not propagated to the config `from_pretrained` restores**,
   whose default is 0. Real training loads the hub checkpoint, so the YAML value would have
   been discarded, **no cross-attention module would have been built, and s2c would have
   trained as an s2b-shaped model while every log line said s2c** — a null indistinguishable
   from a falsified hypothesis. Same trap the neighbouring `grassmann_modality_pair_bias`
   line already warns about.

### One design point sharpened during the build

The learned lead-time embedding is applied **whether or not the batch carries vision**. It
parameterises the forecast positions; it is not part of the visual pathway. Gating it on
`use_video` would make the vision-free forward a different model, and the marginal-gain pass
would then be measuring tau as well as vision.

### One idealisation the tests document rather than assert

`GroupSelfAttention` runs *after* cross-attention inside the same block and attends along the
**batch** axis. So a row with vision masked off still sees the visual update if another row in
its group had vision on — at future positions only. This is structurally identical to s2b,
where the pooled visual tokens are likewise group-visible. The load-bearing claim is the one
about **time** positions (context positions stay clean), and that is asserted directly.
