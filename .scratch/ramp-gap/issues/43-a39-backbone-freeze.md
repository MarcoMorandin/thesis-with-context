# 43 — How frozen is the backbone? 0 vs 3 trainable encoder blocks (A39), and the wording fix

Type: task
Status: resolved (prose fix 2026-09-09; A39 run and called 2026-09-11)

## Prose fix — landed 2026-09-09

Part (b) of the resolution criterion is done and does not wait on A39:

- `knowledge/architecture.md` §2.6 — the stage table's Chronos column now reads `"frozen"`
  in scare quotes, an **S2d row** was added (its freeze policy is deliberately identical to
  S2b, per `stage/s2d.yaml`), and a ⚠ block states the two ways the claim is false, with
  the file:line evidence (`lightning_module.py:225-243` for the by-name trainable set,
  `:245-254` for the encoder tail) and the wording to use verbatim.
- `manuscript/chapter7.tex` — Table~\ref{tab:params} rows corrected; the
  "concrete meaning of frozen" paragraph now carries the precise statement and defers the
  strong claim to A39; the "curriculum freezes the pretrained backbone entirely" sentence
  in §"The failure mode this guards against" corrected; chapter opening softened to
  "largely fixed".
- `manuscript/abstract.tex` — "fuses a frozen Chronos-2" → "a mostly-fixed Chronos-2".
  V-JEPA is left as "frozen": that one is literally true (cached latents, no gradient).

Wording of record: *"Chronos-2 fine-tuned on the train plants (s1), then held fixed except
its last 3 of 12 encoder blocks at 0.1× LR while the visual path trains."*

~~Found while fixing, and corrected in the same table: the manuscript said the encoder had
**12** group-self-attention blocks, 12 feed-forwards and 12 temporal mixers. Every arm runs
`chronos_core_cfg.num_layers: 6` … so the counts were doubled … That makes the unfreeze
**half the encoder**, not a quarter of it.~~

> ⚠ **The struck-through paragraph above was wrong, and was reverted 2026-09-11.** The
> manuscript's original **12** was correct. `chronos_core_cfg.num_layers` is **inert**:
> `lightning_module.py:148-203` builds a `Chronos2CoreConfig` from the YAML, then, when
> `pretrained_model_name_or_path` is set (it always is), *discards* it in favour of
> `Chronos2CoreConfig.from_pretrained("amazon/chronos-2")` and copies over only the
> Grassmann keys, `visual_cross_attn_blocks` and `_attn_implementation`. `num_layers`,
> `d_model` and `num_heads` are never propagated, so the hub config wins:
> `d_model 768, num_layers 12, num_heads 12, d_kv 64, d_ff 3072` (checkpoint `config.json`).
> `knowledge/architecture.md` §1 already said so. The unfreeze is **3 of 12 = a quarter** of
> the encoder, not half. Fixed in `knowledge/architecture.md` §2.6 and
> `manuscript/chapter7.tex` (Table~\ref{tab:params} rows + the "concrete meaning of frozen"
> paragraph). `report/report.typ` was never wrong — it carried 12/768/12 throughout.

Remaining to close: (a) A39's paired deltas vs s2d on SS, ramp NMAE and Δ ramp.

## Question

Two facts the paper must state and one ablation it needs:

1. `vision_chronos2_s2d.yaml` sets `freeze_chronos: true` **and** `n_unfreeze_encoder_blocks: 3`
   with `backbone_lr_ratio: 0.1` — the last 3 of 12 Chronos-2 encoder blocks train at 0.1× LR
   (`lightning_module.py:245–254`). Same for s2a.
2. `stage/s1.yaml` sets `freeze_chronos: false` — the s1 checkpoint every vision arm warm-starts
   from is a **fully fine-tuned** Chronos-2, not the pretrained one.

So "frozen TSFM" is false twice over. The honest wording is "Chronos-2 fine-tuned on train
plants (s1), then held fixed except its last 3 encoder blocks while the visual path trains".
That is a writeup fix, no run needed.

The ablation: does s2d need those 3 blocks, or does a truly frozen s1 backbone read the sky
just as well? Pure config, no code.

**Launch spec** (register as A39):

| Arm | Config | Seeds | Base |
|---|---|---|---|
| A39 | `model=vision_chronos2_s2d +stage=s2d model.n_unfreeze_encoder_blocks=0 compute_marginal_gain=true` | 42, 43, 44 | `uk_pv_s1_selfattn_s{seed}/best.ckpt` |

Resolved when (a) the paired deltas vs s2d on SS, ramp NMAE and Δ ramp are recorded here,
and (b) `knowledge/architecture.md` §2.6 and the manuscript wording carry the corrected
freeze description.

Reads: A39 ≈ s2d → the projector alone suffices; the paper can claim a fixed backbone with
one sentence of nuance about s1. A39 worse by > floor → the 3 blocks are load-bearing;
report it as the adaptation-capacity cost of the mechanism (this is also the answer to
`baselines.md`'s A14 row, from the numeric side).

Context: `paper-readiness-audit.md` §6 item 6.

## Answer — the 3 unfrozen blocks are not load-bearing on either P0 metric (2026-09-11)

n=3 (seeds 42/43/44), `mmtsfm_A39_s2d_ukpv_s{42,43,44}.json`, `git_sha 15673d1`. Manifest
diff against s2d is exactly one key: `train_strategy.n_unfreeze_encoder_blocks` 3 → **0**
(`freeze_chronos: true`, `backbone_lr_ratio: 0.1` unchanged and now inert). Same dataset
fingerprint, n_plants=14 on every file.

| metric | s2d (3 blocks) | A39 (0 blocks) | paired Δ per seed | mean Δ | floor |
|---|---|---|---|---|---|
| skill score | 0.5510 | 0.5529 | +0.0008 / +0.0015 / +0.0034 | +0.0019 | 0.0037 |
| ramp NMAE | 0.1441 | 0.1441 | +0.0025 / −0.0010 / −0.0012 | +0.0001 | 0.0011 |
| Δ ramp (vision marginal) | 0.0064 | 0.0055 | −0.0022 / +0.0003 / −0.0004 | −0.0008 | 0.0011 |
| coverage_80 | 0.7310 | 0.7095 | — | −0.0215 | — |
| quantile ECE | 0.0359 | 0.0444 | — | +0.0085 | — |

**Reading: null on both P0 metrics, with a calibration cost.** Skill score is nominally *up*
but inside the 0.0037 floor and not consistent in magnitude; ramp NMAE is a sign-flipping
+0.0001, well inside the 0.0011 floor. A truly frozen Chronos-2 encoder reads the sky as well
as one with a quarter of its blocks training. The 3-block unfreeze is not where the gain comes from —
it buys calibration (coverage_80 −0.022 and ECE +0.0085 when removed), not accuracy.

The vision marginal gain drops 0.0064 → 0.0055 (−0.0008), inside the floor and sign-flipping
across seeds; not a result, but it is the one number that is directionally against the frozen
arm and should not be quoted as a null with confidence stronger than "within seed noise".

**Consequences.** The strong claim is *recoverable*, with one sentence of nuance that does not
go away: the backbone A39 freezes is still the **s1 fine-tuned** Chronos-2, not the pretrained
one (`stage/s1.yaml` sets `freeze_chronos: false`), so "frozen TSFM" is true of stage 2 only.
The corrected wording of record from the 2026-09-09 prose fix stands for the shipped s2d arm;
whether the paper reports A39 instead — buying "the visual path alone, over a genuinely frozen
backbone" at a calibration cost — is a framing call for ticket
[27](27-resampler-foil-and-paper-framing.md) / [38](38-rewrite-paper-around-s2d.md).

This is also the numeric answer to `baselines.md`'s A14 row: partial backbone unfreeze is not
required for the fusion gain.

Pairs with [41](41-a37-concat-vs-average.md) (A37, sub-cell detail inert) and
[28](28-late-raw-control-arm.md) (A43, placement inert).
