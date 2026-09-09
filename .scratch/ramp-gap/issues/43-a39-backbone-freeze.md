# 43 — How frozen is the backbone? 0 vs 3 trainable encoder blocks (A39), and the wording fix

Type: task
Status: **prose fix DONE (2026-09-09); A39 not yet run — ticket stays open on (a).**

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
its last 3 of 6 encoder blocks at 0.1× LR while the visual path trains."*

Found while fixing, and corrected in the same table: the manuscript said the encoder had
**12** group-self-attention blocks, 12 feed-forwards and 12 temporal mixers. Every arm runs
`chronos_core_cfg.num_layers: 6` (`Chronos2Encoder` builds exactly `num_layers` blocks,
`model.py:174-178`), so the counts were doubled — the "3 of 12" in several model configs'
inline comments is the same stale number. That makes the unfreeze **half the encoder**, not
a quarter of it, which is the fact A39 is testing.

Remaining to close: (a) A39's paired deltas vs s2d on SS, ramp NMAE and Δ ramp.

## Question

Two facts the paper must state and one ablation it needs:

1. `vision_chronos2_s2d.yaml` sets `freeze_chronos: true` **and** `n_unfreeze_encoder_blocks: 3`
   with `backbone_lr_ratio: 0.1` — the last 3 of 6 Chronos-2 encoder blocks train at 0.1× LR
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
