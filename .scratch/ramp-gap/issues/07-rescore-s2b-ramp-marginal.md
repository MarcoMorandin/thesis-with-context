# 07 — Does vision help ramp, and only at short lead?

Type: task
Status: needs-triage
Blocked by: 08

## Question

Once 02 lands, re-score the existing `grassmann@42` s2b checkpoint with
`model.compute_marginal_gain=true` and read the two numbers that have never existed:

1. **Ramp marginal.** `delta_nmae_ramp` — does the visual stream move the P0 metric this
   whole effort is named after? The aggregate marginal is +2.7% NMAE, but ramp is a
   top-decile subset and there is no reason the effect distributes evenly.
2. **Per-horizon marginal.** `probes/localize.decompose_by_horizon` on the `pred_on` /
   `pred_off` dumps. The measured frame decorrelation (8.3 DN at Δt=15 min vs 35.3 DN at
   Δt=2 h, frame std 38.6) predicts the gain concentrates at `h ≤ 4` and vanishes beyond.

That prediction is falsifiable and it matters. 53.3% of scored steps sit at lead > 2 h, so
if the gain *is* front-loaded, the aggregate marginal understates the local effect by
roughly 2× — and the right presentation of H1 becomes a horizon-resolved one, not a single
number. If the gain is instead flat across horizons, the decorrelation argument is wrong and
something other than cloud advection is driving it (see 06).

Uses the existing checkpoint; no training. Caveat from `knowledge/architecture.md` §4:
checkpoints are known not to reproduce their in-process scores, so verify integrity with
`repair_vjepa_checkpoint.py --inspect` first and report the vision-on number next to the
committed `mmtsfm_s2b_ukpv.json` as a consistency check before trusting the delta.

## Progress — 2026-08-25 (not resolved)

**Cannot be driven AFK from this workstation.** No `*.ckpt`, no `vjepa_cache/`, and no
mmtsfm prediction dumps exist locally; only the v2 dataset is synced. The run needs
Leonardo. Re-labelled `ready-for-human`.

Two failure modes that would have produced an empty result were cleared first, so the
queue slot is not at risk:

1. **`write_results` does not whitelist keys** — it serialises `results` wholesale, so the
   new ramp fields reach the JSON.
2. **The ramp trio survives the Lightning path**, verified by
   `test_lightning_test_step_emits_ramp_decomposition`. This required adding `context_mask`
   to the stub's unpacked batch: `_accumulate_protocol` builds `delta` only when both
   `context` and `context_mask` are present, so a batch missing it yields **no ramp metrics
   at all, silently**. Worth knowing if a future run comes back with the aggregate marginal
   but no ramp fields.

### Checklist for the run

Test-only pass over existing weights — one GPU, one test epoch, no training.

```bash
cd MMTSFM
# 0. integrity first: a stripped V-JEPA encoder silently substitutes the pretrained
#    baseline and makes the checkpoint score differently than the run that produced it
uv run python scripts/repair_vjepa_checkpoint.py --inspect \
  --target $CKPT_DIR/uk_pv_s2b/best.ckpt

# 1. re-score with the marginal pass enabled
uv run python -m mmtsfm.train \
  +experiment=ukpv +stage=s2b model=vision_chronos2_grassmann \
  train=false ckpt_path=$CKPT_DIR/uk_pv_s2b/best.ckpt \
  model.compute_marginal_gain=true \
  model.results_tag=mmtsfm_s2b_ukpv_marginal_ramp \
  model.results_dir=$REPO/baselines/results \
  model.sp_reference_path=$REPO/baselines/results/smart_persistence_s2_ukpv.json \
  data.data_dir=$DATA_DIR \
  data.vjepa_cache_dir=$VJEPA_CACHE_ROOT/uk_pv/vit_large_f8_s224 \
  model.vision_cfg.n_visual_context_steps=1 \
  trainer=slurm trainer.devices=1 trainer.strategy=auto
```

### The gate on trusting the answer

Before reading any delta, check `nmae_vision_on` against the committed
`mmtsfm_s2b_ukpv.json` value **0.0738232107346168**.

- Matches → the checkpoint reproduces; the deltas are trustworthy.
- Does not match → `architecture.md` §4's checkpoint-integrity problem has fired. **Stop.**
  The in-process numbers are the record, and a re-scored delta from a checkpoint that does
  not reproduce is not evidence. Escalate to a fresh run rather than reporting it.

### What to read out

1. `delta_nmae_ramp` and `delta_nrmse_ramp` in `overall` — the first attribution of the P0
   metric to the visual stream. Compare against the aggregate `delta_nmae` = 0.0020026: a
   **larger** ramp delta means vision is doing its job where it matters; a **smaller** one
   means the aggregate gain comes from ordinary steps and the physical story is wrong.
2. Per-horizon, via `probes/localize.py::decompose_by_horizon` over the newly dumped
   `predictions/mmtsfm_s2b_ukpv_marginal_ramp_<site>_pred.npz` and `..._pred_off.npz`.
   Prediction from the frame decorrelation (8.3 DN at 15 min vs 35.3 DN at 2 h, frame std
   38.6): gain concentrated at **h <= 4**, ~zero beyond. Read h<=5 and h>=6 separately —
   the 13:30 origin contributes zero scored steps at h>=6, so they are different
   populations, not one curve.
3. Per-plant `delta_nmae_ramp`. The aggregate marginal is positive on 14/14 plants; if the
   ramp delta is not, say which plants dissent and whether they share anything.

Falsifiable either way: a flat-across-horizon ramp gain contradicts the advection account
and points at whatever [Does the model read cloud motion, or time of day?](06-shuffled-frames-control.md)
is testing.

## Re-scoped 2026-08-25 — the shortcut is gone

[Fix the interleaved mask override](03-mask-override-before-or-after.md) landed option (a),
which invalidates this ticket's original plan. Re-scoring the existing `grassmann@42`
checkpoint now runs the **fixed** interleaved path over weights trained by the **buggy**
one. That is not the same model, so the vision-on consistency check against 0.0738232 would
trip — correctly, but for a reason that has nothing to do with checkpoint integrity, and it
would be easy to misread as the `architecture.md` §4 problem.

Two ways to still get the number, and the cheap one is now the wrong one:

- **Pin the pre-fix commit** and re-score there. Gives the ramp marginal for the pre-fix
  model — a number about code that no longer exists. Not worth a queue slot.
- **Take it from wave 1.** All six chains run with `MARGINAL_GAIN=1`, so each emits its own
  ramp on/off decomposition natively, on post-fix code, at n=3 per configuration. Strictly
  better than what this ticket was going to produce.

So this ticket is now blocked on wave 1 and reduces to the part wave 1 does *not* do: the
**per-horizon** decomposition via `probes/localize.py::decompose_by_horizon` over the
`_pred.npz` / `_pred_off.npz` dumps, testing whether the visual gain concentrates at
`h <= 4` as the ~2 h frame decorrelation predicts. Read `h<=5` and `h>=6` separately — the
13:30 origin contributes zero scored steps beyond h=5, so they are different populations.
