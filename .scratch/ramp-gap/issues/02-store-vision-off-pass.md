# 02 — Store the vision-off pass so ramp can be decomposed

Type: task
Status: resolved
Assignee: claude (session 2026-08-25)

## Question

`ProtocolEvaluator.update` calls `_store_batch` only when `vision_off=False`
(`MMTSFM/src/eval/protocol_eval.py`). Two consequences:

- `_ramp_metrics` reads `self._store`, so **ramp is only ever computed for the vision-on
  pass**. There is no ramp-with-vision-off number anywhere, which means the project's P0
  metric has never been attributed to the visual stream.
- `dump_predictions` writes no `pred_off` npz, so `probes/localize.decompose_by_horizon`
  has no data source and the per-horizon visual decomposition cannot run.

This blocks the headline claim. Vision measures +2.7% NMAE / +2.1% NRMSE on 14/14 plants,
but whether any of that lands on ramp is unknown — and the ~2 h frame decorrelation predicts
the effect should concentrate at `h ≤ 4`, which is testable only once both passes are stored.

Give the accumulator a second store keyed by pass, compute ramp for both, and emit
`nmae_ramp_vision_on` / `nmae_ramp_vision_off` / `delta_nmae_ramp` (and the NRMSE trio)
alongside the existing marginal fields. Dump `pred_off` npz per site. Keep the ramp
thresholds identical across passes — they are a property of the data, not of a model, so
they must be computed once and shared, or the two passes are scored on different subsets.

Must land **before** wave 1 launches, so all five chains emit the decomposition natively
rather than needing a re-score.

## Answer

Done. `MMTSFM/src/eval/protocol_eval.py`, 237 tests green (15 in
`test_protocol_eval.py`, up from 7; the 14 pre-existing ones untouched).

**Shape of the fix.** `_ramp_metrics` split into two pieces:

- `_ramp_masks()` derives the per-site top-decile subset **once, from the
  vision-on store**, and returns it.
- `_score_ramp(store, ramp_masks)` scores any one pass against subsets handed
  to it.

That split is the whole point. Thresholds are a property of the data, exactly as
`baselines/common/runner.compute_ramp_thresholds` treats them — computed once per
eval split, shared by every model. Deriving them per pass would have scored
vision-on and vision-off on *different rows* and made their difference
meaningless, and it would have looked correct in production, because `delta`
comes from `y_true`/`context` and never from the predictions, so both passes
normally see identical deltas.

The regression test defeats that invisibility by feeding the off pass a
deliberately *different* `delta`: shared thresholds yield 0.3, a per-pass
threshold yields 0.5. Literal values, no recomputation of the code's own logic.

**New fields**, present only when `compute_marginal_gain=true`:
`nmae_ramp_vision_on` / `nmae_ramp_vision_off` / `delta_nmae_ramp`, the same trio
for `nrmse_ramp`, at both `overall` and `per_plant`.

**Dumps.** Off pass goes to `<model>_<site>_pred_off.npz`, a separate file rather
than extra keys, so `localize.decompose_by_horizon(pred_on, pred_off, y, mask)`
maps straight onto it and existing readers of the on-pass npz are unaffected.
`dump_predictions` still returns `None` when no on-pass data exists — that
behaviour was asserted by an existing test and it stays a real check.

**Backward compatibility.** With `compute_marginal_gain=False` there is no off
store, no off file, and no new keys. Guarded by
`test_no_off_dump_when_marginal_gain_disabled`.

**Facts later tickets depend on:**

- Test tolerance for anything read out of these buffers is float32, not float64 —
  `_store_batch` stores `float32` deliberately. 1e-9 assertions will fail at
  ~1.2e-8; use 1e-6.
- `_score_ramp` skips a site whose row count differs from its ramp mask rather
  than scoring misaligned rows. Both passes run on the same batch inside
  `test_step`, so this should never fire in production — if it does, the two
  passes saw different windows and the delta is not trustworthy.
- Wave 1 chains must set `model.compute_marginal_gain=true` or they emit none of
  this.
