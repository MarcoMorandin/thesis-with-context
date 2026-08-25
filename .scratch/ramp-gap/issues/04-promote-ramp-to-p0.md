# 04 — Promote ramp to P0 and record the gate rule

Type: task
Status: resolved

## Question

`knowledge/scope.md` lists P0 metrics as MAE/RMSE, CRPS/pinball, and generalization score.
**Ramp appears nowhere**, yet it is the metric this whole effort is defined by and one of two
P0 metrics per the map's standing decisions. `protocol.md` §5 defines NMAE, NRMSE and the
skill score but likewise never defines the ramp subset, even though
`baselines/common/runner.compute_ramp_thresholds` implements it and every result JSON
reports it.

A metric the code computes, the results carry, and the argument turns on, but the contract
never names, is how a viva question becomes a problem.

Do three things:

1. Add ramp NMAE / ramp NRMSE to `scope.md`'s primary-metrics table at P0, alongside
   generalization.
2. Define the ramp subset in `protocol.md` §5 as it is actually implemented: per-plant
   top-decile |Δy| vs the previous step (step 0 uses the last history step), validity =
   `mask_future · daylight · prev_mask`, threshold computed once per eval split over the
   full test set and shared by every model.
3. Record the A03 gate rule (map Notes, standing decision 3) in `protocol.md` so it predates
   the data rather than being chosen after seeing it.

Also reconcile `scope.md`'s hypothesis ladder, which still marks H1 "validated in MMTSFM" —
the curriculum falsified it for late fusion (s2a 0.5086 vs s1 0.5087).

## Answer

**`scope.md`** — ramp NMAE / ramp NRMSE added to the primary-metrics table at **P0**, beside
generalization, with the reasoning stated: sudden-change accuracy is what decides whether a
forecast is usable for grid integration, and it is the one place a visual channel should pay
off. Two consequences written in: every table reporting a skill score reports ramp beside it,
and nothing is a "win" on aggregate error alone if the ramp column moved the other way.

**`protocol.md` §5** — the ramp subset is now defined where the metrics live, as
`baselines/common/runner.compute_ramp_thresholds` actually implements it: step delta against
the previous step (h=1 uses the last history step), validity `mask_future · daylight ·
prev_mask`, **per-plant top decile over the whole test set**, per-plant macro-average. Stated
explicitly that the subset is a property of the data, computed once and shared — including
across the two passes of a marginal-gain run, which is what `_ramp_masks` enforces. Carries
the T4-T6 non-comparability warning (`n_steps` 417k-5.6M against the protocol's 165,295).

**`protocol.md` §5.1 — pre-registered decision rules.** New section; the A03 gate rule is
recorded there *before* wave 1 launches, with the seed floor spelled out. A rule that selects
itself after seeing the data is not a rule, and §5.1 exists so the next one also has a home.

**Stale facts fixed while in there** — three of these would have reached the manuscript:

- `scope.md`'s ladder still read H1 "validated in MMTSFM". Replaced with the split verdict the
  data supports: falsified for late fusion (s2a 0.5086 vs s1 0.5087), supported inside
  interleaved fusion (2.7 % rel., 14/14 plants), seeded confirmation pending. H2 marked
  supported at n=1 with its mask confound named.
- The dataset of record was `/leonardo_scratch/fast/IscrC_MTSFM/data` in `protocol.md`,
  `scope.md`, `proposal.md`, `dataset.md` and `AGENTS.md` (twice). All now `data_v2`.
- `dataset.md`'s code note pointed at a `thesis-dataset/` path that does not exist; it now
  names the real directory and records that MMTSFM needs no code edit because `pv_record`
  resolves both filenames from `data.data_dir`.

**Facts later tickets depend on:**

- The gate rule is pre-registered in `protocol.md` §5.1. [Call the A03 gate](09-call-the-a03-gate.md)
  must apply it as written or re-decide it as a separate, stated act.
- Ramp is P0 in the contract now, so a wave-2 ramp-weighted objective needs the baseline
  fairness question settled (tier-2 baselines optimise plain pinball) — already in the map's
  fog, not yet a ticket.
