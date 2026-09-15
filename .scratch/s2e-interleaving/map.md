# Map: Recover temporal interleaving while retaining S2D as the control

Label: wayfinder:map

## Destination

Explain the S2D–S2E regression and choose a controlled Leonardo ablation plan for a
temporally interleaved multimodal architecture. The user explicitly requires true
interleaving and retains S2D as the forecasting control (2026-09-14).

## Notes

- Planning scope: diagnose and choose experiments; implementation, registration,
  cache extraction and SLURM submission are subsequent work.
- Use Wayfinder, grilling and domain-modeling; use GitNexus for code and
  [the knowledge index](../../knowledge/INDEX.md) for project facts.
- This narrower effort reopens the architectural destination of
  [Closing the ramp gap](../ramp-gap/map.md); it does not rewrite that map's history.
- This session charts the map. Preliminary measurements are comments on tickets,
  not resolved design decisions. Read open child tickets in numeric order.
- Preserve existing working-tree edits, including A45a. No experiment code was
  changed in this session. GitNexus index matches HEAD
  `33923187f64aaf8e80571e7b1b08e6fdd84513d7`; working files were checked directly.
- Graphify was unavailable through the local shell allowlist; canonical knowledge
  files were used. The generated results ledger and report synthesis are absent
  in this checkout; use the in-process JSON records linked in the tickets.

## Decisions so far

- [Choose a budget-safe selector for interleaved anchors](issues/01-budget-safe-selection.md): A46 retains both endpoints of ten novel spatial trajectories per anchor at the original 20-token budget; S2D and historical S2E stay unchanged controls.

## Not yet specified

If repairing selection and controlling representation do not recover performance,
investigate how to preserve the recent visual context while adding historical
anchors, and whether within-patch numerical refinement is necessary. The next
architecture depends on the controlled results, not on a preferred explanation.

## Out of scope

Training or rescoring checkpoints in this session; changing the read-only dataset;
rewriting data pipelines; promoting a forecast improvement without an in-process
training result; claiming publication novelty from this local investigation.
