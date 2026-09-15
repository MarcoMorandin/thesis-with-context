# Choose a budget-safe selector for interleaved anchors

Type: grilling
Label: wayfinder:grilling
Status: resolved
Assignee: codex
Parent: ../map.md
Blocked by: none

## Question

Which selector preserves temporally informative visual tokens at S2E's existing
budget, and what capacity control distinguishes selector recovery from extra tokens?

## Comments

2026-09-14 charting evidence:

- [evs_select](../../../MMTSFM/src/mmtsfm/models/vision/patch_projector.py:116)
  splits the total budget across anchors. Its first-frame priority is infinite
  for every spatial cell, before top-k selection (lines 150–167).
- S2E's 20 slots per anchor compete with 49 infinitely ranked first-frame cells.
  The selected spatial subset is determined by tied top-k ordering; no token from
  the second latent frame survives. This is also present at S2E's recorded source
  commit `05f3eb1`, verified with `git show`.
- Direct execution of the actual function, finite random inputs and seed 42:
  selected frame indices `[0,2,4,6,8]`, 20 tokens each; second-latent gradient sum
  `0.0`; replacing all second latents changes the selected output by `0.0`.
  An assertion requiring any second-latent token fails. S2D keeps the complete
  first latent plus 49 later-latent tokens at its shipped geometry.
- This proves a selector degeneracy, not its contribution to the forecast gap.
  Contextual V-JEPA output tokens may already contain information from other
  frames; do not claim the selected representations contain no temporal signal.
- Existing targeted tests: `uv run --project MMTSFM pytest
  MMTSFM/tests/test_s2d_interleaved_raw.py
  MMTSFM/tests/test_s2d_component_ablations.py
  MMTSFM/tests/test_a44_strided_anchors.py
  MMTSFM/tests/test_a44_burst_ladder.py
  MMTSFM/tests/test_a45a_latent_depth.py -q`: 90 passed. They test grouping and
  pinning, not later-frame survival at production budgets.

## Answer

Implement A46 with a fixed 20-token budget per two-latent anchor: rank spatial
cells by pairwise cosine change and retain both endpoints of the ten selected
trajectories. This is `paired_novelty`; it is opt-in, so S2D and historical S2E
remain unchanged controls. A46's tests require both frame indices, nonzero
gradients through the later latent, and loud rejection of non-pair geometry.
The first experiment remains selector recovery at 100 total tokens; keep=490 is
deferred as a separate capacity control.
