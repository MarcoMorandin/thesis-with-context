# Audit what each historical anchor actually represents

Type: task
Label: wayfinder:task
Status: open
Assignee: unassigned
Parent: ../map.md
Blocked by: none

## Question

Establish representation, time and missingness invariants before interpreting
multi-anchor performance or treating A45a as an isolated depth control.

## Comments

2026-09-14 source observations and remaining checks:

- [Extraction](../../../MMTSFM/scripts/extract_video_embeddings.py:151) sends the
  entire clip to the encoder in one call. S2E's daily bursts are not independently
  encoded by this wrapper. Verify the cached encoder's temporal attention and
  position behavior before treating output slices as local historical observations.
  Test influence from later bursts on earlier output slices. Cross-burst influence
  before the forecast origin is not automatically forecast-target leakage.
- [A45a](../../../MMTSFM/configs/ablation/A45a.yaml) slices the newest two output
  latents from the existing S2D cache. Shape equivalence does not establish
  `slice(encode(8 frames)) == encode(4 frames)`, or equivalence to a slice encoded
  in S2E's 20-frame clip. It also inherits the keep=20 selector failure from
  [Choose a budget-safe selector](01-budget-safe-selection.md).
- A45a also retains single-anchor fractional positions, while S2E uses integer
  anchor positions. Its outcome alone cannot cleanly attribute the gap to anchor
  count. This qualifies the stronger interpretation currently written in
  [the registry](../../../knowledge/ablations.md:267).
- [Forward](../../../MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py:1310)
  zeros unavailable selected latents, subsequently adds modality/segment/type
  embeddings, and initializes visual attention slots as valid (line 1445).
  Quantify missing-anchor exposure and verify whether the intended contract is an
  explicit missing token or exclusion from attention. The coverage guard uses a
  batch maximum, so it is not a per-anchor/per-sample timestamp validator.
- Use train/validation cached samples and read-only raw data on Leonardo. Record
  encoder revision, cache ladder, actual timestamps, masks and slice provenance.
  No full re-extraction is justified before a small representation probe.

Completion: an auditable cache/anchor contract, small diagnostic measurements and
a documented list of representation differences that remain in each comparison.
