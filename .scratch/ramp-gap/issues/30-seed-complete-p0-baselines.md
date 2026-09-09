# 30 — Seed-complete the P0 baselines and settle the Chronos-2 zero-shot row

Type: task
Status: open

## Question

`protocol.md` / `baselines.md` §4.5 require ≥3 seeds for every trained model. On disk:

- PatchTST (P0): seed 42 only → run 43, 44.
- Chronos-2 FT: seeds 42, 43 → run 44.
- Chronos-2 zero-shot: `ablations.md` A00 says **still missing** (only `chronos2_oracle` and
  `chronos2_oracle_ft` exist, an oracle-covariate tier), yet `report/report.typ` prints a
  "Chronos-2, zero-shot 0.4737" row. Determine which JSON that number came from. If it is the
  oracle run, either relabel the row honestly or run A00 (the untuned backbone, the natural
  floor under every MMTSFM arm).

Resolved when the three seeds exist for PatchTST and Chronos-2 FT, the A00 provenance is
written here, and `ablations.md` §2.6 A00 is updated.

Context: `paper-readiness-audit.md` §2.2.
