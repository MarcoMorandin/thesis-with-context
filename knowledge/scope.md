# Scope — research question, hypotheses, venue

**Canonical for**: what this project is and is not trying to answer. Read this before
proposing any experiment. Design → [architecture.md](architecture.md) ·
Evaluation → [protocol.md](protocol.md) · Run tracking → [ablations.md](ablations.md).

## Research question (AI framing)

> Can a frozen multimodal foundation model stack (TS FM + vision FM) achieve **cross-plant PV power forecasting** on **disjoint test plants** by **deep token-level fusion** rather than late fusion or domain-specific architectures?

## In scope

- Multimodal fusion mechanisms (interleaving, RAG, memory adapters)
- Cross-plant / zero-shot generalization protocols on disjoint test sets
- Foundation model adaptation (Chronos-2, V-JEPA 2.1, TS-RAG, Cross-RAG, TS-Memory, MEMTS)
- Probabilistic forecasting (quantile loss, CRPS)
- Systematic ablations on fusion, visual window, and horizon length
- Comparison with Solar-VLM, SPIRIT, Chronos-2 family

## Out of scope

- Dataset construction / ETL (dataset of record: `/leonardo_scratch/fast/IscrC_MTSFM/data_v2/` — schema and provenance in [dataset.md](dataset.md) §1.0, read-only)
- CSI / irradiance-only prediction (stay on **power** unless ablating as baseline)
- Pre-2025 methods as primary contributions
- Energy-market or grid operations research

## Primary metrics

| Metric | Split | Priority |
|--------|-------|----------|
| MAE / RMSE (power) | cross_plant | P0 |
| **Ramp NMAE / ramp NRMSE** (top-decile \|Δy\| subset, [protocol.md §5](protocol.md)) | cross_plant | **P0** |
| CRPS / pinball loss | cross_plant | P0 |
| Generalization score on disjoint plants | cross_plant | P0 |
| TEMPLATE transfer scores | cross_plant | P1 |

**Ramp is P0, not a supporting lens** (promoted 2026-08-25). Correctly forecasting sudden
changes in output — a cloud front crossing the array — is the quantity that decides whether a
forecast is usable for grid integration, and it is the one place a visual channel should pay
off. It sits beside generalization; neither is subordinate to the other. Where the two
disagree, say so rather than silently reporting the flattering one.

Two consequences. Every table that reports a skill score reports ramp beside it. And no
result is a "win" on aggregate error alone if the ramp column moved the other way.

## Hypothesis ladder (ablation order)

1. **H0**: Chronos-2 zero-shot vs custom PV architecture (establish FM baseline)
2. **H1**: Late-fusion V-JEPA adapter improves over TS-only (Stage 2a) — **falsified for late
   fusion**: s2a SS 0.5086 vs s1 0.5087, indistinguishable. Vision *does* contribute inside
   interleaved fusion (forced vision-off on s2b: ΔNMAE 0.0020, 2.7 % rel., positive on 14/14
   plants), so the honest verdict is split by fusion mode, not a flat yes/no. Seeded
   confirmation pending. *(An earlier revision read "validated in MMTSFM" — that predated the
   curriculum completing.)*
3. **H2**: Selective temporal interleaving > late fusion (Stage 2b) — **supported at n=1**:
   s2b SS 0.5284 vs s2a 0.5086. Confounded by the interleaved path's attention-mask handling
   and not yet seeded; both are live tickets in `.scratch/ramp-gap/`.
4. **H3**: Cross-plant generalization enables zero-shot prediction on completely disjoint plants (relying on spatial/temporal features learned from training plants)
5. **H4**: TS-RAG / Cross-RAG on frozen Chronos-2 closes gap to full fine-tune

## Target venue framing

Contribution is **multimodal foundation model fusion + cross-plant generalization**, not PV engineering. Position against: Solar-VLM (2026), Time-VLM (2025), TS-RAG (2025), Cross-RAG (2026), TEMPLATE (NeurIPS 2025).
