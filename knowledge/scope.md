# Research Scope  -  PVTSFM

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

- Dataset construction / ETL (dataset of record: `/leonardo_scratch/fast/IscrC_MTSFM/data/` = `dataset_all.parquet` + `images_all.h5`, read-only)
- CSI / irradiance-only prediction (stay on **power** unless ablating as baseline)
- Pre-2025 methods as primary contributions
- Energy-market or grid operations research

## Primary metrics

| Metric | Split | Priority |
|--------|-------|----------|
| MAE / RMSE (power) | cross_plant | P0 |
| CRPS / pinball loss | cross_plant | P0 |
| Generalization score on disjoint plants | cross_plant | P0 |
| TEMPLATE transfer scores | cross_plant | P1 |

## Hypothesis ladder (ablation order)

1. **H0**: Chronos-2 zero-shot vs custom PV architecture (establish FM baseline)
2. **H1**: Late-fusion V-JEPA adapter improves over TS-only (Stage 2a - validated in MMTSFM)
3. **H2**: Selective temporal interleaving > late fusion (Stage 2b - current focus)
4. **H3**: Cross-plant generalization enables zero-shot prediction on completely disjoint plants (relying on spatial/temporal features learned from training plants)
5. **H4**: TS-RAG / Cross-RAG on frozen Chronos-2 closes gap to full fine-tune

## Target venue framing

Contribution is **multimodal foundation model fusion + cross-plant generalization**, not PV engineering. Position against: Solar-VLM (2026), Time-VLM (2025), TS-RAG (2025), Cross-RAG (2026), TEMPLATE (NeurIPS 2025).
