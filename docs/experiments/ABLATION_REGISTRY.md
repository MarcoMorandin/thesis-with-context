# Ablation Registry

| ID | Hypothesis | Config | Branch | Status | Result |
|----|------------|--------|--------|--------|--------|
| A00 | Chronos-2 zero-shot baseline | `experiment=chronos2_zs` |  -  | TODO |  -  |
| A01 | Late fusion (Stage 2a) | `model.fusion_mode=late_fusion` | MMTSFM | DONE (MMTSFM) | Sanity OK 2026-05-03 |
| A02 | Interleaved fusion (Stage 2b) | `model.fusion_mode=interleaved` |  -  | IN PROGRESS |  -  |
| A03 | Grassmann vs self-attention | `model.temporal_mixer=...` |  -  | TODO |  -  |
| A04 | Visual window 3h vs 6h vs 12h | `data.vis_refinement_hours=...` |  -  | TODO |  -  |
| A05 | Cross-plant held-out | `data.split=cross_plant` |  -  | TODO |  -  |
| A06 | N/A (Few-shot protocol) | N/A |  -  | DEPRECATED | Removed in favor of disjoint cross-plant test sets |
| A07 | TS-RAG on frozen Chronos-2 | `baselines/ts_rag` |  -  | TODO |  -  |
| A08 | Cross-RAG vs TS-RAG | compare wrappers |  -  | TODO |  -  |
| A09 | Shuffled-frames control (vision actually read) | `eval.control=shuffle_frames` |  -  | TODO |  -  |
| A10 | Mismatched-plant frames control (spatial grounding) | `eval.control=swap_plant_frames` |  -  | TODO |  -  |
| A11 | Vision-only upper bound | `model.inputs=vision_only` |  -  | TODO |  -  |
| A12 | Modality-contribution grid (TS / TS+cov / TS+vis / full) | `model.inputs=...` (4 runs) |  -  | TODO |  -  |
| A13 | Visual token budget sweep | `model.vision.num_tokens=...` |  -  | TODO |  -  |
| A14 | Frozen vs partial-unfreeze backbone | `model.unfreeze=...` |  -  | TODO |  -  |
| A15 | RAG datastore size / top-k sweep | `baselines/ts_rag k=..., store=...` |  -  | TODO |  -  |

## How to register

1. Add row above before running
2. Create `configs/ablation/<id>.yaml`
3. Branch `exp/<id>-<short-name>`
4. Update Status ? DONE with W&B run ID and key metric
