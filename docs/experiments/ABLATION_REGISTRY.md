# Ablation Registry

| ID | Hypothesis | Config | Branch | Status | Result |
|----|------------|--------|--------|--------|--------|
| A00 | Chronos-2 zero-shot baseline | `experiment=chronos2_zs` |  -  | TODO |  -  |
| A01 | Late fusion (Stage 2a) | `model.fusion_mode=late_fusion` | MMTSFM | DONE (MMTSFM) | Sanity OK 2026-05-03 |
| A02 | Interleaved fusion (Stage 2b) | `model.fusion_mode=interleaved` |  -  | IN PROGRESS |  -  |
| A03 | Grassmann vs self-attention | `model.temporal_mixer=...` |  -  | TODO |  -  |
| A04 | Visual window 3h vs 6h vs 12h | `data.vis_refinement_hours=...` |  -  | TODO |  -  |
| A05 | Cross-plant held-out — **not an ablation**: this *is* the S2 evaluation protocol (BASELINE_COMPARISON §4.1); kept for ID continuity | `data.split=cross_plant` |  -  | FOLDED into protocol |  -  |
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
| HEADLINE | Proposed architecture (V-JEPA 2.1 + interleaved + Grassmann) is the reported model | `model=vision_chronos2_headline` | fix/generalization-merge | CODE DONE; numbers [cluster, deferred] |  -  |
| W4 | Cross-plant group batching (num_entities>1, train-time) activates GroupSelfAttention and improves zero-shot cross-plant skill | `data.num_entities=4` (train only; val/test forced 1) | fix/generalization-merge | CODE DONE; numbers [cluster, deferred] | CPU: group disjointness + cross-entity gradient verified |
| W5 | Bounding the visual window (visual_window_hours) + true frame Δt to the summarizer improves sub-hourly skill | `data.visual_window_hours=6.0` | fix/generalization-merge | CODE DONE; numbers [cluster, deferred] | CPU: recency bound + Δt plumbing verified |
| W6 | Visual marginal gain (NMAE/NRMSE vision-on vs vision-off) confirms the visual stream is used | `model.compute_marginal_gain=true` (eval) | fix/generalization-merge | CODE DONE; sign/magnitude [cluster, deferred] | CPU: dual-pass + Δ computed; forced-off == masked forward |
| W7 | n_visual_context_steps is derived from cadence and bounded by T_ctx (no silent clamp) | derived; asserted at model init | fix/generalization-merge | DONE | CPU: derivation + init assert verified |

## How to register

1. Add row above before running
2. Create `configs/ablation/<id>.yaml`
3. Branch `exp/<id>-<short-name>`
4. Update Status ? DONE with W&B run ID and key metric
