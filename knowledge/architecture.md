# Architecture Overview

Inherited from MMTSFM, narrowed to PV. Full detail: `MMTSFM/proposal.md`.

## Data flow

```text
PV power Y, covariates X, sky/satellite frames V (recent window)
    ?
    ?? Chronos-2 path ??? patch tokens [T_ctx + T_fut, d]
    ?
    ?? V-JEPA path ??? SensorProjection ??? V-JEPA 2.1 ??? LatentSummarizer
                              ?
                              ? visual tokens [n_vis, d]
    ?
    ? SequenceBuilder (interleaved in refinement window only)
    ?
    ? Chronos2Encoder × L
         1. GrassmannMixer (temporal, O(L))
         2. GroupAttention (cross-plant / cross-entity)
         3. FFN
    ?
    ? QuantileHead ??? ? [H, Q]
```

## Key design choices (AI contributions)

| Component | AI rationale |
|-----------|--------------|
| Decoupled resolution | Avoid destructive resampling; preserve FM native scales |
| Selective interleaving | Deep fusion with ~1 - 2% token overhead |
| Grassmann mixing | O(L) context for year-long history on one GPU |
| Frozen FMs + adapters | Sample-efficient; matches TS-RAG / MEMTS literature |
| Cross-plant zero-shot | Generalization on disjoint held-out plants using small history |

## Training curriculum

| Stage | What trains | Mode |
|-------|---------------|------|
| 1 | Grassmann params | TS-only warmup |
| 2a | Vision adapter + summarizer | Late fusion |
| 2b | + interleaving | Cross-modal Grassmann |
| 3 | Joint (optional unfreeze) | Full PV corpus |

## Port status from MMTSFM

The complete prior codebase is physically imported under the [MMTSFM/](file:///Users/marcomorandin/Desktop/thesis-with-context/MMTSFM/) directory. Components will be extracted and ported into `src/pvtsfm/` according to this map:

| Module | MMTSFM path | PVTSFM path | Status |
|--------|-------------|-------------|--------|
| Grassmann mixer | `MMTSFM/src/mmtsfm/models/chronos2/grassmann.py` | `src/pvtsfm/models/chronos2/grassmann_mixer.py` | TODO |
| V-JEPA encoder | `MMTSFM/src/mmtsfm/models/vision/vidtok_encoder.py` | `src/pvtsfm/models/vision/vjepa_encoder.py` | TODO |
| Interleaving | `MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py` | `src/pvtsfm/models/fusion/sequence_builder.py` | TODO |
| Dataset | `MMTSFM/src/mmtsfm/data/dataset.py` | `src/pvtsfm/data/pv_dataset.py` | TODO |

Use `scripts/port_from_mmtsfm.py` for guided porting (one module at a time).
