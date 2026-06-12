# Naming Conventions

## Package

`pvtsfm`  -  PV Temporal Spatiotemporal Foundation Model

## Directory layout

```
src/pvtsfm/
  types.py                    # Shared TypedDicts, Batch schema
  train.py                    # Hydra CLI: training only
  eval.py                     # Hydra CLI: evaluation only
  ablate.py                   # Hydra CLI: multirun ablations
  data/
    pv_dataset.py             # PVTSFMDataset (one class)
    pv_datamodule.py          # Lightning DataModule (one class)
    batch_builder.py          # Sliding window logic (functions)
  models/
    base_model.py             # PVTSFMBaseModel abstract
    chronos2/
      patch_embed.py
      encoder_block.py
      grassmann_mixer.py
      group_attention.py
      quantile_head.py
    vision/
      sensor_projection.py
      vjepa_encoder.py
      latent_summarizer.py
      cross_modal_adapter.py
    fusion/
      sequence_builder.py     # Interleaved token layout
      fusion_mode.py          # Enum + factory
    lightning/
      lightning_pvtsfm.py     # Main Lightning module
  eval/
    metrics.py
    benchmark_runner.py
  baselines/
    chronos2_zero_shot.py
    solar_vlm_wrapper.py
    ts_rag_wrapper.py
```

## File rules

| Rule | Example |
|------|---------|
| Snake_case modules | `grassmann_mixer.py` |
| Class name = PascalCase of file | `GrassmannMixer` in `grassmann_mixer.py` |
| Config mirrors module path | `configs/model/vision_chronos2_grassmann.yaml` |
| Test mirrors source | `tests/models/fusion/test_sequence_builder.py` |
| Script = verb | `scripts/port_from_mmtsfm.py` |

## Git branches

| Prefix | Use |
|--------|-----|
| `exp/<name>` | Experiments |
| `feat/<name>` | Features |
| `fix/<name>` | Bugfixes |
| `port/<module>` | MMTSFM port |

## Hydra experiment names

`{ablation}_{variant}_{date}`  -  e.g. `fusion_interleaved_crossplant_2026-06-12`
