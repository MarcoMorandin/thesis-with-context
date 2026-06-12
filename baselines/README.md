# PV Baselines — Tiers 0–2

Implements the Tier 0–2 baseline suite from
[docs/experiments/BASELINE_COMPARISON.md](../docs/experiments/BASELINE_COMPARISON.md)
on the disjoint cross-plant protocol
([BASELINE_PROTOCOL.md](../docs/experiments/BASELINE_PROTOCOL.md)).

Data source: `/Volumes/SSD/standardized-dataset/numerical/all_curated.parquet`
(produced by `dataset_exploration/curate_dataset.py`; native 30-min `uk_pv`
and 15-min `goes_pvdaq` grids, capacity-normalized `norm_power` target).

## Implemented baselines

| Tier | Registry name | Model | Quantiles |
|---|---|---|---|
| 0 | `persistence` | Naive last-value | — |
| 0 | `smart_persistence` | Clearness-index persistence (Skill-Score reference) | — |
| 0 | `climatology_hourly` | Train-plant mean by (dataset, month, hour) | — |
| 0 | `seasonal_naive` | Same clock time yesterday | — |
| 1 | `lightgbm` | LightGBM, one model per quantile {0.1…0.9} | ✅ |
| 1 | `tabpfn` | TabPFN regressor (optional dep: `uv sync --group tabpfn`) | ✅ |
| 2 | `mlp` | Flattened-input MLP | — |
| 2 | `dlinear` | DLinear (Y only) | — |
| 2 | `patchtst` | PatchTST (channel-independent, RevIN) | — |
| 2 | `itransformer` | iTransformer (variates as tokens) | — |
| 2 | `tft` | TFT-lite (quantile-native) | ✅ |

**TFT-lite deviations from Lim et al. (2021):** no per-variable selection
networks and no static covariate encoders (the protocol has no static
features beyond capacity, which is already in the normalization); retains
GRNs, LSTM encoder/decoder over (history, future-known covariates),
encoder-decoder attention, and the 9-quantile pinball objective.

## Fairness contract (enforced in code)

- Disjoint plant splits, seeded, persisted to `configs/splits.json`;
  disjointness asserted at every load (`common/splits.py`).
- Capacity normalization only — covariates use *fixed physical scalings*,
  never per-plant or test-set statistics (`common/config.py::COV_SCALES`).
- Future covariates are restricted to deterministic solar geometry /
  calendar / clear-sky channels; observed weather is zeroed beyond the
  history window (`WindowDataset(future_cov="deterministic")`).
- No clear-sky physics inside models — Smart Persistence is the one exempt
  reference, per protocol.
- Metrics masked by `mask_future · daylight`, macro-averaged per plant.

## Usage

```bash
cd baselines
uv sync                          # core deps
uv run python -m common.splits   # generate + commit the plant split (once)
uv run python run_eval.py --model smart_persistence persistence
uv run python run_eval.py --model lightgbm dlinear patchtst itransformer mlp tft
uv run pytest                    # contract + metric tests (synthetic, no SSD needed)
```

Results land in `results/<model>.json` with a reproducibility manifest
(git SHA, config hash, seed, dataset version) per BASELINE_COMPARISON.md §6.7.

## Not in this package (other tiers)

Tier 3 (Chronos-2/TimesFM/TiRex zero-shot), Tier 4 (TS-RAG, CoRA), Tier 5/6
(Time-VLM, SUNSET, CrossViVit, Solar-VLM — see `solar_vlm/`) follow per the
execution order in BASELINE_COMPARISON.md §8.
