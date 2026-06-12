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
| 3 | `chronos2_zs` / `chronos2_ft` | Chronos-2 zero-shot / fine-tuned (MMTSFM source) | ✅ |
| 3 | `timesfm_zs` | TimesFM 2.5 zero-shot | ✅ |
| 3 | `tirex_zs` | TiRex (xLSTM) zero-shot | ✅ |
| 3 | `ttm_zs` / `ttm_ft` | TTM-R3 zero-shot / fine-tuned | — |
| 4 | `ts_rag` | TS-RAG: analog retrieval over frozen backbone, α tuned on val | via backbone |
| 4 | `cross_rag` | Cross-RAG (A08): clear-sky-aware keys, per-step α | via backbone |
| 4 | `cora` | CoRA-style covariate adapter on frozen backbone (zero-init residual) | via backbone |

Tier 3 needs `uv sync --group tier3` (transformers/einops for Chronos-2 via
`MMTSFM/src`, timesfm, tirex, granite-tsfm). Tier 4 wraps any registered
zero-shot backbone (default `chronos2_zs`); contract tests run them against
the dependency-free `persistence` backbone.

**Known TTM limitation:** TTM-R3 has no missing-value mask; short histories
are zero-padded to its fixed context length (P2 baseline, noted for the
paper's appendix).

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

## Protocol toolkit

- `common/stats.py` — Diebold–Mariano (HLN-corrected), paired block
  bootstrap (block = day), Holm–Bonferroni (§4.5). No scipy needed.
- `common/aggregate.py` — win rate, geometric-mean skill, average rank
  (§4.4, fev-bench conventions; never raw cross-dataset averaging).
- `common/controls.py` — §5 eval-time controls: `zero_cov`,
  `low_history_{4,8,12}` (mask-based, shape-preserving), plus the aligned
  `shuffle_along_axis` primitive for the A09/A10 frame controls.
- `common/runner.py` — ramp-subset (S6) thresholds + metrics, per-horizon
  NMAE(h) curves (§4.2), per-window loss sidecars (`*_losses.npz`) for
  significance testing, reproducibility manifest (§6.7).
- `scripts/run_suite.py` — S1–S5 + controls + A15 sweep as run_eval
  commands (dry-run by default; `--execute` to run).
- `scripts/significance.py` — DM + bootstrap + Holm over saved runs; only
  bold a result when `bold_ok` is true.
- `scripts/make_tables.py` — renders §7.1 headline + §4.4 aggregation
  tables from `results/`.
- `scripts/efficiency.py` — §4.6 params / latency / VRAM table.

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

Tier 5/6 (Time-VLM, UniCast, SUNSET, CrossViVit, Solar-VLM — see
`solar_vlm/`) and MEMTS (T4, P2) follow per the execution order in
BASELINE_COMPARISON.md §8.
