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
| 4 | _ts_rag_ | TS-RAG — **cluster-only, vendored original code** (`tier4/vendor/`), not a registry baseline | via backbone |
| 4 | _cross_rag_ | Cross-RAG — **cluster-only, vendored original code** (`tier4/vendor/`), not a registry baseline | via backbone |
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
- `scripts/run_suite.py` — S1–S5 + controls as run_eval commands
  (dry-run by default; `--execute` to run).
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

## Cluster execution (SLURM)

Tiers 0-2 run on a laptop; Tiers 3-4 are GPU-bound. Submit from `baselines/`:

```bash
sbatch scripts/slurm_baselines.sh                          # T3 ZS + T4 trained, S2
sbatch --export=ALL,STAGE=zs scripts/slurm_baselines.sh    # zero-shot only
sbatch --export=ALL,STAGE=lopo scripts/slurm_baselines.sh  # goes_pvdaq LOPO (§4.1)
```

Compute nodes are offline — run the prep on the **login node** first so all HF
weights are cached and the uk_pv CSVs exported:

```bash
bash scripts/login_node_prep.sh            # caches HF models + exports + input contract check
```

The *original* vendored TS-RAG / Cross-RAG (separate numpy-1.25 conda env, not
`run_eval`) have their own offline-guarded runner — `scripts/slurm_rag_original.sh`:

```bash
# baseline-contract gate only (offline, no model):
sbatch --export=ALL,METHOD=ts_rag,REGIME=orig,CONTRACT_CHECK=1,CONDA_ENV=tsrag,\
UKPV_CSV_DIR=…,BASE_CKPT=…,MIXER_CKPT=… scripts/slurm_rag_original.sh
# full run: drop CONTRACT_CHECK=1
```

See `docs/experiments/TIER4_RAG_INTEGRATION.md` for the full recipe.

### Leonardo (ISCRA-C) readiness checklist

Before `sbatch`, on the **login node** (internet), in order:

1. `git clone` the repo (brings `MMTSFM/src` for Chronos-2 and `configs/splits.json`).
2. Stage the data: copy `all_curated.parquet` to
   `$TEAM_SCRATCH/data/numerical/all_curated.parquet` (default
   `TEAM_SCRATCH=/leonardo_scratch/fast/IscrC_MTSFM`; override the env if your
   ISCRA-C project scratch differs).
3. `uv sync --group tier3` (resolves the lock for linux; needs network).
4. `bash scripts/login_node_prep.sh` — caches HF weights (chronos-2, timesfm,
   tirex, ttm; chronos-t5-base + chronos-bolt for RAG) and exports the uk_pv CSVs.
5. Confirm the SLURM account: scripts default to `--account=IscrC_MTSFM`; if your
   ISCRA-C grant differs, submit with `sbatch --account=<your_account> …`.

Then on compute nodes (offline):

```bash
sbatch scripts/slurm_baselines.sh                          # T3 ZS + T4 (cora) trained, S2
sbatch --export=ALL,STAGE=lopo scripts/slurm_baselines.sh  # goes_pvdaq LOPO
```

QOS: scripts use `normal` (≤24 h). `boost_qos_dbg` (30 min cap) only for a smoke
test via `sbatch --qos=boost_qos_dbg --time=00:30:00 …`.

**Still manual for the RAG originals** (not auto-prepared): create the upstream
conda env (`TIER4_RAG_INTEGRATION.md §1`) and download the released ARM /
cross-attn checkpoints (Google Drive / HF). Everything else above is ready.

## Cluster-only vendored tiers (originals, own env)

Tiers 5–6 run the authors' **original** code, vendored under `tier5/vendor/` and
`tier6/vendor/` (own conda env per model; heavy stacks conflict with this venv).
Not in-process registry baselines — `make_tables.py` ingests their results by file
stem via `scripts/import_predictions.py`.

- **Tier 5** (generic multimodal TS): Time-VLM, VisionTS++ (numerical track, runnable),
  UniCast, Aurora (multimodal track, gated). See `docs/experiments/TIER5_INTEGRATION.md`,
  `scripts/slurm_{time_vlm,visionts_pp,unicast,aurora}.sh`.
- **Tier 6** (PV-specialized multimodal, domain SOTA): CrossViViT (`tier6/vendor/crossvivit`,
  MIT) + SUNSET (`tier6/vendor/sunset`, MIT) — both multimodal track (real frames),
  gated on that data; Solar-VLM is the third P0, already ported under `solar_vlm/`.
  See `docs/experiments/TIER6_INTEGRATION.md`,
  `scripts/slurm_{crossvivit,sunset}.sh`.

## Not in this package (other tiers)

MEMTS (T4, P2) and the Tier-6 P1/P2 cite-only rows (SPIRIT, PV-VLM, M3S-Net,
MDCTL-MCI) follow per the execution order in BASELINE_COMPARISON.md §8.
