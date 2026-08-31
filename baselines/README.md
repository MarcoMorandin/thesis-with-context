# PV Baselines — Tiers 0–2

Implements the Tier 0–2 baseline suite from
[knowledge/baselines.md](../knowledge/baselines.md)
on the disjoint cross-plant protocol
([knowledge/protocol.md](../knowledge/protocol.md)).

Data source (dataset of record): `/leonardo_scratch/fast/IscrC_MTSFM/data/dataset_all.parquet`
(+ frames `images_all.h5`, canonical pointer `image_h5_index`). Native 30-min
`uk_pv` (100 plants, 128px gray) and 15-min `goes_pvdaq` (10 plants, 256px RGB)
grids, capacity-normalized `norm_power` target. Both datasets fully present
(knowledge/dataset.md §1.0).

> **Note:** point `common/config.py::DEFAULT_DATA_PATH` at
> `thesis-dataset/dataset_all.parquet` (and frame pointer `image_h5_index`).

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
| 2 | `itransformer_nf` | iTransformer from **neuralforecast**, trained on MMTSFM's protocol (not `run_eval.py`) — `tier2/train_itransformer_nf.py` | ✅ |
| 2 | `tft` | TFT-lite (quantile-native) | ✅ |
| 2 | `tabfm` | Google **TabFM v1.0.0** zero-shot tabular FM (optional dep: `uv sync --group tabfm`). *Filed in tier 2; not a supervised deep-TS model — it is the sibling of tier-1 `tabpfn`* | — |
| 2 | `tabfm_ens` | Same, `TabFMRegressor.ensemble()` (feature crosses + SVD feats + NNLS blend) | — |
| 3 | `chronos2_zs` / `chronos2_ft` | Chronos-2 zero-shot / fine-tuned (MMTSFM source) | ✅ |
| 3 | `timesfm_zs` | TimesFM 2.5 zero-shot | ✅ |
| 3 | `tirex_zs` | TiRex (xLSTM) zero-shot | ✅ |
| 3 | `ttm_zs` / `ttm_ft` | TTM-R3 zero-shot / fine-tuned | — |
| 4 | _ts_rag_ | TS-RAG — **cluster-only, vendored original code** (`tier4/vendor/`), not a registry baseline | via backbone |
| 4 | _cross_rag_ | Cross-RAG — **cluster-only, vendored original code** (`tier4/vendor/`), not a registry baseline | via backbone |
| 4 | `cora` | CoRA-style covariate adapter on frozen backbone (zero-init residual) | via backbone |

### `itransformer_nf` — the like-for-like control (`uv sync --group nf`)

Every other tier-2 row is trained by `run_eval.py`: stride-1 windows (~12x more
than MMTSFM sees), batch 256, lr 1e-3, up to 100 epochs, history-only
covariates. That budget, not the architecture, is what the in-repo
`itransformer` row buys. `itransformer_nf` removes it: the model comes from the
`neuralforecast` library and is trained on MMTSFM's own `PVRecordDataset`
windows with MMTSFM's recipe and scored by MMTSFM's `ProtocolEvaluator`, so the
gap against `mmtsfm_s2b_ukpv` is attributable to the model. It does not go
through `run_eval.py`:

```bash
uv sync --group nf                                   # login node (internet)
for s in 42 43 44; do
  sbatch --export=ALL,SEED=$s scripts/slurm_itransformer.sh
done
```

The parity list (windows, optimizer, schedule, precision, early stopping, loss
mask, Skill-Score reference) is in the script docstring and asserted by
`tests/test_itransformer_nf.py`. Trained with the library's own point loss
(default `MSE`, matching `thuml/iTransformer`'s own
`experiments/exp_long_term_forecasting.py::nn.MSELoss()`; `--loss mae` is also
available) rather than a quantile loss — NF's multivariate projector layout
and `MQLoss.domain_map` disagree for N > 1, so a point loss is the
library-native way to get iTransformer's covariates-as-tokens design without a
hand-decoded reshape. NMAE/NRMSE/Skill-Score are reported; CRPS, coverage and
ECE are N/A for this arm.

Fidelity vs. the paper (checked against `thuml/iTransformer` source directly):
encoder-only inverted embedding, per-variate RevIN normalization, full
self-attention over variate tokens, and supervising only the target variate
(the paper's own `features=="MS"` mode does the same) all match. One
deliberate extension: the paper's non-target input channels are always
historical-only at the target's own window (`layers/Embed.py`), while
`--future-cov all` shifts our covariate tokens to include real future
(known-NWP) weather — needed for PV's deployable-forecast setting, and should
be named as an adaptation rather than vanilla iTransformer usage when cited.

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

- `common/controls.py` — §5 eval-time controls: `zero_cov`,
  `low_history_{4,8,12}` (mask-based, shape-preserving), plus the aligned
  `shuffle_along_axis` primitive for the A09/A10 frame controls.
- `common/runner.py` — ramp-subset (S6) thresholds + metrics, per-horizon
  NMAE(h) curves (§4.2), per-window loss sidecars (`*_losses.npz`) for
  significance testing, reproducibility manifest (§6.7).

Results aggregation, table rendering, significance testing and plotting are
**out of scope for this repo**. `run_eval.py` and `scripts/import_predictions.py`
write per-model JSONs (plus `*_losses.npz` per-window sidecars) into `results/`;
consume them with your own tooling.

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
(git SHA, config hash, seed, dataset version) per knowledge/baselines.md §6.7.

## Cluster execution (SLURM)

Tiers 0-2 run on a laptop; Tiers 3-4 are GPU-bound. Submit from `baselines/`:

```bash
uv run python run_eval.py --model <models…>                             # T3 ZS + T4 trained, S2
uv run python run_eval.py --model <models…> --lopo-dataset goes_pvdaq   # goes_pvdaq LOPO (§4.1)
```

Compute nodes are offline — run the prep on the **login node** first so all HF
weights are cached and the uk_pv CSVs exported:

```bash
bash scripts/precache_login.sh             # caches HF models + exports + input contract check
```

**TabPFN (tier 1) is the exception to "tiers 0-2 run on a laptop."** It is an
optional dep group, its TabPFN-3 weights are license-gated, and it re-encodes the
whole in-context table on every `predict()`, so it needs a GPU and its own job:

```bash
# login node, once: syncs the `tabpfn` group + downloads the gated V3 ckpt.
# Needs TABPFN_TOKEN in baselines/.env (https://ux.priorlabs.ai).
STAGE=weights bash scripts/precache_login.sh

# compute node:
sbatch scripts/slurm_tabpfn.sh                                       # S2 cross-plant
sbatch --export=ALL,SCENARIO=s1 scripts/slurm_tabpfn.sh              # in-domain
sbatch --export=ALL,STAGE=lopo scripts/slurm_tabpfn.sh               # goes_pvdaq LOPO
sbatch --export=ALL,MAX_CONTEXT_ROWS=50000 scripts/slurm_tabpfn.sh   # smaller context
```

The job fails fast if the V3 ckpt is missing from `TABPFN_MODEL_CACHE_DIR`
(default `$TEAM_SCRATCH/weights/tabpfn`) — compute nodes cannot reach the
license gate, so a cold cache is fatal, not slow.

**TabFM (filed in tier 2) is the second tabular FM**, on the same flattened
`(Y, X_cov)` table as `lightgbm` and `tabpfn`. Running both makes the claim a
class-level one — *tabular FMs* close (or fail to close) the gap on PV — instead
of a statement about TabPFN alone. Same GPU-bound reasoning as TabPFN, so it too
gets its own job:

```bash
# login node, once: syncs the `tabfm` group (a GIT dep — TabFM is not on PyPI)
# and downloads google/tabfm-1.0.0-pytorch into $HF_HOME. No token needed.
STAGE=weights bash scripts/precache_login.sh

# compute node:
sbatch scripts/slurm_tabfm.sh                                    # tabfm,     S2
sbatch --export=ALL,CONFIG=ens scripts/slurm_tabfm.sh            # tabfm_ens, S2
sbatch --export=ALL,SCENARIO=s1 scripts/slurm_tabfm.sh           # in-domain
sbatch --export=ALL,STAGE=lopo scripts/slurm_tabfm.sh            # goes_pvdaq LOPO
sbatch --export=ALL,MAX_CONTEXT_ROWS=5000 scripts/slurm_tabfm.sh # smaller context
```

Three things differ from TabPFN and all three belong in any table that reports
the row:

* **Point predictions only.** Upstream has no regression quantile head, so
  `supports_quantiles = False`: NMAE / NRMSE / Skill-Score are reported, CRPS,
  coverage and ECE are **N/A** — same as `itransformer_nf` and `ttm_zs`. There is
  deliberately no zero-width-interval fallback, which would silently flatter the
  pinball loss.
* **Context default is 10 000 rows, not TabPFN's 100 000.** TabFM reads the
  context through `n_estimators` (32) separately-transformed views. Both
  `max_context_rows` and `n_estimators` are `--model-kwargs` knobs and are
  *reported protocol parameters* — quote them beside the number rather than
  retuning them to fit a wall clock.
* **Weights are ungated but non-commercial.** No `TABPFN_TOKEN` analogue; the
  code is Apache-2.0 while the checkpoint is `tabfm-non-commercial-v1.0`
  (research use only, no production or commercial use). The job fails fast if
  `$HF_HOME/hub/models--google--tabfm-1.0.0-pytorch` is missing.

The *original* vendored TS-RAG / Cross-RAG (separate numpy-1.25 conda env, not
`run_eval`) have their own offline-guarded runner — `scripts/slurm_rag.sh`:

```bash
# baseline-contract gate only (offline, no model):
sbatch --export=ALL,METHOD=ts_rag,REGIME=orig,CONTRACT_CHECK=1,CONDA_ENV=tsrag,\
UKPV_CSV_DIR=…,BASE_CKPT=…,MIXER_CKPT=… scripts/slurm_rag.sh
# full run: drop CONTRACT_CHECK=1
```

#### Tier 4 regimes: `ts_rag_orig` vs `ts_rag_proto`

Each vendored RAG baseline runs in two regimes, and both are reported:

| Regime | Windows | Fair under [knowledge/protocol.md](../knowledge/protocol.md)? | Why it exists |
|---|---|---|---|
| `ts_rag_orig` / `cross_rag_orig` | 512 history / 64 horizon (the paper's own setting) | ❌ — different window than the protocol | Reproduces the published number, so a weak protocol result cannot be dismissed as a broken port |
| `ts_rag_proto` / `cross_rag_proto` | 24 / 12 (protocol-aligned) | ✅ | The comparable number; this is the one that enters the leaderboard |

Select with `REGIME=orig|proto`. Only `*_proto` rows may be compared against
other tiers; `*_orig` rows are a port-sanity control and must be labelled as such
wherever they are reported. Vendored-source provenance:
[`tier4/vendor/VENDOR_NOTICE.md`](tier4/vendor/VENDOR_NOTICE.md).

### Leonardo (ISCRA-C) readiness checklist

Before `sbatch`, on the **login node** (internet), in order:

1. `git clone` the repo (brings `MMTSFM/src` for Chronos-2 and `configs/splits.json`).
2. Stage the data: copy `dataset_all.parquet` (+ `images_all.h5` for the
   multimodal tiers) to `$TEAM_SCRATCH/data/` (default
   `TEAM_SCRATCH=/leonardo_scratch/fast/IscrC_MTSFM`; override the env if your
   ISCRA-C project scratch differs).
3. `uv sync --group tier3` (resolves the lock for linux; needs network).
4. `bash scripts/precache_login.sh` — caches HF weights (chronos-2, timesfm,
   tirex, ttm; chronos-t5-base + chronos-bolt for RAG) and exports the uk_pv CSVs.
5. Confirm the SLURM account: scripts default to `--account=IscrC_MTSFM`; if your
   ISCRA-C grant differs, submit with `sbatch --account=<your_account> …`.

Then on compute nodes (offline):

```bash
uv run python run_eval.py --model <models…>                             # T3 ZS + T4 (cora) trained, S2
uv run python run_eval.py --model <models…> --lopo-dataset goes_pvdaq   # goes_pvdaq LOPO
```

QOS: scripts use `normal` (≤24 h). `boost_qos_dbg` (30 min cap) only for a smoke
test via `sbatch --qos=boost_qos_dbg --time=00:30:00 …`.

**Still manual for the RAG originals** (not auto-prepared): create the upstream
conda env (`TIER4_RAG_INTEGRATION.md §1`) and download the released ARM /
cross-attn checkpoints (Google Drive / HF). Everything else above is ready.

## Cluster-only vendored tiers (originals, own env)

Tiers 5–6 run the authors' **original** code, vendored under `tier5/vendor/` and
`tier6/vendor/` (own conda env per model; heavy stacks conflict with this venv).
Not in-process registry baselines — `scripts/import_predictions.py` folds their
`*_pred.npz` dumps into `results/` JSONs by file stem.

- **Tier 5** (generic multimodal TS): Time-VLM, VisionTS++ (numerical track, runnable),
  UniCast, Aurora (multimodal track, gated). See `tier5/vendor/VENDOR_NOTICE.md`,
  `scripts/slurm_{time_vlm,visionts_pp,unicast,aurora}.sh`.
- **Tier 6** (PV-specialized multimodal, domain SOTA): CrossViViT (`tier6/vendor/crossvivit`,
  MIT) + SUNSET (`tier6/vendor/sunset`, MIT) — run on the **uk_pv multimodal track**
  (curated `Y` + `images_all.h5` satellite frames, bridged by `tier6/uk_multimodal.py`);
  Solar-VLM is the third P0, already ported under `solar_vlm/`. See
  `tier6/vendor/VENDOR_NOTICE.md`, `scripts/slurm_{crossvivit,sunset}.sh`.

## Not in this package (other tiers)

MEMTS (T4, P2) and the Tier-6 P1/P2 cite-only rows (SPIRIT, PV-VLM, M3S-Net,
MDCTL-MCI) follow per the execution order in knowledge/baselines.md §8.
