# Tier-4 RAG — running the *original* TS-RAG / Cross-RAG code

**Goal.** Run the authors' **original code** for the Tier-4 RAG numbers, per
BASELINE_COMPARISON.md §1 (Tier 4) and the §3 fairness contract. The in-repo α-mix
`baselines/tier4/rag.py` has been **removed**: `ts_rag` / `cross_rag` are no longer
registry baselines and run only from the vendored upstream code on the cluster.
CoRA (`tier4/cora.py`) remains the only in-process Tier-4 baseline. The original repos
are vendored unmodified under `baselines/tier4/vendor/` (`VENDOR_NOTICE.md` = SHAs +
licensing).

**Offline.** Compute nodes have no internet. Run `scripts/login_node_prep.sh` on the
login node first (caches HF models, exports the uk_pv CSVs, runs the input contract
check); the compute job (`scripts/slurm_rag_original.sh`) then runs fully offline with
`HF_HUB_OFFLINE=1` and fails loud if a cache is missing. `CONTRACT_CHECK=1` runs only the
baseline-contract gate (`tier4/vendor/contract_check.py`) and skips the heavy run.

Per the agreed plan we report **two rows per method**:
- **`*_orig` (faithful)** — Chronos-Bolt backbone, the authors' native **ctx-512 / pred-64**
  regime, their **pretrained** ARM / cross-attention checkpoints.
- **`*_proto` (protocol-conformant)** — same code re-run at **T=24 / H=12** (our protocol),
  mixers **re-pretrained** on uk_pv train plants (the released checkpoints are 512/64 and
  do not transfer to 24/12).

> **This is cluster-only work.** It will not run on the MacBook (needs CUDA, `faiss-gpu`,
> multi-GB checkpoints) and **cannot share the `baselines/` virtualenv** — the upstream
> pins (`numpy==1.25`, `chronos-forecasting==1.5.1`, `autogluon==1.3.0`, `faiss`) conflict
> with our `numpy>=2`. Create a dedicated env.

---

## 1. Environment (separate from `baselines/`)

```bash
conda create -n tsrag python=3.9 && conda activate tsrag
pip install -r baselines/tier4/vendor/ts_rag/requirements.txt   # chronos-forecasting,
                                                                # autogluon, faiss-gpu, numpy 1.25, wandb
pip install gluonts scikit-learn                                # imported by dataset.py / zeroshot.py
# Cross-RAG additionally:
pip install tabpfn
```

faiss: use `faiss-gpu` on the cluster; for a CPU dry-run use `faiss-cpu` (swap the pin).

## 2. Artifacts to download (NOT in git — see vendor `.gitignore`)

| Artifact | Source | Used by |
|---|---|---|
| Chronos-Bolt base weights (`checkpoints/base/`, `autogluon_model.pth`) | HF `amazon/chronos-bolt-base` | both, `--pretrained_model_path` |
| chronos-t5-base (embedding model) | HF `amazon/chronos-t5-base` | retrieval embeddings (`zeroshot.py` L142) |
| TS-RAG ARM checkpoint (`checkpoints/chronos-bolt/best.pth`) | Google Drive / HF `nkh/TS-RAG-Data` | `ts_rag_orig` only |
| Cross-RAG cross-attn checkpoint | Google Drive (Cross-RAG README) | `cross_rag_orig` only |

The `*_proto` rows do **not** use the released checkpoints — they are produced by
`pretrain.py` on our data (step 5).

## 3. uk_pv → upstream data format

Their `custom_retrieve` loader (`data_provider/data_loader.py::Dataset_Custom_retrieve`)
reads an **Informer-style CSV**: first column `date`, then one column per series, last
column the target `OT`; it splits 70/20/10 internally by row order. Retrieval keys/values
come from a pre-built database keyed `{database_name}_{frequency}_{lookback}.pkl`.

**Fairness mapping (critical).** The upstream `--mode only_self_train` retrieves from the
*query series' own* history → for cross-plant that would leak **test-plant** windows into
the datastore, violating BASELINE_COMPARISON §3 ("datastore = train-plant windows only").
We must instead:
1. Export **train-plant** uk_pv series → the retrieval database (`retrieve.py::do_retrieve`
   / `load_database`), and
2. Export **test-plant** uk_pv series → the query CSV (`OT` = the plant being forecast),
   run with the train-only database (`--mode all_vars` pointed at the train DB, or a
   custom mode), never `only_self` on the test series.

The bridge is implemented in `baselines/tier4/vendor/export_ukpv.py` (pandas-only, runs
in the baselines venv; verified on uk_pv). It reads `configs/splits.json` +
`all_curated.parquet` and writes, to `--out`:
- `uk_pv_train.csv` — dense 30-min grid, columns = train `site_id`s, `date` + `OT` (=first
  train plant); the retrieval-datastore source (train plants only, §3);
- `uk_pv_test_<site>.csv` per test plant (`date` + `OT`=that site);
- `capacity.json` (`installed_power_w` per site) for de-normalising predictions (step 6);
- `manifest.json`.

Gaps (night/outage) are filled with 0.0 to give the upstream StandardScaler a dense grid.
`tier4/vendor/contract_check.py --inputs <dir>` validates the result (date column, `OT`
present, uniform 30-min grid, finite, in [0,1]) — wired as the SLURM preflight.

> **Note (retrieval-DB fairness).** The remaining open item is wiring `do_retrieve` so the
> datastore is built from `uk_pv_train.csv` only (never `only_self` on the test series) —
> validate on the cluster before trusting the headline numbers.

Keep cadence honest: uk_pv is 30-min ⇒ `--metadata_frequency half_hourly` (maps to
seasonality 48 in `zeroshot.py::SEASONALITY_MAP`).

> **SLURM wrapper.** `baselines/scripts/slurm_rag_original.sh` drives §4 and §5
> end-to-end (conda env activation, optional re-pretrain, per-test-plant zero-shot)
> with fail-loud prerequisite guards. Submit e.g.
> `sbatch --export=ALL,METHOD=ts_rag,REGIME=orig,CONDA_ENV=tsrag,UKPV_CSV_DIR=…,BASE_CKPT=…,MIXER_CKPT=… scripts/slurm_rag_original.sh`.
> The raw commands below document what it runs.

## 4. Faithful rows (`ts_rag_orig`, `cross_rag_orig`) — native 512/64

From `baselines/tier4/vendor/ts_rag/TS-RAG/` (resp. `cross_rag/cross-rag/`):

```bash
python zeroshot.py \
  --root_path <ukpv_csv_dir> --data_path uk_pv_test_<site>.csv --data custom_retrieve \
  --model ChronosBoltRetrieve --augment_mode moe \
  --seq_len 512 --pred_len 64 --lookback_length 512 --top_k 10 \
  --pretrained_model_path <chronos_bolt_base_dir> \
  --checkpoint_model_path <ARM_checkpoint>/best.pth \
  --retrieval_database_dir <train_db_dir> \
  --metadata_frequency half_hourly --metadata_database_name uk_pv \
  --embedding_model_type chronos --dimension 768 --batch_size 256 --gpu_loc 0
```

Cross-RAG: `script/Cross-RAG-zeroshot.sh` with `RETRIEVE_SPACE=X`, `top_k=15`, same backbone.
Loop over the 14 test plants; ≥3 seeds where the data order is stochastic.

> Caveat for the table: 512/64 is **not** input-parity with the other tiers (T=24/H=12).
> Mark the `*_orig` rows as "native-regime, Chronos-Bolt backbone"; they answer
> "does the published method, as published, beat us?" not the parity question.

## 5. Protocol-conformant rows (`*_proto`) — T=24 / H=12

The released ARM/cross-attn checkpoints are 512/64 and will not transfer. Re-pretrain at
our regime (`pretrain.py`), then zero-shot at 24/12:

```bash
# pretrain the mixer on uk_pv train plants, frozen Chronos-Bolt
python pretrain.py --context_length 24 --prediction_length 12 \
  --retrieve_lookback_length 24 --top_k 10 --augment_mode moe \
  --retrieval_database_path <ukpv_train_pairs>.parquet \
  --data_path <ukpv_pretrain_pairs_dir> --freeze_chronos_bolt \
  --train_steps 10000 --batch_size 256 --learning_rate 3e-5
# then zeroshot.py as in §4 but --seq_len 24 --pred_len 12 --lookback_length 24
#   --checkpoint_model_path <the_new_24_12_checkpoint>
```

These rows ARE input-parity with Tiers 0-3 and belong in the headline table.

## 6. Metrics back into our pipeline

Upstream `utils/tools.py::test_retrieve` reports **MSE/MAE on StandardScaler-normalised**
series — not our capacity-normalised NMAE/NRMSE/SS. To compare fairly, dump per-window
predictions and re-score with ours:

- Add a tiny patch (kept as `baselines/tier4/vendor/patches/dump_predictions.diff`, applied
  at run time — do **not** edit vendored source in place) so `test_retrieve` also writes
  `pred`, `true`, and the per-series `scaler` (mean/std) to an `.npz`.
- Un-StandardScale → multiply by `installed_power_w/installed_power_w` is identity since we
  exported `OT` already capacity-normalised (`norm_power`); so just invert the StandardScaler
  to recover `norm_power`, then feed `(pred, true, mask·daylight)` to
  `common/runner.py::evaluate_model`'s metric core to produce NMAE/NRMSE/SS/CRPS and the
  per-window `*_losses.npz` the DM/bootstrap test consumes.
- Once the `.npz` carries `pred`/`true`, the shared glue
  **`scripts/import_predictions.py`** turns it into our result JSON (same as Tier 5):
  `--model ts_rag_orig --glob '…/*_pred.npz' --reference results/smart_persistence_s2_ukpv.json`
  → `results/ts_rag_orig_s2_ukpv.json`, which `summarize_ukpv.py` / `make_tables.py` pick up
  (rows `ts_rag_orig` / `ts_rag_proto` / `cross_rag_orig` / `cross_rag_proto`). Same two
  caveats as Tier 5 apply (proxy daylight mask; native eval windows ⇒ no DM sidecar).

Only the `*_proto` rows are input-parity; keep `*_orig` in a clearly-labelled companion
block (different backbone + regime), per §4.1.1 cadence rules.

## 7. Status

- [x] Original code vendored (`vendor/ts_rag`, `vendor/cross_rag`) + provenance/licensing.
- [x] SLURM wrapper `baselines/scripts/slurm_rag_original.sh` (orig + proto, offline-guarded).
- [x] `export_ukpv.py` data bridge (§3) — verified on uk_pv (dense `date`+`OT` CSVs in [0,1]).
- [x] `contract_check.py` + `CONTRACT_CHECK=1` gate — input + prediction baseline-contract checks.
- [x] `login_node_prep.sh` — caches HF models + exports CSVs so compute nodes stay offline.
- [ ] `dump_predictions.diff` patch (§6) — emit `*_pred.npz` for the output contract check + metric import.
- [ ] Cluster env + released ARM/cross-attn checkpoint download (manual, off-repo).
- [ ] Faithful + proto runs over the 14 test plants, ≥3 seeds.
- [ ] Results imported, `summarize_ukpv.py` regenerated.

There is no longer an in-repo α-mix fallback: `baselines/tier4/rag.py` was removed and
`ts_rag` / `cross_rag` are **no longer registry baselines** — TS-RAG / Cross-RAG run
exclusively from the vendored original code on the cluster. CoRA (`tier4/cora.py`) remains
the only in-process Tier-4 baseline (no vendored upstream).
