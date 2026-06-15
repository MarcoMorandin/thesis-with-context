# Tier-4 RAG — running the *original* TS-RAG / Cross-RAG code

**Goal.** Replace the lightweight α-mix in `baselines/tier4/rag.py` with the authors'
**original code** for the headline Tier-4 numbers, per BASELINE_COMPARISON.md §1 (Tier 4)
and the §3 fairness contract. The original repos are vendored unmodified under
`baselines/tier4/vendor/` (see `VENDOR_NOTICE.md` for SHAs + licensing).

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

Export contract for the bridge script (to add as `baselines/tier4/vendor/export_ukpv.py`,
pandas-only, runnable in our env):
- read `configs/splits.json` + `all_curated.parquet`;
- reindex each plant onto its native 30-min grid (reuse `common.windows.build_site_series`);
- write `uk_pv_train.csv` (columns = train `site_id`s, `date` index, `OT`=first train plant)
  and one `uk_pv_test_<site>.csv` per test plant (`OT`=that site);
- emit `uk_pv` capacity table (`installed_power_w` per site) so predictions can be
  de-normalised back for our metrics (step 6).

Keep cadence honest: uk_pv is 30-min ⇒ `--metadata_frequency half_hourly` (maps to
seasonality 48 in `zeroshot.py::SEASONALITY_MAP`).

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
- Register the resulting numbers under `ts_rag_orig`, `ts_rag_proto`, `cross_rag_orig`,
  `cross_rag_proto` in `results/` so `scripts/summarize_ukpv.py` and `make_tables.py` pick
  them up.

Only the `*_proto` rows are input-parity; keep `*_orig` in a clearly-labelled companion
block (different backbone + regime), per §4.1.1 cadence rules.

## 7. Status

- [x] Original code vendored (`vendor/ts_rag`, `vendor/cross_rag`) + provenance/licensing.
- [ ] `export_ukpv.py` data bridge (contract in §3).
- [ ] `dump_predictions.diff` patch (§6).
- [ ] Cluster env + checkpoint download.
- [ ] Faithful + proto runs over the 14 test plants, ≥3 seeds.
- [ ] Results imported, `summarize_ukpv.py` regenerated.

The existing `baselines/tier4/rag.py` α-mix stays as the dependency-free fallback and the
contract-test backbone; it is **not** the headline Tier-4 number once the above lands.
