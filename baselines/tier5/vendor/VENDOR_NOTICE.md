# Vendored Tier-5 baselines (generic multimodal TS) — provenance & licensing

Unmodified **code-only** copies of four upstream repos so Tier-5 runs the authors'
*original* implementations (knowledge/baselines.md §1, Tier 5), adapted to our
contract/dataset rather than reimplemented. Stripped on copy: `.git`, images,
notebooks, PDFs, bundled CSV/parquet datasets, checkpoints. No source edited.

| Vendor dir | Upstream | Commit SHA | License | Modality / track |
|---|---|---|---|---|
| `time_vlm/` | https://github.com/CityMind-Lab/ICML25-TimeVLM | `796e6ec963788657207ea2b5553740993ea3ea2b` | **none stated** ⚠️ (ICML 2025, arXiv:2502.04395) | TS→pseudo-image (+text) — **numerical track (uk_pv)** |
| `visionts_pp/` | https://github.com/HALF111/VisionTSpp | `484b2ea363b497217d0c3a078494c6af0251c275` | `LICENSE.txt` present (built on Salesforce `uni2ts`, Apache-2.0) | TS→image (vision MAE) — **numerical track (uk_pv)** |
| `unicast/` | https://github.com/adlnlp/UniCast | `a4af694615fabb9844a1a0f297aca148a3ab9db8` | **none stated** ⚠️ (arXiv:2508.11954) | real vision(CLIP/BLIP)+text soft-prompt into Chronos — **uk_pv multimodal track (images)** |
| `aurora/` | https://github.com/decisionintelligence/Aurora | `a247760abbc9d17a861bc365c032368d317815f2` | **none stated** ⚠️ (arXiv:2509.22295) | generative **TS + TEXT** TSFM (BERT-tokenized text; *no image input*) — **uk_pv track (text)** |

## Track split (what runs where)

- **Time-VLM, VisionTS++** render the time series itself as a pseudo-image — they need
  **no real satellite frames**, so they run on the **numerical uk_pv track** and match our
  `Y → ŷ` contract directly (like Tiers 0-4). These are the runnable Tier-5 rows today.
- **UniCast** soft-prompts **real images** (CLIP/BLIP) + a text string into Chronos →
  it needs real frames, available in `images_all.h5` (pointer `image_h5_index`). `tier5/uk_export.py
  --model unicast` emits its native layout (`inputs.pt`/`targets_<H>.pt`/`img/`) from
  the uk multimodal windows, so it runs on uk_pv now.
- **Aurora** is **TS + TEXT**, not images (`Aurora_Single_Dataset` reads a CSV + a
  JSON text list, BERT-tokenized — no image branch). uk images do not apply; it was
  blocked on per-window text. `tier5/uk_export.py --model aurora` emits the per-series
  CSV + weather text (templated from uk covariates), unblocking it on the same data.

## Licensing caveats (read before any public release)

Three of four ship **no `LICENSE` file** (`time_vlm`, `unicast`, `aurora`). These copies are
for **research reproduction only**, under their authors' rights — not ours. Before
publishing this repo: confirm a license with each author, convert those three to git
submodules (no redistribution), or drop to cite-only. `visionts_pp` carries its
`LICENSE.txt` (keep it); it bundles Apache-2.0 `uni2ts` code.

## Adaptations (the vendored code is NO LONGER pristine)

To make the cluster run "just submit" (no edits at run time), we made minimal in-place
edits — diff against the pinned upstream SHA to see them:

- `time_vlm/exp/exp_long_term_forecasting.py` — `test()` dumps per-window predictions to
  `results/<setting>/<test_csv_stem>_pred.npz` in our baseline-contract format (keyed by
  `data_path`, since one trained checkpoint is reused across all test plants).
- `time_vlm/data_provider/data_loader.py` — **added** `Dataset_UKPV`, registered in
  `data_factory.py` as `--data ukpv`. Upstream's `Dataset_Custom` cuts its own 70/10/20
  chronological split out of whatever CSV it is handed, which broke the disjoint
  cross-plant protocol three ways: it trained on ~48 of the 69 committed train plants,
  early-stopped on a slice of those same plants instead of the 15 committed val plants,
  and scored only the last 20% (winter) of each test plant while the skill-score
  reference covers the whole series — making SS arithmetically invalid. `Dataset_UKPV`
  uses each file whole, takes the split from committed plant membership, fits the scaler
  on the train plants only, and never lets a window straddle a plant boundary.
- `time_vlm/run.py` — upstream hardcoded `fix_seed = 2024` **before** `parse_args()`, so
  `--seed` only ever reached the augmentation code. Seeding now happens after parsing and
  is driven by `--seed` (default 42 = the protocol seed). Also adds `--resume`.
- `time_vlm/exp/exp_long_term_forecasting.py` — **resumable training**. Upstream persists
  only best-val *weights* (`checkpoint.pth`), so a SLURM walltime kill loses the optimizer
  moments, the epoch index and the early-stopping counters — the run restarts from zero.
  A `resume.pth` is now written every epoch with all of that; `--resume 1` picks up from
  it, so a kill costs at most one epoch and long trainings can be chained across jobs.
  `--warm_start 1` is the weaker fallback for a run killed *before* any `resume.pth`
  existed: it initializes from `checkpoint.pth` (weights only, fresh optimizer and LR
  schedule). That is **not** an exact resume and must be recorded as a warm start
  wherever the number is reported; it is ignored once a real `resume.pth` exists.
  Same file: the per-epoch `test` pass is skipped under `--data ukpv`. Upstream builds a
  test loader in `train()` and scores it each epoch only to print `Test Loss`; with
  `Dataset_UKPV`, `flag='test'` routes to `args.data_path`, which during training is the
  *train* CSV — so it re-scored all 69 train plants every epoch for a number nothing
  consumes (early stopping reads `vali_loss` alone). Pure walltime, now `nan`.
- `time_vlm/dataset/prompt_bank/ukpv.txt` — **added**: `utils.tools.load_content` keys the
  text prompt on `args.data`, so the `ukpv` loader needs its own entry.
- `visionts_pp/run_ukpv.py` — **added** (not upstream): self-contained zero-shot runner over
  the exported uk_pv CSVs, dumping `*_pred.npz`.
- The uk_pv → CSV bridge `tier4/vendor/export_ukpv.py` also emits `uk_pv_train_stacked.csv`
  (all train plants concatenated) plus the plant-blocked `uk_pv_train_protocol.csv` /
  `uk_pv_val_protocol.csv` that `Dataset_UKPV` consumes.
- `unicast/test_multi_modal_chronos.py` — added a `--dump_npz` flag (the only in-place
  edit to UniCast): writes `pred`/`true` in our baseline-contract format for
  `scripts/import_predictions.py`. Train/model code unchanged.
- `tier5/uk_export.py` — **added** (not upstream, lives outside `vendor/`): builds the
  UniCast (images) and Aurora (TS+text) on-disk formats from `tier6.uk_multimodal`
  (shared uk window bridge), so both run on uk_pv without further edits to their code.

Aurora's own source is unedited.

## Scoring mask (why the CSV bridge is not enough on its own)

`export_ukpv.py` writes a dense `date,OT` grid with missing steps filled `0.0`, which
destroys the two masks Tiers 0-3 score on: the clear-sky daylight mask
(`clearsky_ghi > 0`) and target validity (`norm_power` NaN). Scoring the dumped npz on
the proxy `true > 0` therefore counts outages as night and drops genuinely-overcast
daytime steps. `scripts/import_predictions.py --ukpv_dir <export dir> --data <parquet>`
rebuilds the exact mask: it re-derives each window's rows in the exported per-plant CSV
(`len(csv) = n_windows + seq_len + horizon − 1`), **verifies** that inference against the
dumped `true`, and reindexes the parquet's `norm_power`/`clearsky_ghi` onto those rows.
On a failed check it warns and falls back to the proxy rather than scoring a misaligned
mask. Every SLURM script that goes through the CSV bridge (`time_vlm`, `visionts_pp`,
`aurora`, `rag`) passes the flag; the mask actually used is recorded in the result
manifest under `config.daylight_mask`.

## Off-repo artifacts (NOT in git — see `.gitignore`)

Pretrained weights (VLM/CLIP backbones for Time-VLM, the VisionTS++ MAE checkpoint,
Chronos/Timer for UniCast, the Aurora checkpoint) and any datasets are downloaded on the
login node — see `baselines/README.md` and `scripts/slurm_tier5.sh`.
