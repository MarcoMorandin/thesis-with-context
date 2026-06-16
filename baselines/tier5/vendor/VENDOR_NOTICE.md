# Vendored Tier-5 baselines (generic multimodal TS) — provenance & licensing

Unmodified **code-only** copies of four upstream repos so Tier-5 runs the authors'
*original* implementations (BASELINE_COMPARISON.md §1, Tier 5), adapted to our
contract/dataset rather than reimplemented. Stripped on copy: `.git`, images,
notebooks, PDFs, bundled CSV/parquet datasets, checkpoints. No source edited.

| Vendor dir | Upstream | Commit SHA | License | Modality / track |
|---|---|---|---|---|
| `time_vlm/` | https://github.com/CityMind-Lab/ICML25-TimeVLM | `796e6ec963788657207ea2b5553740993ea3ea2b` | **none stated** ⚠️ (ICML 2025, arXiv:2502.04395) | TS→pseudo-image (+text) — **numerical track (uk_pv)** |
| `visionts_pp/` | https://github.com/HALF111/VisionTSpp | `484b2ea363b497217d0c3a078494c6af0251c275` | `LICENSE.txt` present (built on Salesforce `uni2ts`, Apache-2.0) | TS→image (vision MAE) — **numerical track (uk_pv)** |
| `unicast/` | https://github.com/adlnlp/UniCast | `a4af694615fabb9844a1a0f297aca148a3ab9db8` | **none stated** ⚠️ (arXiv:2508.11954) | real vision+text soft-prompt into TSFM — **multimodal track** |
| `aurora/` | https://github.com/decisionintelligence/Aurora | `a247760abbc9d17a861bc365c032368d317815f2` | **none stated** ⚠️ (arXiv:2509.22295) | multimodal generative TSFM (image+text) — **multimodal track** |

## Track split (what runs where)

- **Time-VLM, VisionTS++** render the time series itself as a pseudo-image — they need
  **no real satellite frames**, so they run on the **numerical uk_pv track** and match our
  `Y → ŷ` contract directly (like Tiers 0-4). These are the runnable Tier-5 rows today.
- **UniCast, Aurora** consume **real image + text** modalities → they need the
  **multimodal track** (skippd / goes16_nsrdb per DATASET_CONTRACT, still downloading).
  Vendored + scaffolded now; not runnable until that data + the vision/text channels land.

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
- `visionts_pp/run_ukpv.py` — **added** (not upstream): self-contained zero-shot runner over
  the exported uk_pv CSVs, dumping `*_pred.npz`.
- The uk_pv → CSV bridge `tier4/vendor/export_ukpv.py` also emits `uk_pv_train_stacked.csv`
  (all train plants concatenated) for Time-VLM's univariate `--features S` training.

UniCast / Aurora are unedited (multimodal track not available yet).

## Off-repo artifacts (NOT in git — see `.gitignore`)

Pretrained weights (VLM/CLIP backbones for Time-VLM, the VisionTS++ MAE checkpoint,
Chronos/Timer for UniCast, the Aurora checkpoint) and any datasets are downloaded on the
login node — see `docs/experiments/TIER5_INTEGRATION.md` and `scripts/slurm_tier5.sh`.
