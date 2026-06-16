# Vendored Tier-6 baselines (PV-specialized multimodal, domain SOTA) — provenance & licensing

Unmodified **code-only** copies of the upstream repos so Tier-6 runs the authors'
*original* implementations (BASELINE_COMPARISON.md §1, Tier 6), adapted to our
contract/dataset rather than reimplemented. Stripped on copy: `.git`, images,
GIFs, videos, notebooks-as-figures, PDFs, bundled datasets, checkpoints. No
upstream source edited (see "Adaptations").

Both Tier-6 models consume **real images** (sky / satellite frames). They run on
the **uk_pv multimodal track**: the curated numerical `Y` + the per-plant
satellite frames in `images_all.h5`, aligned by the canonical
`image_h5_index` pointer and fed through `tier6/uk_multimodal.py`
(`UKMultimodalDataset`). The vendored models are driven by their **own original
code** via per-model `run_ukpv.py` adapters — see "Adaptations".

| Vendor dir | Upstream | Commit SHA | License | Modality / track |
|---|---|---|---|---|
| `crossvivit/` | https://github.com/gitbooo/CrossViVit | `ce345ff97b11b65cb7a46782695af2140272c1e3` | **MIT** (`LICENSE`, © 2023 Ghait Boukachab) | satellite `V` + irradiance `Y` cross-attention — **multimodal track** |
| `sunset/` | https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset | `c4c3d0acf953d32f06c9748ab9fdee083c65593c` | **MIT** (`LICENSE`, © 2022 Yuhao Nie, Xiatong Li) | sky-image `V` + PV history `Y` CNN — **multimodal track (SKIPP'D-native)** |

Third Tier-6 P0 model, **Solar-VLM**, is already ported under `baselines/solar_vlm/`
(not re-vendored here). SPIRIT (P1), PV-VLM / M3S-Net / MDCTL-MCI (P2) are
cite-only for now — add here if reviewers demand (BASELINE_COMPARISON.md §1 Tier 6).

## What each model is

- **CrossViViT** (Boussif et al., NeurIPS 2023, [arXiv:2306.01112](https://arxiv.org/abs/2306.01112)) —
  the reference *deep* satellite+TS cross-attention model; strongest non-FM
  multimodal competitor. PyTorch 2.0 + Lightning + Hydra. Entry: `main.py
  experiment=cross_vivit`. Native data = the DeepLake `hub://crossvivit/SunLake`
  (EUMETSAT satellite context + ground irradiance stations); for our protocol it
  reads the multimodal-track frames + `Y` instead.
- **SUNSET** (Nie et al., Stanford "Neural Network for Solar Electricity Trend") —
  canonical sky-image CNN baseline used by most PV-vision related work.
  TensorFlow 2.4 / Keras, notebook-based (`models/SUNSET_forecast.ipynb` =
  15-min-ahead PV forecast from a stack of past sky images + PV history;
  `SUNSET_nowcast.ipynb` = contemporaneous nowcast). Native data = SKIPP'D sky
  images as an HDF5 (`forecast_dataset.hdf5`), the same dataset `solar_vlm/`
  already consumes.

## Track split (what runs where)

Neither Tier-6 model renders the series as a pseudo-image (unlike Tier-5
Time-VLM / VisionTS++) — both need **real frames**. The dataset of record carries
them (`images_all.h5`, per-site groups; `uk_pv` 128px gray, 30-min daylight
cadence), so both run on uk_pv via the shared `tier6/uk_multimodal.py`
bridge + the per-model `run_ukpv.py` runners.

## Licensing

Both ship an MIT `LICENSE` (kept in-tree) — redistribution is permitted with the
copyright notice. Unlike three of the four Tier-5 vendors, no relicensing action
is needed before a public release; keep both `LICENSE` files intact.

## Adaptations (where the vendored code stops being pristine)

The upstream model code is **unmodified** (diff against the pinned SHA to
confirm). Each model gets ONE added file — a `run_ukpv.py` adapter that imports
the original model and feeds it the uk_pv multimodal windows — plus the shared
`tier6/uk_multimodal.py` bridge. No upstream source was edited.

- `crossvivit/run_ukpv.py` (added) — imports the original
  `src.models.cross_vivit.RoCrossViViT` **unchanged** and drives it on uk_pv:
  the last `pred_len` steps of each history window form CrossViViT's shared
  context window (satellite `V` + PV/covariate `ts`), trained to forecast the
  next `pred_len` PV steps. Dumps `crossvivit_<site>_pred.npz`.
  **Approximations** (uk_pv ≠ the authors' georeferenced DeepLake SunLake):
  single-channel 128px→S crops (`ctx_channels=1`) vs multi-band frames; no
  optical-flow channels, no elevation; per-pixel `ctx_coords` synthesized as a
  small lat/lon grid around the plant; `ts_coords` = plant lat/lon. These weaken
  CrossViViT's spatial grounding — report the row with this caveat.
- `sunset/run_ukpv.py` (added) — transcribes the original
  `models/SUNSET_forecast.ipynb` Keras graph (2 conv blocks 24→48 + BN/maxpool,
  Flatten ⊕ PV history, 2× Dense(1024)/Dropout, MSE/Adam) and feeds it the
  uk_pv sky-image stack `V` + PV history. **Only change** vs upstream: final
  Dense head widened 1 → H (the original predicts a single 15-min step; our
  protocol forecasts H), masked MSE over `mask_future`. Dumps
  `sunset_<site>_pred.npz`.

Both runners dump `<model>_<site>_pred.npz` (`pred`,`true` (N,H)) for
`scripts/import_predictions.py`; the per-model SLURM scripts
(`scripts/slurm_{crossvivit,sunset}.sh`) wire export-free → train → eval →
contract-check → import end-to-end on uk_pv. Verified to run end-to-end on the
real data (CPU smoke test, tiny windows) before the cluster sweep.

## Off-repo artifacts (NOT in git — see `.gitignore`)

Both models train from scratch on uk_pv — no pretrained weights. The data
(`dataset_all.parquet` + `images_all.h5`) lives on the read-only dataset of record
(`/Volumes/SSD/thesis-dataset/`) / staged to `$TEAM_SCRATCH` on the cluster; checkpoints
(`*_best.pt`, `repetition_*/`) and the dumped `*_pred.npz` are run outputs, not
committed. See `docs/experiments/TIER6_INTEGRATION.md`.
