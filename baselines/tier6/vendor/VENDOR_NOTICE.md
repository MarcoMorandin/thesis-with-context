# Vendored Tier-6 baselines (PV-specialized multimodal, domain SOTA) — provenance & licensing

Unmodified **code-only** copies of the upstream repos so Tier-6 runs the authors'
*original* implementations (BASELINE_COMPARISON.md §1, Tier 6), adapted to our
contract/dataset rather than reimplemented. Stripped on copy: `.git`, images,
GIFs, videos, notebooks-as-figures, PDFs, bundled datasets, checkpoints. No
upstream source edited (see "Adaptations").

Both Tier-6 models consume **real images** (sky / satellite frames) → they live
on the **multimodal track** (skippd / goes16_nsrdb per DATASET_CONTRACT), like
Tier-5's UniCast / Aurora — vendored + scaffolded now, gated on that data.

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
Time-VLM / VisionTS++) — both need **real frames**. They run on the multimodal
track only: SKIPP'D (sky images, SUNSET-native and used by `solar_vlm/`) and
goes16_nsrdb (satellite, CrossViViT-style context). Vendored + scaffolded now;
runnable once the per-window `V` frames are wired to each harness's loader.

## Licensing

Both ship an MIT `LICENSE` (kept in-tree) — redistribution is permitted with the
copyright notice. Unlike three of the four Tier-5 vendors, no relicensing action
is needed before a public release; keep both `LICENSE` files intact.

## Adaptations (where the vendored code stops being pristine)

To make the cluster run "just submit", minimal in-place edits will land as the
multimodal loaders are wired — diff against the pinned SHA to see them. As of
this vendor drop **both trees are pristine** (no edits yet); the per-model SLURM
scripts (`scripts/slurm_crossvivit.sh`, `scripts/slurm_sunset.sh`) fail loud
until the multimodal dataset exists. Planned, when the data lands:

- `crossvivit/` — a thin `tscontext_datamodule` config pointing at our
  multimodal-track frames + `Y`, and a prediction-dump hook in the Lightning
  `test_step` writing `results/<setting>/<site>_pred.npz` in our
  baseline-contract format (`pred`,`true` (N,H[,1])), for `import_predictions.py`.
- `sunset/` — a self-contained `run_skippd.py` runner (converted from
  `SUNSET_forecast.ipynb`, no notebook execution at run time) that trains/evals
  on the SKIPP'D HDF5 and dumps `sunset_<site>_pred.npz`.

## Off-repo artifacts (NOT in git — see `.gitignore`)

Pretrained weights / checkpoints (CrossViViT released ckpts, any SUNSET weights)
and datasets (DeepLake SunLake, SKIPP'D HDF5) are downloaded on the login node —
see `docs/experiments/TIER6_INTEGRATION.md`.
