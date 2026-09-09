# 40 — Does s2d's vision gain survive without future weather covariates? (A36)

Type: task
Status: open

## Question

Every MMTSFM arm on disk trains and scores with `future_cov="all"`: the 14 protocol
covariates — including `cloudcover`, `shortwave_radiation`, `direct_radiation`,
`diffuse_radiation`, `direct_normal_irradiance` — are exposed over the **horizon**, i.e.
next-6h NWP-style irradiance is a known input (`pv_record.py` docstring, project decision).
The reviewer question is unavoidable: *with future irradiance given, what does the sky add?*
No MMTSFM run at history-only covariates exists. The switch exists in `PVRecordDataset`
(`future_cov="history"`, used by the `itransformer_nf_histcov` row) but is **not plumbed**
through `MMTSFMDataModule` (`datamodule.py:89` never passes it).

**Launch spec** (register as A36 via `/register-experiment` first):

| Arm | Config | Seeds | Base |
|---|---|---|---|
| A36-s1 | `model=vision_chronos2_a36 +stage=s1 +ablation=A36s1` (`data.future_cov=deterministic`) | 42, 43, 44 | pretrained Chronos-2 (as s1) |
| A36-s2d | `model=vision_chronos2_s2d +stage=s2d +ablation=A36s2d` (`data.future_cov=deterministic`, `compute_marginal_gain=true`) | 42, 43, 44 | `uk_pv_s1_a36_s{seed}/best.ckpt` (the A36-s1 above, **not** the existing s1) |

Code before launch: add `future_cov` to `MMTSFMDataModule.__init__` hparams and pass it to
`PVRecordDataset`; default `"all"` so every existing config is unchanged; record it in the
results manifest `config` block. Cost: 3 × s1 + 3 × s2d.

Resolved when both arms are `DONE` in `ablations.md` and this ticket records, per seed: SS
and ramp NMAE for A36-s1 and A36-s2d, Δ ramp (vision-off) for A36-s2d, and the s2d−s1 gap at
history-only beside the existing gap at `future_cov=all` (+0.028 SS, −0.0066 ramp).

Reads: gap grows → vision is worth more when NWP is absent, deployable story strengthens.
Gap holds → claim is regime-independent. Gap collapses → the headline is an artefact of the
known-future-weather regime; the paper must lead with history-only or reframe.

Context: `paper-readiness-audit.md` §6 item 1.

## Code landed (2026-09-07) — and one correction to the launch spec

`future_cov` is now plumbed: `datamodule.py` takes it as an hparam (default `"all"`, so every
existing config is unchanged) and passes it to `PVRecordDataset` in `_make_dataset`. It is
recorded in the results manifest through a new `lightning_module.py::_data_cfg()`, which puts
a `data` block into `_run_cfg()` alongside `dataset_name`, `visual_window_hours`,
`video_frames`, `emit_vision`, `train_stride` and `num_entities`.

**Correction — the value is `deterministic`, not `history`.** This ticket's table says
`data.future_cov=history`. `history` is the `baselines/tier2/nf_itransformer.py` vocabulary;
`WindowDataset` (`baselines/common/windows.py:103`) accepts only `("deterministic", "all")`
and raises otherwise. `deterministic` is the same regime: it zeroes every covariate not
knowable in advance and keeps solar geometry / calendar / clearsky GHI
(`baselines/common/config.py::DETERMINISTIC_COVS`). The datamodule passes the string through
without validating it, deliberately — a wrong value then fails loudly at dataset construction
instead of training a silent confound, and no alias was added that could drift from the
baselines' own vocabulary.

Configs: `configs/ablation/A36s1.yaml` and `A36s2d.yaml` — two files, not one, because the
sweep turns a manifest id into `+ablation=<id>`, so the id IS the filename. A third file,
`configs/model/vision_chronos2_a36.yaml`, inherits `vision_chronos2_timeselfattn` unchanged
and exists only to rename the run: `slurm_curriculum.sh` derives the checkpoint dir and the
results tag from `MODEL_CFG`, so launching A36's s1 under the ordinary s1 model config would
overwrite `uk_pv_s1_selfattn_s{seed}` — the checkpoint A37/A39/A43 warm-start from. A36's s1
lands in `uk_pv_s1_a36_s{seed}` instead, which is A36s2d's BASE. Manifest rows `A36s1` /
`A36s2d` — the second is
submitted only after the first writes `uk_pv_s1_a36_s{seed}`, since warm-starting the s2d arm
from the existing all-covariate s1 would import an optimum found under the covariates this
ticket removes.

Status stays **open**: the code exists, the six runs do not.
