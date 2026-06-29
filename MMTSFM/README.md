# MMTSFM — Vision-Time Foundation Model for PV forecasting

Multimodal (time-series + satellite vision) foundation model built on Chronos-2
with a V-JEPA 2.1 / VidTok visual stream. This is the **thesis model**, compared
against the baselines under the shared protocol in
[`docs/experiments/BASELINE_PROTOCOL.md`](../docs/experiments/BASELINE_PROTOCOL.md).

## Protocol-aligned run (dataset of record)

The `uk_pv` / `goes_pvdaq` track reads the dataset of record
(`dataset_all.parquet` + `images_all.h5`) through `PVRecordDataset`, which reuses
`baselines/common` for the committed disjoint cross-plant splits, the
physical-time windows (14-day history / 6-hour horizon) and the protocol
covariates **including known future weather** (NWP-available assumption). At test
time it writes NMAE / NRMSE / Skill-Score (vs Smart Persistence) in the baselines
results schema, so `baselines/scripts/aggregate_all.py` lists MMTSFM next to the
other models.

```bash
# local smoke test (synthetic data)
uv run python -m mmtsfm.train

# protocol-aligned uk_pv run
uv run python -m mmtsfm.train +experiment=ukpv data.data_dir=/path/to/data_dir
```

## On Leonardo

```bash
# 1. login node (internet): env + weights (V-JEPA 2.1, Chronos-2) + checks
bash scripts/precache_login.sh

# 2. GPU node: train + test on uk_pv (and/or goes_pvdaq), results → baselines/results
sbatch scripts/run_all_mmtsfm.sh
```

See [`scripts/run_all_mmtsfm.sh`](scripts/run_all_mmtsfm.sh) for the configurable
env vars (datasets, encoder, epochs, paths).
