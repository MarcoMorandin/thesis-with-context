# Visual Fusion Differential Diagnosis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the gated diagnostic instrument from
[2026-08-19-visual-fusion-diagnosis.md](2026-08-19-visual-fusion-diagnosis.md) — a CPU-only
ceiling probe (G0), localization probes (G1), and two model interventions plus a 4-GPU
factorial runner (G2) — so that "vision contributes 0.4 % " becomes an attributable cause with
a matched fix.

**Architecture:** G0/G1 are standalone scripts that read the *existing* V-JEPA latent cache and
the parquet directly, never instantiating the model — so they run on CPU nodes for free. G2
adds two behaviour-preserving flags to the model (`n_visual_tokens_per_step`,
`visual_aux_loss_weight`), both defaulting to current behaviour, exercised by one sbatch that
saturates a single 4-GPU node.

**Tech Stack:** Python 3.11, PyTorch, Lightning, Hydra, pandas/pyarrow, scikit-learn (ridge),
pytest, SLURM.

## Global Constraints

- Package manager is `uv`. Never bare `pip`/`python` — a hook blocks it.
- Dataset of record is **read-only**: `/leonardo_scratch/fast/IscrC_MTSFM/data/dataset_all.parquet`, `images_all.h5`.
- V-JEPA latent cache: `/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224`, one `.pt` per key, key format `{dataset}_{site_id}_{origin}` where `origin = int(timestamps[T-1])`.
- Target column is `norm_power` (capacity-normalized, `[0,1]`); time column `timestamp_utc`; site column `site_id`; dataset column `dataset`.
- `COV_COLS` = the 14 keys of `COV_SCALES` in `baselines/common/config.py`, in that order. `csi` and `kt` are NOT covariates.
- `DETERMINISTIC_COVS` = `solar_zenith, solar_azimuth, doy_sin, doy_cos, solar_time, clearsky_ghi` — the only covariates valid at future timestamps.
- Plant splits come from `baselines/common/splits.load_splits()`; test plants are disjoint and must never be fit on.
- New files under `MMTSFM/src` target < 150 lines (hook-enforced).
- Every G2 arm verifies checkpoint integrity with `scripts/repair_vjepa_checkpoint.py --inspect` before its number is trusted.
- Reserve SLURM walltime to the 24 h partition cap: `train.py` exports `best.ckpt` and runs the test pass only after `fit()` returns, so a TIMEOUT loses all artifacts.

---

## File Structure

| Path | Responsibility |
|---|---|
| `MMTSFM/scripts/probes/ceiling_dataset.py` | Assemble `(X_vis, X_cov, Y, site, origin)` arrays from cache + parquet. No fitting. |
| `MMTSFM/scripts/probes/fit_ceiling.py` | Fit ridge/MLP over the three predictor sets, emit per-horizon skill JSON. |
| `MMTSFM/scripts/probes/localize.py` | G1: horizon decomposition + condition stratification from existing `*_pred.npz`. |
| `MMTSFM/scripts/probes/g0_ceiling.sbatch` | CPU-partition runner for G0. |
| `MMTSFM/scripts/g2_fusion_factorial.sbatch` | 4-GPU node runner for the G2 factorial. |
| `MMTSFM/src/mmtsfm/models/vision/latent_summarizer.py` | *modify*: emit `k` tokens per visual step. |
| `MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py` | *modify*: interleave `k` tokens; aux head. |
| `MMTSFM/src/mmtsfm/models/chronos2/lightning_module.py` | *modify*: aux loss term. |
| `MMTSFM/tests/test_ceiling_dataset.py` | Task 1 tests. |
| `MMTSFM/tests/test_fit_ceiling.py` | Task 2 tests, incl. planted-signal recovery. |
| `MMTSFM/tests/test_visual_tokens_per_step.py` | Task 5 tests. |
| `MMTSFM/tests/test_visual_aux_loss.py` | Task 6 tests. |

---

### Task 1: Ceiling-probe dataset assembly

**Files:**
- Create: `MMTSFM/scripts/probes/__init__.py` (empty)
- Create: `MMTSFM/scripts/probes/ceiling_dataset.py`
- Test: `MMTSFM/tests/test_ceiling_dataset.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `parse_cache_key(name: str) -> tuple[str, str, int]` returning `(dataset, site_id, origin)`
  - `build_arrays(cache_dir: Path, parquet_path: Path, sites: set[str], horizon: int = 12, step_seconds: int = 1800, max_files: int | None = None) -> dict[str, np.ndarray]` returning keys `X_vis` `[N, 4096]`, `X_cov` `[N, D_cov]`, `Y` `[N, horizon]`, `Y_mask` `[N, horizon]`, `site` `[N]`, `origin` `[N]`

- [ ] **Step 1: Verify the timestamp unit before writing code that assumes it**

The cache key's `origin` is `int(timestamps[T-1])` and `timestamps` is `int64`, but the epoch
unit (s vs ns) is not documented. Run on Leonardo:

```bash
cd ~/thesis-with-context/MMTSFM
uv run python -c "
import pathlib, pandas as pd
f = sorted(pathlib.Path('/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224').glob('*.pt'))[0]
ds, site, origin = f.stem.rsplit('_', 2)
print('key:', f.stem, '-> origin =', origin)
df = pd.read_parquet('/leonardo_scratch/fast/IscrC_MTSFM/data/dataset_all.parquet', columns=['dataset','site_id','timestamp_utc'])
df = df[(df.dataset==ds) & (df.site_id.astype(str)==site)]
t = pd.to_datetime(df.timestamp_utc)
for unit in ('s','ms','us','ns'):
    hit = (t.view('int64') // {'s':10**9,'ms':10**6,'us':10**3,'ns':1}[unit] == int(origin)).sum()
    print(f'  unit={unit}: {hit} matching rows')
"
```

Exactly one unit will report a non-zero count. Record it; it becomes `EPOCH_UNIT` in Step 3.

- [ ] **Step 2: Write the failing test**

```python
# MMTSFM/tests/test_ceiling_dataset.py
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.probes.ceiling_dataset import build_arrays, parse_cache_key


def test_parse_cache_key_splits_from_the_right():
    # site ids are numeric strings, dataset names contain underscores
    assert parse_cache_key("uk_pv_7239_1546300800") == ("uk_pv", "7239", 1546300800)
    assert parse_cache_key("goes_pvdaq_1202_1546300800") == (
        "goes_pvdaq", "1202", 1546300800,
    )


@pytest.fixture
def tiny(tmp_path):
    """One site, 40 half-hourly rows, 2 cached windows."""
    t0 = 1546300800  # 2019-01-01T00:00:00Z
    n = 40
    rows = {
        "dataset": ["uk_pv"] * n,
        "site_id": ["9001"] * n,
        "timestamp_utc": pd.to_datetime(
            [t0 + 1800 * i for i in range(n)], unit="s", utc=True
        ),
        "norm_power": np.linspace(0.0, 1.0, n),
        "installed_power_w": [3000.0] * n,
    }
    from common import config

    for c in config.COV_COLS:
        rows[c] = np.arange(n, dtype=float)
    pq = tmp_path / "d.parquet"
    pd.DataFrame(rows).to_parquet(pq)

    cache = tmp_path / "cache"
    cache.mkdir()
    for origin_i in (10, 11):
        torch.save(
            torch.full((4, 196, 1024), float(origin_i), dtype=torch.float16),
            cache / f"uk_pv_9001_{t0 + 1800 * origin_i}.pt",
        )
    return cache, pq


def test_build_arrays_shapes_and_alignment(tiny):
    cache, pq = tiny
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    assert out["X_vis"].shape == (2, 4096)      # spatial mean-pool, time kept
    assert out["Y"].shape == (2, 12)
    assert out["Y_mask"].shape == (2, 12)
    assert out["X_cov"].shape[0] == 2
    # rows are sorted by origin, so row 0 is the earlier window
    assert out["origin"][0] < out["origin"][1]
    # X_vis row i must come from the file whose origin matches row i
    assert np.allclose(out["X_vis"][0], 10.0)
    assert np.allclose(out["X_vis"][1], 11.0)


def test_build_arrays_masks_missing_future(tiny):
    cache, pq = tiny
    out = build_arrays(cache, pq, sites={"9001"}, horizon=12)
    # window at origin index 11 runs off the end of a 40-row table at h=12
    # only if 11+12 >= 40; here it does not, so every step is valid
    assert out["Y_mask"].all()


def test_build_arrays_rejects_unknown_sites(tiny):
    cache, pq = tiny
    out = build_arrays(cache, pq, sites={"9999"}, horizon=12)
    assert out["X_vis"].shape[0] == 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd MMTSFM && uv run pytest tests/test_ceiling_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.probes'`

- [ ] **Step 4: Implement**

```python
# MMTSFM/scripts/probes/ceiling_dataset.py
"""Assemble the G0 ceiling-probe design matrices from the V-JEPA latent cache.

Reads the cache and the parquet directly — no datamodule, no model, no GPU. Cache
keys are `{dataset}_{site_id}_{origin}` (see PVRecordDataset._entity_cache_key),
so each cached window joins to the table on (dataset, site_id, timestamp==origin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "baselines"))
from common import config  # noqa: E402

# Verified in Task 1 Step 1. Origins are whole seconds since the epoch.
EPOCH_UNIT = "s"
_DIVISOR = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[EPOCH_UNIT]


def parse_cache_key(name: str) -> tuple[str, str, int]:
    """`uk_pv_7239_1546300800` -> `("uk_pv", "7239", 1546300800)`.

    Split from the RIGHT: dataset names contain underscores, site ids do not.
    """
    dataset, site_id, origin = name.rsplit("_", 2)
    return dataset, site_id, int(origin)


def build_arrays(
    cache_dir: Path,
    parquet_path: Path,
    sites: set[str],
    horizon: int = 12,
    step_seconds: int = 1800,
    max_files: int | None = None,
) -> dict[str, np.ndarray]:
    """Design matrices for one split's plants.

    X_vis   [N, 4096]  V-JEPA latents, mean-pooled over the 196 spatial patches,
                       4 latent steps kept and flattened (feature variant F2).
    X_cov   [N, D]     history covariates at the origin + future deterministic
                       covariates at each horizon step — exactly what the model
                       has access to.
    Y       [N, H]     norm_power at origin + h*step_seconds.
    Y_mask  [N, H]     False where that future step is absent from the table.
    """
    cache_dir, parquet_path = Path(cache_dir), Path(parquet_path)

    files = sorted(cache_dir.glob("*.pt"))
    parsed = [(f, *parse_cache_key(f.stem)) for f in files]
    parsed = [p for p in parsed if p[2] in sites]
    parsed.sort(key=lambda p: (p[2], p[3]))
    if max_files is not None:
        parsed = parsed[:max_files]

    cols = sorted(
        {
            config.DATASET_COL,
            config.SITE_COL,
            config.TIME_COL,
            config.TARGET_COL,
            *config.COV_COLS,
        }
    )
    df = pd.read_parquet(parquet_path, columns=cols)
    df[config.SITE_COL] = df[config.SITE_COL].astype(str)
    df["_epoch"] = (
        pd.to_datetime(df[config.TIME_COL], utc=True).view("int64") // _DIVISOR
    )
    table = df.set_index([config.DATASET_COL, config.SITE_COL, "_epoch"]).sort_index()

    det_idx = list(config.DETERMINISTIC_COV_IDX)
    n = len(parsed)
    X_vis = np.zeros((n, 4 * 1024), dtype=np.float32)
    X_cov = np.zeros((n, len(config.COV_COLS) + horizon * len(det_idx)), np.float32)
    Y = np.zeros((n, horizon), dtype=np.float32)
    Y_mask = np.zeros((n, horizon), dtype=bool)
    site_out = np.empty(n, dtype=object)
    origin_out = np.zeros(n, dtype=np.int64)

    for i, (path, ds, site, origin) in enumerate(parsed):
        z = torch.load(path, map_location="cpu", weights_only=True).float()
        X_vis[i] = z.mean(dim=1).reshape(-1).numpy()  # [4, 196, 1024] -> [4, 1024]

        try:
            hist = table.loc[(ds, site, origin)]
        except KeyError:
            continue
        if isinstance(hist, pd.DataFrame):
            hist = hist.iloc[0]
        X_cov[i, : len(config.COV_COLS)] = [
            float(hist[c]) for c in config.COV_COLS
        ]

        base = len(config.COV_COLS)
        for h in range(1, horizon + 1):
            ts = origin + h * step_seconds
            try:
                fut = table.loc[(ds, site, ts)]
            except KeyError:
                continue
            if isinstance(fut, pd.DataFrame):
                fut = fut.iloc[0]
            y = fut[config.TARGET_COL]
            if pd.isna(y):
                continue
            Y[i, h - 1] = float(y)
            Y_mask[i, h - 1] = True
            off = base + (h - 1) * len(det_idx)
            X_cov[i, off : off + len(det_idx)] = [
                float(fut[config.COV_COLS[j]]) for j in det_idx
            ]

        site_out[i] = site
        origin_out[i] = origin

    return {
        "X_vis": X_vis,
        "X_cov": X_cov,
        "Y": Y,
        "Y_mask": Y_mask,
        "site": site_out,
        "origin": origin_out,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd MMTSFM && uv run pytest tests/test_ceiling_dataset.py -v`
Expected: 4 passed. If `EPOCH_UNIT` from Step 1 was not `"s"`, change the constant and re-run.

- [ ] **Step 6: Commit**

```bash
git add MMTSFM/scripts/probes/__init__.py MMTSFM/scripts/probes/ceiling_dataset.py MMTSFM/tests/test_ceiling_dataset.py
git commit -m "feat(probes): assemble G0 ceiling-probe design matrices from the latent cache"
```

---

### Task 2: Ceiling probe fit and per-horizon report

**Files:**
- Create: `MMTSFM/scripts/probes/fit_ceiling.py`
- Test: `MMTSFM/tests/test_fit_ceiling.py`

**Interfaces:**
- Consumes: `build_arrays(...)` from Task 1.
- Produces: `fit_and_score(arrays_train: dict, arrays_test: dict, horizon: int = 12, n_folds: int = 5, alpha: float = 1.0) -> dict` returning `{"a": {...}, "b": {...}, "c": {...}, "conditional": [...], "cv_spread": [...]}` where each set maps to `{"nmae": [H floats], "skill": [H floats]}` and `conditional` is the per-horizon `skill(c) - skill(b)`.

- [ ] **Step 1: Write the failing test — planted-signal recovery first**

The single most important test: a probe that cannot recover a *known* planted signal would
report "no signal" for an implementation reason, and G0 would produce a false negative that
ends the study wrongly.

```python
# MMTSFM/tests/test_fit_ceiling.py
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from scripts.probes.fit_ceiling import fit_and_score


def _arrays(n, d_vis=32, d_cov=8, horizon=4, vis_weight=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X_vis = rng.normal(size=(n, d_vis)).astype(np.float32)
    X_cov = rng.normal(size=(n, d_cov)).astype(np.float32)
    # target = a covariate-driven part + an optional vision-only part
    base = X_cov[:, :1] * 0.5
    vis = (X_vis[:, :1] * vis_weight).astype(np.float32)
    Y = np.repeat(base + vis, horizon, axis=1).astype(np.float32)
    Y += rng.normal(scale=0.01, size=Y.shape).astype(np.float32)
    return {
        "X_vis": X_vis,
        "X_cov": X_cov,
        "Y": Y,
        "Y_mask": np.ones_like(Y, dtype=bool),
        "site": np.array([str(i % 7) for i in range(n)], dtype=object),
        "origin": np.arange(n, dtype=np.int64),
    }


def test_recovers_a_planted_visual_signal():
    """If vision genuinely carries signal, (c)-(b) must be clearly positive."""
    tr = _arrays(4000, vis_weight=1.0, seed=1)
    te = _arrays(1000, vis_weight=1.0, seed=2)
    res = fit_and_score(tr, te, horizon=4)
    assert min(res["conditional"]) > 0.05, res["conditional"]


def test_reports_no_conditional_signal_when_vision_is_noise():
    """Vision uncorrelated with the target must give (c)-(b) ~ 0."""
    tr = _arrays(4000, vis_weight=0.0, seed=3)
    te = _arrays(1000, vis_weight=0.0, seed=4)
    res = fit_and_score(tr, te, horizon=4)
    assert max(res["conditional"]) < 0.02, res["conditional"]


def test_covariate_probe_beats_trivial_baseline():
    """Sanity guard: if (b) cannot beat the mean predictor, feature assembly is wrong."""
    tr, te = _arrays(2000, seed=5), _arrays(500, seed=6)
    res = fit_and_score(tr, te, horizon=4)
    assert min(res["b"]["skill"]) > 0.0


def test_cv_spread_is_reported_per_horizon():
    tr, te = _arrays(2000, seed=7), _arrays(500, seed=8)
    res = fit_and_score(tr, te, horizon=4)
    assert len(res["cv_spread"]) == 4
    assert all(v >= 0.0 for v in res["cv_spread"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MMTSFM && uv run pytest tests/test_fit_ceiling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.probes.fit_ceiling'`

- [ ] **Step 3: Implement**

```python
# MMTSFM/scripts/probes/fit_ceiling.py
"""Fit the G0 ceiling probes and report per-horizon skill.

Three predictor sets: (a) visual only, (b) covariates only, (c) both. The
operative quantity is (c) - (b): what vision adds CONDITIONAL on what the numeric
channel already knows. It upper-bounds what any fusion architecture could extract.

Target is norm_power (the quantity protocol.md scores), so probe skill is directly
comparable with the reported SS numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


def _skill(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, ref: np.ndarray) -> float:
    """1 - NMAE(pred)/NMAE(ref) on the masked entries. Higher is better."""
    if mask.sum() == 0:
        return float("nan")
    num = np.abs(pred[mask] - y[mask]).mean()
    den = np.abs(ref[mask] - y[mask]).mean()
    return float(1.0 - num / den) if den > 0 else float("nan")


def _design(arrays: dict, which: str) -> np.ndarray:
    if which == "a":
        return arrays["X_vis"]
    if which == "b":
        return arrays["X_cov"]
    return np.concatenate([arrays["X_vis"], arrays["X_cov"]], axis=1)


def fit_and_score(
    arrays_train: dict,
    arrays_test: dict,
    horizon: int = 12,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> dict:
    out: dict = {}
    y_tr, m_tr = arrays_train["Y"], arrays_train["Y_mask"]
    y_te, m_te = arrays_test["Y"], arrays_test["Y_mask"]

    # Reference = persistence of the most recent observed value. X_cov column 0
    # is not the target, so use the per-horizon train mean as the naive reference;
    # the real run overrides this with smart persistence (see run_g0()).
    ref_te = np.tile(
        np.array([y_tr[:, h][m_tr[:, h]].mean() for h in range(horizon)]),
        (y_te.shape[0], 1),
    )

    for which in ("a", "b", "c"):
        Xtr, Xte = _design(arrays_train, which), _design(arrays_test, which)
        nmae, skill = [], []
        for h in range(horizon):
            sel = m_tr[:, h]
            model = Ridge(alpha=alpha)
            model.fit(Xtr[sel], y_tr[sel, h])
            pred = model.predict(Xte)
            nmae.append(
                float(np.abs(pred[m_te[:, h]] - y_te[m_te[:, h], h]).mean())
                if m_te[:, h].any()
                else float("nan")
            )
            skill.append(_skill(pred, y_te[:, h], m_te[:, h], ref_te[:, h]))
        out[which] = {"nmae": nmae, "skill": skill}

    out["conditional"] = [
        out["c"]["skill"][h] - out["b"]["skill"][h] for h in range(horizon)
    ]

    # CV spread of the conditional quantity, grouped by plant so folds stay
    # plant-disjoint — the same discipline the protocol demands of the model.
    groups = arrays_train["site"]
    n_splits = min(n_folds, len(set(groups.tolist())))
    spread = []
    for h in range(horizon):
        sel = m_tr[:, h]
        if sel.sum() < n_splits or len(set(groups[sel].tolist())) < n_splits:
            spread.append(float("nan"))
            continue
        vals = []
        gkf = GroupKFold(n_splits=n_splits)
        Xb, Xc = _design(arrays_train, "b")[sel], _design(arrays_train, "c")[sel]
        yh, gh = y_tr[sel, h], groups[sel]
        for tr_i, va_i in gkf.split(Xb, yh, gh):
            rb = Ridge(alpha=alpha).fit(Xb[tr_i], yh[tr_i]).predict(Xb[va_i])
            rc = Ridge(alpha=alpha).fit(Xc[tr_i], yh[tr_i]).predict(Xc[va_i])
            m = np.ones_like(yh[va_i], dtype=bool)
            r = np.full_like(yh[va_i], yh[tr_i].mean())
            vals.append(_skill(rc, yh[va_i], m, r) - _skill(rb, yh[va_i], m, r))
        spread.append(float(np.std(vals)))
    out["cv_spread"] = spread
    return out


def run_g0(
    cache_dir: str,
    parquet_path: str,
    out_path: str,
    horizon: int = 12,
    max_files: int | None = None,
) -> dict:
    """Assemble train/test arrays from the real cache and write the report."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "baselines"))
    from common.splits import load_splits  # noqa: E402

    from scripts.probes.ceiling_dataset import build_arrays

    splits = load_splits()
    train_sites = {str(s) for s in splits["uk_pv"]["train"]}
    test_sites = {str(s) for s in splits["uk_pv"]["test"]}
    assert not (train_sites & test_sites), "train/test plants must be disjoint"

    tr = build_arrays(Path(cache_dir), Path(parquet_path), train_sites, horizon, max_files=max_files)
    te = build_arrays(Path(cache_dir), Path(parquet_path), test_sites, horizon, max_files=max_files)
    res = fit_and_score(tr, te, horizon=horizon)
    res["n_train"], res["n_test"] = int(tr["Y"].shape[0]), int(te["Y"].shape[0])
    Path(out_path).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--max-files", type=int, default=None)
    a = ap.parse_args()
    r = run_g0(a.cache_dir, a.parquet, a.out, a.horizon, a.max_files)
    print("conditional (c)-(b) per horizon:", [round(v, 5) for v in r["conditional"]])
    print("cv spread                     :", [round(v, 5) for v in r["cv_spread"]])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd MMTSFM && uv run pytest tests/test_fit_ceiling.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add MMTSFM/scripts/probes/fit_ceiling.py MMTSFM/tests/test_fit_ceiling.py
git commit -m "feat(probes): G0 ceiling probe with planted-signal recovery tests"
```

---

### Task 3: G0 CPU-partition runner

**Files:**
- Create: `MMTSFM/scripts/probes/g0_ceiling.sbatch`

**Interfaces:**
- Consumes: `scripts/probes/fit_ceiling.py` CLI from Task 2.
- Produces: `baselines/results/probes/g0_ceiling_ukpv.json`.

- [ ] **Step 1: Write the runner**

```bash
#!/bin/bash
#SBATCH --job-name=g0_ceiling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# G0 needs no GPU: it reads cached V-JEPA latents and fits ridge probes. Running
# it on the CPU partition keeps the whole diagnosis off the GPU allocation.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export UV_OFFLINE=1 UV_NO_SYNC=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

OUT=../baselines/results/probes
mkdir -p "$OUT"

uv run python -m scripts.probes.fit_ceiling \
  --cache-dir /leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224 \
  --parquet   /leonardo_scratch/fast/IscrC_MTSFM/data/dataset_all.parquet \
  --out       "$OUT/g0_ceiling_ukpv.json" \
  --horizon   12
```

- [ ] **Step 2: Smoke it locally with a file cap before submitting**

Run: `cd MMTSFM && uv run python -m scripts.probes.fit_ceiling --cache-dir <any dir with a few .pt> --parquet <parquet> --out /tmp/g0_smoke.json --max-files 200`
Expected: prints two 12-element lists, writes `/tmp/g0_smoke.json`. This catches path and
join errors in seconds rather than after a queue wait.

- [ ] **Step 3: Commit**

```bash
git add MMTSFM/scripts/probes/g0_ceiling.sbatch
git commit -m "feat(probes): CPU-partition runner for the G0 ceiling probe"
```

---

### Task 4: G1 localization — horizon decomposition and condition stratification

**Files:**
- Create: `MMTSFM/scripts/probes/localize.py`
- Test: `MMTSFM/tests/test_localize.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `decompose_by_horizon(pred_on: np.ndarray, pred_off: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict` returning `{"nmae_on": [H], "nmae_off": [H], "delta": [H]}`; `stratify_by_variability(delta_per_sample: np.ndarray, csi_var: np.ndarray, n_bins: int = 3) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# MMTSFM/tests/test_localize.py
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from scripts.probes.localize import decompose_by_horizon, stratify_by_variability


def test_decompose_finds_a_short_horizon_only_effect():
    """The C5 signature: vision helps at h=0..1 and not beyond."""
    n, h = 500, 6
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n, h))
    pred_off = y + rng.normal(scale=0.5, size=(n, h))
    pred_on = pred_off.copy()
    pred_on[:, :2] = y[:, :2] + rng.normal(scale=0.1, size=(n, 2))  # much better early
    res = decompose_by_horizon(pred_on, pred_off, y, np.ones((n, h), bool))
    assert res["delta"][0] > 0.1 and res["delta"][1] > 0.1
    assert abs(res["delta"][4]) < 0.05 and abs(res["delta"][5]) < 0.05


def test_decompose_respects_the_mask():
    n, h = 100, 3
    y = np.zeros((n, h))
    pred_on = np.ones((n, h))
    pred_off = np.full((n, h), 2.0)
    mask = np.zeros((n, h), bool)
    mask[:, 0] = True
    res = decompose_by_horizon(pred_on, pred_off, y, mask)
    assert res["nmae_on"][0] == 1.0
    assert np.isnan(res["nmae_on"][1])


def test_stratify_orders_bins_by_variability():
    delta = np.concatenate([np.zeros(300), np.full(300, 0.2)])
    csi_var = np.concatenate([np.zeros(300), np.ones(300)])
    res = stratify_by_variability(delta, csi_var, n_bins=2)
    assert res["mean_delta"][0] < res["mean_delta"][1]
    assert res["counts"] == [300, 300]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MMTSFM && uv run pytest tests/test_localize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.probes.localize'`

- [ ] **Step 3: Implement**

```python
# MMTSFM/scripts/probes/localize.py
"""G1 localization probes.

Reads only artifacts that already exist: the per-site prediction dumps written by
ProtocolEvaluator (`results/predictions/<tag>_<site>_pred.npz`) and the covariate
table. Distinguishes C5 (horizon mismatch / dilution) from a uniform effect.
"""

from __future__ import annotations

import numpy as np


def decompose_by_horizon(
    pred_on: np.ndarray,
    pred_off: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Per-horizon NMAE with vision on vs off, and their difference.

    A positive `delta[h]` means vision helped at horizon step h. The aggregate
    marginal gain is a mask-weighted average of these, so a strong early effect
    can be invisible once averaged over 12 steps.
    """
    h = y.shape[1]
    nmae_on, nmae_off = [], []
    for i in range(h):
        m = mask[:, i]
        if not m.any():
            nmae_on.append(float("nan"))
            nmae_off.append(float("nan"))
            continue
        nmae_on.append(float(np.abs(pred_on[m, i] - y[m, i]).mean()))
        nmae_off.append(float(np.abs(pred_off[m, i] - y[m, i]).mean()))
    delta = [
        (nmae_off[i] - nmae_on[i]) if not (np.isnan(nmae_on[i]) or np.isnan(nmae_off[i]))
        else float("nan")
        for i in range(h)
    ]
    return {"nmae_on": nmae_on, "nmae_off": nmae_off, "delta": delta}


def stratify_by_variability(
    delta_per_sample: np.ndarray,
    csi_var: np.ndarray,
    n_bins: int = 3,
) -> dict:
    """Group the per-sample vision benefit by within-window sky variability.

    If vision only pays on variable-sky windows, the aggregate is diluted by the
    clear and fully-overcast majority — a reporting problem, not a model problem.
    """
    order = np.argsort(csi_var)
    bins = np.array_split(order, n_bins)
    return {
        "mean_delta": [float(delta_per_sample[b].mean()) for b in bins],
        "counts": [int(len(b)) for b in bins],
        "var_edges": [float(csi_var[b].min()) for b in bins]
        + [float(csi_var[order[-1]])],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd MMTSFM && uv run pytest tests/test_localize.py -v`
Expected: 3 passed.

- [ ] **Step 5: Add the gate/gradient inspection probe (spec §4.2 rows 3-4)**

Append to `MMTSFM/scripts/probes/localize.py`:

```python
def gate_stats(ckpt_path: str) -> dict:
    """Fusion-gate and modality-bias statistics from a trained checkpoint.

    Tests C4 (modality laziness). Two readings, both against a known-good
    contrast: modality_pair_bias was EXACTLY 0.0 in all 12 blocks through s1 and
    s2a, because that pathway only becomes active under interleaved fusion. A
    still-zero bias after interleaved training means the pathway received no
    usable gradient.

    Returns per-block: mean |modality_pair_bias|, and W_gate bias mean (a proxy
    for the resting alpha, since alpha = sigmoid(W_gate(u)) and a large positive
    bias pins alpha toward the numeric residual, closing the visual path).
    """
    import torch

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    blocks: dict[int, dict[str, float]] = {}
    for k, v in sd.items():
        if ".encoder.block." not in k:
            continue
        idx = int(k.split(".encoder.block.")[1].split(".")[0])
        e = blocks.setdefault(idx, {})
        if k.endswith("modality_pair_bias"):
            e["modality_pair_bias_absmean"] = float(v.float().abs().mean())
        elif k.endswith("layer.0.W_gate.bias"):
            e["w_gate_bias_mean"] = float(v.float().mean())
    return {
        "per_block": {str(i): blocks[i] for i in sorted(blocks)},
        "n_blocks_with_zero_modality_bias": sum(
            1 for e in blocks.values() if e.get("modality_pair_bias_absmean", 1.0) == 0.0
        ),
    }
```

Add the matching test to `MMTSFM/tests/test_localize.py`:

```python
def test_gate_stats_flags_an_untrained_modality_bias(tmp_path):
    import torch

    from scripts.probes.localize import gate_stats

    ckpt = tmp_path / "c.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.chronos.encoder.block.0.layer.0.modality_pair_bias": torch.zeros(4),
                "model.chronos.encoder.block.0.layer.0.W_gate.bias": torch.full((8,), 2.0),
                "model.chronos.encoder.block.1.layer.0.modality_pair_bias": torch.tensor(
                    [0.3, -0.2, 0.1, 0.0]
                ),
                "model.chronos.encoder.block.1.layer.0.W_gate.bias": torch.zeros(8),
            }
        },
        ckpt,
    )
    res = gate_stats(str(ckpt))
    assert res["n_blocks_with_zero_modality_bias"] == 1
    assert res["per_block"]["0"]["w_gate_bias_mean"] == 2.0
    assert res["per_block"]["1"]["modality_pair_bias_absmean"] > 0.0
```

Run: `cd MMTSFM && uv run pytest tests/test_localize.py -v`
Expected: 4 passed.

Gradient flow needs no new code — `_GRAD_GROUPS` already emits
`train/grad_norm/vision_adapter` and `train/grad_norm/latent_summarizer`. Read them from the
CSV logger of any G2 arm and compare against the aggregate `train/grad_norm`; a vision-path
norm orders of magnitude below the aggregate is the C4 signature.

- [ ] **Step 6: Commit**

```bash
git add MMTSFM/scripts/probes/localize.py MMTSFM/tests/test_localize.py
git commit -m "feat(probes): G1 horizon, stratification, and fusion-gate probes"
```

---

### Task 5: Capacity intervention — `n_visual_tokens_per_step`

**Files:**
- Modify: `MMTSFM/src/mmtsfm/models/vision/latent_summarizer.py`
- Modify: `MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py` (`interleave_sequences`, `VisionChronos2Config`)
- Test: `MMTSFM/tests/test_visual_tokens_per_step.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `VisionChronos2Config.n_visual_tokens_per_step: int = 1`; `interleave_sequences(ts_tokens, vis_tokens, n_vis, tokens_per_step: int = 1) -> tuple[Tensor, Tensor]` where `vis_tokens` is `[B, n_vis * tokens_per_step, d]`.

**Why not simply raise `n_vis`:** `validate_n_visual_context_steps` defines it as the number
of most-recent context *patches* given visual tokens. At `input_patch_size=16`, `n_vis=8` would
assert visual coverage over 128 TS steps = 64 hours while the clip spans 6, and RoPE would
encode those tokens as 64 h apart. This task widens the bottleneck without touching temporal
extent.

- [ ] **Step 1: Write the failing test**

```python
# MMTSFM/tests/test_visual_tokens_per_step.py
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from mmtsfm.models.chronos2.vision_chronos2 import interleave_sequences


def test_default_is_bit_identical_to_the_old_single_token_path():
    torch.manual_seed(0)
    ts = torch.randn(2, 10, 8)
    vis = torch.randn(2, 3, 8)
    a, ma = interleave_sequences(ts, vis, n_vis=3)
    b, mb = interleave_sequences(ts, vis, n_vis=3, tokens_per_step=1)
    assert torch.equal(a, b) and torch.equal(ma, mb)


def test_k_tokens_land_after_their_ts_token_and_are_marked_visual():
    ts = torch.zeros(1, 6, 4)
    ts[0, 4] = 1.0  # marker on the first refinement TS token
    vis = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1).expand(1, 4, 4)
    out, mod = interleave_sequences(ts, vis, n_vis=2, tokens_per_step=2)
    # 4 macro + 2 * (1 ts + 2 vis) = 10
    assert out.shape == (1, 10, 4)
    assert mod.shape == (1, 10)
    assert torch.equal(out[0, 4], torch.ones(4))          # ts_4 kept in place
    assert torch.allclose(out[0, 5], torch.full((4,), 1.0))  # its 2 visual tokens
    assert torch.allclose(out[0, 6], torch.full((4,), 2.0))
    assert mod[0].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]


def test_summarizer_emits_k_tokens_per_step():
    from mmtsfm.models.vision.latent_summarizer import LatentSummarizer

    s = LatentSummarizer(d_v=16, d_model=32, n_vis_steps=2, n_heads=4, dropout=0.0, tokens_per_step=3)
    out = s(video_tokens=torch.randn(2, 4, 9, 16), T_ts=2)
    assert out.shape == (2, 2 * 3, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MMTSFM && uv run pytest tests/test_visual_tokens_per_step.py -v`
Expected: FAIL — `interleave_sequences() got an unexpected keyword argument 'tokens_per_step'`

- [ ] **Step 3: Widen the summarizer**

In `latent_summarizer.py`, add `tokens_per_step: int = 1` to `__init__`, store
`self.tokens_per_step = tokens_per_step`, and allocate `n_vis_steps * tokens_per_step` latent
queries instead of `n_vis_steps`:

```python
        self.tokens_per_step = tokens_per_step
        self.latent_queries = nn.Parameter(
            torch.randn(1, n_vis_steps * tokens_per_step, d_model) * (d_model**-0.5)
        )
```

In `forward`, replace the query slice and the causal-mask width so each *group* of
`tokens_per_step` queries shares one TS step's causal frame limit:

```python
        k = self.tokens_per_step
        n_q = n_vis * k
        queries = self.latent_queries[:, :n_q, :].expand(B, -1, -1)
        queries = self.layer_norm_q(queries)
```

and after building `mask` for `n_vis` rows, expand it to `n_q` rows with
`mask = mask.repeat_interleave(k, dim=-2)`. Finally, when `n_vis > 0`, the null-token concat
uses `T_macro = T_ts - n_vis` unchanged, but `attn_out` now has `n_q` tokens, so return
`attn_out` directly when `T_ts == n_vis` (the interleaved path always calls with `T_ts=n_vis`).

- [ ] **Step 4: Widen the interleaver**

In `vision_chronos2.py`, replace `interleave_sequences` with:

```python
def interleave_sequences(
    ts_tokens: torch.Tensor,   # [B, T_ctx, d]
    vis_tokens: torch.Tensor,  # [B, n_vis * tokens_per_step, d]
    n_vis: int,
    tokens_per_step: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weave `tokens_per_step` visual tokens in after each refinement TS token.

    Builds: [ts_0..ts_{T_M-1}] || [ts_{T_M}, v_{T_M,0..k-1}, ts_{T_M+1}, ...]

    All k visual tokens for a step share that step's position id (see
    build_interleaved_position_ids), so widening capacity does not claim any
    additional temporal extent.

    Returns:
        interleaved:   [B, T_ctx + n_vis * tokens_per_step, d]
        modality_mask: [B, T_ctx + n_vis * tokens_per_step]  0=TS, 1=visual
    """
    B, T_ctx, d = ts_tokens.shape
    k = tokens_per_step
    T_M = T_ctx - n_vis

    macro = ts_tokens[:, :T_M, :]
    ts_refine = ts_tokens[:, T_M:, :].unsqueeze(2)            # [B, n_vis, 1, d]
    vis = vis_tokens.reshape(B, n_vis, k, d)                  # [B, n_vis, k, d]
    pairs = torch.cat([ts_refine, vis], dim=2)                # [B, n_vis, 1+k, d]
    refinement = pairs.reshape(B, n_vis * (1 + k), d)
    interleaved = torch.cat([macro, refinement], dim=1)

    device = ts_tokens.device
    modality_mask = torch.zeros(B, T_ctx + n_vis * k, dtype=torch.long, device=device)
    step_start = T_M + torch.arange(n_vis, device=device) * (1 + k)
    for j in range(k):
        modality_mask[:, step_start + 1 + j] = 1
    return interleaved, modality_mask
```

Update `build_interleaved_position_ids` to repeat each refinement id `1 + k` times, and the
caller at the interleaved branch to pass `tokens_per_step=self.vcfg.n_visual_tokens_per_step`.
Add `n_visual_tokens_per_step: int = 1` to `VisionChronos2Config`, and pass it into
`LatentSummarizer(...)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd MMTSFM && uv run pytest tests/test_visual_tokens_per_step.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the full suite — the default path must be untouched**

Run: `cd MMTSFM && uv run pytest -q`
Expected: all previously passing tests still pass (183 + the new ones). Any failure here means
the default `tokens_per_step=1` path changed behaviour, which would invalidate the factorial's
control cell.

- [ ] **Step 7: Commit**

```bash
git add MMTSFM/src/mmtsfm/models/vision/latent_summarizer.py MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py MMTSFM/tests/test_visual_tokens_per_step.py
git commit -m "feat(fusion): n_visual_tokens_per_step widens the visual bottleneck"
```

---

### Task 6: Forcing intervention — auxiliary clear-sky-index loss

**Files:**
- Modify: `MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py` (aux head + output field)
- Modify: `MMTSFM/src/mmtsfm/models/chronos2/lightning_module.py` (`__init__`, `_step`)
- Test: `MMTSFM/tests/test_visual_aux_loss.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `VisionChronos2LightningModule(..., visual_aux_loss_weight: float = 0.0)`; model output gains `visual_aux_pred: torch.Tensor | None` of shape `[B*N, horizon]`.

**Target derivation:** `csi` is not a covariate, so the aux target is derived inside the batch
as `Y_future / max(clearsky_ghi_future, eps)`. `clearsky_ghi` is `COV_COLS` index 13 and is
scaled by 1000.0 in `COV_SCALES`, so the scaled channel is already in `[0,1]`-ish units.

- [ ] **Step 1: Write the failing test**

```python
# MMTSFM/tests/test_visual_aux_loss.py
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from tests.test_training_loop import _make_batch, _make_module


def _mod(weight):
    return _make_module(visual_aux_loss_weight=weight)


def test_weight_zero_reproduces_current_loss_exactly():
    """The factorial's control cell must be a true control."""
    torch.manual_seed(0)
    a = _mod(0.0)
    torch.manual_seed(0)
    b = _make_module()  # no aux arg at all
    batch = _make_batch(bs=2, N=2, T=32, H=8)
    torch.manual_seed(1)
    la = a._step(batch, "train")
    torch.manual_seed(1)
    lb = b._step(batch, "train")
    la = la[0] if isinstance(la, tuple) else la
    lb = lb[0] if isinstance(lb, tuple) else lb
    assert torch.allclose(la, lb), (float(la), float(lb))


def test_positive_weight_changes_the_loss():
    torch.manual_seed(0)
    m = _mod(1.0)
    batch = _make_batch(bs=2, N=2, T=32, H=8)
    torch.manual_seed(1)
    with_aux = m._step(batch, "train")
    with_aux = with_aux[0] if isinstance(with_aux, tuple) else with_aux
    m.hparams.visual_aux_loss_weight = 0.0
    torch.manual_seed(1)
    without = m._step(batch, "train")
    without = without[0] if isinstance(without, tuple) else without
    assert not torch.allclose(with_aux, without)


def test_aux_loss_is_finite_and_backpropagates():
    torch.manual_seed(0)
    m = _mod(1.0)
    batch = _make_batch(bs=2, N=2, T=32, H=8)
    loss = m._step(batch, "train")
    loss = loss[0] if isinstance(loss, tuple) else loss
    assert torch.isfinite(loss)
    loss.backward()
    grads = [
        p.grad for n, p in m.named_parameters() if "visual_aux_head" in n and p.grad is not None
    ]
    assert grads, "aux head received no gradient"
    assert all(torch.isfinite(g).all() for g in grads)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd MMTSFM && uv run pytest tests/test_visual_aux_loss.py -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'visual_aux_loss_weight'`

- [ ] **Step 3: Add the head to the model**

In `VisionChronos2Model.__init__`, after `self.multimodal_embed = ...`:

```python
        # Auxiliary supervision for the visual pathway. Predicting clear-sky
        # INDEX (not power) is deliberate: a power target is satisfiable by
        # encoding time-of-day, which would not force any cloud information
        # into the visual tokens.
        # Built whenever the vision stack exists; whether it CONTRIBUTES is
        # governed by visual_aux_loss_weight (0.0 = off), so the control cell
        # keeps identical parameter count and initialisation draw order.
        self.visual_aux_head = (
            None
            if vision_config.skip_vision_stack
            else nn.Linear(d_model, vision_config.aux_horizon)
        )
```

Add `aux_horizon: int = 12` to `VisionChronos2Config`. In the interleaved branch, right after
`vis_summary` is computed, set `visual_aux_pred = self.visual_aux_head(vis_summary.mean(dim=1))`
and carry it on the output dataclass as `visual_aux_pred` (default `None`, so the late-fusion
and vision-off paths are unaffected).

- [ ] **Step 4: Add the loss term**

In `VisionChronos2LightningModule.__init__`, accept `visual_aux_loss_weight: float = 0.0`
(it lands in `self.hparams` via the existing `save_hyperparameters`). In `_step`, after the
finite-loss assertion:

```python
        aux_w = float(self.hparams.visual_aux_loss_weight)
        aux_pred = getattr(out, "visual_aux_pred", None)
        if aux_w > 0.0 and aux_pred is not None:
            # csi is not a covariate, so derive the target: normalized power over
            # clear-sky. COV_COLS index 13 is clearsky_ghi (scaled /1000).
            # inputs["future_covariates"] is covariate_channels[0] (temperature_2m),
            # NOT the covariate stack. The per-channel list is the right handle;
            # index 13 is clearsky_ghi, shape [BS*N, H].
            cs = inputs["covariate_channels"][13]
            y = inputs["future_target"]
            valid = cs > 0.05  # protocol daylight floor; below it csi is undefined
            if valid.any():
                target = torch.zeros_like(y)
                target[valid] = y[valid] / cs[valid].clamp(min=1e-3)
                aux = torch.nn.functional.l1_loss(
                    aux_pred[valid], target[valid].clamp(0.0, 2.0)
                )
                loss = loss + aux_w * aux
                self.log(f"{stage}/aux_loss", aux, on_step=(stage == "train"), on_epoch=True)
```

`covariate_channels` is returned by `_unpack_batch` as a list of `[BS*N, H]` tensors, one per
`COV_COLS` channel, already sliced to the future window — so no dataloader change is needed.
Assert `len(inputs["covariate_channels"]) == 14` once at the top of the branch; a different
length means `COV_COLS` changed and index 13 is no longer `clearsky_ghi`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd MMTSFM && uv run pytest tests/test_visual_aux_loss.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run the full suite**

Run: `cd MMTSFM && uv run pytest -q`
Expected: all green. `visual_aux_loss_weight=0.0` must leave every existing number unchanged.

- [ ] **Step 7: Commit**

```bash
git add MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py MMTSFM/src/mmtsfm/models/chronos2/lightning_module.py MMTSFM/tests/test_visual_aux_loss.py
git commit -m "feat(fusion): auxiliary clear-sky-index loss on the visual tokens"
```

---

### Task 7: G2 four-GPU factorial runner

**Files:**
- Create: `MMTSFM/scripts/g2_fusion_factorial.sbatch`

**Interfaces:**
- Consumes: `n_visual_tokens_per_step` (Task 5), `visual_aux_loss_weight` (Task 6).
- Produces: `baselines/results/mmtsfm_g2_{seed43,cap,force,capforce}_ukpv.json`.

- [ ] **Step 1: Write the runner**

```bash
#!/bin/bash
#SBATCH --job-name=mmtsfm_g2_factorial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_MTSFM
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
# 2x2 factorial (capacity x forcing) + seed control, one arm per GPU.
# s2b is the fourth cell and is already run. Pattern follows run_all_mmtsfm.sh.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f .env ]] && source .env
[[ -f ../.env ]] && source ../.env
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1 UV_NO_SYNC=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CK=/leonardo_scratch/fast/IscrC_MTSFM/checkpoints
RES=/leonardo/home/userexternal/mmorand1/thesis-with-context/baselines/results
PREV=${CK}/curriculum/uk_pv_s2a/best.ckpt

# The donor must still carry its encoder, or every arm silently trains against
# the pristine baseline (the bug that invalidated s3).
uv run python scripts/repair_vjepa_checkpoint.py --target "$PREV" --inspect | tee /dev/stderr | grep -q "already carries its encoder" \
  || { echo "FATAL: ${PREV} has no encoder weights — repair before running"; exit 1; }

NAMES=(seed43 cap force capforce)
EXTRA0=(seed=43)
EXTRA1=(+model.vision_cfg.n_visual_tokens_per_step=8)
EXTRA2=(+model.visual_aux_loss_weight=0.5)
EXTRA3=(+model.vision_cfg.n_visual_tokens_per_step=8 +model.visual_aux_loss_weight=0.5)

pids=()
for g in 0 1 2 3; do
  name=${NAMES[$g]}
  dir=${CK}/g2/${name}
  mkdir -p "$dir"
  eval "extra=(\"\${EXTRA${g}[@]}\")"
  log=logs/slurm/g2_${name}_${SLURM_JOB_ID}.log
  CUDA_VISIBLE_DEVICES="$g" uv run python -m mmtsfm.train \
    +stage=s2b model=vision_chronos2_grassmann data=ukpv \
    trainer=slurm trainer.devices=1 trainer.strategy=auto \
    trainer.max_epochs=20 "trainer.default_root_dir=${dir}" \
    data.batch_size=4 trainer.accumulate_grad_batches=4 data.num_workers=8 \
    "init_ckpt=${PREV}" \
    "data.vjepa_cache_dir=/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224" \
    model.vision_cfg.n_visual_context_steps=1 \
    "model.sp_reference_path=${RES}/smart_persistence_s2_ukpv.json" \
    +data.train_stride=12 \
    "model.results_dir=${RES}" "model.results_tag=mmtsfm_g2_${name}_ukpv" \
    'hydra.run.dir=logs/experiments/runs/${now:%Y-%m-%d_%H-%M-%S}_g2_'"${name}" \
    "${extra[@]}" >> "$log" 2>&1 &
  pids+=($!)
  echo "  gpu ${g}: ${name}  -> ${log}"
done

rc=0
for i in 0 1 2 3; do
  wait "${pids[$i]}" || { echo "ARM ${NAMES[$i]} FAILED"; rc=1; }
done
echo ">>> G2 factorial done (rc=${rc}). Results: ${RES}/mmtsfm_g2_*_ukpv.json"
exit $rc
```

- [ ] **Step 2: Dry-run the argument composition without a GPU**

Hydra composition errors (the `train=false` incident) surface before any GPU work. Check each
arm's overrides compose:

```bash
cd MMTSFM
for extra in "" "+model.vision_cfg.n_visual_tokens_per_step=8" "+model.visual_aux_loss_weight=0.5"; do
  uv run python -m mmtsfm.train +stage=s2b model=vision_chronos2_grassmann data=smoke \
    trainer=default trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1 \
    trainer.precision=32 +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 \
    +trainer.limit_test_batches=1 logger=csv data.batch_size=2 data.num_workers=0 \
    +trainer.default_root_dir=/tmp/g2dry model.results_dir=/tmp/g2dry/res \
    model.results_tag=dry hydra.run.dir=/tmp/g2dry/h $extra 2>&1 | tail -2
done
```

Expected: each ends with a test-metrics table, no `ConfigCompositionException`.

- [ ] **Step 3: Commit**

```bash
git add MMTSFM/scripts/g2_fusion_factorial.sbatch
git commit -m "feat(fusion): 4-GPU G2 factorial runner (capacity x forcing + seed control)"
```

---

## Execution order and gates

1. **Tasks 1–3, then submit G0.** Read `conditional` in `g0_ceiling_ukpv.json`. If every entry
   is below `cv_spread`, **stop** — the finding is C1/C2 and the spec's §7 branch applies. Do
   not build Tasks 5–7.
2. **Task 4, then G1** — only if G0 passed. Its output selects which G2 arms matter.
3. **Tasks 5–7, then submit G2** — only if G1 points at C3 or C4.

Success is judged against the seed-43 arm, not against zero: an intervention counts only if it
moves dNMAE beyond 2× the measured seed spread.
