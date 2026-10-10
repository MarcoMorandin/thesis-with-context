"""Shared loading for the manuscript diagnostic figures.

Predictions live in ``baselines/results/predictions`` as ``<stem>_<plant>_pred.npz``
with ``pred`` / ``true`` (N_windows, H) and, for runs exported after the protocol
freeze, an element-wise daylight ``mask``.

Legacy baselines carry no mask. Their window grid is byte-identical to the
MMTSFM runs (``true`` matches to 1e-4 on every plant), so the mask is borrowed
from ``MASK_DONOR``. See knowledge/baselines.md 234 for why some legacy JSONs
were scored against a different mask and are therefore not exactly reproducible
from these arrays.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PRED_DIR = REPO / "baselines" / "results" / "predictions"
FIG_DIR = REPO / "manuscript" / "figures"

MASK_DONOR = "mmtsfm_s2d_ukpv_s2d_s42"
REFERENCE = "smart_persistence"

# Ramp subset: |Δ power| over the window exceeding this fraction of capacity.
RAMP_THRESHOLD = 0.10

# Target history is not stored in the prediction files, so the trace figures
# rebuild the eval windows from the curated parquet. These must stay identical
# to the evaluation protocol (baselines/common/config.py, run_eval.py) or the
# reconstructed history will not line up with the stored predictions:
# uk_pv is 30-minute, so 14 days = 672 history steps and 6 hours = 12 horizon
# steps, with non-overlapping evaluation windows. ``make_traces`` asserts the
# alignment against the stored ``true`` arrays before plotting.
DATA_PARQUET = Path(
    os.environ.get("THESIS_PARQUET", "/Volumes/dataset/dataset/dataset_all.parquet")
)
HISTORY_DAYS = 14.0
HORIZON_HOURS = 6.0
EVAL_STRIDE = 12


def plants() -> list[str]:
    found = {
        re.search(r"_(\d+)_pred\.npz$", p.name).group(1)
        for p in PRED_DIR.glob(f"{MASK_DONOR}_*_pred.npz")
    }
    return sorted(found)


def _path(stem: str, plant: str, vision_off: bool = False) -> Path:
    suffix = "_pred_off.npz" if vision_off else "_pred.npz"
    return PRED_DIR / f"{stem}_{plant}{suffix}"


def available(stem: str) -> bool:
    return all(_path(stem, p).exists() for p in plants())


def load_plant(stem: str, plant: str, vision_off: bool = False):
    """Return (pred, true, mask) for one plant, mask borrowed when absent."""
    z = np.load(_path(stem, plant, vision_off))
    pred, true = z["pred"], z["true"]
    if "mask" in z.files:
        mask = z["mask"]
    else:
        donor = np.load(_path(MASK_DONOR, plant))
        if donor["pred"].shape != pred.shape:
            raise ValueError(
                f"{stem}/{plant}: grid {pred.shape} != donor {donor['pred'].shape}; "
                "re-export through baselines/scripts/import_predictions.py --ukpv_dir"
            )
        mask = donor["mask"]
    return pred, true, mask.astype(bool)


def load_all(stem: str, vision_off: bool = False):
    """Concatenate every plant into flat daylight-only (pred, true) vectors."""
    ps, ts = [], []
    for plant in plants():
        pred, true, mask = load_plant(stem, plant, vision_off)
        ps.append(pred[mask])
        ts.append(true[mask])
    return np.concatenate(ps), np.concatenate(ts)


def scores(stem: str) -> dict[str, float]:
    """Macro-averaged NMAE and skill score vs smart persistence, daylight only."""
    nmae, skill = [], []
    for plant in plants():
        pred, true, mask = load_plant(stem, plant)
        ref_pred, ref_true, _ = load_plant(REFERENCE, plant)
        err = np.abs(pred - true)[mask].mean()
        ref = np.abs(ref_pred - ref_true)[mask].mean()
        nmae.append(err)
        skill.append(1.0 - err / ref)
    return {"nmae": float(np.mean(nmae)), "skill": 100.0 * float(np.mean(skill))}


@functools.lru_cache(maxsize=4)
def _eval_windows(plant: str):
    """Rebuild this plant's evaluation windows from the curated parquet."""
    import sys

    import pandas as pd

    if not DATA_PARQUET.exists():
        raise FileNotFoundError(
            f"{DATA_PARQUET} not found; attach the dataset volume or set "
            "THESIS_PARQUET to a copy of dataset_all.parquet"
        )
    sys.path.insert(0, str(REPO / "baselines"))
    from common.windows import dataset_for_sites

    df = pd.read_parquet(DATA_PARQUET)
    df = df[df["dataset"].astype(str) == "uk_pv"]
    return dataset_for_sites(
        df,
        {plant},
        time_range=None,
        stride=EVAL_STRIDE,
        history_days=HISTORY_DAYS,
        horizon_hours=HORIZON_HOURS,
        future_cov="deterministic",
    )


def history(plant: str, window: int, n_steps: int):
    """Return the last ``n_steps`` observed values before the forecast origin.

    Missing history steps come back as NaN so they break the drawn line rather
    than being silently imputed to zero.
    """
    item = _eval_windows(plant)[window]
    y = np.asarray(item["y_hist"], dtype=float)[-n_steps:]
    valid = np.asarray(item["mask_hist"], dtype=bool)[-n_steps:]
    return np.where(valid, y, np.nan)


def check_alignment(plant: str) -> None:
    """Fail loudly if rebuilt windows do not match the stored predictions."""
    ds = _eval_windows(plant)
    _, true, _ = load_plant(MASK_DONOR, plant)
    if len(ds) != len(true):
        raise ValueError(
            f"{plant}: rebuilt {len(ds)} windows but predictions hold "
            f"{len(true)}; window spec no longer matches the eval protocol"
        )
    rebuilt = np.stack([ds[i]["y_future"] for i in range(len(ds))])
    if not np.allclose(rebuilt, true, atol=1e-6):
        raise ValueError(
            f"{plant}: rebuilt y_future does not match stored true "
            f"(max |Δ| = {np.abs(rebuilt - true).max():.3g})"
        )
