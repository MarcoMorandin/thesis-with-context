"""Eval-time control transforms for the robustness battery (§5).

Each transform takes a batch dict (see WindowDataset) and returns a new,
shape-consistent batch — models are evaluated unchanged, only their inputs
are manipulated. Numerical controls implemented here:

* ``zero_covariates``     — covariate-ablation row of §5
* ``shrink_history``      — low-history regime, T ∈ {4, 8, 12, 24}
* ``shuffle_along_axis``  — generic aligned permutation; the shuffled /
  mismatched-frame controls (A09/A10) reuse it on `V` once the multimodal
  batches carry frames (MMTSFM side)

``CONTROLS`` maps CLI names to transforms for run_eval --control.
"""

from __future__ import annotations

from functools import partial

import numpy as np


def zero_covariates(batch: dict) -> dict:
    """Quantifies covariate vs vision contribution separately (§5)."""
    out = dict(batch)
    out["cov"] = np.zeros_like(batch["cov"])
    return out


def shrink_history(batch: dict, keep: int) -> dict:
    """Mask out all but the last ``keep`` history steps ("deployable on a
    new plant" claim — §5 low-history regime).

    Implemented by masking rather than slicing so tensor shapes stay
    constant: fixed-input trained models (MLP, PatchTST, …) and variable-
    context ZS models see the identical control. Deterministic covariates
    and clear-sky stay available (they are known without plant history)."""
    t = batch["y_hist"].shape[1]
    if not 0 < keep <= t:
        raise ValueError(f"keep={keep} outside (0, {t}]")
    out = dict(batch)
    cut = t - keep
    mask = batch["mask_hist"].copy()
    mask[:, :cut] = 0.0
    out["mask_hist"] = mask
    y = batch["y_hist"].copy()
    y[:, :cut] = 0.0
    out["y_hist"] = y
    return out


def shuffle_along_axis(
    arr: np.ndarray, axis: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random permutation along one axis; returns (shuffled, permutation).

    The permutation is returned so tests can verify the control is
    permuted-but-aligned (§6, test_controls)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(arr.shape[axis])
    return np.take(arr, perm, axis=axis), perm


CONTROLS = {
    "zero_cov": zero_covariates,
    "low_history_4": partial(shrink_history, keep=4),
    "low_history_8": partial(shrink_history, keep=8),
    "low_history_12": partial(shrink_history, keep=12),
}


def apply_control(name: str | None, batch: dict) -> dict:
    if name is None or name == "none":
        return batch
    if name not in CONTROLS:
        raise KeyError(f"unknown control {name!r}; known: {sorted(CONTROLS)}")
    return CONTROLS[name](batch)
