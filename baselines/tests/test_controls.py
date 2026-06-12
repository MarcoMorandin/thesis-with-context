"""Control-harness tests (§6: permuted-but-aligned batches, §5 battery)."""

from __future__ import annotations

import numpy as np
import pytest

from common.base import build
from common.controls import (apply_control, shrink_history,
                             shuffle_along_axis, zero_covariates)
from common.runner import evaluate_model

from .conftest import make_frame, windows_for


@pytest.fixture(scope="module")
def batch():
    ds = windows_for(make_frame(n_sites=1, days=4, seed=11), stride=2)
    return ds.batch(list(range(16)))


def test_zero_covariates_only_touches_cov(batch):
    out = zero_covariates(batch)
    assert (out["cov"] == 0).all()
    for key in ("y_hist", "mask_hist", "y_future", "mask_future", "clearsky"):
        np.testing.assert_array_equal(out[key], batch[key])
    assert (batch["cov"] != 0).any()  # original untouched


def test_shrink_history_masks_early_steps(batch):
    t = batch["y_hist"].shape[1]
    out = shrink_history(batch, keep=8)
    assert out["y_hist"].shape == batch["y_hist"].shape  # shapes constant
    assert (out["mask_hist"][:, : t - 8] == 0).all()
    assert (out["y_hist"][:, : t - 8] == 0).all()
    np.testing.assert_array_equal(out["mask_hist"][:, t - 8:],
                                  batch["mask_hist"][:, t - 8:])
    np.testing.assert_array_equal(out["y_future"], batch["y_future"])


def test_shrink_history_rejects_bad_keep(batch):
    with pytest.raises(ValueError):
        shrink_history(batch, keep=0)
    with pytest.raises(ValueError):
        shrink_history(batch, keep=batch["y_hist"].shape[1] + 1)


def test_shuffle_along_axis_is_aligned_permutation():
    """The §6 control contract: shuffled but recoverable via the returned
    permutation (the A09 shuffled-frames harness relies on this)."""
    arr = np.arange(5 * 3 * 2, dtype=np.float64).reshape(5, 3, 2)
    shuffled, perm = shuffle_along_axis(arr, axis=1, seed=0)
    assert sorted(perm) == [0, 1, 2]
    np.testing.assert_array_equal(np.take(arr, perm, axis=1), shuffled)
    inverse = np.argsort(perm)
    np.testing.assert_array_equal(np.take(shuffled, inverse, axis=1), arr)


def test_apply_control_unknown_name(batch):
    with pytest.raises(KeyError):
        apply_control("definitely_not_a_control", batch)
    assert apply_control("none", batch) is batch


def test_low_history_degrades_persistence_gracefully():
    """§5 low-history: shrinking history must not crash evaluation and the
    forecast still satisfies the contract."""
    ds = windows_for(make_frame(n_sites=1, days=4, seed=13), stride=2)
    results = evaluate_model(
        build("persistence"), ds,
        transform=lambda b: shrink_history(b, keep=4),
    )
    assert np.isfinite(results["overall"]["nmae"])
