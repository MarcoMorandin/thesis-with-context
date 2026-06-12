"""Evaluation metrics (BASELINE_COMPARISON.md §4.2-4.3), numpy.

All point metrics operate on capacity-normalized targets, so MAE/RMSE on
this scale *are* NMAE/NRMSE as defined in BASELINE_PROTOCOL.md §5. Every
function takes an element-wise ``mask`` (mask_future · daylight); masked-out
steps contribute nothing.

Per-plant macro-averaging (each plant weighs equally regardless of sample
count) is handled by ``PerPlantAccumulator``.
"""

from __future__ import annotations

import numpy as np

from . import config


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    total = mask.sum()
    if total == 0:
        return float("nan")
    return float((values * mask).sum() / total)


def nmae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return _masked_mean(np.abs(y_pred - y_true), mask)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(_masked_mean((y_pred - y_true) ** 2, mask)))


def skill_score(metric_model: float, metric_reference: float) -> float:
    """SS = 1 - metric_model / metric_reference (reference = Smart Persistence)."""
    if metric_reference == 0 or np.isnan(metric_reference):
        return float("nan")
    return 1.0 - metric_model / metric_reference


def pinball_loss(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    levels: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Mean pinball loss per quantile level.

    y_true (..., H), quantile_preds (..., H, Q), levels (Q,) → (Q,)
    """
    err = y_true[..., None] - quantile_preds
    q = np.asarray(levels, dtype=err.dtype)
    loss = np.where(err >= 0, q * err, (q - 1) * err)
    total = mask.sum()
    if total == 0:
        return np.full(len(q), np.nan)
    return (loss * mask[..., None]).sum(axis=tuple(range(loss.ndim - 1))) / total


def crps(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    levels: np.ndarray,
    mask: np.ndarray,
) -> float:
    """CRPS ≈ 2 · mean_q pinball (Chronos / GIFT-Eval convention).

    Matches MMTSFM/src/eval/metrics.py::crps.
    """
    return float(2.0 * np.nanmean(pinball_loss(y_true, quantile_preds, levels, mask)))


def empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Fraction of (masked) truths inside [lower, upper] — target 0.80 for q10-q90."""
    inside = ((y_true >= lower) & (y_true <= upper)).astype(np.float64)
    return _masked_mean(inside, mask)


def quantile_ece(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    levels: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean over q of |q − empirical P(y ≤ q-quantile)| (BASELINE_COMPARISON §4.3)."""
    total = mask.sum()
    if total == 0:
        return float("nan")
    below = (y_true[..., None] <= quantile_preds).astype(np.float64)
    emp = (below * mask[..., None]).sum(axis=tuple(range(below.ndim - 1))) / total
    return float(np.mean(np.abs(np.asarray(levels) - emp)))


class PerPlantAccumulator:
    """Accumulates error sums per plant; reports macro-averaged metrics.

    Point metrics are exact under accumulation. Probabilistic metrics
    accumulate the per-quantile pinball sums, from which CRPS is exact;
    coverage and ECE accumulate counts and are exact too.
    """

    def __init__(self, levels: tuple[float, ...] = config.QUANTILE_LEVELS):
        self.levels = np.asarray(levels)
        self._iq10 = list(levels).index(0.1) if 0.1 in levels else None
        self._iq90 = list(levels).index(0.9) if 0.9 in levels else None
        self._plants: dict[str, dict[str, np.ndarray | float]] = {}

    def _bucket(self, plant: str) -> dict:
        if plant not in self._plants:
            q = len(self.levels)
            self._plants[plant] = {
                "n": 0.0, "abs": 0.0, "sq": 0.0,
                "pinball": np.zeros(q), "below": np.zeros(q), "inside": 0.0,
                "has_q": False,
            }
        return self._plants[plant]

    def update(
        self,
        plants: np.ndarray,
        y_true: np.ndarray,           # (N, H)
        y_pred: np.ndarray,           # (N, H)
        mask: np.ndarray,             # (N, H) — mask_future · daylight
        quantile_preds: np.ndarray | None = None,  # (N, H, Q)
    ) -> None:
        for plant in np.unique(plants):
            rows = plants == plant
            yt, yp, m = y_true[rows], y_pred[rows], mask[rows]
            b = self._bucket(str(plant))
            b["n"] += m.sum()
            b["abs"] += (np.abs(yp - yt) * m).sum()
            b["sq"] += (((yp - yt) ** 2) * m).sum()
            if quantile_preds is not None:
                qp = quantile_preds[rows]
                err = yt[..., None] - qp
                loss = np.where(err >= 0, self.levels * err, (self.levels - 1) * err)
                b["pinball"] += (loss * m[..., None]).sum(axis=(0, 1))
                b["below"] += ((yt[..., None] <= qp) * m[..., None]).sum(axis=(0, 1))
                if self._iq10 is not None and self._iq90 is not None:
                    inside = (yt >= qp[..., self._iq10]) & (yt <= qp[..., self._iq90])
                    b["inside"] += (inside * m).sum()
                b["has_q"] = True

    def per_plant(self) -> dict[str, dict[str, float]]:
        out = {}
        for plant, b in self._plants.items():
            n = b["n"]
            if n == 0:
                continue
            row = {
                "nmae": float(b["abs"] / n),
                "nrmse": float(np.sqrt(b["sq"] / n)),
                "n_steps": float(n),
            }
            if b["has_q"]:
                row["crps"] = float(2.0 * np.mean(b["pinball"] / n))
                row["coverage_80"] = float(b["inside"] / n)
                row["quantile_ece"] = float(
                    np.mean(np.abs(self.levels - b["below"] / n))
                )
            out[plant] = row
        return out

    def macro(self) -> dict[str, float]:
        """Macro-average over plants (BASELINE_COMPARISON.md §4.2)."""
        rows = self.per_plant()
        if not rows:
            return {}
        keys = [k for k in next(iter(rows.values())) if k != "n_steps"]
        agg = {
            k: float(np.mean([r[k] for r in rows.values() if k in r])) for k in keys
        }
        agg["n_plants"] = float(len(rows))
        agg["n_steps"] = float(sum(r["n_steps"] for r in rows.values()))
        return agg
