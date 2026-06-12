"""Cross-dataset aggregation (BASELINE_COMPARISON.md §4.4, fev-bench style).

Never average raw metrics across datasets with different scales:
* win rate vs a reference model,
* geometric-mean skill (via the geometric mean of NRMSE ratios),
* average rank across datasets/scenarios.
"""

from __future__ import annotations

import numpy as np


def win_rate(
    model_metric: dict[str, float], reference_metric: dict[str, float]
) -> float:
    """Fraction of units (datasets or plants) where the model beats the
    reference (strictly lower metric). Keys must overlap."""
    common = sorted(set(model_metric) & set(reference_metric))
    if not common:
        return float("nan")
    wins = sum(model_metric[k] < reference_metric[k] for k in common)
    return wins / len(common)


def geometric_mean_skill(
    model_metric: dict[str, float], reference_metric: dict[str, float]
) -> float:
    """SS_geo = 1 − geomean_k(metric_model[k] / metric_ref[k]).

    Computed on the ratio (always positive) rather than on the skill scores
    themselves, which can be negative and have no geometric mean.
    """
    common = sorted(set(model_metric) & set(reference_metric))
    ratios = [
        model_metric[k] / reference_metric[k]
        for k in common
        if reference_metric[k] > 0 and np.isfinite(model_metric[k])
    ]
    if not ratios:
        return float("nan")
    return 1.0 - float(np.exp(np.mean(np.log(ratios))))


def average_rank(metric_by_model: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean rank of each model across units (datasets / scenarios).

    ``metric_by_model[model][unit] = metric`` (lower is better). Models
    missing a unit are excluded from that unit's ranking. Ties share the
    average rank.
    """
    units = sorted({u for m in metric_by_model.values() for u in m})
    ranks: dict[str, list[float]] = {m: [] for m in metric_by_model}
    for unit in units:
        entries = [
            (model, values[unit])
            for model, values in metric_by_model.items()
            if unit in values and np.isfinite(values[unit])
        ]
        entries.sort(key=lambda kv: kv[1])
        i = 0
        while i < len(entries):
            j = i
            while j + 1 < len(entries) and entries[j + 1][1] == entries[i][1]:
                j += 1
            shared = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
            for k in range(i, j + 1):
                ranks[entries[k][0]].append(shared)
            i = j + 1
    return {
        m: float(np.mean(r)) if r else float("nan") for m, r in ranks.items()
    }
