"""Model-agnostic evaluator for time-series forecasting.

Usage
-----
Build a DataLoader whose batches are dicts with keys:

    y_context : (batch, T)  — historical context fed to the model
    y_target  : (batch, H)  — ground-truth forecast horizon

Wrap any forecasting model in a ``predict_fn`` with signature::

    predict_fn(y_context: Tensor) -> Forecast

then call::

    results = evaluate(predict_fn, dataloader, config)

For Chronos-2 use the provided ``wrap_chronos2`` adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from eval.metrics import crps, mae, mase, mse, smape


@dataclass
class Forecast:
    """Container for one batch of forecasts."""

    mean: Tensor                          # (batch, H) — point / median forecast
    quantiles: Tensor | None = None       # (batch, H, Q)
    quantile_levels: list[float] | None = None  # len Q, values in (0, 1)


@dataclass
class EvalConfig:
    horizon: int
    quantile_levels: list[float] | None = None  # required for CRPS


def evaluate(
    predict_fn: Callable[[Tensor], Forecast],
    dataloader: DataLoader,
    config: EvalConfig,
) -> dict[str, float]:
    """Evaluate a forecast model over a DataLoader.

    Parameters
    ----------
    predict_fn  callable that accepts ``y_context (batch, T)`` and returns a
                ``Forecast``.  Must be idempotent (no internal state mutation).
    dataloader  yields dicts with ``y_context`` (batch, T) and ``y_target`` (batch, H).
    config      EvalConfig specifying horizon and optional quantile levels.

    Returns
    -------
    dict with keys ``mse``, ``mae``, ``mase``, ``smape``, and (if quantiles are
    present) ``crps``.  All values are Python floats averaged over the dataset.
    """
    totals: dict[str, float] = {k: 0.0 for k in ("mse", "mae", "mase", "smape")}
    crps_total = 0.0
    has_crps = False
    n_batches = 0

    for batch in dataloader:
        y_context: Tensor = batch["y_context"]   # (batch, T)
        y_target: Tensor = batch["y_target"]     # (batch, H)

        forecast: Forecast = predict_fn(y_context)
        y_pred = forecast.mean                   # (batch, H)

        totals["mse"]   += mse(y_target, y_pred).item()
        totals["mae"]   += mae(y_target, y_pred).item()
        totals["mase"]  += mase(y_target, y_pred, y_context).item()
        totals["smape"] += smape(y_target, y_pred).item()

        if forecast.quantiles is not None and forecast.quantile_levels is not None:
            q_levels = torch.tensor(
                forecast.quantile_levels, dtype=y_target.dtype, device=y_target.device
            )
            crps_total += crps(y_target, forecast.quantiles, q_levels).item()
            has_crps = True

        n_batches += 1

    if n_batches == 0:
        raise ValueError("DataLoader was empty — no batches to evaluate.")

    results = {k: v / n_batches for k, v in totals.items()}
    if has_crps:
        results["crps"] = crps_total / n_batches

    return results


# ---------------------------------------------------------------------------
# Chronos-2 adapter
# ---------------------------------------------------------------------------

def wrap_chronos2(pipeline, prediction_length: int) -> Callable[[Tensor], Forecast]:
    """Adapt a ``Chronos2Pipeline`` to the ``predict_fn`` interface.

    The adapter feeds each batch item as a univariate series and collects the
    quantile + median forecasts returned by ``predict_quantiles``.

    Parameters
    ----------
    pipeline          a ``Chronos2Pipeline`` instance (already on the right device)
    prediction_length forecast horizon H
    """
    q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def predict_fn(y_context: Tensor) -> Forecast:
        # predict_quantiles accepts (batch, 1, T) or a list of (1, T) tensors.
        # We pass a 3-D tensor; pipeline handles left-padding internally.
        context = y_context.unsqueeze(1).float()  # (batch, 1, T)

        with torch.no_grad():
            quantiles_list, mean_list = pipeline.predict_quantiles(
                context,
                prediction_length=prediction_length,
                quantile_levels=q_levels,
            )

        # Each element: quantiles (1, H, Q), mean (1, H) — squeeze entity dim
        q_tensor = torch.stack([q.squeeze(0) for q in quantiles_list])   # (batch, H, Q)
        m_tensor = torch.stack([m.squeeze(0) for m in mean_list])         # (batch, H)

        return Forecast(mean=m_tensor, quantiles=q_tensor, quantile_levels=q_levels)

    return predict_fn
