"""Forecasting metrics.

All functions accept torch Tensors with a trailing horizon dimension H and
return a scalar (mean over all batch and horizon elements).

Shapes
------
y_true           : (..., H)
y_pred           : (..., H)   — point forecast
y_insample       : (..., T)   — historical context, used by MASE
quantile_preds   : (..., H, Q)
quantile_levels  : (Q,)       — values in (0, 1), e.g. [0.1, ..., 0.9]
"""

import torch
from torch import Tensor


def mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Mean Squared Error."""
    return ((y_true - y_pred) ** 2).mean()


def mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Mean Absolute Error."""
    return (y_true - y_pred).abs().mean()


def mase(y_true: Tensor, y_pred: Tensor, y_insample: Tensor, period: int = 1) -> Tensor:
    """Mean Absolute Scaled Error.

    Scale is the MAE of the naive lag-``period`` forecast on the in-sample
    series, which is the standard denominator from Hyndman & Koehler (2006).

    Parameters
    ----------
    y_true:     (..., H)  ground-truth forecast horizon
    y_pred:     (..., H)  point forecast
    y_insample: (..., T)  historical context (T >= period + 1)
    period:     seasonal period; 1 gives the non-seasonal naive benchmark
    """
    # Naive lag-period forecast error on in-sample data
    naive_errors = (y_insample[..., period:] - y_insample[..., :-period]).abs()
    scale = naive_errors.mean() + 1e-8
    return mae(y_true, y_pred) / scale


def smape(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Symmetric Mean Absolute Percentage Error (in %).

    sMAPE = 100 * mean( 2|y - ŷ| / (|y| + |ŷ|) )

    The denominator is floored at 1e-8 to guard against both y and ŷ being 0.
    """
    denominator = (y_true.abs() + y_pred.abs()).clamp(min=1e-8)
    return 100.0 * (2.0 * (y_true - y_pred).abs() / denominator).mean()


def crps(y_true: Tensor, quantile_preds: Tensor, quantile_levels: Tensor) -> Tensor:
    """Continuous Ranked Probability Score estimated from quantile forecasts.

    Uses the pinball-loss identity:
        CRPS ≈ 2 · mean_q [ L_q(y, F_q) ]
    where the pinball loss is
        L_q(y, F_q) = max( q·(y − F_q),  (q−1)·(y − F_q) )

    This estimator is exact when the quantile levels uniformly cover (0, 1).

    Parameters
    ----------
    y_true          : (..., H)
    quantile_preds  : (..., H, Q)  Q quantile forecasts per horizon step
    quantile_levels : (Q,)         quantile levels, e.g. torch.tensor([0.1,...,0.9])
    """
    # Broadcast y_true to (..., H, Q)
    y_expanded = y_true.unsqueeze(-1)  # (..., H, 1)
    q = quantile_levels.to(y_true.device).view(*([1] * y_true.dim()), -1)  # (..., 1, ..., Q)

    errors = y_expanded - quantile_preds          # (..., H, Q)
    pinball = torch.where(errors >= 0, q * errors, (q - 1) * errors)

    return 2.0 * pinball.mean()
