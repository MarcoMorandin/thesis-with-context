"""Convert a window batch into Chronos-2 covariate task dicts.

Chronos-2's covariate interface (official ``chronos-forecasting`` package) takes
a list of per-series tasks, each a dict with ``target`` / ``past_covariates`` /
``future_covariates``. The only knob that separates the leakage-free baselines
from the perfect-foresight oracle is *which* covariate columns are exposed over
the horizon (``future_cov_idx``):

  * leakage-free  -> ``config.DETERMINISTIC_COV_IDX`` (solar geometry / calendar,
                     computable for any future timestamp, no lookahead)
  * oracle        -> every column, incl. observed weather (a CEILING only)

``mode`` selects the framing:

  * ``"predict"`` -> target is the history only; the model forecasts forward.
                     Future covariates carry their actual horizon values.
  * ``"fit"``     -> target is the full history+horizon series (the HF trainer
                     samples context/label internally). Future-covariate values
                     are unused during fine-tuning, so the keys are present with
                     ``None`` values purely to flag which covariates are known
                     into the future; past-only covariates are masked over the
                     horizon by the trainer.
"""

from __future__ import annotations

import numpy as np

from common import config


def build_chronos_inputs(
    batch: dict,
    future_cov_idx: tuple[int, ...],
    *,
    mode: str = "predict",
) -> list[dict]:
    """One Chronos-2 task dict per window in ``batch``."""
    if mode not in ("predict", "fit"):
        raise ValueError(f"unknown mode: {mode!r}")

    names = config.COV_COLS
    cov = np.asarray(batch["cov"], dtype=np.float32)      # (N, T+H, C)
    n = batch["y_hist"].shape[0]
    t = batch["y_hist"].shape[1]
    fut_names = [names[c] for c in future_cov_idx]

    if mode == "predict":
        # NaN = missing, so masked-out history steps don't enter as real zeros.
        y = np.where(batch["mask_hist"] > 0, batch["y_hist"], np.nan).astype(np.float32)
        return [
            {
                "target": y[i],
                "past_covariates": {names[c]: cov[i, :t, c] for c in range(len(names))},
                "future_covariates": {names[c]: cov[i, t:, c] for c in future_cov_idx},
            }
            for i in range(n)
        ]

    # mode == "fit": full series target, future-cov values unused (None).
    y_hist = np.where(batch["mask_hist"] > 0, batch["y_hist"], np.nan)
    y_future = np.where(batch["mask_future"] > 0, batch["y_future"], np.nan)
    y_full = np.concatenate([y_hist, y_future], axis=1).astype(np.float32)  # (N, T+H)
    return [
        {
            "target": y_full[i],
            "past_covariates": {names[c]: cov[i, :, c] for c in range(len(names))},
            "future_covariates": {name: None for name in fut_names},
        }
        for i in range(n)
    ]
