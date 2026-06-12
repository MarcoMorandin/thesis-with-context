"""LightGBM quantile baseline (Tier 1, P0).

One gradient-boosted model per quantile level (objective="quantile"),
median used as the point forecast — fills the full probabilistic table
(CRPS / pinball / coverage) per BASELINE_COMPARISON.md §4.3.
"""

from __future__ import annotations

import numpy as np

from common import config
from common.base import Baseline, Forecast, register
from common.windows import WindowDataset

from .features import build_features, training_table


@register
class LightGBMQuantile(Baseline):
    name = "lightgbm"
    tier = 1
    requires_fit = True
    supports_quantiles = True

    def __init__(
        self,
        num_leaves: int = 63,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_train_rows: int = 1_000_000,
        early_stopping_rounds: int = 50,
        seed: int = config.SEED,
    ):
        self.params = dict(
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=seed,
            verbose=-1,
        )
        self.max_train_rows = max_train_rows
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self._models: dict[float, object] = {}

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        import lightgbm as lgb

        x_train, y_train = training_table(train, self.max_train_rows, self.seed)
        x_val, y_val = training_table(val, self.max_train_rows // 4, self.seed)
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        for q in config.QUANTILE_LEVELS:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
            model.fit(
                x_train, y_train,
                eval_set=[(x_val, y_val)] if len(x_val) else None,
                callbacks=callbacks if len(x_val) else None,
            )
            self._models[q] = model

    def predict(self, batch: dict) -> Forecast:
        feats = build_features(batch)
        n, h, f = feats.shape
        flat = feats.reshape(n * h, f)
        quantiles = np.stack(
            [self._models[q].predict(flat) for q in config.QUANTILE_LEVELS], axis=-1
        ).reshape(n, h, -1)
        quantiles = np.clip(np.sort(quantiles, axis=-1), 0.0, 1.0)
        median = quantiles[..., list(config.QUANTILE_LEVELS).index(0.5)]
        return Forecast(
            point=median.astype(np.float32),
            quantiles=quantiles.astype(np.float32),
        )
