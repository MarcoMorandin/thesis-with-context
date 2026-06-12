"""TabPFN baseline (Tier 1) — tabular foundation-model counterpoint to TSFMs.

Uses the TabPFN regressor on the same feature table as LightGBM, with the
training set subsampled to TabPFN's context budget. Optional dependency:
install with `uv sync --group tabpfn`. Quantiles come from TabPFN's native
predictive distribution ("quantiles" output) when available.
"""

from __future__ import annotations

import numpy as np

from common import config
from common.base import Baseline, Forecast, register
from common.windows import WindowDataset

from .features import build_features, training_table


@register
class TabPFNRegressorBaseline(Baseline):
    name = "tabpfn"
    tier = 1
    requires_fit = True
    supports_quantiles = True

    def __init__(self, max_context_rows: int = 10_000, seed: int = config.SEED):
        self.max_context_rows = max_context_rows
        self.seed = seed
        self._model = None

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "tabpfn not installed — run `uv sync --group tabpfn`"
            ) from err

        x, y = training_table(train, self.max_context_rows, self.seed)
        self._model = TabPFNRegressor(random_state=self.seed)
        self._model.fit(x, y)

    def predict(self, batch: dict) -> Forecast:
        feats = build_features(batch)
        n, h, f = feats.shape
        flat = feats.reshape(n * h, f)
        levels = list(config.QUANTILE_LEVELS)
        try:
            quantiles = np.stack(
                self._model.predict(flat, output_type="quantiles", quantiles=levels),
                axis=-1,
            )
        except TypeError:  # older API without quantile output
            point = self._model.predict(flat)
            quantiles = np.repeat(point[:, None], len(levels), axis=1)
        quantiles = np.clip(np.sort(quantiles, axis=-1), 0.0, 1.0).reshape(n, h, -1)
        median = quantiles[..., levels.index(0.5)]
        return Forecast(
            point=median.astype(np.float32),
            quantiles=quantiles.astype(np.float32),
        )
