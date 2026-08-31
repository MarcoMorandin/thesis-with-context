"""TabPFN baseline (Tier 1) — tabular foundation-model counterpoint to TSFMs.

Uses the TabPFN-3 regressor (ModelVersion.V3, default in tabpfn>=8) on the same
feature table as LightGBM, with the training set subsampled to TabPFN's context
budget. TabPFN-3 is documented up to 1M rows x 200 features (TabPFNv2 was capped
at 10k x 500), so the context is set well above the v2 ceiling; it is still a
subsample of the 1M rows LightGBM sees, traded against per-predict inference
cost. Optional dependency: install with `uv sync --group tabpfn`. Local weight
download needs a one-time license (set ``TABPFN_TOKEN``; see
https://ux.priorlabs.ai). Quantiles come from TabPFN's native predictive
distribution ("quantiles" output).
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

    def __init__(self, max_context_rows: int = 100_000, seed: int = config.SEED):
        self.max_context_rows = max_context_rows
        self.seed = seed
        self._model = None

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        try:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "tabpfn not installed — run `uv sync --group tabpfn`"
            ) from err

        x, y = training_table(train, self.max_context_rows, self.seed)
        # TabPFN-3 is already the default in tabpfn>=8; pin ModelVersion.V3
        # explicitly so a future default bump cannot silently change the model.
        # ignore_pretraining_limits: the context can exceed the sizes TabPFN was
        # pretrained on; V3 is documented to 1M rows, but the guard fires earlier.
        self._model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3,
            random_state=self.seed,
            ignore_pretraining_limits=True,
        )
        self._model.fit(x, y)

    def predict(self, batch: dict) -> Forecast:
        feats = build_features(batch)
        n, h, f = feats.shape
        flat = feats.reshape(n * h, f)
        levels = list(config.QUANTILE_LEVELS)
        # Native predictive distribution: returns a list of (n*h,) arrays, one
        # per level. No point-prediction fallback on purpose — degrading to
        # zero-width intervals would corrupt pinball loss silently.
        quantiles = np.stack(
            self._model.predict(flat, output_type="quantiles", quantiles=levels),
            axis=-1,
        )
        quantiles = np.clip(np.sort(quantiles, axis=-1), 0.0, 1.0).reshape(n, h, -1)
        median = quantiles[..., levels.index(0.5)]
        return Forecast(
            point=median.astype(np.float32),
            quantiles=quantiles.astype(np.float32),
        )
