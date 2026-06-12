"""TimesFM 2.5 baseline wrapper (Tier 3, P0).

Provides zero-shot forecasting using Google's TimesFM 2.5 model.
"""

from __future__ import annotations

import numpy as np

from common import config
from common.base import Baseline, Forecast, register


@register
class TimesFM25ZS(Baseline):
    name = "timesfm_zs"
    tier = 3
    requires_fit = False
    supports_quantiles = True

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device_name = device
        self._model = None

    def predict(self, batch: dict) -> Forecast:
        if self._model is None:
            if self.model_id == "dummy":
                class DummyTimesFM:
                    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
                        n = len(inputs)
                        point = np.zeros((n, horizon), dtype=np.float32)
                        # Mean + 9 quantiles = 10 elements in last dim
                        quantiles = np.zeros((n, horizon, 10), dtype=np.float32)
                        # Populate some non-zero values for test sanity
                        for i in range(10):
                            quantiles[..., i] = float(i) / 10.0
                        return point, quantiles
                self._model = DummyTimesFM()
            else:
                import timesfm

                if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
                    self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
                else:
                    self._model = timesfm.TimesFm.from_pretrained(self.model_id)

                # device placement is handled internally by timesfm at compile time
                self._model.compile(
                    timesfm.ForecastConfig(
                        max_context=1024,
                        max_horizon=256,
                        use_continuous_quantile_head=True,
                    )
                )

        horizon = batch["y_future"].shape[1]
        inputs = list(batch["y_hist"])

        point_forecast, quantile_forecast = self._model.forecast(horizon=horizon, inputs=inputs)

        # Extract the 9 quantiles (indices 1 to 9 in the last dimension)
        quantiles = quantile_forecast[:, :, 1:]
        median_idx = list(config.QUANTILE_LEVELS).index(0.5)
        median = quantiles[:, :, median_idx]

        point = np.clip(median, 0.0, 1.0)
        quantiles = np.clip(quantiles, 0.0, 1.0)

        return Forecast(point=point, quantiles=quantiles)
