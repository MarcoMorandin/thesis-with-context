"""TiRex baseline wrapper (Tier 3, P1).

Provides zero-shot forecasting using NX-AI's TiRex model based on xLSTM.
"""

from __future__ import annotations

import numpy as np

from common import config
from common.base import Baseline, Forecast, register


@register
class TiRexZS(Baseline):
    name = "tirex_zs"
    tier = 3
    requires_fit = False
    supports_quantiles = True

    def __init__(
        self,
        model_id: str = "NX-AI/TiRex",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device_name = device
        self._model = None
        self._device = None

    def predict(self, batch: dict) -> Forecast:
        import torch

        if self._model is None:
            self._device = torch.device(
                self.device_name if self.device_name else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            if self.model_id == "dummy":
                class DummyTiRex:
                    def forecast(self, context: torch.Tensor, prediction_length: int) -> tuple[torch.Tensor, torch.Tensor]:
                        n = context.shape[0]
                        quantiles = torch.zeros((n, prediction_length, 9), dtype=torch.float32)
                        for i in range(9):
                            quantiles[..., i] = float(i) / 9.0
                        mean = torch.zeros((n, prediction_length), dtype=torch.float32)
                        return quantiles, mean
                self._model = DummyTiRex()
            else:
                import tirex
                self._model = tirex.load_model(self.model_id)
                self._model = self._model.to(self._device)

        horizon = batch["y_future"].shape[1]
        context = torch.from_numpy(batch["y_hist"]).float()
        if self.model_id != "dummy":
            context = context.to(self._device)

        with torch.no_grad():
            quantiles, mean = self._model.forecast(context=context, prediction_length=horizon)

        quantiles_np = quantiles.cpu().numpy()
        median_idx = list(config.QUANTILE_LEVELS).index(0.5)
        median = quantiles_np[..., median_idx]

        point = np.clip(median, 0.0, 1.0)
        quantiles_np = np.clip(quantiles_np, 0.0, 1.0)

        return Forecast(point=point, quantiles=quantiles_np)
