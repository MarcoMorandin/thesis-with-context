"""Registry adapters wrapping the tier-2 torch models behind the Baseline API."""

from __future__ import annotations

import numpy as np
import torch

from common import config
from common.base import Baseline, Forecast, register
from common.windows import WindowDataset

from .models import MODEL_CLASSES
from .trainer import TrainerConfig, resolve_device, train_model


class TSLibBaseline(Baseline):
    tier = 2
    requires_fit = True
    model_name: str = ""

    def __init__(self, trainer: TrainerConfig | None = None,
                 seed: int | None = None, **model_kwargs):
        # `seed` shortcut so the multi-seed protocol (§4.5) can rebuild any
        # tier-2 model without constructing a TrainerConfig explicitly
        self.trainer_cfg = trainer or TrainerConfig(
            seed=seed if seed is not None else config.SEED
        )
        if seed is not None:
            self.trainer_cfg.seed = seed
        self.model_kwargs = model_kwargs
        self._model = None
        self._device = None

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        sample = train[0]
        model = MODEL_CLASSES[self.model_name](
            history=len(sample["y_hist"]),
            horizon=len(sample["y_future"]),
            n_cov=sample["cov"].shape[1],
            **self.model_kwargs,
        )
        self._model = train_model(model, train, val, self.trainer_cfg)
        self._device = resolve_device(self.trainer_cfg.device)

    def predict(self, batch: dict) -> Forecast:
        if self._model is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")
        self._model.eval()
        to = lambda k: torch.as_tensor(  # noqa: E731
            batch[k], dtype=torch.float32, device=self._device
        )
        with torch.no_grad():
            pred = self._model(to("y_hist"), to("cov"), to("mask_hist"))
        pred = pred.cpu().numpy().astype(np.float32)
        if self._model.output_quantiles:
            quantiles = np.clip(pred, 0.0, 1.0)
            median = quantiles[..., list(config.QUANTILE_LEVELS).index(0.5)]
            return Forecast(point=median, quantiles=quantiles)
        return Forecast(point=np.clip(pred, 0.0, 1.0))


@register
class MLPBaseline(TSLibBaseline):
    name = "mlp"
    model_name = "mlp"


@register
class DLinearBaseline(TSLibBaseline):
    name = "dlinear"
    model_name = "dlinear"


@register
class PatchTSTBaseline(TSLibBaseline):
    name = "patchtst"
    model_name = "patchtst"


@register
class ITransformerBaseline(TSLibBaseline):
    name = "itransformer"
    model_name = "itransformer"


@register
class TFTBaseline(TSLibBaseline):
    name = "tft"
    model_name = "tft"
    supports_quantiles = True
