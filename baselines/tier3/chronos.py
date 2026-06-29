"""Chronos-2 baselines (Tier 3) on the official ``chronos-forecasting`` package.

Standalone: depends on the official ``chronos`` package (pip:
``chronos-forecasting``), not the vendored MMTSFM copy. Covariates are injected
through Chronos-2's native task interface (see ``build_chronos_inputs``).

Three variants:

  ``chronos2_zs``      zero-shot, leakage-free. Deterministic covariates (solar
                       geometry / calendar) are exposed as known-future inputs;
                       observed weather is past-only.
  ``chronos2_ft``      same covariate framing, fine-tuned on train-plant windows.
  ``chronos2_oracle``  perfect-foresight CEILING — every covariate (incl. true
                       future weather) is known into the future. NOT a
                       deployable forecaster; reports an upper bound on how much
                       headroom perfect covariate knowledge would buy. Requires
                       windows built with ``future_cov="all"`` so the batch
                       actually carries future weather.
"""

from __future__ import annotations

import numpy as np

from common import config
from common.base import Baseline, Forecast, register
from common.windows import WindowDataset

from .build_chronos_inputs import build_chronos_inputs

_DETERMINISTIC_IDX = config.DETERMINISTIC_COV_IDX
_ALL_IDX = tuple(range(len(config.COV_COLS)))


def _resolve_device(device_name: str | None) -> str:
    if device_name:
        return device_name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pipeline(model_id: str, device: str):
    """Load a Chronos-2 pipeline, or build a tiny untrained one for tests."""
    from chronos import Chronos2Pipeline

    if model_id == "dummy":
        from chronos.chronos2 import Chronos2CoreConfig, Chronos2Model

        cfg = Chronos2CoreConfig(
            d_model=32,
            num_layers=1,
            num_heads=2,
            chronos_config={
                "context_length": 64,
                "input_patch_size": 8,
                "input_patch_stride": 8,
                "output_patch_size": 8,
                "quantiles": list(config.QUANTILE_LEVELS),
                # reg token is required for the grouped covariate attention path
                "use_reg_token": True,
                "use_arcsinh": False,
                "max_output_patches": 8,
            },
        )
        pipeline = Chronos2Pipeline(model=Chronos2Model(cfg))
        pipeline.model.to(device)
        return pipeline

    return Chronos2Pipeline.from_pretrained(model_id, device_map=device)


def _forecast(pipeline, batch: dict, future_cov_idx: tuple[int, ...]) -> Forecast:
    """Run Chronos-2 quantile forecasting over one window batch."""
    import torch

    pipeline.model.eval()  # disable dropout → deterministic forecasts
    horizon = batch["y_future"].shape[1]
    inputs = build_chronos_inputs(batch, future_cov_idx, mode="predict")

    with torch.no_grad():
        quantiles_list, mean_list = pipeline.predict_quantiles(
            inputs,
            prediction_length=horizon,
            quantile_levels=list(config.QUANTILE_LEVELS),
        )

    # Each task is univariate: quantiles (1, H, Q), mean (1, H).
    quantiles = np.stack([q[0].cpu().numpy() for q in quantiles_list])  # (N, H, Q)
    point = np.stack([m[0].cpu().numpy() for m in mean_list])           # (N, H)
    # Enforce non-crossing quantiles (sort along the level axis).
    quantiles = np.sort(quantiles, axis=-1)
    return Forecast(
        point=np.clip(point, 0.0, 1.0),
        quantiles=np.clip(quantiles, 0.0, 1.0),
    )


def _resolve_future_cov_idx(future_cov: str, force_all: bool) -> tuple[int, ...]:
    """All covariate columns when future weather is treated as available
    (future_cov="all", the deployable NWP assumption) or for the oracle ceiling;
    otherwise only the leakage-free deterministic (solar geometry / calendar) set."""
    return _ALL_IDX if (force_all or future_cov == "all") else _DETERMINISTIC_IDX


@register
class Chronos2ZS(Baseline):
    name = "chronos2_zs"
    tier = 3
    requires_fit = False
    supports_quantiles = True
    _force_all_cov = False

    def __init__(self, model_id: str = "amazon/chronos-2", device: str | None = None,
                 future_cov: str = "deterministic"):
        self.model_id = model_id
        self.device_name = device
        self._pipeline = None
        self._future_cov_idx = _resolve_future_cov_idx(future_cov, self._force_all_cov)

    def _ensure_pipeline(self) -> None:
        if self._pipeline is None:
            self._pipeline = _load_pipeline(self.model_id, _resolve_device(self.device_name))

    def predict(self, batch: dict) -> Forecast:
        self._ensure_pipeline()
        return _forecast(self._pipeline, batch, self._future_cov_idx)


@register
class Chronos2Oracle(Chronos2ZS):
    """Perfect-foresight ceiling: all covariates known into the future.

    Not a deployable forecaster. Run only on windows built with
    ``future_cov="all"`` so the batch carries true future weather.
    """

    name = "chronos2_oracle"
    _force_all_cov = True


@register
class Chronos2FT(Baseline):
    name = "chronos2_ft"
    tier = 3
    requires_fit = True
    supports_quantiles = True
    _force_all_cov = False

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        num_steps: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        finetune_mode: str = "full",
        seed: int = config.SEED,
        device: str | None = None,
        future_cov: str = "deterministic",
    ):
        self.model_id = model_id
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.finetune_mode = finetune_mode
        self.seed = seed
        self.device_name = device
        self._pipeline = None
        self._future_cov_idx = _resolve_future_cov_idx(future_cov, self._force_all_cov)

    def _all_inputs(self, ds: WindowDataset, mode: str) -> list[dict]:
        rows: list[dict] = []
        for batch in ds.iter_batches(self.batch_size):
            rows.extend(build_chronos_inputs(batch, self._future_cov_idx, mode=mode))
        return rows

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        import tempfile

        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = _resolve_device(self.device_name)
        base = _load_pipeline(self.model_id, device)

        horizon = len(train[0]["y_future"])
        train_inputs = self._all_inputs(train, mode="fit")
        val_inputs = self._all_inputs(val, mode="fit") if len(val) else None

        with tempfile.TemporaryDirectory() as out_dir:
            self._pipeline = base.fit(
                train_inputs,
                prediction_length=horizon,
                validation_inputs=val_inputs,
                finetune_mode=self.finetune_mode,
                learning_rate=self.learning_rate,
                num_steps=self.num_steps,
                batch_size=self.batch_size,
                output_dir=out_dir,
                remove_printer_callback=True,
            )

    def predict(self, batch: dict) -> Forecast:
        if self._pipeline is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")
        return _forecast(self._pipeline, batch, self._future_cov_idx)


@register
class Chronos2OracleFT(Chronos2FT):
    """Fine-tuned perfect-foresight ceiling: all covariates known into the future.

    The fine-tuned counterpart to ``chronos2_oracle`` (and the ceiling for
    ``chronos2_ft``). Fine-tunes on, and forecasts with, every covariate known
    over the horizon (incl. true future weather). Not a deployable forecaster —
    an upper bound only. Run on windows built with ``future_cov="all"``.
    """

    name = "chronos2_oracle_ft"
    _force_all_cov = True
