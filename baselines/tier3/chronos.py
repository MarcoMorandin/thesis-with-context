"""Chronos-2 baseline wrappers (Tier 3, P0/P1).

Provides zero-shot and fine-tuned configurations wrapping the internal
Chronos-2 implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from common import config
from common.base import Baseline, Forecast, register
from common.windows import WindowDataset


def _import_chronos():
    """Dynamically import Chronos-2 models from the MMTSFM source directory."""
    project_root = Path(__file__).resolve().parents[2]
    mmtsfm_src = project_root / "MMTSFM" / "src"
    if str(mmtsfm_src) not in sys.path:
        sys.path.insert(0, str(mmtsfm_src))
    from mmtsfm.models.chronos2.config import Chronos2CoreConfig
    from mmtsfm.models.chronos2.model import Chronos2Model
    from mmtsfm.models.chronos2.pipeline import Chronos2Pipeline
    return Chronos2CoreConfig, Chronos2Model, Chronos2Pipeline


@register
class Chronos2ZS(Baseline):
    name = "chronos2_zs"
    tier = 3
    requires_fit = False
    supports_quantiles = True

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device_name = device
        self._pipeline = None
        self._device = None

    def predict(self, batch: dict) -> Forecast:
        import torch

        if self._pipeline is None:
            self._device = torch.device(
                self.device_name if self.device_name else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            Chronos2CoreConfig, Chronos2Model, Chronos2Pipeline = _import_chronos()
            if self.model_id == "dummy":
                config_obj = Chronos2CoreConfig(
                    d_model=32,
                    num_layers=1,
                    num_heads=2,
                    use_grassmann=False,
                    chronos_config={
                        "context_length": 64,
                        "input_patch_size": 8,
                        "input_patch_stride": 8,
                        "output_patch_size": 8,
                        "quantiles": list(config.QUANTILE_LEVELS),
                        "use_reg_token": False,
                        "use_arcsinh": False,
                        "max_output_patches": 2,
                    }
                )
                model = Chronos2Model(config_obj)
            else:
                config_obj = Chronos2CoreConfig.from_pretrained(self.model_id)
                config_obj.use_grassmann = False
                model = Chronos2Model.from_pretrained(
                    self.model_id,
                    config=config_obj,
                    ignore_mismatched_sizes=True,
                )
                model.quantiles.data.copy_(
                    torch.tensor(config_obj.chronos_config["quantiles"], dtype=model.dtype)
                )

            self._pipeline = Chronos2Pipeline(model=model)
            self._pipeline.model.to(self._device)
            self._pipeline.model.eval()

        horizon = batch["y_future"].shape[1]
        # Chronos-2 treats NaN as missing; masked-out steps must not enter as 0s
        y_hist = np.where(batch["mask_hist"] > 0, batch["y_hist"], np.nan)
        y_ctx = torch.from_numpy(y_hist).float().unsqueeze(1)  # (N, 1, T)

        with torch.no_grad():
            quantiles_list, mean_list = self._pipeline.predict_quantiles(
                y_ctx,
                prediction_length=horizon,
                quantile_levels=list(config.QUANTILE_LEVELS),
            )

        q_tensor = torch.stack([q.squeeze(0) for q in quantiles_list]).cpu().numpy()  # (N, H, Q)
        m_tensor = torch.stack([m.squeeze(0) for m in mean_list]).cpu().numpy()      # (N, H)

        point = np.clip(m_tensor, 0.0, 1.0)
        quantiles = np.clip(q_tensor, 0.0, 1.0)

        return Forecast(point=point, quantiles=quantiles)


@register
class Chronos2FT(Baseline):
    name = "chronos2_ft"
    tier = 3
    requires_fit = True
    supports_quantiles = True

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 3,
        seed: int = config.SEED,
        device: str | None = None,
        num_workers: int = 0,
    ):
        self.model_id = model_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.device_name = device
        self.num_workers = num_workers
        self._model = None
        self._device = None

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        import torch
        from torch.utils.data import DataLoader
        import copy
        import math
        from tslib.trainer import TorchWindows

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._device = torch.device(
            self.device_name if self.device_name else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        Chronos2CoreConfig, Chronos2Model, _ = _import_chronos()
        if self.model_id == "dummy":
            config_obj = Chronos2CoreConfig(
                d_model=32,
                num_layers=1,
                num_heads=2,
                use_grassmann=False,
                chronos_config={
                    "context_length": 64,
                    "input_patch_size": 8,
                    "input_patch_stride": 8,
                    "output_patch_size": 8,
                    "quantiles": list(config.QUANTILE_LEVELS),
                    "use_reg_token": False,
                    "use_arcsinh": False,
                    "max_output_patches": 2,
                }
            )
            self._model = Chronos2Model(config_obj)
        else:
            config_obj = Chronos2CoreConfig.from_pretrained(self.model_id)
            config_obj.use_grassmann = False
            self._model = Chronos2Model.from_pretrained(
                self.model_id,
                config=config_obj,
                ignore_mismatched_sizes=True,
            )
            self._model.quantiles.data.copy_(
                torch.tensor(config_obj.chronos_config["quantiles"], dtype=self._model.dtype)
            )

        self._model.to(self._device)

        sample = train[0]
        horizon = len(sample["y_future"])
        output_patch_size = self._model.chronos_config.output_patch_size
        num_output_patches = max(1, math.ceil(horizon / output_patch_size))

        generator = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(
            TorchWindows(train), batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, generator=generator,
        )
        val_loader = DataLoader(
            TorchWindows(val), batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
        )

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss, best_state, bad_epochs = float("inf"), None, 0

        for epoch in range(self.epochs):
            self._model.train()
            for batch in train_loader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                loss_mask = batch["mask_future"] * batch["daylight_future"]

                optimizer.zero_grad()
                out = self._model(
                    context=batch["y_hist"],
                    context_mask=batch["mask_hist"],
                    future_target=batch["y_future"],
                    future_target_mask=loss_mask,
                    num_output_patches=num_output_patches,
                )
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            self._model.eval()
            val_loss_sum, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    loss_mask = batch["mask_future"] * batch["daylight_future"]
                    out = self._model(
                        context=batch["y_hist"],
                        context_mask=batch["mask_hist"],
                        future_target=batch["y_future"],
                        future_target_mask=loss_mask,
                        num_output_patches=num_output_patches,
                    )
                    val_loss_sum += float(out.loss.detach())
                    val_count += 1
            val_loss = val_loss_sum / max(val_count, 1)

            if val_loss < best_loss - 1e-6:
                best_loss, bad_epochs = val_loss, 0
                best_state = copy.deepcopy(self._model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

    def predict(self, batch: dict) -> Forecast:
        import torch
        import math

        if self._model is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")

        self._model.eval()
        horizon = batch["y_future"].shape[1]

        y_ctx = torch.from_numpy(batch["y_hist"]).float().to(self._device)
        mask_hist = torch.from_numpy(batch["mask_hist"]).float().to(self._device)

        output_patch_size = self._model.chronos_config.output_patch_size
        num_output_patches = max(1, math.ceil(horizon / output_patch_size))

        with torch.no_grad():
            out = self._model(
                context=y_ctx,
                context_mask=mask_hist,
                num_output_patches=num_output_patches,
            )
            q_preds = out.quantile_preds.permute(0, 2, 1).cpu().numpy()  # (B, H_padded, Q)
            q_preds = q_preds[:, :horizon, :]

        median_idx = list(config.QUANTILE_LEVELS).index(0.5)
        median = q_preds[..., median_idx]

        point = np.clip(median, 0.0, 1.0)
        quantiles = np.clip(q_preds, 0.0, 1.0)

        return Forecast(point=point, quantiles=quantiles)
