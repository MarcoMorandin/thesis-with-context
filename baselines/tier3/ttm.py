"""TTM-R3 baseline wrappers (Tier 3, P2).

Provides zero-shot and fine-tuned configurations wrapping IBM's TinyTimeMixer-R3 model.
"""

from __future__ import annotations

import numpy as np

from common import config
from common.base import Baseline, Forecast, register
from common.windows import WindowDataset


@register
class TTMR3ZS(Baseline):
    name = "ttm_zs"
    tier = 3
    requires_fit = False
    supports_quantiles = False

    def __init__(
        self,
        model_id: str = "ibm-research/ttm-r3",
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
                class DummyConfig:
                    context_length = 64
                    prediction_length = 96
                
                class DummyTTM(torch.nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.config = DummyConfig()

                    def forward(self, past_values: torch.Tensor, **kwargs) -> object:
                        class DummyOutput:
                            prediction_outputs = torch.zeros(past_values.shape[0], 96, 1, device=past_values.device)
                        return DummyOutput()

                self._model = DummyTTM()
            else:
                try:
                    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
                except ImportError:
                    from transformers import TinyTimeMixerForPrediction

                self._model = TinyTimeMixerForPrediction.from_pretrained(self.model_id)
                self._model = self._model.to(self._device)
                self._model.eval()

        horizon = batch["y_future"].shape[1]
        y_hist = torch.from_numpy(batch["y_hist"]).float().unsqueeze(-1)  # (B, T, 1)
        B, T, C = y_hist.shape
        expected_len = self._model.config.context_length

        # History (T=24) is far shorter than TTM's context (512). Zero-padding
        # injects a 0→signal step that corrupts TTM's internal instance-norm
        # (mean/std over a ~95%-zero window). Instead edge-pad with the earliest
        # real value and mark padded steps as unobserved, so the model's scaler
        # computes statistics over the real history only.
        if T < expected_len:
            pad_len = expected_len - T
            pad_val = y_hist[:, :1, :].expand(B, pad_len, C)
            past_values = torch.cat([pad_val, y_hist], dim=1)
            observed = torch.cat(
                [torch.zeros(B, pad_len, C, dtype=y_hist.dtype),
                 torch.ones(B, T, C, dtype=y_hist.dtype)], dim=1)
        else:
            past_values = y_hist[:, -expected_len:]
            observed = torch.ones(B, expected_len, C, dtype=y_hist.dtype)

        if self.model_id != "dummy":
            past_values = past_values.to(self._device)
            observed = observed.to(self._device)

        with torch.no_grad():
            outputs = self._model(past_values=past_values,
                                  past_observed_mask=observed)
            pred = outputs.prediction_outputs

        pred_np = pred[:, :horizon, 0].cpu().numpy()
        point = np.clip(pred_np, 0.0, 1.0)

        return Forecast(point=point)


@register
class TTMR3FT(Baseline):
    name = "ttm_ft"
    tier = 3
    requires_fit = True
    supports_quantiles = False

    def __init__(
        self,
        model_id: str = "ibm-research/ttm-r3",
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
        from tslib.trainer import TorchWindows, point_loss

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._device = torch.device(
            self.device_name if self.device_name else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.model_id == "dummy":
            class DummyConfig:
                context_length = 64
                prediction_length = 96

            class DummyTTM(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.config = DummyConfig()
                    self.weight = torch.nn.Parameter(torch.randn(1))

                def forward(self, past_values: torch.Tensor, **kwargs) -> object:
                    class DummyOutput:
                        prediction_outputs = past_values[:, :96, :] * self.weight
                    return DummyOutput()

            self._model = DummyTTM()
        else:
            try:
                from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
            except ImportError:
                from transformers import TinyTimeMixerForPrediction

            self._model = TinyTimeMixerForPrediction.from_pretrained(self.model_id)

        self._model.to(self._device)

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
        expected_len = self._model.config.context_length

        for epoch in range(self.epochs):
            self._model.train()
            for batch in train_loader:
                y_hist = batch["y_hist"].unsqueeze(-1)  # (B, T, 1)
                B, T, C = y_hist.shape
                if T < expected_len:
                    pad_len = expected_len - T
                    padding = torch.zeros(B, pad_len, C, dtype=y_hist.dtype)
                    past_values = torch.cat([padding, y_hist], dim=1)
                else:
                    past_values = y_hist[:, -expected_len:]

                past_values = past_values.to(self._device)
                y_future = batch["y_future"].to(self._device)
                mask = (batch["mask_future"] * batch["daylight_future"]).to(self._device)

                optimizer.zero_grad()
                outputs = self._model(past_values=past_values)
                pred = outputs.prediction_outputs[:, :y_future.shape[1], 0]

                loss = point_loss(pred, y_future, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            self._model.eval()
            val_loss_sum, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    y_hist = batch["y_hist"].unsqueeze(-1)
                    B, T, C = y_hist.shape
                    if T < expected_len:
                        pad_len = expected_len - T
                        padding = torch.zeros(B, pad_len, C, dtype=y_hist.dtype)
                        past_values = torch.cat([padding, y_hist], dim=1)
                    else:
                        past_values = y_hist[:, -expected_len:]

                    past_values = past_values.to(self._device)
                    y_future = batch["y_future"].to(self._device)
                    mask = (batch["mask_future"] * batch["daylight_future"]).to(self._device)

                    outputs = self._model(past_values=past_values)
                    pred = outputs.prediction_outputs[:, :y_future.shape[1], 0]

                    loss = point_loss(pred, y_future, mask)
                    val_loss_sum += float(loss.detach())
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

        if self._model is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")

        self._model.eval()
        horizon = batch["y_future"].shape[1]

        y_hist = torch.from_numpy(batch["y_hist"]).float().unsqueeze(-1)  # (B, T, 1)
        B, T, C = y_hist.shape
        expected_len = self._model.config.context_length

        if T < expected_len:
            pad_len = expected_len - T
            padding = torch.zeros(B, pad_len, C, dtype=y_hist.dtype)
            past_values = torch.cat([padding, y_hist], dim=1)
        else:
            past_values = y_hist[:, -expected_len:]

        past_values = past_values.to(self._device)

        with torch.no_grad():
            outputs = self._model(past_values=past_values)
            pred = outputs.prediction_outputs[:, :horizon, 0].cpu().numpy()

        point = np.clip(pred, 0.0, 1.0)
        return Forecast(point=point)
