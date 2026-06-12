"""CoRA-style covariate adaptation of a frozen TSFM (Tier 4, P0).

Re-implementation in the spirit of CoRA (arXiv:2510.12681): the TSFM
backbone stays completely frozen; a small trainable adapter injects
exogenous covariates by predicting a residual correction on top of the
backbone forecast. The adapter's output layer is zero-initialized, so
training starts exactly at the frozen-backbone forecast and can only
improve on it via covariate information.

This is the "the gain could come from covariates alone" rebuttal baseline
(§2.4): if PVTSFM ≤ CoRA-with-covariates, deep vision fusion is not needed.

Backbone forecasts are precomputed once (it is frozen), then the adapter
trains on cached tensors — no backbone passes inside the training loop.
"""

from __future__ import annotations

import copy

import numpy as np

from common import config
from common.base import Baseline, Forecast, build, register
from common.windows import WindowDataset


def _build_adapter(history: int, horizon: int, n_cov: int,
                   d_hidden: int, dropout: float):
    import torch
    from torch import nn

    class CoRAAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            in_dim = horizon + 2 * history + (history + horizon) * n_cov
            self.net = nn.Sequential(
                nn.Linear(in_dim, d_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_hidden, horizon),
            )
            # zero-init the output layer: adapter starts as the identity
            # on the backbone forecast
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, backbone_point, y_hist, mask_hist, cov):
            x = torch.cat(
                [backbone_point, y_hist, mask_hist, cov.flatten(1)], dim=1
            )
            return self.net(x)

    return CoRAAdapter()


@register
class CoRA(Baseline):
    name = "cora"
    tier = 4
    requires_fit = True

    def __init__(
        self,
        backbone: str = "chronos2_zs",
        backbone_kwargs: dict | None = None,
        d_hidden: int = 128,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        max_train_windows: int = 200_000,
        seed: int = config.SEED,
        device: str | None = None,
    ):
        self.backbone_name = backbone
        self.backbone_kwargs = backbone_kwargs or {}
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_train_windows = max_train_windows
        self.seed = seed
        self.device_name = device
        self.backbone: Baseline | None = None
        self._adapter = None
        self._device = None

    def _cache_split(self, dataset: WindowDataset, max_windows: int | None):
        """Precompute frozen-backbone forecasts and adapter inputs."""
        buf = {k: [] for k in
               ("backbone", "y_hist", "mask_hist", "cov", "y_future", "mask")}
        for batch in dataset.iter_batches(512):
            buf["backbone"].append(self.backbone.predict(batch).point)
            buf["y_hist"].append(batch["y_hist"])
            buf["mask_hist"].append(batch["mask_hist"])
            buf["cov"].append(batch["cov"])
            buf["y_future"].append(batch["y_future"])
            buf["mask"].append(batch["mask_future"] * batch["daylight_future"])
        arrays = {k: np.concatenate(v).astype(np.float32) for k, v in buf.items()}
        n = len(arrays["backbone"])
        if max_windows is not None and n > max_windows:
            pick = np.random.default_rng(self.seed).choice(
                n, max_windows, replace=False
            )
            arrays = {k: v[pick] for k, v in arrays.items()}
        return arrays

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from tslib.trainer import point_loss, resolve_device

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = resolve_device(self.device_name)

        self.backbone = build(self.backbone_name, **self.backbone_kwargs)
        if self.backbone.requires_fit:
            self.backbone.fit(train, val)
        self.supports_quantiles = self.backbone.supports_quantiles

        sample = train[0]
        self._adapter = _build_adapter(
            history=len(sample["y_hist"]),
            horizon=len(sample["y_future"]),
            n_cov=sample["cov"].shape[1],
            d_hidden=self.d_hidden,
            dropout=self.dropout,
        ).to(self._device)

        def loader(dataset: WindowDataset, max_windows, shuffle: bool):
            arrays = self._cache_split(dataset, max_windows)
            tensors = TensorDataset(*(torch.from_numpy(arrays[k]) for k in
                                      ("backbone", "y_hist", "mask_hist",
                                       "cov", "y_future", "mask")))
            generator = torch.Generator().manual_seed(self.seed)
            return DataLoader(tensors, batch_size=self.batch_size,
                              shuffle=shuffle,
                              generator=generator if shuffle else None)

        train_loader = loader(train, self.max_train_windows, shuffle=True)
        val_loader = loader(val, self.max_train_windows // 4, shuffle=False)

        optimizer = torch.optim.AdamW(self._adapter.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        best_loss, best_state, bad_epochs = float("inf"), None, 0
        for _ in range(self.epochs):
            self._adapter.train()
            for bb, yh, mh, cov, yf, mask in train_loader:
                bb, yh, mh, cov, yf, mask = (
                    t.to(self._device) for t in (bb, yh, mh, cov, yf, mask)
                )
                optimizer.zero_grad()
                pred = bb + self._adapter(bb, yh, mh, cov)
                loss = point_loss(pred, yf, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._adapter.parameters(), 1.0)
                optimizer.step()

            self._adapter.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for bb, yh, mh, cov, yf, mask in val_loader:
                    bb, yh, mh, cov, yf, mask = (
                        t.to(self._device) for t in (bb, yh, mh, cov, yf, mask)
                    )
                    pred = bb + self._adapter(bb, yh, mh, cov)
                    total += float(point_loss(pred, yf, mask))
                    count += 1
            val_loss = total / max(count, 1)
            if val_loss < best_loss - 1e-6:
                best_loss, bad_epochs = val_loss, 0
                best_state = copy.deepcopy(self._adapter.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self._adapter.load_state_dict(best_state)

    def predict(self, batch: dict) -> Forecast:
        import torch

        if self._adapter is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")
        backbone_fc = self.backbone.predict(batch)
        self._adapter.eval()
        to = lambda x: torch.as_tensor(  # noqa: E731
            x, dtype=torch.float32, device=self._device
        )
        with torch.no_grad():
            residual = self._adapter(
                to(backbone_fc.point), to(batch["y_hist"]),
                to(batch["mask_hist"]), to(batch["cov"]),
            ).cpu().numpy()
        point = np.clip(backbone_fc.point + residual, 0.0, 1.0).astype(np.float32)
        quantiles = None
        if backbone_fc.quantiles is not None:
            shift = (point - backbone_fc.point)[..., None]
            quantiles = np.clip(backbone_fc.quantiles + shift, 0.0, 1.0).astype(
                np.float32
            )
        return Forecast(point=point, quantiles=quantiles)
