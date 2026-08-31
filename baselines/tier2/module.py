"""LightningModule that trains NFITransformer on MMTSFM's exact recipe.

Everything outside the model is copied from
``mmtsfm.models.chronos2.lightning_module`` so a difference in the results can
only come from the model:

* loss    — the library's own point loss (``MAE``/``MSE``), called through its
            public masked signature ``loss(y, y_hat, mask=...)``; mask is
            ``mask_future`` alone (MMTSFM's Chronos-2 loss masks the same way;
            daylight is applied at SCORING time, not in the loss)
* optim   — AdamW, no weight decay on biases/norms, lr 1e-4 / wd 1e-2
* sched   — linear warmup (500 steps) then cosine to ``min_lr_ratio`` * lr,
            stepped per optimizer step
* test    — ``eval.protocol_eval.ProtocolEvaluator``, the same accumulator the
            MMTSFM test epoch uses (per-plant macro, ramp subset, SS vs Smart
            Persistence), writing the same results-JSON schema
"""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .nf_itransformer import NFITransformer

_NO_DECAY_KWS = ("bias", "norm", "layernorm", "embedding")


class ITransformerNFModule(pl.LightningModule):
    def __init__(
        self,
        history: int,
        horizon: int,
        n_cov: int,
        future_cov: str = "all",
        loss_fn: str = "mse",
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 500,
        min_lr_ratio: float = 0.1,
        seed: int = 42,
        results_dir: str = "results",
        results_tag: str = "itransformer_nf_s2_ukpv",
        sp_reference_path: str | None = None,
        data_path: str | None = None,
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = NFITransformer(
            history=history,
            horizon=horizon,
            n_cov=n_cov,
            future_cov=future_cov,
            loss_fn=loss_fn,
            seed=seed,
            **model_kwargs,
        )
        self._protocol_eval = None

    # -- batch (PVRecordDataset canonical dict, entity dim N == 1) -----------

    @staticmethod
    def _unpack(batch: dict) -> dict:
        return {
            "y_hist": batch["Y"][:, 0, :, 0],
            "mask_hist": batch["mask_target"][:, 0, :, 0],
            "cov": batch["X_cov"][:, 0],
            "y_future": batch["Y_future"][:, 0, :, 0],
            "mask_future": batch["mask_future"][:, 0, :, 0],
            "daylight_future": batch["daylight_future"][:, 0, :, 0],
        }

    def _step(self, batch: dict, stage: str) -> Tensor:
        b = self._unpack(batch)
        pred = self.model(b["y_hist"], b["cov"], b["mask_hist"])
        loss = self.model.nf.loss(b["y_future"], pred, mask=b["mask_future"])
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    # -- protocol test pass --------------------------------------------------

    def on_test_start(self):
        from eval.protocol_eval import ProtocolEvaluator

        self._protocol_eval = ProtocolEvaluator(
            horizon=self.hparams.horizon,
            reference_path=self.hparams.sp_reference_path,
        )

    def test_step(self, batch, batch_idx):
        b = self._unpack(batch)
        pred = self.model(b["y_hist"], b["cov"], b["mask_hist"]).detach().float()
        loss = self.model.nf.loss(b["y_future"], pred, mask=b["mask_future"])
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)

        y, mask = b["y_future"].float(), b["mask_future"].float()
        daylight = b["daylight_future"].float()
        median = pred
        site_ids = batch["site_id"]
        site_ids = [
            str(s)
            for s in (site_ids if isinstance(site_ids, (list, tuple)) else [site_ids])
        ]
        # S6 ramp inputs, same rule as the MMTSFM test step: |Δy| against the
        # previous step (step 0 uses the last history step), validity =
        # mask_future·daylight·prev_mask. Thresholding happens in finalize().
        prev = torch.cat([b["y_hist"][:, -1:].float(), y[:, :-1]], dim=1)
        prev_mask = torch.cat([b["mask_hist"][:, -1:].float(), mask[:, :-1]], dim=1)
        self._protocol_eval.update(
            site_ids=site_ids,
            y_true=y.cpu().numpy(),
            median=median.cpu().numpy(),
            mask=(mask * daylight).cpu().numpy(),
            quantiles=None,
            delta=(y - prev).abs().cpu().numpy(),
            delta_valid=(mask * daylight * prev_mask).cpu().numpy(),
        )

    def on_test_epoch_end(self):
        if self._protocol_eval is None or not self.trainer.is_global_zero:
            return
        overall = self._protocol_eval.finalize().get("overall", {})
        for k in ("nmae", "nrmse", "skill_score", "crps"):
            if k in overall:
                self.log(f"test/{k}", float(overall[k]), rank_zero_only=True)
        run_cfg = {
            "seed": self.hparams.seed,
            "model": "itransformer_nf",
            "library": "neuralforecast",
            "future_cov": self.hparams.future_cov,
            "loss": self.hparams.loss_fn,
        }
        write_kwargs = {}
        if self.hparams.data_path:
            # Provenance: hashes the parquet actually read, not the default path.
            write_kwargs["data_path"] = self.hparams.data_path
        path = self._protocol_eval.write(
            self.hparams.results_dir, self.hparams.results_tag, run_cfg, **write_kwargs
        )
        print(
            f"[protocol-eval] NMAE={overall.get('nmae'):.4f} "
            f"NRMSE={overall.get('nrmse'):.4f} "
            f"SS={overall.get('skill_score', float('nan')):.4f} → {path}",
            flush=True,
        )

    # -- optimiser (MMTSFM's schedule, single param group family) ------------

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (
                no_decay if any(k in name.lower() for k in _NO_DECAY_KWS) else decay
            ).append(p)
        groups = [
            {
                "params": decay,
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": no_decay, "lr": self.hparams.lr, "weight_decay": 0.0},
        ]
        optimizer = AdamW([g for g in groups if g["params"]])

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = self.hparams.warmup_steps
        min_ratio = self.hparams.min_lr_ratio

        def lr_schedule(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
