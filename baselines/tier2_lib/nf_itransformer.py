"""iTransformer from `neuralforecast` (Nixtla), adapted to the protocol window.

The library model is used unmodified: ``neuralforecast.models.iTransformer`` is
instantiated as a plain ``nn.Module`` and called with its own ``windows_batch``
contract. Only the plumbing around it lives here.

Two adaptations are needed and both are documented at their site:

1. **Covariates.** NF's iTransformer declares ``EXOGENOUS_HIST/FUTR = False``,
   so its exogenous API drops the 14 protocol covariates entirely. The paper's
   architecture, however, is *variates-as-tokens*: the reference implementation
   feeds covariates as extra tokens through ``DataEmbedding_inverted``. We do
   exactly that by handing NF an ``n_series`` of ``1 + C`` and stacking the
   covariates next to the target, which keeps the covariate modality identical
   to MMTSFM's instead of crippling the baseline.
2. **Quantile de-interleaving.** With a Q-quantile loss, NF sizes the projector
   at ``h * Q`` and the model returns ``[B, h*Q, N] -> reshape(B, h, Q*N)``,
   i.e. the flat axis is quantile-major (``j = q*N + n``). ``MQLoss.domain_map``
   reads it as variate-major (``j = n*Q + q``) — the two agree only at N == 1,
   which is the univariate case NF tests. We therefore never route the loss
   through ``domain_map``: the projector layout is read straight off the model
   source (see ``_target_quantiles``) and supervised by our own pinball loss.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from common import config

QUANTILES: tuple[float, ...] = config.QUANTILE_LEVELS


def pinball_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Masked multi-quantile loss. pred [B, H, Q], target/mask [B, H].

    Mask is ``mask_future`` alone — MMTSFM's Chronos-2 loss masks the same way,
    and daylight is applied at SCORING time, not in the loss.
    """
    q = torch.tensor(QUANTILES, device=pred.device, dtype=pred.dtype)
    err = target.unsqueeze(-1) - pred
    loss = torch.where(err >= 0, q * err, (q - 1.0) * err)
    denom = mask.sum().clamp(min=1.0) * len(QUANTILES)
    return (loss * mask.unsqueeze(-1)).sum() / denom


def build_nf_itransformer(
    history: int,
    horizon: int,
    n_variates: int,
    hidden_size: int = 128,
    n_heads: int = 8,
    e_layers: int = 2,
    d_ff: int = 512,
    dropout: float = 0.1,
    use_norm: bool = True,
    seed: int = config.SEED,
):
    """The library model, sized for our window. Never fitted by NF itself."""
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import iTransformer

    return iTransformer(
        h=horizon,
        input_size=history,
        n_series=n_variates,
        hidden_size=hidden_size,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=dropout,
        use_norm=use_norm,
        # Sizes the projector to h*Q. NF's own training loop is never entered
        # (no .fit()), so max_steps/batch_size here are inert bookkeeping.
        loss=MQLoss(quantiles=list(QUANTILES)),
        random_seed=seed,
        max_steps=1,
    )


class NFITransformer(nn.Module):
    """``forward(y_hist, cov, mask_hist) -> [B, H, Q]`` over the target variate.

    ``future_cov`` selects which slice of the (T+H) covariate window becomes the
    covariate variates:

    * ``"all"`` — the last T rows, so the tokens carry the *future* weather of
      the forecast window. This is the deployable-NWP assumption MMTSFM is
      trained under (``PVRecordDataset(future_cov="all")``); it is what makes
      the two arms modality-identical.
    * ``"history"`` — the first T rows, the library-standard reading. Use it to
      price how much of any gap is the future-weather channel.
    """

    def __init__(
        self,
        history: int,
        horizon: int,
        n_cov: int,
        future_cov: str = "all",
        **model_kwargs,
    ):
        super().__init__()
        if future_cov not in ("all", "history"):
            raise ValueError(f"unknown future_cov mode: {future_cov!r}")
        self.T, self.H, self.n_cov = int(history), int(horizon), int(n_cov)
        self.future_cov = future_cov
        self.n_variates = 1 + self.n_cov
        self.n_quantiles = len(QUANTILES)
        self.nf = build_nf_itransformer(
            history=self.T,
            horizon=self.H,
            n_variates=self.n_variates,
            **model_kwargs,
        )

    def variates(self, y_hist: Tensor, cov: Tensor) -> Tensor:
        """[B, T] target + [B, T+H, C] covariates -> [B, T, 1+C] variate tokens."""
        cov_slice = (
            cov[:, self.H :, :] if self.future_cov == "all" else cov[:, : self.T, :]
        )
        return torch.cat([y_hist.unsqueeze(-1), cov_slice], dim=-1)

    def _target_quantiles(self, y_pred: Tensor, batch: int) -> Tensor:
        """NF output [B, H, Q*N] -> [B, H, Q] for variate 0 (the target).

        Layout per the model source: the projector's ``[B, H*Q, N]`` output is
        reshaped to ``[B, H, -1]``, so the flat axis runs (quantile, variate).
        """
        q = y_pred.view(batch, self.H, self.n_quantiles, self.n_variates)[..., 0]
        # Monotone quantiles (same guard the in-repo TFT port applies).
        return torch.sort(q, dim=-1).values

    def forward(self, y_hist: Tensor, cov: Tensor, mask_hist: Tensor) -> Tensor:
        # mask_hist is unused by the library model: PVRecordDataset already
        # writes NaN history steps as 0, which is the imputation NF assumes.
        x = self.variates(y_hist, cov)
        y_pred = self.nf({"insample_y": x})
        return self._target_quantiles(y_pred, x.shape[0])
