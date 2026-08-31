"""iTransformer from `neuralforecast` (Nixtla), adapted to the protocol window.

The library model is used unmodified: ``neuralforecast.models.iTransformer`` is
instantiated as a plain ``nn.Module`` and called with its own ``windows_batch``
contract. Only the plumbing around it lives here.

One adaptation is needed, and it is safe precisely because we never call
``NeuralForecast.fit()``/``.predict()``:

**Covariates.** NF's iTransformer declares ``EXOGENOUS_HIST/FUTR = False``, but
that gate lives in ``NeuralForecast.fit()``'s own exogenous-check/dataset
plumbing, not in the model's ``forward()``. The paper's architecture is
*variates-as-tokens*: ``DataEmbedding_inverted`` and the projector are applied
per-token and don't depend on variate count or identity. So handing the raw
model an ``n_series`` of ``1 + C`` and stacking covariates next to the target
is a legitimate use of the public ``forward()`` contract — it keeps the
covariate modality identical to MMTSFM's without touching anything private.

We also train with a point loss (default ``MSE``, matching
``experiments/exp_long_term_forecasting.py``'s ``nn.MSELoss()`` in the paper's
own ``thuml/iTransformer`` code; ``mae`` is offered too). With
``outputsize_multiplier == 1`` the projector is exactly ``nn.Linear(d_model,
pred_len)`` — the paper's own projector shape — and ``domain_map`` is the
identity, so the model's output is used as-is. A multivariate + quantile
combination (``MQLoss``) would not have this property: the model's own
flatten order and ``MQLoss.domain_map``'s assumed order disagree for N > 1.

Fidelity, checked directly against ``thuml/iTransformer`` source (not from
memory): the encoder-only inverted-embedding architecture, per-variate RevIN
normalization, and full self-attention over variate tokens all match exactly.
Supervising only the target variate (discarding the covariate tokens' own
"forecasts") is not an invention — the paper's own code has a built-in
``features == "MS"`` mode (``exp_long_term_forecasting.py``, ``f_dim = -1``)
that does the same thing, used in several of the paper's own benchmark rows.

One deliberate deviation remains, and should be named as such in anything
citing this as "the iTransformer model": the paper's non-target input
channels (plain multivariate, or ``MS``) are always **historical-only, same
window as the target** (see ``layers/Embed.py:DataEmbedding_inverted``, which
concatenates ``x_mark`` at the same length as ``x``). Our ``future_cov="all"``
instead shifts the covariate tokens to end at the horizon's end, injecting
real future (known-NWP) weather — needed for PV's deployable-forecast setting,
but not something the paper's multivariate/MS inputs ever do.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from common import config

_LOSSES = {"mae": "MAE", "mse": "MSE"}


def build_nf_itransformer(
    history: int,
    horizon: int,
    n_variates: int,
    loss_fn: str = "mse",
    hidden_size: int = 128,
    n_heads: int = 8,
    e_layers: int = 2,
    d_ff: int = 512,
    dropout: float = 0.1,
    use_norm: bool = True,
    seed: int = config.SEED,
):
    """The library model, sized for our window. Never fitted by NF itself."""
    from neuralforecast import losses as nf_losses
    from neuralforecast.models import iTransformer

    if loss_fn not in _LOSSES:
        raise ValueError(f"unknown loss_fn: {loss_fn!r}; choose one of {list(_LOSSES)}")
    loss = getattr(nf_losses.pytorch, _LOSSES[loss_fn])()

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
        loss=loss,
        # NF's own training loop is never entered (no .fit()), so max_steps
        # here is inert bookkeeping required only by the constructor.
        random_seed=seed,
        max_steps=1,
    )


class NFITransformer(nn.Module):
    """``forward(y_hist, cov, mask_hist) -> [B, H]`` point forecast of the target.

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
        loss_fn: str = "mse",
        **model_kwargs,
    ):
        super().__init__()
        if future_cov not in ("all", "history"):
            raise ValueError(f"unknown future_cov mode: {future_cov!r}")
        self.T, self.H, self.n_cov = int(history), int(horizon), int(n_cov)
        self.future_cov = future_cov
        self.n_variates = 1 + self.n_cov
        self.nf = build_nf_itransformer(
            history=self.T,
            horizon=self.H,
            n_variates=self.n_variates,
            loss_fn=loss_fn,
            **model_kwargs,
        )

    def variates(self, y_hist: Tensor, cov: Tensor) -> Tensor:
        """[B, T] target + [B, T+H, C] covariates -> [B, T, 1+C] variate tokens."""
        cov_slice = (
            cov[:, self.H :, :] if self.future_cov == "all" else cov[:, : self.T, :]
        )
        return torch.cat([y_hist.unsqueeze(-1), cov_slice], dim=-1)

    def forward(self, y_hist: Tensor, cov: Tensor, mask_hist: Tensor) -> Tensor:
        # mask_hist is unused by the library model: PVRecordDataset already
        # writes NaN history steps as 0, which is the imputation NF assumes.
        x = self.variates(y_hist, cov)
        y_pred = self.nf({"insample_y": x})  # [B, H, n_variates], mult=1
        return y_pred[..., 0]  # target variate
