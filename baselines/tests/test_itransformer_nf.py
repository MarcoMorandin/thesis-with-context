"""Guards for the tier-2 library arm (neuralforecast iTransformer).

Two jobs: (1) the usual shape/gradient smoke on the model wrapper, and (2) the
comparability contract — the training script must resolve the SAME windows and
recipe as the MMTSFM curriculum, or the arm stops being a control.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml

_BL = Path(__file__).resolve().parents[1]
if str(_BL) not in sys.path:
    sys.path.insert(0, str(_BL))
sys.path.insert(0, str(_BL / "scripts"))

from common import config  # noqa: E402

from tier2_lib.nf_itransformer import (  # noqa: E402
    QUANTILES,
    NFITransformer,
    pinball_loss,
)

# The library itself is only needed to BUILD a model; the comparability guards
# below read configs and run without it (`uv sync --group nf` installs it).
needs_nf = pytest.mark.skipif(
    importlib.util.find_spec("neuralforecast") is None
    or importlib.util.find_spec("pytorch_lightning") is None,
    reason="uv sync --group nf",
)

T, H, C, B = 96, 12, len(config.COV_COLS), 2
SMALL = dict(hidden_size=16, n_heads=2, e_layers=1, d_ff=32)


def _model(**kw):
    return NFITransformer(history=T, horizon=H, n_cov=C, **{**SMALL, **kw})


def _inputs():
    g = torch.Generator().manual_seed(0)
    return (
        torch.rand(B, T, generator=g),
        torch.rand(B, T + H, C, generator=g),
        torch.ones(B, T),
    )


@needs_nf
def test_forward_shape_is_quantiles_over_the_horizon():
    out = _model()(*_inputs())
    assert out.shape == (B, H, len(QUANTILES))


@needs_nf
def test_quantiles_are_monotone():
    out = _model()(*_inputs())
    assert torch.all(out[..., 1:] >= out[..., :-1] - 1e-6)


@needs_nf
def test_gradients_reach_the_library_model():
    y_hist, cov, mask = _inputs()
    model = _model()
    loss = pinball_loss(model(y_hist, cov, mask), torch.rand(B, H), torch.ones(B, H))
    loss.backward()
    grads = [p.grad for p in model.nf.parameters() if p.requires_grad]
    assert grads and any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads
    )


@needs_nf
def test_future_cov_mode_selects_the_forecast_window_rows():
    """future_cov='all' must expose the FUTURE covariate rows (MMTSFM parity)."""
    cov = torch.arange((T + H) * C, dtype=torch.float32).reshape(1, T + H, C)
    y = torch.zeros(1, T)
    futr = _model(future_cov="all").variates(y, cov)[..., 1:]
    hist = _model(future_cov="history").variates(y, cov)[..., 1:]
    assert torch.equal(futr[:, -1], cov[:, -1])  # last token = horizon end
    assert torch.equal(hist[:, -1], cov[:, T - 1])  # last token = forecast origin
    assert not torch.equal(futr, hist)


@needs_nf
def test_target_quantile_extraction_matches_the_nf_projector_layout():
    """NF flattens [B, H*Q, N] to [B, H, Q*N]; variate 0 is the ::N stride."""
    model = _model()
    q, n = len(QUANTILES), model.n_variates
    dec_out = torch.arange(H * q * n, dtype=torch.float32).reshape(1, H * q, n)
    y_pred = dec_out.reshape(1, H, -1)
    got = model._target_quantiles(y_pred, 1)
    want = torch.sort(dec_out[..., 0].reshape(1, H, q), dim=-1).values
    assert torch.equal(got, want)


# --- comparability with the MMTSFM curriculum --------------------------------


def _script_defaults() -> argparse.Namespace:
    import train_itransformer_nf as script

    old, sys.argv = sys.argv, ["train_itransformer_nf.py"]
    try:
        return script.parse_args()
    finally:
        sys.argv = old


def _ukpv_cfg() -> dict:
    return yaml.safe_load(
        (_BL.parent / "MMTSFM" / "configs" / "data" / "ukpv.yaml").read_text()
    )


def test_windows_match_the_mmtsfm_data_config():
    args, cfg = _script_defaults(), _ukpv_cfg()
    assert args.history_days == cfg["history_days"]
    assert args.horizon_hours == cfg["horizon_hours"]
    import train_itransformer_nf as script

    assert script.window_steps(args) == (672, cfg["horizon"])


def test_recipe_matches_the_mmtsfm_trainer_and_model_configs():
    args = _script_defaults()
    model_cfg = yaml.safe_load(
        (
            _BL.parent
            / "MMTSFM"
            / "configs"
            / "model"
            / "vision_chronos2_grassmann.yaml"
        ).read_text()
    )
    trainer_cfg = yaml.safe_load(
        (
            _BL.parent / "MMTSFM" / "configs" / "trainer" / "vision_chronos2.yaml"
        ).read_text()
    )
    assert (args.lr, args.weight_decay) == (model_cfg["lr"], model_cfg["weight_decay"])
    assert args.warmup_steps == model_cfg["warmup_steps"]
    assert args.precision == trainer_cfg["precision"]
    early = next(c for c in trainer_cfg["callbacks"] if "patience" in c)
    assert args.patience == early["patience"]
    assert args.early_stop_min_delta == early["min_delta"]
    # Effective batch and train stride of the curriculum runner.
    assert args.batch_size * args.accumulate == 16
    assert args.train_stride == 12
    assert args.future_cov == "all"


@needs_nf
def test_module_reports_the_protocol_quantiles():
    from tier2_lib.module import ITransformerNFModule

    module = ITransformerNFModule(history=T, horizon=H, n_cov=C, **SMALL)
    assert tuple(QUANTILES) == tuple(config.QUANTILE_LEVELS)
    assert module.model.n_quantiles == len(config.QUANTILE_LEVELS)
