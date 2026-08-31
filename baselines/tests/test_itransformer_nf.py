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

from common import config  # noqa: E402

from tier2.nf_itransformer import NFITransformer  # noqa: E402

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
def test_forward_shape_is_point_forecast_over_the_horizon():
    out = _model()(*_inputs())
    assert out.shape == (B, H)


@needs_nf
def test_loss_is_the_configured_library_loss():
    for loss_fn, cls_name in (("mae", "MAE"), ("mse", "MSE")):
        model = _model(loss_fn=loss_fn)
        assert model.nf.loss.__class__.__name__ == cls_name
        assert model.nf.loss.outputsize_multiplier == 1


@needs_nf
def test_gradients_reach_the_library_model():
    y_hist, cov, mask = _inputs()
    model = _model()
    pred = model(y_hist, cov, mask)
    loss = model.nf.loss(torch.rand(B, H), pred, mask=torch.ones(B, H))
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


# --- comparability with the MMTSFM curriculum --------------------------------


def _script_defaults() -> argparse.Namespace:
    from tier2 import train_itransformer_nf as script

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
    from tier2 import train_itransformer_nf as script

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
    assert args.loss == "mse"


@needs_nf
def test_module_uses_the_configured_point_loss():
    from tier2.module import ITransformerNFModule

    module = ITransformerNFModule(history=T, horizon=H, n_cov=C, loss_fn="mse", **SMALL)
    assert module.model.nf.loss.__class__.__name__ == "MSE"


@needs_nf
def test_default_loss_matches_the_papers_own_training_script():
    """thuml/iTransformer's exp_long_term_forecasting.py trains with nn.MSELoss()."""
    from tier2.module import ITransformerNFModule

    module = ITransformerNFModule(history=T, horizon=H, n_cov=C, **SMALL)
    assert module.model.nf.loss.__class__.__name__ == "MSE"
