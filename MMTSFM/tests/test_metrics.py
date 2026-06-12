"""Unit tests for eval.metrics and eval.evaluator.

Run with: uv run pytest
"""

import sys
sys.path.insert(0, "src")

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from eval.metrics import crps, mae, mase, mse, smape
from eval.evaluator import EvalConfig, Forecast, evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    torch.manual_seed(0)
    return torch


@pytest.fixture
def simple_series(rng):
    """Returns (y_true, y_pred) of shape (B=8, H=12)."""
    B, H = 8, 12
    y_true = torch.randn(B, H)
    y_pred = torch.randn(B, H)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

class TestMSE:
    def test_perfect_forecast_is_zero(self):
        y = torch.randn(4, 10)
        assert mse(y, y).item() == pytest.approx(0.0, abs=1e-6)

    def test_constant_forecast_equals_variance(self):
        """Forecasting with the global mean gives MSE = population variance."""
        torch.manual_seed(1)
        y = torch.randn(200)
        c = y.mean().expand_as(y)
        assert mse(y, c).item() == pytest.approx(y.var(unbiased=False).item(), rel=1e-5)

    def test_symmetric(self, simple_series):
        y, yhat = simple_series
        assert mse(y, yhat).item() == pytest.approx(mse(yhat, y).item(), rel=1e-5)

    def test_nonnegative(self, simple_series):
        y, yhat = simple_series
        assert mse(y, yhat).item() >= 0.0

    def test_scalar_output(self, simple_series):
        y, yhat = simple_series
        result = mse(y, yhat)
        assert result.shape == torch.Size([])


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class TestMAE:
    def test_perfect_forecast_is_zero(self):
        y = torch.randn(4, 10)
        assert mae(y, y).item() == pytest.approx(0.0, abs=1e-6)

    def test_nonnegative(self, simple_series):
        y, yhat = simple_series
        assert mae(y, yhat).item() >= 0.0

    def test_symmetric(self, simple_series):
        y, yhat = simple_series
        assert mae(y, yhat).item() == pytest.approx(mae(yhat, y).item(), rel=1e-5)

    def test_mse_geq_mae_squared(self, simple_series):
        """Jensen: MSE >= MAE^2 when both use the same arguments."""
        y, yhat = simple_series
        assert mse(y, yhat).item() >= mae(y, yhat).item() ** 2 - 1e-6


# ---------------------------------------------------------------------------
# MASE
# ---------------------------------------------------------------------------

class TestMASE:
    def test_perfect_forecast_is_zero(self):
        torch.manual_seed(2)
        y = torch.randn(5, 12)
        insample = torch.randn(5, 24)
        assert mase(y, y, insample).item() == pytest.approx(0.0, abs=1e-6)

    def test_known_exact_value(self):
        """MASE with a hand-crafted example where the answer is exactly known.

        insample = [0, 1, 2, 3]  → naive lag-1 errors = [1, 1, 1] → scale = 1.0
        y_true   = [2, 2]
        y_pred   = [0, 0]        → forecast errors = [2, 2]       → MAE = 2.0
        expected MASE = 2.0 / 1.0 = 2.0
        """
        y_true   = torch.tensor([[2.0, 2.0]])
        y_pred   = torch.tensor([[0.0, 0.0]])
        insample = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        assert mase(y_true, y_pred, insample).item() == pytest.approx(2.0, rel=1e-5)

    def test_positive(self, simple_series):
        y, yhat = simple_series
        insample = torch.randn(y.shape[0], 24)
        assert mase(y, yhat, insample).item() >= 0.0


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------

class TestSMAPE:
    def test_perfect_forecast_is_zero(self):
        y = torch.randn(4, 10).abs() + 1.0  # keep positive to avoid division issues
        assert smape(y, y).item() == pytest.approx(0.0, abs=1e-5)

    def test_range_is_zero_to_200(self):
        """sMAPE ∈ [0, 200] by definition."""
        y    = torch.ones(10, 12)
        yhat = torch.zeros(10, 12)
        val = smape(y, yhat).item()
        assert 0.0 <= val <= 200.0 + 1e-6

    def test_nonnegative(self, simple_series):
        y, yhat = simple_series
        assert smape(y, yhat).item() >= 0.0

    def test_returns_percent(self):
        y    = torch.tensor([[1.0, 2.0]])
        yhat = torch.tensor([[2.0, 4.0]])
        # |1-2|/(1+2)*2 = 2/3, |2-4|/(2+4)*2 = 2/3 → mean 2/3 → *100 ≈ 66.67
        assert smape(y, yhat).item() == pytest.approx(66.667, rel=1e-3)


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

class TestCRPS:
    def _uniform_quantile_preds(self, y: Tensor, n_q: int = 9) -> tuple[Tensor, Tensor]:
        """Return quantile forecasts equal to y at every quantile (perfect)."""
        q_levels = torch.linspace(0.1, 0.9, n_q)
        # shape: (..., H, Q) — all quantiles identical to y
        q_preds = y.unsqueeze(-1).expand(*y.shape, n_q).clone()
        return q_preds, q_levels

    def test_perfect_quantile_forecast_crps_near_zero(self):
        """When all quantile forecasts equal the truth, CRPS → 0."""
        torch.manual_seed(4)
        y = torch.randn(8, 12)
        q_preds, q_levels = self._uniform_quantile_preds(y)
        val = crps(y, q_preds, q_levels).item()
        assert val == pytest.approx(0.0, abs=1e-5)

    def test_nonnegative(self):
        torch.manual_seed(5)
        y      = torch.randn(8, 12)
        q_preds = torch.randn(8, 12, 9)
        q_levels = torch.linspace(0.1, 0.9, 9)
        assert crps(y, q_preds, q_levels).item() >= 0.0

    def test_worse_forecast_has_larger_crps(self):
        torch.manual_seed(6)
        y = torch.ones(8, 12)
        q_levels = torch.linspace(0.1, 0.9, 9)
        # Good: quantiles centred on the truth
        good  = y.unsqueeze(-1).expand(8, 12, 9) + torch.randn(8, 12, 9) * 0.1
        # Bad: quantiles far from the truth
        bad   = y.unsqueeze(-1).expand(8, 12, 9) + 5.0
        assert crps(y, good, q_levels).item() < crps(y, bad, q_levels).item()


# ---------------------------------------------------------------------------
# Evaluator — constant-forecast model
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _make_loader(self, B: int, T: int, H: int, seed: int = 7) -> DataLoader:
        torch.manual_seed(seed)
        context = torch.randn(B, T)
        target  = torch.randn(B, H)
        ds = TensorDataset(context, target)

        def collate(batch):
            ctx, tgt = zip(*batch)
            return {"y_context": torch.stack(ctx), "y_target": torch.stack(tgt)}

        return DataLoader(ds, batch_size=B, collate_fn=collate)

    def test_constant_forecast_mse_equals_variance(self):
        """Predicting the per-batch mean gives MSE = population variance of target."""
        torch.manual_seed(8)
        B, T, H = 32, 48, 12

        # Build a fixed target so we know the exact variance
        target = torch.randn(B, H)
        context = torch.randn(B, T)

        def collate(batch):
            ctx, tgt = zip(*batch)
            return {"y_context": torch.stack(ctx), "y_target": torch.stack(tgt)}

        loader = DataLoader(
            TensorDataset(context, target),
            batch_size=B,
            collate_fn=collate,
        )

        # Constant forecast: always predict the global mean of the target
        global_mean = target.mean()

        def const_predict_fn(y_ctx: Tensor) -> Forecast:
            return Forecast(mean=global_mean.expand_as(y_ctx[:, :H]))

        results = evaluate(const_predict_fn, loader, EvalConfig(horizon=H))

        expected_var = ((target - global_mean) ** 2).mean().item()
        assert results["mse"] == pytest.approx(expected_var, rel=1e-5)

    def test_evaluate_returns_all_metric_keys(self):
        loader = self._make_loader(16, 24, 8)

        def point_predict(y_ctx: Tensor) -> Forecast:
            return Forecast(mean=torch.zeros(y_ctx.shape[0], 8))

        results = evaluate(point_predict, loader, EvalConfig(horizon=8))
        assert set(results.keys()) >= {"mse", "mae", "mase", "smape"}

    def test_evaluate_with_quantiles_returns_crps(self):
        loader = self._make_loader(16, 24, 8)
        q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        def prob_predict(y_ctx: Tensor) -> Forecast:
            B = y_ctx.shape[0]
            return Forecast(
                mean=torch.zeros(B, 8),
                quantiles=torch.randn(B, 8, len(q_levels)),
                quantile_levels=q_levels,
            )

        results = evaluate(prob_predict, loader, EvalConfig(horizon=8, quantile_levels=q_levels))
        assert "crps" in results
        assert results["crps"] >= 0.0

    def test_evaluate_empty_loader_raises(self):
        def collate(batch):
            return {"y_context": torch.empty(0, 24), "y_target": torch.empty(0, 8)}

        loader = DataLoader(TensorDataset(torch.empty(0, 1)), batch_size=1, collate_fn=collate)

        with pytest.raises(ValueError, match="empty"):
            evaluate(lambda x: Forecast(mean=x[:, :8]), loader, EvalConfig(horizon=8))

    def test_metrics_are_finite(self):
        loader = self._make_loader(8, 16, 6)

        def predict(y_ctx: Tensor) -> Forecast:
            return Forecast(mean=torch.randn(y_ctx.shape[0], 6))

        results = evaluate(predict, loader, EvalConfig(horizon=6))
        for key, val in results.items():
            assert torch.isfinite(torch.tensor(val)), f"{key} is not finite"


# ---------------------------------------------------------------------------
# Chronos-2 zero-shot MSE
# ---------------------------------------------------------------------------

class TestChronos2ZeroShot:
    """Verify evaluate() produces a finite, positive MSE on Chronos-2 zero-shot.

    This test creates a Chronos-2 model from scratch (random weights) so it
    runs without downloading a pretrained checkpoint.
    """

    @pytest.fixture(scope="class")
    def pipeline(self):
        import sys
        sys.path.insert(0, "src")
        from mmtsfm.models.chronos2.config import Chronos2CoreConfig, Chronos2ForecastingConfig
        from mmtsfm.models.chronos2.model import Chronos2Model
        from mmtsfm.models.chronos2.pipeline import Chronos2Pipeline

        config = Chronos2CoreConfig(
            d_model=64,
            d_kv=16,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_grassmann=False,   # use standard attention for speed
            chronos_config={
                "context_length": 64,
                "input_patch_size": 16,
                "input_patch_stride": 16,
                "output_patch_size": 16,
                "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "use_reg_token": False,
                "use_arcsinh": False,
                "max_output_patches": 1,
            },
        )
        model = Chronos2Model(config)
        return Chronos2Pipeline(model)

    def _make_loader(self, B: int = 8, T: int = 48, H: int = 12) -> DataLoader:
        torch.manual_seed(42)
        context = torch.randn(B, T)
        target  = torch.randn(B, H)

        def collate(batch):
            ctx, tgt = zip(*batch)
            return {"y_context": torch.stack(ctx), "y_target": torch.stack(tgt)}

        return DataLoader(TensorDataset(context, target), batch_size=B, collate_fn=collate)

    def test_evaluate_returns_finite_mse(self, pipeline):
        from eval.evaluator import EvalConfig, evaluate, wrap_chronos2

        H = 12
        loader = self._make_loader(H=H)
        predict_fn = wrap_chronos2(pipeline, prediction_length=H)
        results = evaluate(predict_fn, loader, EvalConfig(horizon=H))

        assert "mse" in results
        assert torch.isfinite(torch.tensor(results["mse"]))
        assert results["mse"] > 0.0
        assert "crps" in results
        assert results["crps"] >= 0.0
