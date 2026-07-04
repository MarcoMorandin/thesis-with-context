"""Unit tests for the protocol evaluator (NMAE/NRMSE/SS + results JSON)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _ref(path: Path, nrmse: float = 0.2):
    path.write_text(
        json.dumps(
            {
                "manifest": {},
                "results": {
                    "overall": {"nrmse": nrmse},
                    "per_plant": {"A": {"nrmse": nrmse}},
                },
            }
        )
    )


def test_perfect_forecast_skill_score_one(tmp_path):
    from eval.protocol_eval import ProtocolEvaluator

    ref = tmp_path / "smart_persistence.json"
    _ref(ref, nrmse=0.2)
    ev = ProtocolEvaluator(horizon=12, reference_path=str(ref))

    y = np.random.default_rng(0).uniform(0, 1, (5, 12))
    ev.update(
        site_ids=["A"] * 5,
        y_true=y,
        median=y.copy(),
        mask=np.ones_like(y),
        quantiles=np.repeat(y[..., None], 9, axis=-1),
    )
    res = ev.finalize()
    assert res["overall"]["nmae"] == 0.0
    assert res["overall"]["nrmse"] == 0.0
    assert abs(res["overall"]["skill_score"] - 1.0) < 1e-9


def test_metric_values_and_masking(tmp_path):
    from eval.protocol_eval import ProtocolEvaluator

    ref = tmp_path / "smart_persistence.json"
    _ref(ref, nrmse=0.5)
    ev = ProtocolEvaluator(horizon=4, reference_path=str(ref))

    y = np.zeros((1, 4))
    pred = np.array([[0.1, 0.1, 0.0, 0.0]])  # error 0.1 on first two steps
    mask = np.array([[1.0, 1.0, 0.0, 0.0]])  # only first two count
    ev.update(site_ids=["A"], y_true=y, median=pred, mask=mask)
    res = ev.finalize()
    assert abs(res["overall"]["nmae"] - 0.1) < 1e-9
    assert abs(res["overall"]["nrmse"] - 0.1) < 1e-9
    assert abs(res["overall"]["skill_score"] - (1 - 0.1 / 0.5)) < 1e-9


def test_write_results_schema(tmp_path):
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=4, reference_path=str(tmp_path / "missing.json"))
    y = np.zeros((1, 4))
    ev.update(site_ids=["A"], y_true=y, median=y, mask=np.ones_like(y))
    out = ev.write(str(tmp_path), "mmtsfm_test", {"seed": 42}, data_path="x")
    blob = json.loads(Path(out).read_text())
    assert "manifest" in blob and "results" in blob
    assert "nmae" in blob["results"]["overall"]
    # no reference present → no skill_score
    assert "skill_score" not in blob["results"]["overall"]


def test_visual_marginal_gain():
    """W6: dual on/off accumulators report the visual marginal gain (Δ)."""
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=4, compute_marginal_gain=True)

    y = np.zeros((1, 4))
    pred_on = np.zeros((1, 4))  # vision-on: zero error
    pred_off = np.array([[0.2, 0.2, 0.2, 0.2]])  # vision-off: 0.2 error
    mask = np.ones((1, 4))

    ev.update(site_ids=["A"], y_true=y, median=pred_on, mask=mask, vision_off=False)
    ev.update(site_ids=["A"], y_true=y, median=pred_off, mask=mask, vision_off=True)

    res = ev.finalize()
    assert abs(res["overall"]["nmae_vision_on"] - 0.0) < 1e-9
    assert abs(res["overall"]["nmae_vision_off"] - 0.2) < 1e-9
    assert abs(res["overall"]["delta_nmae"] - 0.2) < 1e-9

    assert abs(res["overall"]["nrmse_vision_on"] - 0.0) < 1e-9
    assert abs(res["overall"]["nrmse_vision_off"] - 0.2) < 1e-9
    assert abs(res["overall"]["delta_nrmse"] - 0.2) < 1e-9

    assert abs(res["per_plant"]["A"]["nmae_vision_on"] - 0.0) < 1e-9
    assert abs(res["per_plant"]["A"]["nmae_vision_off"] - 0.2) < 1e-9
    assert abs(res["per_plant"]["A"]["delta_nmae"] - 0.2) < 1e-9


def test_ramp_metrics_top_decile_subset():
    """S6: ramp NMAE/NRMSE from the per-site top-decile |Δy| subset."""
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=4)
    rng = np.random.default_rng(0)
    n = 50
    y = rng.uniform(0.2, 0.4, (n, 4))
    y[0, 2] = 0.95  # one huge jump → guaranteed top-decile ramp step
    pred = y + 0.05
    pred[0, 2] = y[0, 2] + 0.3  # ramp step has a much larger error
    mask = np.ones_like(y)
    delta = np.abs(np.diff(np.concatenate([y[:, :1], y], axis=1), axis=1))
    ev.update(
        site_ids=["A"] * n,
        y_true=y,
        median=pred,
        mask=mask,
        delta=delta,
        delta_valid=mask,
    )
    res = ev.finalize()
    assert "nmae_ramp" in res["overall"]
    assert "nrmse_ramp" in res["overall"]
    # the ramp subset contains the high-error jump → ramp error > overall error
    assert res["overall"]["nmae_ramp"] > res["overall"]["nmae"]
    assert res["per_plant"]["A"]["nmae_ramp"] == res["overall"]["nmae_ramp"]


def test_ramp_metrics_absent_without_delta():
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=4)
    y = np.zeros((2, 4))
    ev.update(site_ids=["A"] * 2, y_true=y, median=y, mask=np.ones_like(y))
    res = ev.finalize()
    assert "nmae_ramp" not in res["overall"]


def test_dump_predictions_per_site_npz(tmp_path):
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=4, reference_path=str(tmp_path / "missing.json"))
    y = np.random.default_rng(1).uniform(0, 1, (3, 4))
    ev.update(site_ids=["A", "B", "A"], y_true=y, median=y + 0.1, mask=np.ones_like(y))
    ev.write(str(tmp_path), "mmtsfm_test", {"seed": 42}, data_path="x")
    for site, rows in (("A", 2), ("B", 1)):
        f = tmp_path / "predictions" / f"mmtsfm_test_{site}_pred.npz"
        assert f.exists()
        data = np.load(f)
        assert data["pred"].shape == (rows, 4)
        assert data["true"].shape == (rows, 4)
        assert data["mask"].shape == (rows, 4)
    # vision-off updates must not pollute the dump
    ev2 = ProtocolEvaluator(horizon=4, compute_marginal_gain=True)
    ev2.update(
        site_ids=["A"],
        y_true=y[:1],
        median=y[:1],
        mask=np.ones((1, 4)),
        vision_off=True,
    )
    assert ev2.dump_predictions(str(tmp_path), "off_only") is None
