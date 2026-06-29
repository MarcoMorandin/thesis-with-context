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
    path.write_text(json.dumps({
        "manifest": {}, "results": {
            "overall": {"nrmse": nrmse},
            "per_plant": {"A": {"nrmse": nrmse}},
        }
    }))


def test_perfect_forecast_skill_score_one(tmp_path):
    from eval.protocol_eval import ProtocolEvaluator

    ref = tmp_path / "smart_persistence.json"
    _ref(ref, nrmse=0.2)
    ev = ProtocolEvaluator(horizon=12, reference_path=str(ref))

    y = np.random.default_rng(0).uniform(0, 1, (5, 12))
    ev.update(site_ids=["A"] * 5, y_true=y, median=y.copy(),
              mask=np.ones_like(y), quantiles=np.repeat(y[..., None], 9, axis=-1))
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
