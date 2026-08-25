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


def test_ramp_thresholds_are_shared_across_vision_passes():
    """S2: both passes are scored on the SAME per-site top-decile subset.

    Thresholds are a property of the data, not of a model. In production both
    passes see identical `delta` (it derives from y_true/context, never from the
    predictions), so a per-pass threshold bug is invisible there. Feeding the
    off pass a *different* delta makes the two implementations disagree:

      shared thresholds  -> off is scored on the ON pass's ramp row  -> 0.3
      per-pass threshold -> off is scored on its own ramp row        -> 0.5
    """
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=1, compute_marginal_gain=True)
    n = 10
    y = np.zeros((n, 1))
    mask = np.ones((n, 1))

    # delta 0 everywhere except the LAST row -> top decile is row 9 alone
    delta_on = np.zeros((n, 1))
    delta_on[9, 0] = 10.0
    # deliberately different: top decile would be row 0 alone
    delta_off = np.zeros((n, 1))
    delta_off[0, 0] = 10.0

    pred_on = np.zeros((n, 1))  # zero error everywhere
    pred_off = np.zeros((n, 1))
    pred_off[0, 0] = 0.5  # error on the OFF pass's would-be ramp row
    pred_off[9, 0] = 0.3  # error on the ON pass's ramp row

    ev.update(
        site_ids=["A"] * n, y_true=y, median=pred_on, mask=mask,
        delta=delta_on, delta_valid=mask, vision_off=False,
    )
    ev.update(
        site_ids=["A"] * n, y_true=y, median=pred_off, mask=mask,
        delta=delta_off, delta_valid=mask, vision_off=True,
    )

    res = ev.finalize()
    # tolerance is float32: _store_batch stores the buffers as float32 on purpose
    assert abs(res["overall"]["nmae_ramp_vision_on"] - 0.0) < 1e-6
    assert abs(res["overall"]["nmae_ramp_vision_off"] - 0.3) < 1e-6
    assert abs(res["overall"]["delta_nmae_ramp"] - 0.3) < 1e-6
    # the discriminating assertion: a per-pass threshold would score row 0 (0.5)
    assert res["overall"]["nmae_ramp_vision_off"] < 0.4


def test_dump_predictions_writes_vision_off_alongside(tmp_path):
    """S3: the off pass lands in its own npz so localize can read both.

    Separate file, not extra keys: `decompose_by_horizon(pred_on, pred_off, ...)`
    takes two arrays, and any existing consumer of the on-pass npz keeps working.
    """
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=4, compute_marginal_gain=True)
    y = np.random.default_rng(2).uniform(0, 1, (3, 4))
    mask = np.ones_like(y)
    ev.update(site_ids=["A"] * 3, y_true=y, median=y + 0.1, mask=mask, vision_off=False)
    ev.update(site_ids=["A"] * 3, y_true=y, median=y + 0.4, mask=mask, vision_off=True)

    ev.dump_predictions(str(tmp_path), "mmtsfm_marg")

    on = np.load(tmp_path / "predictions" / "mmtsfm_marg_A_pred.npz")
    off = np.load(tmp_path / "predictions" / "mmtsfm_marg_A_pred_off.npz")
    assert on["pred"].shape == (3, 4)
    assert off["pred"].shape == (3, 4)
    # the two passes carry different predictions over identical ground truth
    assert np.allclose(on["true"], off["true"])
    assert not np.allclose(on["pred"], off["pred"])


def test_no_off_dump_when_marginal_gain_disabled(tmp_path):
    """Default path stays byte-identical: no off store, no off file."""
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=4)
    y = np.zeros((2, 4))
    ev.update(site_ids=["A"] * 2, y_true=y, median=y, mask=np.ones_like(y))
    ev.dump_predictions(str(tmp_path), "plain")
    assert (tmp_path / "predictions" / "plain_A_pred.npz").exists()
    assert not (tmp_path / "predictions" / "plain_A_pred_off.npz").exists()
