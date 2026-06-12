"""Evaluation-protocol tests (BASELINE_COMPARISON.md §6.2-6.3, §5).

Split disjointness, normalizer leakage, daylight masking, the night/masked
leakage check, ramp-subset and per-horizon correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from common import config
from common.base import build
from common.runner import (compute_ramp_thresholds, evaluate_model,
                           _ramp_mask)
from common.windows import build_site_series, time_slice_series

from .conftest import make_frame, windows_for


def test_normalizer_is_data_independent():
    """Covariate scalings are fixed physical constants — they cannot leak
    test-plant statistics (§6.3)."""
    df_a = make_frame(n_sites=1, days=3, seed=1)
    df_b = make_frame(n_sites=1, days=3, seed=2)
    s_a = build_site_series(df_a)[0]
    s_b = build_site_series(df_b)[0]
    # same raw covariate value must map to the same scaled value regardless
    # of which data it was built with
    raw = 25.0
    i = config.COV_COLS.index("temperature_2m")
    assert config.COV_SCALES["temperature_2m"] == 40.0
    df_a2 = df_a.copy()
    df_a2["temperature_2m"] = raw
    df_b2 = df_b.copy()
    df_b2["temperature_2m"] = raw
    v_a = build_site_series(df_a2)[0].cov[0, i]
    v_b = build_site_series(df_b2)[0].cov[0, i]
    assert v_a == v_b == np.float32(raw / 40.0)
    assert s_a.cov.shape[1] == s_b.cov.shape[1] == len(config.COV_COLS)


def test_daylight_masking_excludes_night_errors():
    """§5 night test: a forecast wrong only at night scores identically to a
    perfect one — masked steps must not leak into metrics."""
    ds = windows_for(make_frame(n_sites=1, days=4, kt=0.7, nan_fraction=0.0),
                     stride=1)

    class NightVandal:
        name = "night_vandal"

        def predict(self, batch):
            from common.base import Forecast
            point = batch["y_future"].copy()
            night = batch["daylight_future"] == 0
            point[night] = 99.0  # absurd values, but only at night
            return Forecast(point=point.astype(np.float32))

    class Perfect:
        name = "perfect"

        def predict(self, batch):
            from common.base import Forecast
            return Forecast(point=batch["y_future"].astype(np.float32))

    vandal = evaluate_model(NightVandal(), ds)["overall"]
    perfect = evaluate_model(Perfect(), ds)["overall"]
    assert vandal["nmae"] == perfect["nmae"] == 0.0
    assert vandal["nrmse"] == perfect["nrmse"] == 0.0


def test_ramp_thresholds_top_decile():
    ds = windows_for(make_frame(n_sites=2, days=5, seed=7), stride=1)
    thresholds = compute_ramp_thresholds(ds)
    assert set(thresholds) == {s.site_id for s in ds.series}
    assert all(t >= 0 for t in thresholds.values())
    # the ramp mask must select a minority subset of valid steps
    batch = ds.batch(list(range(len(ds))))
    ramp = _ramp_mask(batch, thresholds)
    valid = batch["mask_future"] * batch["daylight_future"]
    assert 0 < ramp.sum() < 0.5 * valid.sum()


def test_ramp_metrics_reported():
    ds = windows_for(make_frame(n_sites=1, days=4, seed=3), stride=2)
    thresholds = compute_ramp_thresholds(ds)
    results = evaluate_model(build("persistence"), ds,
                             ramp_thresholds=thresholds)
    overall = results["overall"]
    assert "nmae_ramp" in overall and "nrmse_ramp" in overall
    # persistence is hurt most exactly on ramps
    assert overall["nmae_ramp"] >= overall["nmae"]


def test_per_horizon_breakdown_monotone_for_persistence():
    """NMAE(h) of persistence must grow (weakly) with horizon on smooth
    synthetic data — and the vector must have length H (§4.2)."""
    ds = windows_for(make_frame(n_sites=1, days=5, nan_fraction=0.0, seed=5),
                     stride=2)
    overall = evaluate_model(build("persistence"), ds)["overall"]
    curve = overall["nmae_per_horizon"]
    assert len(curve) == ds.horizon
    assert curve[-1] > curve[0]  # errors accumulate with lead time


def test_per_sample_losses_aligned_and_finite():
    ds = windows_for(make_frame(n_sites=2, days=4, seed=9), stride=4)
    results = evaluate_model(build("persistence"), ds, collect_losses=True)
    per_sample = results["per_sample"]
    assert len(per_sample["loss"]) == len(ds)
    assert np.isfinite(per_sample["loss"]).all()
    assert per_sample["plant"].shape == per_sample["day"].shape == \
        per_sample["loss"].shape


def test_time_slice_disjoint_ranges():
    """S1 in-domain: train and eval time ranges must not overlap."""
    series = build_site_series(make_frame(n_sites=1, days=5))
    early = time_slice_series(series, 0.0, 0.8)[0]
    late = time_slice_series(series, 0.8, 1.0)[0]
    assert early.timestamps[-1] < late.timestamps[0]
    assert len(early.y) + len(late.y) == len(series[0].y)
    with pytest.raises(ValueError):
        time_slice_series(series, 0.8, 0.2)
