import numpy as np
import pandas as pd
import pytest

from context_windows import clear_sky_scale, image_difference, select_pair


def test_pixel_difference_does_not_wrap_unsigned_values():
    dark = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    light = np.full_like(dark, 255)
    assert image_difference(dark, light) == 1.0
    assert image_difference(light, dark) == 1.0


def test_selection_maximizes_image_difference_within_season_matched_pairs():
    windows = [dict(day="2019-06-01", cloud=0),
               dict(day="2019-06-02", cloud=90),
               dict(day="2019-06-03", cloud=95),
               dict(day="2019-12-01", cloud=100)]
    clips = [np.full((2, 2, 2, 3), x, dtype=np.uint8) for x in [0, 80, 180, 255]]
    clear, cloudy, score = select_pair(windows, clips)
    assert (clear, cloudy) == (0, 2)
    assert score == pytest.approx(180 / 255)


def test_clear_sky_scale_uses_only_valid_clear_history_before_origin():
    origin = pd.Timestamp("2019-06-15 09:00", tz="UTC")
    rows = [{"timestamp_utc": origin - pd.Timedelta(hours=i),
             "clearsky_ghi": 500.0, "norm_power": 0.5, "cloudcover": 0,
             "outage_flag": False, "stuck_flag": False} for i in range(1, 11)]
    for stamp, cloud, outage in [(origin, 0, False),
                                  (origin - pd.Timedelta(days=15), 0, False),
                                  (origin - pd.Timedelta(hours=11), 100, False),
                                  (origin - pd.Timedelta(hours=12), 0, True)]:
        rows.append(dict(timestamp_utc=stamp, clearsky_ghi=500.0,
                         norm_power=100.0, cloudcover=cloud,
                         outage_flag=outage, stuck_flag=False))
    assert clear_sky_scale(pd.DataFrame(rows), origin) == pytest.approx(0.001)


def test_ramp_constraint_excludes_smooth_cloudy_window_before_image_ranking():
    windows = [dict(day="2019-06-01", cloud=0, ramps=0),
               dict(day="2019-06-02", cloud=95, ramps=9, max_ramp=0.12),
               dict(day="2019-06-03", cloud=90, ramps=8, max_ramp=0.50)]
    clips = [np.full((2, 2, 2, 3), x, dtype=np.uint8) for x in [0, 255, 180]]
    clear, cloudy, score = select_pair(windows, clips, min_ramps=8, min_peak_ramp=0.30)
    assert (clear, cloudy) == (0, 2)
    assert score == pytest.approx(180 / 255)
