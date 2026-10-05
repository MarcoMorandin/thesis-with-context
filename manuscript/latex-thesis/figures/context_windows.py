"""Selection and clear-sky reference for the introductory data illustration."""
import numpy as np
import pandas as pd

def image_difference(left, right):
    assert left.shape == right.shape and left.ndim == 4
    return float(np.abs(left.astype(np.float32) - right.astype(np.float32)).mean() / 255)

def select_pair(windows, clips, min_ramps=0, min_peak_ramp=0):
    pairs = []
    for i, clear in enumerate(windows):
        if clear["cloud"] > 10:
            continue
        for j, cloudy in enumerate(windows):
            separation = abs((pd.Timestamp(clear["day"]) - pd.Timestamp(cloudy["day"])).days)
            if (cloudy["cloud"] >= 80 and separation <= 31
                    and cloudy.get("ramps", 0) >= min_ramps
                    and cloudy.get("max_ramp", 0) >= min_peak_ramp):
                pairs.append((i, j, image_difference(clips[i], clips[j])))
    if not pairs:
        raise ValueError("No season-matched clear/cloudy pair meets the selection criteria")
    return max(pairs, key=lambda pair: pair[2])

def clear_sky_scale(table, origin):
    history = table[table.timestamp_utc.ge(origin - pd.Timedelta(days=14))
                    & table.timestamp_utc.lt(origin)
                    & table.cloudcover.le(10) & table.clearsky_ghi.ge(100)
                    & table.norm_power.notna()
                    & ~table.outage_flag.astype(bool) & ~table.stuck_flag.astype(bool)]
    if len(history) < 6:
        raise ValueError("Fewer than six valid clear-sky history samples")
    irradiance = history.clearsky_ghi.to_numpy()
    return float(irradiance @ history.norm_power.to_numpy() / (irradiance @ irradiance))
