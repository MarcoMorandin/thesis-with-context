"""Select the most visually distinct eligible clear/cloudy six-hour windows."""
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from context_windows import clear_sky_scale, select_pair

DATA = Path("/Volumes/dataset/dataset")
OUT = Path(__file__).resolve().parent
table = pd.read_parquet(DATA / "dataset_all.parquet",
                       filters=[("dataset", "=", "uk_pv"), ("site_id", "=", "11287")])
table = table.sort_values("timestamp_utc")
windows, clips = [], []
pointer_mismatches = 0
with h5py.File(DATA / "images_all.h5", "r") as archive:
    group = archive["uk_pv_11287"]
    frame_lookup = {pd.Timestamp(stamp.decode()): index
                    for index, stamp in enumerate(group["timestamps"][:])}
    for day, full_day in table.groupby(table.timestamp_utc.dt.strftime("%Y-%m-%d")):
        hours = full_day.timestamp_utc.dt.hour + full_day.timestamp_utc.dt.minute / 60
        window = full_day[hours.between(9, 15)]
        if len(window) != 13 or window.norm_power.isna().any():
            continue
        if window[["outage_flag", "stuck_flag", "bad_site_flag"]].to_numpy().any():
            continue
        if window.solar_zenith.max() >= 85 or window.image_h5_index.isna().any():
            continue
        if any(stamp not in frame_lookup for stamp in window.timestamp_utc):
            continue
        cloud = float(window.cloudcover.mean())
        if 10 < cloud < 80:
            continue
        origin = window.timestamp_utc.iloc[0]
        try:
            scale = clear_sky_scale(table, origin)
        except ValueError:
            continue
        frames = []
        for _, record in window.iterrows():
            index = frame_lookup[record.timestamp_utc]
            pointer_mismatches += index != int(record.image_h5_index)
            assert pd.Timestamp(group["timestamps"][index].decode()) == record.timestamp_utc
            image = np.asarray(Image.open(io.BytesIO(group["images"][index].tobytes())))
            assert image.shape == (128, 128, 3)
            frames.append(image)
        clips.append(np.stack(frames))
        changes = np.abs(np.diff(window.norm_power.to_numpy()))
        windows.append(dict(day=day, cloud=cloud, scale=scale,
                            ramps=int((changes >= 0.10).sum()), max_ramp=float(changes.max())))
        if len(windows) % 30 == 0:
            print(f"Read {len(windows)} eligible windows", flush=True)
clear, cloudy, score = select_pair(windows, clips, min_ramps=8, min_peak_ramp=0.30)
pair_count = sum(a["cloud"] <= 10 and b["cloud"] >= 80
                 and b["ramps"] >= 8 and b["max_ramp"] >= 0.30
                 and abs((pd.Timestamp(a["day"]) - pd.Timestamp(b["day"])).days) <= 31
                 for a in windows for b in windows)
result = dict(site="11287", start_hour=9, end_hour=15,
              clear=windows[clear], cloudy=windows[cloudy], score=score,
              eligible_windows=len(windows), eligible_pairs=pair_count,
              pointer_mismatches=pointer_mismatches,
              frame_lookup="Exact UTC timestamp in uk_pv_11287/timestamps; table pointers audited",
              score_definition="Mean absolute pixel difference / 255 over all 13 aligned RGB frames",
              eligibility="Valid native 09:00-15:00 UTC windows, solar zenith <85 degrees; clear mean cloud <=10%, cloudy >=80%; dates <=31 days apart; >=6 prior clear calibration samples; cloudy window >=8 changes of >=0.10 capacity-normalized power per 30 min and >=1 change of >=0.30",
              ramp_threshold=0.10, minimum_cloudy_ramps=8, minimum_peak_ramp=0.30,
              forecast="clip(alpha * clearsky_ghi, 0, 1); alpha fitted by least squares to valid clear observations in preceding 14 days, strictly before 09:00 UTC")
(OUT / "context-window-selection.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
