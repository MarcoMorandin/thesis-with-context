"""Plot selected clear/cloudy windows and a past-calibrated clear-sky forecast."""
import io
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from context_windows import clear_sky_scale

DATA = Path("/Volumes/dataset/dataset")
OUT = Path(__file__).resolve().parent
selection = json.loads((OUT / "context-window-selection.json").read_text())
table = pd.read_parquet(DATA / "dataset_all.parquet",
                       filters=[("dataset", "=", "uk_pv"), ("site_id", "=", selection["site"])])
table = table.sort_values("timestamp_utc")
plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})
fig = plt.figure(figsize=(9.4, 4.8), layout="constrained")
grid = fig.add_gridspec(2, 5, width_ratios=[1.7, 1.7, 1, 1, 1],
                       hspace=0.16, wspace=0.12)
provenance = []
with h5py.File(DATA / "images_all.h5", "r") as archive:
    group = archive[f"uk_pv_{selection['site']}"]
    lookup = {pd.Timestamp(stamp.decode()): index
              for index, stamp in enumerate(group["timestamps"][:])}
    display_frames = {}
    for key in ["clear", "cloudy"]:
        for hour in [9, 12, 15]:
            stamp = pd.Timestamp(f"{selection[key]['day']} {hour:02d}:00", tz="UTC")
            frame = np.asarray(Image.open(io.BytesIO(group["images"][lookup[stamp]].tobytes())))
            display_frames[stamp] = frame
    inverted = 1.0 - np.stack(list(display_frames.values())).astype(float) / 255.0
    channel_means = inverted.mean(axis=(0, 1, 2))
    gains = channel_means.mean() / channel_means
    for row, (key, title) in enumerate([("clear", "Clear sky"), ("cloudy", "Cloudy sky")]):
        day = selection[key]["day"]
        origin = pd.Timestamp(day + " 09:00", tz="UTC")
        values = table[table.timestamp_utc.between(origin, origin + pd.Timedelta(hours=6))]
        assert len(values) == 13 and values.norm_power.notna().all()
        assert not values[["outage_flag", "stuck_flag", "bad_site_flag"]].to_numpy().any()
        assert values.timestamp_utc.diff().dropna().eq(pd.Timedelta(minutes=30)).all()
        np.testing.assert_allclose(values.norm_power,
                                   values.power_w / values.installed_power_w,
                                   rtol=1e-5, atol=1e-7)
        scale = clear_sky_scale(table, origin)
        assert np.isclose(scale, selection[key]["scale"])
        forecast = np.clip(scale * values.clearsky_ghi, 0, 1)
        ax = fig.add_subplot(grid[row, :2])
        hours = values.timestamp_utc.dt.hour + values.timestamp_utc.dt.minute / 60
        ax.plot(hours, values.norm_power, "o-", color="#285f8f", lw=1.6, ms=3,
                label="Measured power")
        ax.plot(hours, forecast, "--", color="#9b6423", lw=1.5,
                label="Naive clear-sky forecast")
        ax.set(xlim=(8.9, 15.1), ylim=(0, 1), ylabel="Capacity-normalized power",
               xlabel="Time (UTC)")
        ax.set_xticks([9, 11, 13, 15], ["09:00", "11:00", "13:00", "15:00"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.set_title(f"({chr(97 + row)}) {title}, {pd.Timestamp(day):%d %B %Y}",
                     loc="left", fontsize=10.5, pad=9)
        ax.legend(loc="lower left" if row == 0 else "lower right", fontsize=8, frameon=False)
        for col, hour in enumerate([9, 12, 15], 2):
            stamp = pd.Timestamp(f"{day} {hour:02d}:00", tz="UTC")
            index = lookup[stamp]
            assert pd.Timestamp(group["timestamps"][index].decode()) == stamp
            frame = np.asarray(Image.open(io.BytesIO(group["images"][index].tobytes())))
            assert frame.shape == (128, 128, 3)
            crop = fig.add_subplot(grid[row, col])
            display = np.clip((1.0 - frame.astype(float) / 255.0) * gains, 0, 1)
            neutral = display.mean(axis=-1, keepdims=True)
            display = np.clip((neutral + 0.15 * (display - neutral))
                              * np.array([0.95, 1.0, 1.08]), 0, 1) ** 0.65
            crop.imshow(display, interpolation="nearest")
            crop.set_title(f"{hour:02d}:00 UTC", fontsize=9, pad=5)
            crop.set_xticks([])
            crop.set_yticks([])
            for spine in crop.spines.values():
                spine.set_visible(True)
                spine.set_color("#bcbcbc")
                spine.set_linewidth(0.5)
            provenance.append(f"{stamp.isoformat()} group=uk_pv_{selection['site']} index={index}")
for suffix in ["pdf", "png"]:
    fig.savefig(OUT / f"pv-power-satellite-context.{suffix}", dpi=300)
plt.close(fig)
(OUT / "pv-power-satellite-context-provenance.txt").write_text(
    "Source: /Volumes/dataset/dataset/{dataset_all.parquet,images_all.h5}\n"
    "Selection: context-window-selection.json; maximal eligible mean pixel difference.\n"
    "Power: native samples, no interpolation; verified against power_w / installed_power_w.\n"
    "Forecast: alpha * stored clearsky_ghi, clipped [0,1]; alpha fitted using only prior\n"
    "14-day valid clear observations before 09:00 UTC; illustrative, no benchmark score.\n"
    "Satellite display: all three source bands inverted, then shared gray-world\n"
    f"channel gains {gains.tolist()} applied to all six frames and clipped [0,1].\n"
    "Gains use pooled means across all six inverted frames, never per-image adjustment.\n"
    "Shared display filter: saturation 0.15 around channel mean, RGB multipliers\n"
    "[0.95,1.0,1.08] to remove warm cast, clipping [0,1], gamma exponent 0.65.\n"
    "This is display-enhanced false color, not reconstructed visible-light color.\n"
    "Selection uses original bands; source pixels and dataset remain unchanged.\n"
    + "\n".join(provenance) + "\n")
print("Verified 26 power samples, past-only calibration, and 6 timestamp-exact satellite crops.")
print(OUT / "pv-power-satellite-context.pdf")
