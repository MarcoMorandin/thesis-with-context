import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
import h5py, io
from PIL import Image
from pathlib import Path

# --- Presentation & Typography ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "none"
plt.rcParams["axes.linewidth"] = 0

# --- Exact Palette from Slide Deck Template ---
COLOR_BG = "#fbfcfd"             # Slide background
COLOR_TEXT_MAIN = "#142c45"      # Bold dark navy text (#142c45)
COLOR_TEXT_MUTED = "#627487"     # Muted slate subtitle (#627487)
COLOR_PV = "#bc7d13"             # Golden amber / ochre (#bc7d13)
COLOR_IRR = "#245a80"            # Deep steel blue (#245a80)
COLOR_CLOUD = "#9aaab7"          # Slate grey-blue (#9aaab7)
COLOR_DIVIDER = "#e2e9ef"        # Subtle row divider
COLOR_ANCHOR = "#6085a3"         # Crisp anchor vertical lines (#6085a3)
COLOR_CARD_BORDER = "#88a8c0"    # Satellite frame card border
COLOR_BREAK = "#64849e"          # Axis break lines and badges
COLOR_BADGE_BG = "#eaf1f7"       # Soft pill badge background for axis break annotation

# --- 1. Data Loading ---
site = "10793"
dataset_path = "/Volumes/dataset/dataset/dataset_all.parquet"
h5_path = "/Volumes/dataset/dataset/images_all.h5"

dataset = ds.dataset(dataset_path, format="parquet")
table = dataset.to_table(filter=(ds.field("site_id") == site), columns=[
    "timestamp_utc", "norm_power", "shortwave_radiation", 
    "cloudcover", "clearsky_ghi"
])
df = table.to_pandas().sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)
df = df.set_index("timestamp_utc")

# Continuous 30-min grid
full_grid = pd.date_range(df.index[0], df.index[-1], freq="30min", tz="UTC")
df_full = df.reindex(full_grid)
df_full["norm_power"] = df_full["norm_power"].fillna(0.0)
df_full["shortwave_radiation"] = df_full["shortwave_radiation"].fillna(0.0)
df_full["clearsky_ghi"] = df_full["clearsky_ghi"].fillna(0.0)
df_full["cloudcover"] = df_full["cloudcover"].interpolate(method="time").bfill().ffill()

# Protocol specification: 14-day history window (672 steps) up to 12:00 UTC
t_origin = pd.Timestamp("2020-05-22 12:00:00+00:00")

# 8 satellite frame timestamps: 45 min cadence ending at 11:45 UTC (15m before origin)
frame_times = [
    t_origin - pd.Timedelta(minutes=15) - pd.Timedelta(minutes=45*k)
    for k in range(7, -1, -1)
]

# Load true satellite image crops from H5
sat_images = []
with h5py.File(h5_path, "r") as f:
    h5_site = f[f"uk_pv_{site}"]
    h5_ts = {ts.decode("utf-8") if isinstance(ts, bytes) else str(ts): i for i, ts in enumerate(h5_site["timestamps"])}
    for ft in frame_times:
        fts = ft.strftime("%Y-%m-%dT%H:%M:%SZ")
        idx = h5_ts[fts]
        raw_bytes = h5_site["images"][idx]
        img = Image.open(io.BytesIO(raw_bytes))
        sat_images.append(img)

print(f"Loaded {len(sat_images)} genuine cloud satellite frames from {h5_path}.")

# --- 2. Geometry: 3 Days Numerical History + 8-Frame Visual Alignment Window ---
X_MIN = 0.2
X_MAX = 16.5
X_HIST_START = 0.80
X_HIST_END = 6.20
X_BREAK = 6.925   # Perfectly centered midpoint between 6.20 and 7.65
X_SAT_START = 7.65
X_SAT_END = 15.65
X_END = 16.20     # Terminus cleanly at 12:00 UTC

# 8 frame X positions
x_frames = np.linspace(X_SAT_START, X_SAT_END, len(frame_times))

# Interpolate exact values at frame times + 12:00 terminus
eval_post_times = frame_times + [t_origin]
df_post_eval = df_full.reindex(df_full.index.union(eval_post_times)).interpolate(method="time").loc[eval_post_times]

x_post_eval = np.append(x_frames, X_END)
y_pv_post_pts = df_post_eval["norm_power"].to_numpy(dtype=float)
y_irr_post_pts = df_post_eval["shortwave_radiation"].to_numpy(dtype=float)
y_cloud_post_pts = df_post_eval["cloudcover"].to_numpy(dtype=float)

# Pre-break: 3 continuous days of numerical history (May 19 06:00 to May 21 18:00)
t_hist_start = pd.Timestamp("2020-05-19 06:00:00+00:00")
t_hist_end = pd.Timestamp("2020-05-21 18:00:00+00:00")
skip_label = "10 days"  # 14 days total history - (3 days shown + 1 day visual window) = 10 days

day_labels = [("Day 1", 1/6), ("Day 2", 3/6), ("Day 3", 5/6)]
ticks_pre = [
    (0.0, "06:00"), (1/6, "12:00"), (2/6, "00:00"), 
    (3/6, "12:00"), (4/6, "00:00"), (5/6, "12:00"), (1.0, "18:00")
]

hist_slice = df_full.loc[t_hist_start:t_hist_end].copy()
dt_sec = (hist_slice.index - t_hist_start).total_seconds().to_numpy()
tot_sec = (t_hist_end - t_hist_start).total_seconds()
x_pre_pts = X_HIST_START + (dt_sec / tot_sec) * (X_HIST_END - X_HIST_START)

y_pv_raw = np.clip(gaussian_filter1d(hist_slice["norm_power"].to_numpy(dtype=float), sigma=1.2), 0.0, 1.0)
y_pv_raw[0] = 0.0
y_pv_raw[-1] = 0.0
y_irr_raw = np.clip(gaussian_filter1d(hist_slice["shortwave_radiation"].to_numpy(dtype=float), sigma=1.2), 0.0, None)
y_irr_raw[0] = 0.0
y_irr_raw[-1] = 0.0
y_cloud_raw = np.clip(gaussian_filter1d(hist_slice["cloudcover"].to_numpy(dtype=float), sigma=1.2), 0.0, 100.0)

# High-density splines
x_dense_pre = np.linspace(X_HIST_START, X_HIST_END, 400)
y_pv_pre = np.clip(PchipInterpolator(x_pre_pts, y_pv_raw)(x_dense_pre), 0.0, 1.0)
y_irr_pre = np.clip(PchipInterpolator(x_pre_pts, y_irr_raw)(x_dense_pre), 0.0, None)
y_cloud_pre = np.clip(PchipInterpolator(x_pre_pts, y_cloud_raw)(x_dense_pre), 0.0, 100.0)

x_dense_post = np.linspace(X_SAT_START, X_END, 500)
spl_pv_post = CubicSpline(x_post_eval, y_pv_post_pts, bc_type="natural")
y_pv_post_smooth = np.clip(spl_pv_post(x_dense_post), 0.0, 1.0)

spl_irr_post = Akima1DInterpolator(x_post_eval, y_irr_post_pts)
y_irr_post_smooth = np.clip(spl_irr_post(x_dense_post), 0.0, None)

spl_cloud_post = CubicSpline(x_post_eval, y_cloud_post_pts, bc_type="natural")
y_cloud_post_smooth = np.clip(spl_cloud_post(x_dense_post), 0.0, 100.0)

# --- 3. Figure Layout ---
fig = plt.figure(figsize=(19.5, 9.8), facecolor=COLOR_BG)
gs = gridspec.GridSpec(
    4, 1, 
    height_ratios=[1.15, 1.00, 1.00, 2.15], 
    hspace=0.18, 
    left=0.08, right=0.96, top=0.93, bottom=0.06
)

ax_pv = fig.add_subplot(gs[0])
ax_irr = fig.add_subplot(gs[1])
ax_cloud = fig.add_subplot(gs[2])
ax_sat = fig.add_subplot(gs[3])

for ax in [ax_pv, ax_irr, ax_cloud, ax_sat]:
    ax.set_facecolor(COLOR_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(X_MIN, X_MAX)

# --- 4. ROW 1: PV Output ---
ax_pv.plot(x_dense_pre, y_pv_pre, color=COLOR_PV, lw=3.0, solid_capstyle="round", zorder=5)
ax_pv.plot(x_dense_post, y_pv_post_smooth, color=COLOR_PV, lw=3.0, solid_capstyle="round", zorder=5)
ax_pv.text(0.35, 0.45, "PV output", color=COLOR_TEXT_MAIN, fontsize=15.5, fontweight="bold", ha="right", va="center")
ax_pv.set_ylim(-0.06, 1.12)

# --- 5. ROW 2: Irradiance ---
max_irr = max(np.max(y_irr_pre), np.max(y_irr_post_smooth))
ax_irr.plot(x_dense_pre, y_irr_pre, color=COLOR_IRR, lw=2.9, solid_capstyle="round", zorder=5)
ax_irr.plot(x_dense_post, y_irr_post_smooth, color=COLOR_IRR, lw=2.9, solid_capstyle="round", zorder=5)
ax_irr.text(0.35, max_irr * 0.45, "Irradiance", color=COLOR_TEXT_MAIN, fontsize=15.5, fontweight="bold", ha="right", va="center")
ax_irr.set_ylim(-20, max_irr * 1.15)

# --- 6. ROW 3: Cloud Cover ---
ax_cloud.plot(x_dense_pre, y_cloud_pre, color=COLOR_CLOUD, lw=2.9, solid_capstyle="round", zorder=5)
ax_cloud.plot(x_dense_post, y_cloud_post_smooth, color=COLOR_CLOUD, lw=2.9, solid_capstyle="round", zorder=5)
ax_cloud.text(0.35, 50, "Cloud cover", color=COLOR_TEXT_MAIN, fontsize=15.5, fontweight="bold", ha="right", va="center")
ax_cloud.set_ylim(-5, 105)

# --- 7. Baseline Dividers with Double Slash // Break ---
break_w = 0.16
for ax, y_base in [(ax_pv, 0.0), (ax_irr, 0.0), (ax_cloud, 0.0)]:
    ax.plot([0.45, X_BREAK - break_w], [y_base, y_base], color=COLOR_DIVIDER, lw=1.2, zorder=1)
    ax.plot([X_BREAK + break_w, X_END], [y_base, y_base], color=COLOR_DIVIDER, lw=1.2, zorder=1)
    y_span = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.045
    dx = 0.06
    ax.plot([X_BREAK - 0.06, X_BREAK - 0.06 + dx], [y_base - y_span, y_base + y_span], color=COLOR_BREAK, lw=1.8, zorder=12)
    ax.plot([X_BREAK + 0.06, X_BREAK + 0.06 + dx], [y_base - y_span, y_base + y_span], color=COLOR_BREAK, lw=1.8, zorder=12)

# --- 8. ROW 4: Satellite Frames, Anchor Lines & Timeline ---
ax_sat.set_ylim(-0.76, 0.68)

# Marker dots at anchor points
for xf, yp, yi, yc in zip(x_frames, y_pv_post_pts[:8], y_irr_post_pts[:8], y_cloud_post_pts[:8]):
    ax_pv.plot(xf, yp, marker="o", markersize=6.5, color=COLOR_PV, markeredgecolor="#ffffff", markeredgewidth=1.5, zorder=9)
    ax_irr.plot(xf, yi, marker="o", markersize=6.0, color=COLOR_IRR, markeredgecolor="#ffffff", markeredgewidth=1.4, zorder=9)
    ax_cloud.plot(xf, yc, marker="o", markersize=6.0, color=COLOR_CLOUD, markeredgecolor="#ffffff", markeredgewidth=1.4, zorder=9)

# 8 Large Satellite Frame Cards (78pt, sleek spacing)
for i, (xf, img, ft) in enumerate(zip(x_frames, sat_images, frame_times)):
    thumb = img.resize((256, 256), Image.Resampling.LANCZOS)
    imagebox = OffsetImage(thumb, zoom=0.305)
    ab = AnnotationBbox(
        imagebox, (xf, 0.05), frameon=True, pad=0.06,
        bboxprops=dict(boxstyle="square,pad=0.03", edgecolor=COLOR_CARD_BORDER, facecolor="#ffffff", linewidth=1.2)
    )
    ax_sat.add_artist(ab)
    ax_sat.text(
        xf, 0.45, f"{i+1}", 
        color=COLOR_TEXT_MAIN, fontsize=10.5, fontweight="bold", 
        ha="center", va="center",
        bbox=dict(boxstyle="circle,pad=0.24", facecolor="#ffffff", edgecolor=COLOR_CARD_BORDER, linewidth=1.2)
    )

# --- 9. Full-Width Timeline Axis ---
y_tl = -0.38
tick_len = 0.05

# Left history timeline
ax_sat.plot([X_HIST_START, X_BREAK - break_w], [y_tl, y_tl], color=COLOR_DIVIDER, lw=1.4, zorder=1)
for frac, lbl in ticks_pre:
    xt = X_HIST_START + frac * (X_HIST_END - X_HIST_START)
    ax_sat.plot([xt, xt], [y_tl, y_tl - tick_len], color=COLOR_BREAK, lw=1.2, zorder=2)
    ax_sat.text(xt, y_tl - tick_len - 0.04, lbl, color=COLOR_TEXT_MUTED, fontsize=8.5, fontweight="bold", ha="center", va="top")

# Day 1, Day 2, Day 3 grouping badges
for d_lbl, d_frac in day_labels:
    x_d = X_HIST_START + d_frac * (X_HIST_END - X_HIST_START)
    ax_sat.text(
        x_d, 0.20, d_lbl, 
        color=COLOR_TEXT_MAIN, fontsize=11.0, fontweight="bold", 
        ha="center", va="center", 
        bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.3", facecolor="#edf2f7", edgecolor="#b9cde0", linewidth=0.9)
    )

# Timeline break slashes // and "10 days" pill badge
y_span_tl = 0.05
dx = 0.06
ax_sat.plot([X_BREAK - 0.06, X_BREAK - 0.06 + dx], [y_tl - y_span_tl, y_tl + y_span_tl], color=COLOR_BREAK, lw=1.8, zorder=12)
ax_sat.plot([X_BREAK + 0.06, X_BREAK + 0.06 + dx], [y_tl - y_span_tl, y_tl + y_span_tl], color=COLOR_BREAK, lw=1.8, zorder=12)

ax_sat.text(
    X_BREAK, y_tl - tick_len - 0.045, skip_label, 
    color=COLOR_TEXT_MAIN, fontsize=11.0, fontweight="bold", ha="center", va="top",
    bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.35", facecolor=COLOR_BADGE_BG, edgecolor=COLOR_BREAK, linewidth=1.1, zorder=15)
)

# Right visual window timeline
ax_sat.plot([X_BREAK + break_w, X_END], [y_tl, y_tl], color=COLOR_DIVIDER, lw=1.4, zorder=1)
for i, (xf, ft) in enumerate(zip(x_frames, frame_times)):
    time_str = ft.strftime("%H:%M")
    # Stem connecting card to tick
    ax_sat.plot([xf, xf], [-0.20, y_tl], color=COLOR_CARD_BORDER, lw=0.9, linestyle=":", alpha=0.7, zorder=1)
    ax_sat.plot([xf, xf], [y_tl, y_tl - tick_len], color=COLOR_BREAK, lw=1.2, zorder=2)
    ax_sat.text(xf, y_tl - tick_len - 0.04, time_str, color=COLOR_TEXT_MUTED, fontsize=9.0, fontweight="bold", ha="center", va="top")

# Terminus tick at 12:00
ax_sat.plot([X_END, X_END], [y_tl, y_tl - tick_len], color=COLOR_TEXT_MAIN, lw=1.4, zorder=2)
ax_sat.text(X_END, y_tl - tick_len - 0.04, "12:00", color=COLOR_TEXT_MAIN, fontsize=9.0, fontweight="bold", ha="center", va="top")

# --- 10. Anchor Lines from Satellite Frames Up Through Curves ---
fig.canvas.draw()
def get_fig_x(ax, data_x):
    return ax.transData.transform((data_x, 0))[0] / fig.bbox.width

for xf, yp in zip(x_frames, y_pv_post_pts[:8]):
    fig_x = get_fig_x(ax_sat, xf)
    fig_y_bottom = ax_sat.transData.transform((xf, 0.54))[1] / fig.bbox.height
    fig_y_dot = ax_pv.transData.transform((xf, yp))[1] / fig.bbox.height
    line = mlines.Line2D(
        [fig_x, fig_x], [fig_y_bottom, fig_y_dot], 
        color=COLOR_ANCHOR, lw=1.2, alpha=0.70, linestyle="--", 
        transform=fig.transFigure, zorder=2
    )
    fig.add_artist(line)

# --- 11. Save Figure ---
repo_out = Path("report/figures/arm_families_input_output.png")
repo_out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(repo_out, dpi=300, bbox_inches="tight", facecolor=COLOR_BG)

scratch_out = Path("/Users/marcomorandin/.gemini/antigravity-cli/brain/91dccad1-5457-4684-9174-7756c94c9d24/scratch/arm_families_input_output.png")
fig.savefig(scratch_out, dpi=300, bbox_inches="tight", facecolor=COLOR_BG)

plt.close(fig)
print(f"Generated finalized dataset representation plot at:\n  {repo_out}")
