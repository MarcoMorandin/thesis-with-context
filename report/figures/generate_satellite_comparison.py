import h5py
import io
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

h5_path = "/Volumes/dataset/dataset/images_all.h5"
out_dir = Path("report/figures")
out_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/Users/marcomorandin/.gemini/antigravity-cli/brain/91dccad1-5457-4684-9174-7756c94c9d24/scratch")

# --- 1. Extract Raw Images from H5 ---
with h5py.File(h5_path, "r") as f:
    # UK image: site 10793, 2020-05-22 11:45 UTC (matching dataset pipeline overview)
    grp_uk = f["uk_pv_10793"]
    ts_uk = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in grp_uk["timestamps"]]
    idx_uk = ts_uk.index("2020-05-22T11:45:00Z")
    raw_uk = grp_uk["images"][idx_uk]
    img_uk = Image.open(io.BytesIO(raw_uk))
    
    # Save raw UK PNG (128x128)
    uk_raw_path = out_dir / "satellite_uk.png"
    img_uk.save(uk_raw_path)
    img_uk.save(scratch_dir / "satellite_uk.png")

    # US image: test site 1202, 2019-06-15 19:00 UTC (GOES ABI East Coast crop)
    grp_us = f["goes_pvdaq_1202"]
    ts_us = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in grp_us["timestamps"]]
    idx_us = ts_us.index("2019-06-15T19:00:00Z")
    raw_us = grp_us["images"][idx_us]
    img_us = Image.open(io.BytesIO(raw_us))
    
    # Save raw US PNG (256x256)
    us_raw_path = out_dir / "satellite_us.png"
    img_us.save(us_raw_path)
    img_us.save(scratch_dir / "satellite_us.png")

print(f"Extracted raw images: UK {img_uk.size}, US {img_us.size}")

# --- 2. Styling Constants ---
COLOR_BG = "#fbfcfd"
COLOR_CARD_BG = "#ffffff"
COLOR_TEXT_MAIN = "#142c45"
COLOR_TEXT_MUTED = "#627487"
COLOR_BORDER = "#88a8c0"
COLOR_ACCENT_UK = "#bc7d13"     # Amber / ochre
COLOR_ACCENT_US = "#245a80"     # Steel blue
COLOR_PILL_UK = "#fef6e7"
COLOR_PILL_US = "#eaf1f7"

specs_uk = [
    ("Spatial Resolution", "~3–4 km / pixel (nadir ~3 km, oblique at 51°–58°N)"),
    ("Physical Area", "~380 × 380 km site-centered bounding area"),
    ("Spectral Composite", "3-channel composite (IR/VIS non-HRV, night-visible)"),
    ("Temporal Cadence", "15-minute intervals (paired with 30-min power)"),
    ("Solar Plants", "100 residential rooftop installations (1.5–4.0 kW)")
]

specs_us = [
    ("Spatial Resolution", "~0.5–1.0 km / pixel (near-nadir at 36°–40°N)"),
    ("Physical Area", "~250 × 250 km site-centered bounding area"),
    ("Spectral Composite", "Day Land Cloud RGB (cyan clouds, green land, navy ocean)"),
    ("Temporal Cadence", "15-minute intervals (paired with 15-min power)"),
    ("Solar Plants", "10 commercial / industrial / utility plants (1.8–408 kW)")
]

# --- 3. Combined Side-by-Side Comparison Slide Graphic ---
fig = plt.figure(figsize=(16, 9), facecolor=COLOR_BG)

fig.text(0.08, 0.93, "Satellite Modality Comparison: UK vs. US", fontsize=23, fontweight="bold", color=COLOR_TEXT_MAIN)
fig.text(0.08, 0.89, "Sensor specifications, image crops, and spatial characteristics across evaluation domains", fontsize=12.5, color=COLOR_TEXT_MUTED)

gs = gridspec.GridSpec(1, 2, left=0.08, right=0.92, top=0.83, bottom=0.08, wspace=0.09)

ax_left = fig.add_subplot(gs[0])
ax_right = fig.add_subplot(gs[1])

for ax in [ax_left, ax_right]:
    ax.set_facecolor(COLOR_CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_BORDER)
        spine.set_linewidth(1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

# Exact square inset dimensions
h_frac = 0.38
w_frac = h_frac * (6.75 / 5.95)
left_frac = (1.0 - w_frac) / 2.0
bottom_frac = 0.40

# --- UK Card ---
ax_left.text(0.5, 9.35, "  UK TRACK (uk_pv)  ", fontsize=13.5, fontweight="bold", color=COLOR_ACCENT_UK,
             bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.3", facecolor=COLOR_PILL_UK, edgecolor=COLOR_ACCENT_UK, linewidth=1.2))
ax_left.text(0.5, 8.75, "EUMETSAT Meteosat Second Generation (MSG)", fontsize=11.5, fontweight="bold", color=COLOR_TEXT_MAIN)
ax_left.text(0.5, 8.40, "Sensor: SEVIRI (Spinning Enhanced Visible & IR)", fontsize=10.5, color=COLOR_TEXT_MUTED)

ax_img_uk = ax_left.inset_axes([left_frac, bottom_frac, w_frac, h_frac])
ax_img_uk.imshow(img_uk)
ax_img_uk.set_xticks([])
ax_img_uk.set_yticks([])
for spine in ax_img_uk.spines.values():
    spine.set_edgecolor(COLOR_BORDER)
    spine.set_linewidth(1.1)

ax_left.text(5.0, 3.65, "Native resolution: 128 × 128 px  |  3 channels (RGB)", fontsize=9.8, fontweight="bold", color=COLOR_TEXT_MUTED, ha="center")

y_pos = 3.12
for title, val in specs_uk:
    ax_left.text(0.6, y_pos, f"• {title}:", fontsize=10.2, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax_left.text(0.8, y_pos - 0.26, val, fontsize=9.3, color=COLOR_TEXT_MUTED)
    y_pos -= 0.58

# --- US Card ---
ax_right.text(0.5, 9.35, "  US TRACK (goes_pvdaq)  ", fontsize=13.5, fontweight="bold", color=COLOR_ACCENT_US,
              bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.3", facecolor=COLOR_PILL_US, edgecolor=COLOR_ACCENT_US, linewidth=1.2))
ax_right.text(0.5, 8.75, "NOAA GOES-16 (GOES-East, 75.2°W)", fontsize=11.5, fontweight="bold", color=COLOR_TEXT_MAIN)
ax_right.text(0.5, 8.40, "Sensor: ABI (Advanced Baseline Imager)", fontsize=10.5, color=COLOR_TEXT_MUTED)

ax_img_us = ax_right.inset_axes([left_frac, bottom_frac, w_frac, h_frac])
ax_img_us.imshow(img_us)
ax_img_us.set_xticks([])
ax_img_us.set_yticks([])
for spine in ax_img_us.spines.values():
    spine.set_edgecolor(COLOR_BORDER)
    spine.set_linewidth(1.1)

ax_right.text(5.0, 3.65, "Native resolution: 256 × 256 px  |  3 channels (RGB)", fontsize=9.8, fontweight="bold", color=COLOR_TEXT_MUTED, ha="center")

y_pos = 3.12
for title, val in specs_us:
    ax_right.text(0.6, y_pos, f"• {title}:", fontsize=10.2, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax_right.text(0.8, y_pos - 0.26, val, fontsize=9.3, color=COLOR_TEXT_MUTED)
    y_pos -= 0.58

comp_fig_path = out_dir / "satellite_comparison_uk_us.png"
fig.savefig(comp_fig_path, dpi=300, facecolor=COLOR_BG)
fig.savefig(scratch_dir / "satellite_comparison_uk_us.png", dpi=300, facecolor=COLOR_BG)
plt.close(fig)
print(f"Saved side-by-side comparison figure to {comp_fig_path}")

# --- 4. Individual Presentation Cards ---
def make_single_card(img, track_name, sat_name, sensor_name, res_str, specs, accent_color, pill_bg, filename):
    fig_card = plt.figure(figsize=(7.5, 8.5), facecolor=COLOR_BG)
    ax = fig_card.add_subplot(1, 1, 1)
    ax.set_facecolor(COLOR_CARD_BG)
    for s in ax.spines.values():
        s.set_edgecolor(COLOR_BORDER)
        s.set_linewidth(1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Header badge
    ax.text(0.5, 9.35, f"  {track_name}  ", fontsize=13.5, fontweight="bold", color=accent_color,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.3", facecolor=pill_bg, edgecolor=accent_color, linewidth=1.2))
    ax.text(0.5, 8.75, sat_name, fontsize=11.5, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax.text(0.5, 8.40, sensor_name, fontsize=10.5, color=COLOR_TEXT_MUTED)
    
    h_f = 0.38
    w_f = h_f * (8.5 / 7.5)
    l_f = (1.0 - w_f) / 2.0
    ax_im = ax.inset_axes([l_f, 0.40, w_f, h_f])
    ax_im.imshow(img)
    ax_im.set_xticks([])
    ax_im.set_yticks([])
    for s in ax_im.spines.values():
        s.set_edgecolor(COLOR_BORDER)
        s.set_linewidth(1.1)
        
    ax.text(5.0, 3.65, res_str, fontsize=9.8, fontweight="bold", color=COLOR_TEXT_MUTED, ha="center")
    
    y = 3.12
    for title, val in specs:
        ax.text(0.6, y, f"• {title}:", fontsize=10.2, fontweight="bold", color=COLOR_TEXT_MAIN)
        ax.text(0.8, y - 0.26, val, fontsize=9.3, color=COLOR_TEXT_MUTED)
        y -= 0.58
        
    p = out_dir / filename
    fig_card.savefig(p, dpi=300, facecolor=COLOR_BG)
    fig_card.savefig(scratch_dir / filename, dpi=300, facecolor=COLOR_BG)
    plt.close(fig_card)
    print(f"Saved card {p}")

make_single_card(img_uk, "UK TRACK (uk_pv)", "EUMETSAT Meteosat Second Generation (MSG)", 
                 "Sensor: SEVIRI (Spinning Enhanced Visible & IR)", "Native resolution: 128 × 128 px  |  3 channels (RGB)",
                 specs_uk, COLOR_ACCENT_UK, COLOR_PILL_UK, "satellite_uk_card.png")

make_single_card(img_us, "US TRACK (goes_pvdaq)", "NOAA GOES-16 (GOES-East, 75.2°W)", 
                 "Sensor: ABI (Advanced Baseline Imager)", "Native resolution: 256 × 256 px  |  3 channels (RGB)",
                 specs_us, COLOR_ACCENT_US, COLOR_PILL_US, "satellite_us_card.png")
