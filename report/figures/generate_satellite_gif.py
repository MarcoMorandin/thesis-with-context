import h5py
import io
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from pathlib import Path

h5_path = "/Volumes/dataset/dataset/images_all.h5"
schema_path = "/Users/marcomorandin/.gemini/antigravity-cli/brain/91dccad1-5457-4684-9174-7756c94c9d24/.user_uploaded/uploaded_media_1791475130970.png"
out_dir = Path("report/figures")
out_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = Path("/Users/marcomorandin/.gemini/antigravity-cli/brain/91dccad1-5457-4684-9174-7756c94c9d24/scratch")

# Fonts
font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
font_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
font_badge_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)

site = "10793"
t_origin = pd.Timestamp("2020-05-22 12:00:00+00:00")
frame_times = [
    t_origin - pd.Timedelta(minutes=15) - pd.Timedelta(minutes=45*k)
    for k in range(7, -1, -1)
]

raw_frames = []
with h5py.File(h5_path, "r") as f:
    h5_site = f[f"uk_pv_{site}"]
    h5_ts = {ts.decode("utf-8") if isinstance(ts, bytes) else str(ts): i for i, ts in enumerate(h5_site["timestamps"])}
    for ft in frame_times:
        fts = ft.strftime("%Y-%m-%dT%H:%M:%SZ")
        idx = h5_ts[fts]
        raw_bytes = h5_site["images"][idx]
        img = Image.open(io.BytesIO(raw_bytes))
        raw_frames.append((ft, img))

# 1. Clean Standalone GIF (512x512)
clean_512 = [img.resize((512, 512), Image.Resampling.LANCZOS) for _, img in raw_frames]
p_clean = out_dir / "satellite_uk_8frames.gif"
clean_512[0].save(p_clean, save_all=True, append_images=clean_512[1:], duration=500, loop=0)
clean_512[0].save(scratch_dir / "satellite_uk_8frames.gif", save_all=True, append_images=clean_512[1:], duration=500, loop=0)

# 2. Annotated Standalone GIF (512x512)
annotated_512 = []
for i, (ft, img) in enumerate(raw_frames):
    im_scaled = img.resize((512, 512), Image.Resampling.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", (512, 512), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    time_str = ft.strftime("%H:%M UTC")
    badge_text = f"Frame {i+1}/8   |   {time_str}"
    bbox = draw.textbbox((0, 0), badge_text, font=font_bold)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 18, 9
    bw, bh = tw + pad_x * 2, th + pad_y * 2
    bx0, by0 = (512 - bw) // 2, 512 - bh - 24
    draw.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], radius=14, fill=(20, 44, 69, 215), outline=(255, 255, 255, 200), width=1)
    draw.text((bx0 + pad_x, by0 + pad_y - 2), badge_text, font=font_bold, fill=(255, 255, 255, 255))
    
    dt_origin_min = int((t_origin - ft).total_seconds() / 60)
    lead_str = f"t - {dt_origin_min}m" if dt_origin_min > 0 else "Origin (12:00)"
    bbox_l = draw.textbbox((0, 0), lead_str, font=font_badge_sm)
    lw, lh = bbox_l[2] - bbox_l[0], bbox_l[3] - bbox_l[1]
    lx0, ly0 = 512 - lw - 36, 18
    draw.rounded_rectangle([lx0, ly0, 512 - 16, ly0 + lh + 14], radius=8, fill=(20, 44, 69, 190), outline=(255, 255, 255, 140), width=1)
    draw.text((lx0 + 10, ly0 + 6), lead_str, font=font_badge_sm, fill=(235, 241, 247, 255))
    
    combined = Image.alpha_composite(im_scaled, overlay).convert("RGB")
    annotated_512.append(combined)

p_ann = out_dir / "satellite_uk_8frames_annotated.gif"
annotated_512[0].save(p_ann, save_all=True, append_images=annotated_512[1:], duration=500, loop=0)
annotated_512[0].save(scratch_dir / "satellite_uk_8frames_annotated.gif", save_all=True, append_images=annotated_512[1:], duration=500, loop=0)

# 3. Card-Fitted Clip (321 x 228) for box replacement
box_fitted = []
for i, (ft, img) in enumerate(raw_frames):
    card = Image.new("RGBA", (321, 228), (238, 244, 248, 255))
    draw = ImageDraw.Draw(card)
    
    hdr_text = "Latest 8-frame clip"
    bbox_h = draw.textbbox((0, 0), hdr_text, font=font_title)
    hw = bbox_h[2] - bbox_h[0]
    draw.text(((321 - hw) // 2, 10), hdr_text, font=font_title, fill=(20, 44, 69, 255))
    
    img_size = 142
    im_sq = img.resize((img_size, img_size), Image.Resampling.LANCZOS).convert("RGBA")
    mask = Image.new("L", (img_size, img_size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img_size, img_size], radius=8, fill=255)
    
    im_x = (321 - img_size) // 2
    im_y = 38
    card.paste(im_sq, (im_x, im_y), mask)
    draw.rounded_rectangle([im_x, im_y, im_x + img_size, im_y + img_size], radius=8, outline=(136, 168, 192, 255), width=1)
    
    time_str = ft.strftime("%H:%M UTC")
    lbl = f"Frame {i+1}/8   •   {time_str}"
    bbox_f = draw.textbbox((0, 0), lbl, font=font_small)
    fw = bbox_f[2] - bbox_f[0]
    draw.text(((321 - fw) // 2, 192), lbl, font=font_small, fill=(98, 116, 135, 255))
    
    draw.rounded_rectangle([0, 0, 320, 227], radius=14, outline=(190, 208, 224, 255), width=2)
    box_fitted.append(card.convert("RGB"))

p_box = out_dir / "satellite_uk_8frames_card.gif"
box_fitted[0].save(p_box, save_all=True, append_images=box_fitted[1:], duration=500, loop=0)
box_fitted[0].save(scratch_dir / "satellite_uk_8frames_card.gif", save_all=True, append_images=box_fitted[1:], duration=500, loop=0)

# 4. Animated Full Schema Diagram
if Path(schema_path).exists():
    schema_base = Image.open(schema_path).convert("RGBA")
    schema_animated_frames = []
    for card_frame in box_fitted:
        frame_canvas = schema_base.copy()
        frame_canvas.paste(card_frame.convert("RGBA"), (29, 44))
        schema_animated_frames.append(frame_canvas.convert("RGB"))

    p_schema_gif = out_dir / "schema_vjepa_animated.gif"
    schema_animated_frames[0].save(p_schema_gif, save_all=True, append_images=schema_animated_frames[1:], duration=500, loop=0)
    schema_animated_frames[0].save(scratch_dir / "schema_vjepa_animated.gif", save_all=True, append_images=schema_animated_frames[1:], duration=500, loop=0)

    p_schema_png = out_dir / "schema_vjepa_with_satellite.png"
    schema_animated_frames[0].save(p_schema_png)
    schema_animated_frames[0].save(scratch_dir / "schema_vjepa_with_satellite.png")

print("Generated reproducible GIF assets in report/figures/")
