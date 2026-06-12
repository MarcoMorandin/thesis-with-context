# tools/precompute_vision_feats_qwen3vl.py
import os
import re
import glob
import argparse
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch


def _get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        current_dir,
        os.path.dirname(current_dir),
        os.path.dirname(os.path.dirname(current_dir)),
    ]
    for base in candidates:
        if os.path.isdir(os.path.join(base, 'dataset')) or os.path.isfile(os.path.join(base, 'README.md')):
            return base
    return current_dir


PROJECT_ROOT = _get_project_root()

_HPC_SCRATCH = os.environ.get('SOLARVLM_SCRATCH', '')
SCRATCH_ROOT = _HPC_SCRATCH if _HPC_SCRATCH else PROJECT_ROOT


# ============== 1) 站点列表与经纬度 ==============
STATIONS = [
    'station00','station01','station02','station04',
    'station06','station07','station08','station09'
]

# 你提供的真实经纬度 (lon, lat)
STATION_LONLAT = {
    'station00': (114.95139, 38.04778),
    'station01': (117.45722, 38.18306),
    'station02': (114.19887, 38.05728),
    'station04': (114.86767, 39.51550),
    'station06': (114.54841, 36.89891),
    'station07': (113.64187, 36.64403),
    'station08': (113.89999, 36.70761),
    'station09': (115.059855, 38.731417),
}

# 生成 PNG 时使用的默认经纬度范围（可通过参数覆盖）
DEFAULT_LON_MIN, DEFAULT_LON_MAX = 109.10, 120.26
DEFAULT_LAT_MIN, DEFAULT_LAT_MAX = 32.28, 43.49


# ============== 工具函数 ==============
def extract_ts_key(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.findall(r'\d+', stem)
    return ''.join(m) if m else stem


def log_bad_image(bad_file_path: str, log_path: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(bad_file_path + "\n")


def station_to_pixel(station_id: str, W: int, H: int,
                     lon_min: float, lon_max: float, lat_min: float, lat_max: float):
    """
    站点经纬度 -> 像素坐标 (cx, cy)

    你生成 PNG 时 lat 是从 lat_max(北) 到 lat_min(南) 排列，所以 y=0 对应 lat_max。
    rel_y = (lat_max - lat) / (lat_max - lat_min)
    """
    if station_id not in STATION_LONLAT:
        return None
    lon, lat = STATION_LONLAT[station_id]

    if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
        return None

    rel_x = (lon - lon_min) / (lon_max - lon_min)
    rel_y = (lat_max - lat) / (lat_max - lat_min)

    cx = int(np.clip(rel_x * (W - 1), 0, W - 1))
    cy = int(np.clip(rel_y * (H - 1), 0, H - 1))
    return cx, cy


def extract_roi(img: Image.Image, station_id: str, roi_size: int,
                lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> Image.Image:
    W, H = img.size
    p = station_to_pixel(station_id, W, H, lon_min, lon_max, lat_min, lat_max)
    if p is None:
        return Image.new("RGB", (roi_size, roi_size), (0, 0, 0))

    cx, cy = p
    half = roi_size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + roi_size, y0 + roi_size

    pad = Image.new("RGB", (roi_size, roi_size), (0, 0, 0))

    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(W, x1), min(H, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return pad

    roi = img.crop((ix0, iy0, ix1, iy1))
    px0, py0 = ix0 - x0, iy0 - y0
    pad.paste(roi, (px0, py0))
    return pad


# ============== 2) Qwen3-VL-Embedding 编码 ==============
@torch.no_grad()
def qwen_encode_images(embedder, images, batch_size: int = 32, normalize: bool = True):
    outs = []
    for s in range(0, len(images), batch_size):
        chunk = images[s:s+batch_size]
        inputs = [{"image": im, "text": ""} for im in chunk]
        emb = embedder.process(inputs, normalize=normalize)  # [bs, D]
        outs.append(emb.detach().cpu())
    feats = torch.cat(outs, dim=0).float().numpy().astype("float32")
    return feats


def load_qwen_embedder(qwen_path: str, device: torch.device, dtype: torch.dtype):
    try:
        from src.SolarVLM.qwen3_vl_embedding import Qwen3VLEmbedder
    except Exception as e:
        raise RuntimeError(
            "Failed to import src.SolarVLM.qwen3_vl_embedding.Qwen3VLEmbedder.\n"
            f"Original error: {e}"
        )

    embedder = Qwen3VLEmbedder(qwen_path, torch_dtype=dtype)
    embedder.model.to(device).eval()
    return embedder


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--image_root", type=str,
                    default=os.path.join(SCRATCH_ROOT, 'dataset', 'image_trash', 'all_image'))
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(SCRATCH_ROOT, 'vision_feats_qwen3vl'))

    ap.add_argument("--qwen_path", type=str,
                    default=os.path.join(SCRATCH_ROOT, 'QwenQwen3-VL-Embedding-2B'))

    ap.add_argument("--roi_size", type=int, default=128,
                    help="建议 >=64，推荐 128（16 太小基本无效）")
    ap.add_argument("--n_frames", type=int, default=1)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_images", type=int, default=32)
    ap.add_argument("--normalize", type=int, default=1)
    ap.add_argument("--fp16", type=int, default=1)

    # 经纬度范围：默认与你生成 PNG 时一致
    ap.add_argument("--lonmin", type=float, default=DEFAULT_LON_MIN)
    ap.add_argument("--lonmax", type=float, default=DEFAULT_LON_MAX)
    ap.add_argument("--latmin", type=float, default=DEFAULT_LAT_MIN)
    ap.add_argument("--latmax", type=float, default=DEFAULT_LAT_MAX)

    args = ap.parse_args()

    lon_min, lon_max = args.lonmin, args.lonmax
    lat_min, lat_max = args.latmin, args.latmax

    os.makedirs(args.out_dir, exist_ok=True)

    bad_log_path = os.path.join(os.path.dirname(args.out_dir), "bad_images.txt")
    if not os.path.exists(bad_log_path):
        with open(bad_log_path, "w", encoding="utf-8") as f:
            f.write("# 预计算过程中无法正常读取的图片（会用黑图兜底继续运行）\n")

    print("[1/3] 枚举并排序所有图像文件...")
    img_files = sorted(glob.glob(os.path.join(args.image_root, "*.*")))
    assert img_files, f"在 {args.image_root} 下没有找到图像文件"
    ts_keys = [extract_ts_key(p) for p in img_files]

    print("[2/3] 加载 Qwen3-VL-Embedding-2B...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.float32
    embedder = load_qwen_embedder(args.qwen_path, device=device, dtype=dtype)

    EMB_DIM_FALLBACK = 2048
    normalize = bool(args.normalize)

    print("[3/3] 逐时间点预计算 8×D 特征（D≈2048）...")
    for i, (path, ts) in enumerate(zip(img_files, ts_keys)):
        out_path = os.path.join(args.out_dir, f"{ts}.npy")
        if os.path.exists(out_path):
            continue

        start = max(0, i - (args.n_frames - 1))
        frame_paths = img_files[start:i+1]
        while len(frame_paths) < args.n_frames:
            frame_paths = [frame_paths[0]] + frame_paths

        current_time_bad = False
        rois_by_frame = []

        for fp in frame_paths:
            try:
                img = Image.open(fp).convert("RGB")
                img.load()
            except UnidentifiedImageError:
                print(f"[WARNING] UnidentifiedImageError → 黑图替代: {fp}")
                log_bad_image(fp, bad_log_path)
                img = Image.new("RGB", (args.roi_size, args.roi_size), (0, 0, 0))
                current_time_bad = True
            except (OSError, IOError) as e:
                print(f"[WARNING] OSError/IOError ({e}) → 黑图替代: {fp}")
                log_bad_image(fp, bad_log_path)
                img = Image.new("RGB", (args.roi_size, args.roi_size), (0, 0, 0))
                current_time_bad = True
            except Exception as e:
                print(f"[ERROR] Unexpected error when opening {fp}: {e}")
                log_bad_image(fp + f" | Exception: {e}", bad_log_path)
                img = Image.new("RGB", (args.roi_size, args.roi_size), (0, 0, 0))
                current_time_bad = True

            rois = [
                extract_roi(img, sid, args.roi_size, lon_min, lon_max, lat_min, lat_max)
                for sid in STATIONS
            ]
            rois_by_frame.append(rois)

        # flatten：station-major（每站点 T 帧连续）
        flat_imgs = []
        for sidx in range(len(STATIONS)):
            for t in range(args.n_frames):
                flat_imgs.append(rois_by_frame[t][sidx])

        try:
            flat_feats = qwen_encode_images(
                embedder,
                flat_imgs,
                batch_size=args.batch_images,
                normalize=normalize
            )  # [8*T, D]
            D = flat_feats.shape[-1]
            arr = flat_feats.reshape(len(STATIONS), args.n_frames, D).astype("float32")
        except Exception as e:
            print(f"[ERROR] Qwen embedding failed for time {ts}: {e}")
            log_bad_image(f"{ts} | embedding error: {e}", bad_log_path)
            arr = np.zeros((len(STATIONS), args.n_frames, EMB_DIM_FALLBACK), dtype="float32")

        np.save(out_path, arr)

        if current_time_bad:
            print(f"[{i}/{len(img_files)}] saved (contains bad frames) → {out_path}  shape={arr.shape}")
        elif (i % 100) == 0:
            print(f"[{i}/{len(img_files)}] saved: {out_path}  shape={arr.shape}")

    print("✅ 预计算全部完成！")
    print(f"   特征保存在：{os.path.abspath(args.out_dir)}")
    print(f"   坏图记录在：{os.path.abspath(bad_log_path)}")


if __name__ == "__main__":
    main()
