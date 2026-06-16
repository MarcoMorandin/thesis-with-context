"""Export uk_pv multimodal windows into the on-disk formats the *original*
gated Tier-5 multimodal models consume — so they run on uk data unmodified.

Two of the four Tier-5 vendors render the series as a pseudo-image and already
run on uk_pv (Time-VLM, VisionTS++). The other two were "gated on the multimodal
track"; with the satellite frames (`images_all.h5`) available, this exporter
unblocks them by reusing `tier6.uk_multimodal.UKMultimodalDataset` (shared Y + V
+ covariates bridge, same disjoint plant splits as every other tier):

- **UniCast** (`--model unicast`) — needs **real images** (CLIP/BLIP vision
  soft-prompt). Emits its native layout per split: `inputs.pt`/`targets_<H>.pt`
  dicts keyed by window + an `img/<key>.png` sky frame, consumed verbatim by
  `tier5/vendor/unicast/{train,test}_multi_modal_chronos.py`.
- **Aurora** (`--model aurora`) — its dataset (`Aurora_Single_Dataset`) is
  **time-series + text**, NOT images: a per-series CSV (`date` + value) plus a
  matching JSON text list. Images do not apply to Aurora; this emits the CSV +
  templated weather text generated from the uk covariates so Aurora is unblocked
  on the same data.

    uv run python tier5/uk_export.py --model unicast --out /tmp/unicast_ukpv
    uv run python tier5/uk_export.py --model aurora  --out /tmp/aurora_ukpv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # baselines/ on path

from common import config  # noqa: E402
from tier6.uk_multimodal import UKMultimodalDataset, sites_for_split  # noqa: E402

DATASET_TEXT = ("Photovoltaic power output of a solar plant under varying sky and "
                "cloud conditions; forecast the normalized power for the next steps.")


# --------------------------------------------------------------------------- #
# UniCast: inputs.pt / targets_<H>.pt / img/<key>.png  (needs real frames)
# --------------------------------------------------------------------------- #
def _save_unicast_split(ds: UKMultimodalDataset, out: Path, H: int) -> int:
    from PIL import Image

    (out / "img").mkdir(parents=True, exist_ok=True)
    inputs, targets, n = {}, {}, 0
    for i in range(len(ds)):
        it = ds[i]
        mv = it["mask_visual"]
        if mv.sum() == 0:                      # vision model needs ≥1 real frame
            continue
        j = int(np.where(mv > 0)[0][-1])       # most-recent observed sky frame
        frame = (np.clip(it["V"][j, 0], 0.0, 1.0) * 255).astype(np.uint8)
        key = f"{it['site_id']}_{i}.png"
        Image.fromarray(frame).convert("RGB").save(out / "img" / key)
        inputs[key] = np.asarray(it["y_hist"], np.float32)
        targets[key] = np.asarray(it["y_future"][:H], np.float32)
        n += 1
    import torch

    torch.save({k: torch.tensor(v) for k, v in inputs.items()}, out / "inputs.pt")
    torch.save({k: torch.tensor(v) for k, v in targets.items()}, out / f"targets_{H}.pt")
    return n


def export_unicast(out: Path, H: int, img_size: int, stride: int, data: str, h5: str) -> None:
    def mk(sites):
        return UKMultimodalDataset(site_ids=sites, data_path=data, h5_path=h5,
                                   horizon=H, stride=stride, img_size=img_size)

    for part in ("train", "val"):
        n = _save_unicast_split(mk(sites_for_split(part)), out / part, H)
        print(f"  unicast/{part}: {n} windows")
    for site in sites_for_split("test"):              # per-plant test sets (per-site npz)
        n = _save_unicast_split(mk([site]), out / f"test_{site}", H)
        print(f"  unicast/test_{site}: {n} windows")
    (out / "dataset_text.txt").write_text(DATASET_TEXT)


# --------------------------------------------------------------------------- #
# Aurora: <series>.csv (date,value) + <series>.json (text list)  (TS + text)
# --------------------------------------------------------------------------- #
def _weather_text(cov_row: np.ndarray) -> str:
    """One templated sentence from the (scaled) covariate means of a window."""
    idx = {c: k for k, c in enumerate(config.COV_COLS)}
    cloud = float(cov_row[idx["cloudcover"]]) * config.COV_SCALES["cloudcover"]
    temp = float(cov_row[idx["temperature_2m"]]) * config.COV_SCALES["temperature_2m"]
    sw = float(cov_row[idx["shortwave_radiation"]]) * config.COV_SCALES["shortwave_radiation"]
    sky = "overcast" if cloud > 70 else "partly cloudy" if cloud > 30 else "clear"
    return (f"Sky is {sky} ({cloud:.0f}% cloud cover), air temperature {temp:.0f}C, "
            f"shortwave irradiance {sw:.0f} W/m2.")


def export_aurora(out: Path, H: int, stride: int, data: str, h5: str) -> None:
    import pandas as pd

    ts_dir = out / "path_to_your_datasets"
    txt_dir = out / "path_to_your_corresponding_text"
    ts_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    interval = 2000  # Aurora_Single_Dataset maps text by index // interval

    for part in ("train", "val", "test"):
        for site in sites_for_split(part):
            ds = UKMultimodalDataset(site_ids=[site], data_path=data, h5_path=h5,
                                     horizon=H, stride=1, img_size=8)  # tiny V (unused)
            if len(ds) == 0:
                continue
            series, texts, ds_name = [], [], "uk_pv"
            for i in range(len(ds)):
                it = ds[i]
                ds_name = str(it["dataset"])
                # contiguous series: emit the per-window future head so the CSV is
                # a regular value column Aurora windows over (date is a placeholder).
                series.append(float(it["y_future"][0]))
                if i % interval == 0:
                    texts.append({"text": _weather_text(it["cov"][:config.HISTORY_STEPS].mean(0))})
            freq = "15min" if ds_name == "goes_pvdaq" else "30min"
            dates = pd.date_range("2019-01-01", periods=len(series), freq=freq)
            tag = f"{ds_name}_{site}_{part}"
            pd.DataFrame({"date": dates, "norm_power": series}).to_csv(
                ts_dir / f"{tag}.csv", index=False)
            (txt_dir / f"{tag}.json").write_text(json.dumps(texts or [{"text": DATASET_TEXT}]))
        print(f"  aurora/{part}: exported plant CSVs + text")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["unicast", "aurora"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=config.DEFAULT_DATA_PATH)
    ap.add_argument("--h5", default=config.DEFAULT_IMAGES_H5)
    ap.add_argument("--pred_len", type=int, default=config.HORIZON_STEPS)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.model == "unicast":
        export_unicast(out, args.pred_len, args.img_size, args.stride, args.data, args.h5)
    else:
        export_aurora(out, args.pred_len, args.stride, args.data, args.h5)
    print(f"✓ {args.model} uk export → {out}")


if __name__ == "__main__":
    main()
