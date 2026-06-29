"""Dataset-of-record backend (``uk_pv`` / ``goes_pvdaq``) aligned to the baseline
protocol (BASELINE_PROTOCOL.md).

This is the bridge that makes MMTSFM *comparable* to every baseline: it reuses
``baselines/common`` for all numerical logic — the committed disjoint plant
splits (``configs/splits.json``), physical-time windows (14-day history /
6-hour horizon, resolved per cadence), the protocol covariate set and its fixed
scalings — and reads satellite frames straight from ``images_all.h5`` by the
canonical ``image_h5_index`` pointer. No data ETL happens here, so MMTSFM and
the baselines share one fairness contract.

Future weather is exposed as *known* (``future_cov="all"``): per the project
decision the deployable setting treats NWP weather as available at inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# --- bootstrap: put repo-root/baselines on sys.path so `common` resolves -------
def _baselines_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "baselines" / "common" / "windows.py").exists():
            return parent / "baselines"
    raise RuntimeError("pv_record: could not locate baselines/ from MMTSFM")


_BL = _baselines_dir()
if str(_BL) not in sys.path:
    sys.path.insert(0, str(_BL))

import pandas as pd  # noqa: E402
from common import config  # noqa: E402
from common.splits import load_splits  # noqa: E402
from common.windows import dataset_for_sites  # noqa: E402

FRAME_IDX_COL = config.FRAME_INDEX_COL

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _normalize_part(split: str) -> str:
    return "val" if split in ("val", "validation") else split


def _steps_per_day(df: pd.DataFrame) -> int:
    site = str(df[config.SITE_COL].iloc[0])
    g = df[df[config.SITE_COL] == site].sort_values(config.TIME_COL)
    t = pd.DatetimeIndex(g[config.TIME_COL].drop_duplicates())
    step = t.to_series().diff().median()
    return int(round(pd.Timedelta(days=1) / step))


def _prep_frame(arr: np.ndarray, side: int, c_img: int, imagenet_norm: bool) -> torch.Tensor:
    """Raw uint8 H5 frame → float tensor (c_img, side, side) in [0, 1].

    Handles uk_pv ``(H, W)`` grayscale and goes_pvdaq ``(H, W, 3)`` RGB; resizes
    with PIL (up- or down-sampling) and maps native channels to ``c_img``.
    """
    from PIL import Image

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    native_c = arr.shape[-1]
    bands = []
    for ch in range(native_c):
        band = arr[..., ch].astype(np.float32)
        if band.max() > 1.0:
            band = band / 255.0
        if band.shape != (side, side):
            im = Image.fromarray((band * 255).astype(np.uint8)).resize(
                (side, side), Image.LANCZOS
            )
            band = np.asarray(im, dtype=np.float32) / 255.0
        bands.append(band)
    t = torch.from_numpy(np.stack(bands, axis=0))  # (native_c, side, side)
    if c_img == native_c:
        pass
    elif native_c == 1:  # gray → replicate to c_img
        t = t.expand(c_img, side, side).contiguous()
    elif c_img == 1:  # rgb → luminance
        t = t.mean(dim=0, keepdim=True)
    elif c_img < native_c:
        t = t[:c_img]
    else:  # c_img > native_c > 1: pad by repeating first channel
        t = torch.cat([t, t[:1].expand(c_img - native_c, side, side)], dim=0)
    if imagenet_norm and c_img == 3:
        t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t


class PVRecordDataset(Dataset):
    """Sliding multimodal windows (Y, X_cov, V) over one split's plants.

    Emits the MMTSFM canonical dict (single entity, N=1) so the existing
    ``VisionChronos2LightningModule._unpack_batch`` consumes it unchanged.
    """

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "uk_pv",
        data_dir: str | None = None,
        data_path: str | None = None,
        h5_path: str | None = None,
        history_days: float = config.HISTORY_DAYS,
        horizon_hours: float = config.HORIZON_HOURS,
        hist_steps: int | None = None,
        horizon: int | None = None,
        video_frames: int = 8,
        img_size: int = 224,
        img_channels: int = 3,
        imagenet_norm: bool = True,
        stride: int | None = None,
        future_cov: str = "all",
        **_ignored,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.T_v = int(video_frames)
        self.img_size = int(img_size)
        self.C_img = int(img_channels)
        self.imagenet_norm = bool(imagenet_norm)

        # Resolve a data directory (the run_all_baselines.sh DATA_DIR convention)
        # to the canonical parquet + h5; an explicit *.parquet path is used as-is.
        raw = Path(data_path or data_dir or config.DEFAULT_DATA_PATH)
        if raw.suffix == ".parquet":
            data_path, base = str(raw), raw.parent
        else:
            data_path, base = str(raw / "dataset_all.parquet"), raw
        self.h5_path = str(h5_path or base / "images_all.h5")
        splits = load_splits()
        if dataset_name not in splits:
            raise ValueError(
                f"pv_record: dataset {dataset_name!r} not in splits {sorted(splits)}"
            )
        site_ids = {str(s) for s in splits[dataset_name][_normalize_part(split)]}

        cols = sorted({
            config.DATASET_COL, config.SITE_COL, config.TIME_COL,
            config.TARGET_COL, config.CAPACITY_COL, config.CLEARSKY_COL,
            config.BAD_SITE_COL, FRAME_IDX_COL, *config.COV_COLS,
        })
        df = pd.read_parquet(data_path, columns=cols)
        df = df[df[config.DATASET_COL] == dataset_name].copy()
        df[config.SITE_COL] = df[config.SITE_COL].astype(str)
        df = df[df[config.SITE_COL].isin(site_ids)]
        if df.empty:
            raise ValueError(f"pv_record: no rows for {dataset_name} {split} plants")

        spd = _steps_per_day(df)
        self.T = int(hist_steps) if hist_steps else int(round(history_days * spd))
        self.H = int(horizon) if horizon else int(round(horizon_hours / 24.0 * spd))
        if stride is None:
            stride = 1 if split == "train" else self.H

        self.win = dataset_for_sites(
            df, site_ids, history=self.T, horizon=self.H,
            stride=stride, future_cov=future_cov,
        )

        # per-(dataset, site) {unix_ts -> frame_idx} from the canonical pointer
        fdf = df[df[FRAME_IDX_COL].notna() & (df[FRAME_IDX_COL] >= 0)]
        self.frame_maps: dict[tuple[str, str], dict[int, int]] = {}
        for (ds, site), g in fdf.groupby([config.DATASET_COL, config.SITE_COL]):
            ts = pd.DatetimeIndex(g[config.TIME_COL])
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            unix = ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")).to_numpy()
            idx = g[FRAME_IDX_COL].to_numpy().astype(np.int64)
            self.frame_maps[(str(ds), str(site))] = dict(
                zip(unix.astype(np.int64).tolist(), idx.tolist())
            )
        self._h5 = None  # opened lazily (h5py handles are not fork-safe)

    def __len__(self) -> int:
        return len(self.win)

    def _group(self, dataset: str, site: str):
        import h5py

        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5[f"{dataset}_{site}"]

    def _load_vision(self, item: dict) -> tuple[torch.Tensor, torch.Tensor]:
        T, Tv, S, C = self.T, self.T_v, self.img_size, self.C_img
        key = (str(item["dataset"]), str(item["site_id"]))
        fmap = self.frame_maps.get(key, {})
        hist_ts = item["timestamps"][:T]
        present = [int(t) for t in hist_ts.tolist() if int(t) in fmap]
        sel = sorted(present[-Tv:])  # most recent Tv frame-bearing steps

        V = torch.zeros(1, Tv, C, S, S)
        mask_v = torch.zeros(1, Tv)
        if sel:
            g = self._group(*key)
            images = g["images"]
            for j, t in enumerate(sel):
                V[0, j] = _prep_frame(images[fmap[t]], S, C, self.imagenet_norm)
                mask_v[0, j] = 1.0
        return V, mask_v

    def __getitem__(self, idx: int) -> dict:
        item = self.win[idx]
        T, H = self.T, self.H
        cov = np.asarray(item["cov"], dtype=np.float32)  # (T+H, C) future weather known
        V, mask_visual = self._load_vision(item)
        return {
            "Y": torch.from_numpy(np.asarray(item["y_hist"], np.float32)).view(1, T, 1),
            "Y_future": torch.from_numpy(
                np.asarray(item["y_future"], np.float32)
            ).view(1, H, 1),
            "X_cov": torch.from_numpy(cov).view(1, T + H, -1),
            "V": V,
            "timestamps": torch.from_numpy(
                np.asarray(item["timestamps"], np.int64)
            ).long(),
            "entity_ids": torch.zeros(1, dtype=torch.long),
            "mask_target": torch.from_numpy(
                np.asarray(item["mask_hist"], np.float32)
            ).view(1, T, 1),
            "mask_future": torch.from_numpy(
                np.asarray(item["mask_future"], np.float32)
            ).view(1, H, 1),
            "mask_visual": mask_visual,
            "mask_modality_dropout": torch.tensor([[1.0, float(mask_visual.any())]]),
            "adj_matrix": torch.eye(1),
        }
