"""
MMTSFMDataset — multimodal spatiotemporal dataset for Vision-Time FM.

Supported dataset_name values
──────────────────────────────
"synthetic"       Random tensors matching the canonical schema (fast smoke-tests).
"skippd"          Refactored SKIPP'D  (data/refactored/solar/skippd/).
"solarnet"        Refactored Solarnet (data/refactored/solar/solarnet/).
"goes16_nsrdb"    Refactored GOES-16 + NSRDB (data/refactored/solar/goes16_nsrdb/).
"earthnet2021"    Refactored EarthNet2021 (data/refactored/meteorology/earthnet2021/).
                  Per-entity frames; entity-based train/val/test split.
                  Default window_size (T+H=36) exceeds 30 steps/entity → num_samples=0;
                  use hist_steps≤20, horizon≤10 for non-zero samples.
"era5_eu"         Refactored ERA5 Europe subset (data/refactored/meteorology/era5_eu/).
                  Shared spatial frames; chronological split across 2924 timesteps.
"meteonet"        Refactored MeteoNet (data/refactored/meteorology/meteonet/).
                  Shared spatial frames; NPZ stores 'rainfall'+'reflectivity' keys.

All modes return the same dict schema (Vision-Time FM canonical tensors):
    Y                    (N, T, C_target)     float32
    X_cov                (N, T+H, C_cov)      float32
    V                    (N, T_v, C_img, H_img, W_img)  float32  [0, 1]
    timestamps           (T+H,)               int64
    entity_ids           (N,)                 int64
    timestamps_v         (T_v,)               int64
    mask_target          (N, T, C_target)     float32
    mask_visual          (N, T_v)             float32
    mask_modality_dropout (N, 2)              float32   [numeric, visual]
    adj_matrix           (N, N)               float32

Refactored data layout (produced by scripts/solar/build_all.py):
    data/refactored/solar/{dataset}/
        timeseries.parquet   — entity_id, timestamp_unix,
                               {target_cols}, {target_cols}_norm,
                               [{cov_cols}, {cov_cols}_norm],
                               mask_target, [mask_cov]
        frame_index.parquet  — entity_id, timestamp_unix, rel_path, mask_visual
        frames/              — JPEG (sky cam) or NPZ float16 (satellite)
        graph.json           — {nodes, edges, [adjacency_matrix]}
        metadata.json
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

# ImageNet normalisation constants
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_frame_index(path: Path) -> tuple[list[int], list[str], list[int]]:
    """
    Read frame_index.parquet → (timestamps, rel_paths, mask_visual) lists.
    Returns empty lists if file does not exist.
    mask_visual defaults to 1 for all rows if the column is absent (e.g. GOES-16).
    """
    if not path.exists():
        return [], [], []
    cols = ["timestamp_unix", "rel_path"]
    schema = pq.read_schema(path)
    has_mask = "mask_visual" in schema.names
    if has_mask:
        cols.append("mask_visual")
    tbl  = pq.read_table(path, columns=cols)
    ts   = tbl.column("timestamp_unix").to_pylist()
    rp   = tbl.column("rel_path").to_pylist()
    mask = tbl.column("mask_visual").to_pylist() if has_mask else [1] * len(ts)
    return ts, rp, mask


def _nearest_frame_idx(frame_ts: list[int], target: int) -> Optional[int]:
    """Binary-search nearest frame timestamp; returns None if frame_ts is empty."""
    if not frame_ts:
        return None
    pos = bisect_left(frame_ts, target)
    if pos == 0:
        return 0
    if pos >= len(frame_ts):
        return len(frame_ts) - 1
    before = pos - 1
    if abs(frame_ts[before] - target) <= abs(frame_ts[pos] - target):
        return before
    return pos


def _load_jpeg(path: Path, img_size: int, c_img: int) -> torch.Tensor:
    """Load a JPEG → float32 tensor (C, H, W) in [0, 1], resized to img_size."""
    img = Image.open(path).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3)
    t   = torch.from_numpy(arr).permute(2, 0, 1)           # (3, H, W)
    if c_img < 3:
        t = t[:c_img]
    elif c_img > 3:
        t = torch.cat([t, t[:1].expand(c_img - 3, -1, -1)], dim=0)
    return t


def _load_npz(path: Path, img_size: int, c_img: int,
              npz_key: Optional[str] = "frame") -> torch.Tensor:
    """
    Load a float16 NPZ frame → float32 tensor (c_img, img_size, img_size).

    npz_key:
        Key to read from the NPZ (e.g. ``"frame"`` for GOES-16/EarthNet/ERA5).
        Pass ``None`` to stack all arrays in sorted key order (e.g. MeteoNet
        stores ``"rainfall"`` and ``"reflectivity"`` as separate 2-D arrays).
    """
    f = np.load(path)
    if npz_key is not None:
        data = f[npz_key].astype(np.float32)
        if data.ndim == 2:          # (H, W) → (1, H, W)
            data = data[np.newaxis]
    else:
        arrays = []
        for k in sorted(f.keys()):
            a = f[k].astype(np.float32)
            if a.ndim == 2:
                a = a[np.newaxis]
            arrays.append(a)
        data = np.concatenate(arrays, axis=0)   # (C_src, H, W)
    # data: (C_src, H, W)
    C_src, H, W = data.shape

    # Resize each band to img_size using PIL (nearest avoids float precision loss)
    if img_size > 0 and (H != img_size or W != img_size):
        resized = []
        for c in range(C_src):
            band = np.nan_to_num(data[c], nan=0.0, posinf=0.0, neginf=0.0)
            lo, hi = band.min(), band.max()
            band = ((band - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
            img = Image.fromarray(band).resize((img_size, img_size), Image.BILINEAR)
            resized.append(np.array(img, dtype=np.float32) / 255.0)
        data = np.stack(resized, axis=0)          # (C_src, img_size, img_size)
    else:
        # Normalise to [0, 1] per band
        for c in range(C_src):
            band = np.nan_to_num(data[c], nan=0.0, posinf=0.0, neginf=0.0)
            lo, hi = band.min(), band.max()
            data[c] = (band - lo) / (hi - lo + 1e-6)

    t = torch.from_numpy(data)                    # (C_src, H, W)
    if c_img <= C_src:
        t = t[:c_img]
    else:
        t = torch.cat([t, t[:1].expand(c_img - C_src, -1, -1)], dim=0)
    return t


def _chronological_split(n: int, split: str,
                         train_frac: float = 0.70,
                         val_frac:   float = 0.10) -> slice:
    n_train = int(n * train_frac)
    n_val   = int(n * (train_frac + val_frac))
    if split == "train":
        return slice(0, n_train)
    if split in ("val", "validation"):
        return slice(n_train, n_val)
    return slice(n_val, n)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MMTSFMDataset(Dataset):
    """
    Multimodal Spatiotemporal Foundation Model dataset.
    See module docstring for full schema documentation.
    """

    def __init__(
        self,
        num_samples:           int = 1000,
        num_entities:          int = 10,
        hist_steps:            int = 24,
        horizon:               int = 12,
        target_dim:            int = 1,
        covariate_dim:         int = 5,
        video_frames:          int = 8,
        img_channels:          int = 3,
        img_size:              int = 64,
        data_dir:              str = "./data",
        dataset_name:          str = "synthetic",
        split:                 str = "train",
        imagenet_norm:         bool = False,
        vidtok_cache_dir:      Optional[str] = None,
        train_frac:            float = 0.70,
        val_frac:              float = 0.10,
        vis_cadence_multiplier: int = 1,
    ):
        """
        vis_cadence_multiplier (int, default 1):
            C4 fix — Decoupled Resolution Architecture.
            Controls how many times denser the visual anchor cadence is compared
            to the numeric step cadence.  With the default value of 1 the
            behaviour is identical to the original code (visual anchors coincide
            with the last T_v numeric timestamps).  With multiplier=M the T_v
            anchors are spread across a numeric sub-interval of T_v/M steps,
            giving M× finer temporal resolution for video lookup.
            Example: numeric=1h-hourly, T_v=24, multiplier=4 → anchors every
            ~15 min over the last 6 numeric hours.
        """
        super().__init__()
        self.num_samples           = num_samples
        self.num_entities          = num_entities
        self.T                     = hist_steps
        self.H                     = horizon
        self.C_target              = target_dim
        self.C_cov                 = covariate_dim
        self.T_v                   = video_frames
        self.C_img                 = img_channels
        self.img_size              = img_size
        self.data_dir              = Path(data_dir)
        self.dataset_name          = dataset_name.lower()
        self.split                 = split
        self.imagenet_norm         = imagenet_norm
        self.window_size           = self.T + self.H
        self._cache_dir            = Path(vidtok_cache_dir) if vidtok_cache_dir else None
        self.vis_cadence_multiplier = max(1, vis_cadence_multiplier)

        if self.C_target != 1:
            raise ValueError("target_dim=1 required (single-variate targets per entity).")

        dispatch = {
            "skippd":        self._init_skippd,
            "solarnet":      self._init_solarnet,
            "goes16_nsrdb":  self._init_goes16_nsrdb,
            "earthnet2021":  self._init_earthnet2021,
            "era5_eu":       self._init_era5_eu,
            "meteonet":      self._init_meteonet,
        }
        if self.dataset_name in dispatch:
            dispatch[self.dataset_name]()
        else:
            # synthetic fallback
            self.adj_matrix = torch.eye(self.num_entities)

    # ------------------------------------------------------------------
    # C4: Decoupled visual anchor helper
    # ------------------------------------------------------------------

    def _visual_anchor_timestamps(self, ts_win: list) -> list:
        """Generate T_v evenly-spaced visual anchor timestamps.

        C4 fix: decouples the visual sampling cadence from the numeric step
        cadence.  The T_v anchors are derived by linearly interpolating
        within the last ``T_v // vis_cadence_multiplier`` numeric steps of
        the history window (at least 1 step), spread uniformly over that
        wall-clock interval.

        With ``vis_cadence_multiplier=1`` (default) this reproduces the
        original behaviour: anchors land on the last T_v numeric timestamps.
        With ``vis_cadence_multiplier=M`` the same wall-clock span is sampled
        M× more densely, giving finer temporal resolution for video lookup.

        Parameters
        ----------
        ts_win : list[int]
            Numeric timestamp window of length ``T + H`` (unix seconds or
            any consistent integer unit).

        Returns
        -------
        list[int]
            ``T_v`` anchor timestamps, ascending, within the history interval.
        """
        T_v = self.T_v
        M   = self.vis_cadence_multiplier

        # Number of numeric steps spanned by the visual window.
        # At least 1, at most T (history length).
        numeric_span = max(1, T_v // M)
        start_idx    = max(0, self.T - numeric_span)
        end_idx      = self.T - 1  # last history timestamp index (inclusive)

        t_start = ts_win[start_idx]
        t_end   = ts_win[end_idx]

        if T_v == 1:
            return [t_end]

        # Linearly interpolate T_v anchor timestamps in [t_start, t_end]
        step = (t_end - t_start) / (T_v - 1)
        return [int(round(t_start + i * step)) for i in range(T_v)]

    # ------------------------------------------------------------------
    # SKIPP'D  (refactored layout)
    # ------------------------------------------------------------------

    def _init_skippd(self) -> None:
        self.num_entities = 1
        self.adj_matrix   = torch.eye(1)

        base = self.data_dir / "solar" / "skippd"
        ts_path = base / "timeseries.parquet"
        fi_path = base / "frame_index.parquet"

        if not ts_path.exists():
            raise FileNotFoundError(
                f"SKIPP'D refactored timeseries not found: {ts_path}\n"
                "Run:  python scripts/solar/build_all.py --skip-solarnet --skip-goes16 --skip-unified"
            )

        tbl = pq.read_table(ts_path, columns=["timestamp_unix", "pv_power_norm", "mask_target"])
        self._skippd_ts     = tbl.column("timestamp_unix").to_pylist()
        self._skippd_target = [float(v) if v is not None else 0.0
                               for v in tbl.column("pv_power_norm").to_pylist()]

        # Frame index: sorted by timestamp for binary search
        fi_ts, fi_rp, fi_mask = _load_frame_index(fi_path)
        sort_order = sorted(range(len(fi_ts)), key=lambda i: fi_ts[i])
        self._skippd_fi_ts   = [fi_ts[i]   for i in sort_order]
        self._skippd_fi_rp   = [fi_rp[i]   for i in sort_order]
        self._skippd_fi_mask = [fi_mask[i]  for i in sort_order]
        self._skippd_frames_dir = base / "frames"

        sl = _chronological_split(len(self._skippd_ts), self.split)
        self.num_samples = max(0, len(range(*sl.indices(len(self._skippd_ts)))) - self.window_size + 1)
        self._skippd_sl = sl

    def _getitem_skippd(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self._skippd_sl.start + idx
        ts_win     = self._skippd_ts    [offset: offset + self.window_size]
        target_win  = self._skippd_target[offset: offset + self.T]
        future_win  = self._skippd_target[offset + self.T: offset + self.window_size]

        Y        = torch.tensor(target_win, dtype=torch.float32).view(1, self.T, 1)
        Y_future = torch.tensor(future_win, dtype=torch.float32).view(1, self.H, 1)
        X_cov = torch.zeros(1, self.window_size, self.C_cov)

        # C4 fix: visual anchors at decoupled (potentially denser) cadence
        vis_ts = self._visual_anchor_timestamps(ts_win)
        V            = torch.zeros(1, self.T_v, self.C_img, self.img_size, self.img_size)
        timestamps_v = torch.zeros(self.T_v, dtype=torch.long)
        mask_visual  = torch.zeros(1, self.T_v)

        for i, t in enumerate(vis_ts):
            fi_idx = _nearest_frame_idx(self._skippd_fi_ts, t)
            if fi_idx is None or not self._skippd_fi_mask[fi_idx]:
                continue
            rel = self._skippd_fi_rp[fi_idx]
            frame_path = self._skippd_frames_dir / Path(rel).name
            if not frame_path.exists():
                continue
            try:
                frame = _load_jpeg(frame_path, self.img_size, self.C_img)
                if self.imagenet_norm:
                    frame = (frame - _IMAGENET_MEAN) / _IMAGENET_STD
                V[0, i] = frame
                timestamps_v[i] = self._skippd_fi_ts[fi_idx]
                mask_visual[0, i] = 1.0
            except Exception:
                pass

        return {
            "Y":                     Y,
            "Y_future":              Y_future,
            "X_cov":                 X_cov,
            "V":                     V,
            "timestamps":            torch.tensor(ts_win, dtype=torch.long),
            "entity_ids":            torch.zeros(1, dtype=torch.long),
            "timestamps_v":          timestamps_v,
            "mask_target":           torch.ones(1, self.T, 1),
            "mask_future":           torch.ones(1, self.H, 1),
            "mask_visual":           mask_visual,
            "mask_modality_dropout": torch.tensor([[1.0, float(mask_visual.any())]]),
            "adj_matrix":            self.adj_matrix,
        }

    # ------------------------------------------------------------------
    # Solarnet  (refactored layout — single timeseries.parquet)
    # ------------------------------------------------------------------

    def _init_solarnet(self) -> None:
        self.num_entities = 1
        self.adj_matrix   = torch.eye(1)

        base    = self.data_dir / "solar" / "solarnet"
        ts_path = base / "timeseries.parquet"
        fi_path = base / "frame_index.parquet"

        if not ts_path.exists():
            raise FileNotFoundError(
                f"Solarnet refactored timeseries not found: {ts_path}\n"
                "Run:  python scripts/solar/build_all.py --skip-skippd --skip-goes16 --skip-unified"
            )

        tbl = pq.read_table(ts_path)
        self._solarnet_ts = tbl.column("timestamp_unix").to_pylist()

        # Target: prefer normalised GHI
        for col in ("ghi_norm", "ghi", "GHI_norm", "GHI"):
            if col in tbl.schema.names:
                self._solarnet_target = [float(v) if v is not None else 0.0
                                         for v in tbl.column(col).to_pylist()]
                break
        else:
            raise ValueError("No irradiance target column in Solarnet timeseries.parquet.")

        # Covariates: prefer *_norm columns, skip meta/mask cols
        _skip = {"entity_id", "timestamp_unix", "mask_target", "mask_cov"}
        cov_candidates = [
            c for c in tbl.schema.names
            if c not in _skip
            and not any(c == stem or c == f"{stem}_norm"
                        for stem in ("ghi", "dni", "dhi", "GHI", "DNI", "DHI"))
            and c.endswith("_norm")
        ]
        self._solarnet_covariates: list[list[float]] = []
        for col in cov_candidates[: self.C_cov]:
            self._solarnet_covariates.append(
                [float(v) if v is not None else 0.0
                 for v in tbl.column(col).to_pylist()]
            )
        while len(self._solarnet_covariates) < self.C_cov:
            self._solarnet_covariates.append([0.0] * len(self._solarnet_ts))

        # Frame index
        fi_ts, fi_rp, fi_mask = _load_frame_index(fi_path)
        sort_order = sorted(range(len(fi_ts)), key=lambda i: fi_ts[i])
        self._solarnet_fi_ts   = [fi_ts[i]  for i in sort_order]
        self._solarnet_fi_rp   = [fi_rp[i]  for i in sort_order]
        self._solarnet_fi_mask = [fi_mask[i] for i in sort_order]
        self._solarnet_frames_dir = base / "frames"

        sl = _chronological_split(len(self._solarnet_ts), self.split)
        self._solarnet_sl = sl
        self.num_samples = max(0, len(range(*sl.indices(len(self._solarnet_ts)))) - self.window_size + 1)

    def _getitem_solarnet(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self._solarnet_sl.start + idx
        ts_win     = self._solarnet_ts    [offset: offset + self.window_size]
        target_win  = self._solarnet_target[offset: offset + self.T]
        future_win  = self._solarnet_target[offset + self.T: offset + self.window_size]

        Y        = torch.tensor(target_win, dtype=torch.float32).view(1, self.T, 1)
        Y_future = torch.tensor(future_win, dtype=torch.float32).view(1, self.H, 1)
        X_cov = torch.zeros(1, self.window_size, self.C_cov)
        for ci, col in enumerate(self._solarnet_covariates):
            X_cov[0, :, ci] = torch.tensor(col[offset:offset + self.window_size], dtype=torch.float32)

        # C4 fix: visual anchors at decoupled (potentially denser) cadence
        vis_ts = self._visual_anchor_timestamps(ts_win)
        V            = torch.zeros(1, self.T_v, self.C_img, self.img_size, self.img_size)
        timestamps_v = torch.zeros(self.T_v, dtype=torch.long)
        mask_visual  = torch.zeros(1, self.T_v)

        for i, t in enumerate(vis_ts):
            fi_idx = _nearest_frame_idx(self._solarnet_fi_ts, t)
            if fi_idx is None or not self._solarnet_fi_mask[fi_idx]:
                continue
            rel = self._solarnet_fi_rp[fi_idx]
            frame_path = self._solarnet_frames_dir / Path(rel).name
            if not frame_path.exists():
                continue
            try:
                frame = _load_jpeg(frame_path, self.img_size, self.C_img)
                if self.imagenet_norm:
                    frame = (frame - _IMAGENET_MEAN) / _IMAGENET_STD
                V[0, i] = frame
                timestamps_v[i] = self._solarnet_fi_ts[fi_idx]
                mask_visual[0, i] = 1.0
            except Exception:
                pass

        return {
            "Y":                     Y,
            "Y_future":              Y_future,
            "X_cov":                 X_cov,
            "V":                     V,
            "timestamps":            torch.tensor(ts_win, dtype=torch.long),
            "entity_ids":            torch.zeros(1, dtype=torch.long),
            "timestamps_v":          timestamps_v,
            "mask_target":           torch.ones(1, self.T, 1),
            "mask_future":           torch.ones(1, self.H, 1),
            "mask_visual":           mask_visual,
            "mask_modality_dropout": torch.tensor([[1.0, float(mask_visual.any())]]),
            "adj_matrix":            self.adj_matrix,
        }

    # ------------------------------------------------------------------
    # GOES-16 + NSRDB  (refactored layout)
    # ------------------------------------------------------------------

    def _init_goes16_nsrdb(self) -> None:
        base     = self.data_dir / "solar" / "goes16_nsrdb"
        ts_path  = base / "timeseries.parquet"
        fi_path  = base / "frame_index.parquet"
        graph_p  = base / "graph.json"

        if not ts_path.exists():
            raise FileNotFoundError(
                f"GOES-16/NSRDB refactored timeseries not found: {ts_path}\n"
                "Run:  python scripts/solar/build_all.py --skip-skippd --skip-solarnet --skip-unified"
            )

        tbl = pq.read_table(ts_path)
        all_entity_ids = sorted(set(tbl.column("entity_id").to_pylist()))
        selected_ids   = all_entity_ids[: self.num_entities]
        self.num_entities        = len(selected_ids)
        self._nsrdb_entity_ids   = selected_ids

        # Identify target and covariate columns
        _skip_cols = {"entity_id", "lat", "lon", "timestamp_unix", "mask_target"}
        _target_raw = ("GHI", "ghi")
        _norm_suffix = "_norm"

        self._nsrdb_data: Dict[int, Dict] = {}
        for eid in selected_ids:
            mask_col = pc.equal(tbl.column("entity_id"), pa.scalar(eid, pa.int32()))
            etbl = tbl.filter(mask_col)
            sort_idx = pc.sort_indices(etbl, sort_keys=[("timestamp_unix", "ascending")])
            etbl = etbl.take(sort_idx)

            ts_list = etbl.column("timestamp_unix").to_pylist()

            # Target: prefer GHI_norm
            target_list: list[float] = []
            for col in ("GHI_norm", "ghi_norm", "GHI", "ghi"):
                if col in etbl.schema.names:
                    target_list = [float(v) if v is not None else 0.0
                                   for v in etbl.column(col).to_pylist()]
                    break
            if not target_list:
                target_list = [0.0] * len(ts_list)

            # Covariates: use *_norm columns that are not the target
            cov_candidates = [
                c for c in etbl.schema.names
                if c not in _skip_cols
                and c.endswith(_norm_suffix)
                and not any(c == f"{t}{_norm_suffix}" for t in ("GHI", "ghi", "DNI", "DHI"))
            ]
            cov_data: list[list[float]] = []
            for col in cov_candidates[: self.C_cov]:
                cov_data.append([float(v) if v is not None else 0.0
                                  for v in etbl.column(col).to_pylist()])
            while len(cov_data) < self.C_cov:
                cov_data.append([0.0] * len(ts_list))

            if "mask_target" in etbl.schema.names:
                mt_list = [float(v) if v is not None else 1.0
                           for v in etbl.column("mask_target").to_pylist()]
            else:
                mt_list = [1.0] * len(ts_list)

            self._nsrdb_data[eid] = {
                "timestamps":   ts_list,
                "target":       target_list,
                "covariates":   cov_data,
                "mask_target":  mt_list,
            }

        # All selected entities share the same timestamp grid
        self._nsrdb_timestamps = self._nsrdb_data[selected_ids[0]]["timestamps"]
        n_total = len(self._nsrdb_timestamps)

        sl = _chronological_split(n_total, self.split)
        self._nsrdb_sl = sl
        split_len = len(range(*sl.indices(n_total)))
        self.num_samples = max(0, split_len - self.window_size + 1)

        # Adjacency matrix from graph.json
        self.adj_matrix = self._load_adj_matrix(graph_p, selected_ids)

        # Frame index (satellite NPZ files, shared across entities)
        fi_ts, fi_rp, fi_mask = _load_frame_index(fi_path)
        sort_order = sorted(range(len(fi_ts)), key=lambda i: fi_ts[i])
        self._nsrdb_fi_ts   = [fi_ts[i]  for i in sort_order]
        self._nsrdb_fi_rp   = [fi_rp[i]  for i in sort_order]
        self._nsrdb_fi_mask = [fi_mask[i] for i in sort_order]
        self._nsrdb_frames_dir = base / "frames"

    def _getitem_goes16_nsrdb(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self._nsrdb_sl.start + idx
        N      = self.num_entities
        ts_win = self._nsrdb_timestamps[offset: offset + self.window_size]

        Y           = torch.zeros(N, self.T, 1)
        Y_future    = torch.zeros(N, self.H, 1)
        X_cov       = torch.zeros(N, self.window_size, self.C_cov)
        mask_target = torch.zeros(N, self.T, 1)
        for ni, eid in enumerate(self._nsrdb_entity_ids):
            d = self._nsrdb_data[eid]
            for t in range(self.T):
                v = d["target"][offset + t]
                Y[ni, t, 0] = float(v) if v is not None else 0.0
                mask_target[ni, t, 0] = d["mask_target"][offset + t]
            for h in range(self.H):
                v = d["target"][offset + self.T + h]
                Y_future[ni, h, 0] = float(v) if v is not None else 0.0
            for ci, col in enumerate(d["covariates"]):
                X_cov[ni, :, ci] = torch.tensor(
                    [float(v) if v is not None else 0.0 for v in col[offset:offset + self.window_size]],
                    dtype=torch.float32,
                )

        # C4 fix: visual anchors at decoupled (potentially denser) cadence
        vis_ts = self._visual_anchor_timestamps(ts_win)
        V            = torch.zeros(N, self.T_v, self.C_img, self.img_size, self.img_size)
        timestamps_v = torch.zeros(self.T_v, dtype=torch.long)
        mask_visual  = torch.zeros(N, self.T_v)

        for i, t in enumerate(vis_ts):
            fi_idx = _nearest_frame_idx(self._nsrdb_fi_ts, t)
            if fi_idx is None or not self._nsrdb_fi_mask[fi_idx]:
                continue
            rel = self._nsrdb_fi_rp[fi_idx]
            frame_path = self._nsrdb_frames_dir / Path(rel).name
            if not frame_path.exists():
                continue
            try:
                frame = _load_npz(frame_path, self.img_size, self.C_img)  # (C, H, W)
                V[:, i] = frame.unsqueeze(0).expand(N, -1, -1, -1)
                timestamps_v[i] = self._nsrdb_fi_ts[fi_idx]
                mask_visual[:, i] = 1.0
            except Exception:
                pass

        vis_available = float(mask_visual.any())
        return {
            "Y":                     Y,
            "Y_future":              Y_future,
            "X_cov":                 X_cov,
            "V":                     V,
            "timestamps":            torch.tensor(ts_win, dtype=torch.long),
            "entity_ids":            torch.tensor(self._nsrdb_entity_ids, dtype=torch.long),
            "timestamps_v":          timestamps_v,
            "mask_target":           mask_target,
            "mask_future":           torch.ones(N, self.H, 1),
            "mask_visual":           mask_visual,
            "mask_modality_dropout": torch.tensor([[1.0, vis_available]] * N),
            "adj_matrix":            self.adj_matrix,
        }

    # ------------------------------------------------------------------
    # EarthNet2021  (per-entity frames, entity-based split)
    # ------------------------------------------------------------------

    def _init_earthnet2021(self) -> None:
        self.num_entities = 1
        self.adj_matrix   = torch.eye(1)

        base    = self.data_dir / "meteorology" / "earthnet2021"
        ts_path = base / "timeseries.parquet"
        fi_path = base / "frame_index.parquet"

        if not ts_path.exists():
            raise FileNotFoundError(
                f"EarthNet2021 refactored data not found: {ts_path}\n"
                "Run: python scripts/meteorology/build_earthnet2021.py"
            )

        tbl    = pq.read_table(ts_path)
        fi_tbl = pq.read_table(fi_path)

        all_eids = sorted(set(tbl.column("entity_id").to_pylist()))
        n_all    = len(all_eids)
        n_train  = int(n_all * 0.70)
        n_val    = int(n_all * 0.80)
        if self.split == "train":
            split_ids = all_eids[:n_train]
        elif self.split in ("val", "validation"):
            split_ids = all_eids[n_train:n_val]
        else:
            split_ids = all_eids[n_val:]
        split_id_set = set(split_ids)

        # Identify target and covariate columns
        _skip        = {"entity_id", "timestamp_unix", "mask_target", "mask_cov"}
        _tgt_priority = ("ndvi_norm", "ndvi")
        ts_names     = tbl.schema.names
        target_col   = next((c for c in _tgt_priority if c in ts_names), None)
        if target_col is None:
            raise ValueError("No target column (ndvi_norm/ndvi) in EarthNet2021 timeseries.parquet")
        tgt_stem = target_col.replace("_norm", "")
        cov_cands = [c for c in ts_names
                     if c not in _skip and c.endswith("_norm")
                     and c != f"{tgt_stem}_norm" and c != tgt_stem]

        # Pull all columns to Python lists once (717 K rows — ~0.1 s)
        id_raw  = tbl.column("entity_id").to_pylist()
        ts_raw  = tbl.column("timestamp_unix").to_pylist()
        tgt_raw = [float(v) if v is not None else 0.0
                   for v in tbl.column(target_col).to_pylist()]
        has_mt  = "mask_target" in ts_names
        mt_raw  = [float(v) if v is not None else 1.0
                   for v in tbl.column("mask_target").to_pylist()] if has_mt else None
        cov_raw: list[list[float]] = []
        for col in cov_cands[: self.C_cov]:
            cov_raw.append([float(v) if v is not None else 0.0
                            for v in tbl.column(col).to_pylist()])
        while len(cov_raw) < self.C_cov:
            cov_raw.append([0.0] * len(id_raw))

        # Group by entity_id
        en21: dict = {}
        for i, eid in enumerate(id_raw):
            if eid not in split_id_set:
                continue
            if eid not in en21:
                en21[eid] = {"ts": [], "target": [], "mt": [],
                             "cov": [[] for _ in range(self.C_cov)]}
            en21[eid]["ts"].append(ts_raw[i])
            en21[eid]["target"].append(tgt_raw[i])
            en21[eid]["mt"].append(mt_raw[i] if mt_raw else 1.0)
            for ci in range(self.C_cov):
                en21[eid]["cov"][ci].append(cov_raw[ci][i])

        # Sort by timestamp within each entity
        for d in en21.values():
            order = sorted(range(len(d["ts"])), key=lambda x: d["ts"][x])
            d["ts"]    = [d["ts"][j]     for j in order]
            d["target"] = [d["target"][j] for j in order]
            d["mt"]    = [d["mt"][j]     for j in order]
            for ci in range(self.C_cov):
                d["cov"][ci] = [d["cov"][ci][j] for j in order]

        # Group frame_index by entity_id
        fi_eid_raw = fi_tbl.column("entity_id").to_pylist()
        fi_ts_raw  = fi_tbl.column("timestamp_unix").to_pylist()
        fi_rp_raw  = fi_tbl.column("rel_path").to_pylist()
        fi_mv_raw  = (fi_tbl.column("mask_visual").to_pylist()
                      if "mask_visual" in fi_tbl.schema.names
                      else [1] * len(fi_ts_raw))
        fi_by_eid: dict = {}
        for i, eid in enumerate(fi_eid_raw):
            if eid not in split_id_set:
                continue
            if eid not in fi_by_eid:
                fi_by_eid[eid] = {"ts": [], "rp": [], "mv": []}
            fi_by_eid[eid]["ts"].append(fi_ts_raw[i])
            fi_by_eid[eid]["rp"].append(fi_rp_raw[i])
            fi_by_eid[eid]["mv"].append(fi_mv_raw[i])
        for fd in fi_by_eid.values():
            order = sorted(range(len(fd["ts"])), key=lambda x: fd["ts"][x])
            fd["ts"] = [fd["ts"][j] for j in order]
            fd["rp"] = [fd["rp"][j] for j in order]
            fd["mv"] = [fd["mv"][j] for j in order]

        self._en21_entities:   dict = en21
        self._en21_fi:         dict = fi_by_eid
        self._en21_frames_dir: Path = base / "frames"

        # Flat sample index: (entity_id, window_offset)
        samples: list[tuple[int, int]] = []
        for eid in split_ids:
            if eid not in en21:
                continue
            n_steps = len(en21[eid]["ts"])
            for off in range(max(0, n_steps - self.window_size + 1)):
                samples.append((eid, off))
        self._en21_samples = samples
        self.num_samples   = len(samples)

    def _getitem_earthnet2021(self, idx: int) -> Dict[str, torch.Tensor]:
        eid, offset = self._en21_samples[idx]
        d    = self._en21_entities[eid]
        fd   = self._en21_fi.get(eid, {"ts": [], "rp": [], "mv": []})

        ts_win     = d["ts"][offset: offset + self.window_size]
        target_win = d["target"][offset: offset + self.T]
        future_win = d["target"][offset + self.T: offset + self.window_size]
        mt_win     = d["mt"][offset: offset + self.T]

        Y        = torch.tensor(target_win, dtype=torch.float32).view(1, self.T, 1)
        Y_future = torch.tensor(future_win, dtype=torch.float32).view(1, self.H, 1)
        X_cov    = torch.zeros(1, self.window_size, self.C_cov)
        for ci in range(self.C_cov):
            col = d["cov"][ci]
            X_cov[0, :, ci] = torch.tensor(col[offset:offset + self.window_size], dtype=torch.float32)

        mt_tensor = torch.tensor(mt_win, dtype=torch.float32).view(1, self.T, 1)

        # C4 fix: visual anchors at decoupled (potentially denser) cadence
        vis_ts = self._visual_anchor_timestamps(ts_win)
        V            = torch.zeros(1, self.T_v, self.C_img, self.img_size, self.img_size)
        timestamps_v = torch.zeros(self.T_v, dtype=torch.long)
        mask_visual  = torch.zeros(1, self.T_v)

        for i, t in enumerate(vis_ts):
            fi_idx = _nearest_frame_idx(fd["ts"], t)
            if fi_idx is None or not fd["mv"][fi_idx]:
                continue
            frame_path = self._en21_frames_dir / Path(fd["rp"][fi_idx])
            if not frame_path.exists():
                continue
            try:
                frame = _load_npz(frame_path, self.img_size, self.C_img, npz_key="frame")
                if self.imagenet_norm:
                    frame = (frame - _IMAGENET_MEAN) / _IMAGENET_STD
                V[0, i]          = frame
                timestamps_v[i]  = fd["ts"][fi_idx]
                mask_visual[0, i] = 1.0
            except Exception:
                pass

        return {
            "Y":                     Y,
            "Y_future":              Y_future,
            "X_cov":                 X_cov,
            "V":                     V,
            "timestamps":            torch.tensor(ts_win, dtype=torch.long),
            "entity_ids":            torch.tensor([eid], dtype=torch.long),
            "timestamps_v":          timestamps_v,
            "mask_target":           mt_tensor,
            "mask_future":           torch.ones(1, self.H, 1),
            "mask_visual":           mask_visual,
            "mask_modality_dropout": torch.tensor([[1.0, float(mask_visual.any())]]),
            "adj_matrix":            self.adj_matrix,
        }

    # ------------------------------------------------------------------
    # ERA5-EU  (shared spatial frames, multi-entity grid points)
    # ------------------------------------------------------------------

    def _init_era5_eu(self) -> None:
        base    = self.data_dir / "meteorology" / "era5_eu"
        ts_path = base / "timeseries.parquet"
        fi_path = base / "frame_index.parquet"

        if not ts_path.exists():
            raise FileNotFoundError(
                f"ERA5-EU refactored data not found: {ts_path}\n"
                "Run: python scripts/meteorology/build_era5_eu.py"
            )

        # Load only selected entities via predicate pushdown
        tbl_meta = pq.read_schema(ts_path)
        all_eids = sorted(set(
            pq.read_table(ts_path, columns=["entity_id"])
            .column("entity_id").to_pylist()
        ))
        selected_ids      = all_eids[: self.num_entities]
        self.num_entities = len(selected_ids)
        self._era5_entity_ids = selected_ids

        tbl = pq.read_table(ts_path, filters=[("entity_id", "in", selected_ids)])

        _skip        = {"entity_id", "timestamp_unix", "mask_target", "mask_cov"}
        _tgt_priority = ("t2m_norm", "ssrd_norm")
        ts_names     = tbl.schema.names
        target_col   = next((c for c in _tgt_priority if c in ts_names), None)
        if target_col is None:
            target_col = next((c for c in ts_names if c.endswith("_norm") and c not in _skip), None)
        tgt_stem = target_col.replace("_norm", "") if target_col else ""

        self._era5_data: Dict[int, Dict] = {}
        for eid in selected_ids:
            mask_col = pc.equal(tbl.column("entity_id"), pa.scalar(eid))
            etbl     = tbl.filter(mask_col)
            sort_idx = pc.sort_indices(etbl, sort_keys=[("timestamp_unix", "ascending")])
            etbl     = etbl.take(sort_idx)

            ts_list  = etbl.column("timestamp_unix").to_pylist()
            tgt_list = ([float(v) if v is not None else 0.0
                         for v in etbl.column(target_col).to_pylist()]
                        if target_col else [0.0] * len(ts_list))

            cov_cands = [c for c in ts_names
                         if c not in _skip and c.endswith("_norm")
                         and c != f"{tgt_stem}_norm"]
            cov_data: list[list[float]] = []
            for col in cov_cands[: self.C_cov]:
                cov_data.append([float(v) if v is not None else 0.0
                                  for v in etbl.column(col).to_pylist()])
            while len(cov_data) < self.C_cov:
                cov_data.append([0.0] * len(ts_list))

            mt_list = ([float(v) if v is not None else 1.0
                        for v in etbl.column("mask_target").to_pylist()]
                       if "mask_target" in ts_names else [1.0] * len(ts_list))

            self._era5_data[eid] = {
                "timestamps":  ts_list,
                "target":      tgt_list,
                "covariates":  cov_data,
                "mask_target": mt_list,
            }

        self._era5_timestamps = self._era5_data[selected_ids[0]]["timestamps"]
        n_total = len(self._era5_timestamps)
        sl      = _chronological_split(n_total, self.split)
        self._era5_sl     = sl
        self.num_samples  = max(0, len(range(*sl.indices(n_total))) - self.window_size + 1)
        self.adj_matrix   = self._load_adj_matrix(base / "graph.json", selected_ids)

        fi_ts, fi_rp, fi_mask = _load_frame_index(fi_path)
        order = sorted(range(len(fi_ts)), key=lambda i: fi_ts[i])
        self._era5_fi_ts   = [fi_ts[i]   for i in order]
        self._era5_fi_rp   = [fi_rp[i]   for i in order]
        self._era5_fi_mask = [fi_mask[i]  for i in order]
        self._era5_frames_dir = base / "frames"

    def _getitem_era5_eu(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self._era5_sl.start + idx
        N      = self.num_entities
        ts_win = self._era5_timestamps[offset: offset + self.window_size]

        Y           = torch.zeros(N, self.T, 1)
        Y_future    = torch.zeros(N, self.H, 1)
        X_cov       = torch.zeros(N, self.window_size, self.C_cov)
        mask_target = torch.zeros(N, self.T, 1)
        for ni, eid in enumerate(self._era5_entity_ids):
            d = self._era5_data[eid]
            for t in range(self.T):
                Y[ni, t, 0]           = d["target"][offset + t]
                mask_target[ni, t, 0] = d["mask_target"][offset + t]
            for h in range(self.H):
                Y_future[ni, h, 0] = d["target"][offset + self.T + h]
            for ci, col in enumerate(d["covariates"]):
                X_cov[ni, :, ci] = torch.tensor(col[offset:offset + self.window_size], dtype=torch.float32)

        # C4 fix: visual anchors at decoupled (potentially denser) cadence
        vis_ts = self._visual_anchor_timestamps(ts_win)
        V            = torch.zeros(N, self.T_v, self.C_img, self.img_size, self.img_size)
        timestamps_v = torch.zeros(self.T_v, dtype=torch.long)
        mask_visual  = torch.zeros(N, self.T_v)

        for i, t in enumerate(vis_ts):
            fi_idx = _nearest_frame_idx(self._era5_fi_ts, t)
            if fi_idx is None or not self._era5_fi_mask[fi_idx]:
                continue
            rel        = self._era5_fi_rp[fi_idx]
            frame_path = self._era5_frames_dir / Path(rel).name
            if not frame_path.exists():
                continue
            try:
                frame = _load_npz(frame_path, self.img_size, self.C_img, npz_key="frame")
                V[:, i]        = frame.unsqueeze(0).expand(N, -1, -1, -1)
                timestamps_v[i] = self._era5_fi_ts[fi_idx]
                mask_visual[:, i] = 1.0
            except Exception:
                pass

        vis_available = float(mask_visual.any())
        return {
            "Y":                     Y,
            "Y_future":              Y_future,
            "X_cov":                 X_cov,
            "V":                     V,
            "timestamps":            torch.tensor(ts_win, dtype=torch.long),
            "entity_ids":            torch.tensor(self._era5_entity_ids, dtype=torch.long),
            "timestamps_v":          timestamps_v,
            "mask_target":           mask_target,
            "mask_future":           torch.ones(N, self.H, 1),
            "mask_visual":           mask_visual,
            "mask_modality_dropout": torch.tensor([[1.0, vis_available]] * N),
            "adj_matrix":            self.adj_matrix,
        }

    # ------------------------------------------------------------------
    # MeteoNet  (shared spatial frames, NPZ keys rainfall+reflectivity)
    # ------------------------------------------------------------------

    def _init_meteonet(self) -> None:
        base    = self.data_dir / "meteorology" / "meteonet"
        ts_path = base / "timeseries.parquet"
        fi_path = base / "frame_index.parquet"

        if not ts_path.exists():
            raise FileNotFoundError(
                f"MeteoNet refactored data not found: {ts_path}\n"
                "Run: python scripts/meteorology/build_meteonet.py"
            )

        # Read only entity_id column first to select subset
        all_eids = sorted(set(
            pq.read_table(ts_path, columns=["entity_id"])
            .column("entity_id").to_pylist()
        ))
        selected_ids      = all_eids[: self.num_entities]
        self.num_entities = len(selected_ids)
        self._meteo_entity_ids = selected_ids

        tbl = pq.read_table(ts_path, filters=[("entity_id", "in", selected_ids)])

        _skip         = {"entity_id", "timestamp_unix", "mask_target", "mask_cov"}
        _tgt_priority = ("t_norm", "precip_norm")
        ts_names      = tbl.schema.names
        target_col    = next((c for c in _tgt_priority if c in ts_names), None)
        if target_col is None:
            target_col = next((c for c in ts_names if c.endswith("_norm") and c not in _skip), None)
        tgt_stem = target_col.replace("_norm", "") if target_col else ""

        self._meteo_data: Dict[int, Dict] = {}
        for eid in selected_ids:
            mask_col = pc.equal(tbl.column("entity_id"), pa.scalar(eid))
            etbl     = tbl.filter(mask_col)
            sort_idx = pc.sort_indices(etbl, sort_keys=[("timestamp_unix", "ascending")])
            etbl     = etbl.take(sort_idx)

            ts_list  = etbl.column("timestamp_unix").to_pylist()
            tgt_list = ([float(v) if v is not None else 0.0
                         for v in etbl.column(target_col).to_pylist()]
                        if target_col else [0.0] * len(ts_list))

            cov_cands = [c for c in ts_names
                         if c not in _skip and c.endswith("_norm")
                         and c != f"{tgt_stem}_norm"]
            cov_data: list[list[float]] = []
            for col in cov_cands[: self.C_cov]:
                cov_data.append([float(v) if v is not None else 0.0
                                  for v in etbl.column(col).to_pylist()])
            while len(cov_data) < self.C_cov:
                cov_data.append([0.0] * len(ts_list))

            mt_list = ([float(v) if v is not None else 1.0
                        for v in etbl.column("mask_target").to_pylist()]
                       if "mask_target" in ts_names else [1.0] * len(ts_list))

            self._meteo_data[eid] = {
                "timestamps":  ts_list,
                "target":      tgt_list,
                "covariates":  cov_data,
                "mask_target": mt_list,
            }

        self._meteo_timestamps = self._meteo_data[selected_ids[0]]["timestamps"]
        n_total = len(self._meteo_timestamps)
        sl      = _chronological_split(n_total, self.split)
        self._meteo_sl    = sl
        self.num_samples  = max(0, len(range(*sl.indices(n_total))) - self.window_size + 1)
        self.adj_matrix   = self._load_adj_matrix(base / "graph.json", selected_ids)

        fi_ts, fi_rp, fi_mask = _load_frame_index(fi_path)
        order = sorted(range(len(fi_ts)), key=lambda i: fi_ts[i])
        self._meteo_fi_ts   = [fi_ts[i]   for i in order]
        self._meteo_fi_rp   = [fi_rp[i]   for i in order]
        self._meteo_fi_mask = [fi_mask[i]  for i in order]
        self._meteo_frames_dir = base / "frames"

    def _getitem_meteonet(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self._meteo_sl.start + idx
        N      = self.num_entities
        ts_win = self._meteo_timestamps[offset: offset + self.window_size]

        Y           = torch.zeros(N, self.T, 1)
        Y_future    = torch.zeros(N, self.H, 1)
        X_cov       = torch.zeros(N, self.window_size, self.C_cov)
        mask_target = torch.zeros(N, self.T, 1)
        for ni, eid in enumerate(self._meteo_entity_ids):
            d = self._meteo_data[eid]
            for t in range(self.T):
                Y[ni, t, 0]           = d["target"][offset + t]
                mask_target[ni, t, 0] = d["mask_target"][offset + t]
            for h in range(self.H):
                Y_future[ni, h, 0] = d["target"][offset + self.T + h]
            for ci, col in enumerate(d["covariates"]):
                X_cov[ni, :, ci] = torch.tensor(col[offset:offset + self.window_size], dtype=torch.float32)

        # C4 fix: visual anchors at decoupled (potentially denser) cadence
        vis_ts = self._visual_anchor_timestamps(ts_win)
        V            = torch.zeros(N, self.T_v, self.C_img, self.img_size, self.img_size)
        timestamps_v = torch.zeros(self.T_v, dtype=torch.long)
        mask_visual  = torch.zeros(N, self.T_v)

        for i, t in enumerate(vis_ts):
            fi_idx = _nearest_frame_idx(self._meteo_fi_ts, t)
            if fi_idx is None or not self._meteo_fi_mask[fi_idx]:
                continue
            rel        = self._meteo_fi_rp[fi_idx]
            frame_path = self._meteo_frames_dir / Path(rel).name
            if not frame_path.exists():
                continue
            try:
                # MeteoNet stores rainfall + reflectivity as separate 2-D arrays
                frame = _load_npz(frame_path, self.img_size, self.C_img, npz_key=None)
                V[:, i]           = frame.unsqueeze(0).expand(N, -1, -1, -1)
                timestamps_v[i]   = self._meteo_fi_ts[fi_idx]
                mask_visual[:, i] = 1.0
            except Exception:
                pass

        vis_available = float(mask_visual.any())
        return {
            "Y":                     Y,
            "Y_future":              Y_future,
            "X_cov":                 X_cov,
            "V":                     V,
            "timestamps":            torch.tensor(ts_win, dtype=torch.long),
            "entity_ids":            torch.tensor(self._meteo_entity_ids, dtype=torch.long),
            "timestamps_v":          timestamps_v,
            "mask_target":           mask_target,
            "mask_future":           torch.ones(N, self.H, 1),
            "mask_visual":           mask_visual,
            "mask_modality_dropout": torch.tensor([[1.0, vis_available]] * N),
            "adj_matrix":            self.adj_matrix,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _load_adj_matrix(graph_path: Path, entity_ids: List[int]) -> torch.Tensor:
        """Build a dense float32 adjacency matrix from graph.json for the given entity subset."""
        n   = len(entity_ids)
        adj = torch.eye(n)
        if not graph_path.exists():
            return adj
        graph     = json.loads(graph_path.read_text())
        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
        for edge in graph.get("edges", []):
            s, d = edge.get("src"), edge.get("dst")
            if s in id_to_idx and d in id_to_idx:
                adj[id_to_idx[s], id_to_idx[d]] = 1.0
        return adj

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def _compute_cache_key(self, idx: int) -> str:
        """Stable, content-based key for the pre-computed VidTok latent cache."""
        if self.dataset_name == "skippd":
            return str(self._skippd_ts[self._skippd_sl.start + idx])
        if self.dataset_name == "solarnet":
            return str(self._solarnet_ts[self._solarnet_sl.start + idx])
        if self.dataset_name == "goes16_nsrdb":
            return str(self._nsrdb_timestamps[self._nsrdb_sl.start + idx])
        if self.dataset_name == "earthnet2021":
            eid, off = self._en21_samples[idx]
            return f"{eid}_{off:05d}"
        if self.dataset_name == "era5_eu":
            return str(self._era5_timestamps[self._era5_sl.start + idx])
        if self.dataset_name == "meteonet":
            return str(self._meteo_timestamps[self._meteo_sl.start + idx])
        return str(idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.dataset_name == "skippd":
            item = self._getitem_skippd(idx)
        elif self.dataset_name == "solarnet":
            item = self._getitem_solarnet(idx)
        elif self.dataset_name == "goes16_nsrdb":
            item = self._getitem_goes16_nsrdb(idx)
        elif self.dataset_name == "earthnet2021":
            item = self._getitem_earthnet2021(idx)
        elif self.dataset_name == "era5_eu":
            item = self._getitem_era5_eu(idx)
        elif self.dataset_name == "meteonet":
            item = self._getitem_meteonet(idx)
        else:
            item = self._getitem_synthetic(idx)

        # Attach pre-computed VidTok latents when cache is available.
        # Z shape: [N, T_lat, P, D_v]  (N=1 for single-entity datasets)
        if self._cache_dir is not None:
            key        = self._compute_cache_key(idx)
            cache_file = self._cache_dir / f"{key}.pt"
            if cache_file.exists():
                item["Z"] = torch.load(cache_file, map_location="cpu", weights_only=True)

        return item

    # ------------------------------------------------------------------
    # Synthetic
    # ------------------------------------------------------------------

    def _getitem_synthetic(self, idx: int) -> Dict[str, torch.Tensor]:
        base_time    = idx * self.window_size
        timestamps   = torch.arange(base_time, base_time + self.window_size, dtype=torch.long)
        visual_start = max(base_time, base_time + self.T - self.T_v)
        timestamps_v = torch.linspace(visual_start, base_time + self.T - 1,
                                      self.T_v).long()
        return {
            "Y":                     torch.randn(self.num_entities, self.T, self.C_target),
            "Y_future":              torch.randn(self.num_entities, self.H, self.C_target),
            "X_cov":                 torch.randn(self.num_entities, self.window_size, self.C_cov),
            "V":                     torch.rand(self.num_entities, self.T_v, self.C_img,
                                                self.img_size, self.img_size),
            "timestamps":            timestamps,
            "entity_ids":            torch.arange(self.num_entities, dtype=torch.long),
            "timestamps_v":          timestamps_v,
            "mask_target":           torch.ones(self.num_entities, self.T, self.C_target),
            "mask_future":           torch.ones(self.num_entities, self.H, self.C_target),
            "mask_visual":           torch.ones(self.num_entities, self.T_v),
            "mask_modality_dropout": torch.ones(self.num_entities, 2),
            "adj_matrix":            self.adj_matrix,
        }
