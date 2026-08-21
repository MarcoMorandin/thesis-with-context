"""Dataset-of-record backend (``uk_pv`` / ``goes_pvdaq``) aligned to the baseline
protocol (knowledge/protocol.md).

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


def _frame_maps_from_h5(h5_path: str) -> tuple[bool, dict[tuple[str, str], dict]]:
    """(is_png_encoded, {(dataset, site): {unix_ts -> frame_idx}}) read from the H5.

    The H5's `timestamps` is the authoritative record of which frames exist;
    the parquet only knows about rows where power was observed. Returns an empty
    map when the file is absent so callers can fall back.
    """
    import os

    if not h5_path or not os.path.exists(h5_path):
        return False, {}
    import h5py

    maps: dict[tuple[str, str], dict] = {}
    with h5py.File(h5_path, "r") as f:
        png = f.attrs.get("format") == "png"
        for name in f:
            if "timestamps" not in f[name]:
                continue
            raw = f[name]["timestamps"][:]
            ts = pd.to_datetime(
                [t.decode() if isinstance(t, bytes) else str(t) for t in raw],
                utc=True,
                format="mixed",
            )
            unix = ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")).to_numpy()
            ds_name, site = name.rsplit("_", 1)
            maps[(ds_name, site)] = dict(
                zip(unix.astype(np.int64).tolist(), range(len(unix)))
            )
    return png, maps


def _decode_frame(raw: np.ndarray, png: bool) -> np.ndarray:
    """One stored frame -> H,W[,C] uint8. v2 keeps PNG bytes, v1 kept arrays."""
    if not png:
        return np.asarray(raw)
    import io

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(np.asarray(raw, dtype=np.uint8).tobytes())))


def _prep_frame(
    arr: np.ndarray, side: int, c_img: int, imagenet_norm: bool
) -> torch.Tensor:
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
        visual_window_hours: float = 6.0,
        visual_frame_spacing_min: float | None = None,
        num_entities: int = 1,
        vjepa_cache_dir: str | None = None,
        emit_vision: bool = True,
        **_ignored,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        # False → vision-free runs (model.vision_cfg.skip_vision_stack=true):
        # emit a 1×1-pixel placeholder V, no frame decode, no latent load.
        self.emit_vision = bool(emit_vision)
        # Pre-extracted V-JEPA 2.1 latent cache (scripts/extract_video_embeddings.py).
        # When set, each window's [T_lat, P, D_v] latent is loaded and attached as
        # ``Z`` so training skips the encoder forward (see _attach_latents).
        self._cache_dir = Path(vjepa_cache_dir) if vjepa_cache_dir else None
        self.T_v = int(video_frames)
        self.img_size = int(img_size)
        self.C_img = int(img_channels)
        self.imagenet_norm = bool(imagenet_norm)
        self.visual_window_hours = float(visual_window_hours)
        self.visual_frame_spacing_min = (
            float(visual_frame_spacing_min)
            if visual_frame_spacing_min is not None
            else None
        )
        # W4: number of distinct plants assembled per group (cross-plant mixing).
        # >1 groups disjoint plants from THIS split that share a time window so
        # GroupSelfAttention fuses across entities. Disjointness vs other splits
        # is guaranteed because site_ids is already filtered to this split.
        self.num_entities = max(1, int(num_entities))

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

        cols = sorted(
            {
                config.DATASET_COL,
                config.SITE_COL,
                config.TIME_COL,
                config.TARGET_COL,
                config.CAPACITY_COL,
                config.CLEARSKY_COL,
                config.BAD_SITE_COL,
                FRAME_IDX_COL,
                *config.COV_COLS,
            }
        )
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
            df,
            site_ids,
            history=self.T,
            horizon=self.H,
            stride=stride,
            future_cov=future_cov,
        )

        # per-(dataset, site) {unix_ts -> frame_idx}. Preferred source is the H5's
        # own `timestamps`, because the parquet only has rows where POWER was
        # observed -- daylight -- while v2 also holds night frames that no parquet
        # row points at. Building this map from the parquet would leave every one
        # of those frames unreachable, which is precisely the half of the visual
        # channel the non-HRV re-extraction exists to recover: a 07:30 origin's
        # 6h-backward window is 01:30-07:30 and lands entirely in them.
        self.png_frames, self.frame_maps = _frame_maps_from_h5(self.h5_path)
        if not self.frame_maps:
            # No H5 (unit fixtures, latents-only runs): fall back to the parquet
            # pointer. image_h5_index < 0 marks a row whose frame does not exist
            # -- 500 EUMETSAT archive gaps on uk_pv -- and must not become a
            # lookup into frame 0.
            fdf = df[df[FRAME_IDX_COL].notna() & (df[FRAME_IDX_COL] >= 0)]
            for (ds, site), g in fdf.groupby([config.DATASET_COL, config.SITE_COL]):
                ts = pd.DatetimeIndex(g[config.TIME_COL])
                if ts.tz is None:
                    ts = ts.tz_localize("UTC")
                unix = (
                    (ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
                ).to_numpy()
                idx = g[FRAME_IDX_COL].to_numpy().astype(np.int64)
                self.frame_maps[(str(ds), str(site))] = dict(
                    zip(unix.astype(np.int64).tolist(), idx.tolist())
                )
        # Sorted (timestamps, indices) per group. The dict answers "is this
        # exact instant a frame?"; picking a frame at a REQUESTED spacing needs
        # nearest-neighbour lookup instead, which needs them ordered.
        self._frame_grid = {
            k: (
                np.array(sorted(v), dtype=np.int64),
                np.array([v[t] for t in sorted(v)], dtype=np.int64),
            )
            for k, v in self.frame_maps.items()
        }
        self._h5 = None  # opened lazily (h5py handles are not fork-safe)
        self._build_groups()

    def _build_groups(self) -> None:
        """Assemble groups of ``num_entities`` distinct plants per time window.

        N==1 → one window per group (legacy single-entity behaviour). N>1 →
        bucket windows by their absolute origin timestamp and emit full groups
        of N distinct plants sharing that window. Partial buckets (< N plants)
        are dropped so every group stacks to a fixed [N, ...] shape.
        """
        if self.num_entities == 1:
            self.groups: list[list[int]] = [[i] for i in range(len(self.win))]
            return

        from collections import defaultdict

        buckets: dict[int, list[int]] = defaultdict(list)
        seen: dict[int, set[str]] = defaultdict(set)
        for wi, (si, start) in enumerate(self.win._index):
            s = self.win.series[si]
            t0 = int(s.timestamps[start])
            if s.site_id in seen[t0]:
                continue  # one window per (site, origin)
            seen[t0].add(s.site_id)
            buckets[t0].append(wi)

        N = self.num_entities
        groups: list[list[int]] = []
        for t0 in sorted(buckets):
            members = buckets[t0]
            for k in range(0, len(members) - N + 1, N):
                groups.append(members[k : k + N])

        if not groups:
            # Not enough co-temporal plants to form a single full group of N
            # (e.g. a split with < N plants). Fall back to single-entity windows
            # so the run still proceeds rather than yielding an empty dataset.
            print(
                f"[pv_record] num_entities={N} but no time window has {N} distinct "
                f"plants in this split — falling back to single-entity windows.",
                flush=True,
            )
            self.num_entities = 1
            self.groups = [[i] for i in range(len(self.win))]
            return
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def _group(self, dataset: str, site: str):
        import h5py

        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5[f"{dataset}_{site}"]

    def _load_vision(
        self, item: dict, load_frames: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frame availability mask + Δt always; pixel decode only when needed.

        ``load_frames=False`` (pre-extracted V-JEPA latent hit) skips the H5
        read + PIL resize — the model consumes ``Z`` and ignores ``V`` — but
        keeps mask/Δt, which drive the summarizer's latent mask and W5 window.
        """
        T, Tv, S, C = self.T, self.T_v, self.img_size, self.C_img
        key = (str(item["dataset"]), str(item["site_id"]))
        fmap = self.frame_maps.get(key, {})
        hist_ts = item["timestamps"][:T]
        t_now = int(hist_ts[-1])

        # Frames are chosen at a REQUESTED SPACING off the frame grid, not by
        # intersecting with the power window's timestamps. Selecting from
        # `hist_ts` was correct while frames existed only where power was
        # observed, but v2 puts them on their own 15-minute grid: intersecting
        # would silently snap back to the 30-minute power cadence and ignore
        # every frame in between, making the cadence unusable as a parameter.
        # That is the dial the sub-hourly-imagery question turns on.
        #
        # Default spacing spreads Tv frames across visual_window_hours, which is
        # what "8 frames over a 6h window" plainly means; an explicit
        # visual_frame_spacing_min pins it instead and the span becomes
        # spacing * (Tv - 1).
        spacing_min = self.visual_frame_spacing_min or (
            self.visual_window_hours * 60.0 / Tv
        )
        spacing_sec = spacing_min * 60.0
        # Half a step: a slot takes the nearest frame only if it is closer than
        # the neighbouring slot's target, so an archive gap leaves that slot
        # MASKED rather than quietly duplicating a distant frame.
        tol_sec = spacing_sec / 2.0

        grid_ts, grid_idx = self._frame_grid.get(key, (None, None))
        sel: list[int] = []
        if grid_ts is not None and len(grid_ts):
            for k in range(Tv):
                want = t_now - int(round(k * spacing_sec))
                p = int(np.searchsorted(grid_ts, want))
                best, best_d = None, None
                for c in (p - 1, p):
                    if 0 <= c < len(grid_ts):
                        d = abs(int(grid_ts[c]) - want)
                        if best_d is None or d < best_d:
                            best, best_d = c, d
                if best is not None and best_d <= tol_sec:
                    t_sel = int(grid_ts[best])
                    fmap.setdefault(t_sel, int(grid_idx[best]))
                    sel.append(t_sel)
        # Ascending, newest last, matching the left-pad placement below.
        sel = sorted(set(sel))[-Tv:]

        V = torch.zeros(1, Tv, C, S, S)
        mask_v = torch.zeros(1, Tv)
        video_delta_t = torch.zeros(1, Tv)

        # Left-pad: place active frames at the end if we have fewer than Tv
        pad_len = Tv - len(sel)
        if len(sel) > 0:
            images = None
            if load_frames:
                g = self._group(*key)
                images = g["images"]
            for idx_sel, t in enumerate(sel):
                j = pad_len + idx_sel
                if images is not None:
                    raw = _decode_frame(images[fmap[t]], self.png_frames)
                    V[0, j] = _prep_frame(raw, S, C, self.imagenet_norm)
                mask_v[0, j] = 1.0
                video_delta_t[0, j] = float(t_now - t)

        return V, mask_v, video_delta_t

    def _build_entity(self, item: dict, load_frames: bool = True) -> dict:
        """Per-entity tensors (leading entity dim = 1) for one window."""
        T, H = self.T, self.H
        cov = np.asarray(item["cov"], dtype=np.float32)  # (T+H, C) future weather known
        hist_ts = item["timestamps"][:T]
        t_now = int(hist_ts[-1])
        if self.emit_vision:
            V, mask_visual, video_delta_t = self._load_vision(
                item, load_frames=load_frames
            )
        else:
            # Vision-free run: 1×1-pixel placeholder keeps the batch schema
            # (and collate shapes) without paying decode CPU or pinned memory.
            Tv = self.T_v
            V = torch.zeros(1, Tv, self.C_img, 1, 1)
            mask_visual = torch.zeros(1, Tv)
            video_delta_t = torch.zeros(1, Tv)
        hist_delta_t = torch.from_numpy(t_now - hist_ts.astype(np.float32)).view(1, T)

        return {
            "Y": torch.from_numpy(np.asarray(item["y_hist"], np.float32)).view(1, T, 1),
            "Y_future": torch.from_numpy(np.asarray(item["y_future"], np.float32)).view(
                1, H, 1
            ),
            # daylight horizon mask (clearsky_ghi > 0): protocol metrics are scored
            # only over daylight steps, matching the baselines (common.runner).
            "daylight_future": torch.from_numpy(
                np.asarray(item["daylight_future"], np.float32)
            ).view(1, H, 1),
            "X_cov": torch.from_numpy(cov).view(1, T + H, -1),
            "V": V,
            "mask_target": torch.from_numpy(
                np.asarray(item["mask_hist"], np.float32)
            ).view(1, T, 1),
            "mask_future": torch.from_numpy(
                np.asarray(item["mask_future"], np.float32)
            ).view(1, H, 1),
            "mask_visual": mask_visual,
            "video_delta_t": video_delta_t,
            "hist_delta_t": hist_delta_t,
            "mask_modality_dropout": torch.tensor([[1.0, float(mask_visual.any())]]),
            "site_id": str(item["site_id"]),
        }

    def __getitem__(self, idx: int) -> dict:
        win_indices = self.groups[idx]
        # Latent-cache hit for the whole group → skip the raw frame decode
        # (H5 read + PIL resize per frame): the model consumes Z, not V.
        latent_files = self._latent_files(win_indices) if self.emit_vision else None
        entities = [
            self._build_entity(self.win[w], load_frames=latent_files is None)
            for w in win_indices
        ]
        N = len(entities)

        # Stack per-entity tensors along the leading entity dim → [N, ...].
        stack_keys = (
            "Y",
            "Y_future",
            "daylight_future",
            "X_cov",
            "V",
            "mask_target",
            "mask_future",
            "mask_visual",
            "video_delta_t",
            "hist_delta_t",
            "mask_modality_dropout",
        )
        out = {k: torch.cat([e[k] for e in entities], dim=0) for k in stack_keys}

        # Timestamps are shared across the group (same window) → take entity 0.
        first = self.win[win_indices[0]]
        out["timestamps"] = torch.from_numpy(
            np.asarray(first["timestamps"], np.int64)
        ).long()
        out["entity_ids"] = torch.arange(N, dtype=torch.long)
        out["adj_matrix"] = torch.eye(N)
        # plant ids: keep a single str when N==1 (legacy collate + per-plant
        # protocol path unchanged); list[str] for N>1 cross-plant groups.
        site_ids = [e["site_id"] for e in entities]
        out["site_id"] = site_ids[0] if N == 1 else site_ids

        if latent_files is not None:
            out["Z"] = torch.stack(
                [
                    torch.load(f, map_location="cpu", weights_only=True)
                    for f in latent_files
                ],
                dim=0,
            )
        return out

    def _entity_cache_key(self, win_item: dict) -> str:
        """Stable per-(plant, window-origin) key for the V-JEPA latent cache.

        Shared with scripts/extract_video_embeddings.py so the producer and the
        training loader agree on file names. Origin = last history timestamp.
        """
        ts = np.asarray(win_item["timestamps"])
        origin = int(ts[self.T - 1])
        return f"{win_item['dataset']}_{win_item['site_id']}_{origin}"

    def _latent_files(self, win_indices: list[int]) -> list[Path] | None:
        """Cache files for the group, or None on any miss.

        Z is attached only when *every* entity in the group has a cached latent
        (a partial group would break the collate stack); a miss falls back to
        decoding + encoding raw frames V at train time.
        """
        if self._cache_dir is None:
            return None
        files = [
            self._cache_dir / f"{self._entity_cache_key(self.win[w])}.pt"
            for w in win_indices
        ]
        return files if all(f.exists() for f in files) else None
