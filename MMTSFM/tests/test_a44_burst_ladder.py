"""A44 / ticket 45 — the data side of the daily-anchor ladder.

``PVRecordDataset._load_vision`` draws frames as ``n_anchors`` dense bursts and
must place them SLOT-STABLY: slot ``j`` belongs to anchor ``j // f`` whether or
not a frame was found for it. The historical uniform path left-packs instead
(survivors slide to the end), which is correct there and fatal here — a slid
frame lands in a different anchor's latent group and gets the wrong anchor's
position id, i.e. a mislabelled sky age, which is A10b (-0.135 SS).

The masked slots matter as much as the filled ones: they keep their REQUESTED
offset in ``video_delta_t``, so the coverage guard in ``vision_chronos2.forward``
measures the ladder's SPAN and not which frames happened to survive. uk_pv has
a ~10 h nightly hole in the frame archive (frames exist 02:00-16:00 UTC only),
so partially-populated bursts are the normal case, not an edge case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from test_pv_record import SPLITS, _make_parquet  # noqa: E402

ANCHOR_H, F, TV = 24.0, 4, 20
SPACING_MIN = 45.0


def _dataset(tmp_path: Path, drop: slice | None = None, **kw):
    """Burst-ladder dataset over one plant, optionally with a hole in the grid.

    ``drop`` blanks a contiguous run of frame pointers, standing in for the
    archive's nightly gap.
    """
    h5py = pytest.importorskip("h5py")
    site = SPLITS["uk_pv"]["train"][0]
    parquet = tmp_path / "dataset_all.parquet"
    h5 = tmp_path / "images_all.h5"
    # 30-min cadence x 800 steps = 400 h, enough for a 98.25 h ladder behind a
    # 672-step (14-day) history.
    _make_parquet(parquet, [site], n_steps=800, with_frames=True)
    if drop is not None:
        import pandas as pd

        df = pd.read_parquet(parquet)
        col = df.columns[df.columns.str.contains("h5_index")][0]
        df.loc[drop, col] = -1
        df.to_parquet(parquet)
    with h5py.File(h5, "w") as f:
        g = f.create_group(f"uk_pv_{site}")
        g.create_dataset("images", data=np.full((800, 128, 128), 200, np.uint8))

    from mmtsfm.data.pv_record import PVRecordDataset

    return PVRecordDataset(
        split="train",
        dataset_name="uk_pv",
        data_path=str(parquet),
        h5_path=str(h5),
        img_size=32,
        img_channels=3,
        video_frames=TV,
        visual_window_hours=100.0,
        visual_frame_spacing_min=SPACING_MIN,
        visual_anchor_stride_hours=ANCHOR_H,
        visual_frames_per_anchor=F,
        **kw,
    )


def _requested(j: int) -> float:
    """Seconds before the origin that slot ``j`` asked for (oldest slot first)."""
    k = TV - 1 - j
    return (k // F) * ANCHOR_H * 3600.0 + (k % F) * SPACING_MIN * 60.0


class TestBurstLadder:
    def test_every_slot_carries_its_requested_age(self, tmp_path):
        """Filled or masked, Δt is the LAYOUT's age, not the survivor's."""
        item = _dataset(tmp_path)[0]
        vdt = item["video_delta_t"][0]
        assert vdt.shape == (TV,)
        for j in range(TV):
            want = _requested(j)
            # a filled slot snaps to the nearest frame, within half a step
            assert abs(float(vdt[j]) - want) <= SPACING_MIN * 60.0 / 2.0

    def test_delta_t_runs_oldest_to_newest(self, tmp_path):
        """Slot 0 is the oldest; anchor i == slots [i*F, (i+1)*F)."""
        item = _dataset(tmp_path)[0]
        vdt = item["video_delta_t"][0]
        assert float(vdt[0]) > float(vdt[-1])
        assert float(vdt[-1]) == 0.0  # newest anchor co-temporal with the origin
        assert np.all(np.diff(vdt.numpy()) <= 0)
        # the ladder spans (n_anchors - 1) * 24 h plus the burst
        span_h = float(vdt[0]) / 3600.0
        assert span_h == pytest.approx(
            (TV // F - 1) * ANCHOR_H + (F - 1) * SPACING_MIN / 60.0, abs=0.5
        )

    def test_a_hole_masks_in_place_instead_of_sliding(self, tmp_path):
        """The uk_pv night. Survivors must NOT left-pack onto the recent slots."""
        # blank ~1.5 days of pointers well inside the ladder's reach
        item = _dataset(tmp_path, drop=slice(600, 672))[0]
        mask = item["mask_visual"][0]
        vdt = item["video_delta_t"][0]

        n_missing = int((mask == 0).sum())
        assert 0 < n_missing < TV, "fixture must lose SOME frames, not all"
        # every slot still reports its own requested age, masked ones included
        for j in range(TV):
            assert abs(float(vdt[j]) - _requested(j)) <= SPACING_MIN * 60.0 / 2.0
        # and the gap is a contiguous run in the middle, not a left-pad prefix
        missing = (mask == 0).nonzero().flatten().tolist()
        assert missing != list(range(n_missing)), "frames slid to the tail"

    def test_uniform_path_still_left_packs(self, tmp_path):
        """The regression fence: no burst args -> historical behaviour, intact."""
        h5py = pytest.importorskip("h5py")
        assert h5py
        site = SPLITS["uk_pv"]["train"][0]
        parquet = tmp_path / "dataset_all.parquet"
        h5 = tmp_path / "images_all.h5"
        _make_parquet(parquet, [site], n_steps=800, with_frames=True)
        import pandas as pd

        df = pd.read_parquet(parquet)
        col = df.columns[df.columns.str.contains("h5_index")][0]
        df.loc[660:672, col] = -1  # blank the most recent frames
        df.to_parquet(parquet)
        with h5py.File(h5, "w") as f:
            g = f.create_group(f"uk_pv_{site}")
            g.create_dataset("images", data=np.full((800, 128, 128), 200, np.uint8))

        from mmtsfm.data.pv_record import PVRecordDataset

        ds = PVRecordDataset(
            split="train",
            dataset_name="uk_pv",
            data_path=str(parquet),
            h5_path=str(h5),
            img_size=32,
            img_channels=3,
            video_frames=8,
            visual_window_hours=6.0,
        )
        mask = ds[0]["mask_visual"][0]
        n = int(mask.sum())
        # left-packed: whatever survives occupies the TRAILING slots
        assert float(mask[-n:].sum()) == n
