"""Repair images_all_v2.h5 written before the timestamp convention was fixed.

The first pack wrote uk_pv timestamps WITHOUT the trailing Z that v1 uses and
that goes_pvdaq kept, so the file carried two conventions and reindex matched 0
of 104,792 goes rows. Only the 100 uk_pv `timestamps` datasets are wrong; all
105 GB of frame bytes are correct and are not touched.

Rewriting in place rather than repacking: the datasets are fixed-length strings
whose width changes (S19 -> S20), so each must be deleted and recreated. HDF5
does not reclaim the freed space, which costs a few MB of holes -- against 48
minutes to rebuild the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_images_v2 import reindex  # noqa: E402


def repair(h5_path: Path) -> dict[str, dict[str, int]]:
    index: dict[str, dict[str, int]] = {}
    fixed = 0
    with h5py.File(h5_path, "r+") as f:
        for name in sorted(f):
            ts = [
                t.decode() if isinstance(t, bytes) else str(t)
                for t in f[name]["timestamps"][:]
            ]
            n_img = f[name]["images"].shape[0]
            assert len(ts) == n_img, (name, len(ts), n_img)
            if not ts[0].endswith("Z"):
                new = [t + "Z" for t in ts]
                del f[name]["timestamps"]
                f[name].create_dataset("timestamps", data=np.array(new, dtype="S20"))
                ts = new
                fixed += 1
            index[name] = {t: i for i, t in enumerate(ts)}
    print(f"rewrote timestamps on {fixed} groups; {len(index)} groups indexed")
    # One convention across the whole file, or the next reader hits the same bug.
    widths = set()
    with h5py.File(h5_path, "r") as f:
        for name in f:
            widths.add(f[name]["timestamps"].dtype.itemsize)
    assert widths == {20}, f"timestamps still inconsistent: {widths}"
    print("all groups now |S20 with trailing Z")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--h5", default="/Users/marcomorandin/Desktop/thesis-dataset/images_all_v2.h5"
    )
    ap.add_argument(
        "--parquet",
        default="/Users/marcomorandin/Desktop/thesis-dataset/dataset_all.parquet",
    )
    ap.add_argument(
        "--out-parquet",
        default="/Users/marcomorandin/Desktop/thesis-dataset/dataset_all_v2.parquet",
    )
    a = ap.parse_args()
    idx = repair(Path(a.h5))
    reindex(Path(a.parquet), Path(a.out_parquet), idx)
