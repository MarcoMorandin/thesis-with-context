"""Pack the non-HRV PNG crops into images_all_v2.h5 and reindex the parquet.

v1 stored raw uint8 arrays. v2 cannot: uk_pv now has 4.0M frames at 128x128x3
(night included, 15-minute cadence), which is 196 GB raw against 146 GB free.
Measured alternatives on real frames:

    raw        48.0 KB/frame   196.6 GB
    gzip-4     36.3 KB/frame   148.8 GB
    lzf        45.4 KB/frame   186.0 GB
    png bytes  23.5 KB/frame    96.4 GB    <- the only one that fits

So frames are stored as PNG-ENCODED BYTES in a variable-length uint8 dataset.
`/format` is set to "png" at the file root and consumers must decode; use
`read_frame` below rather than indexing the dataset directly. The cost is one
decode per frame read, which is paid once on Leonardo when the V-JEPA latent
cache is built, not per training epoch.

goes_pvdaq is re-encoded the same way rather than copied raw, so the file has
exactly ONE format. A file where some groups are arrays and others are encoded
bytes is a trap for every future reader.

The parquet is reindexed because the frame set changed shape entirely: v1 had
daylight frames only, v2 covers every 15-minute slot any visual window reaches
back into, so `image_h5_index` from v1 points at the wrong rows. Written to a
NEW parquet, leaving the dataset of record untouched so the old numbers stay
reproducible.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

STAMP_RE = re.compile(r"(\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d)Z\.png$")
VLEN = h5py.vlen_dtype(np.uint8)


def read_frame(h5: h5py.File, group: str, index: int) -> np.ndarray:
    """Decode one frame. The only supported way to read a v2 file."""
    if h5.attrs.get("format") != "png":
        return h5[group]["images"][index]
    return np.asarray(Image.open(io.BytesIO(h5[group]["images"][index].tobytes())))


def _stamp_to_iso(stamp: str) -> str:
    """`2019-06-15T10-45-00` -> `2019-06-15T10:45:00`, matching v1's |S20."""
    d, t = stamp.split("T")
    return f"{d}T{t.replace('-', ':')}"


def pack_uk(png_dir: Path, out: h5py.File) -> dict[str, dict[str, int]]:
    """One group per plant, PNG bytes stored verbatim -- no decode/re-encode.

    Reading the file off disk and writing those same bytes keeps the pack pass
    I/O-bound instead of CPU-bound, and guarantees the stored image is bit
    identical to the one that was verified.
    """
    index: dict[str, dict[str, int]] = {}
    dirs = sorted(p for p in png_dir.iterdir() if p.name.startswith("uk_pv_"))
    t0 = time.time()
    for n, d in enumerate(dirs):
        files = sorted(d.glob("*.png"))
        stamps = []
        for f in files:
            m = STAMP_RE.search(f.name)
            if m:
                stamps.append((_stamp_to_iso(m.group(1)), f))
        stamps.sort()
        g = out.create_group(d.name)
        ds = g.create_dataset("images", (len(stamps),), dtype=VLEN)
        for i, (_, f) in enumerate(stamps):
            ds[i] = np.frombuffer(f.read_bytes(), dtype=np.uint8)
        g.create_dataset(
            "timestamps",
            data=np.array([s for s, _ in stamps], dtype="S20"),
        )
        index[d.name] = {s: i for i, (s, _) in enumerate(stamps)}
        if n % 10 == 0:
            el = time.time() - t0
            print(
                f"  {d.name}  {len(stamps)} frames  "
                f"({n + 1}/{len(dirs)}, {el / 60:.1f} min)",
                flush=True,
            )
    return index


def pack_goes(old_h5: Path, out: h5py.File) -> dict[str, dict[str, int]]:
    """Re-encode goes_pvdaq from v1 arrays so the file has one uniform format."""
    index: dict[str, dict[str, int]] = {}
    with h5py.File(old_h5, "r") as src:
        for name in sorted(k for k in src if k.startswith("goes_pvdaq_")):
            imgs, ts = src[name]["images"], src[name]["timestamps"]
            g = out.create_group(name)
            ds = g.create_dataset("images", (imgs.shape[0],), dtype=VLEN)
            for i in range(imgs.shape[0]):
                buf = io.BytesIO()
                Image.fromarray(imgs[i]).save(buf, format="PNG")
                ds[i] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            g.create_dataset("timestamps", data=ts[:])
            index[name] = {
                t.decode() if isinstance(t, bytes) else str(t): i
                for i, t in enumerate(ts[:])
            }
            print(f"  {name}  {imgs.shape[0]} frames", flush=True)
    return index


def reindex(parquet: Path, out_parquet: Path, index: dict[str, dict[str, int]]) -> None:
    """Point image_h5_index at v2 positions; -1 where no frame exists.

    -1 rather than NaN or a silent drop: a missing frame is a fact the loader
    must be able to see and act on, and an integer column cannot hold NaN
    without becoming a float and losing exactness at large indices.
    """
    df = pd.read_parquet(parquet)
    df["site_id"] = df["site_id"].astype(str)
    key = df["dataset"].astype(str) + "_" + df["site_id"]
    iso = pd.to_datetime(df["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S")
    new = np.full(len(df), -1, dtype=np.int64)
    for grp, table in index.items():
        m = (key == grp).to_numpy()
        if not m.any():
            continue
        new[m] = [table.get(s, -1) for s in iso[m]]
    df["image_h5_index"] = new
    miss = int((new < 0).sum())
    print(
        f"reindexed {len(df)} rows; {miss} with no frame ({100 * miss / len(df):.3f}%)",
        flush=True,
    )
    for ds_name in df["dataset"].unique():
        sel = df["dataset"] == ds_name
        print(
            f"  {ds_name}: {int((new[sel.to_numpy()] < 0).sum())} unmatched "
            f"of {int(sel.sum())}",
            flush=True,
        )
    df.to_parquet(out_parquet, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", default="/Volumes/dataset/uk_nonhrv_png")
    ap.add_argument(
        "--old-h5", default="/Users/marcomorandin/Desktop/thesis-dataset/images_all.h5"
    )
    ap.add_argument(
        "--parquet",
        default="/Users/marcomorandin/Desktop/thesis-dataset/dataset_all.parquet",
    )
    ap.add_argument(
        "--out-h5",
        default="/Users/marcomorandin/Desktop/thesis-dataset/images_all_v2.h5",
    )
    ap.add_argument(
        "--out-parquet",
        default="/Users/marcomorandin/Desktop/thesis-dataset/dataset_all_v2.parquet",
    )
    ap.add_argument("--need-gb", type=float, default=115.0)
    a = ap.parse_args()

    out_h5 = Path(a.out_h5)
    free = shutil.disk_usage(out_h5.parent).free / 1e9
    if free < a.need_gb:
        print(
            f"ABORT: {free:.0f} GB free at {out_h5.parent}, need ~{a.need_gb:.0f} GB. "
            f"Filling the disk mid-write would also endanger the v1 backup living "
            f"beside it.",
            file=sys.stderr,
        )
        return 1
    print(f"{free:.0f} GB free, writing {out_h5}", flush=True)

    t0 = time.time()
    with h5py.File(out_h5, "w") as out:
        out.attrs["format"] = "png"
        out.attrs["note"] = "images datasets hold PNG-encoded bytes; see read_frame"
        index = pack_uk(Path(a.png_dir), out)
        index.update(pack_goes(Path(a.old_h5), out))
    print(
        f"h5 written in {(time.time() - t0) / 60:.1f} min, "
        f"{out_h5.stat().st_size / 1e9:.1f} GB",
        flush=True,
    )

    reindex(Path(a.parquet), Path(a.out_parquet), index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
