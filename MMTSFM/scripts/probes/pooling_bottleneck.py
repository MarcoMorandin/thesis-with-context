"""Does POOLING the satellite crop destroy the ramp signal? (ticket 13)

Model-free, CPU-only, ~10 min on a laptop. Paths default to the local mount;
override with MMTSFM_H5 / MMTSFM_PARQUET on the cluster.

v1 showed the pooled ("one token") feature helps on ALL steps but does nothing —
or hurts — on RAMP steps, while an 8x8 spatial grid helps on both. If that is
really a pooling bottleneck, then ramp skill should rise MONOTONICALLY as the
grid gets finer: 1x1 (pure pooling) < 4x4 < 8x8 < 16x16.

Decodes once at 16x16 and derives the coarser grids by nested block-averaging
(exact), so every arm sees identical pixels. Caches to npz so the sweep is free
to re-run.
"""

import io
import os
import sys

import h5py
import numpy as np
import pandas as pd
from PIL import Image

H5 = os.environ.get("MMTSFM_H5", "/Volumes/dataset/dataset/images_all_v2.h5")
PARQUET = os.environ.get(
    "MMTSFM_PARQUET", "/Volumes/dataset/dataset/dataset_all_v2.parquet"
)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_cache.npz")
TEST = [
    "10793",
    "11176",
    "11287",
    "12642",
    "13388",
    "13817",
    "18989",
    "26854",
    "26933",
    "27020",
    "3333",
    "6648",
    "7648",
    "7836",
]
FINE = 16
FRAME_LAG_MIN = 45
HORIZONS = (1, 2, 4)


def block(a, g):
    b = a.shape[-1] // g
    return a.reshape(*a.shape[:-2], g, b, g, b).mean(axis=(-3, -1))


def decode(buf):
    b = buf.tobytes() if isinstance(buf, np.ndarray) else bytes(buf)
    return np.asarray(Image.open(io.BytesIO(b)).convert("L"), dtype=np.float32)


if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    print(f"loaded cache: {CACHE}")
    FNOW, FDIFF = z["fnow"], z["fdiff"]
    P, TGT, RAMP, SITE, TT = z["P"], z["tgt"], z["ramp"], z["site"], z["tt"]
else:
    df = pd.read_parquet(
        PARQUET,
        columns=[
            "site_id",
            "timestamp_utc",
            "norm_power",
            "clearsky_ghi",
            "csi",
            "dataset",
        ],
    )
    df = df[(df.dataset == "uk_pv") & (df.site_id.isin(TEST))]
    df = df.sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)
    fn, fd, Pl, tg, rp, st, tl = [], [], [], [], [], [], []
    with h5py.File(H5, "r") as f:
        for site, g in df.groupby("site_id", sort=True):
            key = f"uk_pv_{site}"
            if key not in f:
                continue
            g = g.reset_index(drop=True)
            ts = g.timestamp_utc
            y = g.norm_power.to_numpy(float)
            cs = g.clearsky_ghi.to_numpy(float)
            csi = g.csi.to_numpy(float)
            dt = ts.diff().dt.total_seconds().to_numpy()
            contig = np.zeros(len(g), bool)
            contig[1:] = dt[1:] == 1800.0
            day = (cs > 0) & np.isfinite(csi) & np.isfinite(y)
            dy = np.zeros(len(g))
            dy[1:] = np.abs(y[1:] - y[:-1])
            pday = np.zeros(len(g), bool)
            pday[1:] = day[:-1]
            valid = contig & day & pday
            if valid.sum() < 500:
                continue
            thr = np.quantile(dy[valid], 0.9)
            is_ramp = valid & (dy >= thr)
            fts = f[key]["timestamps"][:]
            fmap = {
                (t.decode() if isinstance(t, bytes) else str(t)): i
                for i, t in enumerate(fts)
            }
            images = f[key]["images"]
            cache = {}

            def grid_at(tstamp):
                k = tstamp.strftime("%Y-%m-%dT%H:%M:%SZ")
                if k not in cache:
                    i = fmap.get(k)
                    cache[k] = None if i is None else block(decode(images[i]), FINE)
                return cache[k]

            maxh = max(HORIZONS)
            run = np.ones(len(g), bool)
            for j in range(1, maxh + 3):
                run[j:] &= contig[j:]
            csmax = max(cs[day].max(), 1e-9)
            for i in range(3, len(g) - maxh):
                if not valid[i] or not run[i + maxh]:
                    continue
                if not all(day[i + h] for h in HORIZONS):
                    continue
                if not (
                    np.isfinite(csi[i])
                    and np.isfinite(csi[i - 1])
                    and np.isfinite(csi[i - 2])
                ):
                    continue
                a = grid_at(ts.iloc[i])
                b = grid_at(ts.iloc[i] - pd.Timedelta(minutes=FRAME_LAG_MIN))
                if a is None or b is None:
                    continue
                fn.append(a)
                fd.append(a - b)
                Pl.append([csi[i], csi[i - 1], csi[i - 2], cs[i] / csmax])
                tg.append([csi[i + h] for h in HORIZONS])
                rp.append(is_ramp[i])
                st.append(site)
                tl.append(ts.iloc[i].value)
            print(f"  {site}: {len(fn):>7} cum samples", flush=True)
    FNOW = np.stack(fn)
    FDIFF = np.stack(fd)
    P = np.asarray(Pl, float)
    TGT = np.asarray(tg, float)
    RAMP = np.asarray(rp, bool)
    SITE = np.asarray(st)
    TT = np.asarray(tl, np.int64)
    np.savez_compressed(
        CACHE, fnow=FNOW, fdiff=FDIFF, P=P, tgt=TGT, ramp=RAMP, site=SITE, tt=TT
    )
    print(f"cached -> {CACHE}")

print(f"\nsamples {len(P):,}  ramp {int(RAMP.sum()):,} ({RAMP.mean():.1%})")
print(f"fine grid {FNOW.shape[1]}x{FNOW.shape[2]}\n")


def r2_oos(X, y, eval_mask):
    fin = np.isfinite(X).all(1) & np.isfinite(y)
    fit = np.zeros(len(y), bool)
    for s in np.unique(SITE):
        m = SITE == s
        fit |= m & (TT <= np.quantile(TT[m], 0.70))
    fit &= fin
    ev = (~fit) & eval_mask & fin
    if ev.sum() < 200:
        return np.nan, 0
    Xf, yf = X[fit], y[fit]
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    Xf = (Xf - mu) / sd
    Xe = (X[ev] - mu) / sd
    ybar = yf.mean()
    lam = 1e-2 * len(Xf)
    w = np.linalg.solve(Xf.T @ Xf + lam * np.eye(Xf.shape[1]), Xf.T @ (yf - ybar))
    pred = Xe @ w + ybar
    return 1 - ((y[ev] - pred) ** 2).sum() / ((y[ev] - y[ev].mean()) ** 2).sum(), int(
        ev.sum()
    )


def feats(g):
    """Spatial ANOMALY grid + motion grid at resolution g. g=1 -> pure pooling."""
    a = block(FNOW, g).reshape(len(FNOW), -1)
    d = block(FDIFF, g).reshape(len(FDIFF), -1)
    if g == 1:
        return np.column_stack([a, d])  # the "one token" arm: 2 numbers
    a = a - a.mean(1, keepdims=True)
    d = d - d.mean(1, keepdims=True)
    return np.column_stack(
        [
            a,
            d,
            FNOW.reshape(len(FNOW), -1).mean(1),
            FDIFF.reshape(len(FDIFF), -1).mean(1),
        ]
    )


for hi, h in enumerate(HORIZONS):
    y = TGT[:, hi]
    print(f"=== csi at t+{h * 30} min — out-of-sample R^2 vs spatial resolution ===")
    print(
        f"{'visual features':<40} {'dim':>5} {'ALL':>9} {'RAMP':>9} {'ΔRAMP vs P':>11}"
    )
    r_all, _ = r2_oos(P, y, np.ones(len(y), bool))
    r_ramp, n_r = r2_oos(P, y, RAMP)
    print(
        f"{'none (csi persistence only)':<40} {P.shape[1]:>5} {r_all:>9.4f} {r_ramp:>9.4f} {'—':>11}"
    )
    for g in (1, 2, 4, 8, 16):
        X = np.column_stack([P, feats(g)])
        ra, _ = r2_oos(X, y, np.ones(len(y), bool))
        rr, _ = r2_oos(X, y, RAMP)
        tag = f"{g}x{g} grid" + ("  <- pooled, = 1 token" if g == 1 else "")
        print(f"{tag:<40} {X.shape[1]:>5} {ra:>9.4f} {rr:>9.4f} {rr - r_ramp:>+11.4f}")
    print(f"{'':<40} {'':>5} {'':>9} {'n=' + f'{n_r:,}':>9}")
    print()
