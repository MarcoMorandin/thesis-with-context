"""Do V-JEPA's PATCH tokens carry ramp signal that pooling destroys? (ticket 13 successor)

`pooling_bottleneck.py` showed raw 128 px crops carry ramp signal in their SPATIAL
layout: predicting csi at t+60 min out-of-sample, a 1x1 pooled feature is
neutral-to-negative on ramp steps (R^2 0.2833 -> 0.2787) while a 16x16 grid helps
(0.2983), monotonically in resolution.

Wave 2 acted on that by widening `n_soft_tokens` 1 -> 16 and ramp did not move
(0.1487 -> 0.1484, seed floor 0.0011). The reason is architectural:
LatentSummarizer.latent_queries is [1, n_vis_steps, d_model] — ONE query per visual
step — so ~800 V-JEPA patches are already pooled to a single vector before
CrossModalAdapter expands it to N tokens. Those N tokens add capacity, not
information.

The successor fix is N queries PER STEP in the summarizer. Before spending a
curriculum on it, this probe asks whether the information is there to recover:

    1x1   mean over the patch grid   = exactly what the summarizer emits today
    2x2 / 4x4   the grid, kept       = what N spatially-specialised queries reach

If the grid arms beat 1x1 on RAMP steps, the summarizer rewrite has headroom. If
they tie, V-JEPA's representation is the ceiling, no summarizer change helps, and
the visual branch is a measured negative result rather than an unfinished one.

CPU only, no model forward. Reads the same cached latents training consumes.
Cache layout assumed: <cache>/<dataset>_<site_id>_<origin_epoch_seconds>.pt

Usage (from MMTSFM/ on Leonardo):
    uv run python scripts/probes/latent_pooling_bottleneck.py \
        --cache /leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224_nonhrv_sp45 \
        --parquet /leonardo_scratch/fast/IscrC_MTSFM/data_v2/dataset_all.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch

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
HORIZONS = (1, 2, 4)  # steps of 30 min
GRID = 4  # finest spatial grid retained; 2x2 and 1x1 nest inside
FNAME = re.compile(r"^(?P<ds>.+?)_(?P<site>\d+)_(?P<epoch>\d+)\.pt$")


def block_pool(a: np.ndarray, g: int) -> np.ndarray:
    """[..., G0, G0, D] -> [..., g, g, D] by exact nested block means."""
    G0 = a.shape[-2]
    b = G0 // g
    return (
        a[..., : g * b, : g * b, :]
        .reshape(*a.shape[:-3], g, b, g, b, a.shape[-1])
        .mean(axis=(-4, -2))
    )


def r2_oos(X, y, sites, tt, eval_mask):
    """Ridge, fit on each plant's earliest 70% of time, scored on the rest."""
    fin = np.isfinite(X).all(1) & np.isfinite(y)
    fit = np.zeros(len(y), bool)
    for s in np.unique(sites):
        m = sites == s
        fit |= m & (tt <= np.quantile(tt[m], 0.70))
    fit &= fin
    ev = (~fit) & eval_mask & fin
    if ev.sum() < 150 or fit.sum() < 400:
        return float("nan"), int(ev.sum())
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


def pca(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(0)
    k = min(k, Xc.shape[0] - 1, Xc.shape[1])
    if k >= Xc.shape[1]:
        return Xc
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument(
        "--max-per-plant",
        type=int,
        default=1200,
        help="latents are ~1.6 MB each; 1200 x 14 plants ~= 27 GB of reads",
    )
    ap.add_argument("--pca-dim", type=int, default=64)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.cache, "**", "*.pt"), recursive=True)
    print(f"cache files: {len(files):,}")
    index: dict[tuple[str, int], str] = {}
    for f in files:
        m = FNAME.match(os.path.basename(f))
        if m:
            index[(m.group("site"), int(m.group("epoch")))] = f
    print(f"indexed latents: {len(index):,}")
    if not index:
        raise SystemExit(
            f"filename parse failed — sample: {os.path.basename(files[0])}"
        )

    df = pd.read_parquet(
        args.parquet,
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

    feats, tgt, ramp, sites, tt = [], [], [], [], []
    D = None
    for site, g in df.groupby("site_id", sort=True):
        g = g.reset_index(drop=True)
        y = g.norm_power.to_numpy(float)
        cs = g.clearsky_ghi.to_numpy(float)
        csi = g.csi.to_numpy(float)
        ts = g.timestamp_utc
        dt = ts.diff().dt.total_seconds().to_numpy()
        contig = np.zeros(len(g), bool)
        contig[1:] = dt[1:] == 1800.0
        day = (cs > 0) & np.isfinite(csi) & np.isfinite(y)
        pday = np.zeros(len(g), bool)
        pday[1:] = day[:-1]
        dy = np.zeros(len(g))
        dy[1:] = np.abs(y[1:] - y[:-1])
        valid = contig & day & pday
        if valid.sum() < 500:
            continue
        thr = np.quantile(dy[valid], 0.9)  # protocol_eval._ramp_masks
        is_ramp = valid & (dy >= thr)

        epochs = (ts.astype("int64") // 10**9).to_numpy()
        maxh = max(HORIZONS)
        taken = 0
        for i in range(3, len(g) - maxh):
            if taken >= args.max_per_plant:
                break
            if not valid[i] or not all(day[i + h] for h in HORIZONS):
                continue
            f = index.get((site, int(epochs[i])))
            if f is None:
                continue  # cache is on a coarser origin grid
            z = torch.load(f, map_location="cpu", weights_only=False)
            if isinstance(z, dict):
                z = z.get("latent", next(iter(z.values())))
            z = np.asarray(torch.as_tensor(z).float().numpy())  # [T_lat, P, D]
            if z.ndim == 2:
                z = z[None]
            T_lat, P, Dv = z.shape
            G0 = int(round(P**0.5))
            if G0 * G0 != P:
                raise SystemExit(f"patch count {P} is not square — inspect {f}")
            zg = z.reshape(T_lat, G0, G0, Dv)
            # mirror the pixel probe: a "now" summary and a motion term
            now = zg.mean(0)  # [G0, G0, D]
            mot = zg[-1] - zg[0]  # [G0, G0, D]
            pair = np.stack([now, mot])  # [2, G0, G0, D]
            feats.append(block_pool(pair, GRID).astype(np.float32))  # [2,4,4,D]
            tgt.append([csi[i + h] for h in HORIZONS])
            ramp.append(bool(is_ramp[i]))
            sites.append(site)
            tt.append(int(epochs[i]))
            taken += 1
            D = Dv
        print(f"  {site}: {taken} samples", flush=True)

    if not feats:
        raise SystemExit(
            "no origin matched the cache — the cache sits on a coarser grid than "
            "the 30-min power series; check TRAIN_STRIDE used at extraction"
        )

    F = np.stack(feats)  # [N, 2, 4, 4, D]
    TGT = np.asarray(tgt, float)
    RAMP = np.asarray(ramp, bool)
    SITES = np.asarray(sites)
    TT = np.asarray(tt, np.int64)
    print(f"\nsamples {len(F):,}  ramp {int(RAMP.sum()):,} ({RAMP.mean():.1%})")
    print(
        f"patch grid kept {GRID}x{GRID}, D={D}, feature array {F.nbytes / 1e9:.2f} GB\n"
    )

    ARMS = {}
    for gsz in (1, 2, 4):
        X = block_pool(F, gsz).reshape(len(F), -1)  # exact nested pooling
        tag = f"{gsz}x{gsz} grid" + (
            "   <- what the summarizer emits today" if gsz == 1 else ""
        )
        ARMS[tag] = pca(X, args.pca_dim) if X.shape[1] > args.pca_dim else X

    for hi, h in enumerate(HORIZONS):
        y = TGT[:, hi]
        print(f"=== csi at t+{h * 30} min — out-of-sample R^2 ===")
        print(f"{'visual features':<48} {'ALL':>9} {'RAMP':>9}")
        base = None
        for name, X in ARMS.items():
            ra, _ = r2_oos(X, y, SITES, TT, np.ones(len(y), bool))
            rr, nr = r2_oos(X, y, SITES, TT, RAMP)
            if base is None:
                base = rr
            print(f"{name:<48} {ra:>9.4f} {rr:>9.4f}")
        print(f"{'':<48} {'':>9} {'n=' + f'{nr:,}':>9}\n")

    print("Read: grid arms >> 1x1 on RAMP -> the summarizer rewrite (N queries per")
    print("step) has headroom. grid ~= 1x1 -> V-JEPA is the ceiling, and the visual")
    print("branch is a measured negative result rather than an unfinished one.")


if __name__ == "__main__":
    main()
