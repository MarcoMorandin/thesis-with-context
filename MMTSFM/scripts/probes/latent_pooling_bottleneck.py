"""Do V-JEPA's PATCH tokens carry ramp signal that pooling destroys? (ticket 13 successor)

`pooling_bottleneck.py` showed that raw 128 px crops carry ramp signal in their
SPATIAL layout: predicting csi at t+60 min out-of-sample, a 1x1 pooled feature is
neutral-to-negative on ramp steps (R^2 0.2833 -> 0.2787) while a 16x16 grid helps
(0.2983), monotonically in resolution.

Wave 2 acted on that by widening `n_soft_tokens` 1 -> 16 and ramp did not move
(0.1487 -> 0.1484, seed floor 0.0011). The reason is architectural:
LatentSummarizer.latent_queries is [1, n_vis_steps, d_model] — ONE query per
visual step — so ~800 V-JEPA patches are already pooled to a single vector before
CrossModalAdapter expands it to N tokens. The N tokens add capacity, not
information.

The successor fix is N queries PER STEP in the summarizer. Before spending a
curriculum on it, this probe asks whether the information is there to recover:

    pooled     mean over the P patch tokens        = what the model gets today
    spatial    the patch grid, kept                = what N queries could reach

If spatial >> pooled on RAMP steps, the summarizer rewrite has headroom. If they
are equal, V-JEPA's representation is the ceiling and no summarizer change helps —
which would make the visual branch a measured negative result rather than an
unfinished one.

CPU only, no model forward. Reads the same cached latents training consumes.

Usage (Leonardo login node or a compute node, from MMTSFM/):
    uv run python scripts/probes/latent_pooling_bottleneck.py \
        --cache /leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224_nonhrv_sp45 \
        --parquet /leonardo_scratch/fast/IscrC_MTSFM/data_v2/dataset_all_v2.parquet
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


def r2_oos(X, y, sites, tt, eval_mask):
    """Ridge, fit on each plant's earliest 70% of time, scored on the rest."""
    fin = np.isfinite(X).all(1) & np.isfinite(y)
    fit = np.zeros(len(y), bool)
    for s in np.unique(sites):
        m = sites == s
        fit |= m & (tt <= np.quantile(tt[m], 0.70))
    fit &= fin
    ev = (~fit) & eval_mask & fin
    if ev.sum() < 200 or fit.sum() < 500:
        return float("nan")
    Xf, yf = X[fit], y[fit]
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    Xf = (Xf - mu) / sd
    Xe = (X[ev] - mu) / sd
    ybar = yf.mean()
    lam = 1e-2 * len(Xf)
    w = np.linalg.solve(Xf.T @ Xf + lam * np.eye(Xf.shape[1]), Xf.T @ (yf - ybar))
    pred = Xe @ w + ybar
    return 1 - ((y[ev] - pred) ** 2).sum() / ((y[ev] - y[ev].mean()) ** 2).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="V-JEPA latent cache dir")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-per-plant", type=int, default=4000)
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=64,
        help="PCA dim for the spatial arm; keeps ridge well-posed",
    )
    args = ap.parse_args()

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

    files = glob.glob(os.path.join(args.cache, "**", "*.pt"), recursive=True)
    print(f"cache files: {len(files):,}")
    if not files:
        raise SystemExit(f"no .pt under {args.cache}")

    # cache keys embed site + origin timestamp; adapt the regex if the extractor
    # naming changes (scripts/extract_video_embeddings.py owns it).
    pat = re.compile(r"(?P<site>\d+).*?(?P<ts>\d{8}T?\d{6})")
    index: dict[tuple[str, str], str] = {}
    for f in files:
        m = pat.search(os.path.basename(f))
        if m:
            index[(m.group("site"), m.group("ts"))] = f
    print(f"indexed latents: {len(index):,}")
    if not index:
        raise SystemExit(
            "could not parse site/timestamp from cache filenames — inspect "
            f"{os.path.basename(files[0])} and adjust `pat`"
        )

    rows_pool, rows_spat, tgt, ramp, sites, tt = [], [], [], [], [], []
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

        maxh = max(HORIZONS)
        taken = 0
        for i in range(3, len(g) - maxh):
            if taken >= args.max_per_plant:
                break
            if not valid[i] or not all(day[i + h] for h in HORIZONS):
                continue
            key = (site, ts.iloc[i].strftime("%Y%m%dT%H%M%S"))
            f = index.get(key) or index.get((site, ts.iloc[i].strftime("%Y%m%d%H%M%S")))
            if f is None:
                continue
            z = torch.load(f, map_location="cpu")
            if isinstance(z, dict):
                z = z.get("latent", next(iter(z.values())))
            z = np.asarray(z, dtype=np.float32)  # [T_lat, P, D] or [P, D]
            if z.ndim == 3:
                z = z.mean(0)  # collapse time, keep patches
            rows_pool.append(z.mean(0))  # POOLED: what the model gets
            rows_spat.append(z.reshape(-1))  # SPATIAL: the patch grid
            tgt.append([csi[i + h] for h in HORIZONS])
            ramp.append(bool(is_ramp[i]))
            sites.append(site)
            tt.append(ts.iloc[i].value)
            taken += 1
        print(f"  {site}: {taken} samples", flush=True)

    if not rows_pool:
        raise SystemExit("no samples matched the cache — check the filename regex")

    P = np.stack(rows_pool)
    S = np.stack(rows_spat)
    TGT = np.asarray(tgt, float)
    RAMP = np.asarray(ramp, bool)
    SITES = np.asarray(sites)
    TT = np.asarray(tt, np.int64)
    print(f"\nsamples {len(P):,}  ramp {int(RAMP.sum()):,} ({RAMP.mean():.1%})")
    print(f"pooled dim {P.shape[1]}   spatial dim {S.shape[1]}")

    # PCA the spatial arm so ridge stays well-posed at comparable dimensionality
    Sc = S - S.mean(0)
    k = min(args.pca_dim, Sc.shape[0] - 1, Sc.shape[1])
    _, _, Vt = np.linalg.svd(Sc, full_matrices=False)
    S_pca = Sc @ Vt[:k].T
    print(f"spatial -> PCA {k} dims\n")

    ARMS = {
        "pooled  (mean over patches, = today)": P,
        f"spatial (patch grid, PCA-{k})": S_pca,
        "both": np.column_stack([P, S_pca]),
    }

    for hi, h in enumerate(HORIZONS):
        y = TGT[:, hi]
        print(f"=== csi at t+{h * 30} min — out-of-sample R^2 ===")
        print(f"{'visual features':<40} {'ALL':>9} {'RAMP':>9}")
        for name, X in ARMS.items():
            ra = r2_oos(X, y, SITES, TT, np.ones(len(y), bool))
            rr = r2_oos(X, y, SITES, TT, RAMP)
            print(f"{name:<40} {ra:>9.4f} {rr:>9.4f}")
        print()

    print("Read: spatial >> pooled on RAMP -> the summarizer rewrite (N queries")
    print("per step) has headroom. spatial ~= pooled -> V-JEPA is the ceiling and")
    print("the visual branch is a measured negative result, not an unfinished one.")


if __name__ == "__main__":
    main()
