"""CrossViViT on the uk_pv multimodal track — self-contained runner (added on vendor).

NOT upstream: the authors' code is a Hydra/Lightning project around the DeepLake
`SunLake` dataset (georeferenced EUMETSAT multi-channel satellite + ground
stations + optical flow + elevation). This runner imports and drives the
*original* model `src.models.cross_vivit.RoCrossViViT` unchanged, feeding it our
uk_pv multimodal windows via `tier6.uk_multimodal.UKMultimodalDataset`.

Framing: CrossViViT fuses a context video with a station timeseries over a shared
window. We use the last `pred_len` steps of each history window as that shared
input (satellite `V` frames + PV/covariate `ts`) and train the model to forecast
the next `pred_len` PV steps (`y_future`).

Documented approximations (uk_pv ≠ SunLake; see tier6/vendor/VENDOR_NOTICE.md):
  - single-channel 128px→S crops (ctx_channels=1) vs SunLake's multi-band frames;
  - no optical-flow channels, no elevation;
  - per-pixel `ctx_coords` synthesized as a small lat/lon grid around the plant
    (uk128 crops carry no per-pixel geo-grid); `ts_coords` = plant lat/lon.
These weaken CrossViViT's spatial grounding — report the row with this caveat.

Dumps `crossvivit_<site>_pred.npz` (`pred`,`true` (N,H)) for import_predictions.

    python run_ukpv.py --data <all_curated.parquet> --h5 <images_uk128.h5> \
        --out results_ukpv --epochs 20 --pred_len 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                      # crossvivit/ → `import src...`
sys.path.insert(0, str(_HERE.parents[2]))           # baselines/ → tier6.*, common.*

from src.models.cross_vivit import RoCrossViViT  # noqa: E402
from src.models.modules.positional_encoding import Cyclical_embedding  # noqa: E402
from tier6.uk_multimodal import UKMultimodalDataset, sites_for_split  # noqa: E402
from common import config  # noqa: E402

N_COV = len(config.COV_COLS)
TS_CHANNELS = 1 + N_COV          # PV + covariates
GRID_DEG = 0.5                   # synthetic ctx grid half-extent (deg) around plant


def build_model(img_size: int, pred_len: int, n_quantiles: int) -> RoCrossViViT:
    """Original RoCrossViViT, hyperparams mirroring configs/pl_module/cross_vivit.yaml,
    with channel/length dims set for the uk_pv inputs."""
    return RoCrossViViT(
        image_size=[img_size, img_size], patch_size=[8, 8],
        time_coords_encoder=Cyclical_embedding([12, 31, 24, 60]),
        dim=64, depth=4, heads=4, mlp_ratio=4,
        ctx_channels=1, ts_channels=TS_CHANNELS, ts_length=pred_len,
        out_dim=1, dim_head=64, dropout=0.3, freq_type="lucidrains", pe_type="rope",
        num_mlp_heads=n_quantiles, use_glu=True,
        ctx_masking_ratio=0.0, ts_masking_ratio=0.15,
        decoder_dim=128, decoder_depth=4, decoder_heads=6, decoder_dim_head=128,
        max_freq=64, use_self_attention=True,
    )


def _time_feats(ts_unix: np.ndarray) -> np.ndarray:
    """(L,) unix → (L,4) [month, day, hour, minute] for Cyclical_embedding."""
    t = np.asarray(ts_unix, dtype="datetime64[s]")
    months = t.astype("datetime64[M]").astype(int) % 12 + 1
    days = (t.astype("datetime64[D]") - t.astype("datetime64[M]")).astype(int) + 1
    secs = (t.astype("datetime64[s]").astype(np.int64)) % 86400
    return np.stack([months, days, secs // 3600, (secs % 3600) // 60], axis=1).astype(np.float32)


def window_tensors(ds: UKMultimodalDataset, idx: list[int], H: int, S: int):
    """Assemble a CrossViViT batch (last-H window) from uk multimodal items."""
    ctx, ts, tco, y, m, ctx_co, latlons = [], [], [], [], [], [], []
    for i in idx:
        it = ds[i]
        v = it["V"][-H:, :, :, :]                      # (H,1,S,S)
        hist_cov = it["cov"][:config.HISTORY_STEPS][-H:]      # (H, N_COV)
        pv = it["y_hist"][-H:]                                # (H,)
        ts_feat = np.concatenate([pv[:, None], hist_cov], axis=1)  # (H, TS_CHANNELS)
        lat, lon = it["latlon"]
        # ctx_coords live on the PATCH grid (S/patch), one coord per ViT token
        pg = S // 8
        grid = np.linspace(-GRID_DEG, GRID_DEG, pg, dtype=np.float32)
        gy, gx = np.meshgrid(grid, grid, indexing="ij")
        cco = np.stack([(lat + gy) / 90.0, (lon + gx) / 180.0], axis=0)  # (2,pg,pg)
        tfeat = _time_feats(it["timestamps"][:config.HISTORY_STEPS][-H:])  # (H,4)
        tco_full = np.broadcast_to(tfeat[:, :, None, None], (H, 4, S, S))

        ctx.append(v); ts.append(ts_feat); tco.append(tco_full)
        y.append(it["y_future"]); m.append(it["mask_future"])
        ctx_co.append(cco); latlons.append((lat, lon))
    tt = lambda a: torch.from_numpy(np.asarray(a, np.float32))
    ctx = tt(ctx)                                      # (B,H,1,S,S)
    ts = tt(ts)                                        # (B,H,TS)
    tco = tt(tco)                                      # (B,H,4,S,S) pixel-res
    ctx_co = tt(ctx_co)                                # (B,2,pg,pg)
    ll = np.asarray(latlons, np.float32)               # (B,2)
    ts_co = torch.from_numpy(
        np.stack([ll[:, 0] / 90.0, ll[:, 1] / 180.0], axis=1)
        .reshape(len(idx), 2, 1, 1).astype(np.float32))  # (B,2,1,1)
    return ctx, ctx_co, ts, ts_co, tco, tt(y), tt(m)


def run_epoch(model, ds, order, H, S, bs, opt, device, train: bool):
    model.train(train)
    tot, n = 0.0, 0
    for lo in range(0, len(order), bs):
        idx = order[lo:lo + bs]
        ctx, ctx_co, ts, ts_co, tco, y, mask = window_tensors(ds, idx, H, S)
        ctx, ctx_co, ts, ts_co, tco, y, mask = (
            x.to(device) for x in (ctx, ctx_co, ts, ts_co, tco, y, mask))
        with torch.set_grad_enabled(train):
            out, *_ = model(ctx, ctx_co, ts, ts_co, tco, mask=train)
            yhat = out.mean(dim=2).squeeze(-1)         # (B,H)
            loss = ((yhat - y) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()) * len(idx); n += len(idx)
    return tot / max(n, 1)


def predict(model, ds, H, S, bs, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for lo in range(0, len(ds), bs):
            idx = list(range(lo, min(lo + bs, len(ds))))
            ctx, ctx_co, ts, ts_co, tco, y, _ = window_tensors(ds, idx, H, S)
            ctx, ctx_co, ts, ts_co, tco = (
                x.to(device) for x in (ctx, ctx_co, ts, ts_co, tco))
            out, *_ = model(ctx, ctx_co, ts, ts_co, tco, mask=False)
            preds.append(out.mean(dim=2).squeeze(-1).cpu().numpy())
            trues.append(y.numpy())
    return np.clip(np.concatenate(preds), 0.0, 1.0), np.concatenate(trues)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=config.DEFAULT_DATA_PATH)
    ap.add_argument("--h5", default="/Volumes/SSD/standardized-dataset/images_uk128.h5")
    ap.add_argument("--out", default="results_ukpv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pred_len", type=int, default=config.HORIZON_STEPS)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.6e-3)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--max_train_windows", type=int, default=20000)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, S = args.pred_len, args.img_size
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def mk(part):
        return UKMultimodalDataset(
            site_ids=sites_for_split(part), data_path=args.data, h5_path=args.h5,
            history=config.HISTORY_STEPS, horizon=H, stride=args.stride, img_size=S)

    train_ds, val_ds = mk("train"), mk("val")
    tr_order = np.random.permutation(len(train_ds)).tolist()[: args.max_train_windows]
    print(f">>> train windows={len(tr_order)} val windows={len(val_ds)}")

    model = build_model(S, H, len(config.QUANTILE_LEVELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    best, bad = float("inf"), 0
    for ep in range(args.epochs):
        tl = run_epoch(model, train_ds, tr_order, H, S, args.batch_size, opt, device, True)
        vl = run_epoch(model, val_ds, list(range(len(val_ds))), H, S, args.batch_size, opt, device, False)
        print(f"epoch {ep+1}/{args.epochs}  train_mse={tl:.5f}  val_mse={vl:.5f}")
        if vl < best - 1e-5:
            best, bad = vl, 0
            torch.save(model.state_dict(), out / "crossvivit_best.pt")
        else:
            bad += 1
            if bad >= 5:
                print("early stop"); break
    if (out / "crossvivit_best.pt").exists():
        model.load_state_dict(torch.load(out / "crossvivit_best.pt", map_location=device))

    print(">>> evaluating held-out test plants")
    for site in sites_for_split("test"):
        ds = UKMultimodalDataset(
            site_ids=[site], data_path=args.data, h5_path=args.h5,
            history=config.HISTORY_STEPS, horizon=H, stride=args.stride, img_size=S)
        if len(ds) == 0:
            print(f"    {site}: no windows, skip"); continue
        pred, true = predict(model, ds, H, S, args.batch_size, device)
        np.savez(out / f"crossvivit_{site}_pred.npz",
                 pred=pred.astype(np.float32), true=true.astype(np.float32))
        print(f"    {site}: {len(pred)} windows → crossvivit_{site}_pred.npz")
    print(f"✓ CrossViViT uk_pv done → {out}/crossvivit_*_pred.npz")


if __name__ == "__main__":
    main()
