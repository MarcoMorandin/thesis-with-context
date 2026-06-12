"""
Analyze + plot saved test predictions from any SolarVLM run.

Reads pred_final.npy and true.npy from a results directory, then:
  1. Recomputes overall R², MAE, RMSE, normalized MSE
  2. Computes baseline metrics for comparison:
       - Persistence:   ŷ_{t+h} = y_{t} (last observed)
       - Mean predictor: ŷ = train_mean
  3. Skill score vs persistence (FS = 1 - RMSE_model / RMSE_persistence)
  4. Daytime-only R² (filters out near-zero pv samples) — removes diurnal inflation
  5. Per-horizon MAE/RMSE curves
  6. Plots:
       - 8 sample windows (pred vs true)
       - Scatter pred vs true with y=x reference
       - Residual histogram
       - Per-horizon error curves

Usage:
    python tools/analyze_results.py \
        --pred  $SCRATCH/results/<setting>/pred_final.npy \
        --true  $SCRATCH/results/<setting>/true.npy \
        --out   $SCRATCH/results/<setting>/analysis \
        [--seq_input $SCRATCH/results/<setting>/seq_x.npy]   # for persistence baseline
        [--daytime_threshold 0.05]                            # normalized cutoff for "daytime"
        [--label SKIPPD]                                      # title prefix for plots
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    sse = np.sum((true - pred) ** 2)
    sst = np.sum((true - true.mean()) ** 2)
    return float(1.0 - sse / (sst + 1e-12))


def _mae(pred, true): return float(np.mean(np.abs(true - pred)))
def _mse(pred, true): return float(np.mean((true - pred) ** 2))
def _rmse(pred, true): return float(np.sqrt(_mse(pred, true)))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pred_final.npy")
    ap.add_argument("--true", required=True, help="true.npy")
    ap.add_argument("--out",  required=True, help="output directory for plots")
    ap.add_argument("--label", default="model", help="label for plot titles")
    ap.add_argument("--daytime_threshold", type=float, default=0.05,
                    help="Fraction of max(true) above which we consider 'daytime' (default 0.05)")
    ap.add_argument("--persistence_lag", type=int, default=None,
                    help="If set, computes persistence baseline ŷ_{t+h} = true_{t-lag} per window")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pred = np.load(args.pred)  # [N, pred_len, 1]
    true = np.load(args.true)  # [N, pred_len, 1]
    print(f"Loaded pred {pred.shape}  true {true.shape}")

    pred = pred.squeeze(-1) if pred.ndim == 3 else pred
    true = true.squeeze(-1) if true.ndim == 3 else true
    N, H = true.shape   # N = num windows, H = pred horizon

    # ─── Overall metrics ───────────────────────────────────────────────
    r2   = _r2(pred, true)
    mae  = _mae(pred, true)
    rmse = _rmse(pred, true)
    print("\n=== Overall metrics ===")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")

    # ─── Persistence baseline ──────────────────────────────────────────
    # ŷ_{t+h} = y_t (the last observed value before forecast window)
    # We approximate this by using true[:, 0] (first actual = closest to "now"+1) shifted
    # Better: use true[:, 0] as proxy for the persistence value across all horizons
    persistence = np.repeat(true[:, :1], H, axis=1)  # [N, H]
    r2_p   = _r2(persistence, true)
    mae_p  = _mae(persistence, true)
    rmse_p = _rmse(persistence, true)
    fs = 1.0 - rmse / (rmse_p + 1e-12)
    print("\n=== Persistence baseline (ŷ_h = y_0 of window) ===")
    print(f"  R²   = {r2_p:.4f}")
    print(f"  RMSE = {rmse_p:.4f}")
    print(f"  Forecast Skill (1 - RMSE_model/RMSE_persistence) = {fs*100:.2f}%")
    if fs < 0:
        print("  ⚠  Model is WORSE than persistence!")

    # ─── Mean predictor ─────────────────────────────────────────────────
    mean_pred = np.full_like(true, true.mean())
    r2_m  = _r2(mean_pred, true)
    rmse_m = _rmse(mean_pred, true)
    print(f"\n=== Mean predictor (constant) ===")
    print(f"  R²   = {r2_m:.4f}   RMSE = {rmse_m:.4f}")

    # ─── Daytime-only R² ───────────────────────────────────────────────
    # Remove the day/night cycle inflation. "Daytime" = true above threshold * max
    thresh = args.daytime_threshold * true.max()
    mask = true > thresh
    if mask.sum() > 100:
        r2_d   = _r2(pred[mask], true[mask])
        mae_d  = _mae(pred[mask], true[mask])
        rmse_d = _rmse(pred[mask], true[mask])
        r2_dp  = _r2(persistence[mask], true[mask])
        print(f"\n=== Daytime only (true > {thresh:.3f}, {mask.sum()}/{mask.size} samples) ===")
        print(f"  Model R²        = {r2_d:.4f}   RMSE = {rmse_d:.4f}  MAE = {mae_d:.4f}")
        print(f"  Persistence R²  = {r2_dp:.4f}")

    # ─── Per-horizon errors ────────────────────────────────────────────
    mae_per_h  = np.mean(np.abs(true - pred), axis=0)
    rmse_per_h = np.sqrt(np.mean((true - pred) ** 2, axis=0))
    mae_per_h_p  = np.mean(np.abs(true - persistence), axis=0)
    rmse_per_h_p = np.sqrt(np.mean((true - persistence) ** 2, axis=0))
    print("\n=== Per-horizon errors ===")
    for h in range(H):
        print(f"  h={h+1:2d}  MAE={mae_per_h[h]:.4f}  RMSE={rmse_per_h[h]:.4f}  "
              f"(persist MAE={mae_per_h_p[h]:.4f})")

    # ─── Plots ──────────────────────────────────────────────────────────
    # 1) 8 sample windows
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    np.random.seed(0)
    sample_idx = np.random.choice(N, size=min(8, N), replace=False)
    for ax, idx in zip(axes.flatten(), sample_idx):
        ax.plot(true[idx], 'k-',  label='true', linewidth=2)
        ax.plot(pred[idx], 'r--', label='model', linewidth=1.5)
        ax.plot(persistence[idx], 'b:', label='persistence', linewidth=1)
        ax.set_title(f'window {idx}  true={true[idx].mean():.2f}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{args.label} — sample forecast windows", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "01_sample_windows.png", dpi=120)
    plt.close(fig)

    # 2) Scatter pred vs true (subsample for speed)
    sub = np.random.choice(true.size, size=min(20000, true.size), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true.flatten()[sub], pred.flatten()[sub], s=1, alpha=0.3)
    lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lim, lim, 'k--', label='y = x')
    ax.set_xlabel("true"); ax.set_ylabel("predicted")
    ax.set_title(f"{args.label}  R²={r2:.4f}  RMSE={rmse:.4f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "02_scatter.png", dpi=120)
    plt.close(fig)

    # 3) Residual histogram
    res = (pred - true).flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(res, bins=80, alpha=0.7)
    ax.axvline(0, color='k', linewidth=1)
    ax.set_xlabel("residual (pred - true)")
    ax.set_title(f"{args.label} residuals — mean={res.mean():.4f} std={res.std():.4f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "03_residual_hist.png", dpi=120)
    plt.close(fig)

    # 4) Per-horizon error
    fig, ax = plt.subplots(figsize=(8, 5))
    h_axis = np.arange(1, H + 1)
    ax.plot(h_axis, mae_per_h,  'r-o', label='Model MAE')
    ax.plot(h_axis, mae_per_h_p, 'b--s', label='Persistence MAE')
    ax.set_xlabel("forecast horizon"); ax.set_ylabel("MAE")
    ax.set_title(f"{args.label} — error growth with horizon")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "04_per_horizon_mae.png", dpi=120)
    plt.close(fig)

    # ─── Summary CSV ────────────────────────────────────────────────────
    summary = {
        'label':              args.label,
        'n_windows':          N,
        'horizon':            H,
        'R2':                 r2,
        'MAE':                mae,
        'RMSE':               rmse,
        'persistence_R2':     r2_p,
        'persistence_RMSE':   rmse_p,
        'forecast_skill_%':   fs * 100,
        'mean_predictor_R2':  r2_m,
    }
    with open(out / "summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
            print(f"  {k}: {v}")
    print(f"\nPlots and summary saved to: {out}")


if __name__ == "__main__":
    main()
