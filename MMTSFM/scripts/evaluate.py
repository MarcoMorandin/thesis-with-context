#!/usr/bin/env python
"""Evaluation script for VisionChronos2 trained models and base Chronos-2.

Usage
-----
# Evaluate a trained checkpoint:
    uv run python scripts/evaluate.py \
        --ckpt /path/to/checkpoints/vision_chronos2/epoch=49-step=1234.ckpt \
        --data-dir /path/to/data \
        --dataset synthetic \
        --horizon 12 \
        --output-dir eval_results

# Evaluate pretrained base Chronos-2 (zero-shot, no ckpt needed):
    uv run python scripts/evaluate.py \
        --base-chronos2 \
        --pretrained-model amazon/chronos-2 \
        --data-dir /path/to/data \
        --dataset uk_pv \
        --horizon 12 \
        --output-dir eval_results/base

Output
------
eval_results/
  metrics.json          — dict of all metrics (MSE, MAE, MASE, sMAPE, CRPS, RMSE)
  metrics_table.txt     — pretty-printed table
  forecast_samples.png  — 12 random forecast vs truth panels
  residual_dist.png     — residual histogram + KDE
  quantile_coverage.png — calibration: nominal vs empirical quantile coverage
  error_by_horizon.png  — MAE decomposed per horizon step
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)
sys.path.insert(0, str(root / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from mmtsfm.data.datamodule import MMTSFMDataModule
from mmtsfm.models.chronos2.config import Chronos2CoreConfig
from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule
from mmtsfm.models.chronos2.model import Chronos2Model
from mmtsfm.models.chronos2.pipeline import Chronos2Pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib style
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d2e",
        "axes.edgecolor": "#3d4263",
        "axes.labelcolor": "#c8cfe8",
        "axes.titlecolor": "#e2e6ff",
        "xtick.color": "#8892b0",
        "ytick.color": "#8892b0",
        "grid.color": "#2d3154",
        "grid.linewidth": 0.6,
        "text.color": "#e2e6ff",
        "legend.facecolor": "#1a1d2e",
        "legend.edgecolor": "#3d4263",
        "font.family": "DejaVu Sans",
    }
)

ACCENT   = "#7c6af7"   # purple
ACCENT2  = "#53d8fb"   # cyan
ACCENT3  = "#ff6b9d"   # pink
NEUTRAL  = "#8892b0"
SUCCESS  = "#64ffda"


# ──────────────────────────────────────────────────────────────────────────────
# Collate helper (mirrors datamodule)
# ──────────────────────────────────────────────────────────────────────────────

def _collate_optional_z(batch):
    has_z = ["Z" in b for b in batch]
    if not all(has_z):
        batch = [{k: v for k, v in b.items() if k != "Z"} for b in batch]
    return torch.utils.data.dataloader.default_collate(batch)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics (all return Python floats)
# ──────────────────────────────────────────────────────────────────────────────

def _mse(y_true: Tensor, y_pred: Tensor) -> float:
    return float(((y_true - y_pred) ** 2).mean())


def _mae(y_true: Tensor, y_pred: Tensor) -> float:
    return float((y_true - y_pred).abs().mean())


def _rmse(y_true: Tensor, y_pred: Tensor) -> float:
    return math.sqrt(_mse(y_true, y_pred))


def _mase(y_true: Tensor, y_pred: Tensor, y_ctx: Tensor, period: int = 1) -> float:
    naive = (y_ctx[..., period:] - y_ctx[..., :-period]).abs().mean() + 1e-8
    return float((y_true - y_pred).abs().mean() / naive)


def _smape(y_true: Tensor, y_pred: Tensor) -> float:
    denom = (y_true.abs() + y_pred.abs()).clamp(min=1e-8)
    return float(100.0 * (2.0 * (y_true - y_pred).abs() / denom).mean())


def _crps(y_true: Tensor, q_preds: Tensor, q_levels: list[float]) -> float:
    """Pinball-loss CRPS estimator."""
    ql = torch.tensor(q_levels, dtype=y_true.dtype, device=y_true.device)
    y_exp = y_true.unsqueeze(-1)          # (..., H, 1)
    errors = y_exp - q_preds              # (..., H, Q)
    pinball = torch.where(errors >= 0, ql * errors, (ql - 1) * errors)
    return float(2.0 * pinball.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Forecast runner
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(
    model: VisionChronos2LightningModule,
    dataloader: DataLoader,
    device: torch.device,
    q_levels: list[float],
) -> dict:
    """Return aggregated metrics and arrays for plotting.

    Chronos2Output.quantile_preds shape: (B, Q, H_full)
    where H_full = num_output_patches * output_patch_size >= H.
    We crop to H (the actual forecast horizon) before all calculations.
    """
    model.eval()
    model.to(device)

    all_y_true:  list[Tensor] = []
    all_y_pred:  list[Tensor] = []
    all_y_ctx:   list[Tensor] = []
    all_q_preds: list[Tensor] = []

    mse_sum = mae_sum = mase_sum = smape_sum = crps_sum = 0.0
    n = 0

    # Index of median quantile (0.5) — position 4 in the default 9-level set
    mid_idx = len(q_levels) // 2

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

        inputs = model._unpack_batch(batch)
        out    = model.model.forward(**inputs)

        H     = batch["Y_future"].shape[2]   # true forecast horizon
        y_ctx  = inputs["context"]            # (B, T)
        y_true = inputs["future_target"]      # (B, H)

        # quantile_preds: (B, Q, H_full)  — crop trailing H_full dimension to H
        if out.quantile_preds is not None:
            qp    = out.quantile_preds           # (B, Q, H_full)
            qp_h  = qp[:, :, :H]                # (B, Q, H)  — crop
            y_pred = qp_h[:, mid_idx, :]         # (B, H)     — median
            q_full = qp_h.permute(0, 2, 1)       # (B, H, Q)  — for CRPS / plots
        else:
            # Naïve persistence fallback
            y_pred = y_ctx[:, -1:].expand_as(y_true)
            q_full = None

        # Running metrics
        mse_sum   += _mse(y_true, y_pred)
        mae_sum   += _mae(y_true, y_pred)
        mase_sum  += _mase(y_true, y_pred, y_ctx)
        smape_sum += _smape(y_true, y_pred)
        if q_full is not None:
            crps_sum += _crps(y_true, q_full, q_levels)

        all_y_true.append(y_true.cpu())
        all_y_pred.append(y_pred.cpu())
        all_y_ctx.append(y_ctx.cpu())
        if q_full is not None:
            all_q_preds.append(q_full.cpu())

        n += 1

    if n == 0:
        raise ValueError("DataLoader produced no batches!")

    metrics = {
        "mse":   mse_sum  / n,
        "rmse":  math.sqrt(mse_sum / n),
        "mae":   mae_sum  / n,
        "mase":  mase_sum / n,
        "smape": smape_sum / n,
    }
    if all_q_preds:
        metrics["crps"] = crps_sum / n

    arrays = {
        "y_true":  torch.cat(all_y_true, dim=0).float().numpy(),   # (N, H)
        "y_pred":  torch.cat(all_y_pred, dim=0).float().numpy(),   # (N, H)
        "y_ctx":   torch.cat(all_y_ctx,  dim=0).float().numpy(),   # (N, T)
        "q_preds": torch.cat(all_q_preds, dim=0).float().numpy()   # (N, H, Q)
                   if all_q_preds else None,
    }
    return metrics, arrays


@torch.no_grad()
def run_eval_pipeline(
    pipeline: Chronos2Pipeline,
    dataloader: DataLoader,
    device: torch.device,
    q_levels: list[float],
) -> dict:
    """Eval using a raw Chronos2Pipeline (pretrained / zero-shot).

    Handles MMTSFMDataModule batch format: Y [BS, N, T, 1] / Y_future [BS, N, H, 1].
    Uses pipeline.predict_quantiles; median = q_levels[mid] for y_pred.
    """
    all_y_true:  list[Tensor] = []
    all_y_pred:  list[Tensor] = []
    all_y_ctx:   list[Tensor] = []
    all_q_preds: list[Tensor] = []

    mse_sum = mae_sum = mase_sum = smape_sum = crps_sum = 0.0
    n = 0
    mid_idx = len(q_levels) // 2

    for batch in dataloader:
        Y     = batch["Y"].to(device)        # [BS, N, T, 1]
        Y_fut = batch["Y_future"].to(device) # [BS, N, H, 1]
        BS, N, T, _ = Y.shape
        H = Y_fut.shape[2]

        y_ctx  = Y.reshape(BS * N, T).float()      # (B, T)
        y_true = Y_fut.reshape(BS * N, H).float()  # (B, H)

        # Pipeline creates an internal DataLoader with pin_memory — pass CPU tensors.
        quantiles_list, mean_list = pipeline.predict_quantiles(
            y_ctx.cpu().unsqueeze(1),         # (B, 1, T) on CPU
            prediction_length=H,
            quantile_levels=q_levels,
        )
        # each element: (1, H, Q) / (1, H) — squeeze entity dim, stack over batch
        q_tensor = torch.stack([q.squeeze(0) for q in quantiles_list]).to(device)  # (B, H, Q)
        y_pred   = q_tensor[:, :, mid_idx]                                          # (B, H) median

        mse_sum   += _mse(y_true, y_pred)
        mae_sum   += _mae(y_true, y_pred)
        mase_sum  += _mase(y_true, y_pred, y_ctx)
        smape_sum += _smape(y_true, y_pred)
        crps_sum  += _crps(y_true, q_tensor, q_levels)

        all_y_true.append(y_true.cpu())
        all_y_pred.append(y_pred.cpu())
        all_y_ctx.append(y_ctx.cpu())
        all_q_preds.append(q_tensor.cpu())

        n += 1

    if n == 0:
        raise ValueError("DataLoader produced no batches!")

    metrics = {
        "mse":   mse_sum  / n,
        "rmse":  math.sqrt(mse_sum / n),
        "mae":   mae_sum  / n,
        "mase":  mase_sum / n,
        "smape": smape_sum / n,
        "crps":  crps_sum / n,
    }
    arrays = {
        "y_true":  torch.cat(all_y_true,  dim=0).float().numpy(),
        "y_pred":  torch.cat(all_y_pred,  dim=0).float().numpy(),
        "y_ctx":   torch.cat(all_y_ctx,   dim=0).float().numpy(),
        "q_preds": torch.cat(all_q_preds, dim=0).float().numpy(),
    }
    return metrics, arrays


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_forecast_samples(arrays: dict, q_levels: list[float], out_path: Path, n_samples: int = 12):
    """Grid of n_samples random forecast panels with uncertainty bands."""
    y_true  = arrays["y_true"]   # (N, H)
    y_pred  = arrays["y_pred"]   # (N, H)
    y_ctx   = arrays["y_ctx"]    # (N, T)
    q_preds = arrays["q_preds"]  # (N, H, Q) or None

    N = y_true.shape[0]
    H = y_true.shape[1]
    T = y_ctx.shape[1]
    n_samples = min(n_samples, N)

    rng   = np.random.default_rng(42)
    idxs  = rng.choice(N, size=n_samples, replace=False)
    ncols = 4
    nrows = math.ceil(n_samples / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 2.8))
    fig.suptitle("Forecast Samples — Ground Truth vs Prediction", fontsize=14, y=1.01, color="#e2e6ff")
    axes = axes.flatten()

    ctx_x = np.arange(-T, 0)
    hor_x = np.arange(0, H)

    # Determine quantile band indices (10–90)
    q_lo_idx = 0              # 0.1
    q_hi_idx = len(q_levels) - 1  # 0.9
    q_lo2_idx = 1             # 0.2
    q_hi2_idx = len(q_levels) - 2  # 0.8

    for i, idx in enumerate(idxs):
        ax = axes[i]
        ax.plot(ctx_x, y_ctx[idx], color=NEUTRAL, linewidth=0.9, label="Context")
        ax.plot(hor_x, y_true[idx],  color=SUCCESS, linewidth=1.4, label="Ground Truth", zorder=4)
        ax.plot(hor_x, y_pred[idx],  color=ACCENT,  linewidth=1.4, linestyle="--", label="Median", zorder=5)

        if q_preds is not None:
            ax.fill_between(hor_x,
                            q_preds[idx, :, q_lo_idx],
                            q_preds[idx, :, q_hi_idx],
                            alpha=0.18, color=ACCENT, label="10–90%")
            ax.fill_between(hor_x,
                            q_preds[idx, :, q_lo2_idx],
                            q_preds[idx, :, q_hi2_idx],
                            alpha=0.30, color=ACCENT)

        ax.axvline(0, color="#3d4263", linewidth=1.0, linestyle=":")
        ax.set_title(f"Sample {idx}", fontsize=8, pad=3)
        ax.grid(True, alpha=0.4)
        ax.set_xlabel("")
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")

    # Hide unused axes
    for j in range(n_samples, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path}")


def plot_residual_distribution(arrays: dict, out_path: Path):
    """Histogram + KDE of residuals (y_true - y_pred)."""
    residuals = (arrays["y_true"] - arrays["y_pred"]).ravel()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Residual Distribution  (y_true − ŷ)", fontsize=13)

    counts, bins, _ = ax.hist(
        residuals, bins=80, density=True,
        color=ACCENT, alpha=0.65, edgecolor="none", label="Residuals"
    )
    # Simple Gaussian KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals, bw_method="scott")
    xs = np.linspace(residuals.min(), residuals.max(), 400)
    ax.plot(xs, kde(xs), color=ACCENT2, linewidth=2, label="KDE")
    ax.axvline(0, color=ACCENT3, linewidth=1.5, linestyle="--", label="Zero")
    ax.axvline(float(residuals.mean()), color=SUCCESS, linewidth=1.5,
               linestyle=":", label=f"Mean={residuals.mean():.3f}")

    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path}")


def plot_quantile_coverage(arrays: dict, q_levels: list[float], out_path: Path):
    """Calibration plot: nominal quantile level vs empirical coverage."""
    y_true  = arrays["y_true"]    # (N, H)
    q_preds = arrays["q_preds"]   # (N, H, Q)
    if q_preds is None:
        print("  [SKIP] No quantile forecasts — skipping coverage plot.")
        return

    empirical = []
    for i, q in enumerate(q_levels):
        covered = (y_true <= q_preds[:, :, i]).mean()
        empirical.append(float(covered))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Quantile Calibration (Nominal vs Empirical)", fontsize=13)

    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1.2, linestyle="--", label="Perfect calibration")
    ax.plot(q_levels, empirical, color=ACCENT, linewidth=2, marker="o",
            markersize=6, label="Model")

    # Shade deviation
    ax.fill_between(q_levels, q_levels, empirical,
                    alpha=0.2, color=ACCENT3, label="Deviation")

    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path}")


def plot_error_by_horizon(arrays: dict, out_path: Path):
    """MAE decomposed per horizon step to reveal error accumulation."""
    y_true = arrays["y_true"]   # (N, H)
    y_pred = arrays["y_pred"]   # (N, H)

    H = y_true.shape[1]
    mae_per_step = np.abs(y_true - y_pred).mean(axis=0)   # (H,)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("MAE per Horizon Step", fontsize=13)

    steps = np.arange(1, H + 1)
    bars = ax.bar(steps, mae_per_step, color=ACCENT, alpha=0.85, edgecolor="none")

    # Colour-code by magnitude
    vmin, vmax = mae_per_step.min(), mae_per_step.max()
    cmap = plt.get_cmap("plasma")
    for bar, val in zip(bars, mae_per_step):
        t = (val - vmin) / (vmax - vmin + 1e-8)
        bar.set_facecolor(cmap(t))

    ax.plot(steps, mae_per_step, color=ACCENT2, linewidth=1.5,
            marker="o", markersize=4, zorder=5)
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("MAE")
    ax.set_xticks(steps)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path}")


def plot_metrics_summary(metrics: dict, out_path: Path):
    """Bar chart summarising all scalar metrics."""
    names  = list(metrics.keys())
    values = [metrics[k] for k in names]

    fig, ax = plt.subplots(figsize=(len(names) * 1.4 + 1, 4))
    ax.set_title("Evaluation Metrics Summary", fontsize=13)

    colors = [ACCENT, ACCENT2, ACCENT3, SUCCESS, "#ffb347", "#ff6b6b"]
    bars = ax.bar(names, values, color=colors[:len(names)], alpha=0.88, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, color="#e2e6ff"
        )

    ax.set_ylabel("Value")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Metrics table
# ──────────────────────────────────────────────────────────────────────────────

def print_and_save_metrics(metrics: dict, out_path: Path):
    """Pretty-print metrics and save as plain text."""
    border = "─" * 40
    lines = [
        "╭" + border + "╮",
        f"│{'EVALUATION METRICS':^40}│",
        "├" + border + "┤",
    ]
    for k, v in metrics.items():
        lines.append(f"│  {k.upper():<18} {v:>16.6f}  │")
    lines.append("╰" + border + "╯")
    table = "\n".join(lines)
    print(table)
    out_path.write_text(table)
    print(f"  → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a VisionChronos2 checkpoint or base Chronos-2 (zero-shot)."
    )
    p.add_argument("--ckpt",              default=None,  type=Path, help="Path to .ckpt file (omit with --base-chronos2)")
    p.add_argument("--base-chronos2",     action="store_true",      help="Evaluate pretrained Chronos-2 zero-shot (no ckpt)")
    p.add_argument("--pretrained-model",  default="amazon/chronos-2", help="HF model ID or local path for --base-chronos2")
    p.add_argument("--data-dir",          required=True, type=Path, help="Root data directory")
    p.add_argument("--dataset",    default="synthetic",       help="dataset_name passed to MMTSFMDataModule")
    p.add_argument("--split",      default="test",            help="Split to evaluate: test | val")
    p.add_argument("--horizon",    default=None,   type=int,  help="Forecast horizon H")
    p.add_argument("--batch-size", default=16,     type=int)
    p.add_argument("--num-workers",default=0,      type=int)
    p.add_argument("--output-dir", default="eval_results", type=Path)
    p.add_argument("--device",     default="auto",            help="auto | cpu | cuda | mps")
    p.add_argument("--n-samples",  default=12,     type=int,  help="Number of forecast panels to plot")
    p.add_argument("--num-entities", default=10,   type=int)
    p.add_argument("--hist-steps",   default=None, type=int)
    args = p.parse_args()
    if not args.base_chronos2 and args.ckpt is None:
        p.error("--ckpt is required unless --base-chronos2 is set")
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[eval] device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if args.base_chronos2:
        print(f"[eval] Loading pretrained Chronos-2: {args.pretrained_model}")
        # Mirror lightning_module.py loading pattern: load HF config, force
        # use_grassmann=False (base amazon/chronos-2 has standard attention —
        # our Chronos2CoreConfig defaults to True which causes index OOB on forward),
        # then restore the quantiles buffer which HF from_pretrained zeros out.
        _config = Chronos2CoreConfig.from_pretrained(args.pretrained_model)
        _config.use_grassmann = False
        _base_model = Chronos2Model.from_pretrained(
            args.pretrained_model,
            config=_config,
            ignore_mismatched_sizes=True,
        )
        _base_model.quantiles.data.copy_(
            torch.tensor(_config.chronos_config["quantiles"], dtype=_base_model.dtype)
        )
        pipeline = Chronos2Pipeline(model=_base_model)
        pipeline.model.to(device)
        pipeline.model.eval()
        print("[eval] Pipeline loaded.")
    else:
        print(f"[eval] Loading checkpoint: {args.ckpt}")
        model = VisionChronos2LightningModule.load_from_checkpoint(
            args.ckpt,
            map_location=device,
            strict=False,
            weights_only=False,  # checkpoint contains OmegaConf DictConfig (PyTorch 2.6+)
        )
        model.eval()
        print("[eval] Checkpoint loaded.")

    # ── Build datamodule ────────────────────────────────────────────────────
    print(f"[eval] Building datamodule (split={args.split}, dataset={args.dataset})")
    dm = MMTSFMDataModule(
        data_dir=str(args.data_dir),
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_entities=args.num_entities,
        hist_steps=args.hist_steps,
        horizon=args.horizon,
    )
    dm.setup(stage="test" if args.split == "test" else "fit")

    if args.split == "test":
        dataloader = DataLoader(
            dm.test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=_collate_optional_z,
        )
    elif args.split == "val":
        dataloader = DataLoader(
            dm.val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=_collate_optional_z,
        )
    else:
        raise ValueError(f"Unknown split: {args.split}. Use 'test' or 'val'.")

    # ── Run evaluation ───────────────────────────────────────────────────────
    print("[eval] Running evaluation …")
    if args.base_chronos2:
        metrics, arrays = run_eval_pipeline(pipeline, dataloader, device, q_levels)
    else:
        metrics, arrays = run_eval(model, dataloader, device, q_levels)

    # ── Save metrics JSON ────────────────────────────────────────────────────
    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2))
    print(f"  → {metrics_json}")

    # ── Print + save table ───────────────────────────────────────────────────
    print_and_save_metrics(metrics, out_dir / "metrics_table.txt")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("[eval] Generating plots …")

    plot_metrics_summary(metrics, out_dir / "metrics_summary.png")
    plot_forecast_samples(arrays, q_levels, out_dir / "forecast_samples.png", n_samples=args.n_samples)
    plot_residual_distribution(arrays, out_dir / "residual_dist.png")
    plot_quantile_coverage(arrays, q_levels, out_dir / "quantile_coverage.png")
    plot_error_by_horizon(arrays, out_dir / "error_by_horizon.png")

    print(f"\n✓ All outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
