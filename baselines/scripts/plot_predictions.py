#!/usr/bin/env python3
"""Plot prediction results clustered by model architecture.

Groups models into:
1. classical_naive
2. deep_ts
3. ts_foundation
4. multimodal_vision

Generates 5 plots per cluster inside a dedicated subfolder, and a final
comparison plot in a 'comparison' folder showing the best model per cluster
plus smart_persistence.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Define architectural clusters
CLUSTERS = {
    "classical_naive": [
        "persistence",
        "smart_persistence",
        "climatology_hourly",
        "seasonal_naive",
        "lightgbm",
        "tabpfn",
    ],
    "deep_ts": [
        "mlp",
        "dlinear",
        "patchtst",
        "itransformer",
        "tft",
    ],
    "ts_foundation": [
        "chronos2_zs",
        "chronos2_ft",
        "chronos2_oracle",
        "chronos2_oracle_ft",
        "timesfm_zs",
        "tirex_zs",
        "ttm_zs",
        "ttm_ft",
        "cora",
        "ts_rag_orig",
        "cross_rag_orig",
    ],
    "multimodal_vision": [
        "aurora",
        "visionts_pp",
        "unicast",
        "crossvivit",
        "sunset",
        "solar_vlm",
        "time_vlm",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        default="results/predictions",
        help="Path to predictions directory containing *_pred.npz files",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory containing performance JSON files",
    )
    parser.add_argument(
        "--site",
        default="10793",
        help="Site ID to plot (e.g. 10793)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Window index of the reference model to plot. If omitted, a random window is chosen.",
    )
    parser.add_argument(
        "--num-plots",
        type=int,
        default=5,
        help="Number of random plots to generate if --window is omitted.",
    )
    parser.add_argument(
        "--out-dir",
        default="plots",
        help="Base output directory to save plots",
    )
    return parser.parse_args()


def load_metrics(results_dir: Path, site: str) -> dict[str, dict]:
    """Load NMAE and Ramp NRMSE for each model at a specific site from results JSON files."""
    metrics = {}
    for f in results_dir.glob("*.json"):
        if "predictions" in f.parts or f.name.endswith("_losses.npz"):
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
            if "results" in d and "per_plant" in d["results"]:
                per_plant = d["results"]["per_plant"]
                if str(site) in per_plant:
                    model_name = d.get("manifest", {}).get("model")
                    if not model_name:
                        # Fallback to parsing filename
                        model_name = f.name.split("_s2_")[0].split("_orig_")[0]
                    plant_data = per_plant[str(site)]
                    nmae = plant_data.get("nmae")
                    nrmse_ramp = plant_data.get("nrmse_ramp")
                    
                    if nmae is not None and nrmse_ramp is not None:
                        # If model already has a score, keep the one with lower nrmse_ramp
                        existing = metrics.get(model_name)
                        if existing is None or nrmse_ramp < existing["nrmse_ramp"]:
                            metrics[model_name] = {
                                "nmae": nmae,
                                "nrmse_ramp": nrmse_ramp
                            }
        except Exception:
            pass
    return metrics


def get_model_metric(model_name: str, metrics: dict) -> dict | None:
    """Helper to retrieve metrics for a model using exact or prefix matching."""
    if model_name in metrics:
        return metrics[model_name]
    for k, v in metrics.items():
        if k == model_name or k.startswith(model_name + "_"):
            return v
    return None


def find_best_models(datasets: dict, metrics: dict[str, dict]) -> dict[str, str]:
    """Find the best available model in each cluster based on Ramp NRMSE."""
    best_models = {}
    for cluster_name, model_list in CLUSTERS.items():
        available = [m for m in model_list if m in datasets]
        if not available:
            continue
        
        # Select available model with lowest Ramp NRMSE
        best_model = None
        best_score = float("inf")
        for m in available:
            m_metrics = get_model_metric(m, metrics)
            score = m_metrics["nrmse_ramp"] if m_metrics else float("inf")
            if score < best_score:
                best_score = score
                best_model = m
                
        # Fallback if no metric found
        if best_model is None:
            best_model = available[0]
            
        best_models[cluster_name] = best_model
        # Print best model with its Ramp NRMSE score
        best_metrics = get_model_metric(best_model, metrics)
        score_str = f"{best_metrics['nrmse_ramp']:.4f}" if best_metrics else "inf"
        print(f"  Best model for cluster '{cluster_name}': {best_model} (Ramp NRMSE: {score_str})")
    return best_models


def plot_window(
    w_idx: int,
    datasets: dict,
    model_list: list[str],
    ref_true: np.ndarray,
    context_y: np.ndarray,
    metrics: dict[str, dict],
    title: str,
    save_path: Path,
):
    """Plot context, ground truth, and a specific list of models for a window."""
    plt.figure(figsize=(10, 6))
    
    # 1. Plot ground truth (Context + Future)
    context_len = len(context_y)
    context_x = np.arange(-context_len, 0)
    plt.plot(context_x, context_y, color="black", linestyle="--", linewidth=2.0, label="True Context")
    
    daylight_mask = ref_true > 1e-5
    ref_true_plot = ref_true.copy()
    ref_true_plot[~daylight_mask] = np.nan
    
    future_x = np.arange(12)
    plt.plot(future_x, ref_true_plot, color="black", linestyle="-", linewidth=2.5, label="True Future")

    # Mark the forecast origin (last observed historical value)
    plt.plot(-1, context_y[-1], "ko", markersize=6, zorder=5, label="Forecast Origin")

    # 2. Plot model predictions
    for model_name in model_list:
        if model_name not in datasets:
            continue
            
        d = datasets[model_name]
        m_true = d["true"]
        m_pred = d["pred"]
        
        # Align by finding index j that minimizes MSE with ref_true
        diff = m_true[:, :12] - ref_true
        mse = np.mean(diff ** 2, axis=1)
        best_j = np.argmin(mse)
        
        if mse[best_j] > 1e-2:  # Threshold allows slight scaling differences
            continue
            
        pred_y = m_pred[best_j, :12]
        pred_y_plot = pred_y.copy()
        pred_y_plot[~daylight_mask] = np.nan
        
        # Connect prediction to the last context point at x = -1
        plot_x = np.arange(-1, 12)
        plot_y = np.concatenate([[context_y[-1]], pred_y_plot])
        
        # Legend label with metric if available
        label = model_name
        m_metrics = get_model_metric(model_name, metrics)
        if m_metrics:
            label += f" (R-NRMSE: {m_metrics['nrmse_ramp']:.4f})"
            
        plt.plot(plot_x, plot_y, linewidth=1.5, alpha=0.85, label=label, marker="o", markersize=4)

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Time Step (Relative to forecast start)", fontsize=10)
    plt.ylabel("Normalized PV Power", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.axvline(x=0, color="gray", linestyle=":", linewidth=1.2)
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_scatter_plots(datasets: dict, base_out_dir: Path, site: str, metrics: dict[str, dict]):
    """Generate actual vs predicted scatter plots for each model."""
    scatter_dir = base_out_dir / "scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, d in datasets.items():
        pred = d["pred"].flatten()
        true = d["true"].flatten()
        
        # Filter out NaN values and night steps (true <= 1e-5)
        mask = ~np.isnan(pred) & ~np.isnan(true) & (true > 1e-5)
        pred = pred[mask]
        true = true[mask]
        
        if len(true) == 0:
            continue
            
        plt.figure(figsize=(8, 8))
        
        # Plot scatter
        plt.scatter(true, pred, alpha=0.3, color="blue", s=10, label="Predictions")
        
        # Identity line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="y = x (Perfect Forecast)")
        
        # Labels and Title
        title = f"Actual vs Predicted — {model_name} (Site {site})"
        m_metrics = get_model_metric(model_name, metrics)
        if m_metrics:
            title += f"\nNMAE: {m_metrics['nmae']:.4f} | Ramp NRMSE: {m_metrics['nrmse_ramp']:.4f}"
            
        plt.title(title, fontsize=12, fontweight="bold")
        plt.xlabel("Actual Normalized PV Power", fontsize=10)
        plt.ylabel("Predicted Normalized PV Power", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        save_path = scatter_dir / f"scatter_{model_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
    print("Generated actual vs predicted scatter plots in 'scatter' folder.")


def main():
    args = parse_args()
    pred_dir = Path(args.predictions_dir)
    results_dir = Path(args.results_dir)
    base_out_dir = Path(args.out_dir)

    if not pred_dir.exists():
        print(f"Error: Predictions directory '{pred_dir}' does not exist.")
        return

    # Find all prediction files for this site
    pattern = f"*_{args.site}_pred.npz"
    pred_files = list(pred_dir.glob(pattern))
    if not pred_files:
        print(f"Error: No prediction files found matching '{pattern}' in '{pred_dir}'.")
        return

    # Load all npz datasets
    datasets = {}
    for f in pred_files:
        name_part = f.name[:-len(f"_pred.npz")]
        model_name = "_".join(name_part.split("_")[:-1])
        try:
            data = np.load(f)
            datasets[model_name] = {
                "pred": data["pred"],
                "true": data["true"]
            }
        except Exception as e:
            print(f"Warning: Failed to load {f.name}: {e}")

    if not datasets:
        print("Error: Failed to load any datasets.")
        return

    # Pick reference model
    ref_candidates = [
        m for m, d in datasets.items()
        if d["true"].ndim == 2 and d["true"].shape[1] == 12
    ]
    if not ref_candidates:
        print("Error: Could not find reference model with horizon 12.")
        return

    ref_model = next((n for n in ["smart_persistence", "dlinear", "mlp"] if n in ref_candidates), ref_candidates[0])
    ref_data = datasets[ref_model]
    n_windows = ref_data["true"].shape[0]
    global_true = ref_data["true"].flatten()

    # Load metrics
    metrics = load_metrics(results_dir, args.site)
    print("Loaded performance metrics.")

    # Find best model in each cluster
    print("Finding best model for each cluster...")
    best_models = find_best_models(datasets, metrics)

    # Determine which windows to plot
    if args.window is not None:
        windows_to_plot = [args.window]
    else:
        # Seed for reproducible windows across runs
        random.seed(42)
        windows_to_plot = sorted(random.sample(range(1, n_windows), min(args.num_plots, n_windows - 1)))

    # Process each window
    for w_idx in windows_to_plot:
        if w_idx < 0 or w_idx >= n_windows:
            continue

        ref_true = ref_data["true"][w_idx]
        s_idx = 12 * w_idx
        
        # Context extraction
        context_len = 5
        if s_idx >= context_len:
            context_y = global_true[s_idx - context_len : s_idx]
        else:
            context_y = np.pad(global_true[:s_idx], (context_len - s_idx, 0), constant_values=np.nan)

        # 1. Plot per-cluster directories
        for cluster_name, model_list in CLUSTERS.items():
            cluster_dir = base_out_dir / cluster_name
            cluster_dir.mkdir(parents=True, exist_ok=True)
            
            title = f"{cluster_name.replace('_', ' ').title()} Group — Site {args.site}, Window {w_idx}"
            save_path = cluster_dir / f"plot_site_{args.site}_w{w_idx}.png"
            
            # Plot only models belonging to this cluster
            plot_window(
                w_idx=w_idx,
                datasets=datasets,
                model_list=model_list,
                ref_true=ref_true,
                context_y=context_y,
                metrics=metrics,
                title=title,
                save_path=save_path,
            )
            print(f"  Saved {cluster_name} plot for window {w_idx}")

        # 2. Plot overall best comparison
        comparison_dir = base_out_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Models to plot: best of each cluster + smart_persistence as base reference
        comparison_models = list(best_models.values())
        if "smart_persistence" not in comparison_models and "smart_persistence" in datasets:
            comparison_models.append("smart_persistence")
            
        title = f"Architecture Comparison (Best of Clusters) — Site {args.site}, Window {w_idx}"
        save_path = comparison_dir / f"plot_site_{args.site}_w{w_idx}.png"
        
        plot_window(
            w_idx=w_idx,
            datasets=datasets,
            model_list=comparison_models,
            ref_true=ref_true,
            context_y=context_y,
            metrics=metrics,
            title=title,
            save_path=save_path,
        )
        print(f"  Saved comparison plot for window {w_idx}")

    # Generate scatter plots
    plot_scatter_plots(datasets, base_out_dir, args.site, metrics)


if __name__ == "__main__":
    main()
