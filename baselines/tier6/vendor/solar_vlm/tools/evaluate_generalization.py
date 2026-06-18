#!/usr/bin/env python3
"""
Professional Evaluation & Diagnostics Test Suite for Solar-VLM Generalization.
Computes overall vs daytime-only metrics, horizon step error profiles, persistence baselines,
and generates diagnostic plots.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.metrics import MAE, MSE, RMSE, R2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Solar-VLM generalization results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing the npy result files")
    parser.add_argument("--setting", type=str, required=True,
                        help="Setting folder name (contains pred_final.npy and true.npy)")
    parser.add_argument("--data_dir", type=str, default="/leonardo_scratch/fast/IscrC_MTSFM/SolarVLM/dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--dataset", type=str, choices=["SKIPPD", "WOLLONGONG"], default="WOLLONGONG",
                        help="Dataset name")
    parser.add_argument("--station", type=str, default="loc3",
                        help="Station name (Wollongong only)")
    parser.add_argument("--seq_len", type=int, default=60, help="Input sequence length")
    parser.add_argument("--label_len", type=int, default=30, help="Label length")
    parser.add_argument("--pred_len", type=int, default=15, help="Prediction horizon")
    return parser.parse_args()


def load_dataset(args):
    print(f"Loading dataset {args.dataset} from {args.data_dir}...")
    size = [args.seq_len, args.label_len, args.pred_len]
    
    if args.dataset == "SKIPPD":
        from data_provider.data_loader_skippd import Dataset_SKIPPD
        dataset = Dataset_SKIPPD(
            root_path=args.data_dir,
            flag="test",
            size=size,
            features="MS",
            target="pv",
            freq="t",
            use_era5=False
        )
    elif args.dataset == "WOLLONGONG":
        from data_provider.data_loader_wollongong import Dataset_WOLLONGONG
        dataset = Dataset_WOLLONGONG(
            root_path=args.data_dir,
            station=args.station,
            flag="test",
            size=size,
            features="MS",
            target="pv",
            freq="t"
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
    return dataset


def main():
    args = parse_args()
    
    # Paths
    run_dir = os.path.join(args.results_dir, args.setting)
    pred_path = os.path.join(run_dir, "pred_final.npy")
    true_path = os.path.join(run_dir, "true.npy")
    
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"Error: Could not find prediction files in {run_dir}")
        sys.exit(1)
        
    # Load predictions and true values [N, pred_len, S]
    pred = np.load(pred_path)
    true = np.load(true_path)
    
    print(f"Loaded predictions shape: {pred.shape}")
    print(f"Loaded true values shape: {true.shape}")
    
    N, pred_len, S = pred.shape
    assert pred_len == args.pred_len, f"Loaded pred_len ({pred_len}) does not match arg ({args.pred_len})"
    
    # Load dataset to get exact timestamps and persistence reference
    dataset = load_dataset(args)
    
    # Reconstruct exact timestamps for each forecast horizon point
    # ts_keys has length = len(data_x).
    # Sample i in dataset corresponds to forecast starting at index s_end = i + seq_len.
    # Prediction steps k (0 to pred_len-1) correspond to index s_end + k in ts_keys.
    ts_keys = dataset.ts_keys
    
    # Construct timestamp array for the entire predictions matrix: [N, pred_len]
    sample_timestamps = []
    for i in range(N):
        s_end = i + args.seq_len
        steps_ts = [ts_keys[s_end + k] for k in range(pred_len)]
        sample_timestamps.append(steps_ts)
    sample_timestamps = np.array(sample_timestamps)  # [N, pred_len]
    
    # Extract hour of day for every prediction point
    # timestamp format: YYYYMMDDHHMM
    hours = np.vectorize(lambda ts: int(ts[8:10]))(sample_timestamps)  # [N, pred_len]
    
    # Compute Persistence Baseline: repeat the last historical value of the input sequence
    # For sample i, the last historical value is at index i + seq_len - 1 in self.data_y
    persistence_pred = []
    for i in range(N):
        last_hist_norm = dataset.data_y[i + args.seq_len - 1]  # [S, 1]
        last_hist_denorm = dataset.inverse_transform(last_hist_norm).squeeze()  # scalar or array
        # Repeat for pred_len steps
        persistence_pred.append([last_hist_denorm] * pred_len)
    persistence_pred = np.array(persistence_pred)  # [N, pred_len]
    if persistence_pred.ndim == 2:
        persistence_pred = persistence_pred[:, :, np.newaxis]  # [N, pred_len, 1]
        
    # Daytime mask: hours between 06:00 and 18:00 (inclusive)
    daytime_mask = (hours >= 6) & (hours <= 18)  # [N, pred_len]
    daytime_mask_expanded = np.expand_dims(daytime_mask, -1)  # [N, pred_len, 1]
    
    # Nighttime mask
    nighttime_mask = ~daytime_mask
    nighttime_mask_expanded = np.expand_dims(nighttime_mask, -1)
    
    # -------------------------------------------------------------
    # Compute Metrics
    # -------------------------------------------------------------
    results = {}
    
    for name, mask in [("Overall", np.ones_like(daytime_mask_expanded, dtype=bool)),
                       ("Daytime", daytime_mask_expanded),
                       ("Nighttime", nighttime_mask_expanded)]:
        
        pred_masked = pred[mask]
        true_masked = true[mask]
        pers_masked = persistence_pred[mask]
        
        mae_m = MAE(pred_masked, true_masked)
        mse_m = MSE(pred_masked, true_masked)
        rmse_m = RMSE(pred_masked, true_masked)
        r2_m = R2(pred_masked, true_masked)
        
        mae_p = MAE(pers_masked, true_masked)
        mse_p = MSE(pers_masked, true_masked)
        rmse_p = RMSE(pers_masked, true_masked)
        r2_p = R2(pers_masked, true_masked)
        
        results[name] = {
            "Model": {"MAE": float(mae_m), "MSE": float(mse_m), "RMSE": float(rmse_m), "R2": float(r2_m)},
            "Persistence": {"MAE": float(mae_p), "MSE": float(mse_p), "RMSE": float(rmse_p), "R2": float(r2_p)}
        }
        
    # Horizon step metrics (step-by-step error profile)
    horizon_metrics = {"Overall": [], "Daytime": []}
    for step in range(pred_len):
        # Overall
        pred_step = pred[:, step, :]
        true_step = true[:, step, :]
        pers_step = persistence_pred[:, step, :]
        
        horizon_metrics["Overall"].append({
            "step": step + 1,
            "Model": {
                "MAE": float(MAE(pred_step, true_step)),
                "RMSE": float(RMSE(pred_step, true_step)),
                "R2": float(R2(pred_step, true_step))
            },
            "Persistence": {
                "MAE": float(MAE(pers_step, true_step)),
                "RMSE": float(RMSE(pers_step, true_step)),
                "R2": float(R2(pers_step, true_step))
            }
        })
        
        # Daytime
        day_mask = daytime_mask[:, step]
        if np.any(day_mask):
            pred_step_day = pred[day_mask, step, :]
            true_step_day = true[day_mask, step, :]
            pers_step_day = persistence_pred[day_mask, step, :]
            
            horizon_metrics["Daytime"].append({
                "step": step + 1,
                "Model": {
                    "MAE": float(MAE(pred_step_day, true_step_day)),
                    "RMSE": float(RMSE(pred_step_day, true_step_day)),
                    "R2": float(R2(pred_step_day, true_step_day))
                },
                "Persistence": {
                    "MAE": float(MAE(pers_step_day, true_step_day)),
                    "RMSE": float(RMSE(pers_step_day, true_step_day)),
                    "R2": float(R2(pers_step_day, true_step_day))
                }
            })
            
    # Diurnal Profile MAE
    diurnal_profile = []
    for hr in range(24):
        hr_mask = (hours == hr)
        hr_mask_expanded = np.expand_dims(hr_mask, -1)
        if np.any(hr_mask):
            p_m = pred[hr_mask_expanded]
            t_m = true[hr_mask_expanded]
            p_p = persistence_pred[hr_mask_expanded]
            
            diurnal_profile.append({
                "hour": hr,
                "Model_MAE": float(MAE(p_m, t_m)),
                "Model_RMSE": float(RMSE(p_m, t_m)),
                "Persistence_MAE": float(MAE(p_p, t_m)),
                "Persistence_RMSE": float(RMSE(p_p, t_m))
            })
            
    # Save JSON report
    report = {
        "args": vars(args),
        "results": results,
        "horizon_metrics": horizon_metrics,
        "diurnal_profile": diurnal_profile
    }
    
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Saved evaluation JSON report to: {eval_dir}/evaluation_report.json")
    
    # -------------------------------------------------------------
    # Plotting Diagnostics
    # -------------------------------------------------------------
    print("Generating diagnostic plots...")
    
    # 1. Horizon Error Plot (MAE & R2 vs step)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps_overall = [m["step"] for m in horizon_metrics["Overall"]]
    steps_day = [m["step"] for m in horizon_metrics["Daytime"]]
    
    # MAE plot
    axes[0].plot(steps_overall, [m["Model"]["MAE"] for m in horizon_metrics["Overall"]], 'o-', label="Model (Overall)", color='tab:blue')
    axes[0].plot(steps_overall, [m["Persistence"]["MAE"] for m in horizon_metrics["Overall"]], 'x--', label="Persistence (Overall)", color='tab:orange')
    if len(steps_day) > 0:
        axes[0].plot(steps_day, [m["Model"]["MAE"] for m in horizon_metrics["Daytime"]], 's-', label="Model (Daytime)", color='tab:green')
        axes[0].plot(steps_day, [m["Persistence"]["MAE"] for m in horizon_metrics["Daytime"]], 'd--', label="Persistence (Daytime)", color='tab:red')
    axes[0].set_xlabel("Forecast Horizon Step")
    axes[0].set_ylabel("MAE (W)")
    axes[0].set_title("Mean Absolute Error (MAE) vs Forecast Horizon")
    axes[0].grid(True, linestyle=':')
    axes[0].legend()
    
    # R2 plot
    axes[1].plot(steps_overall, [m["Model"]["R2"] for m in horizon_metrics["Overall"]], 'o-', label="Model (Overall)", color='tab:blue')
    axes[1].plot(steps_overall, [m["Persistence"]["R2"] for m in horizon_metrics["Overall"]], 'x--', label="Persistence (Overall)", color='tab:orange')
    if len(steps_day) > 0:
        axes[1].plot(steps_day, [m["Model"]["R2"] for m in horizon_metrics["Daytime"]], 's-', label="Model (Daytime)", color='tab:green')
        axes[1].plot(steps_day, [m["Persistence"]["R2"] for m in horizon_metrics["Daytime"]], 'd--', label="Persistence (Daytime)", color='tab:red')
    axes[1].set_xlabel("Forecast Horizon Step")
    axes[1].set_ylabel("R² Score")
    axes[1].set_title("R² Score vs Forecast Horizon")
    axes[1].grid(True, linestyle=':')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "horizon_error.png"), dpi=200)
    plt.close()
    
    # 2. Scatter Plot (Daytime predicted vs true)
    plt.figure(figsize=(7, 6))
    pred_day = pred[daytime_mask_expanded]
    true_day = true[daytime_mask_expanded]
    
    if len(pred_day) > 0:
        # Scatter using density coloring if large, else alpha
        if len(pred_day) > 10000:
            plt.hexbin(true_day, pred_day, gridsize=50, cmap='YlOrRd', mincnt=1)
            cb = plt.colorbar()
            cb.set_label('Count')
        else:
            plt.scatter(true_day, pred_day, alpha=0.15, color='tab:blue')
            
        max_val = max(np.max(true_day), np.max(pred_day))
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label="1:1 Perfect Forecast")
        plt.xlabel("True PV Generation (W)")
        plt.ylabel("Predicted PV Generation (W)")
        plt.title(f"Daytime predicted vs True PV ({args.dataset} {args.station})")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No daytime samples found in this dataset subset.", 
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title("Daytime predicted vs True PV (Empty)")
        
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "scatter_daytime.png"), dpi=200)
    plt.close()
    
    # 3. Diurnal Error Profile (MAE/RMSE by hour of day)
    fig, ax = plt.subplots(figsize=(9, 5))
    hrs = [d["hour"] for d in diurnal_profile]
    ax.plot(hrs, [d["Model_MAE"] for d in diurnal_profile], 'o-', label="Model MAE", color='tab:blue')
    ax.plot(hrs, [d["Persistence_MAE"] for d in diurnal_profile], 'x--', label="Persistence MAE", color='tab:orange')
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("MAE (W)")
    ax.set_title("MAE Profile by Hour of Day")
    ax.grid(True, linestyle=':')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "diurnal_error_profile.png"), dpi=200)
    plt.close()
    
    # 4. Qualitative Case Studies (forecast curve vs true curve)
    # Pick 4 representative indices spread across the test split
    case_indices = [0, N//4, N//2, (3*N)//4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx_num, test_idx in enumerate(case_indices):
        if test_idx >= N:
            continue
        ax = axes[idx_num]
        
        # Horizon timestamps for X axis label (HH:MM format)
        x_labels = [ts[8:12] for ts in sample_timestamps[test_idx]]
        x_labels_formatted = [f"{ts[:2]}:{ts[2:]}" for ts in x_labels]
        
        # Forecast horizon steps
        x_steps = range(1, pred_len + 1)
        
        ax.plot(x_steps, true[test_idx, :, 0], 'k-o', label="True PV", linewidth=2)
        ax.plot(x_steps, pred[test_idx, :, 0], 'b-s', label="Model Forecast")
        ax.plot(x_steps, persistence_pred[test_idx, :, 0], 'r--x', label="Persistence")
        
        ax.set_xticks(x_steps)
        ax.set_xticklabels(x_labels_formatted, rotation=45)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("PV Generation (W)")
        ax.set_title(f"Forecast Case Study: Sample #{test_idx} starting {sample_timestamps[test_idx, 0]}")
        ax.grid(True, linestyle=':')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "case_studies.png"), dpi=200)
    plt.close()
    
    print(f"Saved diagnostic plots to: {eval_dir}/")
    
    # -------------------------------------------------------------
    # Output Markdown Report Table
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Solar-VLM Generalization Metrics & Baselines Report")
    print("=" * 80)
    print(f"Dataset   : {args.dataset}")
    if args.dataset == "WOLLONGONG":
        print(f"Station   : {args.station}")
    print(f"Setting   : {args.setting}")
    print("-" * 80)
    
    # Format a table
    headers = ["Metric subset", "Model MAE", "Model RMSE", "Model R2", "Persist. MAE", "Persist. RMSE", "Persist. R2"]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'-|'*len(headers)}")
    for name in ["Overall", "Daytime", "Nighttime"]:
        m = results[name]["Model"]
        p = results[name]["Persistence"]
        row = [
            name,
            f"{m['MAE']:.2f}",
            f"{m['RMSE']:.2f}",
            f"{m['R2']:.4f}",
            f"{p['MAE']:.2f}",
            f"{p['RMSE']:.2f}",
            f"{p['R2']:.4f}"
        ]
        print(f"| {' | '.join(row)} |")
    print("=" * 80)
    print("\nNote: Standard overall R² is diurnal-inflated. Refer to 'Daytime' for true operational quality.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
