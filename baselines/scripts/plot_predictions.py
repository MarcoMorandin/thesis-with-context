#!/usr/bin/env python3
"""Plot predictions from different models on the same ground-truth windows.

Reads the synced `*_pred.npz` files in `baselines/results/predictions/`,
automatically aligns the forecast windows by matching their ground-truth targets,
and plots predictions of all available models side-by-side for qualitative comparison.
"""

import argparse
import os
import random
from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_dir", default="baselines/results/predictions",
                        help="Directory containing the *_pred.npz files")
    parser.add_argument("--output_dir", default="baselines/results/plots",
                        help="Directory to save the plots")
    parser.add_argument("--site", default=None,
                        help="Site ID to plot (default: auto-select first available)")
    parser.add_argument("--num_plots", type=int, default=5,
                        help="Number of forecast windows to plot")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for window selection")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    pred_dir = Path(args.predictions_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        print(f"Error: Predictions directory '{pred_dir}' does not exist.")
        print("Please sync the predictions folder from Leonardo first.")
        return

    # Find all npz files
    npz_files = list(pred_dir.glob("*_pred.npz"))
    if not npz_files:
        print(f"Error: No '*_pred.npz' files found in '{pred_dir}'.")
        return

    # Group files by site ID
    # Filenames are typically: <model>_<site>_pred.npz or uk_pv_test_<site>_pred.npz
    site_to_files = {}
    for f in npz_files:
        stem = f.stem
        # split by underscore and find the numeric parts
        parts = stem.split("_")
        site = None
        for p in parts:
            if p.isdigit():
                site = p
                break
        if site is None:
            continue
        site_to_files.setdefault(site, []).append(f)

    if not site_to_files:
        print("Error: Could not extract site IDs from filenames.")
        return

    # Select site
    available_sites = sorted(site_to_files.keys())
    selected_site = args.site
    if selected_site is None:
        selected_site = available_sites[0]
        print(f"Auto-selected site: {selected_site} (Available: {', '.join(available_sites)})")
    elif selected_site not in site_to_files:
        print(f"Error: Selected site '{selected_site}' not found. Available: {', '.join(available_sites)}")
        return

    site_files = site_to_files[selected_site]
    print(f"Found {len(site_files)} models for site {selected_site}:")

    # Load predictions and true values for each model
    models_data = {}
    for f in site_files:
        # Determine model name
        # e.g., crossvivit_10793_pred -> crossvivit
        # uk_pv_test_10793_pred -> time_vlm
        # visionts_pp_10793_pred -> visionts_pp
        stem = f.stem
        if "uk_pv_test" in stem:
            model_name = "time_vlm"
        elif "visionts_pp" in stem:
            model_name = "visionts_pp"
        elif "crossvivit" in stem:
            model_name = "crossvivit"
        elif "sunset" in stem:
            model_name = "sunset"
        else:
            model_name = stem.split(f"_{selected_site}")[0]
        
        try:
            data = np.load(f)
            pred = data["pred"]
            true = data["true"]
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred[..., 0]
            if true.ndim == 3 and true.shape[-1] == 1:
                true = true[..., 0]
            models_data[model_name] = {
                "pred": pred,
                "true": true
            }
            print(f"  - {model_name}: shape {pred.shape}")
        except Exception as e:
            print(f"  Warning: failed to load {f.name}: {e}")

    if not models_data:
        print("Error: No prediction data loaded.")
        return

    # Choose a reference model to align others against
    # Prefer visionts_pp, then sunset, then crossvivit, then first available
    ref_candidates = ["visionts_pp", "sunset", "crossvivit"]
    ref_name = None
    for c in ref_candidates:
        if c in models_data:
            ref_name = c
            break
    if ref_name is None:
        ref_name = list(models_data.keys())[0]

    ref_data = models_data[ref_name]
    ref_true = ref_data["true"]
    ref_pred = ref_data["pred"]

    # Filter for interesting daylight windows (max true value > 0.1)
    daylight_indices = [i for i in range(len(ref_true)) if np.max(ref_true[i]) > 0.1]
    if not daylight_indices:
        print("Warning: No windows with true power > 0.1 found. Using all windows.")
        daylight_indices = list(range(len(ref_true)))

    # Pre-calculate matches for all daylight windows to find the best ones to plot
    window_scores = []  # list of (ref_idx, num_matches, matches_dict)
    for ref_idx in daylight_indices:
        target_true = ref_true[ref_idx]
        matches_dict = {}
        num_matches = 0
        for model_name, data in models_data.items():
            if model_name == ref_name:
                matches_dict[model_name] = ref_idx
                num_matches += 1
                continue
            m_true = data["true"]
            diffs = np.abs(m_true - target_true[None, :])
            matches = np.where(np.all(diffs < 1e-3, axis=1))[0]
            if len(matches) > 0:
                matches_dict[model_name] = matches[0]
                num_matches += 1
        window_scores.append((ref_idx, num_matches, matches_dict))

    # Sort windows by number of matches (descending)
    window_scores.sort(key=lambda x: x[1], reverse=True)
    best_windows = window_scores[:args.num_plots]
    print(f"Selected {len(best_windows)} windows with maximum model alignment (each matches {best_windows[0][1]} models)...")

    import matplotlib.pyplot as plt

    for plot_num, (ref_idx, num_matches, matches_dict) in enumerate(best_windows):
        target_true = ref_true[ref_idx]
        horizon = len(target_true)
        steps = np.arange(1, horizon + 1)

        plt.figure(figsize=(10, 6))
        
        # Plot ground truth once
        plt.plot(steps, target_true, 'k--', label="Ground Truth (True)", linewidth=2.5, zorder=5)

        # Plot predictions for matched models
        for model_name, m_idx in matches_dict.items():
            m_pred = models_data[model_name]["pred"]
            plt.plot(steps, m_pred[m_idx], label=f"{model_name} (pred)", linewidth=1.8)

        missing_models = [m for m in models_data if m not in matches_dict]
        if missing_models:
            print(f"  Note: Window {ref_idx} did not match: {', '.join(missing_models)}")

        plt.title(f"Qualitative Forecast Comparison - Site {selected_site} (Window {ref_idx})")
        plt.xlabel("Forecast Step (Half-Hourly, 1 to 12)")
        plt.ylabel("Normalized Power")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="upper right")
        
        plot_path = out_dir / f"site_{selected_site}_window_{ref_idx}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved plot {plot_num + 1}/{len(best_windows)} to {plot_path}")

    print(f"\n✓ Done! All plots saved to: {out_dir}/")

if __name__ == "__main__":
    main()
