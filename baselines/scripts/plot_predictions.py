#!/usr/bin/env python3
"""Plot predictions from ALL baseline models on the same ground-truth windows.

Reads `*_pred.npz` files in `baselines/results/predictions/`, automatically
aligns forecast windows across models (which may have different strides and
even different target scales such as Z-scored targets from time_vlm), and
produces per-site comparison plots.

Usage (from project root):
    uv run --with matplotlib --with numpy python baselines/scripts/plot_predictions.py
    uv run --with matplotlib --with numpy python baselines/scripts/plot_predictions.py --site 10793 --num_plots 10
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# ── Known multi-word model prefixes (order matters: longest first) ──────────
# These are model names that contain digits or underscores that could confuse
# the generic parser.  The filename convention is <model>_<site>_pred.npz.
KNOWN_PREFIXES = [
    "uk_pv_test",          # time_vlm's naming on Leonardo
    "climatology_hourly",
    "smart_persistence",
    "seasonal_naive",
    "visionts_pp",
]

# Display-name overrides
DISPLAY_NAMES: dict[str, str] = {
    "uk_pv_test": "Time-VLM",
    "climatology_hourly": "Climatology (Hourly)",
    "smart_persistence": "Smart Persistence",
    "seasonal_naive": "Seasonal Naïve",
    "visionts_pp": "VisionTS++",
    "crossvivit": "CrossViViT",
    "sunset": "SunSet",
    "persistence": "Persistence",
    "lightgbm": "LightGBM",
    "dlinear": "DLinear",
    "mlp": "MLP",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "tft": "TFT",
    "timesfm_zs": "TimesFM (ZS)",
    "ttm_zs": "TTM (ZS)",
    "ttm_ft": "TTM (FT)",
    "chronos2_zs": "Chronos-2 (ZS)",
    "chronos2_ft": "Chronos-2 (FT)",
    "cora": "CORA",
    "tirex_zs": "TiRex (ZS)",
    "aurora": "Aurora",
    "solar_vlm": "Solar-VLM",
    "ts_rag": "TS-RAG",
    "cross_rag": "Cross-RAG",
}

# Tier ordering for legend: lower = plotted/listed first
TIER_ORDER: dict[str, int] = {
    "persistence": 0, "climatology_hourly": 0, "seasonal_naive": 0,
    "smart_persistence": 0,
    "lightgbm": 1,
    "dlinear": 2, "mlp": 2, "patchtst": 2, "itransformer": 2, "tft": 2,
    "timesfm_zs": 3, "ttm_zs": 3, "chronos2_zs": 3, "ttm_ft": 3,
    "chronos2_ft": 3, "cora": 3, "tirex_zs": 3,
    "ts_rag": 4, "cross_rag": 4,
    "crossvivit": 5, "sunset": 5, "uk_pv_test": 5,
    "visionts_pp": 6, "aurora": 6, "solar_vlm": 6,
}

# Color palette: distinct per tier
TIER_COLORS: dict[int, list[str]] = {
    0: ["#9e9e9e", "#bdbdbd", "#757575", "#616161"],    # greys for reference
    1: ["#8d6e63"],                                       # brown
    2: ["#42a5f5", "#1e88e5", "#1565c0", "#0d47a1", "#5c6bc0"],  # blues
    3: ["#66bb6a", "#43a047", "#2e7d32", "#1b5e20", "#00897b", "#00695c"],  # greens
    4: ["#ffa726", "#fb8c00"],                            # oranges
    5: ["#ef5350", "#e53935", "#c62828"],                  # reds
    6: ["#ab47bc", "#8e24aa", "#6a1b9a"],                  # purples
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions_dir", default="baselines/results/predictions",
                    help="Directory containing *_pred.npz files")
    p.add_argument("--output_dir", default="baselines/results/plots",
                    help="Directory to save plots")
    p.add_argument("--site", default=None,
                    help="Site ID to plot (default: auto-select most-covered site)")
    p.add_argument("--num_plots", type=int, default=10,
                    help="Number of forecast windows to plot")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Filename → (model, site) parsing ───────────────────────────────────────

def _parse_filename(stem: str) -> tuple[str, str] | None:
    """Extract (model_key, site_id) from a filename stem like 'dlinear_10793_pred'."""
    if not stem.endswith("_pred"):
        return None
    body = stem[: -len("_pred")]  # remove '_pred' suffix

    # Try known multi-word prefixes first
    for prefix in KNOWN_PREFIXES:
        if body.startswith(prefix + "_"):
            site = body[len(prefix) + 1:]
            if site.isdigit():
                return prefix, site

    # Generic: last purely-numeric segment is the site id
    m = re.match(r"^(.+?)_(\d+)$", body)
    if m:
        return m.group(1), m.group(2)
    return None


# ── Target-scale alignment (handles Z-score targets from time_vlm) ─────────

def _needs_rescale(true_arr: np.ndarray) -> bool:
    """Heuristic: if targets go outside [−0.1, 1.2] they are Z-scored."""
    return float(true_arr.min()) < -0.2 or float(true_arr.max()) > 1.5


def _rescale_to_norm(true_z: np.ndarray, pred_z: np.ndarray,
                     ref_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map Z-scored (true, pred) back to [0,1] norm_power space using a
    reference model's true values for the same windows.

    Uses per-window linear regression: y_norm = slope * z + intercept.
    Falls back to global stats if per-window fit is unstable.
    """
    # Global affine mapping from z → norm
    z_flat, y_flat = true_z.ravel(), ref_true.ravel()
    mask = np.isfinite(z_flat) & np.isfinite(y_flat) & (np.abs(z_flat) < 100)
    if mask.sum() < 10:
        return true_z, pred_z  # can't map
    slope, intercept = np.polyfit(z_flat[mask], y_flat[mask], 1)
    return true_z * slope + intercept, pred_z * slope + intercept


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    pred_dir = Path(args.predictions_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        print(f"Error: '{pred_dir}' does not exist.")
        return

    # ── 1. Discover & group files by (model, site) ──────────────────────
    npz_files = sorted(pred_dir.glob("*_pred.npz"))
    if not npz_files:
        print(f"Error: no *_pred.npz files in '{pred_dir}'.")
        return

    # site → {model_key: path}
    site_models: dict[str, dict[str, Path]] = {}
    for f in npz_files:
        parsed = _parse_filename(f.stem)
        if parsed is None:
            print(f"  ⚠ skipping unrecognised file: {f.name}")
            continue
        model_key, site = parsed
        site_models.setdefault(site, {})[model_key] = f

    if not site_models:
        print("Error: no parseable prediction files found.")
        return

    # ── 2. Pick site (most models, or user-specified) ────────────────────
    if args.site:
        if args.site not in site_models:
            print(f"Error: site '{args.site}' not found. "
                  f"Available: {sorted(site_models)}")
            return
        selected_site = args.site
    else:
        selected_site = max(site_models, key=lambda s: len(site_models[s]))

    models_for_site = site_models[selected_site]
    print(f"\nSite {selected_site}: {len(models_for_site)} models available")

    # ── 3. Load all model predictions for this site ──────────────────────
    models_data: dict[str, dict[str, np.ndarray]] = {}
    for model_key, fpath in sorted(models_for_site.items()):
        try:
            data = np.load(fpath)
            pred = np.asarray(data["pred"], dtype=np.float64)
            true = np.asarray(data["true"], dtype=np.float64)
            # Squeeze trailing singleton dims (time_vlm stores (N,12,1))
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred[..., 0]
            if true.ndim == 3 and true.shape[-1] == 1:
                true = true[..., 0]
            models_data[model_key] = {"pred": pred, "true": true}
            disp = DISPLAY_NAMES.get(model_key, model_key)
            print(f"  ✓ {disp:25s}  windows={pred.shape[0]:6d}  "
                  f"H={pred.shape[1]}  "
                  f"true∈[{true.min():.3f}, {true.max():.3f}]")
        except Exception as e:
            print(f"  ✗ {model_key}: {e}")

    if not models_data:
        print("No data loaded.")
        return

    # ── 4. Rescale Z-scored models ───────────────────────────────────────
    # Find a reference model in [0,1] scale to calibrate against
    ref_key = None
    for candidate in ["smart_persistence", "persistence", "visionts_pp",
                       "sunset", "crossvivit"]:
        if candidate in models_data and not _needs_rescale(
                models_data[candidate]["true"]):
            ref_key = candidate
            break
    if ref_key is None:
        for k, v in models_data.items():
            if not _needs_rescale(v["true"]):
                ref_key = k
                break

    for model_key in list(models_data):
        if _needs_rescale(models_data[model_key]["true"]):
            if ref_key is None:
                print(f"  ⚠ {model_key} appears Z-scored but no reference "
                      "model available; skipping rescale")
                continue
            print(f"  → Rescaling {DISPLAY_NAMES.get(model_key, model_key)} "
                  f"from Z-score to [0,1] using {ref_key}")
            ref_true = models_data[ref_key]["true"]
            zt = models_data[model_key]["true"]
            zp = models_data[model_key]["pred"]
            # Match windows between Z-scored and reference using correlation
            # to build the affine mapping
            nt, nz = ref_true.shape[0], zt.shape[0]
            # Use random subset for speed
            sample_ref = rng.choice(nt, min(500, nt), replace=False)
            sample_z = rng.choice(nz, min(500, nz), replace=False)
            ref_sub = ref_true[sample_ref]
            z_sub = zt[sample_z]
            # Normalise rows to unit variance for correlation
            ref_std = ref_sub.std(axis=1, keepdims=True)
            z_std = z_sub.std(axis=1, keepdims=True)
            ref_std[ref_std == 0] = 1.0
            z_std[z_std == 0] = 1.0
            ref_norm = (ref_sub - ref_sub.mean(axis=1, keepdims=True)) / ref_std
            z_norm = (z_sub - z_sub.mean(axis=1, keepdims=True)) / z_std
            corr = ref_norm @ z_norm.T / ref_sub.shape[1]
            # Find high-correlation pairs for affine fit
            pairs = np.argwhere(corr > 0.999)
            if len(pairs) > 5:
                ref_vals = ref_true[sample_ref[pairs[:, 0]]].ravel()
                z_vals = zt[sample_z[pairs[:, 1]]].ravel()
                mask = np.isfinite(ref_vals) & np.isfinite(z_vals)
                if mask.sum() > 20:
                    slope, intercept = np.polyfit(z_vals[mask],
                                                  ref_vals[mask], 1)
                    models_data[model_key]["true"] = zt * slope + intercept
                    models_data[model_key]["pred"] = zp * slope + intercept
                    print(f"    slope={slope:.4f}  intercept={intercept:.4f}")

    # ── 5. Choose reference model for window indexing ────────────────────
    # Pick the model with the fewest windows (highest stride) as reference
    # so every reference window has a match in wider-strided models
    ref_for_windows = min(models_data, key=lambda k: models_data[k]["true"].shape[0])
    ref_true = models_data[ref_for_windows]["true"]
    n_ref = ref_true.shape[0]
    H = ref_true.shape[1]
    print(f"\nReference model for windowing: "
          f"{DISPLAY_NAMES.get(ref_for_windows, ref_for_windows)} "
          f"({n_ref} windows, H={H})")

    # ── 6. Build window-match index ──────────────────────────────────────
    # For each reference window, find the corresponding window in each model
    # via exact target match (tolerance 1e-3 for [0,1]-scale models).
    # For Z-score-rescaled models use correlation instead.
    print("Building cross-model window index...")

    # Filter to daylight windows (max true > 0.1)
    daylight = [i for i in range(n_ref)
                if np.nanmax(ref_true[i]) > 0.1]
    if len(daylight) == 0:
        daylight = list(range(n_ref))
    print(f"  {len(daylight)} daylight windows")

    # For speed, pre-compute match indices for all models
    match_index: dict[str, dict[int, int]] = {}  # model → {ref_idx: model_idx}
    for model_key, data in models_data.items():
        if model_key == ref_for_windows:
            match_index[model_key] = {i: i for i in range(n_ref)}
            continue
        m_true = data["true"]
        matches: dict[int, int] = {}
        # Try exact match first (fast)
        for ri in daylight:
            diffs = np.abs(m_true - ref_true[ri][None, :])
            exact = np.where(np.all(diffs < 1e-3, axis=1))[0]
            if len(exact) > 0:
                matches[ri] = int(exact[0])
        if len(matches) < min(10, len(daylight)):
            # Fall back to correlation-based match
            ref_sub = ref_true[daylight]
            r_std = ref_sub.std(axis=1, keepdims=True)
            r_std[r_std == 0] = 1.0
            r_norm = (ref_sub - ref_sub.mean(axis=1, keepdims=True)) / r_std
            m_std = m_true.std(axis=1, keepdims=True)
            m_std[m_std == 0] = 1.0
            m_norm = (m_true - m_true.mean(axis=1, keepdims=True)) / m_std
            corr = r_norm @ m_norm.T / H
            for j, ri in enumerate(daylight):
                best = int(np.argmax(corr[j]))
                if corr[j, best] > 0.999:
                    matches[ri] = best
        match_index[model_key] = matches
        disp = DISPLAY_NAMES.get(model_key, model_key)
        print(f"  {disp:25s}: {len(matches)}/{len(daylight)} windows matched")

    # ── 7. Pick windows that maximise model coverage ─────────────────────
    window_coverage = []
    for ri in daylight:
        n_matched = sum(1 for mi in match_index.values() if ri in mi)
        window_coverage.append((ri, n_matched))
    window_coverage.sort(key=lambda x: -x[1])

    # Among the top-coverage windows, pick a diverse random subset
    top_coverage = window_coverage[: max(len(window_coverage) // 4, args.num_plots * 3)]
    if len(top_coverage) > args.num_plots:
        chosen_indices = rng.choice(len(top_coverage), args.num_plots, replace=False)
        chosen = [top_coverage[i] for i in sorted(chosen_indices)]
    else:
        chosen = top_coverage[: args.num_plots]

    print(f"\nPlotting {len(chosen)} windows "
          f"(coverage range: {chosen[-1][1]}–{chosen[0][1]} models)")

    # ── 8. Assign colours ────────────────────────────────────────────────
    tier_counts: dict[int, int] = {}
    model_colors: dict[str, str] = {}
    sorted_models = sorted(models_data.keys(),
                           key=lambda k: (TIER_ORDER.get(k, 99), k))
    for mk in sorted_models:
        tier = TIER_ORDER.get(mk, 99)
        idx = tier_counts.get(tier, 0)
        palette = TIER_COLORS.get(tier, ["#000000"])
        model_colors[mk] = palette[idx % len(palette)]
        tier_counts[tier] = idx + 1

    # ── 9. Plot ──────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for plot_num, (ref_idx, coverage) in enumerate(chosen):
        target = ref_true[ref_idx]
        steps = np.arange(1, H + 1)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Ground truth
        ax.plot(steps, target, "k--", label="Ground Truth",
                linewidth=2.5, zorder=10)

        # Model predictions in tier order
        plotted = []
        for mk in sorted_models:
            mi = match_index.get(mk, {})
            if ref_idx not in mi:
                continue
            m_idx = mi[ref_idx]
            pred = models_data[mk]["pred"][m_idx]
            disp = DISPLAY_NAMES.get(mk, mk)
            ax.plot(steps, pred, label=disp,
                    color=model_colors[mk], linewidth=1.6, alpha=0.85)
            plotted.append(mk)

        ax.set_title(f"Forecast Comparison — Site {selected_site}, "
                     f"Window {ref_idx}  ({len(plotted)} models)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Forecast Step (half-hourly)")
        ax.set_ylabel("Normalised Power")
        ax.set_xlim(0.5, H + 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=7, ncol=2,
                  framealpha=0.9)

        plot_path = out_dir / f"site_{selected_site}_window_{ref_idx}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ [{plot_num+1}/{len(chosen)}] "
              f"{plot_path.name}  ({len(plotted)} models)")

    print(f"\n✓ Done — {len(chosen)} plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
