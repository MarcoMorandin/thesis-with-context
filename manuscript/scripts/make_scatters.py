"""Predicted-versus-observed diagnostic panels (manuscript fig:baseline-scatters).

One PNG per model, daylight-only, all fourteen held-out UK plants pooled.
Regenerate with::

    uv run --no-project --with numpy --with matplotlib \
        python manuscript/scripts/make_scatters.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import diagnostics_common as dc  # noqa: E402

plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "axes.linewidth": 0.7})

# figure stem -> (npz stem, panel title)
PANELS = {
    "scatter_chronos2_zs": ("chronos2_oracle", "Chronos-2 zero-shot"),
    "scatter_itransformer": (
        "itransformer_nf_histcov_s2_ukpv_seed42",
        "iTransformer with covariates",
    ),
    "scatter_lightgbm": ("lightgbm", "LightGBM"),
    "scatter_aurora_ft": ("aurora_ft", "Aurora fine-tuned"),
    "scatter_solar_vlm": ("solar_vlm", "Solar-VLM"),
    "scatter_time_vlm": ("time_vlm", "Time-VLM"),
    "scatter_visionts_pp": ("visionts_pp", "VisionTS++"),
    "scatter_mmtsfm_s1": ("mmtsfm_s1_ukpv_selfattn_s42", "Chronos-2 fine-tuned"),
    "scatter_mmtsfm_s2a": ("mmtsfm_s2a_ukpv_selfattn_s42", "Pooled-image baseline"),
    "scatter_mmtsfm_s2d": ("mmtsfm_s2d_ukpv_s2d_s42", "Spatial-token fusion"),
    "scatter_mmtsfm_s2e": ("mmtsfm_s2e_ukpv_s2e_s42", "Multi-anchor fusion"),
}

LIM = 1.05


def panel(ax, pred, true, title):
    ax.hexbin(
        true,
        pred,
        gridsize=70,
        bins="log",
        cmap="magma_r",
        extent=(0, LIM, 0, LIM),
        linewidths=0,
        mincnt=1,
    )
    ax.plot([0, LIM], [0, LIM], color="#1b6ca8", lw=1.0, ls="--", zorder=3)
    ax.set_xlim(0, LIM)
    ax.set_ylim(0, LIM)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("observed normalized power", fontsize=8)
    ax.set_ylabel("predicted normalized power", fontsize=8)
    ax.tick_params(labelsize=7)


def load_scatter_data(stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Load plotted predictions, using model-specific grids for these runs."""
    if stem not in {"solar_vlm", "time_vlm"}:
        return dc.load_all(stem)

    predictions, targets = [], []
    for plant in dc.plants():
        path = dc.PRED_DIR / f"{stem}_{plant}_pred.npz"
        with np.load(path) as saved:
            predictions.append(saved["pred"].reshape(-1))
            targets.append(saved["true"].reshape(-1))
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> int:
    missing = []
    for out, (stem, title) in PANELS.items():
        if not dc.available(stem):
            missing.append((out, stem, "no predictions"))
            continue
        try:
            pred, true = load_scatter_data(stem)
        except ValueError as exc:
            missing.append((out, stem, str(exc).split(";")[0]))
            continue
        try:
            stats = dc.scores(stem)
        except ValueError:
            stats = None
        fig, ax = plt.subplots(figsize=(3.1, 3.1))
        panel(ax, np.clip(pred, 0, LIM), np.clip(true, 0, LIM), title)
        fig.tight_layout(pad=0.4)
        fig.savefig(dc.FIG_DIR / f"{out}.png", bbox_inches="tight")
        plt.close(fig)
        if stats is None:
            print(f"wrote {out}.png")
        else:
            print(f"wrote {out}.png  NMAE={stats['nmae']:.4f} SS={stats['skill']:.2f}")

    for out, stem, why in missing:
        print(f"SKIPPED {out}: {stem} -> {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
