"""Trajectory diagnostics (manuscript fig:tier-traces, fig:worked-trace).

Visual language follows MMTSFM/scripts/probes/plot_forecast_panel.py: muted
observed context, dark ground truth, dashed forecasts with direct end labels.

Two deliberate differences from that script.

History source. ``plot_forecast_panel`` takes the context from the *previous*
window's ``true``, assuming evaluation windows tile the timeline. They do not:
only about half of them have a contiguous predecessor (start gaps of 12, 36, 48
and 84 steps all occur), so that context is silently displaced whenever a gap
precedes the window. Here the history is rebuilt from the curated parquet
through the evaluation window spec, which is correct for any gap, and
``dc.check_alignment`` asserts the rebuild matches the stored predictions
before anything is drawn.

Window choice. Neither figure uses neutral selection, and the manuscript says so.
``margin_ranked`` picks the tier windows: ramp windows of one plant ordered by
how far S2d's MAE falls below the better of the numerical and pooled-image arms.
The worked example is fixed at plant 11287, window 1021. It shows a clear
upward ramp and was originally selected for a large S2d advantage. Keeping
that window fixed lets the added multi-anchor curve be compared on the same
event. The aggregate tables carry the quantitative comparison.

Night is not recorded, and every fully-daylit window has an 08:00 UTC origin,
so the sixteen hours before the origin are empty. Rather than draw that as a
flat plateau, the x-axis is broken: the left panel holds the previous day's
daylit context, the right panel the horizon. Regenerate with::

    uv run --no-project --with numpy --with matplotlib --with pandas \
        --with pyarrow python manuscript/scripts/make_traces.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import diagnostics_common as dc  # noqa: E402

plt.rcParams.update({"figure.dpi": 200, "font.size": 9, "axes.linewidth": 0.7})

PLANT = "7648"  # tier figures: largest relative S2d gain over S1 (11.5%),
# and the highest per-window win rate among the fourteen held-out plants.
STEP_MINUTES = 30
CONTEXT_STEPS = 48  # 24 h: every fully-daylit window has an 08:00 UTC origin,
# so a 12 h context would be entirely night and carry no information.

GT, INK, MUTED = "#898781", "#0b0b0b", "#c3c2b7"

# Palette from plot_forecast_panel.py (CVD-validated, light surface).
S1 = "mmtsfm_s1_ukpv_selfattn_s42"
S2A = "mmtsfm_s2a_ukpv_selfattn_s42"
S2D = "mmtsfm_s2d_ukpv_s2d_s42"
S2E = "mmtsfm_s2e_ukpv_s2e_s42"

BENCHMARK = [
    (S1, "Chronos-2 fine-tuned", "#2a78d6"),
    ("itransformer_nf_histcov_s2_ukpv_seed42", "iTransformer", "#1baf7a"),
    ("lightgbm", "LightGBM", "#a968bd"),
    ("aurora_ft", "Aurora fine-tuned", "#eb6834"),
]

MULTIMODAL = [
    (S2A, "Pooled-image baseline", "#2a78d6"),
    (S2D, "Spatial-token fusion", "#eb6834"),
    (S2E, "Multi-anchor", "#1baf7a"),
]

# Preserve the original window selection after changing the displayed models.
WINDOW_SELECTION = [BENCHMARK[0], MULTIMODAL[0], MULTIMODAL[1]]

WORKED = [
    (S1, "Chronos-2 fine-tuned", "#6da7ec"),
    (S2A, "Pooled-image baseline", "#2a78d6"),
    (S2D, "Spatial-token fusion", "#eb6834"),
    (S2E, "Multi-anchor", "#c0448d"),
    ("itransformer_nf_histcov_s2_ukpv_seed42", "iTransformer", "#1baf7a"),
    ("chronos2_oracle", "Chronos-2 zero-shot", "#4a3aa7"),
]

WORKED_PLANT = "11287"
WORKED_WINDOW = 1021

N_WINDOWS = 3


def swing_ranked(plant: str) -> list[int]:
    """Fully-daylit window indices, ordered by decreasing end-to-end swing."""
    _, true, mask = dc.load_plant(dc.MASK_DONOR, plant)
    idx = np.flatnonzero(mask.all(axis=1))
    swing = np.abs(true[idx, -1] - true[idx, 0])
    return list(idx[np.argsort(-swing)])


def margin_ranked(plant: str, models) -> list[int]:
    """Ramp windows ordered by how far S2d's MAE falls below the best rival.

    Restricted to ramp windows (|Δ| over the horizon above ``RAMP_THRESHOLD``)
    so the panels show the regime the ramp metric scores, but the ordering is
    still by S2d's own advantage. That is a selected view: across the whole
    record S2d wins only about half of individual ramp windows, and its real
    advantage is a small consistent shift that shows up in the per-plant
    aggregate (lowest NMAE on all fourteen plants), not in any one window.
    The manuscript must disclose this selection.
    """
    s2d, true, mask = dc.load_plant(S2D, plant)
    swing = np.abs(true[:, -1] - true[:, 0])
    idx = np.flatnonzero(mask.all(axis=1) & (swing > dc.RAMP_THRESHOLD))
    others = [s for s, _, _ in models if s != S2D]
    field = np.stack([dc.load_plant(s, plant)[0] for s in others])
    margin = np.abs(field - true[None]).mean(2).min(0) - np.abs(s2d - true).mean(1)
    return list(idx[np.argsort(-margin[idx])])


def best_s2d_window(models) -> tuple[str, int, float, float, float]:
    """Scan every plant for the window where S2d's lead over the field is widest.

    Margin is ``min_over_competitors(MAE) - MAE(S2d)`` on fully-daylit windows,
    with smart persistence included in the field. This is deliberately a
    best-case search; the caller is responsible for saying so in the caption.
    """
    others = [s for s, _, _ in models if s != S2D] + [dc.REFERENCE]
    best = ("", -1, -np.inf, 0.0, 0.0)
    for plant in dc.plants():
        s2d, true, mask = dc.load_plant(S2D, plant)
        keep = np.flatnonzero(mask.all(axis=1))
        if not len(keep):
            continue
        field = np.stack([dc.load_plant(s, plant)[0] for s in others])
        mae_s2d = np.abs(s2d - true).mean(1)
        margin = np.abs(field - true[None]).mean(2).min(0) - mae_s2d
        w = keep[np.argmax(margin[keep])]
        if margin[w] > best[2]:
            best = (
                plant,
                int(w),
                float(margin[w]),
                float(mae_s2d[w]),
                float(mae_s2d[w] + margin[w]),
            )
    return best


def trace_pair(axl, axr, models, plant, window, title, direct_labels: bool):
    """Draw one window across a broken x-axis: daylit context | forecast.

    The origin is 08:00 UTC for every fully-daylit window, so the sixteen hours
    immediately before it are night: not recorded, and worth no horizontal
    space. ``axl`` therefore spans only the previous day's daylit block and
    ``axr`` the horizon, with the break marked between them. Nothing is
    interpolated across the break, and no line joins the context to the origin,
    because no observation exists at the origin itself.
    """
    step_h = STEP_MINUTES / 60.0
    ctx = dc.history(plant, window, CONTEXT_STEPS)
    valid = np.flatnonzero(~np.isnan(ctx))
    lo, hi = valid[0], valid[-1]
    x_ctx = (np.arange(-len(ctx), 0) * step_h)[lo : hi + 1]
    y_ctx = ctx[lo : hi + 1]
    x_fut = (np.arange(12) + 1) * step_h

    _, true, _ = dc.load_plant(dc.MASK_DONOR, plant)
    axl.plot(
        x_ctx, y_ctx, color=MUTED, lw=1.6, marker="o", ms=3, label="observed context"
    )
    axr.plot(
        x_fut,
        true[window],
        color=GT,
        lw=1.7,
        marker="o",
        ms=3.5,
        label="ground truth",
        zorder=5,
    )
    # Keep the legend complete on the left axis without drawing anything there.
    axl.plot([], [], color=GT, lw=1.7, marker="o", ms=3.5, label="ground truth")

    ends = []
    for stem, name, color in models:
        y = dc.load_plant(stem, plant)[0][window]
        axr.plot(
            x_fut, y, color=color, lw=1.9, ls="--", marker="o", ms=4.0, mew=0, zorder=4
        )
        axl.plot([], [], color=color, lw=1.9, ls="--", marker="o", ms=4.0, label=name)
        ends.append([float(y[-1]), name, color])

    axl.set_xlim(x_ctx[0] - 0.4, x_ctx[-1] + 0.4)
    axr.set_xlim(0.0, x_fut[-1] + 0.3)
    axr.axvline(0.0, color=MUTED, lw=1.0, ls=":")

    ymax = max(np.nanmax(y_ctx), true[window].max(), max(e[0] for e in ends))
    for ax in (axl, axr):
        ax.set_ylim(-0.03, ymax * 1.18)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7.5)
    axl.set_ylabel("normalised power", fontsize=8)
    axl.spines[["top", "right"]].set_visible(False)
    axr.spines[["top", "left", "right"]].set_visible(False)
    axr.tick_params(left=False, labelleft=False)

    # Break marks on the facing edges.
    kw = dict(
        marker=[(-1, -0.6), (1, 0.6)],
        ms=5,
        ls="none",
        color=INK,
        mec=INK,
        mew=0.8,
        clip_on=False,
    )
    axl.plot([1, 1], [0, 1], transform=axl.transAxes, **kw)
    axr.plot([0, 0], [0, 1], transform=axr.transAxes, **kw)

    axl.set_title(title, color=INK, fontsize=9.5, loc="left")

    if direct_labels:
        y0, y1 = axr.get_ylim()
        gap = (y1 - y0) * 0.05
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y_lab, name, color in ends:
            axr.annotate(
                name,
                (x_fut[-1], y_lab),
                xytext=(6, 0),
                textcoords="offset points",
                color=color,
                fontsize=7.5,
                va="center",
                clip_on=False,
            )
    return ends


def broken_axes(fig, gs_cell):
    """Two axes sharing a row, with a narrow gap for the break."""
    sub = gs_cell.subgridspec(1, 2, width_ratios=(1.35, 1.0), wspace=0.06)
    return fig.add_subplot(sub[0]), fig.add_subplot(sub[1])


def tier_figure(models, out, suptitle, windows):
    fig = plt.figure(figsize=(5.4, 6.8))
    gs = fig.add_gridspec(N_WINDOWS, 1, hspace=0.55, top=0.86, bottom=0.08)
    pairs = []
    for row, window in enumerate(windows):
        axl, axr = broken_axes(fig, gs[row])
        trace_pair(
            axl, axr, models, PLANT, window, f"plant {PLANT}, window {window}", False
        )
        pairs.append((axl, axr))
    for axl, axr in pairs[:-1]:
        axl.tick_params(labelbottom=True)
    fig.supxlabel("hours from forecast origin", fontsize=8, y=0.015)
    handles, labels = pairs[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.5,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle(suptitle, fontsize=10.5, color=INK, y=0.985)
    fig.savefig(dc.FIG_DIR / f"{out}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png  plant={PLANT} windows={windows}")
    return windows


def worked_figure(plant, window):
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(1, 1, top=0.88, bottom=0.13, left=0.08, right=0.77)
    axl, axr = broken_axes(fig, gs[0])
    trace_pair(
        axl,
        axr,
        WORKED,
        plant,
        window,
        f"plant {plant} — 6 h forecast, window {window}",
        True,
    )
    fig.supxlabel("hours from forecast origin", fontsize=8, y=0.02)
    h, lbl = axl.get_legend_handles_labels()
    axl.legend(h[:2], lbl[:2], frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    out = f"traces_comparison_{plant}_w{window}"
    fig.savefig(dc.FIG_DIR / f"{out}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")
    return out


def us_figure(pred_dir=dc.PRED_DIR, output=dc.FIG_DIR / "traces_us.png"):
    """Plot seed-42 US horizons, selected by observed swing, never model error.

    Use the first three lexicographically sorted eligible plant IDs and their
    largest fully valid end-to-end swing. History is omitted because NPZ exports
    contain only the horizon. No observations are inferred across window gaps.
    """
    models = [
        ("mmtsfm_s1_goespvdaq_selfattn_all_s42", "Numerical control", "#6da7ec"),
        ("mmtsfm_s2a_goespvdaq_selfattn_all_s42", "Pooled-image fusion", "#2a78d6"),
        ("mmtsfm_s2d_goespvdaq_s2d_all_s42", "Spatial-token fusion", "#eb6834"),
        ("mmtsfm_s2e_goespvdaq_s2e_all_s42", "Multi-anchor", "#c0448d"),
    ]
    sites = sorted(p.name.removeprefix(models[0][0] + "_").removesuffix("_pred.npz")
                   for p in pred_dir.glob(f"{models[0][0]}_*_pred.npz"))[:3]
    if len(sites) != 3:
        raise ValueError("US traces require at least three plant exports")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5), sharex=True)
    selected = []
    try:
        for ax, plant in zip(axes, sites):
            arrays = []
            for stem, _, _ in models:
                with np.load(pred_dir / f"{stem}_{plant}_pred.npz") as z:
                    arrays.append((z["pred"], z["true"], z["mask"].astype(bool)))
            _, true, mask = arrays[0]
            for pred, target, valid in arrays:
                if (pred.shape != true.shape or target.shape != true.shape
                        or mask.shape != true.shape or not np.array_equal(valid, mask)
                        or not np.allclose(target, true, atol=1e-6, rtol=0)):
                    raise ValueError(f"unaligned US exports for plant {plant}")
            if true.ndim != 2 or true.shape[1] != 12:
                raise ValueError(f"unaligned US horizon for plant {plant}")
            candidates = np.flatnonzero(mask.all(axis=1))
            if not len(candidates):
                raise ValueError(f"no fully valid US horizon for plant {plant}")
            window = int(candidates[np.argmax(np.abs(
                true[candidates, -1] - true[candidates, 0]))])
            selected.append((plant, window))
            hours = (np.arange(12) + 1) / 2
            ax.plot(hours, true[window], color=GT, lw=1.7, marker="o", ms=3,
                    label="Observed power", zorder=5)
            for (pred, _, _), (_, label, color) in zip(arrays, models):
                ax.plot(hours, pred[window], color=color, lw=1.7, ls="--",
                        marker="o", ms=3, label=label)
            ax.set_title(f"US plant {plant}, window {window}", loc="left", fontsize=9)
            ax.set_ylabel("Normalised power", fontsize=8)
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, 6.2)
        axes[-1].set_xlabel("Hours from forecast origin")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, 1))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(output, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"wrote {output}: {selected}")
    return selected


def main() -> int:
    us_figure()
    for group in (BENCHMARK, MULTIMODAL, WORKED):
        for stem, label, _ in group:
            if not dc.available(stem):
                print(f"missing predictions: {label} ({stem})", file=sys.stderr)
                return 1
    dc.check_alignment(PLANT)
    windows = margin_ranked(PLANT, WINDOW_SELECTION)[:N_WINDOWS]
    tier_figure(BENCHMARK, "traces_tsfm", "Benchmark models", windows)
    tier_figure(MULTIMODAL, "traces_multimodal", "Visual fusion arms", windows)

    dc.check_alignment(WORKED_PLANT)
    worked_figure(WORKED_PLANT, WORKED_WINDOW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
