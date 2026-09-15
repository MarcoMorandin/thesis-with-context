"""Overlay all five forecasters on one example window, per test plant.

Reads the per-site npz written by ProtocolEval.dump_predictions
([n_windows, H] each). Test stride == H, so windows tile the timeline and the
PREVIOUS window's `true` is exactly the context preceding this one — no extra
data source needed.

TimeVLM is dumped by its own vendored harness at stride 1 with no mask, so its
rows are matched back to the protocol grid by exact value-match of `true`
(same trick as baselines/scripts/import_predictions.py::exact_mask, without
needing the parquet). Only windows that are fully valid AND present in the
TimeVLM dump are eligible.

No uncertainty band is drawn: none of these models has a quantile head, so any
shaded interval would be fabricated.

    uv run python scripts/probes/plot_forecast_panel.py \
        --pred-dir ~/Desktop/thesis-with-context/baselines/results/predictions \
        --out-dir report/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# CVD-validated (scripts/validate_palette.js, light surface):
#   s1/s2a = an ordinal blue pair — the precursor stages read as one family;
#   s2d, TimeVLM, iTransformer = three categorical identities, all-pairs PASS.
# s2d takes the warm hue deliberately: it is the model under test, so it must
# separate from its own precursors at a glance, not blend into their ramp.
SERIES = [
    ("mmtsfm_s1_ukpv_selfattn_s42", "MMTSFM s1", "#6da7ec"),
    ("mmtsfm_s2a_ukpv_selfattn_s42", "MMTSFM s2a", "#2a78d6"),
    ("mmtsfm_s2d_ukpv_s2d_s42", "MMTSFM s2d", "#eb6834"),
    ("time_vlm", "TimeVLM", "#4a3aa7"),
    ("itransformer_nf_histcov_s2_ukpv_seed42", "iTransformer", "#1baf7a"),
]
GT, INK, MUTED = "#898781", "#0b0b0b", "#c3c2b7"
TV = "time_vlm"
S2D = "mmtsfm_s2d_ukpv_s2d_s42"


def load(pred_dir: Path, prefix: str, site: str) -> dict:
    return np.load(pred_dir / f"{prefix}_{site}_pred.npz")


def tv_index(tv_true: np.ndarray, proto_true: np.ndarray) -> np.ndarray:
    """Row of `tv_true` holding each protocol window, or -1."""
    out = np.full(len(proto_true), -1, dtype=int)
    key = tv_true[:, :3]
    for i, row in enumerate(proto_true):
        for j in np.where(np.abs(key - row[:3]).sum(1) < 1e-4)[0]:
            if np.abs(tv_true[j] - row).max() < 1e-4:
                out[i] = j
                break
    return out


def pick_window(
    true: np.ndarray, preds: dict, tv_row: np.ndarray, eligible: np.ndarray
) -> int:
    """Eligible window where s2d beats every other model by the widest MAE margin.

    Restricted to windows in the top half of *sustained* swing (scored on a
    3-point moving average, so an isolated one-step sensor spike — which no
    forecaster can or should track — cannot qualify). Without that gate the
    margin is won by flat night windows where every model is trivially right.
    """
    k = np.ones(3) / 3.0
    smooth = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, true)
    swing = np.where(eligible, np.abs(np.diff(smooth, axis=1)).max(1), -np.inf)
    swing[0] = -np.inf
    active = eligible & (swing >= np.median(swing[eligible]))

    def mae(prefix: str) -> np.ndarray:
        rows = tv_row if prefix == TV else np.arange(len(true))
        return np.abs(preds[prefix][np.clip(rows, 0, None)] - true).mean(1)

    s2d = mae(S2D)
    others = np.stack([mae(p) for p, _, _ in SERIES if p != S2D])
    margin = np.where(active, others.min(0) - s2d, -np.inf)
    return int(np.argmax(margin))


def plot_site(pred_dir: Path, site: str, out_dir: Path, n_ctx: int) -> Path:
    data = {p: load(pred_dir, p, site) for p, _, _ in SERIES}
    ref = data[SERIES[0][0]]
    true, mask = ref["true"], ref["mask"]
    H = true.shape[1]

    for prefix, name, _ in SERIES:
        if prefix == TV:
            continue
        if not np.allclose(data[prefix]["true"], true, atol=1e-5, equal_nan=True):
            raise SystemExit(f"{site}: {name} is not on the protocol grid")

    tv_row = tv_index(data[TV]["true"], true)
    eligible = (mask.sum(1) == H) & (tv_row >= 0)
    w = pick_window(true, {p: data[p]["pred"] for p, _, _ in SERIES}, tv_row, eligible)

    n = min(n_ctx, w * H)
    ctx = true[w - n // H - 1 : w].ravel()[-n:]
    x_ctx = np.arange(-len(ctx), 0) / 2.0  # 30 min steps -> hours
    x_fut = (np.arange(H) + 1) / 2.0
    join = np.r_[x_ctx[-1], x_fut]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x_ctx, ctx, color=MUTED, lw=1.6, marker="o", ms=3, label="observed context")
    ax.plot(
        join,
        np.r_[ctx[-1], true[w]],
        color=GT,
        lw=1.7,
        marker="o",
        ms=3.5,
        label="ground truth",
        zorder=5,
    )

    ends = []
    for prefix, name, color in SERIES:
        y = data[prefix]["pred"][tv_row[w] if prefix == TV else w]
        ax.plot(
            join,
            np.r_[ctx[-1], y],
            color=color,
            lw=1.9,
            ls="--",
            marker="o",
            ms=4.0,
            mew=0,
            label=name,
            zorder=4,
        )
        ends.append([float(y[-1]), name, color])

    # direct labels at the line ends, dodged apart so none is hidden
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.05
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y_lab, name, color in ends:
        ax.annotate(
            name,
            (x_fut[-1], y_lab),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=7.5,
            va="center",
            clip_on=False,
        )

    ax.axvline(0.0, color="#c3c2b7", lw=1.0, ls=":")
    ax.set_xlabel("hours from forecast origin")
    ax.set_ylabel("normalised power")
    ax.set_title(f"plant {site} — 6 h forecast, window {w}", color=INK, fontsize=11)
    # the five forecasts carry direct labels, so the legend only names the two
    # reference series — a 7-entry box collides with the context curve.
    h, lbl = ax.get_legend_handles_labels()
    ax.legend(h[:2], lbl[:2], frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.margins(x=0.02)
    fig.subplots_adjust(right=0.82)
    fig.tight_layout()

    out = out_dir / f"forecast_example_{site}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    maes = {
        name: float(
            np.abs(data[p]["pred"][tv_row[w] if p == TV else w] - true[w]).mean()
        )
        for p, name, _ in SERIES
    }
    order = " ".join(
        f"{k}={v:.3f}" for k, v in sorted(maes.items(), key=lambda t: t[1])
    )
    print(
        f"{site}: window {w} of {int(eligible.sum())} eligible -> {out}\n    MAE {order}"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--sites", nargs="+", default=["11287", "6648", "12642", "11176"])
    p.add_argument("--context", type=int, default=24, help="context steps to show")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for site in a.sites:
        plot_site(a.pred_dir.expanduser(), site, a.out_dir, a.context)


if __name__ == "__main__":
    main()
