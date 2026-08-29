"""Plot one baseline's inference: context, ground truth, prediction.

Reads the per-site npz written by ``eval.protocol_eval.ProtocolEvaluator.
dump_predictions`` ([n_windows, H] each) — the same evaluator every baseline
and MMTSFM itself write through, so this works for any tier's dump, not just
one model. Test stride == H, so windows tile the timeline and the PREVIOUS
window's `true` is exactly the context preceding this one — no extra data
source needed. Mirrors MMTSFM/scripts/probes/plot_example_forecast.py.

    uv run python scripts/plot_example_forecast.py \
        --npz results/predictions/itransformer_nf_s2_ukpv_seed42_10793_pred.npz \
        --label "iTransformer (library, MMTSFM protocol)" \
        --window ramp --out example_forecast.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--npz", required=True)
p.add_argument(
    "--window",
    default="ramp",
    help="index, or 'ramp' for the largest within-window swing",
)
p.add_argument("--context", type=int, default=24, help="context steps to show")
p.add_argument(
    "--label", default="forecast", help="legend/series name for the prediction line"
)
p.add_argument(
    "--step-minutes",
    type=int,
    default=30,
    help="cadence for the x-axis label (uk_pv=30, goes_pvdaq=15)",
)
p.add_argument("--out", default="example_forecast.png")
a = p.parse_args()

d = np.load(a.npz)
pred, true, mask = d["pred"], d["true"], d["mask"]
H = true.shape[1]

if a.window == "ramp":
    swing = np.where(mask.sum(1) == H, np.abs(np.diff(true, axis=1)).max(1), -np.inf)
    w = int(np.argmax(swing))
else:
    w = int(a.window)
if w == 0:
    raise SystemExit("window 0 has no preceding window to draw context from")

n_ctx = min(a.context, w * H)
ctx = true[w - n_ctx // H - 1 : w].ravel()[-n_ctx:]  # tiling windows -> flat history
x_ctx = np.arange(-len(ctx), 0)
x_fut = np.arange(H)
m = mask[w] > 0

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x_ctx, ctx, color="0.45", lw=1.4, marker="o", ms=3, label="context (observed)")
ax.plot(
    np.r_[x_ctx[-1], x_fut],
    np.r_[ctx[-1], true[w]],
    color="#1a7f37",
    lw=2,
    marker="o",
    ms=4,
    label="ground truth",
)
ax.plot(
    np.r_[x_ctx[-1], x_fut[m]],
    np.r_[ctx[-1], pred[w][m]],
    color="#1f6feb",
    lw=2,
    ls="--",
    marker="s",
    ms=4,
    label=a.label,
)
ax.axvline(-0.5, color="0.7", lw=1, ls=":")
ax.text(-0.4, ax.get_ylim()[1], " forecast origin", va="top", fontsize=8, color="0.4")

ax.set_xlabel(f"steps from forecast origin ({a.step_minutes} min each)")
ax.set_ylabel("normalised power")
ax.set_title(f"window {w} — max |Δy| in horizon = {np.abs(np.diff(true[w])).max():.3f}")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(a.out, dpi=160)
print(f"window {w}  ->  {a.out}")
