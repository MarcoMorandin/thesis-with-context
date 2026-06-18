"""PVTSFM adaptation: zero-shot VisionTS++ over the exported uk_pv CSVs.

Added (not upstream) so VisionTS++ consumes our dataset with no further edits at
run time. Reuses tier4/vendor/export_ukpv.py output (`uk_pv_test_<site>.csv`,
`date`+`OT`). Builds non-overlapping (T+H) windows, forecasts H from T zero-shot
via the continual-pretrained MAE, and dumps per-plant predictions in our
baseline-contract format (N, H) for scripts/contract_check.py + the metric import.

    python run_ukpv.py --csv_dir <ukpv_dir> --ckpt_path <mae.ckpt> \
        --context_len 24 --pred_len 12 --periodicity 48 --out results_ukpv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# vendored package path (scripts/VisionTS) so `import visionts` works in-place
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "VisionTS"))
from visionts import VisionTS  # noqa: E402


def windows(series: np.ndarray, ctx: int, pred: int):
    """Non-overlapping (ctx history, pred future) windows."""
    xs, ys = [], []
    step = pred
    for start in range(0, len(series) - ctx - pred + 1, step):
        xs.append(series[start : start + ctx])
        ys.append(series[start + ctx : start + ctx + pred])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--ckpt_path", required=True, help="VisionTS++ MAE checkpoint")
    ap.add_argument("--arch", default="mae_base")
    ap.add_argument("--context_len", type=int, default=24)
    ap.add_argument("--pred_len", type=int, default=12)
    ap.add_argument("--periodicity", type=int, default=48)  # 30-min daily
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default="results_ukpv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The released Lefei/VisionTSpp ckpt is a *quantile* VisionTS++ (decoder_pred
    # + 8 decoder_pred_quantile_list heads), so build the model with quantile=True
    # (quantile_head_num defaults to 9 = 1 mean + 8 quantiles) or the extra keys
    # fail the strict state_dict load. Inference takes the mean/point head only.
    model = VisionTS(arch=args.arch, ckpt_path=args.ckpt_path, load_ckpt=True,
                     quantile=True).to(device)
    model.update_config(context_len=args.context_len, pred_len=args.pred_len,
                        periodicity=args.periodicity)
    model.eval()
    os.makedirs(args.out, exist_ok=True)

    for csv in sorted(glob.glob(os.path.join(args.csv_dir, "uk_pv_test_*.csv"))):
        site = os.path.basename(csv)[len("uk_pv_test_"):-len(".csv")]
        ot = pd.read_csv(csv)["OT"].to_numpy(np.float32)
        x, y = windows(ot, args.context_len, args.pred_len)
        if not len(x):
            print(f"skip {site}: too short"); continue
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), args.batch_size):
                xb = torch.from_numpy(x[i : i + args.batch_size]).unsqueeze(-1).to(device)
                out = model(xb)                       # [b, pred_len, 1]
                if isinstance(out, (list, tuple)):    # quantile mode → [mean, quantiles]
                    out = out[0]
                preds.append(out.squeeze(-1).cpu().numpy())
        pred = np.clip(np.concatenate(preds), 0.0, 1.0).astype(np.float32)
        np.savez(os.path.join(args.out, f"visionts_pp_{site}_pred.npz"),
                 pred=pred, true=y)
        print(f"{site}: pred {pred.shape}")
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
