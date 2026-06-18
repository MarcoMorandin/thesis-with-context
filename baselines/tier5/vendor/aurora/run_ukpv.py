"""Aurora (decisionintelligence/Aurora) ZERO-SHOT on uk_pv.

Aurora is a multimodal TS foundation model with a zero-shot `generate()` API
(NOT training-only — runner.py/train_from_scratch.py are the *training* path the
old slurm wrongly invoked). We use the unimodal TS path: feed each plant's power
history `[B, ctx]`, sample forecasts, average. Aurora instance-normalizes inputs
internally, so the output is already in the input (norm_power) scale.

Mirrors the upstream TFB wrapper (ts_benchmark/baselines/aurora/aurora.py):
    model = AuroraForPrediction.from_pretrained(ckpt)
    out = model.generate(inputs=[B,L], max_output_length=H,
                         inference_token_len=48, num_samples=100)  # [B, S, H]
    forecast = out.mean(samples)
Dumps per-plant aurora_<site>_pred.npz for scripts/import_predictions.py.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from aurora.modeling_aurora import AuroraForPrediction


def windows(x: np.ndarray, ctx: int, pred: int):
    n = len(x) - ctx - pred + 1
    if n <= 0:
        return np.empty((0, ctx), np.float32), np.empty((0, pred), np.float32)
    X = np.stack([x[i:i + ctx] for i in range(n)]).astype(np.float32)
    Y = np.stack([x[i + ctx:i + ctx + pred] for i in range(n)]).astype(np.float32)
    return X, Y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--ckpt_path", required=True, help="DecisionIntelligence/Aurora HF dir")
    ap.add_argument("--context_len", type=int, default=24)
    ap.add_argument("--pred_len", type=int, default=12)
    ap.add_argument("--inference_token_len", type=int, default=48)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default="results_ukpv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AuroraForPrediction.from_pretrained(args.ckpt_path).to(device).eval()
    os.makedirs(args.out, exist_ok=True)

    for csv in sorted(glob.glob(os.path.join(args.csv_dir, "uk_pv_test_*.csv"))):
        if "_retrieve_" in csv:                      # skip RAG retrieval intermediates
            continue
        site = os.path.basename(csv)[len("uk_pv_test_"):-len(".csv")]
        ot = pd.read_csv(csv)["OT"].to_numpy(np.float32)
        X, Y = windows(ot, args.context_len, args.pred_len)
        if not len(X):
            print(f"skip {site}: too short")
            continue
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), args.batch_size):
                xb = torch.from_numpy(X[i:i + args.batch_size]).to(device)   # [B, ctx]
                out = model.generate(inputs=xb, max_output_length=args.pred_len,
                                     inference_token_len=args.inference_token_len,
                                     num_samples=args.num_samples)            # [B, S, H]
                out = out.float().mean(dim=1)                                 # [B, H]
                preds.append(out.cpu().numpy())
        pred = np.clip(np.concatenate(preds), 0.0, 1.0).astype(np.float32)
        np.savez(os.path.join(args.out, f"aurora_{site}_pred.npz"), pred=pred, true=Y)
        print(f"{site}: pred {pred.shape}")
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
