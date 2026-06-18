"""Solar-VLM on our uk_pv multimodal track — TRAIN (train-split plant groups)
then EVAL (unseen test-split plant groups) → cross-plant zero-shot.

This is the in-suite entrypoint (the original run_pv.py targets the Hebei
8-station CSV layout). It drives the *unchanged* Solar-VLM model/Experiment with
``--data UKPV`` (Dataset_UKPV) and dumps per-plant ``solar_vlm_<site>_pred.npz``
(norm_power, horizon H) that scripts/import_predictions.py folds into our
NMAE/NRMSE/SS metrics, exactly like crossvivit/sunset.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))          # baselines/

from common import config as cfg                                  # noqa: E402
from data_provider.data_loader_ukpv import Dataset_UKPV           # noqa: E402
from exp.experiment import Experiment                             # noqa: E402

F_DIM = len(cfg.COV_COLS) + 1                      # covariates + norm_power


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solar-VLM on uk_pv (cross-plant)")
    p.add_argument("--data", default="UKPV")
    p.add_argument("--data_path", default=os.environ.get("DATA", cfg.DEFAULT_DATA_PATH))
    p.add_argument("--out", default="results_ukpv")
    p.add_argument("--num_stations", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=cfg.HISTORY_STEPS)
    p.add_argument("--label_len", type=int, default=cfg.HORIZON_STEPS)
    p.add_argument("--pred_len", type=int, default=cfg.HORIZON_STEPS)
    p.add_argument("--vision_feat_dir", required=True)
    p.add_argument("--qwen3_vl_model_path", default="")
    p.add_argument("--vlm_embed_dim", type=int, default=2048)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--image_freq_minutes", type=int, default=30)  # uk_pv cadence
    p.add_argument("--train_epochs", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--multimodal_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--gnn_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--disable_visual", type=int, default=0)
    p.add_argument("--disable_text", type=int, default=0)
    p.add_argument("--disable_gnn", type=int, default=0)
    p.add_argument("--disable_cross_site_attn", type=int, default=0)
    a = p.parse_args()

    # Fill the full arg surface the unchanged Experiment/model expect.
    defaults = dict(
        is_training=1,   # exp_basic.py branches on args.is_training
        task_name="long_term_forecast", model="SolarVLM", model_id="ukpv",
        features="MS", target="power", freq="t", embed="timeF",
        enc_in=F_DIM, dec_in=F_DIM, c_out=a.num_stations,
        d_model=128, n_heads=16, e_layers=3, d_layers=1, d_ff=2048,
        dropout=0.2, activation="gelu", patch_len=10, stride=8, padding=8,
        periodicity=48, norm_const=0.4, top_k=5, memory_bank_size=20,
        patch_memory_size=100, vlm_type="qwen3vl", image_size=224, roi_size=64,
        use_gnn=True, gnn_layers=3, use_offline_vision=True,
        vision_temporal_layers=2, modal_dropout_rate=0.0, multimodal_lr_ratio=0.2,
        multimodal_loss_weight=0.1, memory_loss_weight=0.05, modal_temp=0.7,
        min_modal_weight=0.0, multimodal_scale=0.0, nonnegative=False,
        grad_clip_norm=1.0, lr_warmup_steps=500, lradj="type1", use_amp=False,
        loss_type="mse", huber_beta=1.0, patience=5, num_workers=4, itr=1,
        percent=1.0, use_dtw=False, inverse=False, scale=False, seasonal_patterns="",
        use_mem_gate=True, learnable_image=False, save_images=False,
        content="uk_pv multi-station PV power forecasting",
        checkpoints=os.path.join(a.out, "checkpoints"),
        results_dir=os.path.join(a.out, "results"),
        test_results_dir=os.path.join(a.out, "test_results"),
        # model.py parses start_time with strptime — must be a valid datetime, not
        # empty. It is only a reference epoch; our loader emits real ts_keys for the
        # vision lookup, so the exact value does not affect the uk_pv windows.
        root_path=a.data_path,
        start_time="2018-01-01 00:00", end_time="2020-01-01 00:00",
        # fixed-size station set: only the COUNT matters to the model
        station_list=[f"plant{i}" for i in range(a.num_stations)],
        use_gpu=torch.cuda.is_available(), use_multi_gpu=False, gpu=0,
        device_ids=[0], devices="0",
    )
    for k, v in defaults.items():
        setattr(a, k, v)
    a.disable_cross_site_attn = bool(a.disable_cross_site_attn)
    for k in ("disable_visual", "disable_text", "disable_gnn"):
        setattr(a, k, bool(getattr(a, k)))
    return a


@torch.no_grad()
def dump_per_site(exp, args, setting):
    """Predict on test-split groups; write solar_vlm_<site>_pred.npz per plant."""
    ds = Dataset_UKPV(root_path="", flag="test",
                      size=[args.seq_len, args.label_len, args.pred_len],
                      num_stations=args.num_stations, data_path=args.data_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = exp.model.eval()
    dev = exp.device
    pred_by, true_by = defaultdict(list), defaultdict(list)
    H = args.pred_len
    for bx, by, bxm, bym, tsk in loader:
        bx, by = bx.float().to(dev), by.float().to(dev)
        bxm, bym = bxm.float().to(dev), bym.float().to(dev)
        dec = torch.cat([by[:, :args.label_len, :],
                         torch.zeros(by.size(0), H, by.size(2), device=dev)], dim=1)
        out, _, _ = model(bx, bxm, dec, bym, ts_keys=list(tsk))
        final = out[:, -H:, :].cpu().numpy()       # [B,H,S]
        true = by[:, -H:, :].cpu().numpy()         # [B,H,S]
        for b in range(final.shape[0]):
            gi = int(str(tsk[b]).split("__")[0])
            seen = set()
            for si, site in enumerate(ds._ggroups[gi]):
                if site in seen:                    # drop padded duplicates
                    continue
                seen.add(site)
                pred_by[site].append(final[b, :, si])
                true_by[site].append(true[b, :, si])
    out_dir = os.path.join(args.out, "preds")
    os.makedirs(out_dir, exist_ok=True)
    for site in sorted(pred_by):
        np.savez(os.path.join(out_dir, f"solar_vlm_{site}_pred.npz"),
                 pred=np.stack(pred_by[site]), true=np.stack(true_by[site]))
    print(f"✓ dumped {len(pred_by)} plants → {out_dir}/solar_vlm_<site>_pred.npz")


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    setting = f"ukpv_SolarVLM_S{args.num_stations}_sl{args.seq_len}_pl{args.pred_len}"
    exp = Experiment(args)
    print(f">>> TRAIN Solar-VLM (uk_pv train-split groups): {setting}")
    exp.train(setting)
    print(">>> EVAL Solar-VLM (uk_pv test-split groups, cross-plant)")
    dump_per_site(exp, args, setting)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
