"""Train + test the `neuralforecast` iTransformer on MMTSFM's exact protocol.

Why this exists
---------------
The tier-2 `itransformer` row (in-repo port, run_eval.py harness) beats MMTSFM
on uk_pv, but the two arms were never trained under the same conditions: the
port sees stride-1 windows (~12x more), batch 256, lr 1e-3, up to 100 epochs,
and history-only covariates. This script removes every one of those degrees of
freedom by reusing MMTSFM's own datamodule, trainer settings and test-time
evaluator, so the comparison isolates the model.

Identical to the MMTSFM curriculum by construction (all reused, not re-coded):

    windows    mmtsfm.data.pv_record.PVRecordDataset (MMTSFM's own dataset)
               14-day history / 6-h horizon, committed disjoint plant splits,
               future_cov="all", train stride 12, val/test stride = H
    recipe     AdamW lr 1e-4 wd 1e-2, warmup 500 + cosine, bf16-mixed,
               grad-clip 1.0, effective batch 16, EarlyStopping(val/loss, 7)
    scoring    eval.protocol_eval.ProtocolEvaluator -> results/<tag>.json

Differences that remain, and are the point of the run: the model, and the fact
that iTransformer has no visual stream (it is the numerical control).

    uv run --group nf python tier2/train_itransformer_nf.py \
        --data-dir /leonardo_scratch/fast/IscrC_MTSFM/data --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BASELINES = Path(__file__).resolve().parents[1]
_MMTSFM_SRC = _BASELINES.parent / "MMTSFM" / "src"
for _p in (_BASELINES, _MMTSFM_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from pytorch_lightning.callbacks import (  # noqa: E402
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger  # noqa: E402

from common import config  # noqa: E402
from tier2.data import build_dataset, build_loader  # noqa: E402
from tier2.module import ITransformerNFModule  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # data — defaults mirror MMTSFM/configs/data/ukpv.yaml
    p.add_argument("--data-dir", default=str(Path(config.DEFAULT_DATA_PATH).parent))
    p.add_argument("--dataset", default="uk_pv")
    p.add_argument("--history-days", type=float, default=config.HISTORY_DAYS)
    p.add_argument("--horizon-hours", type=float, default=config.HORIZON_HOURS)
    p.add_argument(
        "--train-stride",
        type=int,
        default=12,
        help="MMTSFM curriculum TRAIN_STRIDE; val/test always stride H",
    )
    p.add_argument(
        "--future-cov",
        default="all",
        choices=["all", "history"],
        help="'all' = future weather in the covariate variates (MMTSFM parity)",
    )
    p.add_argument(
        "--loss",
        default="mse",
        choices=["mae", "mse"],
        help="library-native point loss; mse matches the paper's own "
        "training script (thuml/iTransformer nn.MSELoss())",
    )
    # recipe — defaults mirror MMTSFM/configs/{trainer,model}
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="effective batch = batch-size * accumulate (MMTSFM: 16)",
    )
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--precision", default="bf16-mixed")
    # model — same width/depth as the in-repo tslib port, so implementation and
    # training modality are the only moving parts
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    # bookkeeping
    p.add_argument("--out", default=str(_BASELINES / "results"))
    p.add_argument("--tag", default="")
    p.add_argument(
        "--ckpt-dir", default=str(_BASELINES / "checkpoints" / "itransformer_nf")
    )
    p.add_argument(
        "--sp-reference",
        default=None,
        help="Smart-Persistence results JSON for the Skill Score",
    )
    p.add_argument(
        "--limit-train-batches",
        type=float,
        default=None,
        help="smoke knob; leave unset for a real run",
    )
    return p.parse_args()


# Native cadence per dataset (knowledge/dataset.md): uk_pv 30-min, goes_pvdaq
# 15-min. PVRecordDataset derives the same numbers from the parquet; resolving
# them here too lets the model be sized without a second full parquet read.
STEPS_PER_DAY = {"uk_pv": 48, "goes_pvdaq": 96}


def window_steps(args: argparse.Namespace) -> tuple[int, int]:
    """(T, H) in steps — identical arithmetic to PVRecordDataset."""
    spd = STEPS_PER_DAY.get(args.dataset)
    if spd is None:
        raise SystemExit(
            f"unknown cadence for dataset {args.dataset!r}; add it to STEPS_PER_DAY"
        )
    return int(round(args.history_days * spd)), int(
        round(args.horizon_hours / 24.0 * spd)
    )


def build_loaders(args: argparse.Namespace, history: int, horizon: int):
    """(train, val, test) loaders over MMTSFM's own PVRecordDataset windows."""
    splits = {}
    for split in ("train", "val", "test"):
        ds = build_dataset(
            split=split,
            data_dir=args.data_dir,
            dataset_name=args.dataset,
            history=history,
            horizon=horizon,
            train_stride=args.train_stride,
        )
        splits[split] = build_loader(
            ds, args.batch_size, args.num_workers, train=(split == "train")
        )
        print(f"[data] {split}: {len(ds)} windows", flush=True)
    return splits["train"], splits["val"], splits["test"]


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    tag = (
        args.tag
        or f"itransformer_nf_s2_{'ukpv' if args.dataset == 'uk_pv' else args.dataset}_seed{args.seed}"
    )
    root = Path(args.ckpt_dir) / f"seed{args.seed}"
    root.mkdir(parents=True, exist_ok=True)

    history, horizon = window_steps(args)
    train_loader, val_loader, test_loader = build_loaders(args, history, horizon)
    module = ITransformerNFModule(
        history=history,
        horizon=horizon,
        n_cov=len(config.COV_COLS),
        future_cov=args.future_cov,
        loss_fn=args.loss,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        results_dir=args.out,
        results_tag=tag,
        sp_reference_path=args.sp_reference,
        data_path=str(Path(args.data_dir) / "dataset_all.parquet"),
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )

    ckpt = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        filename="epoch{epoch:02d}",
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate,
        default_root_dir=str(root),
        logger=CSVLogger(save_dir=str(root), name="csv"),
        log_every_n_steps=50,
        enable_progress_bar=False,
        limit_train_batches=args.limit_train_batches or 1.0,
        callbacks=[
            ckpt,
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=args.patience,
                min_delta=args.early_stop_min_delta,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(
        module, dataloaders=test_loader, ckpt_path=ckpt.best_model_path or None
    )


if __name__ == "__main__":
    main()
