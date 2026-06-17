"""SUNSET on the uk_pv multimodal track — self-contained runner (added on vendor).

NOT upstream: SUNSET ships as Jupyter notebooks (`models/SUNSET_forecast.ipynb`).
This runner transcribes the *exact* original Keras architecture from that
notebook (2 conv blocks 24→48 filters, BN, 2×2 maxpool; Flatten ⊕ PV-history;
two Dense(1024)+Dropout(0.4); MSE/Adam) and only:
  (1) feeds it our uk_pv multimodal windows (sky-image stack `V` + PV history
      `y_hist`) via `tier6.uk_multimodal.UKMultimodalDataset`, and
  (2) widens the final Dense head from 1 (the original single 15-min-ahead step)
      to H, our forecast horizon — the one architectural change, needed for the
      multi-step protocol (documented in tier6/vendor/VENDOR_NOTICE.md).

Trains on the train plants, evaluates each held-out test plant, and dumps
`sunset_<site>_pred.npz` (`pred`,`true` (N,H)) in our baseline-contract format
for `scripts/import_predictions.py`.

    python run_ukpv.py --data <dataset_all.parquet> --h5 <images_all.h5> \
        --out results_ukpv --epochs 20 --pred_len 12

Runs on the whole dataset (uk_pv + goes_pvdaq) by default — `sites_for_split`
returns plants of every dataset in the split, and the bridge grayscales both
128px and 256px frames to a single channel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

# baselines/ on path → tier6.uk_multimodal + common.*
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tier6.uk_multimodal import UKMultimodalDataset, sites_for_split  # noqa: E402
from common import config  # noqa: E402


def build_arrays(ds: UKMultimodalDataset, history: int, horizon: int,
                 max_windows: int = 0, seed: int = 0):
    """Materialize (X_img (N,S,S,T), X_pv (N,T), Y (N,H), M (N,H), sites).

    When ``max_windows`` > 0 and the split has more windows, a random subset of
    window *indices* is chosen up front so we never hold the entire split in
    RAM at once. Eagerly building every window and capping afterwards OOM-killed
    the job: the train split is hundreds of thousands of (S,S,T) frames.
    """
    n = len(ds)
    if max_windows and n > max_windows:
        idx = np.random.default_rng(seed).permutation(n)[:max_windows]
    else:
        idx = np.arange(n)
    X_img, X_pv, Y, M, sites = [], [], [], [], []
    for i in idx:
        it = ds[int(i)]
        # V (T,1,S,S) → stack frames along channels (S,S,T) as SUNSET does
        v = it["V"][:, 0, :, :]                      # (T,S,S)
        X_img.append(np.transpose(v, (1, 2, 0)))     # (S,S,T)
        X_pv.append(it["y_hist"].astype(np.float32))  # (T,)
        Y.append(it["y_future"].astype(np.float32))   # (H,)
        M.append(it["mask_future"].astype(np.float32))
        sites.append(str(it["site_id"]))
    return (
        np.asarray(X_img, np.float32), np.asarray(X_pv, np.float32),
        np.asarray(Y, np.float32), np.asarray(M, np.float32), np.asarray(sites),
    )


def sunset_model(img_side: int, n_frames: int, history: int, horizon: int):
    """The original SUNSET_forecast Keras graph; head widened 1 → horizon."""
    from tensorflow import keras

    num_filters, kernel_size, pool_size, strides = 24, [3, 3], [2, 2], 2
    dense_size, drop_rate = 1024, 0.4

    x_in = keras.Input(shape=(img_side, img_side, n_frames))   # sky-image stack
    x2_in = keras.Input(shape=(history,))                       # PV history

    x = keras.layers.Conv2D(num_filters, kernel_size, padding="same", activation="relu")(x_in)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size, strides)(x)
    x = keras.layers.Conv2D(num_filters * 2, kernel_size, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size, strides)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Concatenate(axis=1)([x, x2_in])
    x = keras.layers.Dense(dense_size, activation="relu")(x)
    x = keras.layers.Dropout(drop_rate)(x)
    x = keras.layers.Dense(dense_size, activation="relu")(x)
    x = keras.layers.Dropout(drop_rate)(x)
    y_out = keras.layers.Dense(units=horizon)(x)               # 1 → H (our protocol)
    return keras.Model(inputs=[x_in, x2_in], outputs=y_out)


def masked_mse(y_true, y_pred):
    import tensorflow as tf

    y_t, m = y_true[..., :y_pred.shape[-1]], y_true[..., y_pred.shape[-1]:]
    se = tf.square(y_pred - y_t) * m
    return tf.reduce_sum(se) / (tf.reduce_sum(m) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=config.DEFAULT_DATA_PATH)
    ap.add_argument("--h5", default=config.DEFAULT_IMAGES_H5)
    ap.add_argument("--out", default="results_ukpv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pred_len", type=int, default=config.HORIZON_STEPS)
    ap.add_argument("--history", type=int, default=config.HISTORY_STEPS)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--max_train_windows", type=int, default=60000,
                    help="cap training windows for tractability (0 = all)")
    ap.add_argument("--max_val_windows", type=int, default=20000,
                    help="cap validation windows (early-stop only; 0 = all)")
    args = ap.parse_args()

    import tensorflow as tf

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    H, T = args.pred_len, args.history
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def make_ds(part):
        return UKMultimodalDataset(
            site_ids=sites_for_split(part), data_path=args.data, h5_path=args.h5,
            history=T, horizon=H, stride=args.stride, img_size=args.img_size,
        )

    print(">>> building train/val arrays")
    Xi, Xp, Y, M, _ = build_arrays(make_ds("train"), T, H,
                                   max_windows=args.max_train_windows, seed=args.seed)
    Vxi, Vxp, Vy, Vm, _ = build_arrays(make_ds("val"), T, H,
                                       max_windows=args.max_val_windows, seed=args.seed)
    print(f"    train windows={len(Xi)}  val windows={len(Vxi)}")

    model = sunset_model(args.img_size, T, T, H)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=masked_mse)
    model.fit(
        [Xi, Xp], np.concatenate([Y, M], axis=1),
        validation_data=([Vxi, Vxp], np.concatenate([Vy, Vm], axis=1)),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    print(">>> evaluating held-out test plants")
    for site in sites_for_split("test"):
        ds = UKMultimodalDataset(
            site_ids=[site], data_path=args.data, h5_path=args.h5,
            history=T, horizon=H, stride=args.stride, img_size=args.img_size,
        )
        if len(ds) == 0:
            print(f"    {site}: no windows, skip")
            continue
        Xi_t, Xp_t, Y_t, _, _ = build_arrays(ds, T, H)
        pred = np.clip(model.predict([Xi_t, Xp_t], batch_size=args.batch_size, verbose=0), 0.0, 1.0)
        np.savez(out / f"sunset_{site}_pred.npz",
                 pred=pred.astype(np.float32), true=Y_t.astype(np.float32))
        print(f"    {site}: {len(pred)} windows → sunset_{site}_pred.npz")
    print(f"✓ SUNSET uk_pv done → {out}/sunset_*_pred.npz")


if __name__ == "__main__":
    main()
