"""Fit the G0 ceiling probes and report per-horizon skill.

Three predictor sets: (a) visual only, (b) covariates only, (c) both. The
operative quantity is (c) - (b): what vision adds CONDITIONAL on what the numeric
channel already knows. It upper-bounds what any fusion architecture could extract.

Target is norm_power (the quantity protocol.md scores), so probe skill is directly
comparable with the reported SS numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


def _skill(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, ref: np.ndarray) -> float:
    """1 - NMAE(pred)/NMAE(ref) on the masked entries. Higher is better."""
    if mask.sum() == 0:
        return float("nan")
    num = np.abs(pred[mask] - y[mask]).mean()
    den = np.abs(ref[mask] - y[mask]).mean()
    return float(1.0 - num / den) if den > 0 else float("nan")


def _design(arrays: dict, which: str) -> np.ndarray:
    if which == "a":
        return arrays["X_vis"]
    if which == "b":
        return arrays["X_cov"]
    return np.concatenate([arrays["X_vis"], arrays["X_cov"]], axis=1)


def fit_and_score(
    arrays_train: dict,
    arrays_test: dict,
    horizon: int = 12,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> dict:
    out: dict = {}
    y_tr, m_tr = arrays_train["Y"], arrays_train["Y_mask"]
    y_te, m_te = arrays_test["Y"], arrays_test["Y_mask"]

    # Reference = persistence of the most recent observed value. X_cov column 0
    # is not the target, so use the per-horizon train mean as the naive reference;
    # the real run overrides this with smart persistence (see run_g0()).
    ref_te = np.tile(
        np.array([y_tr[:, h][m_tr[:, h]].mean() for h in range(horizon)]),
        (y_te.shape[0], 1),
    )

    for which in ("a", "b", "c"):
        Xtr, Xte = _design(arrays_train, which), _design(arrays_test, which)
        nmae, skill = [], []
        for h in range(horizon):
            sel = m_tr[:, h]
            model = Ridge(alpha=alpha)
            model.fit(Xtr[sel], y_tr[sel, h])
            pred = model.predict(Xte)
            nmae.append(
                float(np.abs(pred[m_te[:, h]] - y_te[m_te[:, h], h]).mean())
                if m_te[:, h].any()
                else float("nan")
            )
            skill.append(_skill(pred, y_te[:, h], m_te[:, h], ref_te[:, h]))
        out[which] = {"nmae": nmae, "skill": skill}

    out["conditional"] = [
        out["c"]["skill"][h] - out["b"]["skill"][h] for h in range(horizon)
    ]

    # CV spread of the conditional quantity, grouped by plant so folds stay
    # plant-disjoint — the same discipline the protocol demands of the model.
    groups = arrays_train["site"]
    n_splits = min(n_folds, len(set(groups.tolist())))
    spread = []
    for h in range(horizon):
        sel = m_tr[:, h]
        if sel.sum() < n_splits or len(set(groups[sel].tolist())) < n_splits:
            spread.append(float("nan"))
            continue
        vals = []
        gkf = GroupKFold(n_splits=n_splits)
        Xb, Xc = _design(arrays_train, "b")[sel], _design(arrays_train, "c")[sel]
        yh, gh = y_tr[sel, h], groups[sel]
        for tr_i, va_i in gkf.split(Xb, yh, gh):
            rb = Ridge(alpha=alpha).fit(Xb[tr_i], yh[tr_i]).predict(Xb[va_i])
            rc = Ridge(alpha=alpha).fit(Xc[tr_i], yh[tr_i]).predict(Xc[va_i])
            m = np.ones_like(yh[va_i], dtype=bool)
            r = np.full_like(yh[va_i], yh[tr_i].mean())
            vals.append(_skill(rc, yh[va_i], m, r) - _skill(rb, yh[va_i], m, r))
        spread.append(float(np.std(vals)))
    out["cv_spread"] = spread
    return out


def run_g0(
    cache_dir: str,
    parquet_path: str,
    out_path: str,
    horizon: int = 12,
    max_files: int | None = None,
) -> dict:
    """Assemble train/test arrays from the real cache and write the report."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "baselines"))
    from common.splits import load_splits  # noqa: E402

    from scripts.probes.ceiling_dataset import build_arrays

    splits = load_splits()
    train_sites = {str(s) for s in splits["uk_pv"]["train"]}
    test_sites = {str(s) for s in splits["uk_pv"]["test"]}
    assert not (train_sites & test_sites), "train/test plants must be disjoint"

    tr = build_arrays(
        Path(cache_dir), Path(parquet_path), train_sites, horizon, max_files=max_files
    )
    te = build_arrays(
        Path(cache_dir), Path(parquet_path), test_sites, horizon, max_files=max_files
    )
    res = fit_and_score(tr, te, horizon=horizon)
    res["n_train"], res["n_test"] = int(tr["Y"].shape[0]), int(te["Y"].shape[0])
    Path(out_path).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--max-files", type=int, default=None)
    a = ap.parse_args()
    r = run_g0(a.cache_dir, a.parquet, a.out, a.horizon, a.max_files)
    print("conditional (c)-(b) per horizon:", [round(v, 5) for v in r["conditional"]])
    print("cv spread                     :", [round(v, 5) for v in r["cv_spread"]])
