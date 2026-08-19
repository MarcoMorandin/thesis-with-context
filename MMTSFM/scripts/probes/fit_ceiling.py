"""Fit the G0 ceiling probes and report per-horizon skill.

Three predictor sets: (a) visual only, (b) covariates only, (c) both. The
operative quantity is (c) - (b): what vision adds CONDITIONAL on what the numeric
channel already knows. It upper-bounds what any fusion architecture could extract.

Target is norm_power (the quantity protocol.md scores), so probe skill is directly
comparable with the reported SS numbers.

Three properties are load-bearing for reading a ZERO result as evidence of
absence rather than as a probe artefact:

* **Features are standardized** (train statistics, applied to test), so the
  arbitrary activation scale of the visual encoder does not decide how hard that
  block gets penalized relative to covariates already scaled into [0, 1].
* **The two blocks carry SEPARATE penalties.** Set (c) is ~4k visual dimensions
  bolted onto ~100 strongly predictive covariates. Under one shared penalty
  there is no setting that both keeps the covariate fit and suppresses the
  visual block, so (c) is forced to be *worse* than (b) whenever vision is
  uninformative -- the probe would report "vision hurts" when it only overfit.
* **The visual penalty grid reaches far enough to switch the block OFF** (1e10
  against standardized eigenvalues of order n). With the covariate penalty
  pinned to the one (b) selected, set (c) can therefore always reproduce (b)
  exactly, so `conditional_rel` is bounded below by ~0 by construction. That is
  the right null for a CEILING probe: it can report that vision helps or that it
  does nothing, but it can never manufacture evidence that vision harms.

See test_fit_ceiling.py::test_block_penalties_prevent_a_false_negative_from_overfitting.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

# Features are standardized, so the Gram's diagonal is ~n_rows and the useful
# penalty range runs from "essentially none" up to "block switched off". The
# top of the grid must stay far above n_rows or the off state is unreachable.
# `alpha_selected` is reported so a reader can see whether a grid edge was hit.
DEFAULT_ALPHAS: tuple[float, ...] = tuple(float(a) for a in np.logspace(-2, 10, 13))

# Rows per pass when accumulating the Gram / applying a fit. Bounds the float64
# working copy of a 4k-wide design matrix to ~130 MB instead of ~3 GB.
_CHUNK = 4096


def _skill(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, ref: np.ndarray) -> float:
    """1 - NMAE(pred)/NMAE(ref) on the masked entries. Higher is better."""
    if mask.sum() == 0:
        return float("nan")
    num = np.abs(pred[mask] - y[mask]).mean()
    den = np.abs(ref[mask] - y[mask]).mean()
    return float(1.0 - num / den) if den > 0 else float("nan")


def _relative_reduction(nmae_b: float, nmae_c: float) -> float:
    """Fraction of (b)'s error that (c) removes: (nmae_b - nmae_c) / nmae_b.

    Reference-free companion to `skill`: no external baseline enters this
    quantity, so it cannot be thrown off by a mismatched or arbitrary
    reference the way a difference of two `skill` values can be.
    """
    if not np.isfinite(nmae_b) or nmae_b == 0:
        return float("nan")
    return float((nmae_b - nmae_c) / nmae_b)


def _fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std in float64, returned float32 so applying cannot promote."""
    mean = X.mean(axis=0, dtype=np.float64)
    std = X.std(axis=0, dtype=np.float64)
    # A constant column carries no information; scale 1.0 leaves it as all-zeros
    # after centering rather than producing inf.
    std[~np.isfinite(std) | (std < 1e-12)] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_scaler(X: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = scaler
    out = np.empty(X.shape, dtype=np.float32)
    for i in range(0, X.shape[0], _CHUNK):
        out[i : i + _CHUNK] = (X[i : i + _CHUNK] - mean) / std
    return out


class _RidgeSolver:
    """Block-penalized ridge, refit cheaply across a penalty grid.

    Forms the Gram X'X and X'y ONCE -- the O(n d^2) part -- then each penalty
    setting is a single O(d^3) solve of (X'X + diag(p)) w = X'y. That is what
    makes a penalty search affordable at 4k visual dimensions inside the probe's
    CPU walltime; refitting from the raw rows per grid point would not be.

    `blocks` gives the column count of each contiguous feature block, so the two
    modalities can be penalized independently. Centering X and y reproduces
    `fit_intercept=True`; the intercept is never penalized, which matters
    because the target has a large non-zero mean.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, blocks: Sequence[int]) -> None:
        d = X.shape[1]
        self.blocks = tuple(int(b) for b in blocks)
        assert sum(self.blocks) == d, (self.blocks, d)
        self.x_mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.y_mean = float(np.mean(y, dtype=np.float64))
        gram = np.zeros((d, d), dtype=np.float64)
        xty = np.zeros(d, dtype=np.float64)
        for i in range(0, X.shape[0], _CHUNK):
            b = (X[i : i + _CHUNK] - self.x_mean).astype(np.float64)
            gram += b.T @ b
            xty += b.T @ (y[i : i + _CHUNK].astype(np.float64) - self.y_mean)
        self._gram = gram
        self._xty = xty

    def coef(self, alphas: Sequence[float]) -> np.ndarray:
        penalty = np.repeat(np.asarray(alphas, dtype=np.float64), self.blocks)
        a = self._gram.copy()
        # Add the penalty to the diagonal in place; a full diag() would allocate
        # a second d x d matrix (~140 MB at d=4194) for nothing.
        a.flat[:: a.shape[0] + 1] += penalty
        return np.linalg.solve(a, self._xty)

    def predict(self, X: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
        w = self.coef(alphas).astype(np.float32)
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(0, X.shape[0], _CHUNK):
            out[i : i + _CHUNK] = (X[i : i + _CHUNK] - self.x_mean) @ w
        return out + self.y_mean


def _design(arrays: dict, which: str) -> np.ndarray:
    if which == "a":
        return arrays["X_vis"]
    if which == "b":
        return arrays["X_cov"]
    # Visual block FIRST: _blocks() below and every penalty vector assume this
    # order, and swapping it would silently penalize the wrong modality.
    return np.concatenate([arrays["X_vis"], arrays["X_cov"]], axis=1)


def _blocks(arrays: dict, which: str) -> tuple[int, ...]:
    d_vis = arrays["X_vis"].shape[1]
    d_cov = arrays["X_cov"].shape[1]
    if which == "a":
        return (d_vis,)
    if which == "b":
        return (d_cov,)
    return (d_vis, d_cov)


def _standardized_designs(
    arrays_train: dict, arrays_test: dict
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build (a)/(b)/(c) once each, standardized on TRAIN statistics.

    Scaling is fit on all train rows rather than per-horizon: the horizon mask
    is a property of the TARGET, so every row's features are in-distribution
    regardless, and a global scaler keeps one fixed feature basis across all
    horizons and folds.
    """
    tr = {w: _design(arrays_train, w) for w in ("a", "b", "c")}
    te = {w: _design(arrays_test, w) for w in ("a", "b", "c")}
    for w in ("a", "b", "c"):
        scaler = _fit_scaler(tr[w])
        tr[w] = _apply_scaler(tr[w], scaler)
        te[w] = _apply_scaler(te[w], scaler)
    return tr, te


def _cv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    blocks: Sequence[int],
    folds: list[tuple[np.ndarray, np.ndarray]],
    penalties: Sequence[Sequence[float]],
) -> list[list[np.ndarray]]:
    """[fold][penalty] -> validation predictions. One Gram per fold, not per penalty."""
    out = []
    for tr_i, va_i in folds:
        solver = _RidgeSolver(X[tr_i], y[tr_i], blocks)
        out.append([solver.predict(X[va_i], p) for p in penalties])
    return out


def _pick(
    fold_preds: list[list[np.ndarray]],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_penalties: int,
) -> int:
    """Index of the penalty with the lowest fold-mean validation NMAE."""
    scores = [
        float(
            np.mean(
                [
                    np.abs(fold_preds[k][j] - y[folds[k][1]]).mean()
                    for k in range(len(folds))
                ]
            )
        )
        for j in range(n_penalties)
    ]
    return int(np.argmin(scores))


def fit_and_score(
    arrays_train: dict,
    arrays_test: dict,
    horizon: int = 12,
    n_folds: int = 5,
    alphas: Sequence[float] | None = None,
) -> dict:
    alphas = tuple(DEFAULT_ALPHAS if alphas is None else alphas)
    out: dict = {"alpha_grid": list(alphas)}
    y_tr, m_tr = arrays_train["Y"], arrays_train["Y_mask"]
    y_te, m_te = arrays_test["Y"], arrays_test["Y_mask"]

    # Reference = per-horizon train mean. The gate quantity is a DIFFERENCE
    # (skill(c) - skill(b)), so any fixed reference cancels out of it; the
    # reference's absolute value only matters when reading a single skill
    # number in isolation. Absolute per-horizon NMAE is also reported below
    # and is what compares directly against the model's own NMAE.
    ref_by_h = [float(y_tr[:, h][m_tr[:, h]].mean()) for h in range(horizon)]
    ref_te = np.tile(np.array(ref_by_h), (y_te.shape[0], 1))

    # Built once and shared by the penalty search, the CV spread and the
    # headline fits: (c)'s concatenation is ~1.7 GB on real data.
    Xtr, Xte = _standardized_designs(arrays_train, arrays_test)
    blocks = {w: _blocks(arrays_train, w) for w in ("a", "b", "c")}

    # Folds are grouped by plant so they stay plant-disjoint -- the same
    # discipline the protocol demands of the model, and the reason the penalty
    # chosen here reflects cross-plant generalization rather than within-plant
    # fit. n_splits is floored at 2 so GroupKFold cannot be handed 1.
    groups = arrays_train["site"]
    n_splits = max(2, min(n_folds, len(set(groups.tolist()))))
    # Fallback when a horizon cannot be cross-validated: the grid's midpoint,
    # recorded in alpha_selected like any other choice so it is never silent.
    fallback = len(alphas) // 2

    sets = ("a", "b", "c")
    alpha_idx: dict[str, list[int]] = {w: [] for w in sets}
    nmae: dict[str, list[float]] = {w: [] for w in sets}
    skill: dict[str, list[float]] = {w: [] for w in sets}
    spread: list[float] = []
    spread_rel: list[float] = []

    for h in range(horizon):
        sel = m_tr[:, h]
        yh, gh = y_tr[sel, h], groups[sel]
        cv_ok = sel.sum() >= n_splits and len(set(gh.tolist())) >= n_splits
        folds = (
            list(GroupKFold(n_splits=n_splits).split(np.empty(len(yh)), yh, gh))
            if cv_ok
            else []
        )

        chosen: dict[str, int] = {}
        fold_preds: dict[str, list[list[np.ndarray]]] = {}
        # (b) first: the covariate penalty it selects is what (c) holds fixed,
        # which is what lets (c) reproduce (b) exactly at the top of the grid.
        for w in sets:
            if not cv_ok:
                chosen[w] = fallback
                continue
            grid = _penalty_grid(w, alphas, alphas[chosen["b"]] if w == "c" else None)
            fold_preds[w] = _cv_predictions(Xtr[w][sel], yh, blocks[w], folds, grid)
            chosen[w] = _pick(fold_preds[w], yh, folds, len(grid))

        for w in sets:
            alpha_idx[w].append(chosen[w])
            grid = _penalty_grid(w, alphas, alphas[chosen["b"]] if w == "c" else None)
            solver = _RidgeSolver(Xtr[w][sel], yh, blocks[w])
            pred = solver.predict(Xte[w], grid[chosen[w]])
            nmae[w].append(
                float(np.abs(pred[m_te[:, h]] - y_te[m_te[:, h], h]).mean())
                if m_te[:, h].any()
                else float("nan")
            )
            skill[w].append(_skill(pred, y_te[:, h], m_te[:, h], ref_te[:, h]))

        if not cv_ok:
            spread.append(float("nan"))
            spread_rel.append(float("nan"))
            continue

        # CV spread of both conditional quantities, at the SAME penalties used
        # for the headline numbers -- otherwise the spread would describe a
        # differently-regularized model than the one being gated. Uses the same
        # global per-horizon reference (ref_by_h) as the headline skill numbers,
        # not a fold-local mean, so cv_spread is on the same normalization basis
        # as conditional and the two are directly comparable (that comparison is
        # the study's decision gate).
        vals, vals_rel = [], []
        for k, (_, va_i) in enumerate(folds):
            rb = fold_preds["b"][k][chosen["b"]]
            rc = fold_preds["c"][k][chosen["c"]]
            yv = yh[va_i]
            m = np.ones_like(yv, dtype=bool)
            r = np.full_like(yv, ref_by_h[h])
            vals.append(_skill(rc, yv, m, r) - _skill(rb, yv, m, r))
            vals_rel.append(
                _relative_reduction(
                    float(np.abs(rb - yv).mean()), float(np.abs(rc - yv).mean())
                )
            )
        spread.append(float(np.std(vals)))
        spread_rel.append(float(np.std(vals_rel)))

    for w in sets:
        out[w] = {"nmae": nmae[w], "skill": skill[w]}
    # For (a) and (b) this is the block's own penalty. For (c) it is the VISUAL
    # penalty; (c)'s covariate penalty is alpha_selected["b"] by construction.
    # A (c) value pinned at the grid maximum means "vision switched off won".
    out["alpha_selected"] = {w: [alphas[j] for j in alpha_idx[w]] for w in sets}

    # conditional[h] = skill(c) - skill(b). Float subtraction propagates NaN
    # rather than coercing it to 0 -- a NaN here means "no data at this
    # horizon" (see n_test_valid below), not "no signal at this horizon".
    # Do not "fix" this by defaulting a NaN term to 0; that would turn a
    # missing-data horizon into a false "no signal" reading.
    out["conditional"] = [skill["c"][h] - skill["b"][h] for h in range(horizon)]
    out["n_test_valid"] = [int(m_te[:, h].sum()) for h in range(horizon)]

    # Reference-free companion to `conditional`: fraction of (b)'s error
    # that (c) removes. No external baseline, so it is immune to any
    # mismatch between the reference used for `skill` and the one used for
    # `cv_spread`.
    out["conditional_rel"] = [
        _relative_reduction(nmae["b"][h], nmae["c"][h]) for h in range(horizon)
    ]
    out["cv_spread"] = spread
    out["cv_spread_rel"] = spread_rel
    return out


def _penalty_grid(
    which: str, alphas: Sequence[float], alpha_cov: float | None
) -> list[tuple[float, ...]]:
    """Per-block penalty vectors to search, one entry per grid point.

    (a) and (b) are single-block, so the grid is the plain alpha sweep. (c)
    sweeps only the VISUAL penalty and pins the covariate penalty to the value
    (b) already selected: sweeping both would be a quadratic search for a
    covariate optimum that barely moves, and pinning it is what guarantees the
    grid's top end reproduces (b).
    """
    if which == "c":
        assert alpha_cov is not None
        return [(a, alpha_cov) for a in alphas]
    return [(a,) for a in alphas]


def run_g0(
    cache_dir: str,
    parquet_path: str,
    out_path: str,
    horizon: int = 12,
    max_files: int | None = None,
    alphas: Sequence[float] | None = None,
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
    res = fit_and_score(tr, te, horizon=horizon, alphas=alphas)
    res["n_train"], res["n_test"] = int(tr["Y"].shape[0]), int(te["Y"].shape[0])
    res["n_skipped_train"] = int(tr["n_skipped"])
    res["n_skipped_test"] = int(te["n_skipped"])
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
    ap.add_argument(
        "--alphas",
        default=None,
        help="comma-separated ridge penalties; default is logspace(-2, 10, 13)",
    )
    a = ap.parse_args()
    grid = [float(v) for v in a.alphas.split(",")] if a.alphas else None
    r = run_g0(a.cache_dir, a.parquet, a.out, a.horizon, a.max_files, grid)
    print("conditional (c)-(b) per horizon:", [round(v, 5) for v in r["conditional"]])
    print(
        "conditional_rel                :", [round(v, 5) for v in r["conditional_rel"]]
    )
    print("cv spread_rel                  :", [round(v, 5) for v in r["cv_spread_rel"]])
    print("alpha* covariate block         :", r["alpha_selected"]["b"])
    print("alpha* visual block in (c)     :", r["alpha_selected"]["c"])
