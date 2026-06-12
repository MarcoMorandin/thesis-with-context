"""Statistical rigor toolkit (BASELINE_COMPARISON.md §4.5).

* Diebold–Mariano test on per-sample loss differentials, with the
  Harvey–Leybourne–Newbold small-sample correction and Newey–West variance
  for h-step-ahead forecasts.
* Paired block bootstrap (block = day, 1000 resamples) → 95 % CI on ΔNMAE.
* Holm–Bonferroni step-down correction over the baseline set.

Bold a table entry only when the bootstrap CI excludes 0 AND the
Holm-adjusted DM p-value is < 0.05.
"""

from __future__ import annotations

import numpy as np


def _t_sf(x: float, df: int) -> float:
    """Two-sided survival of |t| under Student-t via the incomplete beta.

    Avoids a scipy dependency; standard identity
    P(|T| > x) = I_{df/(df+x²)}(df/2, 1/2).
    """
    from math import lgamma

    if df <= 0:
        return float("nan")
    z = df / (df + x * x)

    # regularized incomplete beta I_z(a, b) by continued fraction (Lentz)
    def betainc(a: float, b: float, x: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        ln_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = np.exp(a * np.log(x) + b * np.log(1 - x) - ln_beta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(200):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + num / (c if abs(c) > 1e-30 else 1e-30)
            f *= c * d
            if abs(1.0 - c * d) < 1e-12:
                break
        return front * (f - 1.0)

    # symmetry: use the smaller tail for numerical stability
    p = betainc(df / 2.0, 0.5, z)
    return float(min(max(p, 0.0), 1.0))


def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
) -> dict[str, float]:
    """Diebold–Mariano test: H0 'model A and B equally accurate'.

    loss_a / loss_b are aligned per-sample losses (same eval windows, same
    order — guaranteed by the deterministic WindowDataset ordering).
    Negative statistic ⇒ A better (lower loss). Returns the HLN-corrected
    statistic and the two-sided p-value.
    """
    if loss_a.shape != loss_b.shape:
        raise ValueError("loss arrays must be aligned (same eval windows)")
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = len(d)
    if n < 2:
        return {"stat": float("nan"), "p_value": float("nan"), "mean_diff": float("nan")}
    dbar = d.mean()
    centered = d - dbar
    var = (centered**2).mean()
    for lag in range(1, h):  # Newey–West for h-step-ahead correlation
        cov = (centered[lag:] * centered[:-lag]).mean()
        var += 2.0 * cov
    var = max(var, 1e-300)
    dm = dbar / np.sqrt(var / n)
    # Harvey–Leybourne–Newbold small-sample correction
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * k)
    return {
        "stat": stat,
        "p_value": _t_sf(abs(stat), df=n - 1),
        "mean_diff": float(dbar),
        "n": float(n),
    }


def block_bootstrap_ci(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    blocks: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Paired block bootstrap CI on the mean loss difference (A − B).

    ``blocks`` assigns each sample to a block (day key); whole blocks are
    resampled with replacement, preserving intra-day correlation.
    CI excluding 0 ⇒ significant difference.
    """
    if not (loss_a.shape == loss_b.shape == blocks.shape):
        raise ValueError("loss and block arrays must be aligned")
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    unique_blocks = np.unique(blocks)
    by_block = [d[blocks == blk] for blk in unique_blocks]
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    n_blocks = len(by_block)
    for i in range(n_resamples):
        pick = rng.integers(0, n_blocks, n_blocks)
        sample = np.concatenate([by_block[j] for j in pick])
        means[i] = sample.mean()
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0.0 or hi < 0.0),
        "n_blocks": float(n_blocks),
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm–Bonferroni step-down over a family of comparisons (§4.5).

    Returns, per comparison, the adjusted p-value and whether H0 is
    rejected at family-wise level ``alpha``.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    adjusted_running = 0.0
    rejecting = True
    for rank, (name, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        adjusted_running = max(adjusted_running, adjusted)  # monotone
        if adjusted_running > alpha:
            rejecting = False
        out[name] = {
            "p_value": p,
            "p_adjusted": adjusted_running,
            "reject": rejecting and adjusted_running <= alpha,
        }
    return out
