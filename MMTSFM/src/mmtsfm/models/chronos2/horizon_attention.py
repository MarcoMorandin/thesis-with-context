"""Horizon-attention diagnostics for the s2c arm (ticket 15).

s2c gives the forecast three future positions, each with its own learned lead-time
embedding, and lets them cross-attend a retained 4x4x4 V-JEPA field. Three queries
that collapse into near-duplicates produce exactly the same flat ramp metric as a
genuinely falsified hypothesis; without this module the two are indistinguishable
after the fact. The attention distribution is the primary evidence, because it is
what the model DOES; lead-time embedding distances are secondary (large distances do
not prove the model uses them, small ones do not prove the attention is identical).

The noise floor is measured, not assumed: the same divergence is computed WITHIN one
tau, between two random half-splits of its own samples. Their ratio is the deciding
statistic and needs no magic constant -- 1.0 means indistinguishable from noise.
"""

from __future__ import annotations

import numpy as np

# Pre-registered before any s2c checkpoint exists (2026-08-28), for the same reason
# the 0.0011 ramp threshold was: a threshold chosen after seeing the number is not a
# threshold. Both conditions must hold to call the queries genuinely different.
SEPARATION_THRESHOLD = 2.0  # between-tau divergence vs. the within-tau noise floor
MIN_BETWEEN_L1 = 0.05  # total-variation floor: 2.5% of attention mass must move


def _norm(p: np.ndarray) -> np.ndarray:
    s = p.sum(axis=-1, keepdims=True)
    return np.divide(p, s, out=np.full_like(p, 1.0 / p.shape[-1]), where=s > 0)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.sum(p * np.log(p + eps)))


def _pair_metrics(p: np.ndarray, q: np.ndarray) -> dict:
    return {
        "l1": float(np.abs(p - q).sum()),
        "kl_fwd": _kl(p, q),
        "kl_rev": _kl(q, p),
        "js": _js(p, q),
    }


class HorizonAttentionAccumulator:
    """Streams per-tau attention over the visual field across a test epoch."""

    def __init__(self, n_blocks: int, n_tau: int, n_kv: int, seed: int = 0):
        self._sum = np.zeros((n_blocks, n_tau, n_kv), dtype=np.float64)
        self._half = np.zeros((2, n_blocks, n_tau, n_kv), dtype=np.float64)
        self._rows = 0
        self._half_rows = np.zeros(2, dtype=np.int64)
        # Half-split assignment is per ROW and seeded, not per batch: the test loader
        # is not shuffled, so a batch-parity split would correlate the two halves with
        # site order and understate the noise floor.
        self._rng = np.random.default_rng(seed)

    @property
    def n_rows(self) -> int:
        return int(self._rows)

    def update(self, attn: np.ndarray) -> None:
        """``attn``: ``[n_blocks, rows, n_tau, n_kv]``, already head-averaged and
        restricted to rows whose visual query was active."""
        if attn.size == 0 or attn.shape[1] == 0:
            return
        side = self._rng.integers(0, 2, size=attn.shape[1])
        self._sum += attn.sum(axis=1)
        self._rows += attn.shape[1]
        for h in (0, 1):
            sel = side == h
            if sel.any():
                self._half[h] += attn[:, sel].sum(axis=1)
                self._half_rows[h] += int(sel.sum())

    def report(self, tau_embed: np.ndarray | None = None) -> dict:
        if self._rows == 0:
            return {"n_rows": 0, "verdict": "not_measured"}
        nb, ntau, nkv = self._sum.shape
        p = _norm(self._sum)  # [blocks, tau, kv]
        ph = _norm(self._half)  # [2, blocks, tau, kv]
        pairs = [(i, j) for i in range(ntau) for j in range(i + 1, ntau)]

        per_block, between_l1, within_l1 = [], [], []
        for b in range(nb):
            btw = {f"{i}_vs_{j}": _pair_metrics(p[b, i], p[b, j]) for i, j in pairs}
            wtn = {f"{i}": _pair_metrics(ph[0, b, i], ph[1, b, i]) for i in range(ntau)}
            b_l1 = [v["l1"] for v in btw.values()]
            w_l1 = [v["l1"] for v in wtn.values()]
            between_l1 += b_l1
            within_l1 += w_l1
            per_block.append(
                {
                    "block": b,
                    "between_tau": btw,
                    "within_tau_noise_floor": wtn,
                    "between_l1_mean": float(np.mean(b_l1)),
                    "within_l1_mean": float(np.mean(w_l1)),
                    "entropy_nats_per_tau": [_entropy(p[b, i]) for i in range(ntau)],
                }
            )

        b_mean, w_mean = float(np.mean(between_l1)), float(np.mean(within_l1))
        ratio = float(b_mean / w_mean) if w_mean > 0 else float("inf")
        if self._half_rows.min() < 2:
            verdict = "inconclusive_too_few_rows"
        elif ratio > SEPARATION_THRESHOLD and b_mean > MIN_BETWEEN_L1:
            verdict = "queries_differ"
        else:
            # Not a falsified hypothesis -- a degenerate parameterisation. The three
            # queries learned the same thing, so any ramp result is uninformative
            # about whether horizon-specific spatial attention would have helped.
            verdict = "degenerate_queries_collapsed"

        out = {
            "n_rows": int(self._rows),
            "n_blocks": nb,
            "n_tau": ntau,
            "n_kv": nkv,
            "between_l1_mean": b_mean,
            "within_l1_noise_floor": w_mean,
            "separation_ratio": ratio,
            "separation_threshold": SEPARATION_THRESHOLD,
            "min_between_l1": MIN_BETWEEN_L1,
            "verdict": verdict,
            "uniform_entropy_nats": float(np.log(nkv)),
            "per_block": per_block,
        }
        if tau_embed is not None and tau_embed.shape[0] >= ntau:
            e = np.asarray(tau_embed, dtype=np.float64)[:ntau]
            n = np.linalg.norm(e, axis=1) + 1e-12
            out["tau_embedding"] = {
                f"{i}_vs_{j}": {
                    "l2": float(np.linalg.norm(e[i] - e[j])),
                    "cosine_dist": float(1.0 - e[i] @ e[j] / (n[i] * n[j])),
                }
                for i, j in pairs
            }
        return out
