"""Retrieval-augmented baselines over a frozen TSFM (Tier 4).

**Lightweight, dependency-free reference implementation.** For the *headline*
Tier-4 numbers we run the authors' original code, vendored under
``baselines/tier4/vendor/{ts_rag,cross_rag}`` (TS-RAG arXiv:2503.07649 NeurIPS
2025; Cross-RAG arXiv:2603.14709) — see ``docs/experiments/TIER4_RAG_INTEGRATION.md``
for the cluster run recipe (their code needs Chronos-Bolt + faiss-gpu + numpy 1.25,
a separate env). The classes below are kept as (a) a CPU/no-GPU fallback and (b)
the contract-test backbone; they are an α-blend approximation, not the published
ARM / cross-attention fusion.

* TS-RAG (`ts_rag`, P0)    — analog retrieval over train-plant history
  windows blended with the frozen backbone forecast via a single mixing
  weight α tuned on the validation plants.
* Cross-RAG (`cross_rag`)  — stronger fusion (A08): retrieval keys augmented
  with the future clear-sky profile (clock/season-aware analogs) and a
  per-horizon-step mixing vector α_h tuned on validation plants.

Fairness rules enforced (BASELINE_COMPARISON.md §3):
- the datastore is populated from **train-plant windows only** (the
  datastore is whatever ``fit`` receives as the train split; site ids are
  recorded in ``datastore_sites`` for auditability);
- mixing weights are tuned on the validation plants, never on test
  (A15: "RAG baselines tuned, not strawmanned").

The backbone is any registered zero-shot baseline (default: chronos2_zs,
frozen). Contract tests use the dependency-free `persistence` backbone.
"""

from __future__ import annotations

import numpy as np

from common import config
from common.base import Baseline, Forecast, build, register
from common.windows import WindowDataset

_EPS = 1e-3


def _znorm(y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask-aware per-row z-normalization. Returns (z, mean, std)."""
    count = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    mean = (y * mask).sum(axis=1, keepdims=True) / count
    var = ((y - mean) ** 2 * mask).sum(axis=1, keepdims=True) / count
    std = np.maximum(np.sqrt(var), _EPS)
    return (y - mean) / std * mask, mean, std


class _RAGBase(Baseline):
    tier = 4
    requires_fit = True
    per_step_alpha = False

    def __init__(
        self,
        backbone: str = "chronos2_zs",
        backbone_kwargs: dict | None = None,
        top_k: int = 8,
        temperature: float = 0.5,
        max_datastore: int = 200_000,
        max_tune_batches: int = 50,
        seed: int = config.SEED,
    ):
        self.backbone_name = backbone
        self.backbone_kwargs = backbone_kwargs or {}
        self.top_k = top_k
        self.temperature = temperature
        self.max_datastore = max_datastore
        self.max_tune_batches = max_tune_batches
        self.seed = seed
        self.backbone: Baseline | None = None
        self.alpha: np.ndarray | None = None  # scalar array or (H,) vector
        self.datastore_sites: set[str] = set()
        self._keys = None        # (M, K)
        self._futures = None     # (M, H) z-normed by own history stats
        self._key_sq = None      # (M,) squared norms, cached

    # -- keys ---------------------------------------------------------------

    def _make_keys(self, batch: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (keys (N,K), mean (N,1), std (N,1)) for retrieval."""
        return _znorm(batch["y_hist"], batch["mask_hist"])

    # -- fit ----------------------------------------------------------------

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        self.backbone = build(self.backbone_name, **self.backbone_kwargs)
        if self.backbone.requires_fit:
            self.backbone.fit(train, val)
        self.supports_quantiles = self.backbone.supports_quantiles
        self._build_datastore(train)
        self._tune_alpha(val)

    def _build_datastore(self, train: WindowDataset) -> None:
        keys, futures, sites = [], [], []
        for batch in train.iter_batches(512):
            t = batch["y_hist"].shape[1]
            usable = (batch["mask_future"].min(axis=1) > 0) & (
                batch["mask_hist"].sum(axis=1) >= t // 2
            )
            if not usable.any():
                continue
            k, mean, std = self._make_keys(batch)
            f = (batch["y_future"] - mean) / std
            keys.append(k[usable])
            futures.append(f[usable])
            sites.append(batch["site_id"][usable])
        if not keys:
            self._keys = np.empty((0, 1), np.float32)
            self._futures = np.empty((0, 1), np.float32)
            return
        keys = np.concatenate(keys).astype(np.float32)
        futures = np.concatenate(futures).astype(np.float32)
        sites = np.concatenate(sites)
        if len(keys) > self.max_datastore:
            pick = np.random.default_rng(self.seed).choice(
                len(keys), self.max_datastore, replace=False
            )
            keys, futures, sites = keys[pick], futures[pick], sites[pick]
        self._keys, self._futures = keys, futures
        self._key_sq = (keys**2).sum(axis=1)
        self.datastore_sites = set(np.unique(sites))

    def _tune_alpha(self, val: WindowDataset) -> None:
        """Grid-search the backbone/retrieval mixing weight on val plants."""
        horizon = val.horizon
        grid = np.linspace(0.0, 1.0, 11)
        if len(self._keys) == 0:
            self.alpha = np.ones(1 if not self.per_step_alpha else horizon)
            return
        b_all, r_all, y_all, m_all = [], [], [], []
        for bi, batch in enumerate(val.iter_batches(256)):
            if bi >= self.max_tune_batches:
                break
            b_all.append(self.backbone.predict(batch).point)
            r_all.append(self._retrieval_forecast(batch))
            y_all.append(batch["y_future"])
            m_all.append(batch["mask_future"] * batch["daylight_future"])
        if not b_all:  # empty validation split: keep the backbone untouched
            self.alpha = np.ones(horizon if self.per_step_alpha else 1)
            return
        b = np.concatenate(b_all)
        r = np.concatenate(r_all)
        y = np.concatenate(y_all)
        m = np.concatenate(m_all)
        # masked MAE per alpha (and per step for Cross-RAG)
        errs = np.stack(
            [np.abs(a * b + (1 - a) * r - y) * m for a in grid]
        )  # (A, N, H)
        if self.per_step_alpha:
            denom = np.maximum(m.sum(axis=0), 1.0)            # (H,)
            losses = errs.sum(axis=1) / denom                 # (A, H)
            self.alpha = grid[np.argmin(losses, axis=0)]      # (H,)
        else:
            losses = errs.sum(axis=(1, 2)) / max(m.sum(), 1.0)  # (A,)
            self.alpha = np.array([grid[np.argmin(losses)]])

    # -- predict ------------------------------------------------------------

    def _retrieval_forecast(self, batch: dict) -> np.ndarray:
        q, mean, std = self._make_keys(batch)
        q = q.astype(np.float32)
        k = min(self.top_k, len(self._keys))
        # squared L2 distances via the matmul identity
        d2 = (q**2).sum(axis=1, keepdims=True) - 2.0 * q @ self._keys.T + self._key_sq
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]               # (N, k)
        d_top = np.take_along_axis(d2, idx, axis=1) / q.shape[1]      # scale by dim
        w = np.exp(-d_top / max(self.temperature, 1e-6))
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        analogs = self._futures[idx]                                  # (N, k, H)
        ret_norm = (w[..., None] * analogs).sum(axis=1)               # (N, H)
        return np.clip(ret_norm * std + mean, 0.0, 1.0)

    def predict(self, batch: dict) -> Forecast:
        if self.backbone is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")
        backbone_fc = self.backbone.predict(batch)
        if len(self._keys) == 0:
            return backbone_fc
        retrieval = self._retrieval_forecast(batch)
        alpha = self.alpha if self.per_step_alpha else self.alpha[0]
        point = np.clip(
            alpha * backbone_fc.point + (1.0 - alpha) * retrieval, 0.0, 1.0
        ).astype(np.float32)
        quantiles = None
        if backbone_fc.quantiles is not None:
            shift = (point - backbone_fc.point)[..., None]
            quantiles = np.clip(backbone_fc.quantiles + shift, 0.0, 1.0).astype(
                np.float32
            )
        return Forecast(point=point, quantiles=quantiles)


@register
class TSRAG(_RAGBase):
    """TS-RAG: history-shape analogs, single tuned mixing weight."""

    name = "ts_rag"
    per_step_alpha = False


@register
class CrossRAG(_RAGBase):
    """Cross-RAG (A08): clear-sky-aware retrieval keys, per-step mixing."""

    name = "cross_rag"
    per_step_alpha = True

    def _make_keys(self, batch: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z, mean, std = _znorm(batch["y_hist"], batch["mask_hist"])
        t = batch["y_hist"].shape[1]
        # future clear-sky profile makes analogs clock- and season-aware
        cs_future = batch["clearsky"][:, t:] / config.STC_IRRADIANCE
        return np.concatenate([z, cs_future], axis=1), mean, std
