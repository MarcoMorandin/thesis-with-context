"""TabFM baseline (Google) — second tabular foundation model on the tier-1 table.

`google-research/tabfm` (v1.0.0) is a zero-shot tabular FM: alternating
row/column attention, then row compression, then in-context learning over the
compressed embeddings (TabICL-style). Same premise as TabPFN-3
(``tier1/tabpfn_model.py``), different pretraining and architecture — so running
both turns "TabPFN does not close the gap on PV" into the broader claim about
tabular FMs that a reviewer will actually ask for.

Two arms, mirroring Google's own TabArena reporting:

* ``tabfm``     — the default ``TabFMRegressor``
* ``tabfm_ens`` — ``TabFMRegressor.ensemble()``: sqrt feature crosses, sqrt SVD
                  features, NNLS-weighted blending. Both use 32 ensemble
                  members; only the feature/blending schedule differs.

Placement note: this module lives in ``tier2/`` by explicit request, but it is a
zero-shot tabular FM, *not* a supervised deep-TS model like the rest of the
tier. It is fed by ``tier1.features`` — the exact flattened ``(Y, X_cov)`` table
LightGBM and TabPFN see — which is what keeps the three tabular arms internally
comparable. Report it as such; the tier number here is filing, not taxonomy.

**Point predictions only.** Upstream has no regression quantile head (the
estimator's only ``quantile`` machinery is ``QuantileTransformer``
preprocessing), so ``supports_quantiles`` is False and CRPS / coverage / ECE are
N/A for this arm — same handling as ``itransformer_nf`` and ``ttm_zs``. A
zero-width interval fallback would corrupt pinball loss silently, so there is
none.

Optional dependency: ``uv sync --group tabfm`` (a git dependency — TabFM is not
on PyPI). Weights (``google/tabfm-1.0.0-pytorch``) are **ungated**, unlike
TabPFN's token gate, but are released under ``tabfm-non-commercial-v1.0``:
non-commercial, non-production use only. The Apache-2.0 licence covers the
source code, not the checkpoint.
"""

from __future__ import annotations

import numpy as np

from common import config
from common.base import Baseline, Forecast, register
from common.windows import WindowDataset

from tier1.features import build_features, training_table


@register
class TabFMRegressorBaseline(Baseline):
    name = "tabfm"
    tier = 2
    requires_fit = True
    supports_quantiles = False
    _ENSEMBLE = False

    def __init__(
        self,
        max_context_rows: int = 10_000,
        n_estimators: int = 32,
        ens_batch_size: int = 1,
        device: str | None = None,
        cache_context: bool = True,
        quantize_kv_cache: bool = False,
        keep_cache_on_device: bool = False,
        row_chunk_size: int | None = 1024,
        seed: int = config.SEED,
    ):
        # 10k, not TabPFN's 100k: TabFM reads the context through n_estimators
        # separately-transformed views, so an equal context costs ~32x TabPFN's
        # already GPU-bound inference. This is a REPORTED protocol parameter --
        # it is a --model-kwargs knob and must be quoted alongside any number,
        # never quietly retuned to fit a wall-clock limit.
        self.max_context_rows = max_context_rows
        self.n_estimators = n_estimators
        # Members-per-forward. Pure activation tiling: members are independent,
        # so this changes peak device memory and nothing else. Upstream's own
        # default is 1; anything higher multiplies the transient Fourier cell
        # tensor [B, row_chunk, H, G, E] by B and OOMs a 64 GB A100 on this
        # table (705 raw features -> 500 subsampled per member).
        self.ens_batch_size = ens_batch_size
        self.device = device
        self.cache_context = cache_context
        self.quantize_kv_cache = quantize_kv_cache
        # Park each member's prefill cache on the host instead of accumulating
        # 32 un-quantized ICL K/V caches on the GPU (with quantize_kv_cache
        # False they are fp-wide, which is what fills the card before the first
        # big allocation even lands). Costs one H2D copy per predict() batch;
        # the cached values themselves are untouched, so predictions stay
        # bit-identical -- which is the whole reason quantisation stays off.
        self.keep_cache_on_device = keep_cache_on_device
        # Rows per chunk in the cell embedder / row interactor. Upstream ships
        # 4096; the chunks are concatenated, so this is a memory knob with no
        # effect on the output. NOT a protocol parameter -- unlike
        # max_context_rows and n_estimators it may be retuned freely.
        self.row_chunk_size = row_chunk_size
        self.seed = seed
        self._model = None

    def _estimator_kwargs(self, model) -> dict:
        return dict(
            model=model,
            n_estimators=self.n_estimators,
            batch_size=self.ens_batch_size,
            random_state=self.seed,
            # Encode the in-context rows once at fit() instead of re-encoding
            # them on every predict(); run_eval calls predict() once per eval
            # batch, so the uncached path re-reads the whole context thousands
            # of times. With quantize_kv_cache=False the cached path is
            # documented upstream as numerically identical to the uncached one
            # -- exactness over device memory, since the contract test demands
            # bit-identical repeated predictions.
            cache_context=self.cache_context,
            maybe_quantize_kv_cache=self.quantize_kv_cache,
            keep_cache_on_device=self.keep_cache_on_device,
        )

    def _set_row_chunk(self, model) -> None:
        """Override upstream's activation-chunk width on every submodule."""
        if self.row_chunk_size is None:
            return
        for module in model.modules():
            if hasattr(module, "row_chunk_size"):
                module.row_chunk_size = self.row_chunk_size

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        try:
            import torch
            from tabfm import TabFMRegressor
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "tabfm not installed — run `uv sync --group tabfm`"
            ) from err

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bfloat16 is the compute design of both releases (the JAX one defaults
        # to jnp.bfloat16); the checkpoint itself is stored fp32.
        model = tabfm_v1_0_0.load(
            model_type="regression", device=device, dtype=torch.bfloat16
        )
        self._set_row_chunk(model)

        x, y = training_table(train, self.max_context_rows, self.seed)
        kwargs = self._estimator_kwargs(model)
        self._model = (
            TabFMRegressor.ensemble(**kwargs)
            if self._ENSEMBLE
            else TabFMRegressor(**kwargs)
        )
        self._model.fit(x, y)

    def predict(self, batch: dict) -> Forecast:
        feats = build_features(batch)
        n, h, f = feats.shape
        point = np.asarray(self._model.predict(feats.reshape(n * h, f)))
        point = np.clip(point, 0.0, 1.0).reshape(n, h)
        # quantiles stay None on purpose — see the module docstring.
        return Forecast(point=point.astype(np.float32))


@register
class TabFMEnsembleBaseline(TabFMRegressorBaseline):
    """TabFM-Ensemble: cross + SVD features and NNLS-blended ensemble weights."""

    name = "tabfm_ens"
    _ENSEMBLE = True
