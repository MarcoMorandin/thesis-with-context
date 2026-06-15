"""Tier-4 specific tests: the zero-initialized CoRA adapter.

Shape/finite/determinism contracts are covered by the parametrized
test_baseline_contract.py; here we test what is specific to tier 4:
the zero-initialized CoRA adapter starting at the backbone forecast.

TS-RAG / Cross-RAG are cluster-only (vendored original code, see
docs/experiments/TIER4_RAG_INTEGRATION.md); their provenance is guarded by
test_tier4_vendor.py, not by in-process unit tests.
"""

from __future__ import annotations

import numpy as np

from common.base import build

from .conftest import make_frame, windows_for


def _splits():
    df = make_frame(n_sites=5, days=6, seed=3)
    sites = lambda *ids: df[df.site_id.isin(ids)]  # noqa: E731
    train = windows_for(sites("site_0", "site_1", "site_2"))
    val = windows_for(sites("site_3"))
    test = windows_for(sites("site_4"))
    return train, val, test


def test_cora_zero_init_starts_at_backbone():
    """Before any training step the adapter must be the identity on the
    backbone forecast (zero-initialized output layer)."""
    import torch

    from tier4.cora import _build_adapter

    adapter = _build_adapter(history=24, horizon=12, n_cov=14,
                             d_hidden=16, dropout=0.0)
    adapter.eval()
    with torch.no_grad():
        residual = adapter(
            torch.rand(4, 12), torch.rand(4, 24),
            torch.ones(4, 24), torch.rand(4, 36, 14),
        )
    assert torch.equal(residual, torch.zeros(4, 12))


def test_cora_improves_or_matches_backbone_on_train_like_data():
    """Sanity: trained CoRA should not be (much) worse than its backbone
    on held-out plants of the same synthetic process."""
    train, val, test = _splits()
    model = build("cora", backbone="persistence", epochs=5, batch_size=64,
                  device="cpu", patience=3, max_train_windows=2000)
    model.fit(train, val)
    batch = test.batch(list(range(min(64, len(test)))))
    mask = batch["mask_future"] * batch["daylight_future"]

    def masked_mae(pred):
        return (np.abs(pred - batch["y_future"]) * mask).sum() / mask.sum()

    backbone_mae = masked_mae(build("persistence").predict(batch).point)
    cora_mae = masked_mae(model.predict(batch).point)
    assert cora_mae <= backbone_mae * 1.10  # allow small training noise
