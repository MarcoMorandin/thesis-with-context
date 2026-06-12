"""Tier-4 specific tests: retrieval fairness, tuning, adapter identity.

Shape/finite/determinism contracts are covered by the parametrized
test_baseline_contract.py; here we test what is specific to tier 4:
the datastore rule (§3: train-plant data only), val-based tuning, and
the zero-initialized CoRA adapter starting at the backbone forecast.
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


def test_ts_rag_datastore_uses_train_plants_only():
    train, val, test = _splits()
    model = build("ts_rag", backbone="persistence", top_k=4, max_datastore=1000)
    model.fit(train, val)
    train_sites = {s.site_id for s in train.series}
    assert model.datastore_sites  # non-empty
    assert model.datastore_sites <= train_sites  # the §3 datastore rule


def test_ts_rag_alpha_tuned_on_val_within_grid():
    train, val, _ = _splits()
    model = build("ts_rag", backbone="persistence", top_k=4, max_datastore=1000)
    model.fit(train, val)
    assert model.alpha.shape == (1,)
    assert 0.0 <= model.alpha[0] <= 1.0


def test_cross_rag_per_step_alpha():
    train, val, _ = _splits()
    model = build("cross_rag", backbone="persistence", top_k=4, max_datastore=1000)
    model.fit(train, val)
    assert model.alpha.shape == (val.horizon,)
    assert (model.alpha >= 0.0).all() and (model.alpha <= 1.0).all()


def test_cross_rag_keys_include_clearsky_profile():
    train, val, _ = _splits()
    model = build("cross_rag", backbone="persistence", top_k=4, max_datastore=1000)
    model.fit(train, val)
    t, h = train.history, train.horizon
    assert model._keys.shape[1] == t + h  # z-history ⊕ future clear-sky


def test_rag_empty_datastore_falls_back_to_backbone():
    train, val, test = _splits()
    model = build("ts_rag", backbone="persistence", top_k=4, max_datastore=1000)
    model.fit(train, val)
    model._keys = model._keys[:0]  # simulate empty datastore
    batch = test.batch(list(range(8)))
    point = model.predict(batch).point
    expected = build("persistence").predict(batch).point
    np.testing.assert_array_equal(point, expected)


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
