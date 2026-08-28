"""Ticket 15 — the horizon-attention diagnostic must tell a degenerate
parameterisation apart from a genuine null.

The point of these tests is narrow and specific: if the three lead-time queries
collapse into copies of each other, s2c produces a flat ramp metric that looks
*exactly* like a falsified hypothesis. Only this diagnostic separates the two, so
the property under test is the verdict, not the arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmtsfm.models.chronos2.horizon_attention import (  # noqa: E402
    MIN_BETWEEN_L1,
    SEPARATION_THRESHOLD,
    HorizonAttentionAccumulator,
)

N_KV = 64
N_TAU = 3


def _rows(centres, rows=400, sharp=6.0, noise=0.02, seed=0):
    """Attention rows peaked at ``centres[tau]`` over the 64-token field."""
    rng = np.random.default_rng(seed)
    grid = np.arange(N_KV)[None, :]
    logits = -(((grid - np.asarray(centres)[:, None]) / sharp) ** 2)  # [tau, kv]
    base = np.exp(logits)
    base /= base.sum(axis=-1, keepdims=True)
    out = base[None, :, :] + rng.normal(0, noise / N_KV, size=(rows, N_TAU, N_KV))
    out = np.clip(out, 1e-9, None)
    out /= out.sum(axis=-1, keepdims=True)
    return out[None]  # [1 block, rows, tau, kv]


def _run(attn, chunk=50):
    acc = HorizonAttentionAccumulator(n_blocks=1, n_tau=N_TAU, n_kv=N_KV, seed=7)
    for i in range(0, attn.shape[1], chunk):
        acc.update(attn[:, i : i + chunk])
    return acc


def test_distinct_queries_are_called_distinct():
    rep = _run(_rows([8, 32, 56])).report()
    assert rep["verdict"] == "queries_differ"
    assert rep["separation_ratio"] > SEPARATION_THRESHOLD
    assert rep["between_l1_mean"] > MIN_BETWEEN_L1


def test_collapsed_queries_are_called_degenerate_not_null():
    # Three queries attending the same place with independent sampling noise: the
    # ramp metric would be flat, and WITHOUT this the flatness reads as evidence
    # against the hypothesis rather than as a broken parameterisation.
    rep = _run(_rows([32, 32, 32], seed=1)).report()
    assert rep["verdict"] == "degenerate_queries_collapsed"
    assert rep["separation_ratio"] < SEPARATION_THRESHOLD


def test_the_noise_floor_is_measured_not_assumed():
    # Same collapsed field at two noise levels. A fixed absolute threshold would
    # flip its verdict as noise grows; the measured within-tau floor must not.
    quiet = _run(_rows([32, 32, 32], noise=0.02, seed=2)).report()
    loud = _run(_rows([32, 32, 32], noise=0.40, seed=3)).report()
    assert quiet["verdict"] == loud["verdict"] == "degenerate_queries_collapsed"
    assert loud["within_l1_noise_floor"] > quiet["within_l1_noise_floor"]


def test_a_tiny_shift_is_not_promoted_by_a_large_sample():
    # Separation must be about the SIZE of the difference, not the sample count.
    # 4000 rows shrink the measured noise floor until the ratio alone calls a
    # fraction-of-a-token shift "different"; MIN_BETWEEN_L1 is the guard that
    # stops a statistically clean but physically meaningless difference from
    # being reported as horizon-specific attention.
    rep = _run(_rows([32, 32.1, 32.2], rows=4000, seed=4)).report()
    assert rep["separation_ratio"] > SEPARATION_THRESHOLD
    assert rep["between_l1_mean"] < MIN_BETWEEN_L1
    assert rep["verdict"] == "degenerate_queries_collapsed"


def test_no_rows_is_reported_as_not_measured():
    rep = HorizonAttentionAccumulator(1, N_TAU, N_KV).report()
    assert rep["verdict"] == "not_measured" and rep["n_rows"] == 0


def test_entropy_is_reported_against_a_uniform_reference():
    rep = _run(_rows([8, 32, 56])).report()
    ent = rep["per_block"][0]["entropy_nats_per_tau"]
    assert len(ent) == N_TAU
    # A peaked distribution must sit below uniform; the reference is emitted so a
    # reader can tell "attends everywhere" from "attends one place".
    assert max(ent) < rep["uniform_entropy_nats"]
    assert rep["uniform_entropy_nats"] == pytest.approx(np.log(N_KV))


def test_tau_embedding_distances_ride_along_but_do_not_decide():
    # Identical embeddings with genuinely different attention: the verdict comes
    # from the attention, and the embedding block is reported, not consulted.
    emb = np.ones((N_TAU, 8), dtype=np.float64)
    rep = _run(_rows([8, 32, 56])).report(tau_embed=emb)
    assert rep["verdict"] == "queries_differ"
    assert rep["tau_embedding"]["0_vs_1"]["l2"] == pytest.approx(0.0)


def test_the_half_split_ignores_batch_boundaries():
    # The test loader is not shuffled, so a batch-parity split would tie the two
    # halves to site order. Same rows, different chunking => same noise floor.
    attn = _rows([8, 32, 56], seed=5)
    a = _run(attn, chunk=1).report()["within_l1_noise_floor"]
    b = _run(attn, chunk=137).report()["within_l1_noise_floor"]
    assert a == pytest.approx(b, rel=0.25)


def test_extra_diagnostics_survive_the_double_finalize():
    """The exact trap this plumbing exists to avoid: ``write()`` calls
    ``finalize()`` a SECOND time internally, so a diagnostic injected into the
    dict ``finalize()`` returned would never reach the JSON on disk."""
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=12, reference_path="/nonexistent/ref.json")
    ev.extra["horizon_attention"] = {"verdict": "queries_differ"}
    for _ in range(2):
        assert ev.finalize()["horizon_attention"]["verdict"] == "queries_differ"


def test_extra_never_shadows_a_protocol_metric():
    from eval.protocol_eval import ProtocolEvaluator

    ev = ProtocolEvaluator(horizon=12, reference_path="/nonexistent/ref.json")
    ev.extra["overall"] = {"nmae": -1.0}
    # setdefault, not assignment: a diagnostic is evidence ABOUT the run and must
    # never be able to overwrite the run's own metrics.
    assert ev.finalize()["overall"].get("nmae") != -1.0


def test_the_lightning_hooks_harvest_the_right_rows_and_positions():
    """Exercises the wiring without a Trainer. The slicing is the part most
    likely to be silently wrong: covariate rows are appended along the BATCH axis
    and vision-off rows sit inside the target block, so taking either would fold
    attention that never gated a residual into the distribution."""
    import types

    from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule

    blocks = [types.SimpleNamespace(capture_visual_attn=True) for _ in range(2)]
    rows, heads, seq, n_kv, t_fut = 4, 2, 8, 16, 3
    torch.manual_seed(0)
    for b in blocks:
        b.last_visual_attn = torch.softmax(torch.randn(rows, heads, seq, n_kv), -1)
    ev = types.SimpleNamespace(extra={})
    lm = types.SimpleNamespace(
        _horizon_attn=None,
        _horizon_attn_blocks=blocks,
        _num_output_patches=t_fut,
        _protocol_eval=ev,
        # rows 2 and 3 are covariate rows; row 1 carried no vision this batch.
        model=types.SimpleNamespace(
            _last_visual_active=torch.tensor([True, False]),
            lead_time_embed=torch.nn.Parameter(torch.randn(t_fut, 8)),
        ),
    )
    VisionChronos2LightningModule._capture_horizon_attention(lm)
    assert lm._horizon_attn.n_rows == 1, "harvested a masked or covariate row"
    assert lm._horizon_attn._sum.shape == (2, t_fut, n_kv)
    # Cleared, so a second call cannot double-count the same batch.
    assert all(b.last_visual_attn is None for b in blocks)

    VisionChronos2LightningModule._emit_horizon_attention(lm)
    rep = ev.extra["horizon_attention"]
    # One row cannot support a half-split, and the report must say so rather than
    # emit a separation ratio computed against a noise floor that does not exist.
    assert rep["verdict"] == "inconclusive_too_few_rows"
    assert all(not b.capture_visual_attn for b in blocks)


def test_an_arm_without_cross_attention_emits_nothing_at_all():
    import types

    from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule

    ev = types.SimpleNamespace(extra={})
    lm = types.SimpleNamespace(
        _horizon_attn=None, _horizon_attn_blocks=[], _protocol_eval=ev
    )
    VisionChronos2LightningModule._emit_horizon_attention(lm)
    assert ev.extra == {}, "s1/s2a/s2b results JSONs must be byte-identical"


def test_a_present_but_never_fired_diagnostic_says_not_measured():
    """Distinct from the previous case on purpose. A missing key means the arm
    has no cross-attention; ``not_measured`` means it had it and produced nothing,
    which is a bug report, not a null result."""
    import types

    from mmtsfm.models.chronos2.lightning_module import VisionChronos2LightningModule

    ev = types.SimpleNamespace(extra={})
    lm = types.SimpleNamespace(
        _horizon_attn=None,
        _horizon_attn_blocks=[types.SimpleNamespace(capture_visual_attn=True)],
        _protocol_eval=ev,
    )
    VisionChronos2LightningModule._emit_horizon_attention(lm)
    assert ev.extra["horizon_attention"]["verdict"] == "not_measured"


def test_capture_is_off_by_default_and_adds_no_state_dict_keys():
    from test_s2c_future_query import _block_cfg

    from mmtsfm.models.chronos2.model import Chronos2EncoderBlock

    blk = Chronos2EncoderBlock(_block_cfg(num_layers=2, k=2), block_idx=1, num_layers=2)
    assert blk.visual_cross_attn is not None
    assert blk.capture_visual_attn is False
    assert blk.last_visual_attn is None
    # Plain attributes, not buffers: an s2c checkpoint must stay loadable by any
    # arm that does not run the diagnostic.
    keys = list(blk.state_dict().keys())
    assert not [k for k in keys if "capture_visual_attn" in k or "last_visual" in k]


def test_capture_records_a_distribution_over_the_visual_field():
    from test_s2c_future_query import _block_cfg, _block_inputs

    from mmtsfm.models.chronos2.model import Chronos2EncoderBlock

    cfg = _block_cfg(num_layers=2, k=2)
    blk = Chronos2EncoderBlock(cfg, block_idx=1, num_layers=2).eval()
    blk.capture_visual_attn = True
    seq, n_kv = 8, 16
    kwargs, kv = _block_inputs(batch=2, seq=seq, n_kv=n_kv)
    qmask = torch.zeros(2, seq, dtype=torch.bool)
    qmask[:, -3:] = True
    with torch.no_grad():
        blk(**kwargs, visual_kv=kv, visual_query_mask=qmask)
    a = blk.last_visual_attn
    assert a is not None and a.shape == (2, cfg.num_heads, seq, n_kv)
    # Softmax over the visual field: every query row sums to 1 across the KV
    # tokens, which is what makes a KL between two tau rows meaningful at all.
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)


def test_capture_stays_off_unless_asked():
    """The default path must not pay for the diagnostic. Capture forces eager
    attention; leaving it armed would slow every future arm and silently retain a
    tensor per block."""
    from test_s2c_future_query import _block_cfg, _block_inputs

    from mmtsfm.models.chronos2.model import Chronos2EncoderBlock

    blk = Chronos2EncoderBlock(_block_cfg(num_layers=2, k=2), block_idx=1, num_layers=2)
    blk.eval()
    kwargs, kv = _block_inputs(batch=2, seq=8, n_kv=16)
    qmask = torch.zeros(2, 8, dtype=torch.bool)
    qmask[:, -3:] = True
    with torch.no_grad():
        blk(**kwargs, visual_kv=kv, visual_query_mask=qmask)
    assert blk.last_visual_attn is None
