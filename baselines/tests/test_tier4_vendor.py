"""Provenance checks for the vendored original TS-RAG / Cross-RAG code.

These do not import the upstream modules (they need chronos-forecasting + faiss +
numpy 1.25, a separate env — see docs/experiments/TIER4_RAG_INTEGRATION.md). They
only assert the unmodified sources and their provenance/licensing notice are present,
so the "use original code" requirement can't silently rot.
"""

from __future__ import annotations

from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "tier4" / "vendor"

# upstream commit SHAs recorded in VENDOR_NOTICE.md (update both together)
TS_RAG_SHA = "73ac807789d2e61b8a3dfc8514e3fc947fe185cc"
CROSS_RAG_SHA = "b9a5428365b8ada43a986b2501ece12dd3844e95"


def test_vendor_notice_records_shas():
    notice = (VENDOR / "VENDOR_NOTICE.md").read_text()
    assert TS_RAG_SHA in notice
    assert CROSS_RAG_SHA in notice
    # Cross-RAG license caveat must stay visible
    assert "No license file stated upstream" in notice


def test_ts_rag_original_sources_present():
    base = VENDOR / "ts_rag" / "TS-RAG"
    for rel in ("models/ChronosBolt.py", "retrieve.py", "dataset.py",
                "zeroshot.py", "pretrain.py"):
        assert (base / rel).is_file(), f"missing vendored TS-RAG file: {rel}"
    assert (VENDOR / "ts_rag" / "LICENSE").is_file()  # MIT, keep it


def test_cross_rag_original_sources_present():
    base = VENDOR / "cross_rag" / "cross-rag"
    for rel in ("models/CrossRAG.py", "models/base.py", "zeroshot.py",
                "retrieve_X.py"):
        assert (base / rel).is_file(), f"missing vendored Cross-RAG file: {rel}"


def test_integration_doc_exists():
    doc = (VENDOR.parents[2] / "docs" / "experiments"
           / "TIER4_RAG_INTEGRATION.md")
    assert doc.is_file()
    text = doc.read_text()
    assert "ts_rag_orig" in text and "ts_rag_proto" in text


# ---- data bridge + baseline-contract checks --------------------------------

import numpy as np  # noqa: E402

from tier4.vendor import contract_check  # noqa: E402
from tier4.vendor.export_ukpv import _grid_frame  # noqa: E402

from .conftest import make_frame  # noqa: E402


def test_export_grid_frame_is_dense_and_in_range():
    df = make_frame(n_sites=3, days=4, nan_fraction=0.1)
    df["site_id"] = df["site_id"].astype(str)
    wide = _grid_frame(df, ["site_0", "site_2"])
    assert list(wide.columns) == ["site_0", "site_2"]      # committed order, dense
    assert not wide.isna().any().any()                      # gaps filled
    step = wide.index.to_series().diff().dropna().dt.total_seconds().unique()
    assert step.tolist() == [1800.0]                        # 30-min grid
    assert wide.to_numpy().min() >= 0.0


def test_contract_check_inputs(tmp_path):
    import pandas as pd
    dates = pd.date_range("2021-06-01", periods=48, freq="30min", tz="UTC")
    good = pd.DataFrame({"date": dates, "OT": np.linspace(0, 1, 48)})
    good.to_csv(tmp_path / "uk_pv_test_x.csv", index=False)
    (tmp_path / "manifest.json").write_text("{}")
    assert contract_check.check_inputs(tmp_path) == []      # clean

    bad = pd.DataFrame({"date": dates, "OT": np.linspace(0, 2, 48)})  # >1
    bad.to_csv(tmp_path / "uk_pv_test_bad.csv", index=False)
    assert any("outside [0,1]" in e for e in contract_check.check_inputs(tmp_path))


def test_contract_check_predictions(tmp_path):
    ok = tmp_path / "ok.npz"
    np.savez(ok, pred=np.random.rand(20, 12).astype("float32"))
    assert contract_check.check_predictions(ok, horizon=12) == []

    bad = tmp_path / "bad.npz"
    np.savez(bad, pred=(np.random.rand(20, 8) * 2).astype("float32"))
    errs = contract_check.check_predictions(bad, horizon=12)
    assert any("horizon" in e for e in errs)
    assert any("outside [0,1]" in e for e in errs)
