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
