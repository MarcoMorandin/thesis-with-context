"""Vendored original TS-RAG / Cross-RAG code + the uk_pv bridge utilities.

The upstream packages (``ts_rag/``, ``cross_rag/``) are unmodified third-party
code run on the cluster from their own env; they are NOT imported here. Only the
in-repo helpers (``export_ukpv``, ``contract_check``) are importable. See
``VENDOR_NOTICE.md`` and ``docs/experiments/TIER4_RAG_INTEGRATION.md``.
"""
