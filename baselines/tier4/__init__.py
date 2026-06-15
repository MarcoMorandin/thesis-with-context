from . import cora  # noqa: F401 — registers the tier-4 CoRA baseline

# TS-RAG / Cross-RAG are GPU/cluster-only: they run from the authors' original
# vendored code (tier4/vendor/, see docs/experiments/TIER4_RAG_INTEGRATION.md),
# not as in-process registry baselines. No `ts_rag` / `cross_rag` here by design.
