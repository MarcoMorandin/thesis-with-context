# Vendored Tier-4 RAG baselines — provenance & licensing

This directory holds **unmodified copies** of two upstream research repositories so
the Tier-4 retrieval-augmented baselines run the authors' *original code* rather than
a reimplementation (per BASELINE_COMPARISON.md §1, Tier 4). Images and `.git` were
stripped; no source files were edited. Integration lives in
`baselines/tier4/rag_original.py` + `docs/experiments/TIER4_RAG_INTEGRATION.md`; the
lightweight α-mix in `baselines/tier4/rag.py` is retained as a dependency-free
fallback / contract-test backbone.

| Vendor dir | Upstream | Commit SHA | License |
|---|---|---|---|
| `ts_rag/` | https://github.com/UConn-DSIS/TS-RAG | `73ac807789d2e61b8a3dfc8514e3fc947fe185cc` | **MIT** (`ts_rag/LICENSE`), NeurIPS 2025, arXiv:2503.07649 |
| `cross_rag/` | https://github.com/seunghan96/cross-rag | `b9a5428365b8ada43a986b2501ece12dd3844e95` | **No license file stated upstream** ⚠️ — see caveat below; arXiv:2603.14709 |

## Licensing caveats

- **TS-RAG** is MIT-licensed; redistribution here is compliant. Keep `ts_rag/LICENSE`.
- **Cross-RAG** ships **no `LICENSE` file**. It also states it "builds upon TS-RAG"
  and bundles Amazon's Chronos-Bolt pipeline code (`cross_rag/cross-rag/models/base.py`,
  `models/utils.py`; authored by Amazon, originally **Apache-2.0**). Vendoring it here
  is a *convenience copy for research reproduction only*. Before any public release of
  this thesis repo:
  1. open an issue / email the Cross-RAG authors (seunghan.lee@lgresearch.ai) to confirm
     a license, **or**
  2. replace the vendored copy with a git submodule (no redistribution), **or**
  3. drop it and cite-only.
  Until then treat `cross_rag/` as third-party code under its authors' rights, not ours.

## What is NOT vendored (download separately, not in git)

Both methods need artifacts hosted off-repo (multi-GB; cluster only):
- Chronos-Bolt base weights (HF `amazon/chronos-bolt-*`).
- Pretrained ARM / cross-attention checkpoints (Google Drive folder in each README;
  TS-RAG also on HF `nkh/TS-RAG-Data`).
- Their preprocessed retrieval databases (we build ours from uk_pv instead — see the
  integration doc).

These are excluded by `baselines/tier4/vendor/.gitignore`.
