# Knowledge Corpus (Graphify input)

Point Graphify at this folder to build a research knowledge graph.

```bash
/graphify knowledge/ --wiki --update
```

## Contents (symlinks or copies)

| Subfolder | Source |
|-----------|--------|
| `papers/baselines/` | Reorganized PDF files of baselines (tracked directly) |
| `papers/related/` | Reorganized PDF files of related work (tracked directly) |
| `docs/` | `../MMTSFM/*.md`, `../docs/` |
| `notes/` | Your experiment notes (create as you go) |

Graphify tags edges as EXTRACTED / INFERRED / AMBIGUOUS  -  use for literature, not code deps.
For code structure, use **GitNexus** on the repo root instead.
