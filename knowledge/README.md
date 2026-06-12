# Knowledge Corpus (Graphify input)

Point Graphify at this folder to build a research knowledge graph.

```bash
/graphify knowledge/ --wiki --update
```

## Contents (symlinks or copies)

| Subfolder | Source |
|-----------|--------|
| `papers/baselines/` | `../baselines/*.pdf` |
| `papers/related/` | `../solar-related-work/*.pdf` |
| `docs/` | `../MMTSFM/*.md`, `../docs/` |
| `notes/` | Your experiment notes (create as you go) |

Graphify tags edges as EXTRACTED / INFERRED / AMBIGUOUS  -  use for literature, not code deps.
For code structure, use **GitNexus** on the repo root instead.
