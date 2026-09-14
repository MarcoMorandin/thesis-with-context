---
name: graphify
description: Query or refresh this repository's prose and literature graph. Use for project knowledge or paper questions, never source-code exploration.
---

# Graphify for this repository

Graphify owns `knowledge/` only. GitNexus owns code.

- For a question, run `graphify query "<question>"`; use `explain` or `path` only when needed.
- After intentional prose changes, run `graphify update knowledge/`.
- Never run Graphify on the repository root, source code, or generated graph artifacts.
- Use `uv` for dependency work; do not install packages from this skill.

Return a concise answer or update result and identify the knowledge sources used.
