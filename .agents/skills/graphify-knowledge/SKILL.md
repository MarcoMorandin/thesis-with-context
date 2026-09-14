---
name: graphify-knowledge
description: Query or refresh the knowledge and literature graph only. Invoke as `$graphify-knowledge <question|update>`.
disable-model-invocation: true
---

Graphify owns project prose and papers under `knowledge/`; GitNexus owns code.

- For a question, run `graphify query "<question>"`, then use `explain` or `path` only if needed.
- For routine updates after intentional prose changes, run `graphify update knowledge/`.
- Never graphify the repository root, source code, or generated `knowledge/graphify-out/` artifacts.

Report the answer or update result concisely and identify the knowledge sources used.
