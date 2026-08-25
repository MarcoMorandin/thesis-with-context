# Domain docs — consumer rules

**Canonical for**: how the `mattpocock-skills` engineering skills consume this repo's
domain documentation. Where issues live is in [issue-tracker.md](issue-tracker.md).

Layout is **single-context** — no monorepo, no `package.json`, one codebase.

## Before exploring, read these

This repo has **no `docs/` tree and no root `CONTEXT.md`**. `docs/` was merged into
`knowledge/` on 2026-08-04 and `knowledge/` is the single source of truth
([../../AGENTS.md](../../AGENTS.md) §5). The equivalents are:

| Skill expects | Read instead |
|---|---|
| `CONTEXT.md` — glossary, domain terms | [../INDEX.md](../INDEX.md) → routes to `scope.md`, `architecture.md`, `protocol.md`, `dataset.md` |
| `docs/adr/` — architecture decisions | [../specs/](../specs/) — dated design decisions and superseded designs |
| `CONTEXT-MAP.md` | n/a — single-context repo |

If a file doesn't exist, **proceed silently**. Don't flag its absence; don't suggest
creating it upfront. `/domain-modeling` creates them lazily when terms or decisions
actually get resolved — and when it does, they go under `knowledge/` with a row in
`INDEX.md`, never at the repo root.

## File structure

```
knowledge/
├── INDEX.md          ← routing table; read first (stands in for CONTEXT.md)
├── scope.md          ← research question, hypothesis ladder
├── architecture.md   ← what the model is, why each piece exists
├── protocol.md       ← fairness rules, metrics, splits
├── dataset.md        ← schema, splits, canonical batch dict
├── specs/            ← dated design decisions (stands in for docs/adr/)
└── agents/           ← this file + issue-tracker.md
```

## Use the project's vocabulary

When your output names a domain concept — an issue title, a refactor proposal, a
hypothesis, a test name — use the term as `knowledge/` defines it. Don't drift to
synonyms: `origin` not `timestamp`, `plant` not `station`, `skill score` not
`improvement`, and `ramp` only in the top-decile-|Δy| sense of
[../protocol.md](../protocol.md).

If the concept isn't in `knowledge/` yet, that's a signal: either you're inventing
language the project doesn't use (reconsider), or there's a real gap (note it for
`/domain-modeling`).

## Flag spec conflicts

If your output contradicts a spec in [../specs/](../specs/), surface it explicitly
rather than silently overriding:

> _Contradicts `2026-08-19-visual-fusion-diagnosis.md` §1, but worth reopening because…_

Specs are dated and supersedable. Say which one and why, and add a run-log row rather
than editing history.
