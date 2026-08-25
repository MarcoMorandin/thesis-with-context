# knowledge/ — single source of truth

Everything the project knows in prose: scope, design, protocol, results-interpretation,
and the literature corpus. **One topic, one file.** If a fact appears twice, one of the two
is wrong — fix the canonical file and link to it.

Rules and non-negotiables live in [`../AGENTS.md`](../AGENTS.md); this tree is the *content*
those rules point at.

---

## 1. Routing — where to look

| Question | File |
|---|---|
| What is the research question? What is in / out of scope? Which hypothesis? | [scope.md](scope.md) |
| What is the model, and why is each piece there? | [architecture.md](architecture.md) |
| The full scientific argument, problem setup, math | [proposal.md](proposal.md) |
| What data exists? Schema, splits, canonical batch dict | [dataset.md](dataset.md) |
| What counts as a fair comparison? Windows, metrics, splits | [protocol.md](protocol.md) |
| Which baselines, in which tier, and what does each paper claim? | [baselines.md](baselines.md) |
| How do I run / train / submit anything? | [runbook.md](runbook.md) |
| Where does code go? Naming, Hydra, git | [conventions.md](conventions.md) |
| Which ablations exist and what is their status? | [ablations.md](ablations.md) |
| Dated design decisions, superseded designs | [specs/](specs/) |
| Where do issues live? How do agent skills read/write tickets? | [agents/issue-tracker.md](agents/issue-tracker.md) |
| How should agent skills consume domain docs? | [agents/domain.md](agents/domain.md) |
| The papers themselves | [papers/](papers/) — query via Graphify, don't read PDFs |

Outside this tree:

| Question | Where |
|---|---|
| What do the numbers actually say? | [`../report/`](../report/) — `REPORT.md` (dataset EDA), `BASELINE_TEST_REPORT.md` (leaderboard synthesis) |
| Live baseline numbers | `../baselines/results/ALL_RESULTS.md` — **generated**, never hand-edit |
| The thesis text | [`../manuscript/`](../manuscript/) |
| How does this code work? What breaks if I change it? | **GitNexus** — never grep first |

---

## 2. Canonical sources — do not duplicate

Facts that exist in exactly one place. Anything else citing them must **link**, not copy.

| Fact | Canonical home |
|---|---|
| Fairness rules, history/horizon windows, metric definitions | `protocol.md` |
| Dataset schema, plant splits, batch dict | `dataset.md` |
| Model structure, shapes, component→file map | `architecture.md` |
| Live baseline numbers | `baselines/results/ALL_RESULTS.md` (generated) |
| Results interpretation | `report/BASELINE_TEST_REPORT.md` |
| Ablation status | `ablations.md` |
| Code layout, naming, git rules | `conventions.md` |
| Agent rules and non-negotiables | `../AGENTS.md` |

---

## 3. The two graphs — never crossed

| Tool | Domain | Index | Refresh |
|---|---|---|---|
| **GitNexus** | **Code** — call chains, blast radius, "how does X work" | `.gitnexus/` | `node .gitnexus/run.cjs analyze` |
| **Graphify** | **Prose + papers** — this whole `knowledge/` tree | `knowledge/graphify-out/` | `graphify update knowledge/` |

```bash
graphify query   "<question>"     # scoped subgraph — start here for prose/literature
graphify explain "<concept>"
graphify path    "<A>" "<B>"
graphify knowledge/ --wiki        # full rebuild (costs API); update is AST-only and free
```

**Never** run `graphify` over the repo root — it floods the prose graph with thousands of
code files. Graphify's input is always `knowledge/`.

---

## 4. Context budgeting

The point of this tree is that an agent reads *less*, not more.

1. **Point, don't paste.** Link `architecture.md`; never inline the proposal into a prompt.
2. **Graph first.** GitNexus / Graphify return scoped subgraphs orders of magnitude smaller
   than grep output or a full file.
3. **Registry, not logs.** `ablations.md` records what ran; an agent should never re-read
   training logs to reconstruct history.
4. **lean-ctx for I/O.** `ctx_read` caches — re-reads cost ~13 tokens. Prefer it over
   re-reading large files, and use `mode=map` for context-only files.
5. **Exclude noise.** Artifacts, checkpoints, `logs/`, `knowledge/graphify-out/`, `.gitnexus/` are
   gitignored and must never be indexed.

## 5. Anti-patterns

| Anti-pattern | Fix |
|---|---|
| `graphify update .` over the repo root | `graphify update knowledge/` |
| Graphify for code questions | GitNexus — Graphify is prose/papers only |
| Copying `AGENTS.md` content into `CLAUDE.md` | `CLAUDE.md` holds Claude-specific deltas only |
| Restating protocol numbers in a new doc | Link `protocol.md` |
| Pasting result numbers into prose by hand | Link `baselines/results/ALL_RESULTS.md` |
| A new doc for a topic that already has one | Edit the existing file |
| Undocumented ablation | `ablations.md` row + `/register-experiment` |
| Data or checkpoints in the repo | External path + `data_dir` config |
