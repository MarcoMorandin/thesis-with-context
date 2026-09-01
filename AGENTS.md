# AGENTS.md — rules of record

Applies to every coding agent in this repo. **Rules only.** All *content* lives in
[`knowledge/`](knowledge/INDEX.md) — read it, do not restate it here.

## Mission

Build a research-grade **AI foundation model** for PV power forecasting. The primary metric
is **zero-shot cross-plant generalization** on disjoint test plants from a short history,
without sacrificing point-forecast quality.

This is an **AI science project**. PV is the testbed, not the subject. Target framing is
ICLR/NeurIPS: the contribution is multimodal foundation-model fusion + cross-plant
generalization, never PV engineering.

---

## 1. Read first

Do not answer an architecture, protocol, or data question from memory. Route it:

| Question | Go to |
|---|---|
| Anything about the project's prose | [`knowledge/INDEX.md`](knowledge/INDEX.md) — the routing table |
| How does this code work / what breaks if I change it | **GitNexus** MCP (`query`, `context`, `impact`) |
| What does paper X claim | `graphify query "<question>"` |
| What do the numbers say | `report/` + `baselines/results/ALL_RESULTS.md` |

The canonical-source table in `knowledge/INDEX.md` §2 says which file owns which fact.
**A fact has exactly one home.** If you need it elsewhere, link — never copy.

---

## 2. Non-negotiables

| Rule | Detail |
|---|---|
| **Python** | `uv` only. Never `pip`, `poetry`, or `conda`. `uv add` / `uv sync` for deps. |
| **Config** | Hydra only. No `argparse`, no `yaml.load` outside Hydra. Baseline configs self-contained per baseline. |
| **Files** | One class or one script capability per file. Target < 150 lines. |
| **Tests** | Every module has `test_<module>.py` with shape + gradient smoke tests. `uv run pytest` **before** claiming any fix works. |
| **Git** | Never work on `main`. Branch `exp/`, `feat/`, or `fix/`. Micro-commit per verified sub-task. Merge locally; push `main` only. |
| **Models** | Multimodal foundation models (TS FM + vision FM). No classical ML (XGBoost, LightGBM, scikit-learn) without explicit approval. |
| **Literature** | Prefer 2026, then late 2025. Nothing before 2025. |
| **Data** | Dataset of record `/leonardo_scratch/fast/IscrC_MTSFM/data_v2/` is **read-only**. Do not refactor data pipelines. Schema → `knowledge/dataset.md`. |

Detail behind these rules: [`knowledge/conventions.md`](knowledge/conventions.md).

---

## 3. Tool routing

Cheapest-correct first. Never start with grep.

| Need | Use | Not |
|---|---|---|
| Understand code, find a symbol, trace a flow | `gitnexus` MCP: `query` → `context` | grep, reading files |
| Blast radius before an edit | `gitnexus impact({target, direction:"upstream"})` | guessing |
| Literature, proposal, project prose | `graphify query` / `explain` / `path` | reading PDFs |
| Read a specific file | `ctx_read` (`mode=map` when context-only) | `cat` / `head` / `tail` |
| Search text | `ctx_search` | `grep` / `rg` |
| Shell output | `ctx_shell` | raw bash |
| Directory map | `ctx_tree` | `ls` / `find` |

Edits always use native `Edit` / `Write`. `ctx_edit` only when `Edit` needs a `Read` you
cannot do — never loop on a failing `Edit`.

**Two graphs, never crossed:** GitNexus = code (`.gitnexus/`). Graphify = prose + papers
(`knowledge/` → `knowledge/graphify-out/`). Never run Graphify over the repo root.

---

## 4. Workflow

**Session start** — `git status`; if on `main`, branch before touching anything.

**Editing code**
1. `gitnexus impact` on the target symbol. Report the blast radius; **warn on HIGH/CRITICAL**.
2. Edit ONE module + its test + its Hydra config.
3. `uv run pytest <the test>`, then the suite.
4. `gitnexus detect_changes()` — confirm only expected symbols moved.
5. `git diff HEAD`, strip debug prints, micro-commit.

**Running an experiment** — every experiment declares, before it launches:
1. **Hypothesis** — one sentence.
2. **Config diff** — a Hydra override under `MMTSFM/configs/` (or `baselines/configs/`).
3. **Registry row** — in [`knowledge/ablations.md`](knowledge/ablations.md).
4. **Baseline** — which standard baseline it is compared against, per
   [`knowledge/protocol.md`](knowledge/protocol.md).

Use `/register-experiment`; pre-flight with the `experiment-reviewer` agent.

**Evaluation splits** — `cross_plant` (disjoint held-out plants) is the primary metric.
`intra_plant` is a sanity check only, never a headline number.

---

## 5. Writing: knowledge/, report/, manuscript/

| Tree | Contains | Rule |
|---|---|---|
| `knowledge/` | project prose + papers | One topic per file. Adding a doc? First check `INDEX.md` — if the topic exists, **edit that file**. New file ⇒ new `INDEX.md` row. |
| `report/` | ongoing results: EDA, leaderboard synthesis | Numbers are **cited from** `baselines/results/ALL_RESULTS.md`, never retyped. State the run that produced them. |
| `manuscript/` | thesis LaTeX | Every claim traces to `report/` or a `knowledge/papers/` citation. Never invent a number here. |

Never hand-edit generated artifacts: `knowledge/graphify-out/`, `.gitnexus/`,
`baselines/results/ALL_RESULTS.md`, compiled PDFs.

---

## 6. Must not

- Introduce energy-domain physics heuristics (CSI conversion, irradiance physics) unless
  explicitly ablating them out.
- Add `scikit-learn`, `lightgbm`, or `xgboost` without explicit user approval.
- Create monolithic files with multiple classes.
- Modify `/leonardo_scratch/fast/IscrC_MTSFM/data_v2` (read-only).
- Commit data, checkpoints, logs, or large binaries.
- Report a checkpoint's score re-derived post-hoc — checkpoints are known not to reproduce
  their in-process numbers. In-process numbers are the record.
- Claim a fix works without running the tests and reading the output.

## 7. Agent skills

Configuration the `mattpocock-skills` engineering skills read. These files are a
contract for those skills; they restate nothing from `knowledge/` — they point at it.

### Issue tracker

Local markdown under `.scratch/<feature-slug>/`. No `gh` CLI in this repo, and the
GitHub remote is not used for issue tracking. `.scratch/` is working state, exempt from
§5 — promote anything that outlives its ticket into `knowledge/`.
See [`knowledge/agents/issue-tracker.md`](knowledge/agents/issue-tracker.md).

### Triage labels

The five canonical roles, each label string equal to its name. Recorded as a `Status:`
line in the ticket file, since the tracker is local markdown with nothing to label.
See [`knowledge/agents/triage-labels.md`](knowledge/agents/triage-labels.md).

### Domain docs

Single-context, rehomed into `knowledge/`: this repo has no `docs/` tree and no root
`CONTEXT.md` (§5). `knowledge/INDEX.md` stands in for the glossary and
`knowledge/specs/` for ADRs.
See [`knowledge/agents/domain.md`](knowledge/agents/domain.md).

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **thesis-with-context** (3253 symbols, 5716 relationships, 127 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/thesis-with-context/context` | Codebase overview, check index freshness |
| `gitnexus://repo/thesis-with-context/clusters` | All functional areas |
| `gitnexus://repo/thesis-with-context/processes` | All execution flows |
| `gitnexus://repo/thesis-with-context/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

## Graphify — prose & literature graph

Graphify indexes **`knowledge/` only** (project prose + `papers/`) into `knowledge/graphify-out/`.
It is **not** a code graph — for code, use GitNexus above.

```bash
graphify query   "<question>"     # start here; returns a scoped subgraph
graphify explain "<concept>"
graphify path    "<A>" "<B>"
graphify update  knowledge/       # AST-only, free — keeps the graph current
graphify knowledge/ --wiki        # full rebuild (costs API)
```

- `knowledge/graphify-out/wiki/index.md` for broad navigation; `knowledge/graphify-out/GRAPH_REPORT.md` only for a full
  literature sweep.
- **Never** `graphify update .` or `graphify .` — the repo root floods the prose graph with
  code files.
