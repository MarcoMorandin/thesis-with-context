# CLAUDE.md — Claude Code deltas

**[AGENTS.md](AGENTS.md) is the rules of record — read it first.** This file holds only what
is specific to the Claude Code harness. Nothing here restates AGENTS.md or `knowledge/`.

## Skills & agents in this repo

| Invoke | For |
|---|---|
| `/register-experiment` | Register an ablation: `knowledge/ablations.md` row + config diff + `exp/` branch |
| `/new-baseline` | Scaffold a tier baseline: dir, config, SLURM script, registry stub |
| `/graphify` | Rebuild or query the prose/literature graph over `knowledge/` |
| `experiment-reviewer` (agent) | Pre-flight an ablation for protocol compliance **before** it runs |
| `result-aggregator` (agent) | Validate `baselines/results/*.json` before any number is trusted |
| `slurm-log-triager` (agent) | Classify a failed SLURM job → minimal fix |
| `gitnexus-*` skills | Explore / impact / debug / refactor via the code graph |

Plugins enabled: `superpowers` (plans, TDD, verification), `academic-research-skills`
(paper pipeline — useful for `manuscript/`), `caveman`, `last30days`.

## Hooks active (`.claude/settings.json`)

Guardrails run automatically; you do not invoke them. If one blocks you it is enforcing an
AGENTS.md rule — fix the call, do not work around it.

| When | Hook | Enforces |
|---|---|---|
| SessionStart | `session-brief.py` | branch, graph staleness, results-vs-report drift, routing rule |
| Before `Bash` | `guard-uv-only.py` | `uv` only — blocks bare `pip` / `python` / `conda` |
| Before `Edit`/`Write` | `guard-data-readonly.py` | dataset of record is read-only |
| Before `Edit`/`Write` | `guard-generated-files.py` | no hand-editing `graphify-out/`, `.gitnexus/`, `ALL_RESULTS.md`, compiled PDFs |
| Before `Read`/`Glob`/`Grep` | `route-exploration.py` | GitNexus / Graphify before raw grep |
| After `Write` | `check-new-file-size.py` | < 150-line target for new `MMTSFM/src` files |
| After `Write` | `check-knowledge-index.py` | every `knowledge/` doc has an `INDEX.md` row; no stray prose |
| After `Edit`/`Write` on `*.py` | `ruff format` | formatting |
| On stop | gitnexus / graphify refresh | both graphs stay current |

## Commands

```bash
uv run pytest                                     # from MMTSFM/ or baselines/
node .gitnexus/run.cjs analyze                    # refresh code graph
graphify update knowledge/                        # refresh prose graph (never `.`)
```

Known local quirk: `baselines/tests/test_baseline_contract.py` segfaults on this macOS
box (duplicate libomp, pre-existing). Run the rest with
`--ignore=tests/test_baseline_contract.py`.

Everything else — training, SLURM, curriculum — is in
[`knowledge/runbook.md`](knowledge/runbook.md).

## Google Workspace (`gws` CLI)

```bash
gws sheets read   --id <id> --range <range>
gws sheets append --id <id> --range <range> --values "v1,v2,v3"
gws gmail  list   --query <query> --limit <n>
gws gmail  send   --to <email> --subject <subject> --body <body>
gws docs   create --title <title> --content <text>
gws drive  upload --file <path> --folder <folder_id>
```

JSON output unless `--format plain`. Sheet/Doc IDs come from the URL.

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
