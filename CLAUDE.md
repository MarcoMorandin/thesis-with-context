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
