# Context Engineering Guide (2026)

A guide for building AI-agent-native research codebases. Tailored to this repo (MMTSFM / PV forecasting).

## What is context engineering?

The deliberate design of **what an AI agent sees, when, and in what order** — so it works reliably on complex research code without hallucinating architecture or breaking dependencies. Not prompt engineering: **repository design** — files, rules, graphs, and workflows that compound across sessions.

## The 2026 stack (four layers)

```
+---------------------------------------------------------+
|  Layer 4: Skills (superpowers, gitnexus suite, ARS, ...) |
+---------------------------------------------------------+
|  Layer 3: Knowledge graphs (GitNexus=code, Graphify=lit)|
+---------------------------------------------------------+
|  Layer 2: Rules (AGENTS.md = source of truth, CLAUDE.md) |
+---------------------------------------------------------+
|  Layer 1: Code structure (modular, short, tested)       |
+---------------------------------------------------------+
```

### Layer 1 — Code structure

Agents read files; they do not hold your architecture in memory.

| Practice | Why |
|----------|-----|
| One class per file | Agent edits one concern; tests map 1:1 |
| Files <150 lines | Fits context; forces decomposition |
| Predictable naming | Agent guesses correct path without search |
| Hydra configs (`MMTSFM/configs/`) | Behavior change without touching code |
| Tests per module (`MMTSFM/tests/`) | Verify edits without a full training run |

**Real layout:**
- `MMTSFM/src/mmtsfm/` — main package (`mmtsfm`): `data/`, `models/{base.py,chronos2/,vision/}`, `train.py`. Run `uv run python -m mmtsfm.train`.
- `MMTSFM/{configs,tests,scripts}/` — Hydra configs, tests, SLURM scripts.
- `baselines/` — `tier0..tier6`, `run_eval.py` (canonical runner), `scripts/`, `results/`, `common/`. Vendored third-party baselines under `baselines/tier6/vendor/` (e.g. `solar_vlm/`, `crossvivit/`, `sunset/`) — excluded from the code graph.
- `knowledge/` — literature corpus: `papers/` (PDFs), `docs/`, `proposal`. **Graphify domain.**
- `docs/` — `architecture/`, `context/`, `experiments/` (ABLATION_REGISTRY, BASELINE_PROTOCOL).

### Layer 2 — Rules and entry docs

| File | Role |
|------|------|
| `AGENTS.md` | **Single source of truth**: mission, non-negotiables, conventions, testing, ablation, git. |
| `CLAUDE.md` | Claude-specific deltas only: commands, tool routing, GWS. Points to AGENTS.md. (Guard-protected — edit manually.) |

**Rule of thumb**: if you explain it twice in chat, put it in `AGENTS.md`. Do not duplicate AGENTS.md content into CLAUDE.md.

### Layer 3 — Knowledge graphs (strict split)

Two graphs, **never crossed**:

| Tool | Domain | Index | Refresh |
|------|--------|-------|---------|
| **GitNexus** | **Code** — call chains, blast radius, "how does X work" | `.gitnexus/` (~158 files, no vendor) | `node .gitnexus/run.cjs analyze` (Stop hook auto-runs on `.py` edits) |
| **Graphify** | **Literature** — `knowledge/` papers + proposal | `graphify-out/` | `graphify update knowledge/` (cheap) / `graphify knowledge/ --wiki` (full rebuild) |

```bash
# Code graph
node .gitnexus/run.cjs analyze

# Literature graph — knowledge/ ONLY, never repo root
graphify knowledge/ --wiki
```

**Critical anti-pattern:** never `graphify update .` / `graphify .` over the repo root — it pollutes the literature graph with thousands of code files (the misrouting bug). Graphify input is always `knowledge/`.

### Layer 4 — Skills & agents

Custom (this repo, in `.claude/`):

| Asset | Use |
|-------|-----|
| `/new-baseline` | Scaffold a tier baseline (dir, config, SLURM, registry stub) |
| `/register-experiment` | Register an ablation (registry row + config diff + `exp/` branch) |
| `experiment-reviewer` (agent) | Pre-flight an ablation for protocol compliance |
| `result-aggregator` (agent) | Validate baseline result JSONs before trusting numbers |
| `slurm-log-triager` (agent) | Classify failed SLURM jobs → minimal fix |
| `gitnexus-*` skills | Explore / impact / debug / refactor via the code graph |

Plugins: `superpowers` (plans/TDD/verification), `academic-research-skills` (paper pipeline), `caveman` (dense status), `lean-ctx` (token-efficient I/O).

## Session workflow for Claude Code

```
1. Read AGENTS.md (+ docs/context/RESEARCH_SCOPE.md)
2. Code Q?  -> gitnexus query/context/impact on the target module
   Lit  Q?  -> graphify query "<question>"
3. Branch: exp/<name>
4. Edit ONE module + matching test + Hydra config
5. uv run pytest MMTSFM/tests/test_<module>.py
6. git commit  (Stop hook refreshes the code graph)
7. (optional) uv run python -m mmtsfm.train  for integration
```

## Context budgeting

1. **Point, don't paste** — reference `docs/architecture/OVERVIEW.md`, not the raw proposal.
2. **Registry pattern** — `docs/experiments/ABLATION_REGISTRY.md` tracks experiments; agent does not re-read all logs.
3. **Graph-first** — gitnexus/graphify return scoped subgraphs far smaller than raw grep.
4. **lean-ctx** — `ctx_read` caches; prefer over re-reading large files.
5. **Exclude noise** — `.gitignore` artifacts; never index `logs/`, checkpoints, `graphify-out/`, `.gitnexus/`.

## Anti-patterns

| Anti-pattern | Fix |
|--------------|-----|
| `graphify update .` (root) | `graphify update knowledge/` — graphify is literature-only |
| Using graphify for code questions | Use GitNexus; graphify = `knowledge/` |
| Monolithic `model.py` | Split under `models/{chronos2,vision}/` |
| Duplicating AGENTS.md into CLAUDE.md | CLAUDE.md = deltas only |
| Data/checkpoints in repo | External path + `data_dir` config |
| Undocumented ablations | `ABLATION_REGISTRY.md` + `/register-experiment` |
