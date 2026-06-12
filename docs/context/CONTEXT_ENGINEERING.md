# Context Engineering Guide (2026)

A brief guide for building AI-agent-native research codebases. Tailored to PVTSFM.

## What is context engineering?

Context engineering is the deliberate design of **what an AI agent sees, when, and in what order**  -  so it can work reliably on complex research code without hallucinating architecture or breaking dependencies.

It is not prompt engineering. It is **repository design**: files, rules, graphs, and workflows that compound across sessions.

## The 2026 stack (four layers)

```
???????????????????????????????????????????????????????????
?  Layer 4: Skills (superpowers, caveman, ars-plan, )    ?
???????????????????????????????????????????????????????????
?  Layer 3: Knowledge graphs (GitNexus + Graphify)        ?
???????????????????????????????????????????????????????????
?  Layer 2: Rules (.cursor/rules/, AGENTS.md, CLAUDE.md)?
???????????????????????????????????????????????????????????
?  Layer 1: Code structure (modular, short, tested)     ?
???????????????????????????????????????????????????????????
```

### Layer 1  -  Code structure

**Principle**: Agents read files; they do not hold your architecture in memory.

| Practice | Why |
|----------|-----|
| One class per file | Agent edits one concern; tests map 1:1 |
| Files <150 lines | Fits in context window; forces decomposition |
| Predictable naming | Agent guesses correct path without search |
| Hydra configs | Agent changes behavior without touching code |
| Tests per module | Agent verifies edits without full training run |

### Layer 2  -  Rules and entry docs

| File | Role |
|------|------|
| `AGENTS.md` | Single source of truth: mission, constraints, read order |
| `CLAUDE.md` | Claude-specific hooks (graphify, gitnexus) |
| `.cursor/rules/*.mdc` | Scoped rules (Python, research, configs) |

**Rule of thumb**: If you explain it twice in chat, put it in `AGENTS.md`.

### Layer 3  -  Knowledge graphs (dual system)

Use **two** graphs  -  they solve different problems:

| Tool | Best for | Stars (Jun 2026) | Integration |
|------|----------|------------------|-------------|
| **GitNexus** | Code structure, call chains, blast radius | ~42k | MCP native, `npx gitnexus analyze` |
| **Graphify** | Papers, proposals, cross-document synthesis | growing | `/graphify`, MCP, wiki output |
| Understand Anything | General doc Q&A | smaller | Skip for this project |

**Recommendation**: GitNexus for `src/` + Graphify for `knowledge/papers/baselines/`, `knowledge/papers/related/`, `knowledge/docs/`, `MMTSFM/`.

```bash
# Code graph
npx gitnexus analyze
npx gitnexus setup   # MCP for Cursor/Claude

# Research graph
/graphify knowledge/ --wiki --watch
```

### Layer 4  -  Skills

Install and invoke explicitly:

| Skill | Use |
|-------|-----|
| `superpowers` | Plans, TDD, verification gates |
| `caveman` | Dense status during long runs |
| `ars-plan` | Experiment design |
| `deep-research` | Literature (filter: 2026 ? late 2025) |
| `academic-paper` | Writing pipeline |
| `systematic-debugging` | Training bugs |
| `verification-before-completion` | Never claim success without pytest/logs |

**Also consider** (high GitHub traction, research-relevant):

- `lean-ctx`  -  token-efficient reads (you already use this)
- `academic-research-skills`  -  paper/review pipeline
- `graphify`  -  research corpus graph

## Session workflow for Claude Code

```
1. Agent reads AGENTS.md + RESEARCH_SCOPE.md
2. GitNexus: impact analysis on target module
3. Graphify: check related papers / prior decisions
4. Branch: exp/<name>
5. Edit ONE module + matching test + hydra config
6. uv run pytest tests/models/test_<module>.py
7. graphify update . ; git commit
8. (optional) uv run pvtsfm-train for integration
```

## Context budgeting tips

1. **Point, don't paste**  -  reference `docs/architecture/OVERVIEW.md` instead of dumping proposal
2. **Registry pattern**  -  `ABLATION_REGISTRY.md` tracks experiments; agent does not re-read all logs
3. **Fn refs**  -  lean-ctx caches files; prefer `ctx_read` over re-reading large files
4. **Exclude noise**  -  `.gitignore` artifacts; never let agent index `logs/` or checkpoints
5. **Staged context**  -  `AGENTS.md` read order prevents loading everything at once

## Anti-patterns

| Anti-pattern | Fix |
|--------------|-----|
| Monolithic `model.py` (800 lines) | Split into `models/fusion/`, `models/vision/` |
| Secrets in repo | `.env` gitignored; Hydra env resolver |
| Data in repo | External SSD + `data_root` in config |
| Undocumented ablations | `ABLATION_REGISTRY.md` |
| Agent invents baselines | `baselines/*.md` + thin wrappers in `src/pvtsfm/baselines/` |

## Migration from MMTSFM

PVTSFM narrows MMTSFM to PV-only:

| MMTSFM (general) | PVTSFM (this repo) |
|------------------|-------------------|
| Multi-domain datasets | `/Volumes/SSD/standardized-dataset` only |
| `scripts/meteorology/` | Removed |
| `scripts/solar/` refactor | Out of scope (data external) |
| Cross-entity GroupAttention | Cross-**plant** context tokens |
| MeteoNet, traffic eval | PV baselines only (Solar-VLM, SPIRIT, Chronos-2+RAG) |

Port modules from MMTSFM incrementally; one file per PR.
