# Claude Code  -  Project Instructions

Please adhere to all guidelines in [AGENTS.md](file:///Users/marcomorandin/Desktop/thesis-with-context/AGENTS.md) first. This file defines specific CLI commands and guidelines for Claude Code (claude.ai/code).

---

## 1. General Instructions

* **Python execution**: Always run code/scripts using `uv run` (e.g., `uv run pytest`). Never run `python` or `pip` directly.
* **Adding dependencies**: Use `uv add <package>` or `uv sync` to manage python dependencies.
* **Git reliance**:
  * Run `git status` at startup to ensure you are on a clean, task-specific branch (never work directly on `main`).
  * Commit incrementally (micro-commits) immediately after completing and verifying each logical sub-task.
  * Run `git diff` to review your edits and clean up leftover debug print statements before completing a task.
  * If changes fail tests and debugging is not obvious, roll back immediately (`git checkout` or `git reset --hard`) rather than accumulating untested fixes.
* **Response style**:
  * Keep replies concise and direct.
  * Cite exact file paths with line numbers when referencing code.
  * Prefer incremental, small changes over massive refactors.
  * Verify your changes using tests (`uv run pytest`) before claiming completion.

---

## 2. Useful Commands

### Running Tests
```bash
# Run all smoke tests
uv run pytest

# Run a specific test file
uv run pytest tests/models/test_vision_chronos2.py
```

### Knowledge Graph Tools
```bash
# Analyze code structure & call chains
npx gitnexus analyze

# Setup / re-index code graph
npx gitnexus setup

# Compile literature and proposal papers to Graphify Wiki
/graphify knowledge/ --wiki --update

# Update graph after code changes
graphify update .
```

### Running MMTSFM Baseline
```bash
# Local dev training smoke-test (synthetic data)
uv run python -m mmtsfm.train

# Local training on SKIPP'D
uv run python -m mmtsfm.train data.dataset_name=skippd data.data_dir=/Volumes/SSD/standardized-dataset/solar/skippd

# Submit training run to SLURM cluster
sbatch MMTSFM/scripts/slurm_train.sh
```

### Running SolarVLM Baseline
```bash
# Set up environment for SolarVLM
source baselines/solar_vlm/setup_env.sh

# Train SolarVLM on SKIPP'D (using offline precomputed features)
python baselines/solar_vlm/run_skippd.py --is_training 1 --use_offline_vision --vision_feat_dir /path/to/feats
```


# lean-ctx — Context Engineering Layer

PREFER lean-ctx MCP tools over native equivalents for token savings:

| PREFER | OVER | Why |
|--------|------|-----|
| `ctx_read(path)` | Read / cat / head / tail | Session caching, 8 compression modes, re-reads cost ~13 tokens |
| `ctx_shell(command)` | Bash (shell commands) | Pattern-based compression for git, npm, cargo, docker, tsc |
| `ctx_search(pattern, path)` | Grep / rg | Compact context, token-efficient results |
| `ctx_tree(path, depth)` | ls / find | Compact directory maps with file counts |

## ctx_read Modes

- `full` — cached read (use for files you will edit)
- `map` — deps + API signatures (use for context-only files)
- `signatures` — API surface only
- `diff` — changed lines only (after edits)
- `aggressive` — syntax stripped
- `entropy` — Shannon + Jaccard filtering
- `lines:N-M` — specific range

## File Editing

Use native Edit/StrReplace when available. If Edit requires Read and Read is unavailable,
use `ctx_edit(path, old_string, new_string)` — it reads, replaces, and writes in one MCP call.
NEVER loop trying to make Edit work. If it fails, switch to ctx_edit immediately.
Write, Delete have no lean-ctx equivalent — use them normally.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **thesis-with-context** (2004 symbols, 3487 relationships, 137 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
