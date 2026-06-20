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

# Local training on the dataset of record (uk_pv / goes_pvdaq)
uv run python -m mmtsfm.train data.dataset_name=uk_pv data.data_dir=/Volumes/SSD/thesis-dataset

# Submit training run to SLURM cluster
sbatch MMTSFM/scripts/slurm_train.sh
```

### Running SolarVLM Baseline
```bash
# Set up environment for SolarVLM
source baselines/solar_vlm/setup_env.sh

# Train SolarVLM on the uk_pv multimodal track (offline precomputed vision features)
# (run_skippd.py is the legacy-named entrypoint)
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

# PVTSFM  -  Agent Context & Development Rules

## Mission
Build a **research-grade AI foundation model** for PV power forecasting. Primary metric is **zero-shot cross-plant generalization** on disjoint test sets using small history, without sacrificing power prediction quality. This is an **AI science project**, not an energy-domain engineering project.

---

## 1. Core Non-Negotiables

| Rule | Detail |
|------|--------|
| **Python** | `uv` only - never use pip, poetry, or conda. |
| **Config** | Hydra only - no argparse, no `yaml.load` outside Hydra. Configurations must be self-contained per baseline codebase. |
| **Git Discipline** | Branch per experiment: `exp/<name>`. Commit message format: `exp(<name>): <what changed and why>`. One logical change per commit. Never commit data, checkpoints, logs, or large binaries. |
| **Files** | One class or one script capability per file; keep files short (target < 150 lines). |
| **Models** | Multimodal foundation models (TS FM + vision FM) preferred. Avoid classical ML (XGBoost, etc.) unless explicitly justified as baseline. |
| **Literature** | Prefer 2026 papers, then late 2025; nothing before 2025. |
| **Data** | Read-only dataset of record `/Volumes/SSD/thesis-dataset/` (`dataset_all.parquet` + `images_all.h5`, frame pointer `image_h5_index`; both `uk_pv` and `goes_pvdaq`). Do not refactor data pipelines here. |

---

## 2. Python Module & Naming Conventions

### File Naming
* `{component}.py` (e.g., `grassmann_mixer.py`) — Contains a single `nn.Module` class.
* `{verb}_{noun}.py` (e.g., `build_batch.py`) — Contains a single pure function.
* `lightning_{stage}.py` (e.g., `lightning_stage2a.py`) — Contains one Lightning module variant.

### Import Rules
* Use relative imports within the `pvtsfm` package only.
* No circular imports (specifically between `models/chronos2/` and `models/vision/`).
* Keep shared types in `pvtsfm/types.py`.

### Hydra Integration
* All hyperparameters must live in `configs/`. Do not hardcode magic numbers in model code.
* Use `@dataclass` + `instantiate` pattern for complex submodules.

---

## 3. Testing & Verification Rules
* Mirror the `src/pvtsfm/models/` structure under `tests/models/`.
* Each module file must have a corresponding `test_<module>.py` containing shape and gradient smoke tests.
* **Verification**: Run `uv run pytest` before claiming a fix works. Never claim a fix works without running tests and reviewing logs.

---

## 4. Experiment & Ablation Workflow

Every experiment must define:
1. **Hypothesis**: A single-sentence statement of what you are testing.
2. **Config Diff**: A config diff file under `configs/ablation/` (or within baseline-specific configs).
3. **Registry Entry**: Register the run in `docs/experiments/ABLATION_REGISTRY.md`.
4. **Baseline Comparison**: Compare against the standard baselines defined in `docs/experiments/BASELINE_PROTOCOL.md`.

### Evaluation Splits
* `intra_plant`: Same plant, held-out time (sanity check only).
* `cross_plant`: Disjoint held-out plants (primary test metric for zero-shot generalization).

### Baselines Priority
1. **Solar-VLM** (multimodal PV SOTA baseline)
2. **Chronos-2** + `TS-RAG` / `Cross-RAG` / `TS-Memory`
3. **SPIRIT** (vision FM zero-shot)
4. **TiRex**, **Reverso** (TS-only foundation models)
5. **TEMPLATE** metrics for transferability

---

## 5. What Agents Must NOT Do

* Introduce energy-domain physics heuristics (CSI conversion, irradiance physics) unless explicitly ablating them out.
* Introduce `scikit-learn`, `lightgbm`, or `xgboost` without explicit user approval.
* Create monolithic files with multiple classes.
* Modify `/Volumes/SSD/thesis-dataset` (read-only dataset of record).
* Copy large checkpoints or datasets into the repository.

---

## 6. Knowledge Graphs & Tools

* **Code Graph (GitNexus)**: Run `npx gitnexus analyze` and `npx gitnexus setup` to analyze call chains and blast radius. Use `impact` before editing shared modules.
* **Research Graph (Graphify)**: Run `/graphify knowledge/ --wiki --update` to compile background literature (papers and proposals).
* Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` if present.
* After code changes, run `graphify update .` and re-index `gitnexus`.

---

## 7. Git & Version Control Protocol

AI agents must strongly rely on Git to maintain repository safety, trace changes, and ensure logical code isolation.

### 7.1 Startup Verification
* **Check Status**: Always run `git status` at the beginning of a session to verify you are working on a clean tree.
* **Isolate Work**: Never perform research or feature development directly on `main`. Ensure you are on a task-specific branch (`exp/<name>`, `feat/<name>`, or `fix/<name>`).

### 7.2 Incremental Committing (Micro-Commits)
* **One Step, One Commit**: Commit immediately after completing and verifying a logical sub-task (e.g., implementing a single class, fixing a specific bug, creating a test).
* **Do Not Accumulate Changes**: Do not wait until the entire task is finished to commit. Large, multi-file changes are an anti-pattern.
* **Commit Messages**: Adhere strictly to: `exp(<name>): <short desc>`, `feat(<name>): <short desc>`, or `fix(<name>): <short desc>`.

### 7.3 Git-Assisted Debugging & Rollbacks
* **Safety Net**: If tests fail after your modifications and the fix is not immediately obvious, do not pile up temporary workarounds. Use `git checkout` or `git reset --hard` to roll back to the last clean, verified commit and re-approach the problem.
* **Diff Reviews**: Review your changes with `git diff` before running verification tests to check for unintended edits or left-over debugging code.

### 7.4 Pre-Completion Audit
* **Audit Diff**: Before claiming a task is complete, run `git diff --cached` or `git diff HEAD` and review every line changed. Remove all leftover print statements, commented-out test code, or temporary files.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **thesis-with-context** (2321 symbols, 4095 relationships, 134 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Available Tools

You have access to the GWS CLI for Google Workspace operations.

### Sheets
- Read: `gws sheets read --id <id> --range <range>`
- Append: `gws sheets append --id <id> --range <range> --values "v1,v2,v3"`
- Update: `gws sheets update --id <id> --range <range> --values "v1,v2,v3"`

### Gmail
- List: `gws gmail list --query <query> --limit <n>`
- Read: `gws gmail read --id <message_id>`
- Send: `gws gmail send --to <email> --subject <subject> --body <body>`

### Docs
- Create: `gws docs create --title <title> --content <text>`
- Read: `gws docs read --id <doc_id>`
- Append: `gws docs append --id <doc_id> --content <text>`

### Drive
- List: `gws drive list --folder <folder_id> --limit <n>`
- Upload: `gws drive upload --file <path> --folder <folder_id>`
- Download: `gws drive download --id <file_id> --output <path>`

All commands output JSON unless --format plain is specified.
Sheet IDs and Doc IDs come from the URL of the respective file.
