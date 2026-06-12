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
