#!/usr/bin/env python3
"""PostToolUse hook: warn when a newly-created MMTSFM/src file exceeds 150 lines."""
import json
import subprocess
import sys

LIMIT = 150

data = json.load(sys.stdin)
file_path = data.get("tool_input", {}).get("file_path", "")

if not file_path.endswith(".py") or "/MMTSFM/src/" not in file_path:
    sys.exit(0)

# Only flag newly-created files (untracked in git), not edits to existing ones.
status = subprocess.run(
    ["git", "status", "--porcelain", "--", file_path],
    capture_output=True, text=True
).stdout.strip()

if not status.startswith("??"):
    sys.exit(0)

with open(file_path) as f:
    n_lines = sum(1 for _ in f)

if n_lines > LIMIT:
    print(
        f"New file {file_path} has {n_lines} lines (>{LIMIT}). "
        f"AGENTS.md: one class/script per file, target <{LIMIT} lines. Consider splitting.",
        file=sys.stderr,
    )
    sys.exit(2)
