#!/usr/bin/env python3
"""SessionStart hook: a compact orientation brief, computed not stored.

Everything here is state that CLAUDE.md cannot know: which branch you are on,
whether the two graphs are stale, and whether results are newer than the report
that interprets them. Keeping it in a hook is what lets AGENTS.md stay static.
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())


def sh(*args):
    try:
        return subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=8
        ).stdout.strip()
    except Exception:
        return ""


def newest_mtime(path, exts=None):
    """Most recent mtime under path (or of path itself). 0 if absent."""
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return 0.0
    if os.path.isfile(p):
        return os.path.getmtime(p)
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv")]
        for fn in filenames:
            if exts and not fn.endswith(exts):
                continue
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, fn)))
            except OSError:
                pass
    return newest


lines = []

branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD")
dirty = sh("git", "status", "--porcelain")
n_dirty = len([ln for ln in dirty.splitlines() if ln.strip()])

if branch == "main":
    lines.append(
        "⚠️  On `main`. AGENTS.md §2: branch (`exp/`, `feat/`, `fix/`) before "
        "editing anything."
    )
else:
    lines.append(
        f"Branch `{branch}`"
        + (f", {n_dirty} uncommitted path(s)." if n_dirty else ", clean.")
    )

# Graph freshness — stale graphs are the main source of wrong answers.
code_src = max(
    newest_mtime("MMTSFM/src", (".py",)), newest_mtime("baselines/common", (".py",))
)
if code_src > newest_mtime(".gitnexus/meta.json"):
    lines.append(
        "Code graph is STALE → `node .gitnexus/run.cjs analyze` before code questions."
    )

prose = newest_mtime("knowledge", (".md", ".pdf"))
if prose > newest_mtime("graphify-out/graph.json"):
    lines.append(
        "Prose graph is STALE → `graphify update knowledge/` before literature questions."
    )

# Results newer than the narrative that interprets them.
if newest_mtime("baselines/results", (".json",)) > newest_mtime(
    "report/BASELINE_TEST_REPORT.md"
):
    lines.append(
        "New result JSONs are newer than report/BASELINE_TEST_REPORT.md — "
        "re-aggregate (`baselines/scripts/aggregate_all.py`) before quoting numbers."
    )

lines.append(
    "Routing: prose → knowledge/INDEX.md · code → gitnexus MCP · papers → "
    "`graphify query` · numbers → baselines/results/ALL_RESULTS.md. "
    "One fact, one home — link, never copy."
)

print(
    json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "PROJECT BRIEF\n"
                + "\n".join(f"- {ln}" for ln in lines),
            }
        }
    )
)
sys.exit(0)
