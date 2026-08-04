#!/usr/bin/env python3
"""PostToolUse hook: keep knowledge/ a single source of truth.

Two invariants, both from AGENTS.md §5:

1. Every doc in knowledge/ has a row in knowledge/INDEX.md. An unindexed doc is
   invisible to the routing table and becomes a second home for a fact.
2. Project prose lives in knowledge/ (or report/ / manuscript/). A new .md
   dropped at the repo root or into a code tree is how duplication starts.

Warn-only (exit 2 feeds the message back to the model); never blocks a write.
"""

import json
import os
import subprocess
import sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
INDEX = os.path.join(ROOT, "knowledge", "INDEX.md")

# Trees allowed to hold prose without an INDEX.md row.
PROSE_OK = ("report/", "manuscript/", "knowledge/papers/", "knowledge/specs/")
# Component-local READMEs are legitimate: they document how to run that subtree.
LOCAL_DOC_NAMES = ("README.md", "VENDOR_NOTICE.md", "AGENTS.md", "CLAUDE.md")

data = json.load(sys.stdin)
raw = str(data.get("tool_input", {}).get("file_path", "")).replace("\\", "/")

if not raw.endswith(".md"):
    sys.exit(0)

rel = os.path.relpath(raw, ROOT) if os.path.isabs(raw) else raw
rel = rel.replace("\\", "/")

if rel.startswith("..") or "/.claude/" in f"/{rel}" or rel.startswith(".claude/"):
    sys.exit(0)
if os.path.basename(rel) in LOCAL_DOC_NAMES:
    sys.exit(0)
if any(rel.startswith(p) for p in PROSE_OK):
    sys.exit(0)

# Only complain about files this write actually created.
status = subprocess.run(
    ["git", "status", "--porcelain", "--", rel],
    cwd=ROOT,
    capture_output=True,
    text=True,
).stdout.strip()
if not status.startswith("??"):
    sys.exit(0)

if rel.startswith("knowledge/"):
    try:
        with open(INDEX) as f:
            index_text = f.read()
    except OSError:
        sys.exit(0)
    name = os.path.basename(rel)
    if name not in index_text:
        print(
            f"{rel} is a new knowledge/ doc with no row in knowledge/INDEX.md.\n"
            "AGENTS.md §5: one topic per file, and every file is routable. Either\n"
            "  (a) add a row to the INDEX.md routing table, or\n"
            "  (b) merge this content into the existing file that owns the topic.\n"
            "Check INDEX.md §2 (canonical sources) before creating a second home "
            "for a fact.",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(0)

print(
    f"{rel} is new project prose outside knowledge/.\n"
    "AGENTS.md §5: project prose belongs in knowledge/ (one topic per file, "
    "indexed in INDEX.md); results narrative in report/; thesis text in "
    "manuscript/. Move it, or merge it into the file that already owns the topic.",
    file=sys.stderr,
)
sys.exit(2)
