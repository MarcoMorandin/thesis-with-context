#!/usr/bin/env python3
"""PreToolUse hook: force graph-first exploration instead of raw grep/read.

Routing (per project convention):
  - CODE files            -> GitNexus  (priority; code intelligence graph)
  - knowledge/ docs+papers -> Graphify  (literature/doc graph)

Blocks (permissionDecision: deny) broad content/structure exploration so the
agent orients via the graph first. Targeted line reads and edits are allowed.
"""

import json
import os
import sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
GRAPHIFY_INDEX = os.path.join(ROOT, "graphify-out", "graph.json")
GITNEXUS_INDEX = os.path.join(ROOT, ".gitnexus", "run.cjs")

SOURCE_EXTS = (
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".cs",
    ".kt",
    ".swift",
    ".php",
    ".scala",
    ".lua",
    ".sh",
)
DOC_EXTS = (".md", ".rst", ".txt", ".mdx", ".pdf", ".tex", ".bib")


def allow():
    sys.exit(0)


def deny(reason: str):
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


data = json.load(sys.stdin)
tool = data.get("tool_name", "")
ti = data.get("tool_input", data) or {}

raw_path = str(ti.get("file_path") or ti.get("path") or "")
norm = raw_path.replace("\\", "/")
low = norm.lower()

# Never interfere with the graphs' own artifacts or VCS internals.
if "graphify-out/" in low or "/.gitnexus/" in low or "/.git/" in low:
    allow()

in_knowledge = "/knowledge/" in low or low.startswith("knowledge/")
is_source = low.endswith(SOURCE_EXTS)
is_doc = low.endswith(DOC_EXTS)

# Escape hatch: targeted line reads (offset/limit) are edit/debug work, not
# exploration. Let them through.
if tool == "Read" and (ti.get("offset") or ti.get("limit")):
    allow()

GITNEXUS_MSG = (
    "Code exploration must go through GitNexus first (not raw grep/read). "
    'Use mcp__gitnexus__query({query:"<concept>"}) to find flows, '
    'mcp__gitnexus__context({name:"<symbol>"}) for a symbol, or '
    "mcp__gitnexus__impact before edits. Read raw source only afterward, "
    "or with offset/limit for a targeted line read."
)
GRAPHIFY_MSG = (
    "knowledge/ is literature+docs: query Graphify first (not raw grep/read). "
    'Run `graphify query "<question>"`, `graphify explain "<concept>"`, or '
    '`graphify path "<A>" "<B>"`. Read raw files only afterward.'
)


def route_block(default_code: bool):
    if in_knowledge and is_doc and os.path.exists(GRAPHIFY_INDEX):
        deny(GRAPHIFY_MSG)
    if is_source and os.path.exists(GITNEXUS_INDEX):
        deny(GITNEXUS_MSG)
    # Grep/Glob with no/dir path and no clear ext: default to code -> GitNexus.
    if default_code and not is_doc and os.path.exists(GITNEXUS_INDEX):
        deny(GITNEXUS_MSG)
    allow()


if tool == "Grep":
    # Pure content exploration -> always route through the graph.
    route_block(default_code=True)

if tool == "Read":
    # Broad whole-file read -> orient via graph first.
    route_block(default_code=False)

# Glob (filename discovery) and everything else: graphs don't replace it.
allow()
