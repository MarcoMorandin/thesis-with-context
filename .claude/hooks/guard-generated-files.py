#!/usr/bin/env python3
"""PreToolUse hook: block hand-edits to generated artifacts.

AGENTS.md §5 / knowledge/conventions.md §6: these paths are produced by a tool and
must be regenerated, never edited. A hand-edit silently desynchronizes the artifact
from its source and is undone by the next refresh.
"""

import json
import sys

# (path fragment, how to regenerate it)
GENERATED = [
    ("graphify-out/", "graphify update knowledge/"),
    ("/.gitnexus/", "node .gitnexus/run.cjs analyze"),
    ("baselines/results/ALL_RESULTS.md", "python baselines/scripts/aggregate_all.py"),
]

# Compiled outputs: never authored by hand.
COMPILED_SUFFIXES = (
    ".pdf",
    ".fls",
    ".fdb_latexmk",
    ".synctex.gz",
    ".aux",
    ".bbl",
    ".blg",
    ".toc",
)

data = json.load(sys.stdin)
ti = data.get("tool_input", {}) or {}
path = str(ti.get("file_path") or ti.get("path") or "").replace("\\", "/")

if not path:
    sys.exit(0)

for fragment, regen in GENERATED:
    if fragment in path:
        print(
            f"Blocked: {path} is generated, not authored. "
            f"Regenerate it with `{regen}` instead of editing it by hand "
            "(AGENTS.md §5).",
            file=sys.stderr,
        )
        sys.exit(2)

if path.endswith(COMPILED_SUFFIXES):
    print(
        f"Blocked: {path} is a compiled build product. Edit the source "
        "(.tex / .md) and rebuild — do not write the artifact directly.",
        file=sys.stderr,
    )
    sys.exit(2)
