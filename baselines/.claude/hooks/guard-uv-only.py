#!/usr/bin/env python3
"""PreToolUse hook: block pip/poetry/conda dependency commands. AGENTS.md: uv only."""
import json
import re
import sys

BANNED = re.compile(r"(^|[;&|]\s*)(pip3?|poetry|conda)\s+(install|add|remove|uninstall|update)\b")

data = json.load(sys.stdin)
command = data.get("tool_input", {}).get("command", "")

if BANNED.search(command):
    print(
        "Blocked: AGENTS.md mandates `uv` only for dependency management. "
        "Use `uv add <package>` / `uv remove <package>` / `uv sync` instead.",
        file=sys.stderr,
    )
    sys.exit(2)
