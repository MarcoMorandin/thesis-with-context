#!/usr/bin/env python3
"""Block dependency-manager commands other than uv."""
import json
import re
import sys

command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
if re.search(r"(^|[;&|]\s*)(pip3?|poetry|conda)\s+(install|add|remove|uninstall|update)\b", command):
    print("Blocked: use `uv add`, `uv remove`, or `uv sync` for dependencies.", file=sys.stderr)
    sys.exit(2)
