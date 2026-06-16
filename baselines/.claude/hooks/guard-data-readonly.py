#!/usr/bin/env python3
"""PreToolUse hook: block writes into the external standardized dataset. AGENTS.md: read-only."""
import json
import sys

PROTECTED = "/Volumes/SSD/standardized-dataset"

data = json.load(sys.stdin)
tool_input = data.get("tool_input", {})
file_path = tool_input.get("file_path", "") or tool_input.get("path", "")

if file_path.startswith(PROTECTED):
    print(
        f"Blocked: {PROTECTED} is read-only (AGENTS.md). "
        "Do not modify the standardized dataset from this repo.",
        file=sys.stderr,
    )
    sys.exit(2)
