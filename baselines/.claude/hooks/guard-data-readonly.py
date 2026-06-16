#!/usr/bin/env python3
"""PreToolUse hook: block writes into the read-only dataset of record. AGENTS.md: read-only."""
import json
import sys

PROTECTED = "/Volumes/SSD/thesis-dataset"

data = json.load(sys.stdin)
tool_input = data.get("tool_input", {})
file_path = tool_input.get("file_path", "") or tool_input.get("path", "")

if file_path.startswith(PROTECTED):
    print(
        f"Blocked: {PROTECTED} is read-only (AGENTS.md). "
        "Do not modify the dataset of record from this repo.",
        file=sys.stderr,
    )
    sys.exit(2)
