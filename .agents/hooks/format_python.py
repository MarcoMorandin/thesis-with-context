#!/usr/bin/env python3
"""Format a Python file after a native edit or write."""
import json
import subprocess
import sys

path = json.load(sys.stdin).get("tool_input", {}).get("file_path", "")
if path.endswith(".py"):
    subprocess.run(["uv", "run", "ruff", "format", path], capture_output=True)
