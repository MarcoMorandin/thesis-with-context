#!/usr/bin/env python3
"""Warn if a new MMTSFM source module exceeds the repository target."""
import json
import subprocess
import sys

path = json.load(sys.stdin).get("tool_input", {}).get("file_path", "")
if not path.endswith(".py") or "/MMTSFM/src/" not in path:
    sys.exit(0)
status = subprocess.run(["git", "status", "--porcelain", "--", path], capture_output=True, text=True).stdout
if status.startswith("??") and sum(1 for _ in open(path)) > 150:
    print(f"{path} exceeds the 150-line target; split a capability before continuing.", file=sys.stderr)
    sys.exit(2)
