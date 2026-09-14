#!/usr/bin/env python3
"""Block writes to the dataset of record."""
import json
import sys

data = json.load(sys.stdin).get("tool_input", {})
path = data.get("file_path") or data.get("path") or ""
if str(path).startswith("/leonardo_scratch/fast/IscrC_MTSFM/data"):
    print("Blocked: the dataset of record is read-only.", file=sys.stderr)
    sys.exit(2)
