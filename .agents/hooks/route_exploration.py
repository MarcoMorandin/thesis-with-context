#!/usr/bin/env python3
"""Route broad Claude/Codex exploration through the appropriate graph first."""
import json
import os
import subprocess
import sys

root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
event = json.load(sys.stdin)
tool = event.get("tool_name", "")
data = event.get("tool_input", {})
path = str(data.get("file_path") or data.get("path") or "").replace("\\", "/").lower()
if tool == "Read" and (data.get("offset") or data.get("limit")):
    sys.exit(0)
if "knowledge/" in path and path.endswith((".md", ".pdf", ".tex", ".bib")) and os.path.exists(os.path.join(root, "knowledge", "graphify-out", "graph.json")):
    print("Use Graphify or knowledge/INDEX.md before broad prose exploration.", file=sys.stderr)
    sys.exit(2)
if path.endswith((".py", ".ts", ".js", ".sh")) and os.path.exists(os.path.join(root, ".gitnexus", "run.cjs")):
    print("Use GitNexus before broad code exploration.", file=sys.stderr)
    sys.exit(2)
