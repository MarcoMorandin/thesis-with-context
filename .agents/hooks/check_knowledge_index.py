#!/usr/bin/env python3
"""Warn when a new knowledge document is not routed from INDEX.md."""
import json
import os
import subprocess
import sys

root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
raw = json.load(sys.stdin).get("tool_input", {}).get("file_path", "")
rel = os.path.relpath(raw, root).replace("\\", "/") if os.path.isabs(raw) else raw.replace("\\", "/")
if not rel.endswith(".md") or rel.startswith((".claude/", ".codex/", ".scratch/", "report/", "manuscript/", "knowledge/papers/", "knowledge/specs/")):
    sys.exit(0)
if os.path.basename(rel) in ("README.md", "VENDOR_NOTICE.md", "AGENTS.md", "CLAUDE.md"):
    sys.exit(0)
status = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=root, capture_output=True, text=True).stdout
if not status.startswith("??"):
    sys.exit(0)
if rel.startswith("knowledge/") and os.path.basename(rel) not in open(os.path.join(root, "knowledge", "INDEX.md")).read():
    print(f"{rel} needs an INDEX.md routing-table row.", file=sys.stderr)
    sys.exit(2)
if not rel.startswith("knowledge/"):
    print(f"{rel} is new project prose; place it in knowledge/, report/, or manuscript/.", file=sys.stderr)
    sys.exit(2)
