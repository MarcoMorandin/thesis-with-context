#!/usr/bin/env python3
"""Block edits to generated and compiled artifacts."""
import json
import sys

data = json.load(sys.stdin).get("tool_input", {})
path = str(data.get("file_path") or data.get("path") or "").replace("\\", "/")
generated = {
    "graphify-out/": "graphify update knowledge/",
    "/.gitnexus/": "node .gitnexus/run.cjs analyze",
    "baselines/results/ALL_RESULTS.md": "uv run python baselines/scripts/aggregate_all.py",
}
if any(fragment in path for fragment in generated):
    command = next(command for fragment, command in generated.items() if fragment in path)
    print(f"Blocked: {path} is generated; regenerate it with `{command}`.", file=sys.stderr)
    sys.exit(2)
if path.endswith((".pdf", ".fls", ".fdb_latexmk", ".synctex.gz", ".aux", ".bbl", ".blg", ".toc")):
    print(f"Blocked: {path} is a compiled artifact; edit its source instead.", file=sys.stderr)
    sys.exit(2)
