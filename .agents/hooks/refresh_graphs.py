#!/usr/bin/env python3
"""Refresh derived graphs only when their authoritative inputs changed."""
import subprocess


def changed_paths() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True
    )
    return [path for path in result.stdout.splitlines() if path]


paths = changed_paths()
if any(path.endswith(".py") and path.startswith(("MMTSFM/src/", "baselines/common/")) for path in paths):
    subprocess.run(["node", ".gitnexus/run.cjs", "analyze"], check=False)
if any(path.startswith("knowledge/") and "graphify-out/" not in path for path in paths):
    subprocess.run(["graphify", "update", "knowledge/"], check=False)
