#!/usr/bin/env python3
"""Emit compact, computed orientation state at the start of a session."""
import os
import subprocess

root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.splitlines()
lines = [f"Branch `{branch}`" + (f", {len(dirty)} uncommitted path(s)." if dirty else ", clean.")]
if branch == "main":
    lines.append("Branch before editing: use `exp/`, `feat/`, or `fix/`.")
lines.append("Route code through GitNexus; route project prose and papers through knowledge/INDEX.md and Graphify.")
print("PROJECT BRIEF\n" + "\n".join(f"- {line}" for line in lines))
