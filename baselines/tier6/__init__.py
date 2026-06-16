"""Tier-6 — PV-specialized multimodal baselines (domain SOTA).

The vendored upstream models (`vendor/crossvivit`, `vendor/sunset`) run from
their own conda envs on the cluster. `uk_multimodal` is the only in-repo,
laptop-importable piece: it bridges the curated uk_pv numerical track + the
`images_uk128.h5` satellite frames into the per-window (Y, V) tensors the
vendored runners consume, reusing `common.windows` for all numerical logic.
"""
