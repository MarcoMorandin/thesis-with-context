"""Tier-6 — PV-specialized multimodal baselines (domain SOTA).

The vendored upstream models (`vendor/crossvivit`, `vendor/sunset`) run from
their own conda envs on the cluster. `uk_multimodal` is the only in-repo,
laptop-importable piece: it bridges the curated numerical track + the
`images_all.h5` satellite frames (both uk_pv and goes_pvdaq) into the per-window
(Y, V) tensors the vendored runners consume, reusing `common.windows` for all
numerical logic.
"""
