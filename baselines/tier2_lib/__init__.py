"""Tier-2 baselines built on a PRE-BUILT library, trained on MMTSFM's own windows.

``tslib/`` holds compact in-repo ports trained by the run_eval.py harness
(stride-1 windows, batch 256, lr 1e-3). This package instead wraps an upstream
library implementation and drives it through the *MMTSFM* data + training +
evaluation path, so the only difference between the two arms is the model —
not the protocol, not the budget. See scripts/train_itransformer_nf.py.
"""
