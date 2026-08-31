"""Tier-2 baselines: upstream implementations driven through the MMTSFM protocol.

``tslib/`` holds compact in-repo ports trained by the run_eval.py harness
(stride-1 windows, batch 256, lr 1e-3). This package instead wraps upstream
library implementations and drives them through the *MMTSFM* data + training +
evaluation path, so the only difference between the two arms is the model —
not the protocol, not the budget. See tier2/train_itransformer_nf.py.

Two kinds of arm live here:

* **Supervised deep TS** — ``train_itransformer_nf.py`` (neuralforecast
  iTransformer), trained on the train plants. This is what the tier name means.
* **Zero-shot tabular FM** — ``tabfm_model.py`` (Google TabFM v1.0.0), filed
  here by request. It is *not* supervised and *not* deep-TS: it is the sibling
  of the tier-1 TabPFN arm and consumes the same ``tier1.features`` table.
  Label it accordingly wherever the tier-2 rows are reported.
"""

try:  # tabfm is an optional dependency group
    from . import tabfm_model  # noqa: F401
except ImportError:
    pass
