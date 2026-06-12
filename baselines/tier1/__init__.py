from . import gbm  # noqa: F401 — registers tier-1 baselines

try:  # tabpfn is an optional dependency group
    from . import tabpfn_model  # noqa: F401
except ImportError:
    pass
