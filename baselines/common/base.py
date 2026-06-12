"""Baseline interface and registry (BASELINE_COMPARISON.md §6, contract test).

Every baseline implements ``fit`` (no-op for zero-shot/reference models) and
``predict``, which consumes a batch dict from ``WindowDataset.batch`` and
returns a ``Forecast``. The registry feeds both ``run_eval.py`` and the
parametrized contract tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from . import config
from .windows import WindowDataset


@dataclass
class Forecast:
    point: np.ndarray                       # (N, H) float32
    quantiles: np.ndarray | None = None     # (N, H, Q) float32, sorted by level
    quantile_levels: tuple[float, ...] = field(default=config.QUANTILE_LEVELS)


class Baseline(ABC):
    """Base class for all tier 0-2 baselines."""

    name: str = "baseline"
    tier: int = -1
    requires_fit: bool = False
    supports_quantiles: bool = False

    def fit(self, train: WindowDataset, val: WindowDataset) -> None:
        """Train on train-plant windows; tune/early-stop on val plants."""

    @abstractmethod
    def predict(self, batch: dict) -> Forecast:
        """Forecast a batch dict (see WindowDataset.__getitem__ for keys)."""


REGISTRY: dict[str, type[Baseline]] = {}


def register(cls: type[Baseline]) -> type[Baseline]:
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate baseline name: {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def build(name: str, **kwargs) -> Baseline:
    """Instantiate a registered baseline, importing tier modules on demand."""
    if name not in REGISTRY:
        import importlib

        for module in ("tier0.reference", "tier1.gbm", "tier1.tabpfn_model",
                       "tslib.adapter", "tier3.chronos", "tier3.timesfm",
                       "tier3.tirex", "tier3.ttm", "tier4.rag", "tier4.cora"):
            try:
                importlib.import_module(module)
            except ImportError:
                pass
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
