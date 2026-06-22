# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

from .config import Chronos2CoreConfig, Chronos2ForecastingConfig
from .model import Chronos2Model
from .pipeline import Chronos2Pipeline
from .dataset import Chronos2Dataset
from .vision_chronos2 import VisionChronos2Model, VisionChronos2Config, VisionChronos2Output


def __getattr__(name):
    # Lazy: VisionChronos2LightningModule pulls in `lightning.pytorch`, which the
    # inference-only baseline envs (tier3) do not install. Importing it on demand
    # keeps `from mmtsfm.models.chronos2 import Chronos2Model` working without it,
    # while training contexts (which have lightning) still get the symbol.
    if name == "VisionChronos2LightningModule":
        from .lightning_module import VisionChronos2LightningModule

        return VisionChronos2LightningModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Chronos2CoreConfig",
    "Chronos2ForecastingConfig",
    "Chronos2Model",
    "Chronos2Pipeline",
    "Chronos2Dataset",
    "VisionChronos2Model",
    "VisionChronos2Config",
    "VisionChronos2Output",
    "VisionChronos2LightningModule",
]
