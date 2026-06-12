"""Reproducibility helpers.

Call set_seed(seed) once at the top of every train/eval entry point, before
any model, dataset, or dataloader is created.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all RNG sources for fully reproducible runs.

    Covers:
    - Python built-in random
    - NumPy
    - PyTorch CPU and all CUDA devices
    - cuDNN (deterministic kernels, no auto-tuning)
    - Lightning DataLoader worker seeds via env vars
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = "1"
