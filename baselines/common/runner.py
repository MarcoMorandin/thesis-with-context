"""Evaluation loop + reproducibility manifest (BASELINE_COMPARISON.md §4, §6.7).

``evaluate_model`` streams batches through a baseline, accumulates per-plant
masked errors, and returns macro-averaged metrics. ``write_results`` persists
results next to a manifest with git SHA, config hash, seed and dataset
version, as required before any result is logged.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .base import Baseline, Forecast
from .metrics import PerPlantAccumulator
from .windows import WindowDataset


def evaluate_model(
    model: Baseline,
    dataset: WindowDataset,
    batch_size: int = 256,
) -> dict:
    """Evaluate on daylight, valid future steps; macro-average over plants."""
    acc = PerPlantAccumulator()
    for batch in dataset.iter_batches(batch_size):
        forecast: Forecast = model.predict(batch)
        point = np.asarray(forecast.point, dtype=np.float64)
        if point.shape != batch["y_future"].shape:
            raise ValueError(
                f"{model.name}: forecast shape {point.shape} != "
                f"target shape {batch['y_future'].shape}"
            )
        if not np.isfinite(point[batch["mask_future"] == 1]).all():
            raise ValueError(f"{model.name}: non-finite forecast on valid steps")
        mask = batch["mask_future"] * batch["daylight_future"]
        acc.update(
            plants=batch["site_id"],
            y_true=batch["y_future"],
            y_pred=point,
            mask=mask,
            quantile_preds=forecast.quantiles,
        )
    return {"overall": acc.macro(), "per_plant": acc.per_plant()}


def add_skill_scores(results: dict, reference: dict) -> dict:
    """Attach SS = 1 − NRMSE/NRMSE_smart_persistence (overall + per plant)."""
    from .metrics import skill_score

    ref = reference["overall"].get("nrmse")
    if ref:
        results["overall"]["skill_score"] = skill_score(
            results["overall"]["nrmse"], ref
        )
    for plant, row in results["per_plant"].items():
        ref_row = reference["per_plant"].get(plant)
        if ref_row:
            row["skill_score"] = skill_score(row["nrmse"], ref_row["nrmse"])
    return results


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _dataset_version(data_path: str) -> str:
    p = Path(data_path)
    if not p.exists():
        return "unknown"
    stat = p.stat()
    return f"{p.name}:{stat.st_size}:{int(stat.st_mtime)}"


def write_results(
    out_dir: str | Path,
    model_name: str,
    results: dict,
    run_config: dict,
    data_path: str = config.DEFAULT_DATA_PATH,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_blob = json.dumps(run_config, sort_keys=True, default=str)
    manifest = {
        "model": model_name,
        "git_sha": _git_sha(),
        "config_hash": hashlib.sha256(config_blob.encode()).hexdigest()[:16],
        "seed": run_config.get("seed", config.SEED),
        "dataset_version": _dataset_version(data_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": run_config,
    }
    path = out / f"{model_name}.json"
    path.write_text(
        json.dumps({"manifest": manifest, "results": results}, indent=2) + "\n"
    )
    return path
