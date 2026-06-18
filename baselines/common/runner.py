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


def compute_ramp_thresholds(
    dataset: WindowDataset, quantile: float = 0.9
) -> dict[str, float]:
    """Per-plant top-decile |ΔY| threshold for the S6 ramp subset (§4.2).

    Thresholds are a property of the data, not of any model, so they are
    computed once per eval split and shared by every baseline.
    """
    deltas: dict[str, list[np.ndarray]] = {}
    for batch in dataset.iter_batches(1024):
        d, m = _future_deltas(batch)
        for plant in np.unique(batch["site_id"]):
            rows = batch["site_id"] == plant
            valid = m[rows] > 0
            deltas.setdefault(str(plant), []).append(d[rows][valid])
    return {
        plant: float(np.quantile(np.concatenate(parts), quantile))
        for plant, parts in deltas.items()
        if sum(len(p) for p in parts) > 0
    }


def _future_deltas(batch: dict) -> tuple[np.ndarray, np.ndarray]:
    """|ΔY| per future step (vs the previous step) and its validity mask."""
    prev = np.concatenate(
        [batch["y_hist"][:, -1:], batch["y_future"][:, :-1]], axis=1
    )
    prev_mask = np.concatenate(
        [batch["mask_hist"][:, -1:], batch["mask_future"][:, :-1]], axis=1
    )
    delta = np.abs(batch["y_future"] - prev)
    valid = batch["mask_future"] * prev_mask * batch["daylight_future"]
    return delta, valid


def _ramp_mask(batch: dict, thresholds: dict[str, float]) -> np.ndarray:
    delta, valid = _future_deltas(batch)
    thr = np.array(
        [thresholds.get(str(p), np.inf) for p in batch["site_id"]],
        dtype=np.float64,
    )
    return ((delta >= thr[:, None]) & (valid > 0)).astype(np.float64)


def evaluate_model(
    model: Baseline,
    dataset: WindowDataset,
    batch_size: int = 256,
    ramp_thresholds: dict[str, float] | None = None,
    collect_losses: bool = False,
    transform=None,
) -> dict:
    """Evaluate on daylight, valid future steps; macro-average over plants.

    With ``collect_losses=True`` the result carries one masked-MAE loss per
    window plus its plant and day key — the per-sample loss differentials
    required by the DM test and the paired block bootstrap (§4.5).

    ``transform`` is an optional eval-time control (common.controls): it
    manipulates the model's *inputs*; metrics are computed against the
    untouched targets and masks.
    """
    acc = PerPlantAccumulator()
    losses: dict[str, list[np.ndarray]] = {"loss": [], "plant": [], "day": []}
    preds_list = []
    trues_list = []
    plants_list = []
    for batch in dataset.iter_batches(batch_size):
        forecast: Forecast = model.predict(
            transform(batch) if transform is not None else batch
        )
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
            ramp_mask=(
                _ramp_mask(batch, ramp_thresholds)
                if ramp_thresholds is not None else None
            ),
        )
        if collect_losses:
            t = batch["y_hist"].shape[1]
            losses["loss"].append(
                (np.abs(point - batch["y_future"]) * mask).sum(axis=1)
                / np.maximum(mask.sum(axis=1), 1.0)
            )
            losses["plant"].append(batch["site_id"])
            losses["day"].append(batch["timestamps"][:, t] // 86_400)
        
        preds_list.append(point)
        trues_list.append(batch["y_future"])
        plants_list.append(batch["site_id"])
        
    results = {"overall": acc.macro(), "per_plant": acc.per_plant()}
    if collect_losses:
        results["per_sample"] = {
            k: np.concatenate(v) for k, v in losses.items()
        }
        
    if preds_list:
        from pathlib import Path
        pred_out_dir = Path("results/predictions")
        pred_out_dir.mkdir(parents=True, exist_ok=True)
        all_preds = np.concatenate(preds_list, axis=0)
        all_trues = np.concatenate(trues_list, axis=0)
        all_plants = np.concatenate(plants_list, axis=0)
        for plant in np.unique(all_plants):
            rows = all_plants == plant
            plant_preds = all_preds[rows]
            plant_trues = all_trues[rows]
            np.savez(
                pred_out_dir / f"{model.name}_{plant}_pred.npz",
                pred=plant_preds.astype(np.float32),
                true=plant_trues.astype(np.float32)
            )
    return results


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
    per_sample = results.pop("per_sample", None)
    if per_sample is not None:
        # per-window losses for the DM test / block bootstrap (§4.5);
        # kept out of the JSON, written as a sidecar npz
        np.savez_compressed(out / f"{model_name}_losses.npz", **per_sample)
    path = out / f"{model_name}.json"
    path.write_text(
        json.dumps({"manifest": manifest, "results": results}, indent=2) + "\n"
    )
    return path
