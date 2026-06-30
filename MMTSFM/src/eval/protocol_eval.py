"""Protocol-aligned evaluation for MMTSFM (BASELINE_PROTOCOL.md §5).

Reuses ``baselines/common`` metrics so MMTSFM's NMAE / NRMSE / per-horizon /
CRPS and the Skill Score vs Smart Persistence are computed by the *identical*
code as every baseline (per-plant macro-average, mask = mask_future·daylight).
The result JSON is written in the baselines schema so ``scripts/aggregate_all.py``
ingests it next to the other models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Reuse pv_record's repo-root locator to put baselines/common on sys.path.
from mmtsfm.data.pv_record import _baselines_dir  # noqa: F401  (side effect: path)

from common import config  # noqa: E402
from common.metrics import PerPlantAccumulator, skill_score  # noqa: E402
from common.runner import write_results  # noqa: E402

_Q50 = list(config.QUANTILE_LEVELS).index(0.5)


def default_reference_path() -> Path:
    """Committed Smart-Persistence reference written by the baselines suite."""
    return (
        _baselines_dir().parent
        / "baselines"
        / "results"
        / "smart_persistence_s2_ukpv.json"
    )


class ProtocolEvaluator:
    """Accumulates test predictions and reports protocol metrics + Skill Score."""

    def __init__(
        self,
        horizon: int,
        reference_path: str | None = None,
        compute_marginal_gain: bool = False,
    ):
        self.acc = PerPlantAccumulator()
        self.H = int(horizon)
        self.reference_path = reference_path
        # W6: when enabled, a second accumulator collects the vision-off pass so
        # finalize() can report the visual marginal gain (Δ on/off).
        self.compute_marginal_gain = compute_marginal_gain
        if self.compute_marginal_gain:
            self.acc_off = PerPlantAccumulator()

    def update(
        self,
        site_ids: list[str],
        y_true: np.ndarray,  # (B, H)
        median: np.ndarray,  # (B, H)
        mask: np.ndarray,  # (B, H) = mask_future · daylight
        quantiles: np.ndarray | None = None,  # (B, H, Q)
        vision_off: bool = False,
    ) -> None:
        target_acc = (
            self.acc_off if (self.compute_marginal_gain and vision_off) else self.acc
        )
        target_acc.update(
            plants=np.asarray([str(s) for s in site_ids]),
            y_true=np.asarray(y_true, dtype=np.float64),
            y_pred=np.asarray(median, dtype=np.float64),
            mask=np.asarray(mask, dtype=np.float64),
            quantile_preds=None
            if quantiles is None
            else np.asarray(quantiles, np.float64),
        )

    def _reference_nrmse(self) -> tuple[float | None, dict]:
        path = Path(self.reference_path or default_reference_path())
        if not path.exists():
            return None, {}
        ref = __import__("json").loads(path.read_text())["results"]
        return ref.get("overall", {}).get("nrmse"), ref.get("per_plant", {})

    def finalize(self) -> dict:
        results = {"overall": self.acc.macro(), "per_plant": self.acc.per_plant()}
        ref_nrmse, ref_per_plant = self._reference_nrmse()
        if ref_nrmse:
            results["overall"]["skill_score"] = skill_score(
                results["overall"]["nrmse"], ref_nrmse
            )
            for plant, row in results["per_plant"].items():
                r = ref_per_plant.get(plant)
                if r and r.get("nrmse"):
                    row["skill_score"] = skill_score(row["nrmse"], r["nrmse"])

        if self.compute_marginal_gain:
            overall_off = self.acc_off.macro()
            per_plant_off = self.acc_off.per_plant()

            results["overall"]["nmae_vision_on"] = results["overall"]["nmae"]
            results["overall"]["nmae_vision_off"] = overall_off["nmae"]
            results["overall"]["delta_nmae"] = (
                overall_off["nmae"] - results["overall"]["nmae"]
            )

            results["overall"]["nrmse_vision_on"] = results["overall"]["nrmse"]
            results["overall"]["nrmse_vision_off"] = overall_off["nrmse"]
            results["overall"]["delta_nrmse"] = (
                overall_off["nrmse"] - results["overall"]["nrmse"]
            )

            for plant, row in results["per_plant"].items():
                row_off = per_plant_off.get(plant, {})
                row["nmae_vision_on"] = row["nmae"]
                row["nmae_vision_off"] = row_off.get("nmae")
                if row_off.get("nmae") is not None:
                    row["delta_nmae"] = row_off["nmae"] - row["nmae"]

                row["nrmse_vision_on"] = row["nrmse"]
                row["nrmse_vision_off"] = row_off.get("nrmse")
                if row_off.get("nrmse") is not None:
                    row["delta_nrmse"] = row_off["nrmse"] - row["nrmse"]

        return results

    def write(
        self,
        out_dir: str,
        model_name: str,
        run_config: dict,
        data_path: str = config.DEFAULT_DATA_PATH,
    ) -> Path:
        return write_results(
            out_dir, model_name, self.finalize(), run_config, data_path
        )
