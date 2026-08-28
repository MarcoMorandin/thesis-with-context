"""Protocol-aligned evaluation for MMTSFM (knowledge/protocol.md §5).

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
        # Free-form diagnostics a caller wants carried into the results JSON
        # (ticket 15: s2c horizon-attention). Merged inside finalize() rather than
        # into the dict it returns, because write() calls finalize() a SECOND time
        # internally -- anything added to the returned dict never reaches the file.
        self.extra: dict = {}
        self.reference_path = reference_path
        # W6: when enabled, a second accumulator collects the vision-off pass so
        # finalize() can report the visual marginal gain (Δ on/off).
        self.compute_marginal_gain = compute_marginal_gain
        if self.compute_marginal_gain:
            self.acc_off = PerPlantAccumulator()
        # Raw per-site buffers (vision-on pass only). Ramp thresholds are a
        # per-site top-decile over the WHOLE test set, so the S6 ramp metrics
        # cannot be streamed — they need a second pass in finalize(). The same
        # buffers back the per-site prediction npz dump in write().
        self._store: dict[str, dict[str, list[np.ndarray]]] = {}
        # Vision-OFF buffers, kept separate so the on-pass dump and its ramp
        # numbers stay byte-identical when compute_marginal_gain is off. The
        # ramp SUBSET is never recomputed from these: thresholds are a property
        # of the data, so finalize() derives them once from the on-pass store
        # and scores both passes against them (see _ramp_metrics).
        self._store_off: dict[str, dict[str, list[np.ndarray]]] = {}

    def update(
        self,
        site_ids: list[str],
        y_true: np.ndarray,  # (B, H)
        median: np.ndarray,  # (B, H)
        mask: np.ndarray,  # (B, H) = mask_future · daylight
        quantiles: np.ndarray | None = None,  # (B, H, Q)
        vision_off: bool = False,
        delta: np.ndarray | None = None,  # (B, H) |Δy| vs previous step
        delta_valid: np.ndarray | None = None,  # (B, H) mask·daylight·prev_mask
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
        if not vision_off:
            self._store_batch(site_ids, y_true, median, mask, delta, delta_valid)
        elif self.compute_marginal_gain:
            self._store_batch(
                site_ids,
                y_true,
                median,
                mask,
                delta,
                delta_valid,
                store=self._store_off,
            )

    def _store_batch(
        self, site_ids, y_true, median, mask, delta, delta_valid, store=None
    ):
        if store is None:
            store = self._store
        arrs = {
            "true": np.asarray(y_true, dtype=np.float32),
            "pred": np.asarray(median, dtype=np.float32),
            "mask": np.asarray(mask, dtype=np.float32),
        }
        if delta is not None and delta_valid is not None:
            arrs["delta"] = np.asarray(delta, dtype=np.float32)
            arrs["delta_valid"] = np.asarray(delta_valid, dtype=np.float32)
        sites = np.asarray([str(s) for s in site_ids])
        for site in np.unique(sites):
            rows = sites == site
            bucket = store.setdefault(str(site), {k: [] for k in arrs})
            for k, v in arrs.items():
                bucket.setdefault(k, []).append(v[rows])

    def _ramp_metrics(self) -> dict[str, dict[str, float]]:
        """Per-site S6 ramp NMAE/NRMSE (protocol rule: top-decile |Δy| per site).

        Same subset rule as ``baselines/common/runner`` (delta vs previous step
        incl. the last history step, validity = mask·daylight·prev_mask,
        per-site top-decile threshold over the full test set) — the MMTSFM
        windows are protocol-aligned, so these ramp numbers are comparable with
        tiers 0-3, unlike the T4-T6 native-window proxies.
        """
        return self._score_ramp(self._store, self._ramp_masks())

    def _ramp_masks(self) -> dict[str, np.ndarray]:
        """Per-site ramp subset, derived ONCE from the vision-on store.

        The threshold is a property of the data, not of a model — exactly as in
        ``baselines/common/runner.compute_ramp_thresholds``, which computes it
        once per eval split and shares it across every baseline. Deriving it
        per pass would score vision-on and vision-off on different subsets and
        make their difference meaningless.
        """
        masks: dict[str, np.ndarray] = {}
        for site, bucket in self._store.items():
            # ramp needs delta for EVERY stored batch — a mixed stream (some
            # updates without history) would misalign delta rows with pred/true
            if not bucket.get("delta") or len(bucket["delta"]) != len(bucket["true"]):
                continue
            delta = np.concatenate(bucket["delta"], axis=0).astype(np.float64)
            valid = np.concatenate(bucket["delta_valid"], axis=0).astype(np.float64)
            if (valid > 0).sum() == 0:
                continue
            thr = float(np.quantile(delta[valid > 0], 0.9))
            masks[site] = ((delta >= thr) & (valid > 0)).astype(np.float64)
        return masks

    @staticmethod
    def _score_ramp(
        store: dict[str, dict[str, list[np.ndarray]]],
        ramp_masks: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Score one pass's predictions on the supplied per-site ramp subsets."""
        out: dict[str, dict[str, float]] = {}
        acc = PerPlantAccumulator()
        for site, ramp in ramp_masks.items():
            bucket = store.get(site)
            if not bucket:
                continue
            true = np.concatenate(bucket["true"], axis=0).astype(np.float64)
            pred = np.concatenate(bucket["pred"], axis=0).astype(np.float64)
            mask = np.concatenate(bucket["mask"], axis=0).astype(np.float64)
            # A pass that saw a different number of windows cannot be aligned to
            # this subset; skip rather than score the wrong rows.
            if true.shape != ramp.shape:
                continue
            acc.update(
                plants=np.asarray([site] * len(true)),
                y_true=true,
                y_pred=pred,
                mask=mask,
                ramp_mask=ramp,
            )
        for site, row in acc.per_plant().items():
            if "nmae_ramp" in row:
                out[site] = {
                    "nmae_ramp": row["nmae_ramp"],
                    "nrmse_ramp": row["nrmse_ramp"],
                }
        return out

    def dump_predictions(self, out_dir: str, model_name: str) -> Path | None:
        """Write per-site raw pred/true/mask npz (baselines predictions layout)."""
        if not self._store:
            return None
        pred_dir = Path(out_dir) / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for site, bucket in self._store.items():
            np.savez(
                pred_dir / f"{model_name}_{site}_pred.npz",
                pred=np.concatenate(bucket["pred"], axis=0),
                true=np.concatenate(bucket["true"], axis=0),
                mask=np.concatenate(bucket["mask"], axis=0),
            )
        # Vision-off pass in its own file, so `localize.decompose_by_horizon`
        # has a pred_off to read and existing consumers of the on-pass npz are
        # untouched. Absent entirely when compute_marginal_gain is off.
        for site, bucket in self._store_off.items():
            np.savez(
                pred_dir / f"{model_name}_{site}_pred_off.npz",
                pred=np.concatenate(bucket["pred"], axis=0),
                true=np.concatenate(bucket["true"], axis=0),
                mask=np.concatenate(bucket["mask"], axis=0),
            )
        return pred_dir

    def _reference_nrmse(self) -> tuple[float | None, dict]:
        path = Path(self.reference_path or default_reference_path())
        if not path.exists():
            return None, {}
        ref = __import__("json").loads(path.read_text())["results"]
        return ref.get("overall", {}).get("nrmse"), ref.get("per_plant", {})

    def finalize(self) -> dict:
        results = {"overall": self.acc.macro(), "per_plant": self.acc.per_plant()}
        ramp_masks = self._ramp_masks()
        ramp = self._score_ramp(self._store, ramp_masks)
        if ramp:
            for plant, row in results["per_plant"].items():
                if plant in ramp:
                    row.update(ramp[plant])
            results["overall"]["nmae_ramp"] = float(
                np.mean([r["nmae_ramp"] for r in ramp.values()])
            )
            results["overall"]["nrmse_ramp"] = float(
                np.mean([r["nrmse_ramp"] for r in ramp.values()])
            )
            # Ramp decomposed across the visual passes, on the SAME subsets.
            # Without this the project's P0 metric is never attributed to the
            # visual stream: the aggregate marginal gain says nothing about the
            # top-decile |Δy| rows, which are the ones vision is supposed to buy.
            if self.compute_marginal_gain:
                ramp_off = self._score_ramp(self._store_off, ramp_masks)
                if ramp_off:
                    for key in ("nmae_ramp", "nrmse_ramp"):
                        on_v = results["overall"][key]
                        off_v = float(np.mean([r[key] for r in ramp_off.values()]))
                        results["overall"][f"{key}_vision_on"] = on_v
                        results["overall"][f"{key}_vision_off"] = off_v
                        results["overall"][f"delta_{key}"] = off_v - on_v
                    for plant, row in results["per_plant"].items():
                        if plant not in ramp or plant not in ramp_off:
                            continue
                        for key in ("nmae_ramp", "nrmse_ramp"):
                            row[f"{key}_vision_on"] = ramp[plant][key]
                            row[f"{key}_vision_off"] = ramp_off[plant][key]
                            row[f"delta_{key}"] = (
                                ramp_off[plant][key] - ramp[plant][key]
                            )
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

        # Last, and never overwriting a computed metric: diagnostics are evidence
        # ABOUT the run, not part of the protocol's metric contract.
        for key, value in self.extra.items():
            results.setdefault(key, value)
        return results

    def write(
        self,
        out_dir: str,
        model_name: str,
        run_config: dict,
        data_path: str = config.DEFAULT_DATA_PATH,
    ) -> Path:
        path = write_results(
            out_dir, model_name, self.finalize(), run_config, data_path
        )
        try:
            self.dump_predictions(out_dir, model_name)
        except Exception as e:  # dump is auxiliary — never fail the results write
            print(f"[protocol-eval] prediction dump skipped: {e}", flush=True)
        return path
