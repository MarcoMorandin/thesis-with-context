"""Import dumped predictions (Tier-4 RAG / Tier-5 vendored runs) into our metrics.

Each `<...>_<site>_pred.npz` carries `pred` (N,H[,1]) and `true` (N,H[,1]) — the
per-window forecasts the vendored harnesses dump (Time-VLM, VisionTS++/run_ukpv,
and the RAG originals once patched). This reduces them to the SAME result JSON
the in-repo baselines write (`PerPlantAccumulator` → overall + per-plant
NMAE/NRMSE/SS/CRPS) in the same schema as run_eval.py writes.

    uv run python scripts/import_predictions.py --model time_vlm --tag s2_ukpv \
        --glob 'tier5/vendor/time_vlm/results/*/uk_pv_test_*_pred.npz'

Caveats (written into the result manifest):
- Daylight mask: pass `--ukpv_dir <export dir>` (and `--data <parquet>`) to score
  against the EXACT clear-sky daylight + validity mask Tiers 0-3 use, recovered
  by re-aligning each dumped window onto the exported per-plant CSV. Without it
  the mask falls back to the proxy `true > 0` (night norm_power is exactly 0),
  which drops daytime near-zero overcast steps and, because the CSV export fills
  gaps with 0.0, silently scores outages as night.
- Legacy dumps use each harness's native eval windows. Archives marked
  ``protocol_aligned`` (including Aurora) use the common UK-PV windows and can
  retain the last history target for exact first-step ramp scoring.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config  # noqa: E402
from common.metrics import PerPlantAccumulator  # noqa: E402
from common.runner import add_skill_scores, write_results  # noqa: E402


def _2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a


def site_of(path: Path) -> str:
    return path.name[: -len("_pred.npz")].split("_")[-1]


def exact_mask_source(data_path: str, sites: list[str]) -> dict:
    """Per-plant clear-sky + target validity from the parquet, indexed by time.

    The CSV bridge the vendored harnesses consume carries only ``date``/``OT``
    on a dense grid with gaps filled 0.0, so both the clear-sky daylight mask
    and the missing-data mask are destroyed by the export. Recover both from the
    dataset of record so Tiers 4-6 are scored on the SAME mask as Tiers 0-3.
    """
    import pandas as pd

    cols = [
        config.DATASET_COL,
        config.SITE_COL,
        config.TIME_COL,
        config.TARGET_COL,
        config.CLEARSKY_COL,
    ]
    df = pd.read_parquet(data_path, columns=cols)
    df = df[
        (df[config.DATASET_COL] == "uk_pv")
        & (df[config.SITE_COL].astype(str).isin(set(sites)))
    ]
    df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL], utc=True)
    out = {}
    for site, g in df.groupby(config.SITE_COL):
        g = g.sort_values(config.TIME_COL)
        g = g[~g[config.TIME_COL].duplicated(keep="first")]
        out[str(site)] = g.set_index(config.TIME_COL)[
            [config.TARGET_COL, config.CLEARSKY_COL]
        ]
    return out


def exact_mask(true: np.ndarray, site: str, ukpv_dir: Path, source: dict):
    """Exact ``valid & daylight`` mask for the windows in ``true``, or None.

    The npz carries no timestamps, so re-derive them: the vendored loaders slide
    a stride-1 window over the whole per-plant CSV, hence
    ``len(csv) == n_windows + seq_len + horizon - 1`` fixes ``seq_len`` and
    window ``i`` forecasts CSV rows ``[i+seq_len, i+seq_len+H)``. That is an
    inference, so it is VERIFIED against the dumped ``true`` before use; on any
    mismatch return None and let the caller fall back to the proxy mask.
    """
    import pandas as pd

    csv_path = ukpv_dir / f"uk_pv_test_{site}.csv"
    if not csv_path.exists() or site not in source:
        print(f"WARN: {site}: no {csv_path.name} / not in parquet — proxy mask")
        return None

    csv = pd.read_csv(csv_path)
    ot = csv["OT"].to_numpy(dtype=np.float64)
    n, h = true.shape
    seq_len = len(ot) - n - h + 1
    if seq_len < 1:
        print(f"WARN: {site}: {n} windows do not fit {len(ot)} CSV rows — proxy mask")
        return None

    idx = np.arange(n)[:, None] + seq_len + np.arange(h)[None, :]
    if not np.allclose(ot[idx], true, atol=1e-3):
        print(
            f"WARN: {site}: dumped 'true' does not match the CSV at the "
            f"derived offsets (seq_len={seq_len}) — proxy mask"
        )
        return None

    ref = source[site].reindex(pd.to_datetime(csv["date"], utc=True))
    valid = ref[config.TARGET_COL].notna().to_numpy()
    daylight = np.nan_to_num(ref[config.CLEARSKY_COL].to_numpy(dtype=np.float64)) > 0
    return (valid & daylight)[idx].astype(np.float64)


def ramp_mask_from_true(
    true: np.ndarray,
    quantile: float = 0.9,
    mask: np.ndarray | None = None,
    previous: np.ndarray | None = None,
    previous_valid: np.ndarray | None = None,
    target_valid: np.ndarray | None = None,
    daylight: np.ndarray | None = None,
) -> np.ndarray:
    """Model-independent S6 ramp subset from a site's own targets (§4.2).

    Mirrors ``common.runner._ramp_mask``. New archives may provide the previous
    history value and separate validity/daylight arrays for exact reconstruction;
    legacy archives fall back to the dumped ``true`` and combined mask:
    - daylight/validity from ``mask`` when the exact one was recovered, else the
      proxy ``true > 0``;
    - per-step |Δtrue| vs the previous in-window step, top-decile threshold per
      site (thresholds are a property of the data, not the model);
    - step 0 has no in-window predecessor (history is not dumped) so it is
      excluded from the ramp subset.
    Legacy native-window archives remain unaligned with the common protocol.
    """
    if true.shape[1] < 2 and previous is None:
        return np.zeros_like(true)
    if previous is None:
        ok = (true > 0) if mask is None else (mask > 0)
        delta = np.abs(true[:, 1:] - true[:, :-1])
        valid = ok[:, 1:] & ok[:, :-1]
        offset = 1
    else:
        previous = np.asarray(previous).reshape(-1, 1)
        if len(previous) != len(true):
            raise ValueError("previous must have one value per forecast window")
        prev = np.concatenate([previous, true[:, :-1]], axis=1)
        delta = np.abs(true - prev)
        if target_valid is not None and daylight is not None:
            target_valid = np.asarray(target_valid) > 0
            previous_valid = (
                np.ones(len(true), dtype=bool)
                if previous_valid is None
                else np.asarray(previous_valid).reshape(-1) > 0
            )
            prev_valid = np.concatenate(
                [previous_valid[:, None], target_valid[:, :-1]], axis=1
            )
            valid = target_valid & prev_valid & (np.asarray(daylight) > 0)
        else:
            ok = (true > 0) if mask is None else (mask > 0)
            valid = ok
        offset = 0
    if valid.sum() == 0:
        return np.zeros_like(true)
    thr = float(np.quantile(delta[valid], quantile))
    rm = np.zeros_like(true)
    rm[:, offset:] = ((delta >= thr) & valid).astype(np.float64)
    return rm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="result stem, e.g. time_vlm")
    ap.add_argument("--glob", required=True, help="glob for *_<site>_pred.npz")
    ap.add_argument("--tag", default="s2_ukpv")
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--reference",
        default=None,
        help="Smart Persistence result json for SS "
        "(default: <out>/smart_persistence_<tag>.json)",
    )
    ap.add_argument(
        "--ukpv_dir",
        default=None,
        help="export dir holding uk_pv_test_<site>.csv; enables the "
        "EXACT clear-sky+validity mask (else proxy true>0)",
    )
    ap.add_argument(
        "--data",
        default=config.DEFAULT_DATA_PATH,
        help="parquet of record, for --ukpv_dir mask recovery",
    )
    args = ap.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no npz matched: {args.glob}")

    mask_source, ukpv_dir = None, None
    if args.ukpv_dir:
        ukpv_dir = Path(args.ukpv_dir)
        mask_source = exact_mask_source(args.data, [site_of(f) for f in files])
    n_exact = 0
    n_embedded = 0
    n_history = 0
    n_aligned = 0

    acc = PerPlantAccumulator()
    # Centralize every model's raw pred/true under results/predictions/ (same
    # naming as the in-repo runner) so metrics can be recomputed/corrected later
    # without re-running the model.
    pred_dir = Path(args.out) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        data = np.load(f, allow_pickle=False)
        pred, true = _2d(data["pred"]), _2d(data["true"])
        if pred.shape != true.shape:
            raise SystemExit(f"{f.name}: pred {pred.shape} != true {true.shape}")

        site = site_of(f)

        # raw (post-inverse, pre-clip) pred/true for later re-scoring
        np.savez(
            pred_dir / f"{args.model}_{site}_pred.npz",
            pred=pred.astype(np.float32),
            true=true.astype(np.float32),
        )

        mask = _2d(data["valid"]) if "valid" in data else None
        if mask is not None:
            n_embedded += 1
        elif mask_source is not None:
            mask = exact_mask(true, site, ukpv_dir, mask_source)
        if mask is None:
            mask = (true > 0).astype(np.float64)  # daylight proxy
        else:
            n_exact += 1

        q = _2d(data["quantiles"]) if "quantiles" in data else None
        if q is None and "samples" in data:
            q = np.quantile(data["samples"], config.QUANTILE_LEVELS, axis=1)
            q = np.moveaxis(q, 0, -1)
        previous = _2d(data["last_history"]) if "last_history" in data else None
        if previous is not None:
            n_history += 1
        if "protocol_aligned" in data and bool(data["protocol_aligned"].item()):
            n_aligned += 1
        acc.update(
            plants=np.array([site] * len(pred)),
            y_true=true,
            y_pred=np.clip(pred, 0.0, 1.0),
            mask=mask,
            quantile_preds=q,
            ramp_mask=ramp_mask_from_true(
                true,
                mask=mask,
                previous=previous,
                previous_valid=(
                    data["last_history_valid"] if "last_history_valid" in data else None
                ),
                target_valid=(data["target_valid"] if "target_valid" in data else None),
                daylight=(data["daylight"] if "daylight" in data else None),
            ),
        )

    results = {"overall": acc.macro(), "per_plant": acc.per_plant()}

    ref_path = Path(args.reference or f"{args.out}/smart_persistence_{args.tag}.json")
    if not ref_path.exists() and not args.reference:
        fallback_path = Path(args.out) / "smart_persistence_s2.json"
        if fallback_path.exists():
            ref_path = fallback_path

    if ref_path.exists():
        ref = json.loads(ref_path.read_text())["results"]
        results = add_skill_scores(results, ref)
        dest_ref_path = Path(args.out) / f"smart_persistence_{args.tag}.json"
        if not dest_ref_path.exists():
            import shutil

            shutil.copyfile(ref_path, dest_ref_path)
            print(f"Copied smart persistence reference to {dest_ref_path}")
    else:
        print(f"WARN: no Smart Persistence reference at {ref_path}; SS omitted")

    run_config = {
        "model": args.model,
        "tag": args.tag,
        "source": "vendored harness",
        "glob": args.glob,
        "n_plants": len(acc.per_plant()),
        "daylight_mask": (
            f"embedded exact mask ({n_embedded}/{len(files)} plants)"
            if n_embedded == len(files)
            else f"exact clear-sky+validity from {args.data} via {args.ukpv_dir} "
            f"({n_exact}/{len(files)} plants; rest fell back to proxy true>0)"
            if mask_source is not None
            else "proxy true>0 (not exact clear-sky mask)"
        ),
        "ramp_subset": (
            "exact top-decile |Δtrue| per site including step 0"
            if n_history == len(files)
            else "proxy top-decile |Δtrue| per site; step 0 excluded when history "
            "is unavailable"
        ),
        "eval_windows": (
            "protocol-aligned common UK-PV windows"
            if n_aligned == len(files)
            else "native harness split — not aligned with tiers 0-4; no "
            "DM/bootstrap sidecar (compare via SS/rank, §4.4)"
        ),
        "quantile_levels": config.QUANTILE_LEVELS,
    }
    path = write_results(args.out, f"{args.model}_{args.tag}", results, run_config)
    o = results["overall"]
    print(
        f"{args.model}: plants={len(acc.per_plant())} "
        f"NMAE={o.get('nmae', float('nan')):.4f} "
        f"NRMSE={o.get('nrmse', float('nan')):.4f} "
        f"SS={o.get('skill_score', float('nan')):.4f} → {path}"
    )


if __name__ == "__main__":
    main()
