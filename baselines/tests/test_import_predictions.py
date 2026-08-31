"""Unit test for scripts/import_predictions.py (npz → our results JSON glue)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "import_predictions", SCRIPTS / "import_predictions.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_site_parsing():
    m = _load()
    assert m.site_of(Path("uk_pv_test_3432_pred.npz")) == "3432"
    assert m.site_of(Path("visionts_pp_6648_pred.npz")) == "6648"


def test_import_writes_results_json(tmp_path, monkeypatch):
    m = _load()
    rng = np.random.default_rng(0)
    for site in ("3432", "6648"):
        true = rng.random((30, 12)).astype("float32")
        true[:, ::4] = 0.0                     # night zeros → masked out
        pred = np.clip(true + 0.02, 0, 1).astype("float32")
        np.savez(tmp_path / f"visionts_pp_{site}_pred.npz", pred=pred, true=true)

    monkeypatch.setattr(sys, "argv", [
        "import_predictions.py", "--model", "visionts_pp", "--tag", "t",
        "--glob", str(tmp_path / "visionts_pp_*_pred.npz"), "--out", str(tmp_path),
    ])
    m.main()

    out = json.loads((tmp_path / "visionts_pp_t.json").read_text())
    o = out["results"]["overall"]
    assert o["n_plants"] == 2
    assert o["nmae"] >= 0 and o["nrmse"] >= 0
    assert set(out["results"]["per_plant"]) == {"3432", "6648"}
    # caveats recorded in the manifest
    cfg = out["manifest"]["config"]
    assert "proxy true>0" in cfg["daylight_mask"]
    assert "not aligned" in cfg["eval_windows"]


def test_exact_mask_recovers_clearsky_and_nan_gaps(tmp_path):
    """The CSV bridge fills gaps with 0.0 and drops clear-sky, so the `true>0`
    proxy scores outages as night and hides overcast daytime steps. The exact
    mask must come back from the parquet, and must refuse to guess."""
    import pandas as pd

    from common import config

    m = _load()
    n_rows, seq_len, h = 200, 48, 12
    dates = pd.date_range("2021-06-01", periods=n_rows, freq="30min", tz="UTC")
    power = np.abs(np.sin(np.arange(n_rows) / 10.0)).round(4)
    power[[50, 51, 120]] = np.nan                      # real data gaps
    clearsky = np.where((np.arange(n_rows) % 48) < 20, 0.0, 500.0)
    ot = np.nan_to_num(power)                          # what the exporter writes
    pd.DataFrame({"date": dates, "OT": ot}).to_csv(
        tmp_path / "uk_pv_test_S1.csv", index=False)

    n = n_rows - seq_len - h + 1
    idx = np.arange(n)[:, None] + seq_len + np.arange(h)[None, :]
    true = ot[idx]
    source = {"S1": pd.DataFrame(
        {config.TARGET_COL: power, config.CLEARSKY_COL: clearsky}, index=dates)}

    mask = m.exact_mask(true, "S1", tmp_path, source)
    expected = ((~np.isnan(power)) & (clearsky > 0))[idx].astype(np.float64)
    assert mask is not None and np.array_equal(mask, expected)
    assert not np.array_equal(mask, (true > 0).astype(np.float64))

    # window geometry is INFERRED, so a failed self-check must return None
    # (caller falls back to the proxy) rather than a silently misaligned mask
    assert m.exact_mask(true + 0.5, "S1", tmp_path, source) is None
    assert m.exact_mask(true, "absent_site", tmp_path, source) is None
