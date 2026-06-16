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
