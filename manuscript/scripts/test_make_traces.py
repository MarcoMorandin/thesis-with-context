"""Check that US trace panels compare identical saved targets."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_traces as traces


@pytest.fixture
def exports(tmp_path):
    stems = [
        "mmtsfm_s1_goespvdaq_selfattn_all_s42",
        "mmtsfm_s2a_goespvdaq_selfattn_all_s42",
        "mmtsfm_s2d_goespvdaq_s2d_all_s42",
        "mmtsfm_s2e_goespvdaq_s2e_all_s42",
    ]
    for plant in ("1202", "1203", "1277"):
        true = np.stack([np.linspace(0.1, end, 12) for end in (0.2, 0.6, 0.9)])
        mask = np.ones_like(true, dtype=bool)
        mask[2, -1] = False  # Largest swing is ineligible.
        for stem in stems:
            np.savez(tmp_path / f"{stem}_{plant}_pred.npz",
                     pred=true + 0.01, true=true, mask=mask)
    return tmp_path, stems


def test_us_figure_selects_observed_swing_and_writes_panels(exports, tmp_path):
    pred_dir, _ = exports
    output = tmp_path / "traces.png"
    selected = traces.us_figure(pred_dir, output)
    assert selected == [("1202", 1), ("1203", 1), ("1277", 1)]
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("field", ["true", "mask", "pred"])
def test_us_figure_rejects_incompatible_exports(exports, tmp_path, field):
    pred_dir, stems = exports
    path = pred_dir / f"{stems[-1]}_1202_pred.npz"
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if field == "true":
        arrays[field][0, 0] += 0.1
    elif field == "mask":
        arrays[field][0, 0] = False
    else:
        arrays[field] = arrays[field][:, :-1]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="unaligned"):
        traces.us_figure(pred_dir, tmp_path / "bad.png")
