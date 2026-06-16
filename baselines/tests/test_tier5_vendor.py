"""Provenance guard for the vendored original Tier-5 multimodal-TS baselines.

Like test_tier4_vendor.py: does not import the heavy upstream stacks (CLIP / uni2ts /
Chronos / Aurora, GPU + conflicting deps — see docs/experiments/TIER5_INTEGRATION.md);
only asserts the unmodified sources + provenance/licensing notice are present.
"""

from __future__ import annotations

from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "tier5" / "vendor"

SHAS = {
    "time_vlm": "796e6ec963788657207ea2b5553740993ea3ea2b",
    "visionts_pp": "484b2ea363b497217d0c3a078494c6af0251c275",
    "unicast": "a4af694615fabb9844a1a0f297aca148a3ab9db8",
    "aurora": "a247760abbc9d17a861bc365c032368d317815f2",
}


def test_vendor_notice_records_all_shas():
    notice = (VENDOR / "VENDOR_NOTICE.md").read_text()
    for sha in SHAS.values():
        assert sha in notice
    assert "none stated" in notice  # the 3 missing-license caveats stay visible


def test_all_four_vendor_dirs_present():
    for name in SHAS:
        d = VENDOR / name
        assert d.is_dir(), f"missing vendored Tier-5 dir: {name}"
        assert any(d.rglob("*.py")), f"no python sources under {name}"


def test_entry_points_present():
    assert (VENDOR / "time_vlm" / "run.py").is_file()        # TSLib harness
    assert (VENDOR / "unicast" / "test_multi_modal_chronos.py").is_file()
    assert (VENDOR / "aurora" / "runner.py").is_file()
    assert any((VENDOR / "visionts_pp").rglob("batch_evaluate.py"))


def test_integration_doc_exists():
    doc = (VENDOR.parents[2] / "docs" / "experiments" / "TIER5_INTEGRATION.md")
    assert doc.is_file()
    text = doc.read_text()
    assert "Time-VLM" in text and "multimodal track" in text


def test_dedicated_slurm_scripts_present():
    scripts = VENDOR.parents[1] / "scripts"
    for name in ("slurm_time_vlm.sh", "slurm_visionts_pp.sh",
                 "slurm_unicast.sh", "slurm_aurora.sh"):
        assert (scripts / name).is_file(), f"missing dedicated SLURM script: {name}"


def test_adaptations_present():
    # VisionTS++ zero-shot uk_pv runner we added
    assert (VENDOR / "visionts_pp" / "run_ukpv.py").is_file()
    # Time-VLM prediction-dump patch (contract-format npz)
    exp = (VENDOR / "time_vlm" / "exp" / "exp_long_term_forecasting.py").read_text()
    assert "_pred.npz" in exp and "PVTSFM adaptation" in exp
    # VENDOR_NOTICE documents that the code is no longer pristine
    assert "NO LONGER pristine" in (VENDOR / "VENDOR_NOTICE.md").read_text()
