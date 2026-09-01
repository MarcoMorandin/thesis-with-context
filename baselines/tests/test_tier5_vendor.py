"""Provenance guard for the vendored original Tier-5 multimodal-TS baselines.

Like test_tier4_vendor.py: does not import the heavy upstream stacks (CLIP / uni2ts /
Chronos / Aurora, GPU + conflicting deps — see baselines/README.md);
only asserts the unmodified sources + provenance/licensing notice are present.
"""

from __future__ import annotations

from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "tier5" / "vendor"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

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
    assert (VENDOR / "time_vlm" / "run.py").is_file()  # TSLib harness
    assert (VENDOR / "unicast" / "test_multi_modal_chronos.py").is_file()
    assert (VENDOR / "aurora" / "runner.py").is_file()
    assert any((VENDOR / "visionts_pp").rglob("batch_evaluate.py"))


def test_integration_doc_exists():
    # The tier-5 recipe lives in the baselines README (the old standalone
    # docs/experiments/TIER5_INTEGRATION.md was folded into it).
    doc = VENDOR.parents[1] / "README.md"
    assert doc.is_file()
    text = doc.read_text()
    assert "Time-VLM" in text and "multimodal track" in text


def test_dedicated_slurm_scripts_present():
    scripts = VENDOR.parents[1] / "scripts"
    for name in (
        "slurm_time_vlm.sh",
        "slurm_visionts_pp.sh",
        "slurm_unicast.sh",
        "slurm_aurora.sh",
    ):
        assert (scripts / name).is_file(), f"missing dedicated SLURM script: {name}"


def test_adaptations_present():
    # VisionTS++ zero-shot uk_pv runner we added
    assert (VENDOR / "visionts_pp" / "run_ukpv.py").is_file()
    # Time-VLM prediction-dump patch (contract-format npz)
    exp = (VENDOR / "time_vlm" / "exp" / "exp_long_term_forecasting.py").read_text()
    assert "_pred.npz" in exp and "PVTSFM adaptation" in exp
    # VENDOR_NOTICE documents that the code is no longer pristine
    assert "NO LONGER pristine" in (VENDOR / "VENDOR_NOTICE.md").read_text()


def test_time_vlm_uses_committed_split_loader():
    """`Dataset_Custom` would cut its own 70/10/20 split out of every uk_pv CSV,
    which breaks the disjoint cross-plant protocol three ways (see the class
    docstring). The `ukpv` loader must be present, registered, and wired up."""
    tvlm = VENDOR / "time_vlm"
    loader = (tvlm / "data_provider" / "data_loader.py").read_text()
    assert "class Dataset_UKPV" in loader
    assert "uk_pv_train_protocol.csv" in loader and "uk_pv_val_protocol.csv" in loader

    factory = (tvlm / "data_provider" / "data_factory.py").read_text()
    assert "'ukpv': Dataset_UKPV" in factory

    # the exporter must actually produce the two protocol files
    export = (VENDOR.parents[1] / "tier4" / "vendor" / "export_ukpv.py").read_text()
    assert "uk_pv_train_protocol.csv" in export and "uk_pv_val_protocol.csv" in export

    # and the recipe must select it (plus the 30-min solar period)
    sh = (VENDOR.parents[1] / "scripts" / "slurm_time_vlm.sh").read_text()
    assert "--data ukpv" in sh and "--periodicity 48" in sh

    # prompt bank entry, keyed on args.data by utils.tools.load_content
    assert (tvlm / "dataset" / "prompt_bank" / "ukpv.txt").is_file()


def test_time_vlm_seed_is_parsed_not_hardcoded():
    run = (VENDOR / "time_vlm" / "run.py").read_text()
    assert "fix_seed = 2024" not in run, "upstream seeded before parse_args()"
    assert "random.seed(args.seed)" in run


def test_time_vlm_training_is_resumable():
    """A walltime kill must cost one epoch, not the run. Upstream saves only
    best-val weights, so optimizer moments / epoch / early-stop counters die
    with the job."""
    tvlm = VENDOR / "time_vlm"
    assert "--resume" in (tvlm / "run.py").read_text()
    exp = (tvlm / "exp" / "exp_long_term_forecasting.py").read_text()
    assert "resume.pth" in exp
    for key in ("'optimizer': model_optim.state_dict()", "'epoch': epoch + 1",
                "'counter': early_stopping.counter"):
        assert key in exp, f"resume state missing {key}"
    assert "for epoch in range(start_epoch, self.args.train_epochs)" in exp
    # the per-epoch 'test' pass re-scored the TRAIN csv under --data ukpv
    assert "if self.args.data != 'ukpv':" in exp
    assert 'RESUME="${RESUME:-0}"' in (
        SCRIPTS / "slurm_time_vlm.sh").read_text()
