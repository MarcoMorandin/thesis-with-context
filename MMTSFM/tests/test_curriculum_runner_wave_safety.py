"""Wave safety for ``scripts/slurm_curriculum.sh``.

Wave 1 submits five chains that differ only in ``MODEL_CFG`` and ``SEED``. If the
run tag or the checkpoint directory does not carry both, the chains share a
results JSON and a checkpoint dir, clobbering each other's ``best.ckpt`` mid-run
while ``afterok`` chains stages onto whichever file won the race — five chains of
compute, one corrupted answer, and no error raised.

These drive the script through ``DRY_RUN=1``, which prints the planned
submissions instead of calling ``sbatch``, so the guarantee is assertable
without a cluster.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "slurm_curriculum.sh"

# (MODEL_CFG, SEED) for the five wave-1 arms; grassmann@42 already exists.
WAVE_1 = [
    ("vision_chronos2_grassmann", "43"),
    ("vision_chronos2_grassmann", "44"),
    ("vision_chronos2_timeselfattn", "42"),
    ("vision_chronos2_timeselfattn", "43"),
    ("vision_chronos2_timeselfattn", "44"),
]


def _plan(tmp_path: Path, **env_overrides: str) -> str:
    """Run the submitter in dry-run mode and return its printed plan."""
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "DATA_DIR": str(tmp_path / "data"),
            "CKPT_DIR": str(tmp_path / "ckpt"),
            "RESULTS_DIR": str(tmp_path / "results"),
            "VJEPA_CACHE_ROOT": str(tmp_path / "cache"),
        }
    )
    env.update(env_overrides)
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        cwd=SCRIPT.parent.parent,
    )
    assert proc.returncode == 0, f"dry run failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def _field(plan: str, key: str) -> list[str]:
    return re.findall(rf"^\s*{key}=(\S+)\s*$", plan, flags=re.MULTILINE)


def test_wave_1_arms_get_distinct_tags_and_checkpoint_dirs(tmp_path):
    tags: list[str] = []
    dirs: list[str] = []
    for model_cfg, seed in WAVE_1:
        plan = _plan(tmp_path, MODEL_CFG=model_cfg, SEED=seed, END_STAGE="s2b")
        tags.extend(_field(plan, "TAG"))
        dirs.extend(_field(plan, "STAGE_DIR"))

    assert tags, "dry run printed no TAG lines"
    assert len(tags) == len(set(tags)), f"tag collision across wave-1 arms: {tags}"
    assert len(dirs) == len(set(dirs)), f"checkpoint dir collision: {dirs}"


def test_canonical_arm_keeps_its_historical_tag(tmp_path):
    """grassmann@42 must still write mmtsfm_s2b_ukpv.json.

    That file is the baseline the A03 gate compares against and is referenced by
    ALL_RESULTS and the manuscript; renaming it would orphan them.
    """
    plan = _plan(
        tmp_path, MODEL_CFG="vision_chronos2_grassmann", SEED="42", END_STAGE="s2b"
    )
    assert "mmtsfm_s2b_ukpv" in _field(plan, "TAG")


def test_end_stage_stops_the_chain(tmp_path):
    plan = _plan(tmp_path, END_STAGE="s2b")
    stages = _field(plan, "STAGE")
    assert stages == ["s1", "s2a", "s2b"], stages
    assert "s3" not in stages


def test_submitter_exports_marginal_gain(tmp_path):
    """The submitter's job is to EXPORT the flag; translating it is the stage's."""
    off = _plan(tmp_path, END_STAGE="s2b")
    assert "MARGINAL_GAIN=1" not in off
    on = _plan(tmp_path, END_STAGE="s2b", MARGINAL_GAIN="1")
    assert "MARGINAL_GAIN=1" in on


def test_submitter_exports_the_resolved_stage_dir(tmp_path):
    """curriculum_stage.sbatch must not recompute STAGE_DIR.

    It used to derive `${CKPT_DIR}/${DS}_${STAGE}` itself, which drops the arm
    suffix and puts every chain of a wave back into one checkpoint directory —
    defeating the tag fix entirely.
    """
    plan = _plan(
        tmp_path, MODEL_CFG="vision_chronos2_timeselfattn", SEED="43", END_STAGE="s2b"
    )
    assert "STAGE_DIR=" in plan
    for line in _field(plan, "STAGE_DIR"):
        assert line.endswith("_selfattn_s43"), line


def _stage_cmd(tmp_path: Path, **env_overrides: str) -> str:
    """Drive curriculum_stage.sbatch in dry-run and return the composed command."""
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "STAGE": "s2b",
            "DS": "uk_pv",
            "DCFG": "ukpv",
            "TAG": "t",
            "DATA_DIR": str(tmp_path / "data"),
            "CKPT_DIR": str(tmp_path / "ckpt"),
            "RESULTS_DIR": str(tmp_path / "results"),
        }
    )
    env.update(env_overrides)
    stage = SCRIPT.parent / "curriculum_stage.sbatch"
    proc = subprocess.run(
        ["bash", str(stage)],
        env=env,
        capture_output=True,
        text=True,
        cwd=SCRIPT.parent.parent,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_stage_translates_marginal_gain_to_hydra_override(tmp_path):
    assert "compute_marginal_gain=true" not in _stage_cmd(tmp_path)
    assert "compute_marginal_gain=true" in _stage_cmd(tmp_path, MARGINAL_GAIN="1")


def test_stage_honours_the_exported_stage_dir(tmp_path):
    out = _stage_cmd(tmp_path, STAGE_DIR=str(tmp_path / "ckpt" / "uk_pv_s2b_selfattn_s43"))
    assert "uk_pv_s2b_selfattn_s43" in out


def _stage_rc(tmp_path: Path, **env_overrides: str):
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1", "STAGE": "s2b", "DS": "uk_pv", "DCFG": "ukpv", "TAG": "t",
            "DATA_DIR": "/data_v2",
            "CKPT_DIR": str(tmp_path / "ckpt"),
            "RESULTS_DIR": str(tmp_path / "results"),
        }
    )
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SCRIPT.parent / "curriculum_stage.sbatch")],
        env=env, capture_output=True, text=True, cwd=SCRIPT.parent.parent,
    )


def test_missing_cache_dir_is_fatal(tmp_path):
    """A requested cache that isn't there must not silently fall back to live encode.

    The old behaviour dropped the override and logged a WARN, so a typo'd
    VJEPA_CACHE_VER cost ~10x the compute with nothing but a warning in a log
    nobody reads until the job is done.
    """
    r = _stage_rc(tmp_path, VJEPA_CACHE=str(tmp_path / "nope"))
    assert r.returncode != 0
    assert "cache" in (r.stderr + r.stdout).lower()


def test_cache_without_provenance_is_fatal(tmp_path):
    """No _extract_meta.txt means unknown imagery. Refuse rather than guess.

    The stale vit_large_f8_s224 cache sits at the DEFAULT path. If it predates
    the extractor writing provenance there is no mismatch guard at all, and a
    wave would train on v1 HRV latents against a v2 DATA_DIR without a word.
    """
    cache = tmp_path / "cache_no_meta"
    cache.mkdir()
    r = _stage_rc(tmp_path, VJEPA_CACHE=str(cache))
    assert r.returncode != 0
    assert "_extract_meta.txt" in (r.stderr + r.stdout)


def test_cache_with_matching_provenance_is_accepted(tmp_path):
    cache = tmp_path / "cache_ok"
    cache.mkdir()
    (cache / "_extract_meta.txt").write_text(
        "data_dir=/data_v2|arch=vit_large|frames=8|img=224\n"
    )
    r = _stage_rc(tmp_path, VJEPA_CACHE=str(cache))
    assert r.returncode == 0, r.stderr
    assert f"data.vjepa_cache_dir={cache}" in r.stdout


def test_cache_from_a_different_data_dir_is_fatal(tmp_path):
    cache = tmp_path / "cache_v1"
    cache.mkdir()
    (cache / "_extract_meta.txt").write_text("data_dir=/data|arch=vit_large\n")
    r = _stage_rc(tmp_path, VJEPA_CACHE=str(cache))
    assert r.returncode != 0
    assert "/data" in (r.stderr + r.stdout)
