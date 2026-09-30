"""Tests for scripts/eval_us_generalization.sbatch.

Validates command composition, checkpoint resolution, tag naming, and
model-specific options under DRY_RUN=1 without cluster or GPU dependencies.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_us_generalization.sbatch"


def _run_dry_plan(**env_overrides: str) -> str:
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "DATA_DIR": "/test/data_v2",
            "CKPT_DIR": "/test/checkpoints",
            "RESULTS_DIR": "/test/results",
        }
    )
    env.update(env_overrides)
    proc = subprocess.run(
        [str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        cwd=SCRIPT.parent.parent,
    )
    assert proc.returncode == 0, f"dry run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc.stdout


def test_eval_us_generalization_all_models():
    plan = _run_dry_plan(SEEDS="42")
    models = re.findall(r"^>>> \[EVAL\] model=(\S+)", plan, flags=re.MULTILINE)
    assert models == ["itransformer", "s1", "s2a", "s2d", "s2e", "A46b", "timevlm"], f"Unexpected models: {models}"


def test_eval_us_generalization_tags_and_checkpoints():
    plan = _run_dry_plan(SEEDS="42", ONLY="s1 s2a s2d s2e A46b")
    tags = re.findall(r"model\.results_tag=(\S+)", plan)
    ckpts = re.findall(r"ckpt_path=(\S+)", plan)

    assert len(tags) == 5
    assert len(set(tags)) == 5, f"Tag collision: {tags}"
    assert len(set(ckpts)) == 5, f"Checkpoint collision: {ckpts}"

    expected_tags = {
        "mmtsfm_s1_goespvdaq_selfattn_all_s42",
        "mmtsfm_s2a_goespvdaq_selfattn_all_s42",
        "mmtsfm_s2d_goespvdaq_s2d_all_s42",
        "mmtsfm_s2e_goespvdaq_s2e_all_s42",
        "mmtsfm_A46b_s2e_goespvdaq_all_s42",
    }
    assert set(tags) == expected_tags


def test_eval_us_generalization_only_filter():
    plan = _run_dry_plan(ONLY="s1 s2d", SEEDS="42")
    models = re.findall(r"^>>> \[EVAL\] model=(\S+)", plan, flags=re.MULTILINE)
    assert models == ["s1", "s2d"]


def test_eval_us_generalization_batch_sizes_and_flags():
    plan = _run_dry_plan(SEEDS="42", ONLY="A46b s1")
    # A46b must have batch_size=2 and +ablation=A46b
    a46b_lines = [line for line in plan.splitlines() if "model=A46b" in line or ("A46b" in line and "uv run" in line)]
    a46b_cmd = [line for line in a46b_lines if line.strip().startswith("uv run")][0]
    assert "data.batch_size=2" in a46b_cmd
    assert "+ablation=A46b" in a46b_cmd
    assert "data=goespvdaq" in a46b_cmd
    assert "train=false" in a46b_cmd
    assert "test=true" in a46b_cmd
    assert "data.test_split=all" in a46b_cmd

    # s1 does not have compute_marginal_gain
    s1_cmd = [line for line in plan.splitlines() if line.strip().startswith("uv run") and "stage=s1" in line][0]
    assert "compute_marginal_gain" not in s1_cmd
    assert "data.batch_size=8" in s1_cmd


def test_eval_us_generalization_visual_scale_and_evs_keep():
    plan = _run_dry_plan(SEEDS="42", ONLY="s2a s2d s2e A46b")
    s2a_cmd = [line for line in plan.splitlines() if line.strip().startswith("uv run") and "stage=s2a" in line][0]
    assert "model.vision_cfg.visual_scale=0.5" in s2a_cmd

    s2d_cmd = [line for line in plan.splitlines() if line.strip().startswith("uv run") and "model=vision_chronos2_s2d" in line][0]
    assert "model.vision_cfg.visual_scale=0.5" in s2d_cmd
    assert "model.vision_cfg.visual_evs_keep=14" in s2d_cmd

    s2e_cmd = [line for line in plan.splitlines() if line.strip().startswith("uv run") and "tag=mmtsfm_s2e_goespvdaq_s2e_all_s42" in line][0]
    assert "model.vision_cfg.visual_scale=0.5" in s2e_cmd
    assert "model.vision_cfg.visual_evs_keep=20" in s2e_cmd

    a46b_cmd = [line for line in plan.splitlines() if line.strip().startswith("uv run") and "tag=mmtsfm_A46b_s2e_goespvdaq_all_s42" in line][0]
    assert "model.vision_cfg.visual_scale=0.5" in a46b_cmd
    assert "model.vision_cfg.visual_evs_keep=70" in a46b_cmd


def test_eval_us_generalization_multi_seed():
    plan = _run_dry_plan(SEEDS="42 43", ONLY="s1")
    tags = re.findall(r"model\.results_tag=(\S+)", plan)
    assert tags == ["mmtsfm_s1_goespvdaq_selfattn_all_s42", "mmtsfm_s1_goespvdaq_selfattn_all_s43"]


def test_eval_us_generalization_sp_reference(tmp_path):
    sp_file = tmp_path / "smart_persistence_s2_goespvdaq.json"
    plan_missing = _run_dry_plan(SP_REF=str(sp_file), ONLY="s1")
    assert "[SP] Generating US Smart Persistence baseline at 30-min cadence" in plan_missing

    sp_file.write_text("{}")
    plan_exists = _run_dry_plan(SP_REF=str(sp_file), ONLY="s1")
    assert f"model.sp_reference_path={sp_file}" in plan_exists
