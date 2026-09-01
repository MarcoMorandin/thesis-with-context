"""Safety net for the packed ablation sweep.

``scripts/ablation_sweep.sh`` expands a manifest into one run per (ablation,
seed) and packs several of them onto one whole GPU node, where
``scripts/ablation_pack.sbatch`` runs them concurrently. Two runs sharing a tag
share a results JSON and a checkpoint directory; a run whose ``+ablation=`` was
lost is a plain re-run of the arm it was supposed to ablate. Neither raises an
error at runtime — both produce a well-formed, meaningless number — so they are
caught here instead.

Both scripts print their plan under ``DRY_RUN=1`` rather than calling ``sbatch``
or ``uv run``, which is what makes the guarantee assertable without a cluster.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SWEEP = SCRIPTS / "ablation_sweep.sh"
PACK = SCRIPTS / "ablation_pack.sbatch"
REPO = SCRIPTS.parent

# 6 runs: an eval pair, a 2-seed train row, and two single-seed train rows.
MANIFEST = """\
# id | mode | stage | model_cfg | seeds | base
A09 | eval  | s2c | vision_chronos2_s2c | 42,43 | uk_pv_s2c_s2c_s{seed}
A17 | train | s2c | vision_chronos2_s2c | 42,43 | uk_pv_s1_selfattn_s{seed}
A22 | train | s2c | vision_chronos2_s2c | 42    | uk_pv_s1_selfattn_s{seed}
A23 | train | s2a | vision_chronos2     | 42    | uk_pv_s1_selfattn_s{seed}
"""

BASES = [
    "uk_pv_s2c_s2c_s42",
    "uk_pv_s2c_s2c_s43",
    "uk_pv_s1_selfattn_s42",
    "uk_pv_s1_selfattn_s43",
]


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    """A sweep-shaped environment with every warm-start checkpoint present."""
    ckpt = tmp_path / "ckpt"
    for base in BASES:
        (ckpt / base).mkdir(parents=True, exist_ok=True)
        (ckpt / base / "best.ckpt").write_bytes(b"")
    (tmp_path / "results").mkdir(exist_ok=True)
    manifest = tmp_path / "sweep.manifest"
    manifest.write_text(MANIFEST)

    e = dict(os.environ)
    e.update(
        {
            "DRY_RUN": "1",
            "MANIFEST": str(manifest),
            "DATA_DIR": str(tmp_path / "data"),
            "CKPT_DIR": str(ckpt),
            "RESULTS_DIR": str(tmp_path / "results"),
            "SWEEP_DIR": str(tmp_path / "sweep"),
            "VJEPA_CACHE_ROOT": str(tmp_path / "cache"),
            "MAIL_USER": "",
        }
    )
    return e


def _run(
    script: Path, env: dict[str, str], **overrides: str
) -> subprocess.CompletedProcess:
    e = dict(env)
    e.update(overrides)
    return subprocess.run(
        ["bash", str(script)], env=e, capture_output=True, text=True, cwd=REPO
    )


def _plan(env: dict[str, str], **overrides: str) -> str:
    proc = _run(SWEEP, env, **overrides)
    assert proc.returncode == 0, f"sweep failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def _field(text: str, key: str) -> list[str]:
    return re.findall(rf"^\s*{key}=(\S+)\s*$", text, flags=re.MULTILINE)


def _packs(env: dict[str, str], **overrides: str) -> list[list[str]]:
    """Tags per pack, in submission order."""
    plan = _plan(env, **overrides)
    out: list[list[str]] = []
    for block in plan.split("--- would submit ---")[1:]:
        out.append(_field(block, "TAG"))
    return out


def _job_file(env: dict[str, str], **overrides: str) -> str:
    """Run the sweep with a single pack and return the job file it wrote."""
    _plan(env, NPACKS="1", **overrides)
    return str(Path(env["SWEEP_DIR"]) / "pack0.jobs")


def _dispatch(
    env: dict[str, str], job_file: str, **overrides: str
) -> dict[str, dict[str, str]]:
    """Run the pack worker over ``job_file`` and parse one record per run.

    GPUS=1 keeps the slots serial so the per-run blocks cannot interleave on
    stdout; the concurrent path is exercised separately.
    """
    e = dict(env)
    e.update({"JOB_FILE": job_file, "GPUS": "1", "POLL": "0.05"})
    e.update(overrides)
    proc = subprocess.run(
        ["bash", str(PACK)], env=e, capture_output=True, text=True, cwd=REPO
    )
    assert proc.returncode == 0, f"pack failed:\n{proc.stdout}\n{proc.stderr}"
    keys = ("STAGE", "STAGE_DIR", "GPU", "MODE", "ABLATION", "CMD")
    records: dict[str, dict[str, str]] = {}
    cur: dict[str, str] | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("TAG="):
            cur = {}
            records[line[len("TAG=") :]] = cur
            continue
        if cur is None:
            continue
        for k in keys:
            if line.startswith(f"{k}="):
                cur[k] = line[len(k) + 1 :]
                break
    records["__stdout__"] = {"CMD": proc.stdout}
    return records


# --------------------------------------------------------------------------- #
# submitter
# --------------------------------------------------------------------------- #


def test_every_run_gets_a_unique_tag(env):
    tags = _field(_plan(env, NPACKS="1"), "TAG")
    assert len(tags) == 6, tags
    assert len(set(tags)) == len(tags), f"tag collision: {tags}"
    # The ablation ID has to be in the tag: A17 and A22 differ by nothing else.
    assert {"mmtsfm_A17_s2c_ukpv_s42", "mmtsfm_A22_s2c_ukpv_s42"} <= set(tags)


def test_duplicate_manifest_rows_are_rejected_before_anything_is_queued(env, tmp_path):
    dupe = tmp_path / "dupe.manifest"
    dupe.write_text(
        "A17 | train | s2c | vision_chronos2_s2c | 42 | uk_pv_s1_selfattn_s{seed}\n"
        "A17 | train | s2c | vision_chronos2_s2c | 42 | uk_pv_s1_selfattn_s{seed}\n"
    )
    proc = _run(SWEEP, env, MANIFEST=str(dupe))
    assert proc.returncode != 0
    assert "duplicate run tags" in proc.stdout


def test_short_and_long_runs_never_share_a_pack(env):
    # A pack drains a queue with GPUS slots; once the queue is empty a slot that
    # finished early sits allocated and idle until the pack's slowest run ends.
    # An eval (minutes) sharing a pack with a train (hours) is that GPU wasted
    # for the whole difference, which is what makes the parallelism nominal.
    packs = [p for p in _packs(env, NPACKS="3") if p]
    assert len(packs) >= 2
    for pack in packs:
        modes = {"eval" if "_A09_" in t or "_A10_" in t else "train" for t in pack}
        assert len(modes) == 1, f"pack mixes cost classes: {pack}"
    evals = [p for p in packs if all("_A09_" in t or "_A10_" in t for t in p)]
    trains = [p for p in packs if p not in evals]
    assert evals and trains
    assert sum(len(p) for p in evals) == 2
    assert sum(len(p) for p in trains) == 4


def test_round_robin_balances_packs_within_a_cost_class(env):
    # Within a class the jobs are interchangeable, so no pack should draw a
    # disproportionate share and run out of walltime while another finishes early.
    packs = [p for p in _packs(env, NPACKS="3") if p]
    trains = [p for p in packs if not all("_A09_" in t or "_A10_" in t for t in p)]
    sizes = [len(p) for p in trains]
    assert max(sizes) - min(sizes) <= 1, sizes


def test_every_job_lands_in_exactly_one_pack(env):
    packs = _packs(env, NPACKS="3")
    flat = [t for p in packs for t in p]
    assert len(flat) == 6
    assert len(set(flat)) == 6, flat


def test_empty_packs_are_not_submitted(env):
    # NPACKS is clamped to the job count, but a cost class can still end up with
    # fewer jobs than the packs allotted to it. Submitting the empty one would
    # allocate a node for a worker that exits FATAL on an empty slice.
    plan = _plan(env, NPACKS="5", ONLY="A09")
    assert "PACK=" in plan
    for pack in _packs(env, NPACKS="5", ONLY="A09"):
        assert pack, "an empty pack was submitted"


def test_missing_checkpoint_is_fatal_at_submit_time(env, tmp_path):
    # A train row from random init and an eval row on an untrained model both
    # write a results JSON nothing downstream can tell from a real one, so this
    # must fail on the login node before an allocation is spent.
    proc = _run(SWEEP, env, DRY_RUN="0", CKPT_DIR=str(tmp_path / "empty"))
    assert proc.returncode != 0
    assert "checkpoint not found" in proc.stdout
    assert "missing checkpoint" in proc.stdout


def test_only_filter_selects_a_single_ablation(env):
    tags = _field(_plan(env, NPACKS="1", ONLY="A17"), "TAG")
    assert len(tags) == 2
    assert all("_A17_" in t for t in tags)


def test_seeds_override_replaces_every_row_seed_list(env):
    tags = _field(_plan(env, NPACKS="1", SEEDS="42"), "TAG")
    assert len(tags) == 4
    assert all(t.endswith("_s42") for t in tags)


def test_chain_links_resubmissions_with_afterany(env):
    plan = _plan(env, NPACKS="1", CHAIN="3")
    deps = _field(plan, "DEPENDENCY")
    assert deps[0] == "<none>"
    # afterany, not afterok: the 24 h cap shows up as TIMEOUT (a failure), and
    # resuming from it is the entire point of the chain.
    assert sum("afterany" in d for d in deps) == 2, deps
    assert not any("afterok" in d for d in deps)


# --------------------------------------------------------------------------- #
# pack worker
# --------------------------------------------------------------------------- #


def test_every_packed_run_carries_its_ablation_and_marginal_gain(env):
    runs = _dispatch(env, _job_file(env))
    del runs["__stdout__"]
    assert len(runs) == 6
    for tag, rec in runs.items():
        abl = rec["ABLATION"]
        assert f"+ablation={abl}" in rec["CMD"], tag
        # Without this the vision-on/off decomposition is absent from the JSON
        # and the ablation cannot be read at all.
        assert "model.compute_marginal_gain=true" in rec["CMD"], tag


def test_ablation_override_is_appended_last(env):
    runs = _dispatch(env, _job_file(env))
    del runs["__stdout__"]
    for tag, rec in runs.items():
        cmd = rec["CMD"]
        assert cmd.index("+ablation=") > cmd.index("model="), tag
        assert cmd.index("+ablation=") > cmd.index("+stage="), tag


def test_each_packed_run_gets_its_own_checkpoint_directory(env):
    runs = _dispatch(env, _job_file(env))
    del runs["__stdout__"]
    dirs = [r["STAGE_DIR"] for r in runs.values()]
    assert len(set(dirs)) == len(dirs), dirs
    # A17 and A22 share stage, model config and seed — only the ablation ID
    # keeps them from warm-starting into one directory and overwriting each
    # other's best.ckpt.
    for tag, rec in runs.items():
        assert rec["STAGE_DIR"].endswith(f"_{rec['ABLATION']}"), tag


def test_eval_rows_score_the_checkpoint_and_never_warm_start(env):
    runs = _dispatch(env, _job_file(env))
    del runs["__stdout__"]
    evals = {t: r for t, r in runs.items() if r["MODE"] == "eval"}
    assert len(evals) == 2
    for tag, rec in evals.items():
        assert "ckpt_path=" in rec["CMD"], tag
        assert "best.ckpt" in rec["CMD"], tag
        assert "init_ckpt=" not in rec["CMD"], tag


def test_train_rows_warm_start_from_the_manifest_base(env):
    runs = _dispatch(env, _job_file(env))
    del runs["__stdout__"]
    trains = {t: r for t, r in runs.items() if r["MODE"] == "train"}
    assert len(trains) == 4
    for tag, rec in trains.items():
        assert "init_ckpt=" in rec["CMD"], tag
        assert "uk_pv_s1_selfattn_s" in rec["CMD"], tag


def test_skip_done_makes_resubmission_idempotent(env):
    # The continuation chain re-submits the SAME job file. Without this, every
    # link would retrain everything the previous one finished.
    job_file = _job_file(env)
    done = "mmtsfm_A17_s2c_ukpv_s42"
    (Path(env["RESULTS_DIR"]) / f"{done}.json").write_text("{}")
    runs = _dispatch(env, job_file)
    stdout = runs.pop("__stdout__")["CMD"]
    assert f"skip {done}" in stdout
    assert done not in runs
    assert len(runs) == 5


def test_pack_slice_bounds_select_a_subset(env):
    runs = _dispatch(env, _job_file(env), PACK_START="0", PACK_END="2")
    del runs["__stdout__"]
    assert len(runs) == 2


def test_concurrent_slots_do_not_share_command_state(env):
    # build_stage_cmd reads and writes globals; four slots in one shell would
    # interleave TAG/STAGE_DIR/CMD and launch four copies of whichever
    # assignment won. Each slot therefore builds in a subshell.
    e = dict(env)
    e.update({"JOB_FILE": _job_file(env), "GPUS": "4", "POLL": "0.05"})
    proc = subprocess.run(
        ["bash", str(PACK)], env=e, capture_output=True, text=True, cwd=REPO
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    tags = _field(proc.stdout, "TAG")
    assert len(tags) == 6
    assert len(set(tags)) == 6, tags
    assert len(set(_field(proc.stdout, "STAGE_DIR"))) == 6
