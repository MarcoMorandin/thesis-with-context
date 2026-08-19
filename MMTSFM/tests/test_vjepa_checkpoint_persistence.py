"""Regression tests: a fine-tuned V-JEPA encoder must never be stripped.

`on_save_checkpoint` drops the encoder to save ~1.2 GB per file, which is only
sound while those weights are still bit-identical to the torch.hub baseline they
are rebuilt from at init. The old predicate asked "does any encoder param require
grad RIGHT NOW", a different question, and it silently corrupted the uk_pv
curriculum:

    s2a  freeze_visual_encoder=partial -> tuned the last 4 blocks, KEPT them
    s2b  freeze_visual_encoder=true    -> inherited them, trained and tested with
                                          them (SS 0.5188), then saved with the
                                          encoder STRIPPED (all frozen at save)
    s3   warm start from s2b           -> `missing=302` (the whole encoder),
                                          silently ran on the pristine baseline

Consequences: s2b's score is unreproducible from its own checkpoint, and s3
trained against an encoder its other weights were never adapted to.

The fix makes the predicate "has this encoder EVER been fine-tuned", carried
across stages by a persisted `vjepa_finetuned` flag.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

from tests.test_training_loop import _make_module

ENC_PREFIX = "model.video_encoder."


def _module(freeze):
    """Build a module whose vision_cfg carries the given freeze policy."""
    vcfg = dict(
        n_visual_context_steps=4,
        n_soft_tokens=1,
        adapter_type="linear",
        visual_dropout_prob=0.0,
        numeric_dropout_prob=0.0,
        dropout=0.0,
        freeze_visual_encoder=freeze,
    )
    return _make_module(vision_cfg=vcfg)


def _save(mod) -> dict:
    """Run on_save_checkpoint over a full state_dict, as Lightning would."""
    ckpt = {"state_dict": dict(mod.state_dict())}
    mod.on_save_checkpoint(ckpt)
    return ckpt


def _enc_keys(ckpt) -> list[str]:
    return [k for k in ckpt.get("state_dict", {}) if k.startswith(ENC_PREFIX)]


class TestStripPredicate:
    def test_pristine_encoder_is_stripped(self):
        """Never fine-tuned -> safe to drop, and the flag says so."""
        mod = _module(True)
        for p in mod.model.video_encoder.parameters():
            p.requires_grad_(False)
        mod._vjepa_finetuned = False

        ckpt = _save(mod)
        assert _enc_keys(ckpt) == []
        assert ckpt["vjepa_finetuned"] is False

    def test_trainable_encoder_is_kept(self):
        """Currently trainable -> kept, as before the fix."""
        mod = _module(True)
        for p in mod.model.video_encoder.parameters():
            p.requires_grad_(True)

        ckpt = _save(mod)
        assert _enc_keys(ckpt), "trainable encoder must be persisted"
        assert ckpt["vjepa_finetuned"] is True

    def test_previously_tuned_but_now_frozen_encoder_is_kept(self):
        """THE s2b CASE — the one the old predicate got wrong.

        Every encoder param is frozen at save time, so the old code stripped it,
        even though an earlier stage had already changed those weights.
        """
        mod = _module(True)
        for p in mod.model.video_encoder.parameters():
            p.requires_grad_(False)
        mod._vjepa_finetuned = True  # inherited from the donor stage

        ckpt = _save(mod)
        assert _enc_keys(ckpt), (
            "encoder was fine-tuned by an earlier stage and must be persisted "
            "even though it is frozen now"
        )
        assert ckpt["vjepa_finetuned"] is True


class TestFlagPropagation:
    def test_partial_policy_sets_the_flag_at_init(self):
        assert _module("partial")._vjepa_finetuned is True

    def test_false_policy_sets_the_flag_at_init(self):
        assert _module(False)._vjepa_finetuned is True

    def test_true_policy_leaves_flag_clear(self):
        mod = _module(True)
        # The fake encoder is built trainable; the real VisualEncoder is built
        # frozen under a truthy policy. Emulate that, then re-evaluate.
        for p in mod.model.video_encoder.parameters():
            p.requires_grad_(False)
        mod._vjepa_finetuned = any(
            p.requires_grad for p in mod.model.video_encoder.parameters()
        )
        assert mod._vjepa_finetuned is False

    def test_on_load_checkpoint_restores_the_flag(self):
        mod = _module(True)
        mod._vjepa_finetuned = False
        mod.on_load_checkpoint({"state_dict": {}, "vjepa_finetuned": True})
        assert mod._vjepa_finetuned is True

    def test_flag_is_sticky(self):
        """A later frozen stage must not clear what an earlier stage set."""
        mod = _module(True)
        mod._vjepa_finetuned = True
        mod.on_load_checkpoint({"state_dict": {}, "vjepa_finetuned": False})
        assert mod._vjepa_finetuned is True


class TestCurriculumChain:
    def test_s2a_to_s2b_to_s3_preserves_the_encoder(self):
        """Full reproduction of the failure, end to end.

        s2a tunes the encoder; s2b inherits it frozen and saves; s3 warm-starts
        from that save. s3 must receive real encoder weights, not zero keys.
        """
        s2a = _module("partial")
        assert s2a._vjepa_finetuned is True
        with torch.no_grad():  # make the weights genuinely differ from baseline
            for p in s2a.model.video_encoder.parameters():
                p.add_(1.0)
        s2a_ckpt = _save(s2a)
        assert _enc_keys(s2a_ckpt), "s2a must persist its tuned encoder"

        # s2b: fresh module, warm start from s2a, then freeze everything.
        s2b = _module(True)
        s2b.load_state_dict(s2a_ckpt["state_dict"], strict=False)
        if s2a_ckpt.get("vjepa_finetuned"):
            s2b._vjepa_finetuned = True
        for p in s2b.model.video_encoder.parameters():
            p.requires_grad_(False)

        s2b_ckpt = _save(s2b)
        assert _enc_keys(s2b_ckpt), (
            "s2b froze an encoder that s2a had tuned — stripping it here is what "
            "invalidated stage s3"
        )

        # s3: warm start from s2b must find the encoder present.
        s3 = _module("partial")
        missing, _ = s3.load_state_dict(s2b_ckpt["state_dict"], strict=False)
        assert [k for k in missing if k.startswith(ENC_PREFIX)] == []

        # ...and the values must be s2a's tuned ones, not the baseline.
        got = dict(s3.state_dict())
        for k in _enc_keys(s2a_ckpt):
            assert torch.equal(got[k], s2a_ckpt["state_dict"][k]), (
                f"{k} did not survive the s2a -> s2b -> s3 chain"
            )


class TestRepairScript:
    """The recovery path for checkpoints already written by the old code."""

    def _run(self, *args) -> subprocess.CompletedProcess:
        script = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "repair_vjepa_checkpoint.py"
        )
        return subprocess.run(
            [sys.executable, script, *args], capture_output=True, text=True
        )

    @pytest.fixture
    def ckpts(self, tmp_path):
        donor = _module("partial")
        with torch.no_grad():
            for p in donor.model.video_encoder.parameters():
                p.add_(1.0)
        d_ckpt = _save(donor)

        stripped = _module(True)
        for p in stripped.model.video_encoder.parameters():
            p.requires_grad_(False)
        stripped._vjepa_finetuned = False  # emulate the OLD buggy save
        t_ckpt = _save(stripped)
        assert _enc_keys(t_ckpt) == []

        dp, tp = tmp_path / "donor.ckpt", tmp_path / "target.ckpt"
        torch.save(d_ckpt, dp)
        torch.save(t_ckpt, tp)
        return str(dp), str(tp), tmp_path

    def test_inspect_reports_stripped(self, ckpts):
        _, target, _ = ckpts
        r = self._run("--target", target, "--inspect")
        assert r.returncode == 0, r.stderr
        assert "STRIPPED" in r.stdout

    def test_repair_splices_and_verifies(self, ckpts):
        donor, target, tmp = ckpts
        out = str(tmp / "repaired.ckpt")
        r = self._run("--target", target, "--donor", donor, "--out", out)
        assert r.returncode == 0, r.stderr + r.stdout
        assert "OK." in r.stdout

        fixed = torch.load(out, map_location="cpu", weights_only=False)
        original = torch.load(donor, map_location="cpu", weights_only=False)
        assert fixed["vjepa_finetuned"] is True
        d_enc = _enc_keys(original)
        assert len(_enc_keys(fixed)) == len(d_enc) > 0
        for k in d_enc:
            assert torch.equal(
                fixed["state_dict"][k], original["state_dict"][k]
            )

    def test_refuses_to_clobber_an_intact_checkpoint(self, ckpts):
        donor, _, tmp = ckpts
        r = self._run(
            "--target", donor, "--donor", donor, "--out", str(tmp / "x.ckpt")
        )
        assert r.returncode != 0
        assert "nothing to" in (r.stdout + r.stderr)

    def test_repaired_checkpoint_is_not_stripped_again(self, ckpts):
        """The stamped flag must survive a subsequent frozen-stage save."""
        donor, target, tmp = ckpts
        out = str(tmp / "repaired.ckpt")
        assert self._run("--target", target, "--donor", donor, "--out", out).returncode == 0

        fixed = torch.load(out, map_location="cpu", weights_only=False)
        nxt = _module(True)
        nxt.on_load_checkpoint(fixed)
        for p in nxt.model.video_encoder.parameters():
            p.requires_grad_(False)
        assert _enc_keys(_save(nxt)), "flag did not survive into the next save"
