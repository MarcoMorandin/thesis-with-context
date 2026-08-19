#!/usr/bin/env python
"""Restore a stripped V-JEPA encoder into a checkpoint from a donor stage.

Why this exists
---------------
`on_save_checkpoint` used to drop the encoder whenever no encoder parameter was
trainable AT SAVE TIME, on the assumption that it could be rebuilt bit-identical
from the torch.hub cache. That assumption breaks the moment an earlier stage
fine-tunes it. In the uk_pv curriculum:

  s2a  freeze_visual_encoder=partial  -> tuned the last 4 blocks, KEPT them
  s2b  freeze_visual_encoder=true     -> inherited + trained/tested with those
                                         tuned weights (SS 0.5188), then saved
                                         with the encoder STRIPPED
  s3   warm start from s2b            -> `missing=302`, silently fell back to
                                         the pristine baseline

So s2b's checkpoint cannot reproduce its own score, and s3 trained against an
encoder its other weights were never adapted to. This script splices the donor's
encoder weights back into the target and stamps `vjepa_finetuned=True` so the
fixed `on_save_checkpoint` will never strip them again.

Usage
-----
    uv run python scripts/repair_vjepa_checkpoint.py \
        --target <CKPT_DIR>/uk_pv_s2b/best.ckpt \
        --donor  <CKPT_DIR>/uk_pv_s2a/best.ckpt \
        --out    <CKPT_DIR>/uk_pv_s2b/best_repaired.ckpt

    # inspect only, change nothing
    uv run python scripts/repair_vjepa_checkpoint.py --target <ckpt> --inspect
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

PREFIX = "model.video_encoder."


def _load(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"FATAL: no such checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _state(ckpt: dict) -> dict:
    return ckpt.get("state_dict", ckpt)


def _encoder_keys(sd: dict) -> list[str]:
    return [k for k in sd if k.startswith(PREFIX)]


def _describe(path: str, ckpt: dict) -> tuple[dict, list[str]]:
    sd = _state(ckpt)
    enc = _encoder_keys(sd)
    flag = ckpt.get("vjepa_finetuned", "<absent>")
    print(f"  {path}")
    print(f"    epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')}")
    print(f"    state_dict keys : {len(sd)}")
    print(f"    encoder keys    : {len(enc)}")
    print(f"    vjepa_finetuned : {flag}")
    return sd, enc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="checkpoint missing its encoder")
    ap.add_argument("--donor", help="checkpoint holding the fine-tuned encoder")
    ap.add_argument("--out", help="output path (default: <target>_repaired.ckpt)")
    ap.add_argument("--inspect", action="store_true", help="report only, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args()

    print("Target:")
    target = _load(args.target)
    t_sd, t_enc = _describe(args.target, target)

    if args.inspect:
        if t_enc:
            print("\nOK: this checkpoint already carries its encoder.")
        else:
            print(
                "\nSTRIPPED: no encoder weights. Safe only if NO stage ever "
                "fine-tuned the encoder; otherwise repair with --donor."
            )
        return

    if t_enc:
        sys.exit(
            f"\nFATAL: target already has {len(t_enc)} encoder keys — nothing to "
            "repair. Refusing to overwrite them."
        )
    if not args.donor:
        sys.exit("\nFATAL: --donor is required to repair (or pass --inspect).")

    print("\nDonor:")
    donor = _load(args.donor)
    d_sd, d_enc = _describe(args.donor, donor)
    if not d_enc:
        sys.exit(
            "\nFATAL: donor carries no encoder weights either. Pick the last "
            "stage whose freeze_visual_encoder was 'partial' or False — that is "
            "the one that kept them."
        )

    out = args.out or args.target.replace(".ckpt", "_repaired.ckpt")
    if os.path.abspath(out) == os.path.abspath(args.target):
        sys.exit("\nFATAL: --out must differ from --target (keep the original).")
    if os.path.exists(out) and not args.force:
        sys.exit(f"\nFATAL: {out} exists. Pass --force to overwrite.")

    # Splice. Donor tensors are copied verbatim; shapes are not reconciled,
    # so a mismatch means the two stages ran different encoder architectures.
    for k in d_enc:
        t_sd[k] = d_sd[k]
    target["vjepa_finetuned"] = True

    n_params = sum(d_sd[k].numel() for k in d_enc if torch.is_tensor(d_sd[k]))
    print(f"\nSpliced {len(d_enc)} encoder tensors ({n_params:,} parameters).")
    print("Stamped vjepa_finetuned=True so it will never be stripped again.")

    torch.save(target, out)
    print(f"Wrote {out}")

    # Verify what actually landed on disk, rather than trusting the write.
    check = _load(out)
    c_sd = _state(check)
    c_enc = _encoder_keys(c_sd)
    ok = len(c_enc) == len(d_enc) and check.get("vjepa_finetuned") is True
    mismatched = [
        k for k in d_enc
        if torch.is_tensor(d_sd[k]) and not torch.equal(c_sd[k].cpu(), d_sd[k].cpu())
    ]
    print("\nVerification:")
    print(f"  encoder keys present : {len(c_enc)}/{len(d_enc)}")
    print(f"  vjepa_finetuned      : {check.get('vjepa_finetuned')}")
    print(f"  tensors != donor     : {len(mismatched)}")
    if not ok or mismatched:
        sys.exit("FAILED: repaired checkpoint did not verify.")
    print("\nOK. Re-score it to confirm it reproduces the donor stage's number:")
    print(
        "  uv run python -m mmtsfm.train +stage=s2b model=vision_chronos2_grassmann \\\n"
        "    data=ukpv trainer=slurm trainer.devices=1 trainer.strategy=auto \\\n"
        f"    train=false ckpt_path={out} \\\n"
        "    model.results_tag=mmtsfm_s2b_ukpv_repaired"
    )


if __name__ == "__main__":
    main()
