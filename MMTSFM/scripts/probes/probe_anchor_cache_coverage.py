#!/usr/bin/env python3
"""Does the s2d latent cache already contain A46b's historical anchors?

A46b wants N visual anchors, each a full-depth s2d block (8 frames @ 45 min ->
T_lat=4 -> 196 candidates -> EVS keeps 98). The expensive route is a fresh
~420 G / 16 h extraction. The cheap route is that the s2d cache ALREADY holds
those blocks: keys are ``{dataset}_{site}_{origin}`` and do NOT encode the
ladder (``pv_record.py:637-645``), so the entry cached at origin ``t-24h`` IS
an 8-frame burst ending at ``t-24h`` -- exactly one A46b anchor.

That only works if those historical origins were actually extracted. uk_pv is
30-min cadence and val/test run at stride=H=12 steps=6 h, so t-{24,48,72,96}h
land 4/8/12/16 strides back -- on the grid in principle. Train uses
TRAIN_STRIDE, which must divide 48 steps (24 h) or train origins miss.

This probe answers the empirical question by listing cache filenames only. No
parquet read, no h5, no GPU. Run it on a login node BEFORE committing to any
extraction. Precedent for why it matters: the 8 h ladder looked fine on paper
and replayed at 0.0 % origin coverage (knowledge/dataset.md, job 57357373).

    uv run python scripts/probes/probe_anchor_cache_coverage.py \
        --cache-dir $VJEPA_CACHE_ROOT/vit_large_f20_s224_nonhrv_sp45
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Unix-epoch magnitude -> ticks per second, so origins parse whatever unit
# `timestamps` was stored in (s / ms / us / ns) without being told.
_UNITS = ((1e17, 1_000_000_000), (1e14, 1_000_000), (1e11, 1_000), (0.0, 1))


def parse_keys(cache_dir: Path) -> dict[str, set[int]]:
    """{'uk_pv_<site>': {origin, ...}} from '<dataset>_<site>_<origin>.pt'.

    rsplit on the LAST underscore only: the dataset name itself contains one
    ("uk_pv"), and site ids may too, so the prefix is kept intact rather than
    split into its parts.
    """
    per_site: dict[str, set[int]] = defaultdict(set)
    for f in cache_dir.glob("*.pt"):
        prefix, _, origin = f.stem.rpartition("_")
        if prefix and origin.lstrip("-").isdigit():
            per_site[prefix].add(int(origin))
    return per_site


def ticks_per_second(origins: set[int]) -> int:
    probe = max(abs(o) for o in origins)
    return next(tps for threshold, tps in _UNITS if probe >= threshold)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument(
        "--splits", type=Path, default=Path("../baselines/configs/splits.json")
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--dataset", default="uk_pv")
    ap.add_argument("--anchors", type=int, default=5, help="max anchors incl. t itself")
    ap.add_argument("--stride-hours", type=float, default=24.0)
    args = ap.parse_args()

    per_site = parse_keys(args.cache_dir)
    if not per_site:
        print(f"NO CACHE ENTRIES under {args.cache_dir} -- wrong path?")
        return 1

    if args.split != "all" and args.splits.exists():
        wanted = json.loads(args.splits.read_text())[args.dataset][args.split]
        keep = {f"{args.dataset}_{s}" for s in map(str, wanted)}
        per_site = {k: v for k, v in per_site.items() if k in keep}
        print(f"filtered to {args.split}: {len(per_site)}/{len(wanted)} plants present")

    tps = ticks_per_second(set().union(*per_site.values()))
    step = int(args.stride_hours * 3600 * tps)
    total = sum(len(v) for v in per_site.values())
    print(f"{total} entries, {len(per_site)} plants, {tps} ticks/s, anchor step {step}")

    # Grid spacing: confirms the extraction stride actually used (test should
    # show 6 h = H). If the modal gap does not divide the anchor step, no
    # origin can ever have a complete ladder and the answer is structural.
    gaps: dict[int, int] = defaultdict(int)
    for o in per_site.values():
        s = sorted(o)
        for a, b in zip(s, s[1:]):
            gaps[b - a] += 1
    modal = min(sorted(gaps, key=lambda g: -gaps[g])[:3])
    print(
        f"modal origin spacing {modal / tps / 3600:.2f} h "
        f"-> anchor step is {step / modal:.2f} grid steps "
        f"({'divides' if step % modal == 0 else 'DOES NOT DIVIDE'})"
    )

    # depth[k] = origins whose k-anchor ladder is complete (k=1 is trivially all)
    depth = [0] * (args.anchors + 1)
    near = 0
    for origins in per_site.values():
        for o in origins:
            k = 1
            while k < args.anchors and (o - k * step) in origins:
                k += 1
            depth[k] += 1
            if k < args.anchors:
                missing = o - k * step
                if any(missing + d in origins for d in (-modal, modal)):
                    near += 1

    print("\nanchors | origins with a COMPLETE ladder | share")
    run = total
    for k in range(1, args.anchors + 1):
        print(f"   {k:2d}   | {run:>10,} | {run / total:6.1%}")
        run -= depth[k]
    print(
        f"\n{near:,} incomplete ladders missed by exactly one grid step "
        f"({near / total:.1%}) -- a snap-to-nearest loader would recover these, "
        "at the cost of anchors no longer sharing the origin's clock time."
    )
    print(
        "\nReuse is viable only where the share stays high at the anchor count "
        "you want. Compare against the 24 h ladder's 94.4 % and the 8 h "
        "ladder's 0.0 % (knowledge/dataset.md)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
