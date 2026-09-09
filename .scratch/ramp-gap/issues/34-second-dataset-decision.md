# 34 — One dataset or two? (goes_pvdaq for the paper)

Type: grilling
Status: open

## Question

The map ruled `goes_pvdaq` cross-dataset validation out of the *thesis* (2026-08-25): no LOPO
harness for MMTSFM, no V-JEPA latent cache for it, a 1 TB scratch quota that has overflowed
once. For a top-tier submission claiming cross-plant generalization, "one dataset, 14 test
plants" is the most common reject reason. This is the biggest scope lever on the map and
only the user can pull it.

What exists: `configs/data/goespvdaq.yaml`; `run_eval.py --lopo-dataset goes_pvdaq` and the
`lopo` preset in `run_suite.py` for the baselines (`baselines.md` §4.1); the dataset itself
is downloaded (15-min cadence, 10 plants, RGB 256×256). What does not: MMTSFM LOPO training
loop (10 folds × 3 seeds × s1→s2d), V-JEPA cache extraction for goes_pvdaq frames, the P0
baselines under LOPO.

Decide: (a) build it — and if so, which subset (s1, s2a, s2d, ticket-28 arm; P0 baselines
only) and whether 10 folds × 3 seeds or fewer; or (b) single-testbed mechanism paper with the
limitation stated in the abstract and the protocol's scale-free cross-dataset aggregation
(`baselines.md` §4.4) left for follow-up.

Decide **before** tickets 28 and 32 launch — the answer changes what they are run on.

Context: `paper-readiness-audit.md` §2.4 D1; map Out of scope entry for goes_pvdaq.
