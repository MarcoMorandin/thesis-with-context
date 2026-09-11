# 28 — Build and run the late-fusion, resampler-free control arm (isolates H2)

Type: task
Status: resolved

## Question

H2 ("interleaving beats late fusion") is confounded: s2a is resampler-pooled, s2d is
resampler-free, so the +0.0252 SS / −0.0047 ramp margin moves fusion mode, resampler and
resolution together (ticket 10). The thesis stated this as a Limitation; the paper's
contribution *is* the fusion mechanism (standing decision 2), so the confound must be removed.

Build `fusion_mode="late_raw"`: the s2d visual payload exactly (pixel-shuffle r=2, 2-layer
MLP projector, cell embedding, EVS keep=98) appended after the series as late-fusion tokens,
with no `LatentSummarizer` / `CrossModalAdapter` and no fractional positions inside the
context. `vision_chronos2.py:1390` currently hard-wires `"late"` to `CrossModalAdapter`; this
is a new branch, plus a test beside `test_s2d_interleaved_raw.py`. Register via
`/register-experiment` **before** launch (A30 was registered retroactively — do not repeat).
Warm start from s1, seeds 42/43/44, `compute_marginal_gain=true`, stage-0 warmup identical to
s2d. Cost ≈ s2d's.

Resolved when three result JSONs exist and the paired-per-seed deltas against s2d on both P0
metrics and on Δ ramp are recorded here against the 0.0011 / 0.0037 floors.

Reads: late_raw ≈ s2d → placement does not matter, the payload does (H2 falsified, paper
reframes around the payload). late_raw ≈ s2a → placement is the mechanism (H2 supported,
cleanly). Either outcome is publishable; only the current confound is not.

Context: `paper-readiness-audit.md` §2.1 G1; ticket 10's Limitations paragraph.

## Code landed (2026-09-07) — registered as **A43**, and one design correction

`fusion_mode="late_raw"` is built (`vision_chronos2.py`): `raw_visual` now covers both raw
arms so the projector path is shared verbatim, and a new `late_raw` flag switches only the
position assignment. Config `configs/ablation/A43.yaml`, manifest row `A43`, test
`tests/test_s2d_component_ablations.py::TestLateRaw`.

**A-ID: A43.** A40/A41/A42 are already claimed by ticket 32 (linear projector / warm start
from pretrained / `visual_window_hours=3.0`).

**Correction to "appended after the series as late-fusion tokens".** Two findings:

1. A literal group-axis late fusion — s2a's actual mechanism, rows stacked on `dim=0`
   (`vision_chronos2.py:1386-1500`) — is not buildable for this payload. The 98 projector
   tokens have no time axis; putting them on a group row of length `T_full=45` means
   inventing one, which is a second uncontrolled change.
2. It would not have been a different sequence anyway. `interleave_sequences` at `n_vis=1`
   already emits `[ts_0..ts_{T_ctx-1}, v_1..v_K]` (`vision_chronos2.py:77-230`) — "appended
   after the context" is *literally the token order s2d already has*.

So the only thing that actually distinguishes late from interleaved for this payload is
**position**, and that is what `late_raw` changes: all 98 tokens share the single integer
position `T_M` (s2b's scheme) instead of spreading fractionally across `[T_M, T_M+0.99]`.
This is a strictly better control than the one specified — it completes a payload x position
2x2 (s2b pooled+integer, s2d raw+fractional, A43 raw+integer) and isolates H2 with one knob.

The test asserts the two properties that make the comparison legitimate: identical sequence
length to s2d, and **zero** fractional positions.

Status stays **open**: the arm is buildable and tested, the three runs do not exist.

## Answer — H2 is falsified: placement does nothing, the payload is the mechanism (2026-09-11)

n=3 (seeds 42/43/44), `mmtsfm_A43_s2d_ukpv_s{42,43,44}.json`, `git_sha 15673d1`. Manifest
diff against s2d is exactly one key: `vision_cfg.fusion_mode` `interleaved_raw` → **`late_raw`**
(all 98 visual tokens at the single integer position `T_M`, zero fractional positions, identical
sequence length — the two properties `tests/test_s2d_component_ablations.py::TestLateRaw`
asserts). Same dataset fingerprint, n_plants=14 on every file.

| metric | s2d (fractional) | A43 (integer) | paired Δ per seed | mean Δ | floor |
|---|---|---|---|---|---|
| skill score | 0.5510 | 0.5508 | −0.0002 / +0.0010 / −0.0013 | −0.0002 | 0.0037 |
| ramp NMAE | 0.1441 | 0.1444 | +0.0016 / −0.0009 / +0.0005 | +0.0004 | 0.0011 |
| Δ ramp (vision marginal) | 0.0064 | 0.0064 | −0.0001 / +0.0006 / −0.0003 | +0.0001 | 0.0011 |
| NMAE | 0.06971 | 0.06978 | — | +0.00007 | — |
| coverage_80 | 0.7310 | 0.7325 | — | +0.0015 | — |

**Null on every metric, sign-flipping on every one of them.** This is the ticket's first
branch, and the reference points make it unambiguous:

| arm | payload | position | SS | ramp NMAE |
|---|---|---|---|---|
| s1 | none | — | 0.5230 | 0.1506 |
| s2a | resampler-pooled soft tokens | late (group axis) | 0.5258 | 0.1487 |
| **A43** | **raw, 49 cells, EVS 98** | **late (integer `T_M`)** | **0.5508** | **0.1444** |
| s2d | raw, 49 cells, EVS 98 | interleaved (fractional) | 0.5510 | 0.1441 |

A43 sits on top of s2d and keeps the entire +0.0252 SS / −0.0047 ramp margin over s2a while
using s2a's position scheme. The confound ticket 10 flagged is now resolved in the direction
that costs the map its headline: of the two things that moved together between s2a and s2d,
**the resampler removal carries all of it and the interleaving carries none of it.**

**H2 ("interleaving beats late fusion") is falsified**, cleanly, at matched payload. It is no
longer a Limitation to state — it is a negative result to report.

**Consequences.** The contribution cannot be named "placement" or "interleaved fusion". What
is left, with A37 (sub-cell detail inert) and A39 (backbone unfreeze inert) landing the same
day, is the token set itself: 49 spatially resolved cells with cell embedding, novelty-selected
to 98 (A38 shows that selection does real work), projected raw instead of pooled through a
`LatentSummarizer`. Naming the contribution and choosing the reported arm is ticket
[27](27-resampler-foil-and-paper-framing.md)'s call, which this result forces and does not make.
Ticket [38](38-rewrite-paper-around-s2d.md) is unblocked on this dependency.
