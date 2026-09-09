# 28 — Build and run the late-fusion, resampler-free control arm (isolates H2)

Type: task
Status: open

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
