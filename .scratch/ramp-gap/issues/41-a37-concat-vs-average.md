# 41 — Concatenate vs. average at the same 7×7 grid (A37)

Type: task
Status: resolved

## Question

`vision_chronos2_s2d.yaml` states the arm's founding claim: pixel-shuffle "so the 4 merged
patches are CONCATENATED, never averaged — averaging is the operation this arm exists to
remove." That claim has never been tested at the model level. The only evidence is the
model-free probe (A30-e), which mean-pools and cannot stand in for a trained arm.

Build one switch in `VisualPatchProjector`: `pool_mode: "shuffle" | "avg"`. `"avg"` mean-pools
each r×r block to `[B, T, 49, 1024]` and projects 1024 → d_model → d_model (same depth, same
cell embedding, same EVS keep=98, same fractional positions). Token count, positions and
sequence length are identical to s2d; only the 4096-vs-1024 payload per cell differs.

**Launch spec** (register as A37):

| Arm | Config | Seeds | Base |
|---|---|---|---|
| A37 | `model=vision_chronos2_s2d +stage=s2d model.vision_cfg.visual_pool_mode=avg compute_marginal_gain=true` | 42, 43, 44 | `uk_pv_s1_selfattn_s{seed}/best.ckpt` |

Resolved when the paired-per-seed delta vs s2d on SS, ramp NMAE and Δ ramp is recorded here
against the floors (0.0037 / 0.0011).

Reads: avg ≈ shuffle → averaging was never the problem; the claim in the config, the design
doc §1 and the manuscript's "resampler-free" motivation must be withdrawn and replaced by
"spatially resolved tokens at the right positions". avg < shuffle by > floor → the founding
claim stands with direct evidence.

Context: `paper-readiness-audit.md` §6 item 3; design §2.

## Answer — the founding claim is falsified; averaging is not worse, it is better (2026-09-11)

n=3 (seeds 42/43/44), `mmtsfm_A37_s2d_ukpv_s{42,43,44}.json`, `git_sha 15673d1`. Manifest
diff against s2d is exactly one key: `vision_cfg.visual_pool_mode = avg`. Same dataset
fingerprint, same `visual_evs_keep=98`, same `fusion_mode=interleaved_raw`, same
`n_unfreeze_encoder_blocks=3`, n_plants=14 on every file.

| metric | s2d | A37 (avg) | paired Δ per seed | mean Δ | floor |
|---|---|---|---|---|---|
| skill score | 0.5510 | **0.5564** | +0.0058 / +0.0052 / +0.0051 | **+0.0054** | 0.0037 |
| ramp NMAE | 0.1441 | 0.1435 | −0.0000 / −0.0014 / −0.0001 | −0.0005 | 0.0011 |
| Δ ramp (vision marginal) | 0.0064 | 0.0073 | +0.0009 / +0.0001 / +0.0017 | +0.0009 | 0.0011 |
| NMAE | 0.06971 | 0.06879 | — | −0.00092 | — |
| coverage_80 | 0.7310 | 0.7533 | — | +0.0223 | — |
| quantile ECE | 0.0359 | 0.0314 | — | −0.0045 | — |

**Reading.** The ticket's two branches were "avg ≈ shuffle" or "avg < shuffle". Neither
happened: avg ≈ shuffle on both ramp metrics (both inside the seed floor, and Δ ramp if
anything favours avg), and avg **beats** shuffle on skill score by +0.0054, clearing the
0.0037 floor on all three seeds individually with no sign flip. Calibration moves the same
way — coverage_80 +0.022 toward nominal, ECE −0.0045 — which is the metric pair s2d was
known to pay for its ramp gain.

So: sub-cell spatial detail carries **nothing**. The 4096-wide pixel-shuffle payload is not
why s2d works, and the r=2 concatenation costs aggregate accuracy and calibration for no ramp
return. What survives is the *token geometry*: 49 spatially distinct cells, cell embedding,
novelty-selected keep=98 — all of which A37 retains.

**Consequences (writeup, not decided here).** The founding claim quoted in
`configs/model/vision_chronos2_s2d.yaml` ("CONCATENATED, never averaged — averaging is the
operation this arm exists to remove"), design doc §1, and the manuscript's "resampler-free
detail" motivation are all **withdrawn as stated**. The honest replacement is "spatially
resolved, novelty-selected tokens at the right positions"; whether the paper ships s2d or
A37 as the reported arm is a framing call that belongs to ticket
[27](27-resampler-foil-and-paper-framing.md), not here. Note that A37 is the better arm on
three of five headline metrics and never worse than the floor on the other two.

Pairs with ticket [28](28-late-raw-control-arm.md) (A43, placement null) and
[43](43-a39-backbone-freeze.md) (A39, unfreeze null): three of s2d's four design
distinctives are now measured inert; EVS novelty selection (A38) is the one that is not.
