# 41 — Concatenate vs. average at the same 7×7 grid (A37)

Type: task
Status: open

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
