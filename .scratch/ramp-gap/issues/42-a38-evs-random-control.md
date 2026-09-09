# 42 — Is EVS's novelty criterion doing anything? Random-98 control (A38)

Type: task
Status: open

## Question

EVS keeps the 98 of 196 tokens with the highest cosine dissimilarity to the same cell one
frame earlier (`patch_projector.py::evs_select`), frame 0 pinned. The paper will present it
as motion-selective token pruning. The only evidence is the eval-only q-sweep (A30-d), which
is confounded by train/test sequence length. The missing control is the cheapest one: keep a
**random** 98 (seeded per sample, frame 0 pinned identically) so token count, positions and
length are unchanged and only the *selection rule* differs.

Build one switch: `visual_evs_mode: "novelty" | "random"`.

**Launch spec** (register as A38):

| Arm | Mode | Config | Seeds | Base |
|---|---|---|---|---|
| A38-eval | eval-only | `+ablation=A38 model.vision_cfg.visual_evs_mode=random` on the s2d checkpoints | 42, 43, 44 | `uk_pv_s2d_s2d_s{seed}/best.ckpt` |
| A38-train | train | `model=vision_chronos2_s2d +stage=s2d model.vision_cfg.visual_evs_mode=random` | 42, 43, 44 | `uk_pv_s1_selfattn_s{seed}/best.ckpt` |

Run A38-eval first (minutes). Run A38-train only if A38-eval moves ramp NMAE by more than the
floor — an eval-only null already answers the question in the direction that costs nothing.

Resolved when the paired deltas vs s2d are recorded here for whichever arms ran.

Reads: random ≈ novelty → EVS is a compute-saving device, not a mechanism; the paper says so
and drops "motion-selective". random worse than novelty by > floor → the selection rule
carries signal; report it beside A30-d.

Context: `paper-readiness-audit.md` §6 item 4. Retrained q-sweep stays in ticket 32 (A31).
