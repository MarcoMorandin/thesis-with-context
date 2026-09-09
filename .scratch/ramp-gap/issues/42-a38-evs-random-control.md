# 42 — Is EVS's novelty criterion doing anything? Random-98 control (A38)

Type: task
Status: **A38-eval DONE (2026-09-09) — random is worse than novelty by > floor. A38-train now
required; ticket stays open until it lands.**

## Result — A38-eval, seeds 42 + 43 (n=2, protocol asks n=3; s44 not run)

Paired per-seed against `mmtsfm_s2d_ukpv_s2d_s{42,43}.json`, 14 test plants, 165 295 steps each:

| Metric | s2d | A38 | delta | per-seed | floor |
|---|---|---|---|---|---|
| ramp NMAE (`nmae_ramp`) | 0.1440 | 0.1480 | **+0.0039** | +0.0029 / +0.0049 | 0.0011 → **3.6×** |
| skill score | 0.5499 | 0.5347 | **−0.0152** | −0.0119 / −0.0184 | 0.0037 → **4.1×** |
| NMAE | 0.0699 | 0.0725 | +0.0026 | +0.0022 / +0.0030 | — |
| NRMSE | 0.1037 | 0.1072 | +0.0035 | +0.0028 / +0.0042 | — |
| ramp NRMSE | 0.1779 | 0.1825 | +0.0046 | +0.0033 / +0.0058 | — |
| CRPS | 0.0546 | 0.0567 | +0.0021 | +0.0017 / +0.0025 | — |
| quantile ECE | 0.0372 | 0.0431 | +0.0059 | +0.0054 / +0.0064 | — |
| 80% coverage | 0.7273 | 0.7056 | −0.0217 | −0.0186 / −0.0248 | — |

Every metric degrades, on both seeds, in the same direction. 13 of 14 plants worse on seed 42.

Forced vision-off marginal gain (ramp NMAE, off − on) localises it to the visual path:

| Arm | s42 | s43 | s44 |
|---|---|---|---|
| s2d | +0.0065 | +0.0069 | +0.0057 |
| A38 | +0.0036 | +0.0019 | — |

Random selection destroys roughly half to three quarters of what vision contributes, while the
vision-**off** number is unchanged.

**Provenance.** A38's JSONs carry `vision_cfg.visual_evs_mode=random`; s2d's have the key absent
(they predate the parameter and ran the `novelty` default). `visual_evs_keep=98`,
`visual_n_cells=49`, `visual_shuffle_r=2`, `fusion_mode=interleaved_raw`,
`n_unfreeze_encoder_blocks=3` and the dataset fingerprint are identical across all five files.
`git_sha` differs (`b5b1ea4` vs `1346540`) because the mode switch is new code, so the eval-only
contract rests on a stronger check instead: `nmae_ramp_vision_off` is identical to four decimals
per seed between the arms (0.1497 on s42, 0.1517 on s43). Vision-off bypasses `evs_select`
entirely, so an identical off-number can only come from the same weights.

**Reading, and its limit.** The selection rule is not inert at inference — the ticket's null is
refuted and "EVS is a compute-saving device, not a mechanism" is off the table. But A38 alone
cannot say the novelty rule is *better*: s2d's projector and its 3 unfrozen encoder blocks were
fit on novelty-selected tokens, so part of this degradation is test-time distribution shift.
A38-train removes that shift and is the arm that separates the two readings. Per the launch spec
above it is now mandatory, and `configs/ablation/sweep.manifest`'s conditional `A38t` row has
been uncommented.

Remaining to close: A38 seed 44 (`ONLY="A38" SEEDS="44"`), then A38t at 42/43/44.

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
