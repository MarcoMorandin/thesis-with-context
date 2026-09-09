# 33 — A23: recipe-matched s2a, n=3

Type: task
Status: open

## Question

The paper's H1 verdict ("late fusion is inert", A01) rests on s2a, which trains at
`visual_dropout_prob=0.3` while s2d trains at 0.5. Until A23 runs, A01's null confounds
fusion mode with recipe, and s2a is the late-fusion comparator the paper reports.

Config exists (`configs/ablation/A23.yaml`), manifest row exists (`A23 | train | s2a |
vision_chronos2 | 42,43,44 | uk_pv_s1_selfattn_s{seed}`), base checkpoints exist. Launch as
registered; cost ≈ s2a's.

Resolved when Δ ramp and Δ NMAE (vision-off) for A23 are recorded here. Δ ≈ 0 → A01's
negative result is clean and the paper keeps it. Δ clears floor → A01 as written is withdrawn
and the H1 split verdict (ticket 10) is rewritten.

Context: `paper-readiness-audit.md` §2.1 G3; `ablations.md` §2.2 A23.
