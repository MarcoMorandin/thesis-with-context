# 38 — Rewrite `report/report.typ` around s2d

Type: task
Status: open
Blocked by: 27, 28, 29, 30, 31, 33

## Question

`report/report.typ` contains no mention of s2d. Its Method (§"The four configurations"
S1/S2a/S2b/S2c), Results (placement ladder, "open ablations" A29/A17/A22, leaderboard with
S2c at rank 1) and Conclusion ("which half of S2c produces that behaviour… the three
ablations of @tbl-open… are the immediate next step") are all s2c-framed. Related work lacks
SolCAD-Net, PVNet and Nemotron positioning (all verified in ticket 19).

Rewrite once the framing (27) and the numbers (28–31, 33) exist: arm table per ticket 27;
H1/H2 verdicts per ticket 10 updated by 28 and 33; s2d's five controls per ticket 26 (claim the
recovery, not the mechanism); calibration cost per 36; leaderboard from the re-scored rows
(29, 30) with DM/bootstrap bolding (31); iTransformer-vs-s2d ramp stated as a tie inside the
floor; single-dataset limitation per 34.

Resolved when the draft compiles and every number in it traces to a result JSON on disk.

Context: `paper-readiness-audit.md` §3.
