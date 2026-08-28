# 16 — Run s2c, 3 seeds, against s2b

Type: task
Status: blocked
Blocked by: 14, 15
Blocks: 17, 18

## Question

Run the arm and collect the comparison.

## Protocol

Same s1 checkpoints as s2b (`uk_pv_s1_selfattn_s{42,43,44}`, borrowed via `INIT_CKPT`),
same data split, same aligned protocol (165,295 scored steps, 14 disjoint plants), 3 seeds,
same ramp and aggregate metrics, `MARGINAL_GAIN=1` so the forced vision-off pass is scored.

Control is the existing wave-1 `mmtsfm_s2b_ukpv_selfattn_s{42,43,44}`. Note s42 and s43
were killed mid-run and are currently unresumed — the control is n=1 (s44, ramp 0.1487)
until they are re-run. Resume them before or alongside this.

~10 GPU-h per seed. Compute is not the binding constraint.

## Done when

- [ ] 3 seeds completed with results JSONs, no walltime kills
- [ ] ramp NMAE, ramp NRMSE, NMAE, SS, and the vision marginal recorded per seed
- [ ] attention divergence and tau distances (ticket 15) present in each JSON
- [ ] per-seed values reported individually, plus mean +/- sd — not only the mean
