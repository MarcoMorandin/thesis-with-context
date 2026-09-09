# 35 — Efficiency table: params, GPU-hours, latency, VRAM

Type: task
Status: open

## Question

`baselines.md` §4.6: every model reports trainable / total params, GPU-hours to train (or
"0, zero-shot"), single-window inference latency, peak VRAM. Nothing is instrumented — the
design doc says "no measured wall-clock exists on this box", result JSONs carry no timing
keys, and only `pipeline.py` counts parameters. s2d runs 141 tokens against s2b's 44 (3.3×)
and the frozen-backbone story is a selling point only if the numbers are visible.

Deliver: a timing/params hook in the MMTSFM test loop and in `runner.py` that writes
`n_params_trainable`, `n_params_total`, `latency_ms_per_window`, `peak_vram_gb` into the
result manifest; GPU-hours from SLURM accounting (`sacct`) for the trained rows; one table
covering s1, s2a, s2d, the ticket-28 arm, and every P0 baseline.

Resolved when the table exists and is linked here.

Context: `paper-readiness-audit.md` §2.3.
