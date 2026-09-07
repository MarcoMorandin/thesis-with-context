# 23 — A30-e: pooling probe re-run at GRID=7

Type: task
Status: resolved
Blocked by: 20

## Question

Design §5.4: the probe that justified spatially-resolved fusion was only ever run at
GRID=4; s2d ships 7x7. Does re-running it at the native resolution show headroom beyond
4x4, or was 4x4 already the ceiling?

## Answer

Run 2026-09-08 (`logs/slurm/56401768_latent_pooling.out`), after fixing a real bug the
first GRID=7 attempt had (see `MMTSFM/scripts/probes/latent_pooling_bottleneck.py` commit
`9a408e6` — the report table was chaining fixed-size pooling through an already-reduced
7x7 store, silently cropping the "2x2"/"4x4" rows and never printing a true 7x7 row at
all). Fixed version pools each arm independently off the native 14x14 grid.

**RAMP R², all 3 horizons, non-monotonic — 4x4 wins, 7x7 loses to it every time:**

| horizon | 1x1 | 2x2 | 4x4 | 7x7 (s2d native) |
|---|--:|--:|--:|--:|
| t+30  | 0.0215 | −0.0313 | **0.0512** | −0.0020 |
| t+60  | 0.0259 | −0.0056 | **0.0815** | 0.0228 |
| t+120 | 0.0532 |  0.0253 | 0.0516 | 0.0278 |

4x4's t+30 number (0.0512) matches the design doc's cited figure exactly — reproducible.
1x1 moved (0.0215 here vs 0.0060 cited) because the fix pools every arm off the *full*
native 14x14 field instead of nesting inside a GRID-sized store; the old 1x1 was actually
a mean over a 12x12 crop. New basis is the more correct one and is what future citations
should use — the design doc's "0.0060 at 1x1" is now stale, not comparable.

**Reading, with the caveat that matters:** this probe mean-pools to each grid size.
`VisualPatchProjector`'s pixel-shuffle is explicitly *not* mean-pooling — space-to-depth
concatenation (14x14x1024 -> 7x7x4096), built specifically to avoid averaging. So this
result doesn't falsify s2d's architecture directly. It does kill the simple "resolution
match" framing A30-e was hoping to supply: **mean-pooled 7x7 carries no more ramp signal
than mean-pooled 1x1, and clearly less than mean-pooled 4x4.** If s2d's ramp gain is real
(ticket 20; A30-b in ticket 22 says it clearly is, on the stale-sky axis), this probe says
the explanation is more likely EVS motion-selectivity (A30-d, still not run) or the
richer per-cell channel content from concatenation, not raw spatial fineness. Raises the
priority of A30-d.
