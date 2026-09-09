# 32 — s2d's own component-ablation table (trained, not eval-swept)

Type: task
Status: open

## Question

Every training-time component ablation in `knowledge/ablations.md` and
`configs/ablation/sweep.manifest` (A17–A29, A18–A21, A24–A28) is written against an s2c base
and is now out of scope. s2d has **zero** trained component ablations; its five controls are
eval-only or probes, and three of them (A30-a, A30-d, A30-e) came back uninterpretable for
exactly that reason. A reviewer's "ablation study" means this table.

Register (via `/register-experiment`, each with its own `configs/ablation/A3x.yaml` delta on
`model=vision_chronos2_s2d +stage=s2d`, s1 warm start) and run:

| ID | Variable | n | Replaces |
|---|---|---|---|
| A31 | EVS keep q ∈ {0, 0.3, 0.7} **retrained** at that token count | 3 each | A30-d (train/test length confound) |
| A32 | pixel-shuffle r=1 (14×14 native, 196 cells) vs shipped r=2 | 3 | A30-e (probe mean-pooled; architecture concatenates) |
| A33 | all four latents at one shared position vs fractional positions | 3 | makes A30-a's eval-only null a design result |
| A34 | stage-0 projector warmup off (`model.projector_warmup_steps=0`) | 1 | never ablated (Nemotron adoption) |
| A35 | cell embedding off | 1 | — |
| A40 | linear projector (one `Linear(4096, d_model)`) instead of 2-layer GELU MLP | 1 | — |
| A41 | warm start from pretrained Chronos-2 instead of the fine-tuned s1 checkpoint | 1 | curriculum dependence |
| A42 | `data.visual_window_hours=3.0` (2 latent frames) vs shipped 6.0 | 1 | A04/W5 never run on any arm |
| A24/A25/A26 rebased | token-type / modality / segment embedding off | 1 each | component table |
| A27 rebased | `numeric_dropout_prob=0` | 1 | "does numeric dropout manufacture the marginal gain" |

Split off into their own tickets because each carries a distinct claim: concat-vs-average
(ticket 41, A37), EVS random control (ticket 42, A38), backbone freeze (ticket 43, A39),
history-only covariates (ticket 40, A36). Launch order across all of them:
`paper-readiness-audit.md` §6.

Cost check before launch: r=1 is 4× the visual tokens (196/frame → with EVS q=0.5 ≈ 392
visual tokens, seq ≈ 435); set `SWEEP_BATCH` accordingly. A31 q=0 is 239 tokens.

Resolved when each row is `DONE` in `ablations.md` with its paired-per-seed delta against
s2d on both P0 metrics and Δ ramp, judged against the floors.

Blocked by: 27

Blocking note: ticket 27 may readmit s2b_wide and change which rows are worth n=3; the
configs can be written now, the launch waits.

Context: `paper-readiness-audit.md` §2.1 G2; design §5.3 (the four things s2d moves at once).
