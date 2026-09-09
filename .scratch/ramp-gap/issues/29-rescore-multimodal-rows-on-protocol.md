# 29 — Re-score the tier 4–6 rows on the protocol (14 plants, ramp column)

Type: task
Status: open

## Question

Seven result JSONs have no protocol-aligned ramp column, and four of them are aggregated
over the wrong plants:

| Row | n_plants | ramp | issue |
|---|---|---|---|
| `crossvivit_s2_ukpv_mm` | 15 | — | goes_pvdaq **train** site 1202 included |
| `sunset_s2_ukpv_mm` | 15 | — | same |
| `unicast_s2_ukpv_mm` | 15 | — | same |
| `ts_rag_orig_s2_ukpv` | 19 | — | 3 val + 2 train plants included |
| `aurora_s2_ukpv`, `cross_rag_orig_s2_ukpv`, `visionts_pp_s2_ukpv` | 14 | — | no ramp block |
| Time-VLM | — | — | **rank 2** in the current leaderboard; SS carried from vendor eval, never in common format; 14 `time_vlm_*_pred.npz` exist |

Ramp NMAE is P0 and the paper's claim is contested exactly on these rows (they are the
multimodal competitors). Re-score every one of them with
`baselines/scripts/import_predictions.py --ukpv_dir <export> --data <parquet>` so the daylight
mask and ramp thresholds are the protocol's, restricted to the 14 committed test plants, then
`aggregate_all.py`. Eval-only, no GPU.

Resolved when every tier 4–6 JSON reports `n_plants: 14`, `n_steps: 165295`, a ramp block, and
`manifest.config.daylight_mask` is not `proxy true>0`; record the new SS / ramp NMAE for each
row here, with the old value beside it.

Context: `paper-readiness-audit.md` §2.2; memories `goes-pvdaq-1202-leak-mm-baselines`,
`ts-rag-stale-split-leakage`; `protocol.md` §5 item 3 warning.
