# 44 — Confirm the vision-off pass keeps s2d's sequence length

Type: task
Status: resolved

## Question

Every Δ (marginal gain) number for s2d rests on `force_vision_off=True`. On the interleaved
path it sets `vis_on = zeros` (`vision_chronos2.py:1035`) while `use_video` — which gates
whether the 98 visual tokens are assembled at all (`:1046`) — is computed from the presence of
`video_latents`, not from `vis_on`. Read: tokens are still built and zeroed, so the off pass
sees the same 141-token sequence the model trained on. If instead the tokens are dropped, the
off pass runs a 43-token sequence the weights never saw, and part of Δ is the same train/test
length mismatch that made A30-d uninterpretable.

Verify by reading `interleave_sequences` and the `vis_on` consumers on the `interleaved_raw`
branch, then confirm empirically: one seed, eval-only, log the assembled sequence length in
the on and off passes (a print in `_forward` is enough; no result JSON needed).

Resolved when this ticket states "length preserved: yes/no" with the line numbers. If **no**,
add a same-length control (visual tokens replaced by zeros at their trained positions) and
re-derive every Δ in `ablations.md` §1 from it before any of tickets 28, 32, 40–43 quote a Δ.

Context: `paper-readiness-audit.md` §6 caveat.

## Resolution (2026-09-07)

**Length preserved: yes.**

- `vision_chronos2.py:937-941` — `use_video` is derived from `video_latents is not None`
  (and the encoder being present), never from `force_vision_off`. So the branch that
  assembles the visual tokens is entered identically in both passes.
- `vision_chronos2.py:1208` — `force_vision_off` reaches only `_modality_dropout`.
- `vision_chronos2.py:635-644` — that path multiplies `visual_embeds` by 0.0 and returns
  early. The tensor keeps its shape; the tokens keep their slots.
- The zeroed tokens still go through `interleave_sequences`, and `refine_mask` still marks
  them attendable, so the off pass runs the same 141-token sequence at the same positions.

Empirically asserted, not just read: `tests/test_s2d_component_ablations.py`
::`TestVisionOffPreservesLength` hooks `chronos.encoder`'s pre-forward and compares
`inputs_embeds.shape[1]` and the fractional-position count across the on/off passes, at
`evs_keep=8` and `evs_keep=0` (EVS disabled), plus the same check on `late_raw`.

Consequence: the A30-d length confound does **not** contaminate marginal gain. Tickets 28,
32, 40-43 may quote Δ as recorded. No same-length control arm is needed.
