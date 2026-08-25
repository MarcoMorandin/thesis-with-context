# 03 — Fix the interleaved mask override before wave 1, or carry it?

Type: grilling
Status: resolved

## Question

The interleaved fusion path sets `all_mask = torch.ones(B, T_ctx + n_vis + T_fut, ...)`,
discarding the `attention_mask` returned by `_prepare_patched_context`
(`MMTSFM/src/mmtsfm/models/chronos2/vision_chronos2.py`). The late and numeric paths use it
correctly.

This is not cosmetic. `build_site_series` reindexes each site onto a regular 30-min grid, so
night steps are NaN and `mask_hist` is roughly 35% ones. At `input_patch_size=16` a patch
spans 8 hours, so **fully-night patches exist** — especially in UK winter — and the
interleaved path presents them to the encoder as valid tokens. It also means the s2b-vs-s2a
comparison is not fusion-only: s2b changed the mask handling at the same time it changed the
fusion mode.

The decision is timing, and it is a genuine fork:

- **Fix before wave 1.** All five chains are clean, and the H2 claim stops being confounded.
  Cost: `grassmann@42` (the existing `mmtsfm_s2b_ukpv.json`) was trained with the bug, so it
  is no longer a comparable seed — wave 1 becomes six chains, not five, to rebuild it.
- **Fix after wave 1.** `grassmann@42` stays usable and the wave stays at five chains, but
  the mixer gate is called on a codebase with a known defect, and any reviewer who finds it
  can ask whether the A03 delta is the mixer or the mask.

Weigh the extra chain against the cleanliness of the gate. Note that the confound is
*shared* by both arms in wave 1 — both would carry it — so it threatens the H2 claim more
than the A03 verdict.

## Answer — fix before wave 1 (option a); wave 1 becomes six chains

Decided 2026-08-25. Reasoning: H2 is the headline contribution (standing decision 2), the
deadline is soft, and compute is not the binding constraint — which makes a confounded H2
the expensive option, not the cheap one. The sixth chain costs ~62 GPU-h and buys a clean
claim about the thing the thesis is actually about.

**The fix.** The interleaved path now interleaves the CONTEXT attention mask exactly as it
interleaves the tokens: `macro_mask | (ts_mask, 1) x n_vis | ones(T_fut)`. Visual tokens stay
valid (masking them on modality dropout would be a behaviour change beyond this bug).
`T_M` hoisted above its new first use — it was defined ~20 lines *after* the mask block,
so the first version raised `NameError`.

**Two things the test work exposed, both worth carrying:**

1. **An output-level test cannot detect this bug.** `_prepare_patched_context` concatenates
   `patched_mask` into the patch FEATURES, so `context_mask` changes the embedding whether or
   not the attention mask survives. The first version of the test asserted that forecasts
   differ with and without a dead patch — it passed against the *unfixed* code. Replaced with
   a spy on the encoder boundary asserting the mask that actually crosses it
   (`test_encoder_receives_the_context_mask`), which failed with all twelve positions
   arriving as `1.` and passes now.
2. **That also explains why the bug was subtle rather than catastrophic.** The embedding
   always reflected the mask; only the mixing/attention side was wrong. Unobserved patches
   were embedded correctly but admitted to temporal mixing.

**Magnitude: not measured.** The plan was to quantify what fraction of context patches are
fully unobserved at `T=672, patch=16` from the v2 parquet, but `/Volumes/dataset` was
unmounted before it ran. The fix is correct regardless of magnitude, and the confound
argument does not depend on it — but the number is still worth having for the write-up,
since it bounds how much the s2b-vs-s2a delta could have been affected. Re-run when the
drive is back or on the cluster.

**Consequences for the map:**

- `mmtsfm_s2b_ukpv.json` (`grassmann@42`) was trained with the bug and is **no longer a
  comparable seed**. Wave 1 is now **six chains**: `grassmann@{42,43,44}` +
  `selfattn@{42,43,44}`.
- The canonical bare tag `mmtsfm_s2b_ukpv` now belongs to a rebuilt `grassmann@42`. The old
  file is the historical pre-fix run — **do not compare across that boundary**. Move it aside
  before wave 1 rather than letting it be overwritten silently.
- Every number quoted from the pre-fix s2b (SS 0.5284, ramp 0.1481, the +2.7 % marginal)
  describes the pre-fix code. They stay valid as reported, but the seeded wave supersedes them.
