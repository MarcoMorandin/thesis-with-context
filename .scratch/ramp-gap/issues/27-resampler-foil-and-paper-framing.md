# 27 — With s2b excluded, what is the paper's foil, and what is the contribution's name?

Type: grilling
Status: open

## Question

The user excluded s2b, s2b_wide and s2c from the paper. That removes the only
resampler-*interleaved* arm. The arms that remain are s1 (no vision), s2a (resampler-pooled,
late), s2d (resampler-free, interleaved), plus — if ticket 28 runs — a resampler-free *late*
arm. Two axes, and only one of them (placement: late vs interleaved) will have both cells at
matched payload.

Decide, with the user:

1. Is the contribution named **placement** ("where frozen vision-FM tokens enter a frozen
   TSFM decides whether they are read"), with the resampler removal justified by the latent
   probe (ticket 13) rather than measured? Or is it named **resampler-free interleaving**, in
   which case s2b_wide (n=3, on disk, no new compute) must be readmitted as the
   resampler-interleaved foil so the word "resampler-free" has an in-paper comparator?
2. Which single sentence is the abstract's claim, and which ablation (ticket 28 or the
   readmitted s2b_wide) is the one that sentence rests on?
3. How the A30 registry hypothesis, currently phrased against s2b, is rewritten.

Output: the paper's one-sentence claim, the arm table the paper reports, and the rewritten
A30 hypothesis line for `knowledge/ablations.md`. Ticket 38 (rewrite) is blocked on this.

Context: `paper-readiness-audit.md` §2.4 D2 and §3.
