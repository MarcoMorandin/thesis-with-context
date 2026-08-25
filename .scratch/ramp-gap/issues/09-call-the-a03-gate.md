# 09 — Call the A03 gate: does the mixer get swapped?

Type: grilling
Status: needs-triage
Blocked by: 08

## Question

Apply the rule fixed in the map's Notes before the data existed:

> Swap **iff** self-attention's ramp NMAE beats Grassmann's by more than the seed floor,
> **and** its skill score does not regress by more than the seed floor.
> seed floor = max(MMTSFM 3-seed sd for that config, 2 × iTransformer's 3-seed sd —
> 0.0011 ramp NMAE, 0.0037 skill score).

Compute both 3-seed means and sds, state the floor actually used and why, then call it. The
rule was fixed in advance precisely so this session cannot negotiate with the numbers; if the
rule now looks wrong, say so explicitly and re-decide it as a *separate* act, rather than
quietly reinterpreting it.

What the outcomes mean:

- **Swap.** Grassmann underperforms the pretrained attention it replaced. The finding is
  real and reportable — a geometric O(L) operator through a magnitude-blind 60-dim bottleneck
  loses to content-based routing on sudden change — and H2 survives untouched, because
  interleaving is the contribution and it is operator-independent. Wave 2 re-baselines on the
  new operator.
- **Hold.** Grassmann is defensible on its own terms and wave 2 goes to the objective work
  and whatever G0 (05) licenses on the visual branch.
- **Split verdict.** Ramp and skill score disagree in a way the rule does not cover, or the
  deltas fall inside the floor. Then the honest answer is "no measurable difference at n=3",
  which is itself a result about the operator: it is not buying what it was introduced to buy.

Whichever way it falls, this ticket also graduates the wave-2 fog into tickets, since almost
every remaining decision hangs on it.
