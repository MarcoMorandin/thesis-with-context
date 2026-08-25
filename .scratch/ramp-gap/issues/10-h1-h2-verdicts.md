# 10 — What verdicts do H1 and H2 get?

Type: grilling
Status: needs-triage
Blocked by: 06, 07, 08

## Question

`knowledge/scope.md`'s ladder and the drafted Ch9 both leave H1 and H2 **Open**, on the
stated grounds that the model's arms had not completed. They have. This ticket closes both,
on evidence, at n=3.

**H1 — "a visual stream improves over an identical time-series-only model."** The honest
verdict is split and should be written as such:

- *Falsified for late fusion.* s2a (0.5086) is indistinguishable from s1 (0.5087).
- *Supported inside interleaved fusion.* The forced vision-off pass on s2b's own weights
  gives +2.7% NMAE / +2.1% NRMSE, positive on 14/14 plants.

What is not yet settled, and what 06 and 07 supply: whether that gain survives the
shuffled-frames control, whether it lands on **ramp** at all, and whether it is
horizon-local as the ~2 h decorrelation predicts. Decide how much of H1 each of those
licenses, and what the claim reduces to if the control comes back flat.

**H2 — "selective interleaving improves over late fusion."** s2b (0.5284) over s2a (0.5086)
at n=1 each; wave 1 supplies the seed floor that makes it a verdict rather than an
observation. Note the confound from 03: s2b changed the mask handling at the same time it
changed the fusion mode, so decide explicitly whether the H2 claim rests on the seeded delta
alone or needs the mask-clean rebuild.

Output is the sentence each hypothesis gets in the rewritten conclusions, plus the table row
in the verdict table, plus what goes in Limitations for whichever part the evidence does not
reach. Standing decision 2 says the fusion mechanism is the contribution, so H2's wording
carries the most weight in the thesis and deserves the most care here.
