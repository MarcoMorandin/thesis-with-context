# 17 — Call the pre-registered s2c gate

Type: grilling
Status: blocked
Blocked by: 16

## Question

Decide what the s2c result means, against a criterion fixed *before* the data existed.

## The criterion, pre-registered 2026-08-28

**Supported** — mean ramp-NMAE improvement over s2b exceeds **0.0011**, and **all 3 seeds**
improve over s2b.

**Strong** — mean improvement >= **0.00275**, i.e. closes >= 50 % of the documented 0.0055
ramp gap to iTransformer.

**Below threshold** — recorded as a null; the thesis takes the mechanism story.

### 0.0011 is a practical effect threshold, NOT a significance threshold

It comes from `max(MMTSFM 3-seed sd, 2 x iTransformer 3-seed sd)` and from the wave-2
null (−0.0003 moved nothing). That makes it a defensible *noise scale*, and it does not
license the claim "improvement > 0.0011 therefore statistically significant". With n=3 no
such claim is available. Report per-seed values and mean +/- sd regardless of which tier
is reached, and describe the threshold as pre-registered and practical.

## The second axis, which is not optional

Do not judge this experiment on ramp NMAE alone. Ticket 15's diagnostics decide a separate
question: did the horizon queries learn horizon-specific spatial attention? The four
outcomes are different results, not degrees of one:

| ramp NMAE | attention diverges | reading |
|---|---|---|
| improves | yes | mechanism confirmed — the strong thesis result |
| improves | no | improvement is not from horizon-specific advection; find what it is |
| flat | yes | the model transports spatial information and it still does not pay — a ceiling claim |
| flat | no | degenerate parameterisation, not a falsified hypothesis; fix and re-run |

The bottom row is the one that must not be mistaken for a null.
