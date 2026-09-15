# Set the evidence gate for temporal interleaving

Type: grilling
Label: wayfinder:grilling
Status: open
Assignee: unassigned
Parent: ../map.md
Blocked by: none

## Question

What comparison will establish that temporal interleaving contributes to
cross-plant forecasting beyond simply providing more useful visual features?

## Comments

The user requires temporal interleaving as an architectural property. A positive
scientific claim about its benefit still requires an empirical control.

Proposed gate: train two arms on exactly the same selected multi-anchor visual
payload, cache, timestamps, token budget, initialization and optimizer schedule:
one interleaves after the matched TS anchors; the other appends that payload at the
context tail. Specify position IDs and attention masks explicitly. First isolate
sequence placement with timestamps held fixed; evaluate changed timestamp encoding
as a separate comparison if needed. Current late_raw rejects multiple anchors, so
this is a future implementation, not a ready-to-run override.

Use disjoint validation plants for choices, then a frozen protocol test evaluation
with paired seeds and per-plant results. Judge skill score and ramp error together,
and report calibration and same-weight vision-off results. Existing historical
test results motivate hypotheses; new variants must not be selected on test scores.
The protocol's existing decision floors are practical reference thresholds, not
confidence intervals. Do not infer neutrality from a difference inside a floor.

S2D remains the strong performance control even if the interleaving contribution
has not yet been demonstrated. Do not translate an architectural requirement into
a guaranteed performance win.
