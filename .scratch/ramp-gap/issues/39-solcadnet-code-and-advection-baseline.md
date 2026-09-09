# 39 — Is there runnable code for SolCAD-Net or an equivalent explicit-advection PV forecaster?

Type: research
Status: open

## Question

Ticket 19 confirmed SolCAD-Net (*Energy* 361, 2026, DOI `10.1016/j.energy.2026.141988`) as
the closest published prior art — explicit cloud-advection dynamics for very-short-term PV
forecasting on a ramp-style evaluation — but it is closed-access and no code was looked for.
The reviewer question the paper must pre-empt is "why not compare against the method that
hard-codes the motion prior you say is unnecessary?"

Find out: (1) whether SolCAD-Net has a public implementation or enough architectural detail in
the full text (institutional access) to reimplement in a day; (2) failing that, whether a
classical optical-flow / advection baseline (pySTEPS-style, Carpentieri et al. 2023 lineage)
producing a CSI-persistence forecast from the same 8-frame satellite window is the accepted
stand-in at the target venues, and what it would cost to run on `images_all.h5`.

Output: a recommendation — baseline it, stand-in it, or cite-and-position only — with the
cost of each. Not fired as a subagent this session (user did not ask for one); run with the
`research` skill when picked up.

Context: `knowledge/specs/2026-09-08-prior-art-verification.md`; map fog entry "Explicit
cloud-motion vectors as a model input".
