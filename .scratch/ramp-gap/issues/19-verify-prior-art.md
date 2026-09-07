# 19 — Verify the prior art the reviewers cited

Type: research
Status: resolved

## Question

Which of the citations returned by the external review are real, and does any of them
already do what s2c proposes?

## Why now

The reviews were produced by language models, not human referees, and several carry specific
citations that would matter a great deal if real. One in particular:

- **SolCAD-Net** — reported as *Energy* vol. 361 (2026), described as advection-guided
  cross-attention for ramp-aware PV nowcasting, evaluated on a top-10 % ramp subset almost
  identical to this project's. If that exists it is close prior art for s2c and must be
  cited and positioned against.

Others to check: OCF **PVNet** and **Cloudcasting**, **pySTEPS**, **SolarSTEPS**, **Prithvi**
frozen-vs-fine-tuned numbers, **PV-VLM**, and the KNMI optical-flow-vs-DL benchmark.

Forward-dated journal volumes are the first thing to check. A fabricated citation reaching
the thesis is unrecoverable.

## Done when

- [x] each citation marked real / not found / different from described
- [x] for any real one, a note on whether it pre-empts s2c and how to position against it
- [x] verified references captured somewhere `knowledge/` can point at

## Answer

Resolved via a research subagent (2026-09-08), verified against Crossref/Semantic
Scholar/OpenAlex/Unpaywall by direct API call, not LLM-summarized search (a first
LLM-summarized attempt hallucinated a plausible-looking SolCAD-Net match, which is why the
switch to raw bibliographic queries was necessary for every claim). Full writeup:
[`knowledge/specs/2026-09-08-prior-art-verification.md`](../../knowledge/specs/2026-09-08-prior-art-verification.md).

**SolCAD-Net is real** — *Energy* vol. 361 (2026), art. 141988, DOI
`10.1016/j.energy.2026.141988`, confirmed identically across three independent sources plus a
live DOI resolution. The "vol. 361 (2026)" detail that looked like a forward-dated red flag
checks out as routine Elsevier in-press assignment. It pre-empts **s2c's old cross-attention
framing** directly (advection-guided cross-attention, ramp-focused eval — nearly identical to
what s2c would have claimed). It does **not** pre-empt s2d's actual mechanism (resampler-free
interleaved fusion, no cross-attention, no explicit motion modeling) — the doc recommends
citing it as the closest published comparator and using it as an explicit foil: SolCAD-Net
argues motion must be architecturally hard-coded; s2d's ablations argue implicit
content-grounded interleaving captures the signal without that prior. This is the one citation
that changes the related-work section.

Everything else real-but-non-threatening (PVNet, Cloudcasting, pySTEPS, Prithvi, PV-VLM, KNMI
benchmark) or not found under the given name (SolarSTEPS — real analog identified: Carpentieri
et al. 2023, *Applied Energy*). Full verdicts and positioning notes per citation in the linked
doc.
