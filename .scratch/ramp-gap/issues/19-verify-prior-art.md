# 19 — Verify the prior art the reviewers cited

Type: research
Status: ready-for-agent

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

- [ ] each citation marked real / not found / different from described
- [ ] for any real one, a note on whether it pre-empts s2c and how to position against it
- [ ] verified references captured somewhere `knowledge/` can point at
