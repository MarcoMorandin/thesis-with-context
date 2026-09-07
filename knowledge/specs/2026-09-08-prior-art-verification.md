# Prior-art verification — ticket 19

Resolves `.scratch/ramp-gap/issues/19-verify-prior-art.md`. Verified against Crossref,
Semantic Scholar, OpenAlex, and Unpaywall via direct API calls (`curl`+`jq`), not LLM-summarized
search — an LLM-summarized fetch of "SolCAD-Net" hallucinated a plausible-looking match before
this switch, which is why raw JSON was required for every claim below. DOIs resolve live
(checked via `doi.org` redirect to a real ScienceDirect/publisher page) as of 2026-09-07.

Positioning notes below judge each citation against **s2d's actual mechanism**: resampler-free
interleaved fusion — V-JEPA patches spliced into the shared sequence via a pixel-shuffle
projector at fractional temporal positions, no cross-attention, no explicit motion/advection
modeling, vision contribution content-grounded via ablation controls. s2c (cross-attention
fusion) is discarded; where a citation only threatens s2c, that is called out explicitly.

## SolCAD-Net — REAL, and it is prior art (positioning required)

**Verdict: REAL.** Confirmed independently via Crossref, Semantic Scholar, and OpenAlex
(identical title/authors/DOI/volume across all three), plus a live DOI resolution to a real
ScienceDirect PII (`S0360544226020955`).

- Title: "SolCAD-Net: Solar cloud advection dynamics for very-short-term photovoltaic power
  forecasting"
- Authors: Ziao Li, Guotian Yang, Yingming Guan, Xinli Li, Xiaofei Sun, Guogang Gu, Tian-Yi Shao
- Venue: *Energy* (Elsevier, ISSN 0360-5442), vol. 361, article 141988
- DOI: `10.1016/j.energy.2026.141988` — https://doi.org/10.1016/j.energy.2026.141988
- DOI registered 2026-07-22, Crossref-indexed 2026-08-20, print cover date 2026-10 — this is
  routine Elsevier in-press volume assignment, not a forward-dated fabrication. The "vol. 361
  (2026)" detail that made this look suspicious is correct and checks out.
- Closed access — abstract not retrievable via Crossref/S2/OpenAlex/Unpaywall (all confirm
  `abstract: null` / `is_oa: false`). The specific claim that the mechanism is
  **cross-attention** could not be independently verified from metadata alone; the title
  ("cloud advection dynamics") strongly corroborates explicit advection/motion modeling. Get
  full-text (institutional access) before writing the related-work paragraph, to confirm the
  cross-attention detail precisely.

**Positioning.** This is close prior art for **s2c's old framing** (cross-attention +
advection-guided fusion, ramp-focused evaluation) — if s2c had shipped, this paper would have
pre-empted it directly. Against **s2d's actual mechanism it does not pre-empt**: s2d uses no
cross-attention and no explicit advection/motion-vector modeling — motion information, if used
at all, is learned implicitly through interleaved content tokens at fractional temporal
positions, validated by ablation rather than architecturally hard-coded. The distinction to make
in the thesis: SolCAD-Net argues motion must be explicitly modeled (advection dynamics as an
architectural prior); s2d's ablations argue the opposite — that a resampler-free, content-grounded
interleaving scheme captures the useful ramp-relevant signal without hand-coding motion. Cite it
as the closest published comparator on the same top-10%-ramp-style evaluation protocol, and use
the s1/s2d ablation results as the counter-evidence that explicit advection is not necessary for
the gains observed. This is the one citation in this batch that changes the introduction/related
work — do not skip it.

## OCF PVNet — REAL

**Verdict: REAL.** Active OCF project, live GitHub org (`openclimatefix/PVNet`, 59 stars,
updated 2026-09-04), peer-reviewed workshop paper.

- "Forecasting regional PV power in Great Britain with a multi-modal late fusion network"
- Authors: James Fulton, Jacob Bieker, Peter Dudfield, Solomon Cotton, Zakari Watts, Jack Kelly
- Venue: Tackling Climate Change with Machine Learning workshop, ICLR 2024 —
  https://www.climatechange.ai/papers/iclr2024/46
- Mechanism confirmed from repo README: NWP and satellite streams each encoded separately, then
  **concatenated** (late fusion) with generation history, solar geometry, and a location
  embedding, followed by an output head.

**Positioning.** Late/concat fusion of separately-encoded modalities — architecturally distant
from s2d's shared-sequence interleaving (no token-level mixing, no fractional temporal
positions). Doesn't pre-empt s2d. Useful as the standard "how the field currently fuses satellite
into PV forecasting" reference and as a baseline-fusion-strategy contrast in a related-work
paragraph (late fusion vs. interleaved fusion).

## OCF Cloudcasting — REAL (engineering project; no dedicated paper found)

**Verdict: REAL**, but as production code/model, not a citable paper. `openclimatefix/cloudcasting-app`
exists on GitHub; README describes it as running "the OCF-ATI cloudcasting model live in
production" — takes prior EUMETSAT satellite frames and forecasts future frames (video
prediction of cloud fields, not PV power itself). No standalone arXiv/journal paper found under
"cloudcasting" (searched arXiv + Crossref); it appears to be documented only as OCF engineering
work, feeding models like PVNet downstream.

**Positioning.** Different task entirely (satellite-frame video prediction vs. PV power
regression) — not architectural prior art for s2d. If cited, cite as evidence of a separate
"predict-then-fuse" pipeline design (forecast the sky, then forecast power from the forecast sky)
as the alternative to s2d's fuse-then-forecast approach.

## pySTEPS — REAL

**Verdict: REAL.** Well-established, widely cited.

- "Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0)"
- Authors: Seppo Pulkkinen, Daniele Nerini, Andrés A. Pérez Hortal, Carlos Velasco-Forero, Alan
  Seed, Urs Germann, Loris Foresti
- Venue: *Geoscientific Model Development*, 12, 4185–4219 (2019)
- DOI: `10.5194/gmd-12-4185-2019`

**Positioning.** Classical (non-DL) optical-flow + Lagrangian advection + stochastic
perturbation nowcasting, built for radar precipitation, commonly repurposed for solar/cloud
fields. This is baseline-class prior art (a classical extrapolation method to benchmark ramp
performance against), not a competing learned-fusion architecture. Does not pre-empt s2d;
appropriate as a non-DL baseline in the results table.

## SolarSTEPS — NOT FOUND (closest real analog identified)

**Verdict: NOT FOUND** under this exact name. Zero exact matches in Crossref or OpenAlex
(`"SolarSTEPS"` and `"Solar STEPS"` both return zero hits). Likely a garbled/loosely-remembered
reference by the reviewing LLM.

Closest real work in the same lineage (STEPS-style probabilistic advection nowcasting, applied to
solar radiation, written partly by pySTEPS's own authors):

- "Intraday probabilistic forecasts of surface solar radiation with cloud scale-dependent
  autoregressive advection"
- Authors: A. Carpentieri, D. Folini, D. Nerini, S. Pulkkinen, M. Wild, A. Meyer (Nerini and
  Pulkkinen are pySTEPS core developers)
- Venue: *Applied Energy*, 2023
- DOI: `10.1016/j.apenergy.2023.121775`

**Positioning.** If the thesis needs a "STEPS adapted to solar" citation, use this real paper,
not "SolarSTEPS." Same baseline-class positioning as pySTEPS above — classical advection
extrapolation, not a fusion architecture, does not pre-empt s2d.

## Prithvi — REAL (frozen-vs-fine-tuned numbers not independently verified — check before citing)

**Verdict: REAL.** NASA–IBM open geospatial foundation model family, confirmed via arXiv +
Crossref (multiple derivative papers citing/building on it).

- Base model of interest: "Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth
  Observation Applications" — arXiv:2412.02732
- ViT/MAE-style pretraining on 4.2M HLS (Harmonized Landsat/Sentinel-2) time series samples;
  benchmarked on GEO-Bench; reports ~8% average improvement over Prithvi-EO-1.0 across tasks.

**Positioning.** Prithvi is a general EO vision backbone, not a PV-forecasting architecture — it
competes as an alternative vision encoder choice, not as a fusion mechanism. GEO-Bench-style
evaluation conventionally reports both frozen (linear-probe) and fine-tuned encoder numbers, so
a frozen-vs-fine-tuned comparison existing is plausible, but the abstract does not state specific
figures and the full text was not accessible here (no paywall bypass attempted). **Do not cite
specific frozen-vs-fine-tuned numbers from memory** — pull them from the arXiv PDF directly
before writing that comparison. Relevant to the thesis only as a candidate alternative vision
backbone (vs. V-JEPA), not as competing prior art on fusion mechanism.

## PV-VLM — REAL, but different from the described mechanism

**Verdict: REAL**, confirmed via arXiv.

- "PV-VLM: A Multimodal Vision-Language Approach Incorporating Sky Images for Intra-Hour
  Photovoltaic Power Forecasting"
- Authors: Huapeng Lin, Miao Yu
- arXiv:2504.13624 (2025-04-18)
- Three modules: Time-Aware (PatchTST-style transformer on PV time series), Prompt-Aware
  (LLM-encoded textual prompts from historical stats/dataset descriptors), Vision-Aware
  (pretrained vision-language model extracts high-level semantic features from **ground-based
  sky images**, not satellite). Evaluated on a 30 kW Stanford rooftop array (SUNSET-style) with a
  transfer study to University of Wollongong, Australia. Reports ~5% RMSE / ~6% MAE improvement
  (transfer study: ~7%/9.5%).

**Positioning.** Different sensing modality (ground sky-cam vs. this thesis's satellite
imagery), different fusion mechanism (VLM-derived semantic feature vectors concatenated in, not
raw patches interleaved as tokens), different horizon regime (intra-hour vs. this thesis's ramp
task). Does not pre-empt s2d. Relevant as related work under "vision-language approaches to PV
forecasting" — worth one sentence distinguishing semantic/caption-level fusion (PV-VLM) from
raw-patch/token-level fusion (s2d).

## KNMI optical-flow-vs-DL benchmark — REAL, and useful supporting evidence for s2d's argument

**Verdict: REAL.** Confirmed via Crossref; last author is KNMI's satellite/cloud-physics
researcher Jan Fokke Meirink.

- "Deep Learning for Solar Irradiance Nowcasting: A Comparison of a Recurrent Neural Network and
  Two Traditional Methods"
- Authors: Dennis Knol, Fons de Leeuw, Jan Fokke Meirink, Valeria V. Krzhizhanovskaya
- Venue: *Computational Science – ICCS 2021* (Springer LNCS)
- DOI: `10.1007/978-3-030-77977-1_24`
- Compares an RNN against two traditional (persistence/optical-flow-advection style) methods for
  solar irradiance nowcasting.

**Positioning.** Does not pre-empt s2d — it's a methods-comparison paper, not a fusion
architecture. Useful supporting citation for exactly the argument s2d's ablations make: that a
learned model can match or beat hand-crafted optical-flow/advection extrapolation without
explicitly encoding motion — i.e., independent precedent (KNMI, precipitation/irradiance domain)
for the claim that implicit, content-grounded learning can substitute for explicit motion
modeling, which is the same claim s2d's ablation controls are built to support.

## Summary table

| Citation | Verdict | Pre-empts s2d? |
|---|---|---|
| SolCAD-Net | REAL — *Energy* 361 (2026), DOI `10.1016/j.energy.2026.141988` | Pre-empts old s2c cross-attention framing directly; does not pre-empt s2d's actual mechanism, but must be cited and positioned against |
| OCF PVNet | REAL — ICLR 2024 workshop paper | No — late/concat fusion, different mechanism |
| OCF Cloudcasting | REAL — production project, no dedicated paper found | No — different task (frame prediction, not power forecasting) |
| pySTEPS | REAL — Pulkkinen et al. 2019, GMD | No — classical baseline, not a fusion architecture |
| SolarSTEPS | NOT FOUND — closest real analog: Carpentieri et al. 2023, Applied Energy | N/A — cite the real paper instead if needed |
| Prithvi | REAL — Prithvi-EO-2.0, arXiv:2412.02732 | No — alternative vision backbone, not a fusion mechanism; verify frozen/fine-tuned numbers from source before citing |
| PV-VLM | REAL but different — arXiv:2504.13624 | No — ground sky-cam + VLM captions, not satellite + interleaved patches |
| KNMI optical-flow-vs-DL | REAL — Knol et al. 2021, ICCS/LNCS | No — supports s2d's implicit-motion argument as independent precedent |
