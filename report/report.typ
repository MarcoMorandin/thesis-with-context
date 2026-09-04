#set document(
  title: "Where Does Vision Belong in a Frozen Time-Series Forecaster?",
)
#set page(
  paper: "a4",
  margin: (x: 2.1cm, top: 2.3cm, bottom: 2.1cm),
  numbering: "1",
  number-align: center,
)
#set text(font: ("Libertinus Serif", "New Computer Modern", "Times New Roman"), size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.6em, below: 0.9em)[
  #set text(size: 14pt, weight: 700)
  #it
]
#show heading.where(level: 2): it => block(above: 1.25em, below: 0.6em)[
  #set text(size: 11.5pt, weight: 700)
  #it
]
#show heading.where(level: 3): it => block(above: 1.0em, below: 0.45em)[
  #set text(size: 10.5pt, weight: 700, style: "italic")
  #it
]
#set figure(gap: 0.9em)
#show figure: set block(breakable: true)
#show figure.caption: set text(size: 9pt)
#set table(stroke: (x, y) => (
  top: if y == 0 { 0.9pt } else if y == 1 { 0.6pt } else { 0pt },
  bottom: 0pt,
))
#show table.cell.where(y: 0): set text(weight: 700)

#let note(body) = block(
  width: 100%,
  inset: (x: 0.9em, y: 0.75em),
  radius: 3pt,
  fill: luma(246),
  stroke: (left: 2pt + luma(150)),
  text(size: 9.6pt, body),
)

#let tbl(caption, ..args) = figure(
  block(width: 100%, breakable: true)[
    #set par(justify: false, leading: 0.52em)
    #set text(size: 9pt)
    #table(..args)
  ],
  caption: caption,
  kind: table,
  supplement: [Table],
)

// ── primitives for the schematic diagrams (no external packages) ──
#let dbox(body, fill: luma(250), stroke: 0.6pt + luma(150)) = block(
  width: 100%,
  inset: (x: 0.55em, y: 0.5em),
  radius: 2.5pt,
  fill: fill,
  stroke: stroke,
  align(center + horizon, text(size: 8.4pt)[
    #set par(justify: false, leading: 0.5em)
    #set text(hyphenate: false)
    #body
  ]),
)
#let hit(body) = dbox(body, fill: luma(233), stroke: 1pt + luma(90))
#let ar = align(center + horizon, text(size: 11pt, fill: luma(120))[#sym.arrow.r])
#let ad = align(center, text(size: 11pt, fill: luma(120))[#sym.arrow.b])
#let dcap(body) = align(center, text(size: 8.2pt, style: "italic", fill: luma(90), body))

// a one-line architecture schematic: five stages, arrows between
#let pipe(..cells) = block(width: 100%)[
  #grid(
    columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
    column-gutter: 0.25em,
    align: horizon,
    ..cells.pos().intersperse(ar),
  )
  #v(0.35em)
]

// ─────────────────────────────────────────────────────────────── title

#align(center)[
  #block(text(size: 18pt, weight: 700)[
    Where Does Vision Belong in a \ Frozen Time-Series Forecaster?
  ])
]

#v(0.9em)

#align(center)[
  #block(width: 92%)[
    #set par(justify: true)
    #set text(size: 9.8pt)
    *Abstract.* Multimodal time-series forecasters are normally trained end to end, which makes it
    impossible to attribute an accuracy gain to the additional modality rather than to the extra
    capacity or the extra gradient path it brings. This report removes that confound by freezing
    both backbones — a pretrained time-series transformer and a self-supervised video encoder — and
    varying only the point at which the visual representation is attached. Four configurations are
    trained on an identical schedule from an identical initialisation and evaluated on a cross-plant
    photovoltaic benchmark of held-out installations, using a counterfactual pass that switches the
    imagery off at inference to measure how much of each model's accuracy actually depends on having
    seen the sky. Pooled fusion on the batch axis and appended fusion on the sequence axis both
    yield reliance indistinguishable from zero, and widening the visual channel from one token to
    sixteen does not change that. Reliance becomes measurable only when the imagery is kept unpooled
    as an external memory and queried by the forecast positions themselves, one query per lead-time
    slot. The placement, not the bandwidth, is what determines whether a frozen forecaster uses
    vision at all.
  ]
]

#v(1.0em)

= Introduction

The multimodal forecasting literature is largely a literature of proposals: an architecture is
introduced, trained end to end, and shown to beat a set of unimodal baselines. The contribution of
the additional modality is not left untested — ablation studies are near-universal, and they are
usually the right ones to run. Solar-VLM removes the visual encoder, Time-VLM reports a 9.0% MSE
degradation when its vision branch is dropped, M3S-Net reports roughly 15% higher MAE for a variant
"devoid of visual information", and PV-VLM evaluates every combination of its temporal, prompt and
vision modules @solarvlm @timevlm @m3snet @pvvlm.

The difficulty is what such a test can measure. In each of these cases the ablated variant is
*retrained* — Solar-VLM states it explicitly, that all variants are trained and evaluated under the
same protocol — so removing the module changes the modality, the parameter count and the gradient
path at the same time. The reported degradation is therefore consistent with at least three
explanations: the model was reading the images, the extra capacity helped, or the extra gradient
path acted as regularisation. The standard ablation reproduces the confound it is meant to resolve.

This work fixes the confound by construction. Both encoders are *pretrained and frozen*, so neither
can adapt to the other; only a small bridge between them is learned. The architecture is then held
fixed and the *attachment point* is varied. The question becomes:

#align(center)[
  #block(width: 88%, inset: 0.7em, text(size: 10.6pt, style: "italic")[
    Given a frozen sequence forecaster and a frozen visual encoder, at which point in the
    forecaster's computation must the visual representation be introduced before the forecaster
    demonstrably uses it?
  ])
]

Freezing is what makes the question answerable. With both backbones fixed, no configuration can win
by having more capacity than another, and any measured difference is attributable to the wiring. It
is also what makes the result portable: the finding is a statement about placement in a frozen
pipeline, not about one particular set of weights.

The contributions are as follows.

+ A *placement study* rather than an architecture proposal: three fusion sites are compared under a
  frozen-backbone protocol that holds capacity, initialisation and training schedule constant
  (@sec-configs, @tbl-ladder).
+ A *counterfactual reliance measure* that quantifies how much of a trained model's accuracy
  depends on the imagery, by re-evaluating the same frozen weights with the visual pathway switched
  off (@sec-instrument).
+ A *consolidated cross-region multimodal photovoltaic corpus* — 110 installations on two
  continents, satellite frames co-registered per site and stored with timestamp-exact pointers,
  infrared bands that remain informative at night, and a cross-plant generalisation split
  (@sec-data-novelty, @tbl-data-novelty).
+ A *full-suite comparison* against 32 baselines spanning statistical references, classical machine
  learning, supervised deep forecasters, time-series foundation models, retrieval-augmented
  adaptation and published multimodal forecasters (@tbl-leaderboard-multimodal to
  @tbl-leaderboard-baselines).

= Related work <sec-related>

Published multimodal forecasters differ in many respects, but for the purposes of this study they
differ in exactly one: *where the auxiliary representation is attached relative to the point at
which the horizon is resolved*. Five representative systems are reproduced schematically below in a
common visual language so that the attachment point can be compared directly: three satellite-based
multimodal forecasters (@fig-sunset to @fig-solarvlm), and the two strongest non-MMTSFM entries of the
baseline suite (@fig-timevlm, @fig-itransformer), which are strongest for reasons that bear directly
on the placement question. All five are run as baselines under this report's protocol
(@tbl-leaderboard-multimodal, @tbl-leaderboard-deep).

== Convolutional joint encoding — SUNSET

#figure(
  pipe(
    dbox[sky frames stacked on the *channel* axis],
    dbox[2 conv blocks, 24 #sym.arrow.r 48 filters, BN + max-pool],
    hit[flatten ⊕ *PV history vector*],
    dbox[2 #sym.times Dense(1024) + dropout],
    dbox[horizon emitted in one shot],
  ),
  caption: [SUNSET @sunset. The image branch is reduced to a flat feature vector and concatenated
  with the numeric history before the prediction head; both branches are trained jointly.],
) <fig-sunset>

SUNSET is the canonical convolutional solar baseline and the ancestor of most sky-image forecasters.
Its fusion is *concatenative*: a flattened convolutional descriptor is joined to the power history,
and a multilayer perceptron maps the concatenation to the whole horizon. The visual descriptor is
therefore computed once, independently of the horizon, and every lead time reads the same vector.

== Cross-attention from history timesteps — CrossViViT

#figure(
  pipe(
    dbox[satellite patches per timestep, axial rotary lat/lon encoding],
    dbox[ViT context encoder],
    hit[mixer: each *history* timestep queries its own frame],
    dbox[temporal transformer over the fused history],
    dbox[MLP quantile heads emit all lead times],
  ),
  caption: [CrossViViT @crossvivit. Cross-attention is present, but the queries originate in the
  *history* representation; the horizon is decoded afterwards from the already-fused sequence.],
) <fig-crossvivit>

CrossViViT is the strongest published satellite-plus-time-series architecture and the closest
structural relative of the configuration proposed here. It encodes each satellite frame with a
vision transformer using axial rotary embeddings over latitude and longitude, encodes the station
series with a separate transformer, and mixes the two with cross-attention. The decisive detail is
the direction of that cross-attention: the *query* is the station token at a given history timestep
and the keys and values are that timestep's image patches. Fusion therefore completes before the
temporal decoder runs, and the horizon is produced from a sequence in which the visual evidence has
already been condensed.

== Gated late fusion of pooled embeddings — Solar-VLM

#figure(
  pipe(
    dbox[satellite crop + text prompt (historical weather description)],
    dbox[frozen vision--language embedder],
    dbox[*one pooled vector per station*],
    hit[concatenated with the temporal state, MLP fusion, cross-station attention],
    dbox[gate blends the multimodal and numeric-only predictions],
  ),
  caption: [Solar-VLM @solarvlm. The scene is compressed to a single per-station embedding; a
  learned gate decides how far the multimodal prediction is allowed to move the numeric one.],
) <fig-solarvlm>

Solar-VLM represents the vision--language family @pvvlm @unicast. Its visual pathway ends
in one pooled vector per station, which is concatenated with the temporal state and passed through a
fusion network; a modality gate then interpolates between the multimodal prediction and the
numeric-only prediction. As in the previous two cases, the visual summary is fixed before the model
has committed to any particular lead time.

== Vision without a scene — Time-VLM

#figure(
  pipe(
    dbox[the *series itself* rendered as a 3-channel image, plus a text prompt of its own statistics],
    dbox[frozen vision--language embedder],
    dbox[*one pooled vision + text vector* per window],
    hit[cross-attention with the *query on the temporal side*; a gate blends the result against the numeric-only prediction],
    dbox[linear head emits all lead times per variate],
  ),
  caption: [Time-VLM @timevlm. The imagery is manufactured from the numeric input rather than
  observed, and the pooled multimodal vector serves as keys and values for a query formed from the
  temporal state.],
) <fig-timevlm>

Time-VLM is the strongest published multimodal entry in the suite (rank 2, SS 0.5404,
@tbl-leaderboard-multimodal) and it is instructive precisely because its second modality carries no
external information. There is no camera and no satellite: the past window is plotted into a
three-channel image, a prompt is assembled from the window's own minimum, maximum, median and trend
direction, and both are embedded by a frozen vision--language model. Whatever the visual branch
contributes is therefore a re-encoding of data the numeric branch already has. Its accuracy comes
from reusing visual pretraining as an inductive bias over the series — the same direction as
VisionTS++ @visionts — not from seeing anything new.

Structurally it repeats the pattern of the preceding three. Both modalities are pooled to a single
vector per window; that vector is used as keys and values while the query is formed from the
temporal features; and a learned gate interpolates between the multimodal path and a numeric-only
prediction. The visual representation is again complete before any lead time is distinguished.

The consequence for this report is a measurement one. A model that scores well with a visual channel
that provably carries no new information is a warning about attributing a leaderboard gain to a
modality, and it is the reason the counterfactual instrument of @sec-instrument is defined on the
imagery rather than on the architecture.

== Exogenous information aligned to the horizon — iTransformer

#figure(
  pipe(
    dbox[target series + weather and solar-geometry covariates],
    dbox[each *variate* becomes one token spanning the whole window],
    hit[covariate window *shifted onto the forecast interval*: the tokens carry the horizon's own weather],
    dbox[self-attention *across variates*, feed-forward along time],
    dbox[linear head emits all lead times],
  ),
  caption: [iTransformer @itransformer with covariates. Inverting the transformer's axes makes each
  variate a token; with the covariate window shifted forward, the exogenous channel is already
  aligned to the lead times being predicted.],
) <fig-itransformer>

iTransformer is not a multimodal architecture and is included for a different reason: it holds the
best ramp NMAE of the whole suite (0.1445, @tbl-leaderboard-deep), better than every multimodal
model tested including S2c. It inverts the usual axes — a token is a whole variate rather than a
timestep — so attention runs across the target and its eleven covariates while the feed-forward path
runs along time. Under this report's protocol it is run with the covariate window shifted onto the
forecast interval, which is the deployable-NWP assumption the MMTSFM arms are also trained under and
what makes the two modality-identical.

That shift is the point. The exogenous channel that wins on ramps is one whose content is *already
indexed by the lead time it is meant to inform*: the token describing cloud cover describes the cloud
cover of the hours being predicted, not of the hours already observed. No architecture is needed to
route it to the right horizon, because the alignment is in the data. Satellite imagery has the
opposite property — it is an observation of the past, and something in the model must decide which
part of it pertains to which future step. iTransformer therefore sets the ramp-metric reference point
this report is measured against, and simultaneously illustrates, from the numeric side, the property
that S2c has to construct architecturally.

== The shared assumption <sec-shared-assumption>

@fig-sunset, @fig-crossvivit, @fig-solarvlm and @fig-timevlm differ in encoder family, in training
regime, in whether the second modality is even observed, and in whether cross-attention is used at
all, yet they agree on one structural choice: *the auxiliary representation is finalised before the
horizon is resolved, and every lead time receives the same representation*. Retrieval-augmented and
covariate-adaptation methods for frozen backbones @tsrag @crossrag @cora make the same choice for a
different auxiliary modality. @fig-itransformer is the exception that locates the cost of that
choice: it is the only strong entry whose exogenous channel is aligned to the forecast interval
rather than summarised ahead of it, and it is the one that wins on ramps. The experiments in this
report are designed to test whether the shared choice is what prevents a frozen forecaster from using
satellite imagery.

= The dataset <sec-dataset>

== Construction

The corpus fuses four public sources into one standardised table keyed by
`(site_id, timestamp_utc)`, plus a single image archive addressed by a timestamp-exact per-row
pointer (@tbl-data-sources). Power is normalised by audited installed capacity, so the target is
dimensionless and comparable across installations of very different size.

#tbl(
  [Sources fused into the corpus. Every numeric row carries its own frame pointer, so the numeric
  and visual tracks are aligned by construction rather than by post-hoc matching.],
  columns: (auto, 1fr),
  align: (left, left),
  table.header[Track][Source and processing],
  [PV power, UK], [`openclimatefix/uk_pv` @ukpv: 30-minute generation for 2019--2020 with per-site
  capacity and rounded coordinates. Energy over the interval is converted to power; gaps of at most
  three steps are linearly interpolated, longer ones dropped.],
  [Satellite, UK], [EUMETSAT SEVIRI Rapid Scanning Service @seviri, *non-HRV* multi-band product,
  reprojected per site and cropped to 128 #sym.times 128 px (#sym.tilde.op 128 km).],
  [PV power and satellite, US], [NREL PVDAQ systems from the OEDI data lake @pvdaq paired with
  GOES-16 crops at 256 #sym.times 256 px.],
  [Weather covariates], [Open-Meteo historical archive @openmeteo: eight variables joined to each
  row by nearest timestamp. Solar geometry and a Haurwitz clear-sky GHI are computed analytically.],
) <tbl-data-sources>

== What is new about it <sec-data-novelty>

Four properties distinguish this corpus from the public multimodal PV datasets it is closest to.
They are summarised in @tbl-data-novelty; @tbl-data-compare places the corpus against those datasets
directly.

#tbl(
  [The four properties that are new, and what each one makes measurable. Each is a prerequisite for
  a claim this report makes.],
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  table.header[Property][What it is][What it enables],
  [Cross-region, single schema],
  [110 installations on two continents — 100 UK residential rooftops at 1.5--4.0 kW and 10 US
  systems at 1.8--408 kW — in one table with one column contract, one normalisation and one set of
  quality flags.],
  [Capacity- and climate-transfer questions can be asked without an ETL rewrite; the same model code
  consumes both regions.],

  [Multi-band infrared imagery],
  [Three genuinely distinct bands (inter-channel correlation 0.67--0.91, mean $|R-G| = 16.1$ DN)
  rather than the replicated grayscale channel an HRV crop provides, and coverage through the night:
  December pre-dawn frames carry mean 134 / std 37, against midday mean 123 / std 38.],
  [Cloud fields are observable before sunrise, so a 14-day history window is visually covered rather
  than half blank. 96.1% of scored windows carry all eight frames.],

  [Frames denser than power],
  [Imagery at 15-minute cadence against a 30-minute power grid, 4,103,892 frames over 110 per-site
  groups (98 GB), addressed by a local-to-group index that is timestamp-exact for both regions.],
  [Advection can be measured rather than assumed: frame-to-frame mean absolute difference is 8.3 DN
  at #sym.Delta$t$ = 15 min and 35.3 DN at 2 h, against a within-frame std of 38.6 DN — cloud-motion
  information is effectively exhausted by roughly two hours ahead.],

  [Cross-plant split by design],
  [Train, validation and test partitions are disjoint *by installation*, not by time; all three
  share the same two calendar years and share no rooftop.],
  [The evaluation measures generalisation to an unseen installation, which is the deployment case,
  and removes the per-site memorisation that a chronological split leaves available.],
) <tbl-data-novelty>

#tbl(
  [Positioning against the public multimodal PV corpora. The comparison is structural: each of the
  others is organised as fixed short windows for a fixed task, whereas this corpus is a tall
  `(site, timestamp)` table from which any window definition can be cut.],
  columns: (auto, auto, 1.15fr, 1.15fr, 0.95fr),
  align: (left, left, left, left, left),
  stroke: (x, y) => (
    top: if y == 0 { 0.9pt } else if y == 1 { 0.6pt } else { 0.35pt + luma(215) },
    bottom: 0pt,
  ),
  table.header[Corpus][Sites / region][Imagery][Layout][Split],
  [*This work*], [110, UK + US], [SEVIRI non-HRV 128², GOES-16 256², 15-min], [tall table + frame
  pointer; any window], [*cross-plant*, disjoint sites],
  [ClimateHackAI 2023 @climatehackai], [Great Britain], [SEVIRI HRV + 11-band non-HRV, 5-min],
  [fixed 1 h in #sym.arrow.r 4 h out windows, plus NWP and air-quality tensors],
  [competition split; no paper],
  [MMSP / FusionSF @fusionsf], [88, one Chinese province], [Himawari-8/9, 64², hourly],
  [fixed day-ahead 24 h in #sym.arrow.r 24 h out], [as released],
  [Sky-camera corpora @skippd], [single site each], [ground fisheye RGB, sub-minute],
  [continuous, single location], [chronological, within site],
) <tbl-data-compare>

Neither ClimateHackAI nor MMSP is unusable for the question posed here, but both would require
re-windowing and re-sampling before they could serve a long-history cross-plant protocol: they are
stored as pre-cut input--output tensors for a fixed nowcasting or day-ahead task, whereas the
protocol in @sec-protocol needs 14 days of history per origin and a split that holds whole
installations out.

== Quality control

Two UK installations carry a data-quality flag and are dropped, leaving 98. Row-level flags mark
outages (15,486 rows), stuck sensors (1,318) and night-clamped values (1,535); flagged targets are
excluded from scoring rather than imputed.

= Experimental protocol <sec-protocol>

The experiments in this report are run on the UK track of @sec-dataset. All metrics are normalised by installed
capacity, so they are comparable across systems of different size, and all are computed on daytime
steps only. @tbl-setup gives the full setting; every configuration in this report shares it.

#tbl(
  [The experimental setting and evaluated metrics. All configurations in this report share this
  setting.],
  columns: (auto, 1fr),
  align: (left, left),
  table.header[Quantity / Metric][Specification / Description],
  [Installations], [100 UK residential rooftops, 1.5–4.0 kW; two carry a data-quality flag and are dropped, leaving 98],
  [Sampling], [every 30 minutes, years 2019 and 2020],
  [History window], [14 days = 672 samples of the plant's own power],
  [Forecast horizon], [6 hours = 12 samples, 9 quantiles emitted per step],
  [Covariates], [_Weather_: temperature, cloud cover, wind speed, precipitation, radiation, irradiance. _Solar geometry_: solar zenith, solar azimuth, day-of-year, solar time, clear-sky GHI],
  [Visual signal], [8 satellite frames at 224 × 224 covering the 6 hours ending at the origin, centred on the installation],
  [Split], [70/15/15 *by installation*, not by time: train, validation and test share the same two years and share no rooftop],
  [Evaluation set], [14 test installations, 165,295 scored daytime steps],
  [Seeds], [42, 43, 44],
  [Skill score ($"SS"$, $arrow.t$)], [$"SS" = 1 - "NRMSE"_"model" \/ "NRMSE"_"ref"$ against Smart Persistence ($"SS" = 0$). Higher is better; one means perfection.],
  [Ramp NMAE ($arrow.b$)], [NMAE restricted to steps where true power changes sharply between consecutive samples — a cloud edge crossing the array (lower is better).]
) <tbl-setup>

Smart Persistence is the field's standard null hypothesis and it is a strong one: on clear or
uniformly overcast days it is nearly unbeatable. Ramps deserve the same emphasis. They are a small
minority of steps, but they carry nearly all of the operational cost of a forecast error, and they
are the steps at which a satellite view could in principle help.

= Method

== Components

@tbl-components lists the four components and their training state. Only the last two carry
gradients; together they account for a small fraction of the parameters in the stack.

#tbl(
  [Components of the stack and their training state. The fusion module is the only element that
  differs between the four configurations of @sec-configs.],
  columns: (auto, 1fr, auto),
  align: (left, left, center),
  table.header[Component][Role and dimensions][State],
  [Chronos-2 @chronos2], [Pretrained encoder-only time-series transformer.], [frozen, top 3 of 12 blocks unfrozen],
  [V-JEPA 2 @vjepa2], [Self-supervised video encoder. Encodes the 8-frame clip once, offline, into 4 temporal slices × 196 spatial patches × 1024 channels, cached.], [frozen entirely],
  [Channel projection], [Maps the visual channel width onto the backbone's width.], [trained],
  [Fusion module], [The only thing that differs between configurations.], [trained],
) <tbl-components>

=== The forecasting backbone

The backbone treats a series the way a transformer treats any sequence: it is cut into
non-overlapping patches of 16 consecutive samples, each patch is linearly projected into a token,
and a 672-step history becomes 42 context tokens. Future positions are represented by additional
placeholder tokens carrying only their known covariates; self-attention over the whole assembly lets
those future positions absorb information from the past, and a projection head reads the answer off
them.

The backbone is general-purpose: it was pretrained on heterogeneous series and has no notion of
irradiance, panel tilt or clear-sky curves, so nothing about the solar domain is baked in. It also
consumes covariates natively, which means the numeric-only control is already a strong model and the
visual signal must earn its place against a high bar — as @tbl-leaderboard-deep confirms, the
fine-tuned backbone alone outranks every published multimodal forecaster in the suite except one.

=== The visual encoder

The choice of a self-supervised video model rather than a caption-aligned one is deliberate. The
relevant visual content — the motion and deformation of cloud fields — is temporal and has no useful
textual description; a representation trained to match captions would discard exactly the structure
that matters. V-JEPA's predictive objective in latent space is, by construction, a model of *how a
scene changes*, which is the property the forecaster needs. The encoder is never run during
training: clips are encoded once, offline, and cached.

== The four configurations <sec-configs>

The three fusion arms are best understood not as three architectures but as three answers to a
single structural question: *along which axis does the visual information enter?* @fig-taxonomy
summarises the three answers and the outcome of each; @tbl-ladder gives the corresponding numbers.

#figure(
  block(width: 100%)[
    #grid(
      columns: (1fr, 1fr, 1fr),
      column-gutter: 0.7em,
      row-gutter: 0.45em,
      dbox[*S2a* — batch axis], dbox[*S2b* — sequence axis], hit[*S2c* — neither axis],
      ad, ad, ad,
      dbox[clip pooled to *1 vector*], dbox[clip pooled to *1 or 16 tokens*], dbox[clip kept *unpooled*: 4 × 4 grid × 4 slices = *64 tokens*],
      ad, ad, ad,
      dbox[inserted as an extra parallel row, reachable by group self-attention], dbox[appended to the token sequence, mixed by ordinary self-attention], dbox[held *outside* the sequence as an external key–value memory],
      ad, ad, ad,
      dbox[one summary, identical for every lead time], dbox[one summary, identical for every lead time], hit[each of 3 lead-time slots issues *its own query*, in each of the last 4 encoder blocks],
      ad, ad, ad,
      dbox[reliance indistinguishable from zero], dbox[reliance indistinguishable from zero], hit[reliance measurable and stable across seeds],
    )
  ],
  caption: [The placement taxonomy. The first two arms differ from each other in axis but agree on
  the choice identified in @sec-shared-assumption — a single visual summary, fixed before the
  horizon is resolved. Exact reliance figures are in @tbl-ladder.],
) <fig-taxonomy>

#v(0.3em)

=== S1 — the vision-free control

#figure(
  image("figures/mmtsfm_s1.svg", width: 100%),
  caption: [S1, the numeric-only control: the backbone with the visual pathway absent entirely.],
) <fig-s1>

S1 (@fig-s1) is the same backbone with no imagery at all. It exists to define the baseline against
which reliance is measured, separating a multimodal model's advantage from the effect of the
fine-tuning recipe. Every subsequent arm is initialised from this checkpoint and trained with the
same schedule, so any difference is attributable to the visual pathway.

=== S2a — fusion on the batch axis

#figure(
  image("figures/mmtsfm_s2a.svg", width: 100%),
  caption: [S2a, pooled late fusion: the clip becomes one descriptor presented as an extra parallel
  channel.],
) <fig-s2a>

In S2a (@fig-s2a) the clip is pooled to a single descriptor and presented to the backbone as an
extra parallel channel — a summary of the state of the sky, available to every forecast position
equally. This is textbook late fusion, structurally equivalent to the pooled pathway of
@fig-solarvlm, and it was run first precisely *because* it is the standard answer: a null result
here is informative rather than an omission.

The rationale for expecting it to work is reasonable. The most obvious use of a satellite view is as
a coarse regime indicator — clear, broken, overcast — and a single vector is an adequate carrier for
a regime label. The rationale for expecting it to fail is equally clear in hindsight: the pooling
operation compresses roughly 800 visual elements into one, and it does so *before anything has been
asked of them*. The summary must be computed without knowing which part of the scene is relevant,
and by the time the forecaster is in a position to have an opinion, the discarded detail is gone.

=== S2b — fusion on the sequence axis

#figure(
  image("figures/mmtsfm_s2b.svg", width: 100%),
  caption: [S2b, mid-sequence injection: visual tokens are appended to the token sequence and mixed
  by ordinary self-attention.],
) <fig-s2b>

S2b (@fig-s2b) stops treating vision as a parallel channel and makes it part of the sequence, so
that ordinary self-attention can relate visual tokens to numeric ones directly. Two widths were run
— one visual token and sixteen — to test whether the failure of S2a was a matter of bandwidth.

Widening the channel from 1 token to 16 did not help; reliance stayed at zero within noise in both
settings (@tbl-ladder). That comparison rules out the bandwidth explanation and points at something
structural: the problem is not *how much* visual information is admitted but *when* it is summarised
relative to when it is needed. In both S2a and S2b the imagery is condensed into a fixed
representation before the model has formed any view about the forecast, and every forecast position
then receives the same representation — the assumption identified in @sec-shared-assumption.

=== S2c — the forecast queries the sky

#figure(
  image("figures/mmtsfm_s2c.svg", width: 100%),
  caption: [S2c, future-position cross-attention: the imagery is retained unpooled and queried by
  each lead-time slot.],
) <fig-s2c>

S2c (@fig-s2c) changes the direction of the operation. The imagery is not inserted anywhere. It is
retained *unpooled* as an external memory — the 14 × 14 patch grid is block-pooled to 4 × 4 and all
four temporal slices are kept, giving 64 key–value tokens — and the forecast horizon is subdivided
into three lead-time slots. Each slot issues its own cross-attention query against that memory, in
each of the backbone's last four encoder blocks. No summarisation happens anywhere: the pooled
descriptor and the projection adapter used by the other arms are both bypassed.

*What a query is.* A query is not made of the quantity being predicted. It is a description of what
is being looked for, and it exists before the thing it retrieves. Each forecast slot's query is
assembled from two ingredients, both available before any forecast value exists: a learned lead-time
identity — a vector attached permanently to the slot at a given horizon, shared across all samples
and installations and fixed after training — and the slot's own hidden state after self-attention
over the history and the known covariates, which encodes a belief about what is likely to happen
rather than the outcome. @fig-order gives the resulting order of operations.

#figure(
  block(width: 100%)[
    #grid(
      columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
      column-gutter: 0.25em,
      align: horizon,
      dbox[self-attention populates the slot from history and covariates],
      ar,
      dbox[slot forms a *query*: lead-time identity + its own belief],
      ar,
      hit[query reads the *64 unpooled visual tokens*],
      ar,
      dbox[result *gated back* onto that slot alone],
      ar,
      dbox[projection head emits the number],
    )
    #v(0.4em)
  ],
  caption: [Order of operations for one lead-time slot in S2c, repeated in each of the last four
  encoder blocks. Each pass refines the slot's question with what the previous round returned.],
) <fig-order>

*Why the slots ask different things.* Each lead-time slot receives gradient only from its own
horizon's errors. The two-hour slot is penalised for two-hour mistakes, the five-hour slot for
five-hour mistakes, and cloud fields relevant at those two horizons sit at different distances from
the array. Nothing in the design instructs a slot to attend to cloud edges; the differentiation is
the accumulated residue of one horizon's own past errors.

*Relation to prior fusion.* Conventional late fusion (@fig-sunset, @fig-solarvlm) compresses the
auxiliary modality into a shared representation before combining it with the time-series state,
which forces one visual summary to serve every horizon even where different horizons need different
parts of the scene. Cross-attention alone is not the distinguishing feature either: CrossViViT
(@fig-crossvivit) already uses it, but its queries originate in the *history* representation and
enrich the context before a separate prediction stage. In S2c the queries originate in the *future*
positions and read an unpooled memory directly, so each horizon can retrieve different visual
evidence.

= Measuring reliance <sec-instrument>

After training, the model is frozen and the test set is evaluated twice: once normally, and once
with the visual query switched off. The difference between the two passes is the *reliance* — the
share of the model's accuracy that depends on it having seen the sky.

This is a stronger instrument than comparing a multimodal model to a unimodal one, because it holds
every weight fixed. It cannot be confounded by capacity, initialisation or training schedule; the
two passes differ only in whether the imagery was available.

= Results

== The placement result

@tbl-ladder is the central finding. The reliance column reports the ramp-error improvement
attributable to the imagery, measured by the counterfactual pass of @sec-instrument. Reliance is
flat across the two arms that summarise the clip in advance, including the sixteen-token variant,
and rises by an order of magnitude only when the forecast positions query the memory themselves.

#tbl(
  [The placement ladder. Reliance is the ramp-error improvement attributable to the imagery,
  measured by switching the visual pathway off at inference on frozen weights. Intervals are across
  seeds 42--44.],
  columns: (auto, auto, auto),
  align: (left, left, center),
  table.header[Arm][Where vision enters][Reliance (ramp)],
  [S2a], [batch axis, pooled], [0.0000 ± 0.0015],
  [S2b], [sequence axis, 1 token], [0.0006],
  [S2b wide], [sequence axis, 16 tokens], [0.0002 ± 0.0016],
  [*S2c*], [*future-position queries*], [*0.0056 ± 0.0006*],
) <tbl-ladder>

== Ablation status

@tbl-done reports the two controls that have been executed. The temporal-shuffle control confirms
that the arms whose reliance is zero are genuinely not reading the imagery, and the stale-sky
control establishes that S2c's gain is timing-dependent rather than a plant-level constant.
@tbl-open lists the four further ablations that are configured but not yet launched, together with
the outcome each would produce under the hypothesis that the forecast-side query is the operative
mechanism.

#tbl(
  [Executed ablations and their measured outcomes.],
  columns: (auto, 1fr, 1fr),
  align: (center, left, left),
  table.header[\#][Question it answers][Result],
  [1], [Does frame order matter to S2c? (temporal shuffle)],
  [Bit-identical to the unperturbed run at every digit.],
  [2], [Is the model reading the sky, or is it reading a plant-level constant?],
  [A stale sky is worse than no sky — vision is read, and read for its timing.],
) <tbl-done>

#tbl(
  [Open ablations, configured but not launched.],
  columns: (auto, 1fr, 1fr, auto),
  align: (center, left, left, center),
  table.header[\#][Question it answers][Expected behaviour][GPU-h],
  [3], [Summariser widened to 4 temporal slices of 4 × 4 blocks — S2c's payload with S2b's fixed-latent queries.],
  [Reliance ≈ 0 → the mechanism is the forecast-side query.], [≈ 24],
  [4], [Grid or decoder? S2c with the grid collapsed to 1 × 1],
  [Reliance falls to the S2b level if the grid is what matters; holds if the decoder is.], [≈ 24],
  [5], [Grid held at 4 × 4, decoder collapsed to one position.],
  [Reliance holds if the grid is the mechanism; falls if the 3-slot decoder is.], [≈ 24],
  [6], [Is the grid spatially grounded? Frames swapped with another plant's.],
  [NMAE degrades below the vision-free control and reliance turns negative if the gain is
  plant-specific; indifference would indicate a generic cloudiness prior.], [< 1],
) <tbl-open>

== Placement against the baseline suite

The four arms are scored against 32 baselines under the identical protocol of @tbl-setup.
@tbl-leaderboard-multimodal reports the multimodal field, @tbl-leaderboard-deep the unimodal deep
and foundation-model field, and @tbl-leaderboard-baselines the tabular, classical and reference
tiers. Rank is the global position by skill score across all three tables.

Three observations follow. First, the ordering of the four arms in @tbl-leaderboard-multimodal
matches the reliance ordering of @tbl-ladder exactly, which is what makes the reliance measure
worth trusting as more than an internal diagnostic. Second, the vision-free control alone
(SS 0.5230) already outranks every published multimodal forecaster in the suite except Time-VLM —
whose second modality is manufactured from the numeric input rather than observed (@fig-timevlm) —
which is a statement about the strength of the frozen backbone rather than about those methods.
Third, the best ramp NMAE in the entire suite belongs to a unimodal supervised model, iTransformer
with forward-shifted covariates (@fig-itransformer), so S2c's advantage is specific to the reliance
measurement and does not yet translate into a ramp-metric win; the gap is against an exogenous
channel that is aligned to the forecast interval by construction rather than by architecture.

#tbl(
  [Multimodal leaderboard.],
  columns: (auto, auto, auto, 1fr, auto, auto),
  align: (center, left, left, left, center, center),
  stroke: (x, y) => (
    top: if y == 0 { 0.9pt } else if y == 1 { 0.6pt } else { 0.35pt + luma(215) },
    bottom: 0pt,
  ),
  table.header[Rank][Model][Second modality][Where the fusion sits][Skill score][Ramp NMAE],

  table.cell(colspan: 6, fill: luma(240))[*This work (MMTSFM placement arms)*],
  [1], [*MMTSFM S2c* (ours)], [satellite], [cross-attention from *future* positions to unpooled memory], [*0.5470*], [0.1461],
  [3], [MMTSFM S2b wide (ours)], [satellite], [sequence axis, 16 appended tokens (self-attention)], [0.5352], [0.1484],
  [4], [MMTSFM S2b (ours)], [satellite], [sequence axis, 1 appended token (self-attention)], [0.5322], [0.1487],
  [5], [MMTSFM S2a (ours)], [satellite], [batch axis, pooled vector (group self-attention)], [0.5258], [0.1487],
  [7], [MMTSFM S1 control (ours)], [none], [— (vision-free control)], [0.5230], [0.1506],

  table.cell(colspan: 6, fill: luma(240))[*Multimodal forecasters (satellite & pseudo-image)*],
  [2], [Time-VLM @timevlm], [series *rendered as* images], [pooled vision + text, query on the temporal side (@fig-timevlm)], [0.5404], [—],
  [13], [Solar-VLM @solarvlm], [satellite + text], [vision–language fusion, multi-site (@fig-solarvlm)], [0.4396], [0.1514],
  [21], [CrossViViT @crossvivit], [satellite], [cross-attention from history timesteps (@fig-crossvivit)], [0.3491], [—],
  [26], [Aurora @aurora], [several], [joint multimodal pretraining], [0.2324], [—],
  [27], [SUNSET @sunset], [sky/satellite], [convolutional precedent; joint encoding (@fig-sunset)], [0.2162], [—],
  [28], [UniCast @unicast], [several], [prompting a foundation forecaster], [0.1211], [—],
  [30], [VisionTS++ @visionts], [series *rendered as* images], [continual pretraining of a visual backbone], [0.0167], [—],
) <tbl-leaderboard-multimodal>

#tbl(
  [Unimodal deep learning and foundation model baselines.],
  columns: (auto, auto, auto, 1fr, auto, auto),
  align: (center, left, left, left, center, center),
  stroke: (x, y) => (
    top: if y == 0 { 0.9pt } else if y == 1 { 0.6pt } else { 0.35pt + luma(215) },
    bottom: 0pt,
  ),
  table.header[Rank][Model][Second modality][Where the fusion sits][Skill score][Ramp NMAE],

  table.cell(colspan: 6, fill: luma(240))[*Supervised deep learning*],
  [6], [iTransformer + covariates @itransformer], [covariates], [channel-inverted self-attention over variates (@fig-itransformer)], [0.5257], [*0.1445*],
  [12], [PatchTST @patchtst], [none], [—], [0.4581], [0.1543],
  [14], [Temporal Fusion Transformer @tft], [covariates], [gated residual & temporal self-attention], [0.4264], [0.1605],
  [15], [MLP], [none], [—], [0.4219], [0.1624],
  [22], [DLinear @dlinear], [none], [—], [0.3231], [0.1746],

  table.cell(colspan: 6, fill: luma(240))[*Retrieval & frozen-backbone adaptation*],
  [9], [TS-RAG @tsrag], [retrieved *numeric* history], [concatenated to the context], [0.4779], [—],
  [10], [Cross-RAG @crossrag], [retrieved *numeric* history], [cross-attention between query and retrievals], [0.4768], [—],
  [18], [CoRA @cora], [covariates], [residual adapter on frozen backbone], [0.3798], [0.1624],

  table.cell(colspan: 6, fill: luma(240))[*Time-series foundation models (zero-shot & fine-tuned)*],
  [8], [Chronos-2, fine-tuned @chronos2], [covariates], [group self-attention, fine-tuned], [0.5042], [0.1494],
  [11], [Chronos-2, zero-shot @chronos2], [covariates], [group self-attention, zero-shot], [0.4737], [0.1544],
  [20], [TTM, fine-tuned @ttm], [covariates], [MLP-Mixer temporal/channel mixing, fine-tuned], [0.3601], [0.1716],
  [23], [TiRex, zero-shot @tirex], [none], [—], [0.2873], [0.1826],
  [24], [TimesFM, zero-shot @timesfm], [none], [—], [0.2708], [0.1902],
  [32], [TTM, zero-shot @ttm], [covariates], [MLP-Mixer, zero-shot], [−0.0807], [0.2922],
) <tbl-leaderboard-deep>

#tbl(
  [Tabular, classical ML, and reference baselines.],
  columns: (auto, auto, auto, 1fr, auto, auto),
  align: (center, left, left, left, center, center),
  stroke: (x, y) => (
    top: if y == 0 { 0.9pt } else if y == 1 { 0.6pt } else { 0.35pt + luma(215) },
    bottom: 0pt,
  ),
  table.header[Rank][Model][Second modality][Where the fusion sits][Skill score][Ramp NMAE],

  table.cell(colspan: 6, fill: luma(240))[*Tabular models & classical ML*],
  [16], [TabPFN @tabpfn], [none], [—], [0.4063], [0.1631],
  [17], [LightGBM], [covariates], [gradient-boosted trees over tabular lags], [0.3854], [0.1672],
  [19], [TabFM ensemble], [none], [—], [0.3626], [0.1573],

  table.cell(colspan: 6, fill: luma(240))[*Reference & statistical baselines*],
  [25], [Hourly climatology], [none], [—], [0.2337], [0.1665],
  [29], [Seasonal naive], [none], [—], [0.1068], [0.2056],
  [31], [Persistence], [none], [—], [0.0141], [0.2550],
) <tbl-leaderboard-baselines>

= Conclusion

Under a frozen-backbone protocol that holds capacity, initialisation and schedule constant, the
point at which a visual representation is attached determines whether a forecaster uses it at all.
Pooling the clip onto the batch axis and appending it to the token sequence both produce reliance
indistinguishable from zero, and the sixteen-token variant shows that this is not a bandwidth
limitation. Reliance becomes measurable only when the visual evidence is kept unpooled and queried
by the forecast positions themselves. The shared assumption of the prior architectures reproduced in
@sec-related — one visual summary, fixed before the horizon is resolved — is therefore a plausible
explanation for why frozen multimodal forecasters so often fail to use the modality they were given.

The open ablations of @tbl-open are what would separate the two candidate mechanisms inside S2c, the
unpooled spatial grid and the per-lead-time decoder, and they are the immediate next step.

#bibliography("refs.bib", title: "References", style: "ieee")
