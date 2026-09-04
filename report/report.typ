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
  block(width: 100%, breakable: true, text(size: 9pt, table(..args))),
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
  align(center + horizon, text(size: 8.4pt, body)),
)
#let hit(body) = dbox(body, fill: luma(233), stroke: 1pt + luma(90))
#let ar = align(center + horizon, text(size: 11pt, fill: luma(120))[#sym.arrow.r])
#let ad = align(center, text(size: 11pt, fill: luma(120))[#sym.arrow.b])
#let dcap(body) = align(center, text(size: 8.2pt, style: "italic", fill: luma(90), body))

// ─────────────────────────────────────────────────────────────── title

#align(center)[
  #block(text(size: 18pt, weight: 700)[
    Where Does Vision Belong in a \ Frozen Time-Series Forecaster?
  ])
]

#v(1.2em)

= Settings

All metrics are normalised by installed capacity, so they are comparable across systems of
different size, and all are computed on daytime steps only.

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
uniformly overcast days it is nearly unbeatable.

Ramps deserve the same emphasis. They are a small minority of steps, but they carry nearly all of the
operational cost of a forecast error.

= The research question

The multimodal forecasting literature is largely a literature of proposals: an architecture is
introduced, it is trained end to end, it beats a set of unimodal baselines, and the improvement is
attributed to the additional modality. That attribution is rarely tested. When both encoders are
trained jointly, an accuracy gain is consistent with at least three explanations — the model
learned to read the images, the extra capacity helped, or the extra gradient path acted as
regularisation.

This work inverts the emphasis. Both encoders are *pretrained and frozen*, so neither can adapt to
the other; only a small bridge between them is learned. The architecture is then held fixed and
the *attachment point* is varied. The question becomes:

#align(center)[
  #block(width: 88%, inset: 0.7em, text(size: 10.6pt, style: "italic")[
    Given a frozen sequence forecaster and a frozen visual encoder, at which point in the
    forecaster's computation must the visual representation be introduced before the forecaster
    demonstrably uses it?
  ])
]

Freezing is what makes the question answerable. With both backbones fixed, no configuration can
win by having more capacity than another, and any measured difference is attributable to the
wiring. It is also what makes the result portable: the finding is a statement about placement in a
frozen pipeline, not about one particular set of weights.

= Components

#tbl(
  [],
  columns: (auto, 1fr, auto),
  align: (left, left, center),
  table.header[Component][Role and dimensions][State],
  [Chronos-2], [Pretrained encoder-only time-series transformer.], [frozen, top 3 of 12 blocks unfrozen],
  [V-JEPA 2], [Self-supervised video encoder. Encodes the 8-frame clip once, offline, into 4 temporal slices × 196 spatial patches × 1024 channels, cached.], [frozen entirely],
  [Channel projection], [Maps the visual channel width onto the backbone's width.], [trained],
  [Fusion module], [The only thing that differs between configurations.], [trained],
) <tbl-components>

== The forecasting backbone

The backbone treats a series the way a transformer treats any sequence: it is cut into
non-overlapping patches of 16 consecutive samples, each patch is linearly projected into a token,
and a 672-step history becomes 42 context tokens. Future positions are represented by additional
placeholder tokens carrying only their known covariates; self-attention over the whole assembly
lets those future positions absorb information from the past, and a projection head reads the
answer off them.

The backbone is genuinely general-purpose: it was pretrained on heterogeneous series and has no
notion of irradiance, panel tilt or clear-sky curves, so nothing about the solar domain is baked
in. It also consumes covariates natively, which means the numeric-only control is already a strong
model and the visual signal must earn its place against a high bar.

== The visual encoder

The choice of a self-supervised video model rather than a caption-aligned one is deliberate. The
relevant visual content — the motion and deformation of cloud fields — is temporal and has no
useful textual description; a representation trained to match captions would discard exactly the
structure that matters. V-JEPA's predictive objective in latent space is, by construction, a model
of *how a scene changes*, which is the property the forecaster needs. The encoder is never run
during training.

= The four configurations

The three fusion arms are best understood not as three architectures but as three answers to a
single structural question: *along which axis does the visual information enter?* 

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
      dbox[reliance *0.0000*], dbox[reliance *0.0002*], hit[reliance *0.0056*],
    )
  ],
  caption: [],
) <fig-taxonomy>

#v(0.3em)

== S1 — the vision-free control

#figure(
  image("figures/mmtsfm_s1.svg", width: 100%),
  caption: [S1, the numeric-only control.],
)

S1 is the same backbone with no imagery at all. It exists to define the baseline against which
reliance is measured, ensuring that a multimodal
model's advantage came from the fine-tuning recipe rather than from the second modality. Every
subsequent arm is initialised from this checkpoint and trained with the same schedule, so any
difference is attributable to the visual pathway.

== S2a — fusion on the batch axis

#figure(
  image("figures/mmtsfm_s2a.svg", width: 100%),
  caption: [S2a, pooled late fusion.],
)

The clip is pooled to a single descriptor and presented to the backbone as an extra parallel
channel — a summary of "what today's sky is like", available to every forecast position equally.
This is textbook late fusion, and it was the first arm precisely *because* it is the textbook
answer: if the standard method works, the thesis is finished, and if it does not, the null is a
result rather than an omission.

The rationale for expecting it to work is reasonable. The most obvious use of a satellite view is
as a coarse regime indicator — clear, broken, overcast — and a single vector is an adequate carrier
for a regime label. The rationale for expecting it to fail is equally clear in hindsight: the
pooling operation compresses roughly 800 visual elements into 1, and it does so *before anything
has been asked of them*. The summary must be computed without knowing which part of the scene is
relevant, and by the time the forecaster is in a position to have an opinion, the discarded detail
is gone.

== S2b — fusion on the sequence axis

#figure(
  image("figures/mmtsfm_s2b.svg", width: 100%),
  caption: [S2b, mid-sequence injection.],
)

The natural next move is to stop treating vision as a parallel channel and make it part of the
sequence, so that ordinary self-attention can relate visual tokens to numeric ones directly. Two
widths were run — one visual token, and sixteen — to test whether the failure of S2a was simply a
matter of bandwidth.

The reasoning is sound and the result is instructive. Widening the channel from 1 token to 16 did
not help; the reliance stayed at zero within noise in both settings. That comparison is what
rules out the bandwidth explanation and points at something structural: the problem is not *how
much* visual information is admitted but *when* it is summarised relative to when it is needed.
In both S2a and S2b the imagery is condensed into a fixed representation before the model has
formed any view about the forecast, and every forecast position then receives the same
representation.


== S2c — the forecast queries the sky

#figure(
  image("figures/mmtsfm_s2c.svg", width: 100%),
  caption: [S2c, future-position cross-attention.],
)

S2c changes the direction of the operation. The imagery is not inserted anywhere. It is retained
*unpooled* as an external memory — the 14 × 14 patch grid is block-pooled to 4 × 4 and all four
temporal slices are kept, giving 64 key–value tokens — and the forecast horizon is subdivided into
three lead-time slots. Each slot issues its own cross-attention query against that memory, in each
of the backbone's last four encoder blocks. No summarisation happens anywhere: the pooled
descriptor and the projection adapter used by the other arms are both bypassed.


=== What a query actually is

A query is not made of the thing that is being predicted. It is a *description of what is being
looked for* — the same object as a search string, which exists before the document it retrieves.
Each forecast slot's query is assembled from exactly two ingredients, both available before any
forecast value exists: a learned lead-time identity, a vector attached permanently to "the slot two
hours out", shared across all samples and installations and fixed after training; and the slot's own
hidden state after self-attention over the history and the known covariates, which encodes a
*belief* about what is likely to happen, not the outcome.

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

=== Why the slots ask different things

Each lead-time slot receives gradient only from its own horizon's errors. The two-hour slot is
penalised for two-hour mistakes, the five-hour slot for five-hour mistakes, and cloud fields
relevant at those two horizons sit at different distances from the array. Nothing in the design
instructs a slot to attend to cloud edges; the differentiation is the accumulated residue of one
horizon's own past errors.

=== Why this is not late fusion, and not simply another cross-attention model

_Why this is not late fusion._ Conventional late fusion typically compresses an auxiliary modality into a shared representation before combining it with the time-series state. This creates a potential bottleneck: the same visual summary must support predictions at different future horizons, even though different horizons may require different information from the scene.

_Why this is not simply another cross-attention model._ Cross-attention itself is not the contribution. What matters is where the queries originate and what they retrieve. Existing multimodal forecasting approaches typically use historical time-series representations to query auxiliary features, enriching the context before a separate prediction stage. Here, instead, future lead-time slots query an unpooled visual memory directly. Each horizon can therefore retrieve different visual information relevant to its prediction.

= The measurement instrument

After training, the model is frozen
and the test set is evaluated twice: once normally, and once with the visual query switched off. The
difference between the two passes is the *reliance* — the amount of the model's accuracy that
depends on it having seen the sky.

This is a stronger instrument than comparing a multimodal model to a unimodal one, because it
holds every weight fixed. It cannot be confounded by capacity, initialisation, or training
schedule; the two passes differ only in whether the imagery was available.


= Results

== The placement result

@tbl-ladder is the central finding. The reliance column reports the ramp-error improvement
attributable to the imagery, measured by the counterfactual pass.

#tbl(
  [The placement ladder.],
  columns: (auto, auto, auto),
  align: (left, left, center),
  table.header[Arm][Where vision enters][Reliance (ramp)],
  [S2a], [batch axis, pooled], [0.0000 ± 0.0015],
  [S2b], [sequence axis, 1 token], [0.0006],
  [S2b wide], [sequence axis, 16 tokens], [0.0002 ± 0.0016],
  [*S2c*], [*future-position queries*], [*0.0056 ± 0.0006*],
) <tbl-ladder>


== Ablation status

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
  [2], [Time-VLM], [series *rendered as* images], [reuses visual pretraining in the opposite direction], [0.5404], [—],
  [13], [Solar-VLM], [satellite + text], [vision–language fusion, multi-site], [0.4396], [0.1514],
  [21], [CrossViViT], [satellite], [cross-attention from history timesteps], [0.3491], [—],
  [26], [Aurora], [several], [joint multimodal pretraining], [0.2324], [—],
  [27], [SUNSET], [sky/satellite], [convolutional precedent; joint encoding], [0.2162], [—],
  [28], [UniCast ], [several], [prompting a foundation forecaster], [0.1211], [—],
  [30], [VisionTS++], [series *rendered as* images], [continual pretraining of a visual backbone], [0.0167], [—],
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
  [6], [iTransformer + covariates], [covariates], [channel-inverted self-attention over variates], [0.5257], [*0.1445*],
  [12], [PatchTST], [none], [—], [0.4581], [0.1543],
  [14], [Temporal Fusion Transformer], [covariates], [gated residual & temporal self-attention], [0.4264], [0.1605],
  [15], [MLP], [none], [—], [0.4219], [0.1624],
  [22], [DLinear], [none], [—], [0.3231], [0.1746],

  table.cell(colspan: 6, fill: luma(240))[*Retrieval & frozen-backbone adaptation*],
  [9], [TS-RAG], [retrieved *numeric* history], [concatenated to the context], [0.4779], [—],
  [10], [Cross-RAG], [retrieved *numeric* history], [cross-attention between query and retrievals], [0.4768], [—],
  [18], [CoRA], [covariates], [residual adapter on frozen backbone], [0.3798], [0.1624],

  table.cell(colspan: 6, fill: luma(240))[*Time-series foundation models (zero-shot & fine-tuned)*],
  [8], [Chronos-2, fine-tuned], [covariates], [group self-attention, fine-tuned], [0.5042], [0.1494],
  [11], [Chronos-2, zero-shot], [covariates], [group self-attention, zero-shot], [0.4737], [0.1544],
  [20], [TTM, fine-tuned], [covariates], [MLP-Mixer temporal/channel mixing, fine-tuned], [0.3601], [0.1716],
  [23], [TiRex, zero-shot], [none], [—], [0.2873], [0.1826],
  [24], [TimesFM, zero-shot], [none], [—], [0.2708], [0.1902],
  [32], [TTM, zero-shot], [covariates], [MLP-Mixer, zero-shot], [−0.0807], [0.2922],
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
  [16], [TabPFN], [none], [—], [0.4063], [0.1631],
  [17], [LightGBM], [covariates], [gradient-boosted trees over tabular lags], [0.3854], [0.1672],
  [19], [TabFM ensemble], [none], [—], [0.3626], [0.1573],

  table.cell(colspan: 6, fill: luma(240))[*Reference & statistical baselines*],
  [25], [Hourly climatology], [none], [—], [0.2337], [0.1665],
  [29], [Seasonal naive], [none], [—], [0.1068], [0.2056],
  [31], [Persistence], [none], [—], [0.0141], [0.2550],
) <tbl-leaderboard-baselines>

