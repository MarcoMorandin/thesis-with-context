#set document(
  title: "Interleaved Vision Fusion for Time-Series Foundation Models: Evaluating Inductive Biases, Information Bottlenecks, and Representation Grounding in Photovoltaic Ramp Forecasting",
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
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.85em)[
  #set text(size: 14pt, weight: 700)
  #it
]
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.65em)[
  #set text(size: 11.5pt, weight: 700)
  #it
]
#show heading.where(level: 3): it => block(above: 0.9em, below: 0.5em)[
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
#let dcap(body) = align(center, text(size: 8.2pt, style: "italic", fill: luma(90), body))
#let ar = align(center + horizon, text(size: 8.5pt, fill: luma(120), sym.arrow.r))
#let ad = align(center, text(size: 8.5pt, fill: luma(120), sym.arrow.b))

#let pipe(..cells) = block(width: 100%)[
  #grid(
    columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
    column-gutter: 0.25em,
    align: horizon,
    ..cells.pos().intersperse(ar),
  )
  #v(0.35em)
]

// --- Title block ---
#align(center)[
  #block(width: 90%)[
    #set text(size: 17pt, weight: 700)
    Interleaved Vision Fusion for \ Time-Series Foundation Models
  ]
  #v(0.4em)
  #block(width: 85%)[
    #set text(size: 11.5pt, fill: luma(70), weight: 500)
    Evaluating Inductive Biases, Information Bottlenecks, and Representation Grounding in Photovoltaic Ramp Forecasting
  ]
]

#v(0.8em)

#block(
  width: 100%,
  inset: (x: 1.2em, y: 0.9em),
  radius: 3pt,
  fill: luma(248),
  stroke: (left: 2.5pt + luma(80)),
)[
  #set par(justify: true)
  #set text(size: 9.6pt)
  *Abstract.* Feeding high-dimensional visual observations into pretrained time-series foundation models addresses a failure mode of numeric forecasting: sudden physical regime shifts that leave no trace in the historical series. In solar power forecasting, most of the operational loss falls on sharp ramp events triggered by advecting cloud edges. Whether multimodal forecasters actually exploit visual observations is hard to establish from the existing literature, because end-to-end retraining conflates informational value with parameter capacity, optimization trajectories, and regularizing gradient paths. Modern time-series foundation models make the problem harder: they transfer well zero-shot, which leaves coarse visual summaries redundant. We remove these confounds with a controlled evaluation on fixed backbones, a fully frozen self-supervised video encoder and a time-series foundation transformer held fixed apart from its upper encoder blocks, where the cross-modal bridge is the only component trained from scratch. We introduce a counterfactual reliance protocol that scores identical frozen weights twice, once with the visual pathway switched off at inference, which gives the causal contribution of imagery denominated in ramp error. The standard architectural choice in multimodal forecasting, compressing the visual field into a low-dimensional summary via a learned-query bottleneck resampler, yields visual reliance indistinguishable from zero ($0.0000 plus.minus 0.0015$). A resampler-free architecture, which preserves spatial locality by delivering the patch field to the forecaster as 49 individually addressable cells, projects them position-wise into the sequence self-attention, and prunes redundant background via temporal novelty selection, restores statistically significant reliance ($0.0073 plus.minus 0.0002$) and reaches the lowest ramp error ($0.1435$) across a 27-baseline suite. A matched-budget component ablation then localizes that recovery and simplifies the architecture. Our first design gave each cell a 4096-channel payload built by concatenating its four sub-patches, on the premise that ramp prediction needs sub-cell spatial detail. Averaging the same four sub-patches into 1024 channels matches that design on ramp error, improves skill score by $0.0054$, and calibrates better, so we report the cheaper cell and withdraw the premise. What is load-bearing is the removal of the pooling bottleneck itself, together with the novelty criterion that selects which cells to keep. Diagnostic falsification controls locate that gain in localized spatial cloud boundaries acting as a non-stationary spatial feature field; kinematic temporal tracking and frame order contribute nothing measurable. The sharper ramp response carries a cost we document as well: prediction interval coverage degrades.
]

#v(1.0em)

= Introduction

In physical and energy systems, the cost of a forecast error is rarely spread evenly across time. In photovoltaic (PV) power generation, under clear skies or uniform overcast, the underlying generation curve is smooth, quasi-deterministic, and governed primarily by solar geometry. In these regimes, trivial persistence heuristics and simple autoregressive models are very hard to beat. The expensive failures, the ones that destabilize power grid balancing, incur imbalance penalties, and force fast-ramping reserve dispatch, occur during sudden ramp events, where power output drops or surges by a large fraction of installed capacity within minutes @sunset @fusionsf.

These ramp events expose a physical limit of unimodal time-series forecasting: nothing in a plant's historical power series announces an advecting cloud that has not yet reached the array. A cloud front approaching an installation at 40 km/h produces no signature in local numeric generation until the moment of optical occlusion. Exogenous numerical weather predictions (NWP) supply regional forecasts, but their spatial and temporal resolutions are too coarse to resolve localized cloud boundaries. Geostationary satellite imagery supplies the missing observation: optical and infrared sensors see cloud formation, deformation, and advection across the region hours before local irradiance is affected @seviri @solarcube. That is the premise behind modern multimodal nowcasting and vision--language solar models @crossvivit @solarvlm @pvvlm.

Showing that an architecture genuinely exploits visual information to resolve these ramps has proven harder than expected. Two obstacles run through the literature.

The first is a moving baseline. General-purpose time-series foundation models (TSFMs) pretrained on diverse multi-domain corpora have moved the empirical frontier @chronos2 @timesfm @ttm. Modern TSFMs tokenize numeric series via patching, capture long-range temporal dependencies through self-attention, consume numeric weather covariates natively, and generalize zero-shot across unseen geographies. In our benchmark, a fine-tuned unimodal foundation model on its own reaches a skill score of $0.5230$ against Smart Persistence on held-out plants, above nearly all previously published multimodal solar forecasters (@tbl-leaderboard-multimodal, @tbl-leaderboard-deep). A second modality paired with a foundation forecaster is no longer competing against a weak autoregressive baseline; it has to supply high-entropy spatial information that the existing numerical covariates do not already carry.

The second is an ablation confound. Existing multimodal forecasting architectures are almost universally trained end to end, and when authors ablate the visual branch to quantify its value, the ablated model is retrained from scratch without the visual parameters @solarvlm @timevlm @m3snet. Retraining conflates informational utility with architectural capacity, parameter count, and optimization dynamics. A degradation in accuracy on removing a visual branch is equally consistent with three different hypotheses: (i) the model extracted predictive physical signals from the images, (ii) the extra capacity of the visual encoder acted as a capacity reservoir, or (iii) the multimodal gradient path acted as an implicit regularizer. Standard retraining protocols cannot separate these mechanisms.

#figure(
  block(width: 100%)[
    #grid(
      columns: (1fr, 1fr),
      column-gutter: 0.8em,
      row-gutter: 0.5em,
      dbox[*Bottlenecked Late Fusion* (Conventional Paradigm)], hit[*Resampler-Free Interleaved Fusion* (Proposed Paradigm)],
      ad, ad,
      dbox[Spatiotemporal visual field compressed into a single global vector via learned-query cross-attention.], dbox[Visual field kept spatially resolved; 49 distinct cells reach the forecaster individually rather than as one summary.],
      ad, ad,
      dbox[Injected as an auxiliary parallel channel; identical summary broadcast across all forecast lead times.], dbox[Position-wise projected tokens placed directly into sequence self-attention alongside numeric patches.],
      ad, ad,
      dbox[High-frequency spatial boundaries averaged out; localized cloud fronts irreversibly blurred.], dbox[Temporal novelty selection dynamically filters static background, dedicating tokens to dynamic cloud edges.],
      ad, ad,
      dbox[*Empirical Result: Zero reliance* ($0.0000 plus.minus 0.0015$); scores identical with imagery absent.], hit[*Empirical Result: Significant reliance* ($0.0073 plus.minus 0.0002$); best ramp NMAE in 27-baseline suite.],
    )
  ],
  caption: [Conceptual taxonomy of multimodal fusion paradigms for time-series foundation models. Conventional architectures compress the visual field into a low-dimensional summary vector before decoding, creating an irreversible information bottleneck. The proposed resampler-free interleaved architecture preserves localized spatial entropy and integrates novelty-selected visual tokens directly into native sequence self-attention.],
) <fig-taxonomy>

We remove both confounds with a controlled experimental framework. We hold both foundational backbones fixed and hold training data, optimization schedules, and capacity constant across arms. The self-supervised video transformer is frozen outright and pre-computed offline. The time-series foundation model is fine-tuned once on the training plants in a unimodal stage, and then, identically in every multimodal arm, held fixed apart from its upper encoder blocks, which train at a reduced learning rate. Restricting the freely learnable parameters to the cross-modal interface leaves the geometric structure of the bridge as the single independent variable.

Under this framework, we ask two questions. First, how much does satellite imagery causally contribute to a foundation forecaster, independent of capacity and optimization confounds? Second, what representation geometry must the cross-modal bridge have to stop the foundation model from discarding visual information?

For the first question, we introduce a *counterfactual reliance protocol* on frozen weights. After training, the test set is scored twice on identical parameters: once with all inputs present, once with the visual pathway counterfactually disabled. The performance differential isolates the share of accuracy that depends on having observed the sky, denominated directly in ramp error. We pair this with falsification controls that corrupt visual freshness (stale sky) and spatial alignment (cross-plant donor sky), which test whether reliance is grounded in local physical phenomena or in superficial scene statistics.

For the second question, we contrast two representation paradigms against a unimodal foundation baseline (@fig-taxonomy). The information bottleneck paradigm compresses the visual field into a low-dimensional summary vector via a learned-query cross-attention resampler (Perceiver-style pooling) and broadcasts it as a parallel covariate channel. This widely adopted paradigm yields visual reliance indistinguishable from zero ($0.0000 plus.minus 0.0015$), because pooling destroys high-frequency spatial boundaries before the forecaster can query localized cloud features.

The resampler-free interleaved paradigm eliminates that bottleneck. A local $2 times 2$ average reduces the patch field to 49 spatially distinct cells without collapsing it to a global summary, a position-wise feed-forward network projects each cell, parameter-free temporal novelty selection prunes static background regions, and the surviving visual tokens are interleaved directly into the foundation forecaster's native sequence. Under native bidirectional self-attention, this restores visual reliance that clears the measurement floor by more than six times ($0.0073 plus.minus 0.0002$) and delivers the top ramp accuracy ($0.1435$) across a 27-baseline suite.

That arm bundles several design decisions, and reporting the bundle would repeat the ablation confound we set out to remove. We therefore ablate them separately at matched token count and matched sequence length (@sec-components), and the ablation changed the architecture we report. Our first design gave each cell a 4096-channel payload built by concatenating the four sub-patches it merges, on the premise that a cloud edge cutting diagonally through a cell has to stay recoverable downstream. Averaging those four sub-patches instead matches that design on ramp error, improves skill score by $0.0054$, and improves calibration with it, so the arm we report carries the cheaper cell and the sub-cell premise is withdrawn. What carries the effect is coarser and more portable than that premise: the forecaster receives the scene as many spatially distinct tokens instead of one pooled vector, and a novelty criterion decides which of them to spend the budget on.

The falsification controls also produce a mechanistic result we did not expect. Shuffling frames or collapsing micro-temporal indices has no measurable effect, so the model makes no use of visual frame order or micro-temporal positioning. It reads the spatially resolved visual tokens as an unordered, high-resolution spatial feature field that marks where localized cloud edges sit, and ignores their advective trajectory. That sharp ramp responsiveness carries an empirical calibration cost, reducing prediction interval coverage.

= Related Work and the Information Bottleneck Hypothesis <sec-related>

Multimodal time-series forecasting architectures differ in encoders, alignment objectives, and domain adaptations. From an information-theoretic perspective, though, they share one inductive bias: they summarize the visual representation into a low-dimensional bottleneck before the forecast horizon is resolved. We examine representative architectures below and contrast their structural choices with what physical ramp forecasting requires.

== Concatenative Joint Encoders: SUNSET
SUNSET @sunset is the classical convolutional joint encoding paradigm. The visual stream is ingested as a temporally stacked multi-frame tensor, processed through sequential convolutional and max-pooling blocks, flattened into a 1D vector, and concatenated directly with the flattened numerical history vector before entering a dense multilayer perceptron.

Global spatial pooling and flattening into a monolithic vector conflate localized spatial gradients irreversibly. The model cannot preserve the spatial relationship between a specific cloud boundary and the geographic location of the installation, so all forecast lead times condition on an identical, spatially degraded summary.

== History-Queried Cross-Attention: CrossViViT
CrossViViT @crossvivit brings spatio-temporal vision transformers to solar forecasting. The visual frames are tokenized with axial rotary embeddings, while the numerical power series is encoded via a separate temporal transformer. The two streams interact via cross-attention in which the queries originate from the historical time-series tokens, and the keys and values are image patch tokens.

CrossViViT avoids naive convolutional flattening, but its cross-attention is unidirectional and bounded by history: past generation steps query past visual patches. Fusion terminates before the temporal decoder operates. The representations propagating into the future horizon have therefore already undergone temporal condensation, which prevents future lead-time queries from probing uncompressed visual features.

== Gated Late Fusion of Bottlenecked Embeddings: Solar-VLM
Solar-VLM @solarvlm is representative of recent vision--language architectures @pvvlm @unicast. Satellite crops and historical meteorological text prompts are mapped through a frozen vision--language foundation model, compressed into a single pooled embedding per station via cross-attention pooling, concatenated with the temporal time-series state, and modulated via a learned scalar gate.

Compressing an entire $128 times 128$ km atmospheric scene into a single vector leaves the encoder representing global average cloudiness; fine-grained cloud edges do not survive. The learned gate frequently collapses toward the numeric branch, since the scalar summary adds little to what the numerical weather covariates already supply.

== Synthetic Imagery as Representation Regularizer: Time-VLM
Time-VLM @timevlm performs well in our benchmark suite (rank 2 overall, SS $0.5404$) while its second modality carries zero external physical information. The model plots the historical numeric power curve into a synthetic 2D image, pairs it with statistical text prompts, and embeds both using a vision--language model.

Time-VLM's visual branch re-encodes the numeric history through the representational prior of a pretrained vision backbone @visionts; it never observes the physical sky. The branch works as a geometric regularizer over 1D series patterns. This is one reason our counterfactual evaluation is necessary: a multimodal forecaster can reach superior benchmark accuracy through representation regularization alone, without perceiving the physical world.

== Forward-Aligned Exogenous Covariates: iTransformer
iTransformer @itransformer inverts the conventional transformer axes: each physical variable (target generation, solar zenith, cloud cover, temperature) is embedded as an independent token spanning the entire temporal window, while self-attention operates across variates. Under our evaluation protocol, iTransformer is supplied with numerical weather covariates shifted forward across the forecast horizon, representing operational NWP forecasts.

The difference from our setting is alignment. iTransformer reaches near-optimal ramp accuracy ($0.1445$) because its exogenous covariates are pre-aligned to the future horizon by construction: the cloud-cover token describes the atmospheric opacity of the exact future hour being predicted. Satellite imagery is an observation of the past and present. A multimodal architecture has no pre-aligned future data to lean on; it has to learn internally the spatial-to-temporal mapping from current cloud boundaries to future occlusion events.

== The Information Bottleneck Hypothesis <sec-shared-assumption>
As synthesized in @tbl-related-summary, prior multimodal architectures converge on one structural assumption: the visual field is compressed into a low-dimensional summary or pooled bottleneck before the forecast horizon is resolved. We hypothesize that this premature summarization is the architectural bottleneck that prevents foundation forecasters from using visual imagery. A cloud ramp is a localized, high-frequency spatial discontinuity. Compressing an auxiliary visual stream into a single vector eliminates spatial gradients and reduces the image to a redundant proxy for numerical cloud-cover covariates. Genuine visual reliance requires the visual field to enter the forecaster's sequence as many spatially distinct tokens rather than one summary, so that self-attention can query localized spatial features.

#tbl(
  [Structural comparison of multimodal forecasting paradigms. Prior methods condense the visual scene prior to future horizon resolution, eliminating the high-frequency spatial gradients required for ramp prediction.],
  columns: (auto, auto, 1.2fr, 1fr, auto),
  align: (left, left, left, left, center),
  table.header[Architecture][Visual Source][Fusion Mechanism][Horizon Resolution][Spatially Resolved?],
  [SUNSET @sunset], [Satellite / Sky], [Flattened concatenation + MLP], [Static vector broadcast], [No],
  [CrossViViT @crossvivit], [Satellite], [History-queried cross-attention], [Fused prior to temporal decoding], [No],
  [Solar-VLM @solarvlm], [Satellite + Text], [Learned-query pooled vector + gating], [Scalar gate on pooled state], [No],
  [Time-VLM @timevlm], [Synthetic plot], [Cross-attention on temporal query], [Pre-decoded pooled representation], [No],
  [iTransformer @itransformer], [NWP Covariates], [Cross-variate attention], [Forward-aligned by construction], [N/A],
  [*Proposed (S2d)*], [Satellite], [Resampler-free sequence interleaving], [Direct self-attention at horizon], [*Yes*],
) <tbl-related-summary>

= Multimodal Corpus and Physical Bounds <sec-dataset>

== Dataset Construction and Standardization
To evaluate cross-plant zero-shot transfer, we assemble a standardized multimodal corpus combining high-resolution photovoltaic generation, geostationary satellite imagery, and localized atmospheric covariates across two independent geographic tracks (@tbl-data-sources).

#tbl(
  [Data sources comprising the multimodal corpus. Continuous multi-band infrared satellite frames are co-registered per site for exact spatiotemporal synchronization with ground generation.],
  columns: (auto, 1fr),
  align: (left, left),
  table.header[Modality / Track][Source, Processing, and Properties],
  [PV Power (UK Track)], [Open Climate Fix `uk_pv` @ukpv: 30-minute generation across residential rooftop systems (2019--2020). Normalised strictly by audited installed capacity ($P / P_"cap" in [0, 1]$).],
  [Satellite (UK Track)], [EUMETSAT SEVIRI Rapid Scanning Service @seviri: Multi-band non-HRV infrared and visible channels. Reprojected per site and cropped to $128 times 128$ km spatial windows at 15-minute intervals.],
  [PV & Satellite (US Track)], [NREL PVDAQ industrial arrays @pvdaq paired with GOES-16 multi-band geostationary crops ($256 times 256$ km).],
  [Meteorological Covariates], [Open-Meteo historical archive @openmeteo: 8 surface variables joined by nearest coordinates, paired with analytical solar geometry and clear-sky irradiance @solarcube.],
) <tbl-data-sources>

== Physical Cloud Dynamics and Predictability Bounds <sec-data-novelty>
An open question in physical nowcasting is the temporal horizon over which optical cloud observations remain informative. To establish the physical upper bound on satellite nowcasting without model-induced artifacts, we analyze the spatiotemporal decorrelation of the SEVIRI multi-band imagery across all installations.

#figure(
  block(width: 90%)[
    #align(center)[
      #rect(width: 100%, inset: 1.2em, fill: luma(252), stroke: 0.5pt + luma(180))[
        *Atmospheric Decorrelation and Physical Limits of Cloud Advection*
        #v(0.4em)
        #grid(
          columns: (1fr, 1fr, 1fr),
          gutter: 1em,
          align: center,
          [#text(weight: 700)[$Delta t = 15$ min]\ Mean Absolute Diff: $8.3$ DN\ High spatial coherence],
          [#text(weight: 700)[$Delta t = 60$ min]\ Mean Absolute Diff: $22.7$ DN\ Cloud deformation emerging],
          [#text(weight: 700)[$Delta t = 120$ min]\ Mean Absolute Diff: $35.3$ DN\ Approaches spatial std ($38.6$ DN)],
        )
        #v(0.4em)
        #text(size: 8.5pt, fill: luma(90))[
          Temporal frame-to-frame difference decays into the background spatial noise floor by $approx 2$ hours. Optical motion tracking provides strong physical signals for immediate horizons ($t + 30$ to $t + 60$ min), beyond which cloud formation and dissipation dominate advection.
        ]
      ]
    ]
  ],
  caption: [Empirical measurement of geostationary satellite cloud decorrelation. Optical cloud motion information is physically bounded, decaying toward the spatial noise floor within two hours.],
) <fig-physics-bound>

As quantified in @fig-physics-bound, the mean absolute frame difference across consecutive 15-minute observations is $8.3$ Digital Numbers (DN). By $Delta t = 2$ hours, the difference reaches $35.3$ DN, converging toward the intrinsic intra-frame spatial standard deviation ($38.6$ DN). Optical advection is therefore informative for short-range horizons ($<= 2$ hours), beyond which cloud deformation and thermodynamic dissipation dominate over translational motion. A visual forecasting model has to capture localized spatial features quickly; long-range deterministic advection tracking is not available to it.

== Continuous Multi-Band Infrared Coverage
A common failure mode in optical solar forecasting is nocturnal data blindness. Models relying on visible-spectrum imagery (such as high-resolution visible HRV channels) receive completely blank frames before sunrise and during winter dawns. In our corpus, multi-band infrared channels keep active thermal contrast through the diurnal cycle: December pre-dawn frames have mean DN $134 plus.minus 37$, comparable to midday frames ($123 plus.minus 38$). Over $96.1%$ of evaluated test sequences have complete 8-frame satellite sequences, so foundation models can observe approaching cloud decks long before dawn generation begins.

= Experimental Protocol and Metrics <sec-protocol>

== Cross-Plant Disjoint Evaluation
To evaluate foundation-model transfer, we enforce a strict *cross-plant split*. The 98 quality-controlled UK installations are partitioned into 70 training, 14 validation, and 14 test plants. The partition is disjoint by installation: train, validation, and test sets span the identical two-year calendar period (2019--2020) across completely non-overlapping rooftop installations. This forces the model to generalize across microclimates, array orientations, and local shading profiles, and rules out the spatial memorization available in standard chronological splits.

== Standardized Task and Metrics
All models operate on a standardized input--output window:
- *History Window*: 14 days (672 steps at 30-minute sampling) of capacity-normalized historical power $y_t in [0, 1]$, combined with 14 atmospheric and solar covariates.
- *Visual Window*: 8 satellite frames covering the 6 hours immediately preceding the forecast origin, centered over the installation.
- *Forecast Horizon*: 6 hours ahead (12 lead times), predicting 9 non-parametric quantiles ($q in {0.1, dots, 0.9}$) per lead time to quantify predictive uncertainty.

Evaluation is conducted strictly on daytime generation intervals ($165,295$ scored steps across 14 test plants). We evaluate two primary metrics:
1. *Generalization Skill Score ($SS$)*: Denominated against Smart Persistence ($SS = 0$):
   $ SS = 1 - frac("NRMSE"_"model", "NRMSE"_"persistence") $
   where higher is better, and $1.0$ denotes perfect forecasting.
2. *Ramp Normalized Mean Absolute Error ($"NMAE"_"ramp"$)*: Evaluated exclusively on steps exhibiting high volatility, defined as the top decile of true generation changes between consecutive sampling intervals ($|y_t - y_{t-1}|$). This isolates the physical regime where cloud edges induce operational penalties.

#note[
  *Pre-Registered Significance Floors*: Across 3-seed evaluation ($42, 43, 44$), the empirical noise floor is *0.0011 for Ramp NMAE* and *0.0037 for Skill Score*. Any performance differential below these thresholds is statistical variation.
]

= Method: Controlled Foundation Model Fusion

== Foundation Backbones

The architecture integrates two pretrained foundation models, both held fixed during multimodal training, though not to the same degree:

#tbl(
  [Component stack of the multimodal architecture. The visual backbone is frozen outright; the forecaster is fixed apart from its upper encoder blocks, which train identically in every arm and which @sec-components shows are not load-bearing. This isolates the bridge geometry as the single independent variable.],
  columns: (auto, 1.2fr, 1fr),
  align: (left, left, left),
  table.header[Component][Architecture & Latent Dimensions][Optimization State],
  [Time-Series Backbone], [Chronos-2 transformer @chronos2: 12 blocks, hidden dimension 768, 12 attention heads. Patch size 16.], [Fine-tuned end to end on the training plants in the unimodal stage; thereafter fixed except the upper 3 of 12 encoder blocks at $0.1 times$ learning rate (identical across all arms)],
  [Visual Backbone], [V-JEPA 2.1 ViT-L/16 @vjepa21: Self-supervised spatiotemporal video encoder. 8 input frames $arrow.r$ 4 latent temporal slices $times$ 196 spatial patches $times$ 1024 channels.], [Frozen entirely; pre-computed offline to prevent gradient leakage],
  [Cross-Modal Bridge], [Dimension alignment and spatial/temporal fusion interface.], [Trained from scratch],
) <tbl-components>

=== The Time-Series Foundation Forecaster
The time-series backbone is an encoder-only transformer pretrained on heterogeneous cross-domain time-series datasets @chronos2. Historical power is tokenized via non-overlapping 16-sample patches, mapping a 672-step history into 42 macro-context tokens. Known future weather covariates enter as parallel token sequences. Bidirectional self-attention operates across all positions, and a multi-quantile linear projection head maps future hidden states into 9 output quantiles across the 12-step horizon in a single non-autoregressive forward pass.

#figure(
  image("figures/mmtsfm_s1.svg", width: 90%),
  caption: [Unimodal Foundation Baseline (S1): The time-series transformer operates solely on numerical generation history and weather covariates without visual input.],
) <fig-s1>


=== The Self-Supervised Video Encoder
We use V-JEPA 2.1 @vjepa21 instead of a vision--language encoder trained on text-image alignment (e.g., CLIP). Text-aligned encoders optimize for static object semantics and discard fine-grained cloud deformation. V-JEPA is pretrained via latent feature prediction across masked spatiotemporal tubelets, which gives it an inductive prior over how physical scenes evolve and deform across time. The encoder processes the 8-frame satellite clip into a latent tensor $bold(Z)_v in bb(R)^(4 times 196 times 1024)$, representing 4 temporal slices of $14 times 14$ spatial patches.

== The Fusion Bridges

We formulate two contrasting architectural paradigms for connecting the visual latent field $bold(Z)_v$ to the foundation forecaster:

=== Paradigm 1: Bottlenecked Late Fusion (S2a)
In this conventional paradigm (@fig-s2a), the visual field undergoes learned-query cross-attention pooling. A set of learned query vectors attends over the $4 times 196$ visual tokens, compressing the entire spatiotemporal scene into a single summary vector $bold(v) in bb(R)^(768)$. This vector is projected through a linear adapter and injected into the transformer's batch axis as an auxiliary covariate row. Group self-attention fuses the numeric target sequence with this visual summary at each layer.

The summary is intuitive as a macro-regime indicator, distinguishing overcast from clear days, but compressing 784 spatial patches into a single vector is an irreversible information bottleneck. The high-frequency spatial gradients, the localized boundaries of advancing clouds, are averaged out. Once the bottleneck resampler has discarded them, downstream cross-attention layers cannot reconstruct the spatial coordinates of approaching fronts.

#figure(
  image("figures/mmtsfm_s2a.svg", width: 90%),
  caption: [Bottlenecked Late Fusion (S2a): The satellite clip is compressed via a learned-query resampler into a single global summary vector, injected as an auxiliary parallel channel.],
) <fig-s2a>
=== Paradigm 2: Resampler-Free Interleaved Sequence Fusion (S2d)
The proposed architecture (@fig-s2d) removes the learned-query resampler entirely and replaces it with a position-wise projector, so the visual path reduces to encoder, projection, forecaster. The shape is borrowed from the encoder-projector-decoder design of Nemotron 3 Nano Omni @nemotron, in which modality encoders reach the decoder through MLP projectors with no compressor in between, a pixel shuffle reduces the token count before projection, and redundant video tokens are pruned by Efficient Video Sampling @evs. Those three elements transfer to our setting, one of them with a modification we arrived at empirically, and we describe each below. We adopt the projector-only shape for the reason established in @sec-shared-assumption: every arm that routes the patch field through a pooling resampler sits at the ramp measurement floor, so the compressor itself is the component under suspicion. Removing it raises a practical problem the rest of this section addresses. The V-JEPA latent field holds $4 times 196 = 784$ tokens, which is an order of magnitude more than the 42 numeric macro-patches the forecaster consumes, and passing all of them into self-attention is neither affordable nor useful when most of a satellite crop is static sky.

The first reduction cuts the token count fourfold, at the place and rate Nemotron's pixel shuffle occupies @nemotron. We partition the $14 times 14$ patch grid into non-overlapping $2 times 2$ neighborhoods and average each neighborhood, which leaves a $7 times 7$ field of 49 cells at 1024 channels. Nemotron instead concatenates the four sub-patches of a cell along the channel axis ($14 times 14 times 1024 arrow.r 7 times 7 times 4096$), a space-to-depth pixel shuffle that discards no value: each sub-patch stays individually addressable in its own channel range, so a cloud edge cutting diagonally through a cell remains recoverable downstream. We built the arm that way first, on the premise that ramp prediction needs that sub-cell structure. It does not. At matched cell count, cell embedding, token budget, positions and sequence length, the averaging variant matches the concatenating one on ramp error, beats it on skill score, and calibrates better (@sec-components), so we report the averaging variant and treat the sub-cell premise as falsified. What the step does in either form is hold 49 spatially distinct cells where a resampler would leave one vector.

A two-layer MLP with a GELU nonlinearity then maps each cell's channels into the forecaster's width ($1024 arrow.r 768$). The projector is applied independently at each of the 49 spatial cells and never mixes across them. Any spatial mixing that happens before the backbone is mixing the backbone cannot undo, and the whole point of the arm is to let the foundation model's own self-attention decide which cells matter for a given forecast. A learned embedding table of 49 vectors, indexed by cell coordinate, is added after projection so that each token carries its position in the field.

That table is deliberately spatial only. We add no temporal-slice embedding, because the frame time of each token is already encoded by its sequence position, and a second, independent encoding of the same axis would weaken the frame-order control in @sec-instrument: shuffling the latent slices permutes positions, but a slice embedding travelling with the token could leak the original ordering back in. Keeping frame identity in exactly one place is what makes the null result in Control 1 interpretable.

Pruning happens next, and it is parameter-free. The rule is Efficient Video Sampling @evs, which Nemotron @nemotron uses on its video tokens, and we take it unchanged including its operating point. For every spatial cell we score each frame by the cosine dissimilarity between that cell's token and the same cell one frame earlier, which is high where the scene has changed and near zero where it has not. The first frame is pinned at $+infinity$, so the anchor field survives intact and the model always receives one complete view of the sky rather than a set of difference tokens. Scores are ranked globally across all $4 times 49 = 196$ candidates, the top 98 are retained at a pruning rate of $q = 0.5$, and the surviving indices are re-sorted so the sequence stays in frame-then-cell order at a fixed length. The one substantive change is when the rule runs. In Nemotron, EVS is a runtime knob applied after training to buy throughput at some cost in accuracy, and $q$ is tuned per deployment. Here it is inside the training loop as well, so the 98-token budget is the budget the forecaster learns to read, and the vision-off counterfactual pass of @sec-instrument stays inside the distribution the weights were fitted on. That also turns $q$ from a latency dial into an architectural constant, which lets us test the selection criterion itself: half the budget is spent on the static anchor and half on whichever cells moved, which is the behavior we want when a single advancing cloud edge occupies a small fraction of a $128 times 128$ km crop. Because the criterion has no parameters, the rule can be swapped on a trained checkpoint, so the random-selection control in @tbl-controls is a clean test of the rule rather than of sequence length.

Placing the surviving tokens is complicated by patch granularity. At an input patch size of 16 on 30-minute data, one numeric token spans 8 hours, so the entire 6-hour visual window falls inside the final context patch and there is no integer sequence position left to interleave at. We therefore give visual tokens fractional positions. A token from a frame $Delta t$ seconds before the forecast origin is placed at $T_M + "clamp"(1 - Delta t \/ S, 0, 1) dot 0.99$, where $T_M$ is the index of the co-temporal numeric token and $S$ is the span of one patch in seconds. Visual tokens then occupy the interval $[T_M, T_M + 0.99]$, ordered oldest to newest, after the numeric token they belong to and strictly before the first future position at $T_M + 1$. The upper bound of $0.99$ rather than $1.0$ keeps a frame with $Delta t = 0$ from colliding with the forecast query. Rotary position embeddings take positions as floats, so this needs no change to the attention kernel.

The assembled sequence holds the 42 numeric macro-patches, the 98 visual tokens, and the future horizon query, and all 141 positions are processed jointly under the backbone's native bidirectional self-attention. This is where the arm departs furthest from Nemotron, which concatenates its projected visual tokens into a prefix block ahead of the text stream. Our visual tokens instead sit at their own timestamps inside the numeric sequence, on the reasoning that the numeric axis here is a physical clock and a satellite frame has a place on it. We present that placement as a design choice and attach no claim to it; the contribution established below rests on the removal of the compressor and on the novelty criterion. What the placement does not touch is the absence of a compressor: no cross-attention layer, gate, or auxiliary channel sits between the modalities. A forecast query attends to a visual token under the same mechanism it uses to attend to a numeric patch, and because the visual window lies inside the receptive field of the immediate pre-forecast patch, those tokens describe the sky as it was just before the horizon opens.

During training we drop the visual stream on half of the samples and the numeric stream on a tenth of them, never both for the same sample. The asymmetric rates serve the measurement rather than regularization: the counterfactual protocol of @sec-instrument scores the frozen model a second time with the visual pathway switched off, and that forward pass has to be one the model has seen often during training. Otherwise the measured drop would mix the loss of visual information with a distribution shift the model was never exposed to, which is the confound the protocol exists to remove.



#figure(
  image("figures/mmtsfm_s2d.png", width: 95%),
  caption: [Resampler-Free Interleaved Fusion (S2d): Visual patches are averaged into a $7 times 7$ field of cells, projected position-wise by an MLP, and pruned by temporal novelty. The surviving 98 tokens enter the transformer's native sequence directly, so bidirectional self-attention operates between numeric series and spatially resolved cloud features.],
) <fig-s2d>

= The Counterfactual Reliance Instrument <sec-instrument>

To quantify visual utility without the confounds of model capacity or retraining dynamics, we define the *Counterfactual Reliance Metric*:

$ "Reliance"_"ramp" = "NMAE"_"ramp"^(bold(V)=emptyset) - "NMAE"_"ramp"^(bold(V)="active") $

Following training, the model parameters $bold(theta)$ are frozen. The test set is scored under the standard multimodal forward pass ($bold(V)="active"$). The identical frozen model is then scored with the visual pathway counterfactually ablated ($bold(V)=emptyset$). Because all model weights, learning rates, and random seeds stay identical, a positive reliance means the model's accuracy causally depends on having perceived the imagery.

== Falsification Diagnostic Controls
To determine whether measured reliance comes from genuine physical perception or from spurious visual artifacts, we pair the reliance metric with five diagnostic controls:
1. Shuffling the temporal sequence of visual latent slices. If the model relies on tracking cloud advection vectors, temporal scrambling must degrade accuracy.
2. Collapsing the sub-patch temporal positions of visual tokens into a single shared index, which tests whether micro-temporal granularity carries signal.
3. Lagging the visual clip by one forecast horizon, so generation at $t$ is evaluated using the sky from $t - 6$ hours. This tests whether the model requires contemporaneous weather or exploits generic diurnal brightness patterns.
4. Swapping the contemporaneous visual clip with imagery from a geographically distant installation, which tests whether the model resolves local array-level cloud cover or only regional solar geometry.
5. Replacing the temporal novelty criterion with uniform random token selection at the identical budget, which tests the selection rule rather than the sequence length it produces.

Controls 1--4 corrupt the input of a fixed trained model, and control 5 swaps a parameter-free rule on a trained checkpoint, so all five are evaluation-time interventions on frozen weights. They can therefore establish whether reliance is physically grounded, but not which architectural decision produced it. That second question needs retrained arms, and @sec-components supplies them.

= Results and Analysis

== The Fusion Ladder: Comparing Visual Reliance

@tbl-ladder reports the primary empirical findings across the fusion paradigms.

#tbl(
  [Empirical evaluation of visual reliance across fusion paradigms. Reliance is the reduction in ramp error from having observed the sky, measured counterfactually on frozen weights across seeds 42--44. Bottlenecked pooling yields zero reliance, whereas the resampler-free arm clears the significance floor by more than six times.],
  columns: (auto, 1.2fr, auto, auto, auto),
  align: (left, left, center, center, center),
  table.header[Arm][Architectural Paradigm][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)][Ramp Reliance ($arrow.t$)],
  [S1], [Unimodal Foundation Baseline], [0.5230 #text(size: 8pt)[$plus.minus 0.0041$]], [0.1506 #text(size: 8pt)[$plus.minus 0.0010$]], [---],
  [S2a], [Bottlenecked Late Fusion (Resampler)], [0.5258 #text(size: 8pt)[$plus.minus 0.0043$]], [0.1487 #text(size: 8pt)[$plus.minus 0.0010$]], [0.0000 #text(size: 8pt)[$plus.minus 0.0015$]],
  [*S2d*], [*Resampler-Free Interleaved Fusion*], [*0.5564* #text(size: 8pt)[$plus.minus 0.0018$]], [*0.1435* #text(size: 8pt)[$plus.minus 0.0004$]], [*0.0073* #text(size: 8pt)[$plus.minus 0.0002$]],
) <tbl-ladder>

S2a and S2d separate cleanly. In S2a, nominal ramp error improves slightly over S1 during training ($0.1506 arrow.r 0.1487$), while counterfactual reliance is exactly $0.0000$. Forcing the visual pathway off on frozen weights produces no performance drop at all. S2a's nominal gain over S1 comes from the optimization dynamics of multimodal training; the model stops reading the pooled vector because the resampler has destroyed its spatial utility.

S2d reaches a ramp reliance of $0.0073$, clearing the pre-registered significance floor ($0.0011$) by more than six times across all seeds. Delivering the patch field as 49 separate cells lets the foundation transformer query localized visual boundaries directly, and gives the top ramp accuracy across the suite. The S2a--S2d comparison alone cannot say which of the arm's several differences produces this, since the two arms differ in resampler, token count and placement at once; @sec-components separates them.

The proposed transition from S2d to S2e extends this single-anchor design across the numerical context. Five visual anchors are placed beside Chronos-2 patches at the same clock time on consecutive days, giving the forecaster repeated sky--power pairs at matched solar geometry. Each anchored patch retains S2d's 98 visual tokens, for 490 visual tokens in total, rather than compressing each anchor to 20. A46 tests whether that compression explains the first S2e result while holding the five anchors and 100-token total budget fixed: paired-novelty selection ranks the 49 two-latent spatial trajectories within each anchor and retains both endpoints of the ten that change most. Recovery would identify token allocation as the failure mode, but would not isolate interleaving itself.

== Mechanistic Diagnostics and Falsification Controls

@tbl-controls reports the outcomes of the diagnostic controls. All five were run on the concatenating-payload variant of the arm (S2d-cat in @tbl-components-ablation), which differs from the reported S2d only in the per-cell payload and matches it on both primary metrics; the reference figures the controls are measured against are therefore that variant's ($0.5510$ skill score, $0.1440$ ramp NMAE, $0.0063$ reliance). The results divide cleanly into confirmed content grounding and refuted kinematic tracking.

#tbl(
  [Diagnostic falsification controls on the resampler-free interleaved arm, evaluated across 14 held-out test installations on the concatenating-payload variant (see text). Values are seed means over seeds 42--44, except control 5, which is a two-seed evaluation (42--43).],
  columns: (auto, 1.2fr, 1.2fr),
  align: (center, left, left),
  table.header[\#][Diagnostic Hypothesis Tested][Empirical Finding and Mechanistic Verdict],
  [1], [Does S2d rely on temporal frame ordering? (Kinematic frame shuffle)],
  [Near-inert: $Delta "NMAE"_"ramp" approx 0$ across all seeds ($< 0.0003$). Refutes the hypothesis that the model tracks kinematic cloud trajectories.],
  [2], [Does micro-temporal sub-patch positioning matter? (Collapsing sub-patch indices)],
  [Null effect: Ramp NMAE change ($+0.0008$) falls below the $0.0011$ significance floor; reliance is unaffected ($0.0064$ vs $0.0063$). Temporal indexing within the patch is invariant.],
  [3], [Is a recent sky sufficient? (Stale sky: imagery lagged by 6 hours)],
  [Severely degrades accuracy: Costs $+0.0169$ Ramp NMAE ($+11.7%$ error); marginal gain flips hard negative (stale imagery is worse than no imagery). Confirms strict temporal freshness.],
  [4], [Does the model resolve site-specific sky? (Contemporaneous cross-plant donor sky)],
  [Catastrophic degradation: Costs $+0.0176$ Ramp NMAE; Skill Score collapses from $0.5510$ to $0.3700$. Confirms local spatial grounding over the installation.],
  [5], [Does dynamic novelty pruning carry informational signal? (Uniform random token selection at equal budget)],
  [Degrades performance: Ramp NMAE worsens by $+0.0039$ ($3.6times$ significance floor); Skill Score drops by $-0.0152$ ($4.1times$ floor), over two seeds. Confirms that novelty scoring extracts predictive cloud boundaries.],
) <tbl-controls>

=== Content Grounding
Controls 3 and 4 pin down physical grounding. Supplying a stale sky from 6 hours prior, or a contemporaneous sky from an installation 200 km away, causes catastrophic accuracy drops ($> 15 times$ the significance floor) and flips the visual marginal gain sharply negative. S2d is therefore not exploiting static spatial priors or diurnal brightness heuristics; it depends causally on contemporaneous atmospheric phenomena directly above the target installation.

=== Spatial Entropy over Kinematics
Controls 1 and 2 point the other way. Shuffling the temporal sequence of visual latent slices or collapsing their fractional temporal coordinates has near-zero impact on ramp error. The foundation forecaster does not track directional advective vectors across consecutive frames. Because the satellite window falls within the receptive field of the immediate pre-forecast context token, the model treats the visual tokens as an unordered localized spatial feature field. Its predictive signal is the presence of a sharp, non-stationary cloud boundary near the array; the boundary's minute-by-minute motion goes unused.

=== Validation of Novelty Selection
Control 5 isolates the utility of temporal novelty selection. Replacing the inter-frame cosine dissimilarity criterion with uniform random token selection at the identical 98-token budget degrades Ramp NMAE by $0.0039$ ($3.6times$ the seed floor) and drops Skill Score by $0.0152$ ($4.1times$ the floor) across the two seeds evaluated. Since the budget is held constant, the gain cannot be attributed to sequence length: novelty pruning identifies and keeps high-entropy cloud transitions while filtering uninformative background sky.

== Component Ablations: Which Design Decisions Carry the Gain <sec-components>

The controls above corrupt the visual input of a fixed architecture. They establish that the measured reliance is physically grounded, but they cannot say which part of the architecture produces it. S2d differs from S2a in several ways at once, and the fusion ladder credits the bundle. We therefore vary each decision separately against the arm, one configuration key at a time, at matched seeds (42--44), matched token count, and matched sequence length. Deltas are paired per seed and judged against the pre-registered floors ($0.0037$ skill score, $0.0011$ ramp NMAE). The decision that moved is the per-cell payload (@tbl-components-ablation).

#tbl(
  [Per-cell payload ablation. Both arms hold cell count (49), cell embedding, novelty budget (98 tokens), fractional positions and sequence length identical, and are retrained on seeds 42--44 from the identical unimodal checkpoint. Deltas are paired per seed against the reported arm. Averaging the $2 times 2$ block is what we report; concatenating it was the original design.],
  columns: (auto, 1.3fr, auto, auto, auto),
  align: (left, left, center, center, center),
  table.header[Arm][Per-Cell Payload][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)][Ramp Reliance ($arrow.t$)],
  [*S2d*], [Mean pool over each $2 times 2$ block ($1024$ channels)], [*0.5564*], [*0.1435*], [*0.0073*],
  [S2d-cat], [Space-to-depth concatenation of the same block ($4096$ channels)], [0.5510 #text(size: 8pt)[($-0.0054$)]], [0.1440 #text(size: 8pt)[($+0.0005$)]], [0.0063 #text(size: 8pt)[($-0.0009$)]],
) <tbl-components-ablation>

=== Sub-Cell Spatial Detail Is Inert, and Costs Accuracy
The two arms differ only in how the four sub-patches of a cell are carried into the projector: averaged into 1024 channels, or concatenated into 4096. The pre-registered reads for this ablation were "averaging is equivalent" or "averaging is worse". Neither occurred. Averaging is equivalent on both ramp metrics and better on skill score by $0.0054$, clearing the $0.0037$ floor on all three seeds individually with no sign flip, and it improves calibration in the same direction (@tbl-calibration).

The founding claim of the original design, that the four merged patches must be concatenated because averaging destroys the sub-cell structure ramp prediction needs, is therefore withdrawn. Sub-cell spatial detail carries nothing measurable here, and paying 4096 projector input channels for it costs aggregate accuracy and calibration for no ramp return. What survives is the coarse effect of the step: 49 spatially distinct cells rather than one pooled vector. We report the cheaper arm and keep the concatenating variant on record, because the pair is what licenses the claim, and because the diagnostic controls of @tbl-controls were measured on that variant.

=== What Remains
Two properties separate the reported arm from the bottlenecked one, and both are load-bearing. The first is the removal of the pooling resampler, which the S2a--S2d gap in @tbl-ladder measures at $+0.0306$ skill score and $-0.0052$ ramp NMAE. The second is the temporal novelty criterion, which control 5 in @tbl-controls measures at $0.0039$ ramp NMAE and $0.0152$ skill score against random selection at an identical budget. The payload ablation removes a third candidate: whatever sub-cell resolution contributes, it is not the mechanism, and the cheaper cell wins on both aggregate accuracy and calibration.

The contribution that survives this accounting is coarser than the one we set out to demonstrate, and more portable for it. Deliver the scene to a foundation forecaster as many spatially distinct tokens rather than one summary, and spend a fixed token budget on the cells that changed. Neither half depends on the width of the cell payload, so neither is tied to this backbone's patch granularity.

== The Intrinsic Calibration Cost

S2d gains point-forecast ramp accuracy, and its probabilistic output carries a matching cost:

#tbl(
  [Probabilistic calibration metrics across foundation model configurations. S2d gains ramp accuracy at the expense of empirical coverage and quantile sharpness. The concatenating variant of @tbl-components-ablation is included because the payload moves calibration well beyond what it moves on the primary metrics.],
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  table.header[Configuration][Ramp NMAE ($arrow.b$)][80% Prediction Interval Coverage][Quantile ECE ($arrow.b$)],
  [S1 (Unimodal Control)], [0.1506], [0.772], [0.0297],
  [*S2d (Resampler-Free)*], [*0.1435*], [0.753], [0.0314],
  [S2d-cat (concatenated cells)], [0.1440], [0.731], [0.0359],
  [Nominal Target], [---], [0.800], [0.0000],
) <tbl-calibration>

As documented in @tbl-calibration, S2d's empirical coverage for the nominal 80% prediction interval degrades from $0.772$ in S1 to $0.753$, accompanied by an increase in expected calibration error (ECE) from $0.0297$ to $0.0314$. The continuous ranked probability score moves the other way, improving from $0.0574$ to $0.0536$, so the loss is specific to interval width and quantile placement rather than to the distributional forecast as a whole.

Conditioned on high-resolution spatial cloud boundaries, the model issues confident, sharp forecast drops that anticipate impending ramps. The exact arrival time of a cloud edge across an array carries irreducible micro-scale stochasticity that geostationary satellite resolution cannot pin down. Sharp, deterministic ramp trajectories that are slightly misaligned in time incur severe quantile calibration penalties. High-accuracy multimodal forecasting trades probabilistic conservatism for deterministic ramp responsiveness.

The payload ablation qualifies how intrinsic that trade is. The concatenating variant sits at $0.731$ coverage and $0.0359$ ECE at the same ramp accuracy, so more than half of the coverage gap and roughly three quarters of the ECE gap that our first arm opened against S1 were properties of a 4096-channel cell rather than of the paradigm. The conservatism against responsiveness trade is real, but the operating point is a design variable, and a cheaper cell sits closer to nominal.

= Comprehensive Benchmark Leaderboard

To contextualize these findings, we evaluate the proposed models against 27 re-implemented baselines across three tiers, evaluated under the identical cross-plant protocol on the same 14 test installations (@tbl-leaderboard-multimodal, @tbl-leaderboard-deep, @tbl-leaderboard-baselines).

#tbl(
  [Multimodal solar forecasting leaderboard on 14 held-out test installations. S2d achieves the top rank across all multimodal architectures.],
  columns: (auto, auto, 1fr, auto, auto),
  align: (center, left, left, center, center),
  table.header[Rank][Model][Cross-Modal Fusion Strategy][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)],
  [1], [*MMTSFM S2d (Ours)*], [*Resampler-free sequence interleaving in token space*], [*0.5564*], [*0.1435*],
  [2], [Time-VLM @timevlm], [Synthetic series plot embedded via frozen VLM], [0.5404], [---],
  [3], [MMTSFM S2a (Ours)], [Bottlenecked Perceiver resampler on batch axis], [0.5258], [0.1487],
  [5], [MMTSFM S1 (Ours)], [Unimodal foundation control (no visual stream)], [0.5230], [0.1506],
  [11], [Solar-VLM @solarvlm], [Gated Perceiver pooling of satellite + text prompts], [0.4396], [0.1514],
  [19], [CrossViViT @crossvivit], [Cross-attention from historical station steps], [0.3491], [---],
  [24], [Aurora @aurora], [Joint multimodal foundational pretraining], [0.2324], [---],
  [25], [SUNSET @sunset], [Convolutional joint encoding + feature concatenation], [0.2162], [---],
  [26], [UniCast @unicast], [Prompted multimodal foundation forecaster], [0.1211], [---],
  [28], [VisionTS++ @visionts], [Continual visual pretraining on synthetic plots], [0.0167], [---],
) <tbl-leaderboard-multimodal>

#tbl(
  [Supervised deep learning and foundation model baselines. S2d statistically ties iTransformer's top ramp performance, with measurable reliance on past imagery.],
  columns: (auto, auto, 1fr, auto, auto),
  align: (center, left, left, center, center),
  table.header[Rank][Model][Architectural Paradigm][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)],
  [4], [iTransformer + Covs @itransformer], [Inverted variate tokens with forward-shifted NWP covariates], [0.5257], [0.1445],
  [6], [Chronos-2 Fine-Tuned @chronos2], [Unimodal transformer with native covariate rows], [0.5042], [0.1494],
  [7], [TS-RAG @tsrag], [Retrieval-augmented foundation model on numeric history], [0.4779], [---],
  [8], [Cross-RAG @crossrag], [Cross-attention retrieval on historical series], [0.4768], [---],
  [9], [Chronos-2 Zero-Shot @chronos2], [Direct zero-shot foundation model transfer], [0.4737], [0.1544],
  [10], [PatchTST @patchtst], [Channel-independent patched transformer], [0.4581], [0.1543],
  [12], [Temporal Fusion Transformer @tft], [Gated residual network with temporal self-attention], [0.4264], [0.1605],
  [13], [MLP Baseline], [Dense multi-layer perceptron over flattened history], [0.4219], [0.1624],
  [16], [CoRA @cora], [Covariate-aware adapter on frozen foundation backbone], [0.3798], [0.1624],
  [18], [TTM Fine-Tuned @ttm], [Tiny Time Mixer with channel mixing], [0.3601], [0.1716],
  [20], [DLinear @dlinear], [Decomposed trend-seasonal linear model], [0.3231], [0.1746],
  [21], [TiRex Zero-Shot @tirex], [Pretrained zero-shot time-series model], [0.2873], [0.1826],
  [22], [TimesFM Zero-Shot @timesfm], [Decoder-only zero-shot foundation model], [0.2708], [0.1902],
  [30], [TTM Zero-Shot @ttm], [Pretrained zero-shot Tiny Time Mixer], [-0.0807], [0.2922],
) <tbl-leaderboard-deep>

#tbl(
  [Tabular, classical machine learning, and statistical reference baselines.],
  columns: (auto, auto, 1fr, auto, auto),
  align: (center, left, left, center, center),
  table.header[Rank][Model][Model Type][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)],
  [14], [TabPFN @tabpfn], [Prior-data fitted network for tabular classification/regression], [0.4063], [0.1631],
  [15], [LightGBM], [Gradient-boosted decision trees over lagged tabular features], [0.3854], [0.1672],
  [17], [TabFM Ensemble], [Ensemble of tabular foundation models], [0.3626], [0.1573],
  [23], [Hourly Climatology], [Historical hourly mean per installation], [0.2337], [0.1665],
  [27], [Seasonal Naive], [Persistence of generation from 24 hours prior], [0.1068], [0.2056],
  [29], [Persistence], [Persistence of generation from immediate prior step], [0.0141], [0.2550],
) <tbl-leaderboard-baselines>

== Benchmark Insights

Two observations follow from the leaderboard. The first concerns the height of the unimodal bar. The fine-tuned unimodal control (S1, rank 5, SS $0.5230$) outperforms every published multimodal architecture in the literature except Time-VLM, which uses synthetic plots rather than physical imagery. Published satellite forecasters such as Solar-VLM (rank 11, SS $0.4396$), CrossViViT (rank 19, SS $0.3491$), and SUNSET (rank 25, SS $0.2162$) fall behind strong unimodal transformers. Attaching a visual stream to a modern foundation model buys nothing unless the fusion bridge preserves high-entropy spatial information.

The second concerns ramp accuracy. S2d has the lowest ramp error ($0.1435$) across all 30 evaluated systems. It edges iTransformer ($0.1445$) by $0.0010$, which is inside the $0.0011$ noise floor, so the two are a statistical tie. iTransformer gets its ramp accuracy because its numerical weather covariates are forward-aligned onto the forecast horizon by construction. S2d matches it from historical satellite observations alone, so a spatially resolved token set placed in the transformer's own sequence is enough for the model to perform the spatial-to-temporal projection internally and anticipate future cloud occlusions.

= Discussion and Conclusion

We examine how time-series foundation models interact with dense perceptual modalities. Eliminating retraining confounds and measuring counterfactual reliance directly on frozen weights supports four conclusions:

1. The visual bottleneck is structural. Conventional learned-query resamplers collapse spatial cloud boundaries into a single vector and destroy the high-frequency spatial gradients that ramp prediction needs. The result is zero causal reliance: the model performs identically with imagery absent. Removing the resampler, so that the scene reaches the forecaster as 49 spatially distinct novelty-selected cells rather than one summary, restores significant visual reliance ($0.0073$) and delivers best-in-suite ramp accuracy ($0.1435$).
2. The token set carries the mechanism, and it is cheap. A matched-budget ablation falsifies the design premise that each cell must ship its four sub-patches concatenated. Averaging them instead leaves ramp error unchanged, improves skill score by $0.0054$, and improves calibration with it, so the arm we report carries a quarter of the projector input width we first specified. What is load-bearing is the absence of pooling and the novelty criterion that chooses which cells to keep, neither of which depends on the width of the payload or on this backbone's patch granularity.
3. Spatial presence dominates kinematics. The diagnostic controls overturn our design hypothesis that the forecaster models directional cloud kinematics. Shuffling frame order or collapsing sub-patch temporal indices produces null effects. The foundation model treats the visual stream as an unordered, high-resolution spatial feature field marking the immediate presence of localized cloud boundaries, and does not track advective trajectories.
4. Sharpness costs calibration, but not irreducibly. Conditioned on sharp spatial cloud features, the model issues confident predictions during volatile ramp periods, and these sharp transitions degrade prediction interval coverage and quantile calibration. The concatenating variant we first built lost more than twice the coverage at the same ramp accuracy, so most of the degradation we originally measured was a property of that cell rather than of the paradigm. Future work should look at uncertainty-aware multimodal gating to preserve probabilistic calibration while keeping the deterministic ramp warnings.

#v(0.8em)
#bibliography("refs.bib", title: "References", style: "ieee")
