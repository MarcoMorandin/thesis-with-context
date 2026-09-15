#set document(
  title: "Interleaved Vision Fusion for Time-Series Foundation Models",
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
  *Abstract.* Sudden regime shifts are difficult for numerical forecasters because an approaching cloud front may affect future photovoltaic output before it appears in the local generation history. Multimodal accuracy gains alone cannot establish whether a model uses imagery, since retraining also changes capacity and optimization. We evaluate visual reliance on disjoint test plants while controlling these factors: the video encoder remains frozen, only the upper blocks of the time-series transformer are updated, and the cross-modal bridge is trained from scratch. Our counterfactual protocol scores the same trained weights with vision active and disabled. A learned-query resampler produces reliance indistinguishable from zero ($0.0000 plus.minus 0.0015$). Replacing that pooled representation with 49 addressable spatial cells, sequence interleaving, and temporal-novelty selection increases reliance to $0.0073 plus.minus 0.0002$ and yields the lowest ramp NMAE in the benchmark ($0.1435$). Falsification controls show that performance depends on fresh, site-specific imagery and novelty-selected cells, while frame order has negligible effect. The improvement in point forecasts comes with lower 80% prediction-interval coverage ($0.753$ versus $0.772$ for the unimodal control).
]

#v(1.0em)

= Introduction

Forecast errors in physical and energy systems do not carry uniform costs over time. Under clear skies or uniform overcast, photovoltaic (PV) generation follows a smooth, quasi-deterministic curve governed mainly by solar geometry. Simple persistence or autoregressive models are difficult to beat in these conditions. The largest operational errors occur during sudden ramps, when output changes by a large fraction of installed capacity within minutes. These events complicate grid balancing, incur imbalance penalties, and require fast reserve dispatch @sunset @fusionsf.

Ramp events also expose a physical limit of unimodal forecasting. A plant's historical power series cannot reveal a cloud that has not yet reached the array; a front approaching at 40 km/h leaves no local generation signal before optical occlusion. Numerical weather prediction (NWP) provides regional forecasts, but its spatial and temporal resolution is too coarse to resolve localized cloud boundaries.

Two problems make it difficult to establish whether an architecture uses visual information to resolve ramps.

The first is the strength of the unimodal baseline. General-purpose time-series foundation models (TSFMs), pretrained on multi-domain corpora, patch numerical series, model long dependencies with self-attention, consume weather covariates directly, and transfer zero-shot across unseen geographies @chronos2 @timesfm @ttm. In our benchmark, a fine-tuned unimodal foundation model reaches a skill score of $52.30$ against Smart Persistence on held-out plants, outperforming nearly every published multimodal solar forecaster (@tbl-leaderboard-multimodal, @tbl-leaderboard-tsfms). Imagery must add spatial information absent from the numerical covariates.

We evaluate every model on plants excluded from training and validation. The primary cross-plant split assigns different installations to each set, preventing the model from relying on plant-specific histories or memorized spatial patterns. Performance therefore measures transfer to unseen plants under the same forecasting task instead of interpolation among plants observed during training.

The second problem is that conventional ablations confound visual information with retraining. Most multimodal forecasting architectures train end to end and evaluate visual ablations by retraining a smaller model from scratch @solarvlm @timevlm @m3snet. The resulting loss of accuracy may reflect predictive information in the images, additional capacity from the visual encoder, or regularization through the multimodal gradient path. Standard retraining protocols cannot separate these effects.

Our framework holds the training data, optimization schedule, capacity, and backbone policy constant across arms. We freeze the self-supervised video transformer and pre-compute its outputs offline. After fine-tuning the time-series foundation model on the training plants without vision, we update only its upper encoder blocks during multimodal training, using the same reduced learning rate in every arm. Only the cross-modal interface is learned from scratch, leaving bridge geometry as the experimental variable.

The *counterfactual reliance protocol* scores each trained model twice with identical parameters, first with all inputs and then with the visual pathway disabled. The difference in ramp error measures the accuracy attributable to observing the sky. Falsification controls replace the input with stale imagery or imagery from another plant to determine whether this reliance comes from contemporaneous local conditions or superficial scene statistics.

= Related work and the information bottleneck hypothesis <sec-related>

== Gated late fusion of bottlenecked embeddings: Solar-VLM
Solar-VLM @solarvlm is representative of recent vision-language architectures @pvvlm @unicast. A frozen vision-language model embeds satellite crops and meteorological text prompts. Cross-attention pooling reduces them to one station vector, which is concatenated with the time-series state and modulated by a learned scalar gate.

A single vector for a $128 times 128$ km scene primarily captures average cloudiness and loses fine-grained cloud edges. Its overlap with numerical weather covariates can cause the learned gate to collapse toward the numeric branch.

#figure(
  image("figures/related_solar_vlm_original.png", width: 100%),
  caption: [Original Solar-VLM architecture, reproduced as a screenshot from the source paper @solarvlm.],
) <fig-related-solar-vlm>

== Synthetic imagery as a representation regularizer: Time-VLM
Time-VLM @timevlm ranks second in our benchmark suite (SS $54.04$), although its second modality contains no external physical observation. It renders the historical power curve as a synthetic image and embeds it with statistical text prompts through a vision-language model.

The visual branch re-encodes the same numerical history through a pretrained vision backbone @visionts without observing the sky. Any improvement may therefore come from geometric regularization of one-dimensional patterns. Benchmark accuracy alone cannot show that a multimodal model perceived an external physical process, which motivates our counterfactual evaluation.

#figure(
  image("figures/related_time_vlm_original.png", width: 100%),
  caption: [Original Time-VLM architecture, reproduced as a screenshot from the source paper @timevlm.],
) <fig-related-time-vlm>

== Multimodal foundation pretraining: Aurora
Aurora @aurora pretrains a generative multimodal time-series foundation model on a cross-domain corpus of numerical series and their derived image and text representations. Pretrained encoders produce modality-specific features, which a cross-modality encoder combines through token distillation and modality-guided self-attention.

Aurora studies how multimodal pretraining with derived modalities can improve general-purpose forecasting. Our setting supplies an independent physical observation and tests whether the forecaster retains its spatial structure. This separates representation transfer from reliance on external perceptual evidence.

#figure(
  image("figures/related_aurora_original.png", width: 100%),
  caption: [Original Aurora architecture, reproduced as a screenshot from the source paper @aurora.],
) <fig-related-aurora>

== Multimodal token reduction: Nemotron 3 Nano Omni
Nemotron 3 Nano Omni @nemotron uses an encoder, projector, and decoder. Separate encoders turn audio, images, and video into modality-specific tokens. Projectors align the token widths before passing the sequence to a language-model backbone. The vision path reduces the token count twice: a pixel shuffle applies $4 times$ spatial downsampling before projection, then Efficient Video Sampling (EVS) retains tokens from regions that change over time @evs.

Nemotron's visual path is the closest precedent for S2d and S2e. We reduce a spatial feature grid while keeping each location addressable, then use a position-wise MLP to match the token width of Chronos-2. We also adapt EVS to rank visual tokens by temporal novelty. S2d applies this selection to one recent satellite window, while S2e repeats it at five matched daily anchors. Our forecasting interface omits Nemotron's language decoder and audio branch and interleaves the selected visual tokens directly with the time-series tokens.

#figure(
  image("figures/related_nemotron_visual_path.png", width: 100%),
  caption: [Nemotron 3 Nano Omni architecture, with the visual encoder and token-reduction path expanded. Audio, visual, and text tokens are aligned and concatenated before entering the shared Nemotron language-model backbone.],
) <fig-related-nemotron>

== Forward-aligned exogenous covariates: iTransformer
iTransformer @itransformer embeds each physical variable, including generation, solar zenith, cloud cover, and temperature, as one token spanning the temporal window. Self-attention then operates across variables. Under our protocol, its numerical weather covariates are shifted across the forecast horizon to represent operational NWP forecasts.

The methods differ in temporal alignment. iTransformer reaches a ramp NMAE of $0.1445$ with exogenous covariates that describe the forecasted hours directly. Satellite imagery covers only the past and present, requiring a multimodal model to learn how current cloud boundaries map to future occlusion.

#figure(
  image("figures/related_itransformer_original.png", width: 100%),
  caption: [Original iTransformer architecture, reproduced as a screenshot from the source paper @itransformer.],
) <fig-related-itransformer>

= Multimodal corpus and physical bounds <sec-dataset>

== Dataset construction and standardization
The standardized corpus combines high-resolution photovoltaic generation, geostationary satellite imagery, and local atmospheric covariates from two geographic tracks to evaluate zero-shot transfer across plants (@tbl-data-sources).

The dataset covers plants in the UK and the US, with two complete years in the UK track. Photovoltaic production, weather variables, and satellite images are matched by time and location so that all sources describe the same operating conditions. Cleaning and a common data structure support consistent multimodal training and evaluation.

@tbl-pv-dataset-comparison places the corpus alongside the PV datasets most relevant to multimodal and cross-plant forecasting.

#tbl(
  [Comparison with PV-specific datasets relevant to multimodal and cross-plant forecasting. A dash indicates that the publication does not provide the modality or evaluation property.],
  columns: (0.95fr, 0.75fr, 0.55fr, 1.2fr, 1.1fr, 1.15fr),
  align: (left, left, left, left, left, left),
  table.header[Dataset][PV systems][Frequency][Plant signal and metadata][Environmental context][Evaluation scope],
  [HKUST Rooftop PV @hkpvdata], [60 rooftop systems, Hong Kong], [5 min PV; 1 min weather], [Inverter-level power and electrical measurements; equipment and location metadata], [Measurements from an on-site weather station], [Dataset for PV analytics and forecasting; no imagery or disjoint-plant benchmark],
  [MMSP in FusionSF @fusionsf], [88 plants, one Chinese province], [1 h], [Capacity-normalized PV power; anonymized plant locations], [Himawari-8/9 imagery and ECMWF NWP], [Zero-shot evaluation on unseen plants],
  [SKIPP'D @skippd], [One 30.1 kW system, California], [1 min], [PV power; capacity, tilt, and azimuth], [Ground-based fisheye sky imagery and video], [Chronological benchmark at one plant],
  [PVOD @pvod], [10 plants, China], [15 min], [PV power; capacity, area, panel count, orientation, and location], [Local meteorological measurements and NWP], [Multi-plant dataset; no imagery or disjoint-plant benchmark],
) <tbl-pv-dataset-comparison>

Having situated the corpus among related PV datasets, @tbl-data-sources details the sources and processing used to construct its aligned power, satellite, and meteorological inputs.

#tbl(
  [Data sources comprising the multimodal corpus. Continuous multi-band infrared satellite frames are co-registered per site for exact spatiotemporal synchronization with ground generation.],
  columns: (auto, 1fr),
  align: (left, left),
  table.header[Modality / Track][Source, Processing, and Properties],
  [PV Power (UK Track)], [Open Climate Fix @ukpv: 30-minute generation across residential rooftop systems (2019--2020). Normalised strictly by audited installed capacity ($P / P_"cap" in [0, 1]$).],
  [Satellite (UK Track)], [EUMETSAT @seviri: Multi-band non-HRV infrared and visible channels. Reprojected per site and cropped to $128 times 128$ km spatial windows at 15-minute intervals.],
  [PV & Satellite (US Track)], [NREL PVDAQ industrial arrays @pvdaq paired with GOES-16 multi-band geostationary crops ($256 times 256$ km).],
  [Meteorological Covariates], [Open-Meteo historical archive @openmeteo: 8 surface variables joined by nearest coordinates, paired with analytical solar geometry and clear-sky irradiance.],
) <tbl-data-sources>

= Experimental protocol and metrics <sec-protocol>

== Cross-plant disjoint evaluation
The *cross-plant split* assigns the 98 quality-controlled UK installations to 70 training, 14 validation, and 14 test plants. All three disjoint sets cover the same two-year period (2019--2020). The model must transfer across microclimates, array orientations, and shading profiles without the spatial memorization possible under a chronological split.

== Standardized task and metrics
All models use a standardized input and output window:
- *History Window*: 14 days (672 steps at 30-minute sampling) of capacity-normalized historical power $y_t in [0, 1]$, combined with 14 atmospheric and solar covariates.
- *Visual Window*: S2d uses 8 satellite frames covering the 6 hours immediately preceding the forecast origin, centered over the installation. S2e uses five such 8-frame windows, sampled at the same clock time on five consecutive days, for 40 satellite frames in total.
- *Forecast Horizon*: 6 hours ahead, predicting 9 non-parametric quantiles ($q in {0.1, dots, 0.9}$) to quantify predictive uncertainty.

Evaluation covers 165,295 daytime steps from 14 test plants. We use two primary metrics:
1. *Generalization Skill Score ($SS$)*: Denominated against Smart Persistence ($SS = 0$):
   $ SS = 100 dot (1 - frac("NRMSE"_"model", "NRMSE"_"persistence")) $
   where higher is better, and $100$ denotes perfect forecasting.
2. *Ramp Normalized Mean Absolute Error ($"NMAE"_"ramp"$)*: Evaluated exclusively on steps exhibiting high volatility, defined as the top decile of true generation changes between consecutive sampling intervals ($|y_t - y_{t-1}|$). The metric focuses on the physical regime where cloud edges induce operational penalties.

= Method: controlled foundation model fusion

== Foundation backbones

The architecture combines two pretrained foundation models under different freezing policies during multimodal training:

#tbl(
  [Component stack of the multimodal architecture. The visual backbone is fully frozen. Only the upper encoder blocks of the forecaster remain trainable, under the same policy in every arm. The bridge geometry is therefore the independent variable.],
  columns: (auto, 1.2fr, 1fr),
  align: (left, left, left),
  table.header[Component][Architecture & Latent Dimensions][Optimization State],
  [Time-Series Backbone], [Chronos-2 transformer @chronos2: 12 blocks, hidden dimension 768, 12 attention heads. Patch size 16.], [Fine-tuned end to end on the training plants in the unimodal stage; thereafter fixed except the upper 3 of 12 encoder blocks at $0.1 times$ learning rate (identical across all arms)],
  [Visual Backbone], [V-JEPA 2.1 ViT-L/16 @vjepa21: Self-supervised spatiotemporal video encoder. 8 input frames $arrow.r$ 4 latent temporal slices $times$ 196 spatial patches $times$ 1024 channels.], [Frozen entirely; pre-computed offline to prevent gradient leakage],
  [Cross-Modal Bridge], [Dimension alignment and spatial/temporal fusion interface.], [Trained from scratch],
) <tbl-components>

=== The time-series foundation forecaster
The time-series backbone is an encoder-only transformer pretrained on heterogeneous time-series datasets @chronos2. Non-overlapping 16-sample patches map the 672-step power history to 42 context tokens, while known future weather covariates enter through parallel token sequences. Bidirectional self-attention processes all positions, and a linear multi-quantile head predicts nine quantiles for the 12-step horizon in one non-autoregressive pass.

#figure(
  image("figures/mmtsfm_s1.png", width: 90%),
  caption: [Unimodal Foundation Baseline (S1): The time-series transformer operates solely on numerical generation history and weather covariates without visual input.],
) <fig-s1>


=== The self-supervised video encoder
We use V-JEPA 2.1 @vjepa21, whose masked latent-prediction objective models change across spatiotemporal tubelets. This objective suits fine-grained cloud deformation better than the static object semantics learned through text-image alignment. V-JEPA maps each eight-frame satellite clip to $bold(Z)_v in bb(R)^(4 times 196 times 1024)$: four temporal slices, each containing a $14 times 14$ grid of 1024-dimensional spatial patches.

== The fusion bridges

We evaluate three architectures for connecting the visual latent field $bold(Z)_v$ to the foundation forecaster:

=== Paradigm 1: bottlenecked late fusion (S2a)
S2a applies learned-query cross-attention pooling to the $4 times 196$ visual tokens (@fig-s2a), compressing the complete scene into $bold(v) in bb(R)^(768)$. A linear adapter projects this summary into the transformer's batch axis as an auxiliary covariate row, where group self-attention combines it with the numerical sequence at each layer.

The summary can distinguish broad regimes such as clear and overcast conditions, but one vector cannot retain the localized gradients of advancing cloud boundaries. Once pooling removes those spatial coordinates, downstream layers cannot reconstruct them.

#figure(
  image("figures/mmtsfm_s2a.png", width: 90%),
  caption: [Bottlenecked Late Fusion (S2a): The satellite clip is compressed via a learned-query resampler into a single global summary vector, injected as an auxiliary parallel channel.],
) <fig-s2a>
=== Paradigm 2: resampler-free interleaved sequence fusion (S2d)
S2d removes the learned-query resampler and sends visual tokens to the forecaster through a position-wise projector (@fig-s2d). Its sequence of encoder, projector, and forecaster follows Nemotron 3 Nano Omni @nemotron, which combines an MLP projector with spatial token reduction and Efficient Video Sampling (EVS) @evs. The visual encoder produces $4 times 196 = 784$ tokens, compared with 42 macro-patches in the numerical input. Processing every visual token would spend much of the attention budget on static regions of the satellite crop.

We first reduce the visual field spatially. Dividing the $14 times 14$ patch grid into non-overlapping $2 times 2$ neighborhoods produces a $7 times 7$ field of 49 cells, each with 1024 channels. The reduction lowers the number of visual tokens passed to the forecaster while preserving the field's spatial layout. We then project the 49 cells independently so that subsequent attention layers can still address each location.

A two-layer GELU MLP maps each 1024-channel cell to the forecaster width ($1024 arrow.r 768$). It processes cells independently and does not mix information across locations. A learned table of 49 coordinate-indexed vectors adds each cell's spatial position.

The position encoding treats space and time separately. The learned table records spatial coordinates, while sequence position records the time of each frame. S2d omits a temporal-slice embedding because it would duplicate this information and complicate interpretation of the frame-order control in @sec-instrument. After temporal shuffling, the visual tokens receive new sequence positions. An embedding attached to each slice could still expose its original order.

EVS selects the visual tokens that change most between consecutive frames. For each spatial cell, it computes cosine dissimilarity between a latent slice and the corresponding cell in the preceding slice. High values identify changing regions; low values identify static content. Assigning the first slice a score of $+infinity$ retains one complete anchor field. EVS ranks the $4 times 49 = 196$ candidates globally, keeps the top 98 at $q = 0.5$, and restores frame-then-cell order. The visual sequence contains the 49-cell anchor field and 49 tokens from the most dynamic regions. This fixed budget can retain a localized cloud front without processing the full $128 times 128$ km field.

Nemotron uses EVS after training as a deployment-time throughput control and tunes $q$ for each deployment @nemotron. S2d applies the same parameter-free selection during training and evaluation. We fix $q$ as part of the architecture, so the forecaster always receives 98 visual tokens. The selection rule also makes the vision-off evaluation in @sec-instrument resemble an input pattern seen during training. Because EVS has no learned parameters, the random-selection control in @tbl-controls can replace its ranking rule on a trained checkpoint without changing the sequence length. The control separates the selection criterion from the token budget.

The selected visual tokens are inserted into the numerical sequence according to their observation times. One numeric patch spans $16 times 30$ minutes, or 8 hours, whereas the visual window spans 6 hours within the final context patch.

The final sequence has 141 positions: 42 numeric macro-patches, 98 visual tokens, and one future query. Native bidirectional self-attention processes them together. Nemotron orders modality streams temporally relative to one another; S2d places each visual token at its observation timestamp within the numerical sequence. The modalities share the same attention mechanism, with no intervening cross-attention layer, gate, or auxiliary channel. The future query can attend to both visual and numeric tokens. Because the visual window ends immediately before the forecast horizon, the visual tokens describe the sky observed before prediction.

During training, S2d drops the visual stream in half of the samples and the numeric stream in one tenth, but never drops both together. Visual dropout exposes the model to the vision-off pattern used by @sec-instrument. The difference between the active and disabled passes can therefore be attributed to missing visual information without introducing an unseen input distribution.



#figure(
  image("figures/mmtsfm_s2d.png", width: 95%),
  caption: [Resampler-Free Interleaved Fusion (S2d)],
) <fig-s2d>

=== Paradigm 3: multi-anchor interleaved sequence fusion (S2e)

S2e spreads S2d's visual context over five Chronos-2 patches (@fig-s2e). Each anchor samples the same clock time on one of five consecutive days, giving the model several sky-power pairs at matched solar geometry. Coverage determines the daily stride: UK satellite frames run from 02:00 to 16:00 UTC, and an 8-hour ladder fills all five anchors for none of the 24,605 evaluated origins, compared with $94.4%$ for a 24-hour ladder.

S2e retains the initial configuration's five-anchor layout, 143-token sequence length, warm start, and optimizer. At each anchor, paired novelty ranks 98 candidate visual tokens by scoring 49 two-latent spatial trajectories according to cosine change. It keeps both endpoints for the ten highest-scoring trajectories. The result is 20 visual tokens per anchor and 100 visual tokens in total, accompanied by 42 numeric tokens and one future query. Each visual block takes the integer position of its anchor patch; S2d uses fractional positions inside the final patch.

If S2e performs well, a subsequent experiment will retain all 98 candidates at each anchor. The resulting 490 visual tokens will test whether a larger token budget improves the multi-anchor model.

#figure(
  image("figures/mmtsfm_s2e.png", width: 95%),
  caption: [Canonical Multi-Anchor Interleaving (S2e)],
) <fig-s2e>

= The counterfactual reliance instrument <sec-instrument>

The *Counterfactual Reliance Metric* measures visual utility without changing model capacity or retraining an ablated system:

$ "Reliance"_"ramp" = "NMAE"_"ramp"^(bold(V)=emptyset) - "NMAE"_"ramp"^(bold(V)="active") $

After training, we freeze the parameters $bold(theta)$ and score the test set with vision active ($bold(V)="active"$). We then disable the visual pathway in the same model ($bold(V)=emptyset$). A positive difference means that accuracy depends on the imagery because the weights and all other inputs remain unchanged.

= Results and analysis

== The fusion ladder: comparing visual reliance

@tbl-ladder reports nominal accuracy and counterfactual visual reliance for the three trained arms.

#tbl(
  [Visual reliance across fusion paradigms. Reliance is the reduction in ramp error from observing the sky, measured counterfactually on frozen weights across seeds 42--44. Bottlenecked pooling yields zero reliance; the resampler-free arm exceeds the significance floor by more than six times.],
  columns: (auto, 1.2fr, auto, auto, auto),
  align: (left, left, center, center, center),
  table.header[Arm][Architectural Paradigm][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)][Ramp Reliance ($arrow.t$)],
  [S1], [Unimodal Foundation Baseline], [52.30 #text(size: 8pt)[$plus.minus 0.41$]], [0.1506 #text(size: 8pt)[$plus.minus 0.0010$]], [---],
  [S2a], [Bottlenecked Late Fusion (Resampler)], [52.58 #text(size: 8pt)[$plus.minus 0.43$]], [0.1487 #text(size: 8pt)[$plus.minus 0.0010$]], [0.0000 #text(size: 8pt)[$plus.minus 0.0015$]],
  [*S2d*], [*Resampler-Free Interleaved Fusion*], [*55.64* #text(size: 8pt)[$plus.minus 0.18$]], [*0.1435* #text(size: 8pt)[$plus.minus 0.0004$]], [*0.0073* #text(size: 8pt)[$plus.minus 0.0002$]],
) <tbl-ladder>

S2a reduces nominal ramp NMAE from $0.1506$ to $0.1487$, but its counterfactual reliance is $0.0000$: disabling vision in the frozen model does not change performance. The nominal gain comes from multimodal training rather than visual information used at inference. After the resampler removes spatial structure, the pooled vector contributes no measurable signal.

S2d reaches a ramp reliance of $0.0073$ across all seeds, more than six times the pre-registered significance floor ($0.0011$). Its 49 separate cells give the foundation transformer direct access to localized visual boundaries and produce the best ramp accuracy in the suite. S2a and S2d differ in resampling, token count, and token placement, so this comparison supports only the complete S2d design.

== Mechanistic diagnostics and falsification controls

#tbl(
  [Diagnostic falsification controls on the resampler-free interleaved arm.],
  columns: (auto, 1.2fr, 1.2fr),
  align: (center, left, left),
  table.header[\#][Diagnostic question][Observed effect and interpretation],
  [1], [Does S2d rely on temporal frame ordering? (Kinematic frame shuffle)],
  [Nearly unchanged: $Delta "NMAE"_"ramp" approx 0$ across all seeds ($< 0.0003$). The model does not measurably track kinematic cloud trajectories.],
  [2], [Does micro-temporal sub-patch positioning matter? (Collapsing sub-patch indices)],
  [Little change: Ramp NMAE increases by $+0.0008$; reliance remains $0.0064$ versus $0.0063$. Temporal indexing within the patch has no measurable effect.],
  [3], [Is a recent sky sufficient? (Stale sky: imagery lagged by 6 hours)],
  [Ramp NMAE increases by $+0.0169$ ($+11.7%$ error), showing that the imagery must be recent.],
  [4], [Does the model resolve site-specific sky? (Contemporaneous cross-plant donor sky)],
  [Ramp NMAE increases by $+0.0176$, and Skill Score falls from $55.10$ to $37.00$. The model relies on imagery local to the installation.],
  [5], [Does dynamic novelty pruning carry informational signal? (Uniform random token selection at equal budget)],
  [Ramp NMAE increases by $+0.0039$, and Skill Score falls by $-1.52$ percentage points. Novelty scoring retains information about predictive cloud boundaries.],
) <tbl-controls>


== The intrinsic calibration cost

S2d improves point-forecast ramp accuracy but changes probabilistic calibration:

#tbl(
  [Probabilistic calibration metrics across foundation model configurations.],
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  table.header[Configuration][Ramp NMAE ($arrow.b$)][80% Prediction Interval Coverage][Quantile ECE ($arrow.b$)],
  [S1 (Unimodal Control)], [0.1506], [0.772], [0.0297],
  [*S2d (Resampler-Free)*], [*0.1435*], [0.753], [0.0314],
) <tbl-calibration>


= Comprehensive benchmark leaderboard

The status flag in the tables denotes the training regime: *T* means trained on the benchmark task, *FT* means fine-tuned from pretrained weights, and *ZS* means zero-shot transfer without task-specific training.

#tbl(
  [Multimodal solar forecasting leaderboard on 14 held-out test installations. S2d achieves the top rank across all multimodal architectures.],
  columns: (auto, auto, auto, 1fr, auto, auto),
  align: (center, left, center, left, center, center),
  table.header[Rank][Model][Status][Cross-Modal Fusion Strategy][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)],
  [1], [*MMTSFM S2d (Ours)*], [FT], [*Resampler-free sequence interleaving in token space*], [*55.64*], [*0.1435*],
  [2], [Time-VLM @timevlm], [FT], [Synthetic series plot embedded via frozen VLM], [54.04], [---],
  [3], [MMTSFM S2a (Ours)], [FT], [Bottlenecked Perceiver resampler on batch axis], [52.58], [0.1487],
  [11], [Solar-VLM @solarvlm], [FT], [Gated Perceiver pooling of satellite + text prompts], [43.96], [0.1514],
  [19], [CrossViViT @crossvivit], [FT], [Cross-attention from historical station steps], [34.91], [---],
  [24], [Aurora Fine-Tuned @aurora], [FT], [Joint multimodal foundational pretraining], [TBD], [---],
  [TBD], [Aurora Zero-Shot @aurora], [ZS], [Joint multimodal foundational pretraining], [23.24], [---],
  [25], [SUNSET @sunset], [T], [Convolutional joint encoding + feature concatenation], [21.62], [---],
  [26], [UniCast @unicast], [FT], [Prompted multimodal foundation forecaster], [12.11], [---],
  [28], [VisionTS++ @visionts], [FT], [Continual visual pretraining on synthetic plots], [1.67], [---],
) <tbl-leaderboard-multimodal>

#tbl(
  [Time-series foundation model baselines. S1 is the unimodal foundation control, while the remaining entries use either fine-tuning, retrieval, or zero-shot transfer.],
  columns: (auto, auto, auto, 1fr, auto, auto),
  align: (center, left, center, left, center, center),
  table.header[Rank][Model][Status][Architectural Paradigm][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)],
  [5], [MMTSFM S1 (Ours)], [FT], [Unimodal foundation control], [52.30], [0.1506],
  [7], [TS-RAG @tsrag], [ZS], [Retrieval-augmented foundation model on numeric history], [47.79], [---],
  [8], [Cross-RAG @crossrag], [ZS], [Cross-attention retrieval on historical series], [47.68], [---],
  [9], [Chronos-2 Zero-Shot @chronos2], [ZS], [Direct zero-shot foundation model transfer], [47.37], [0.1544],
  [16], [CoRA @cora], [FT], [Covariate-aware adapter on frozen foundation backbone], [37.98], [0.1624],
  [18], [TTM Fine-Tuned @ttm], [FT], [Tiny Time Mixer with channel mixing], [36.01], [0.1716],
  [21], [TiRex Zero-Shot @tirex], [ZS], [Pretrained zero-shot time-series model], [28.73], [0.1826],
  [22], [TimesFM Zero-Shot @timesfm], [ZS], [Decoder-only zero-shot foundation model], [27.08], [0.1902],
  [30], [TTM Zero-Shot @ttm], [ZS], [Pretrained zero-shot Tiny Time Mixer], [-8.07], [0.2922],
) <tbl-leaderboard-tsfms>

#tbl(
  [Supervised deep learning baselines. S2d statistically ties iTransformer's top ramp performance, with measurable reliance on past imagery.],
  columns: (auto, auto, auto, 1fr, auto, auto),
  align: (center, left, center, left, center, center),
  table.header[Rank][Model][Status][Architectural Paradigm][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)],
  [4], [iTransformer + Covs @itransformer], [T], [Inverted variate tokens with forward-shifted NWP covariates], [52.57], [0.1445],
  [10], [PatchTST @patchtst], [T], [Channel-independent patched transformer], [45.81], [0.1543],
  [12], [Temporal Fusion Transformer @tft], [T], [Gated residual network with temporal self-attention], [42.64], [0.1605],
  [13], [MLP Baseline], [T], [Dense multi-layer perceptron over flattened history], [42.19], [0.1624],
  [20], [DLinear @dlinear], [T], [Decomposed trend-seasonal linear model], [32.31], [0.1746],
) <tbl-leaderboard-supervised>

#tbl(
  [Tabular, classical machine learning, and statistical reference baselines.],
  columns: (auto, auto, auto, 1fr, auto, auto),
  align: (center, left, center, left, center, center),
  table.header[Rank][Model][Status][Model Type][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)],
  [14], [TabPFN @tabpfn], [ZS], [Prior-data fitted network for tabular classification/regression], [40.63], [0.1631],
  [15], [LightGBM], [T], [Gradient-boosted decision trees over lagged tabular features], [38.54], [0.1672],
  [17], [TabFM Ensemble], [ZS], [Ensemble of tabular foundation models], [36.26], [0.1573],
  [23], [Hourly Climatology], [T], [Historical hourly mean per installation], [23.37], [0.1665],
  [27], [Seasonal Naive], [T], [Persistence of generation from 24 hours prior], [10.68], [0.2056],
  [29], [Persistence], [T], [Persistence of generation from immediate prior step], [1.41], [0.2550],
) <tbl-leaderboard-baselines>

== Qualitative inspection of forecast trajectories

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 0.6em,
    image("figures/forecast_example_6648.png", width: 100%),
    image("figures/forecast_example_11176.png", width: 100%),
    image("figures/forecast_example_11287.png", width: 100%),
    image("figures/forecast_example_12642.png", width: 100%),
  ),
  caption: [Representative six-hour forecast trajectories on held-out plants.],
) <fig-forecast-examples>


#pagebreak()
#bibliography("refs.bib", title: "References", style: "ieee")
