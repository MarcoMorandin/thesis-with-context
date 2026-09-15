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
  *Abstract.* Sudden physical regime shifts expose a limitation of numerical forecasting because they leave no trace in the historical series. In solar power forecasting, sharp ramps caused by advecting cloud edges account for most of the operational loss. Establishing whether a multimodal forecaster uses visual observations remains difficult because end-to-end retraining conflates information from the images with parameter capacity, optimization dynamics, and regularization. Strong zero-shot transfer from modern time-series foundation models raises the bar further by making coarse visual summaries redundant. We control these confounds by fixing both foundation backbones: the self-supervised video encoder is frozen, the time-series transformer is held fixed except for its upper encoder blocks, and only the cross-modal bridge is trained from scratch. Our counterfactual reliance protocol scores the same frozen weights with and without vision, measuring the contribution of imagery directly in ramp error. A learned-query bottleneck resampler yields reliance indistinguishable from zero ($0.0000 plus.minus 0.0015$). Removing the resampler preserves 49 individually addressable spatial cells, projects them into sequence self-attention, and prunes static background through temporal novelty selection. This design restores significant reliance ($0.0073 plus.minus 0.0002$) and achieves the lowest ramp error ($0.1435$) in a 27-baseline suite. A matched-budget ablation further shows that averaging four sub-patches into a 1024-channel cell matches the original 4096-channel concatenation on ramp error, improves skill score by $0.54$ percentage points, and calibrates better. The measured gain depends on removing the pooling bottleneck and retaining cells selected by temporal novelty, not on sub-cell payload width. Falsification controls attribute the gain to localized cloud boundaries represented as a non-stationary spatial field; neither frame order nor kinematic tracking contributes measurably. The sharper ramp response reduces prediction-interval coverage.
]

#v(1.0em)

= Introduction

Forecast errors in physical and energy systems do not carry uniform costs over time. Under clear skies or uniform overcast, photovoltaic (PV) generation follows a smooth, quasi-deterministic curve governed mainly by solar geometry and simple persistence or autoregressive models are difficult to beat. The largest operational errors occur during sudden ramps, when output changes by a large fraction of installed capacity within minutes. These events complicate grid balancing, incur imbalance penalties, and require fast reserve dispatch @sunset @fusionsf.

Ramp events also expose a physical limit of unimodal forecasting. A plant's historical power series cannot reveal a cloud that has not yet reached the array; a front approaching at 40 km/h leaves no local generation signal before optical occlusion. Numerical weather prediction (NWP) provides regional forecasts, but its spatial and temporal resolution is too coarse to resolve localized cloud boundaries.

Demonstrating that an architecture actually uses visual information to resolve ramps presents two recurring problems:

First, the unimodal baseline is strong. General-purpose time-series foundation models (TSFMs), pretrained on diverse multi-domain corpora, have improved the empirical frontier @chronos2 @timesfm @ttm. They patch numerical series, model long temporal dependencies with self-attention, consume weather covariates directly and transfer zero-shot across unseen geographies. In our benchmark, a fine-tuned unimodal foundation model reaches a skill score of $52.30$ against Smart Persistence on held-out plants, outperforming nearly every published multimodal solar forecaster (@tbl-leaderboard-multimodal, @tbl-leaderboard-tsfms). A visual modality must therefore add spatial information that the numerical covariates do not already contain.

All models are evaluated on plants that are disjoint from those used for training and validation. The primary cross-plant split assigns separate installations to each set, so a model cannot rely on plant-specific histories or memorized spatial patterns. Performance therefore measures transfer to unseen plants under the same forecasting task, rather than interpolation within plants already observed during training.

Second, conventional ablations confound visual information with retraining. Most multimodal forecasting architectures are trained end to end, and their visual ablations retrain a smaller model from scratch @solarvlm @timevlm @m3snet. Any loss of accuracy can then arise because the images contained predictive physical signals, because the visual encoder supplied additional capacity, or because the multimodal gradient path regularized optimization. Standard retraining protocols cannot distinguish these mechanisms.

Our controlled framework fixes training data, optimization schedules, capacity, and backbone policies across arms. The self-supervised video transformer is frozen and pre-computed offline. The time-series foundation model is first fine-tuned on the training plants without vision; during multimodal training, only its upper encoder blocks remain trainable, at a reduced learning rate shared by every arm. The cross-modal interface is the only component learned from scratch, leaving bridge geometry as the experimental variable.

The *counterfactual reliance protocol* scores each trained model twice with identical parameters: once with all inputs and once with the visual pathway disabled. The difference in ramp error measures the accuracy that depends on observing the sky. Falsification controls then replace the input with stale imagery or imagery from another plant to test whether this reliance comes from contemporaneous local conditions or superficial scene statistics.

= Related Work and the Information Bottleneck Hypothesis <sec-related>

== Gated Late Fusion of Bottlenecked Embeddings: Solar-VLM
Solar-VLM @solarvlm represents recent vision--language architectures @pvvlm @unicast. A frozen vision--language model embeds satellite crops and meteorological text prompts. Cross-attention pooling reduces them to one station vector, which is concatenated with the time-series state and modulated by a learned scalar gate.

One vector for a $128 times 128$ km scene primarily represents average cloudiness and loses fine-grained cloud edges. Because this summary overlaps with numerical weather covariates, the learned gate can collapse toward the numeric branch.

#figure(
  image("figures/related_solar_vlm_original.png", width: 100%),
  caption: [Original Solar-VLM architecture, reproduced as a screenshot from the source paper @solarvlm.],
) <fig-related-solar-vlm>

== Synthetic Imagery as Representation Regularizer: Time-VLM
Time-VLM @timevlm ranks second in our benchmark suite (SS $54.04$), although its second modality contains no external physical observation. It renders the historical power curve as a synthetic image and embeds it with statistical text prompts through a vision--language model.

The visual branch re-encodes the same numerical history through a pretrained vision backbone @visionts and never observes the sky. Its improvement can therefore come from geometric regularization of one-dimensional patterns. This result motivates counterfactual evaluation: benchmark accuracy alone does not show that a multimodal model perceived an external physical process.

#figure(
  image("figures/related_time_vlm_original.png", width: 100%),
  caption: [Original Time-VLM architecture, reproduced as a screenshot from the source paper @timevlm.],
) <fig-related-time-vlm>

== Multimodal Foundation Pretraining: Aurora
Aurora @aurora pretrains a generative multimodal time-series foundation model on a cross-domain corpus of numerical series and their derived image and text representations. Pretrained encoders produce modality-specific features, which a cross-modality encoder combines through token distillation and modality-guided self-attention.

Aurora studies how derived modalities can improve general-purpose forecasting through multimodal pretraining. Our setting instead supplies an independent physical observation and tests whether its spatial structure remains available inside the forecaster. This distinction separates representation transfer from causal reliance on external perceptual evidence.

#figure(
  image("figures/related_aurora_original.png", width: 100%),
  caption: [Original Aurora architecture, reproduced as a screenshot from the source paper @aurora.],
) <fig-related-aurora>

== Multimodal Token Reduction: Nemotron 3 Nano Omni
Nemotron 3 Nano Omni @nemotron uses an encoder--projector--decoder design. Separate encoders turn audio, images, and video into modality-specific tokens, projectors align their widths, and the resulting sequence is passed to a language-model backbone. The vision path also reduces the number of visual tokens before they reach the backbone: spatial compression preserves the image layout, while Efficient Video Sampling (EVS) keeps tokens from regions that change over time @evs.

The visual side of Nemotron provides the closest precedent for S2d and S2e. We retain the idea of reducing a spatial feature grid while keeping each location addressable, then use a position-wise MLP to match the token width of Chronos-2. We also adapt the EVS principle of ranking visual tokens by temporal novelty. S2d applies this selection to one recent satellite window; S2e repeats it at five matched daily anchors. Nemotron's language decoder and audio branch are outside our forecasting interface, where the selected visual tokens are interleaved directly with the time-series tokens.

#figure(
  image("figures/related_nemotron_visual_path.svg", width: 100%),
  caption: [Correspondence between the Nemotron visual path and its adaptation in S2d/S2e. Nemotron reduces and projects spatial visual tokens before EVS selects temporally novel regions; S2d and S2e retain this sequence of operations but interleave the selected tokens with numerical Chronos-2 tokens.],
) <fig-related-nemotron>

== Forward-Aligned Exogenous Covariates: iTransformer
iTransformer @itransformer embeds each physical variable, including generation, solar zenith, cloud cover, and temperature, as one token spanning the temporal window. Self-attention then operates across variables. Under our protocol, its numerical weather covariates are shifted across the forecast horizon to represent operational NWP forecasts.

The relevant difference is temporal alignment. iTransformer reaches a ramp NMAE of $0.1445$ with exogenous covariates that describe the forecasted hours directly. Satellite imagery observes only the past and present, so a multimodal model must learn how current cloud boundaries map to future occlusion.

#figure(
  image("figures/related_itransformer_original.png", width: 100%),
  caption: [Original iTransformer architecture, reproduced as a screenshot from the source paper @itransformer.],
) <fig-related-itransformer>

= Multimodal Corpus and Physical Bounds <sec-dataset>

== Dataset Construction and Standardization
The standardized corpus combines high-resolution photovoltaic generation, geostationary satellite imagery, and local atmospheric covariates from two geographic tracks to evaluate zero-shot transfer across plants (@tbl-data-sources).

The dataset brings together multiple plants from two regions, the UK and the US, with the UK track covering two complete years. Photovoltaic production, weather variables, and satellite images are coordinated in time and location, so the three sources describe the same operating conditions. The raw inputs have also been cleaned and organized into a common structure, making the corpus suitable for consistent multimodal training and evaluation.

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

= Experimental Protocol and Metrics <sec-protocol>

== Cross-Plant Disjoint Evaluation
The *cross-plant split* assigns the 98 quality-controlled UK installations to 70 training, 14 validation, and 14 test plants. These disjoint plant sets cover the same two-year period (2019--2020). The model must therefore transfer across microclimates, array orientations, and shading profiles without using the spatial memorization available under a chronological split.

== Standardized Task and Metrics
All models operate on a standardized input--output window:
- *History Window*: 14 days (672 steps at 30-minute sampling) of capacity-normalized historical power $y_t in [0, 1]$, combined with 14 atmospheric and solar covariates.
- *Visual Window*: S2d uses 8 satellite frames covering the 6 hours immediately preceding the forecast origin, centered over the installation. S2e uses five such 8-frame windows, sampled at the same clock time on five consecutive days, for 40 satellite frames in total.
- *Forecast Horizon*: 6 hours ahead, predicting 9 non-parametric quantiles ($q in {0.1, dots, 0.9}$) to quantify predictive uncertainty.

Evaluation covers 165,295 daytime steps from 14 test plants. We use two primary metrics:
1. *Generalization Skill Score ($SS$)*: Denominated against Smart Persistence ($SS = 0$):
   $ SS = 100 dot (1 - frac("NRMSE"_"model", "NRMSE"_"persistence")) $
   where higher is better, and $100$ denotes perfect forecasting.
2. *Ramp Normalized Mean Absolute Error ($"NMAE"_"ramp"$)*: Evaluated exclusively on steps exhibiting high volatility, defined as the top decile of true generation changes between consecutive sampling intervals ($|y_t - y_{t-1}|$). This isolates the physical regime where cloud edges induce operational penalties.

= Method: Controlled Foundation Model Fusion

== Foundation Backbones

The architecture combines two pretrained foundation models with different freezing policies during multimodal training:

#tbl(
  [Component stack of the multimodal architecture. The visual backbone is fully frozen. Only the upper encoder blocks of the forecaster remain trainable, under the same policy in every arm. The bridge geometry is therefore the independent variable.],
  columns: (auto, 1.2fr, 1fr),
  align: (left, left, left),
  table.header[Component][Architecture & Latent Dimensions][Optimization State],
  [Time-Series Backbone], [Chronos-2 transformer @chronos2: 12 blocks, hidden dimension 768, 12 attention heads. Patch size 16.], [Fine-tuned end to end on the training plants in the unimodal stage; thereafter fixed except the upper 3 of 12 encoder blocks at $0.1 times$ learning rate (identical across all arms)],
  [Visual Backbone], [V-JEPA 2.1 ViT-L/16 @vjepa21: Self-supervised spatiotemporal video encoder. 8 input frames $arrow.r$ 4 latent temporal slices $times$ 196 spatial patches $times$ 1024 channels.], [Frozen entirely; pre-computed offline to prevent gradient leakage],
  [Cross-Modal Bridge], [Dimension alignment and spatial/temporal fusion interface.], [Trained from scratch],
) <tbl-components>

=== The Time-Series Foundation Forecaster
The time-series backbone is an encoder-only transformer pretrained on heterogeneous time-series datasets @chronos2. Non-overlapping 16-sample patches map the 672-step power history to 42 context tokens, while known future weather covariates enter through parallel token sequences. Bidirectional self-attention processes all positions, and a linear multi-quantile head predicts nine quantiles for the 12-step horizon in one non-autoregressive pass.

#figure(
  image("figures/mmtsfm_s1.svg", width: 90%),
  caption: [Unimodal Foundation Baseline (S1): The time-series transformer operates solely on numerical generation history and weather covariates without visual input.],
) <fig-s1>


=== The Self-Supervised Video Encoder
We use V-JEPA 2.1 @vjepa21, whose masked latent-prediction objective models change across spatiotemporal tubelets. This objective is better matched to fine-grained cloud deformation than the static object semantics learned through text--image alignment. V-JEPA maps each eight-frame satellite clip to $bold(Z)_v in bb(R)^(4 times 196 times 1024)$: four temporal slices, each containing a $14 times 14$ grid of 1024-dimensional spatial patches.

== The Fusion Bridges

We evaluate three architectures for connecting the visual latent field $bold(Z)_v$ to the foundation forecaster:

=== Paradigm 1: Bottlenecked Late Fusion (S2a)
S2a applies learned-query cross-attention pooling to the $4 times 196$ visual tokens (@fig-s2a), compressing the complete scene into $bold(v) in bb(R)^(768)$. A linear adapter projects this summary into the transformer's batch axis as an auxiliary covariate row, where group self-attention combines it with the numerical sequence at each layer.

The summary can distinguish broad regimes such as clear and overcast conditions, but one vector cannot retain the localized gradients of advancing cloud boundaries. Once pooling removes those spatial coordinates, downstream layers cannot reconstruct them.

#figure(
  image("figures/mmtsfm_s2a.svg", width: 90%),
  caption: [Bottlenecked Late Fusion (S2a): The satellite clip is compressed via a learned-query resampler into a single global summary vector, injected as an auxiliary parallel channel.],
) <fig-s2a>
=== Paradigm 2: Resampler-Free Interleaved Sequence Fusion (S2d)
S2d removes the learned-query resampler and sends visual tokens to the forecaster through a position-wise projector (@fig-s2d). Its encoder--projector--forecaster layout follows Nemotron 3 Nano Omni @nemotron, which combines an MLP projector with spatial token reduction and Efficient Video Sampling (EVS) @evs. The visual encoder produces $4 times 196 = 784$ tokens, while the numerical input contains 42 macro-patches. Processing all visual tokens would spend much of the attention budget on the static sky in the satellite crop.

The visual field is first reduced spatially. The $14 times 14$ patch grid is divided into non-overlapping $2 times 2$ neighborhoods, producing a $7 times 7$ field of 49 cells with 1024 channels per cell. This reduction preserves the spatial layout of the field while lowering the number of visual tokens passed to the forecaster. The 49 cells are then projected independently, so each location remains addressable by the subsequent attention layers.

Each 1024-channel cell is mapped to the forecaster width with a two-layer GELU MLP ($1024 arrow.r 768$). The MLP processes cells independently, so it does not mix information across locations. A learned table of 49 coordinate-indexed vectors then adds the spatial position of each cell.

The position encoding separates space from time. The learned table represents spatial coordinates, while sequence position represents the time of each frame. S2d does not add a temporal-slice embedding because it would duplicate this information and make the frame-order control in @sec-instrument harder to interpret. After temporal shuffling, the visual tokens receive new sequence positions; an embedding attached to each slice could still expose its original order.

EVS selects the visual tokens that changed most between consecutive frames. For each spatial cell, it computes the cosine dissimilarity between a latent slice and the corresponding cell in the preceding slice. High dissimilarity identifies changing regions, while low dissimilarity identifies static content. The first slice receives a score of $+infinity$, ensuring that one complete anchor field is retained. EVS ranks the $4 times 49 = 196$ candidates globally and keeps the top 98 at $q = 0.5$, restoring frame-then-cell order afterwards. The resulting visual sequence contains the 49-cell anchor field plus 49 tokens from the most dynamic regions. This fixed budget can therefore retain a localized cloud front without processing the full $128 times 128$ km field.

Nemotron uses EVS after training as a deployment-time throughput control and tunes $q$ for each deployment @nemotron. S2d applies the same parameter-free selection during training and evaluation. Thus, $q$ is fixed as part of the architecture, and the forecaster always receives 98 visual tokens. The same selection rule also makes the vision-off evaluation in @sec-instrument resemble an input pattern seen during training. Since EVS has no learned parameters, the random-selection control in @tbl-controls can replace its ranking rule on a trained checkpoint while keeping the sequence length unchanged. This isolates the selection criterion from the token budget.

The selected visual tokens are inserted into the numerical sequence according to their observation times. One numeric patch spans $16 times 30$ minutes, or 8 hours, whereas the visual window spans 6 hours within the final context patch.

The final sequence has 42 numeric macro-patches, 98 visual tokens, and one future query, for 141 positions in total. Native bidirectional self-attention processes all positions together. Nemotron places visual tokens in a prefix block; S2d places each token at its timestamp within the numerical sequence. No cross-attention layer, gate, or auxiliary channel separates the modalities. The future query can attend to visual and numeric tokens through the same attention mechanism. Because the visual window ends immediately before the forecast horizon, these tokens describe the sky observed before the prediction.

During training, S2d drops the visual stream in half of the samples and the numeric stream in one tenth of the samples, never dropping both together. The visual dropout makes the vision-off pass used by @sec-instrument familiar to the model. Consequently, the measured difference between the active and disabled passes reflects the missing visual information without also introducing an unseen input distribution.



#figure(
  image("figures/mmtsfm_s2d.png", width: 95%),
  caption: [Resampler-Free Interleaved Fusion (S2d)],
) <fig-s2d>

=== Paradigm 3: Multi-Anchor Interleaved Sequence Fusion (S2e)

S2e spreads S2d's visual context over five Chronos-2 patches (@fig-s2e). Each anchor samples the same clock time on one of five consecutive days, so the model sees several sky--power pairs at matched solar geometry. The daily stride is imposed by coverage: UK satellite frames run from 02:00 to 16:00 UTC, and an 8-hour ladder fills all five anchors for none of the 24,605 evaluated origins, compared with $94.4%$ for a 24-hour ladder.

S2e retains the five-anchor layout, 143-token sequence length, warm start, and optimizer of the initial configuration. Each anchor ranks its 98 candidate visual tokens using paired novelty: the method scores 49 two-latent spatial trajectories by cosine change and keeps both endpoints for the ten highest-scoring trajectories. This produces 20 visual tokens per anchor, or 100 visual tokens in total, together with 42 numeric tokens and one future query. Each visual block takes the integer position of its anchor patch, whereas S2d uses fractional positions inside the final patch.

If S2e performs well, the next experiment will test whether each anchor should retain all 98 candidates. This would produce 490 visual tokens overall and will show whether the additional token budget improves the multi-anchor model.

#figure(
  image("figures/mmtsfm_s2e.png", width: 95%),
  caption: [Canonical Multi-Anchor Interleaving (S2e)],
) <fig-s2e>

= The Counterfactual Reliance Instrument <sec-instrument>

The *Counterfactual Reliance Metric* measures visual utility without changing model capacity or retraining the ablated system:

$ "Reliance"_"ramp" = "NMAE"_"ramp"^(bold(V)=emptyset) - "NMAE"_"ramp"^(bold(V)="active") $

After training, the parameters $bold(theta)$ are frozen. We score the test set with vision active ($bold(V)="active"$) and then disable the visual pathway in the same model ($bold(V)=emptyset$). A positive difference indicates that the frozen model's accuracy depends on the imagery, since the weights and all other inputs remain unchanged.

= Results and Analysis

== The Fusion Ladder: Comparing Visual Reliance

@tbl-ladder compares nominal accuracy with counterfactual visual reliance across the three trained arms.

#tbl(
  [Empirical evaluation of visual reliance across fusion paradigms. Reliance is the reduction in ramp error from having observed the sky, measured counterfactually on frozen weights across seeds 42--44. Bottlenecked pooling yields zero reliance, whereas the resampler-free arm clears the significance floor by more than six times.],
  columns: (auto, 1.2fr, auto, auto, auto),
  align: (left, left, center, center, center),
  table.header[Arm][Architectural Paradigm][Skill Score ($arrow.t$)][Ramp NMAE ($arrow.b$)][Ramp Reliance ($arrow.t$)],
  [S1], [Unimodal Foundation Baseline], [52.30 #text(size: 8pt)[$plus.minus 0.41$]], [0.1506 #text(size: 8pt)[$plus.minus 0.0010$]], [---],
  [S2a], [Bottlenecked Late Fusion (Resampler)], [52.58 #text(size: 8pt)[$plus.minus 0.43$]], [0.1487 #text(size: 8pt)[$plus.minus 0.0010$]], [0.0000 #text(size: 8pt)[$plus.minus 0.0015$]],
  [*S2d*], [*Resampler-Free Interleaved Fusion*], [*55.64* #text(size: 8pt)[$plus.minus 0.18$]], [*0.1435* #text(size: 8pt)[$plus.minus 0.0004$]], [*0.0073* #text(size: 8pt)[$plus.minus 0.0002$]],
) <tbl-ladder>

S2a reduces nominal ramp NMAE from $0.1506$ to $0.1487$, yet its counterfactual reliance is $0.0000$: disabling vision in the frozen model does not change performance. The nominal gain therefore arises from multimodal training rather than visual information used at inference. The pooled vector contributes no measurable signal after the resampler removes its spatial structure.

S2d reaches a ramp reliance of $0.0073$, more than six times the pre-registered significance floor ($0.0011$) across all seeds. Its 49 separate cells allow the foundation transformer to query localized visual boundaries directly and produce the best ramp accuracy in the suite. Because S2a and S2d differ in resampling, token count, and token placement, this comparison attributes the result only to the complete design.

== Mechanistic Diagnostics and Falsification Controls

#tbl(
  [Diagnostic falsification controls on the resampler-free interleaved arm.],
  columns: (auto, 1.2fr, 1.2fr),
  align: (center, left, left),
  table.header[\#][Diagnostic Hypothesis Tested][Empirical Finding and Mechanistic Verdict],
  [1], [Does S2d rely on temporal frame ordering? (Kinematic frame shuffle)],
  [Near-inert: $Delta "NMAE"_"ramp" approx 0$ across all seeds ($< 0.0003$). Refutes the hypothesis that the model tracks kinematic cloud trajectories.],
  [2], [Does micro-temporal sub-patch positioning matter? (Collapsing sub-patch indices)],
  [Null effect: Ramp NMAE change ($+0.0008$; reliance is unaffected ($0.0064$ vs $0.0063$). Temporal indexing within the patch is invariant.],
  [3], [Is a recent sky sufficient? (Stale sky: imagery lagged by 6 hours)],
  [Severely degrades accuracy: Costs $+0.0169$ Ramp NMAE ($+11.7%$ error). Confirms strict temporal freshness.],
  [4], [Does the model resolve site-specific sky? (Contemporaneous cross-plant donor sky)],
  [Catastrophic degradation: Costs $+0.0176$ Ramp NMAE; Skill Score collapses from $55.10$ to $37.00$. Confirms local spatial grounding over the installation.],
  [5], [Does dynamic novelty pruning carry informational signal? (Uniform random token selection at equal budget)],
  [Degrades performance: Ramp NMAE worsens by $+0.0039$; Skill Score drops by $-1.52$ percentage points. Confirms that novelty scoring extracts predictive cloud boundaries.],
) <tbl-controls>


== The Intrinsic Calibration Cost

S2d improves point-forecast ramp accuracy but changes probabilistic calibration:

#tbl(
  [Probabilistic calibration metrics across foundation model configurations.],
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  table.header[Configuration][Ramp NMAE ($arrow.b$)][80% Prediction Interval Coverage][Quantile ECE ($arrow.b$)],
  [S1 (Unimodal Control)], [0.1506], [0.772], [0.0297],
  [*S2d (Resampler-Free)*], [*0.1435*], [0.753], [0.0314],
) <tbl-calibration>


= Comprehensive Benchmark Leaderboard

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
  [24], [Aurora @aurora], [FT], [Joint multimodal foundational pretraining], [23.24], [---],
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


#v(0.8em)
#bibliography("refs.bib", title: "References", style: "ieee")
