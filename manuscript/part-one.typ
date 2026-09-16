#heading(level: 1, numbering: none)[Abstract]

Photovoltaic forecasting is usually evaluated by withholding later observations from plants a model has already seen. That setup does not answer whether a forecasting system can work for a newly connected plant after a short operating history. This thesis studies that transfer question by holding out complete installations from training.

The first part introduces the research context, reviews time-series and visual foundation models, and documents the multimodal corpus used by the study. The corpus combines power, weather, solar-geometry, and satellite observations from 110 photovoltaic systems in the United Kingdom and the United States. The primary UK evaluation uses 14 held-out plants and 165,295 valid daylight forecast steps. The later parts of the thesis will develop the forecasting methods and report their results.

= Introduction
<cha:intro>

== Forecasting an unseen plant

Photovoltaic generation has a recognizable daily outline and a difficult short-term residual. Solar geometry sets the broad arc of production. Cloud cover, local weather, and the characteristics of individual arrays determine how actual output departs from that arc. A model trained on years of history from one installation can absorb its orientation, shading pattern, and relationship with nearby weather observations. A fleet operator must also forecast installations that have only recently entered service and have no comparable archive.

This thesis studies cross-plant forecasting. Training, validation, and test sets contain different installations. A test plant contributes neither fitting examples nor statistics derived from its own measurements. At inference time the model receives fourteen days of recent context, yet it has never encountered the plant during parameter fitting. The task asks a shared model to infer what matters about an unfamiliar installation from its short history and from relationships learned across the training fleet.

This split changes the meaning of the reported error. A conventional temporal split measures how well a model extrapolates a familiar plant through time. It rewards stable facts about that site, including its capacity factor and systematic bias relative to a weather grid. Cross-plant evaluation asks whether the representation survives the change of entity. Capacity, local weather, data availability, and the relationship between irradiance and output all vary across plants. Separating installations exposes models that depend on information unavailable for a new asset.

== Why vision is a hypothesis

The numerical history at a plant records conditions that have already reached the array. It cannot show a cloud edge approaching from outside the local footprint. A satellite sequence may provide that spatial observation before the resulting occlusion appears in the power trace. Weather covariates add regional context, but they do not provide the same local visual record.

The availability of an image does not establish that a forecasting model uses it. A crop may contain cloud structure, sensor artifacts, coastlines, and features unrelated to the array. More subtly, a multimodal architecture can score better after visual training because the training process changes its optimization, regularization, or trainable capacity. Comparing a numerical model with a separately trained visual model cannot distinguish these explanations.

The thesis therefore treats visual contribution as a counterfactual question. After an arm has been trained, the same weights are scored once with its visual pathway active and once with vision disabled. The marginal change in ramp error measures what the trained forecast actually gains from the images. A score improvement with a negligible counterfactual change is evidence about training, not evidence that the model learned to see. This distinction guides the fusion ladder developed later in the thesis.

== Research question and hypotheses

The research question is whether a time-series foundation model and a visual foundation model can forecast power at disjoint plants through a learned multimodal interface. The work examines transfer, fusion, and the evidential standard required to claim a visual contribution. Photovoltaic power is the test domain because cloud-driven ramps provide a visible source of uncertainty and because plant-disjoint evaluation gives transfer a clear operational meaning.

The investigation is organized around five hypotheses. First, time-series foundation models require in-domain adaptation for this task. Second, visual information should improve a numerical foundation forecaster only when the trained model uses it at inference time. Third, the depth and placement of the fusion interface affect that reliance. Fourth, a plant-disjoint split is sufficiently demanding to discriminate between competing transfer strategies. Fifth, retrieval and lightweight adaptation offer credible alternatives that any multimodal approach must be compared against when their evaluations are protocol aligned.

The hypotheses are deliberately separable. A visual interface can fail to use imagery even if its aggregate score improves. A deep interface can preserve imagery while still losing to a strong numerical forecaster. A model can transfer across plants in one fleet without generalizing across countries. Stating these possibilities in advance prevents the later results from being compressed into one broad claim about multimodality.

== Contributions

This thesis contributes a multimodal cross-plant corpus that combines 1,337,654 numerical rows from 110 sites with 4,103,892 timestamp-addressable satellite frames. It provides a common dataset contract for numerical and visual forecasters, preserves native cadences, and records quality and missingness explicitly.

It also develops a single-protocol benchmark around plant-disjoint evaluation. The benchmark compares reference methods, tabular learners, supervised sequence models, time-series foundation models, retrieval approaches, and multimodal systems under shared windows and metrics. The later chapters explain which comparisons meet the full protocol and which results remain limited by their original implementation assumptions.

The methodological contribution is a fusion ladder with a numerical control, a conventional late-fusion interface, and a multi-anchor interleaved interface. The arms share their numerical backbone and visual encoder policy. Each visual arm is evaluated with a forced vision-off pass. This creates an experimental route from nominal forecast score to evidence about visual reliance.

== Scope and boundaries

The thesis studies power forecasting rather than irradiance reconstruction. Learned models do not receive hand-coded clear-sky or clearness-index transformations as internal modeling steps. The work does not address electricity markets, dispatch optimization, or plant control. These problems motivate the value of accurate forecasts, but they require different data and decision models.

The main evaluation is cross-plant transfer within the UK track. The United States track is part of the corpus and prepares a later cross-dataset experiment, but it does not support the headline conclusions until its quality-controlled split is regenerated. The thesis also distinguishes implemented evidence from design options. A mechanism discussed as a possible temporal operator is not presented as an empirical contribution unless it has been run under the protocol.

== Thesis structure

Part I defines the transfer problem, positions it in the relevant literature, and documents the dataset. Part II specifies the evaluation protocol and the benchmark suite. Part III introduces the fusion methods, reports the ablations, and discusses what the evidence supports. The conclusion returns to the limits of visual transfer and the next experiments required to test it beyond the current fleet.

= Background and related work
<cha:related>

== From plant-specific accuracy to transferable forecasting

Photovoltaic forecasting has traditionally been organized around the history of one installation. The model learns the recurring relationship between past output, weather inputs, and future output for that site. This setting is useful for operations at a well-instrumented asset, yet it leaves a practical gap when a fleet gains a new plant or when historical records are sparse.

Cross-plant forecasting changes the target of generalization. The held-out object is an installation with its own capacity, local weather exposure, equipment, and missing-data pattern. A temporal split can reward learning the stable details of a known site. A plant-disjoint split asks the model to recover the relevant local context from a short history while relying on representations learned elsewhere.

This distinction applies to preprocessing as well as to model fitting. Per-series normalization requires statistics from the held-out series. Capacity normalization uses information known when an installation is commissioned and preserves the transfer setting. The resulting task sits between two familiar extremes: it is neither a conventional within-series forecast nor a metadata-only cold start.

== Numerical forecasting models

The benchmark begins with methods that make different assumptions about temporal structure. Persistence, Smart Persistence, climatology, and seasonal naive forecasts establish how much of the target can be explained by the current state and the daily cycle. They matter because photovoltaic power has a strong astronomical envelope. A method that improves only on this envelope may appear accurate while missing the difficult cloud-driven changes.

Tabular models such as gradient-boosted trees, TabPFN, and TabFM treat lagged power and covariates as features. They test whether a structured feature table already captures the useful information. Supervised sequence models, including DLinear, PatchTST, the Temporal Fusion Transformer, and iTransformer~@itransformer, test different temporal and covariate inductive biases without relying on broad time-series pretraining.

iTransformer is particularly relevant because it treats variables as tokens and models their relationships through attention. In a photovoltaic task, power, solar geometry, cloud cover, radiation variables, and temperature have strong shared structure. A model that receives operationally available covariates may forecast through this numerical context without any visual input. It is therefore an important comparator for a multimodal claim.

== Time-series foundation models

Time-series foundation models are pretrained on many series before zero-shot use or task adaptation. Chronos-2~@chronos2, TimesFM~@timesfm, TiRex~@tirex, and Tiny Time Mixer~@ttm cover several architectural families. Their shared promise is transfer: a model exposed to a broad corpus may bring useful temporal representations to a series it has never seen.

Their treatment of the input differs. Patch-based models compress a long context into blocks of observations. This makes long histories tractable, but it can also blur rapid behavior inside a patch. Covariate support differs across model families, which matters for a task where solar and weather variables carry substantial predictive information. Output representations differ as well, from point forecasts to quantile distributions.

Pretraining does not remove the need for a controlled comparison. Photovoltaic power has daily zeros, seasonality, cloud-driven ramps, and plant-to-plant variation that may not be represented in a general pretraining corpus. A zero-shot result measures the match between prior knowledge and this task. A fine-tuned result measures both that prior and the information obtained from the training fleet.

== Adaptation without a new model

Several methods adapt a pretrained time-series model without fully retraining it. Fine-tuning updates some or all backbone weights on the target fleet. Retrieval methods such as TS-RAG~@tsrag and Cross-RAG~@crossrag condition a frozen forecaster on similar historical windows. Covariate adapters offer a third route by injecting external inputs into an established representation.

These approaches are direct alternatives to visual fusion. Retrieval asks whether analogous numerical histories provide the missing context. A covariate-aware adapter asks whether an external signal can be supplied through a compact residual path. The comparison is only meaningful when the retrieved datastore excludes held-out plants and when all methods use the same forecast windows.

The adaptation literature also clarifies what a multimodal result must show. If a late-fusion adapter and a token-level visual interface receive the same encoder features, their difference concerns how information is represented and routed. It does not demonstrate that vision in general is useful unless the trained model is tested with that input removed.

== Video foundation models as feature extractors

Cloud structure is spatial and evolves through time. A video encoder can represent a short image sequence as a field of spatiotemporal features rather than as a set of independent frames. V-JEPA 2.1~@vjepa2 is used in this thesis as a self-supervised visual feature extractor. Its role is to provide a representation of the satellite clip, not to supply a photovoltaic forecast directly.

Freezing the video encoder makes the experiment easier to interpret. It prevents a large visual network from adapting freely to the target data and focuses attention on the bridge between visual and numerical representations. It also creates a practical constraint: visual latents can be cached before training, which makes repeated seeds feasible but limits experiments that would require end-to-end visual fine-tuning.

The central problem is token budget. A video encoder produces many spatial patches, while a time-series forecaster often has a comparatively short sequence of numerical tokens. Passing every visual patch forward is expensive and may devote most of the attention budget to static regions. Pooling all patches into one vector avoids this cost but discards spatial identity. Token selection occupies the middle ground by preserving a bounded set of addressable visual regions.

== Multimodal forecasting interfaces

The interface between a visual encoder and a forecaster determines what information can survive from the image sequence. A satellite crop may begin as hundreds of spatial features across several frames. The forecast head, in contrast, often receives a short sequence of numerical tokens. Most published systems therefore compress the visual stream before it reaches the numerical model. This is a sensible response to computational cost, but it also creates an information bottleneck whose effect depends on the forecasting problem.

=== Solar-VLM: gated late fusion

Solar-VLM~@solarvlm is a useful example of gated late fusion. It encodes satellite crops and meteorological text with a frozen vision-language model, pools the encoded inputs through cross-attention into a station-level vector, and concatenates that vector with the time-series state. A learned scalar gate then controls the contribution of the visual branch. This design gives the forecaster a compact description of broad cloud conditions and allows it to fall back on numerical history when images are unhelpful. Its limitation is structural: after a large scene has been reduced to one vector, the forecasting model cannot inspect a particular cloud edge or distinguish two regions that had opposite effects on the pooled representation. The gate may also prefer the numerical branch when the pooled image summary overlaps with the cloud and radiation variables already supplied as covariates.

Late fusion remains a necessary baseline because it tests a clear hypothesis: global visual context may be sufficient when the relevant visual evidence is widespread and slowly changing. It is especially valuable here because it fixes the same visual source while changing the way it is introduced to the forecaster. The later comparison uses late fusion as the compact-interface control, not as a proxy for every multimodal system.

#figure(
  image("figures/related_solar_vlm_original.png", width: 94%),
  caption: [Solar-VLM architecture. Satellite crops and meteorological text are encoded, pooled into a station representation, and combined with the time-series branch through a learned gate. Reproduced from the source architecture~@solarvlm.]
) <fig-related-solar-vlm>

=== UniCast: multimodal prompting of a time-series backbone

Prompting methods offer a related route. UniCast~@unicast supplies visual and textual information to a time-series foundation model through a lightweight prompt representation. The approach asks whether a pretrained numerical backbone can accept a new modality without broad retraining. It is attractive when the new input is primarily contextual, since a small prompt leaves the main forecasting representation intact. In a cloud-driven setting, however, a single prompt has the same practical question as pooled fusion: whether the compression retains the local motion and spatial arrangement needed to relate the current sky to a future change in generation.

=== CrossViVit: cross-attention over visual and temporal features

Token-level approaches retain several visual observations and let the forecasting backbone relate them to numerical tokens. CrossViVit~@crossvivit is the central photovoltaic reference in this group. It uses cross-attention between time-series features and visual features, so the model can form different connections for different temporal and visual locations. This is closer to the structure of the physical problem than a single scene vector: the output change at a plant can depend on the location and movement of a cloud, not only on average cloudiness across the crop. The present study examines a narrower design question within this family, namely how a bounded selection of visual tokens can be placed in temporal context while retaining a manageable attention budget.

=== SUNSET: a photovoltaic convolutional reference

SUNSET~@sunset provides a different point of reference. It is a specialized convolutional system for power and satellite data rather than a composition of pretrained foundation models. Its inclusion asks a useful question for the benchmark: can a foundation-model interface improve on an architecture designed directly for solar forecasting? The answer bears on the proposed transfer setting, but a comparison with SUNSET alone cannot distinguish the contribution of pretraining, the fusion interface, and the training protocol. Those factors require the broader set of controls described in this chapter.

=== Nemotron 3 Nano Omni: preserving a selected visual field

Outside photovoltaic forecasting, Nemotron 3 Nano Omni~@nemotron illustrates a practical alternative to global pooling. Its visual pathway aligns visual features with a language-model backbone after reducing their number in two stages: spatial downsampling followed by selection of tokens from regions that change over time. Efficient Video Sampling~@evs supplies the latter idea by pruning temporally redundant video tokens. The application and decoder differ from the present task, yet the design principle transfers directly. A visual feature grid can be compressed while retaining the identity of selected locations. This is useful when a satellite sequence contains large static areas alongside a smaller number of evolving cloud structures.

#figure(
  image("figures/related_nemotron_visual_path.png", width: 92%),
  caption: [Nemotron 3 Nano Omni visual pathway. The visual encoder produces tokens that are spatially reduced and then sampled before projection into the shared backbone. Reproduced from the source architecture~@nemotron.]
) <fig-related-nemotron>

== Real images, synthetic images, and multimodal pretraining

=== Time-VLM and VisionTS++: rendered time-series images

Time-VLM~@timevlm and VisionTS++~@visiontspp take a different view of multimodality. They render the numerical history as an image, then encode that rendering with a visual backbone. Time-VLM also uses statistical text prompts to describe the series. The resulting visual stream may regularize the model or expose a temporal pattern to an encoder trained on images. It remains a transformation of information already present in the power history. It cannot reveal a cloud that has not yet affected the array.

This distinction matters when interpreting a benchmark. A pseudo-image model can be a strong forecasting baseline because it tests a legitimate representation choice. It evaluates the value of re-encoding a history rather than the value of an independently observed modality. Real satellite frames and rendered time-series images answer different questions: whether a visual encoder can re-describe a history, and whether an image provides information outside that history.

#figure(
  image("figures/related_time_vlm_original.png", width: 94%),
  caption: [Time-VLM architecture. Historical time-series values are rendered as an image and combined with statistical text prompts before forecasting. Reproduced from the source architecture~@timevlm.]
) <fig-related-time-vlm>

=== Aurora: pretraining across derived modalities

Aurora~@aurora extends this line of work to multimodal foundation-model pretraining. It combines modality-specific encoders with a cross-modality encoder that uses token distillation and modality-guided self-attention, training on numerical series and derived image and text representations. The work is relevant because it tests whether several representations of a series can improve general forecasting behavior and transfer. It also shows that multimodal architecture can be valuable even where every modality originates in the same underlying signal.

The present task imposes an additional evidential requirement. Satellite imagery is an independent observation of the atmosphere, and a forecast improvement can only be attributed to that observation if the trained model changes when the image is withdrawn. This is why the thesis does not treat multimodal training alone as proof of visual use. The same parameters are evaluated with the visual pathway active and disabled; the difference isolates the contribution of visual information after training.

#figure(
  image("figures/related_aurora_original.png", width: 94%),
  caption: [Aurora architecture. Modality-specific encoders feed a cross-modality encoder that uses token distillation and modality-guided self-attention. Reproduced from the source architecture~@aurora.]
) <fig-related-aurora>

=== iTransformer: variables as tokens

iTransformer~@itransformer is an equally important numerical comparator because it represents each variable as a token across the input window and applies attention across variables. In photovoltaic forecasting, generation, solar position, cloud cover, irradiance, temperature, and wind share substantial structure. When future-aligned weather forecasts are supplied, their timing differs from the satellite stream: weather variables may describe the forecast horizon directly, whereas the images describe conditions observed up to the forecast origin. A multimodal model must learn how the motion and shape of the observed cloud field relate to later occlusion. The iTransformer comparison therefore tests whether the visual pathway adds information beyond a well-structured numerical representation with operational covariates.

#figure(
  image("figures/related_itransformer_original.png", width: 94%),
  caption: [iTransformer architecture. Each variable is embedded as a token spanning the input window, and self-attention models relations across variables. Reproduced from the source architecture~@itransformer.]
) <fig-related-itransformer>

These comparisons motivate the counterfactual reliance protocol used later in the thesis. An architecture can receive images and still base its forecast entirely on numerical history. Scoring the same trained parameters with active and disabled vision distinguishes a nominal multimodal score from evidence that the visual pathway contributes to the prediction.

== Position of the present study

The related work leaves three linked questions. Can a numerical foundation model transfer across disjoint plants? Does an external visual observation add information beyond power history and covariates? Which interface preserves that information without making the visual stream prohibitively expensive?

The present study addresses these questions with a common cross-plant corpus and a benchmark that includes the relevant numerical, foundation, adaptation, and multimodal model families. The methodological comparison is deliberately narrower than a survey of all solar architectures. It contrasts a vision-free numerical control, a pooled late-fusion interface, and an interleaved token-level interface under matched backbone and training policies. The later chapters report the outcome; this chapter identifies the alternatives that outcome must distinguish.

= Dataset construction
<cha:dataset>

== Design requirements

The corpus was constructed for a transfer experiment rather than a single-site forecast. It must preserve plant identity, native cadence, quality information, and the timing between numerical and visual observations. It must also give simple baselines, pretrained models, and multimodal systems the same data contract.

The construction follows four principles. The corpus contains more than one geographic and operational setting. Numerical records and image timestamps remain aligned. Data quality is represented with explicit flags instead of silent repairs. Finally, the storage layout permits reproducible windows without each model inventing its own treatment of missing power values or missing frames.

The dataset has two parts. A flat Parquet table contains 1,337,654 numerical records and 35 columns. A packed HDF5 archive contains 4,103,892 PNG-encoded frames in 110 site groups. The separation keeps numerical data easy to process while preserving timestamp-exact access to large visual payloads.

== The United Kingdom and United States tracks

The United Kingdom track contains 100 residential photovoltaic systems from 1 January 2019 through 31 December 2020. Power is sampled every 30 minutes. The systems have capacities from 1.5 to 4.0 kW and span latitudes 50.7 to 57.8. Satellite frames are available every 15 minutes.

The United States track combines ten NREL PVDAQ systems with GOES imagery. It covers January through September 2019 at 15-minute cadence. Capacity ranges from 1.8 to 408 kW. This spread introduces a different scale of deployment, a different sensor product, and a different geographic context. It is useful for later distribution-shift experiments, even though the current headline evaluation uses the UK track.

== Position among published photovoltaic datasets

The corpus differs from existing published datasets in the combination of its modalities and its evaluation design. HKUST Rooftop PV provides high-frequency inverter-level power and on-site weather observations for 60 rooftop systems, but it does not include satellite imagery or a disjoint-plant benchmark. PVOD provides metadata and meteorological information for ten Chinese plants, again without imagery or a held-out-plant protocol. SKIPP'D pairs one California system with minute-resolution sky imagery, making it useful for visual nowcasting at a single site but unable to test transfer across installations.

FusionSF's MMSP dataset~@fusionsf is the closest published comparison. It combines capacity-normalized power, Himawari imagery, and numerical weather data for 88 plants and evaluates zero-shot prediction on unseen plants. The present corpus extends the available comparison setting in three concrete ways: it has two independently constructed geographic tracks; it preserves the native cadence of each source rather than forcing a single grid; and it stores timestamp-exact visual references inside a common numerical contract. Those properties allow the same cross-plant task to be instantiated with UK residential systems and, after split reconciliation, with US systems that span a much wider capacity range.

#figure(
  align(center)[#table(
    columns: 6,
    align: (left, center, center, left, left, left),
    table.header([Dataset], [PV systems], [Cadence], [Plant information], [Environmental context], [Evaluation scope]),
    table.hline(),
    [HKUST Rooftop PV], [60], [5 min power], [Inverter power and equipment metadata], [On-site weather], [No imagery or disjoint-plant benchmark],
    [MMSP in FusionSF~@fusionsf], [88], [1 h], [Capacity-normalized power and anonymized locations], [Himawari imagery and numerical weather], [Zero-shot unseen plants],
    [SKIPP'D], [1], [1 min], [Power, capacity, tilt, and azimuth], [Ground-based sky images], [Chronological single-plant benchmark],
    [PVOD], [10], [15 min], [Power, capacity, orientation, and location], [Local weather and numerical forecasts], [No imagery or disjoint-plant benchmark],
    [This corpus], [110], [15 or 30 min], [Power, capacity, location, and quality flags], [Satellite frames, atmospheric covariates, solar geometry], [Plant-disjoint UK benchmark; US extension prepared],
  )],
  caption: [Comparison with published photovoltaic datasets relevant to multimodal and cross-plant forecasting.]
  , kind: table
)

The novelty is therefore not simply the addition of images to a power table. The dataset makes three observations jointly available at a forecast origin: a plant's recent generation, atmospheric and astronomical context, and a visual record of the nearby cloud field. It also preserves the information needed to audit their relation, including source identity, UTC timestamps, site coordinates, quality flags, and the local frame index. This combination supports controlled multimodal experiments in which a model's visual input, numerical input, and target window can be traced back to the same plant and time.

#figure(
  align(center)[#table(
    columns: 7,
    align: (left, center, center, center, center, center, left),
    table.header([Track], [Sites], [Rows], [Valid power steps], [Power cadence], [Frame cadence], [Capacity]),
    table.hline(),
    [UK], [100], [1,232,862], [1,217,399], [30 min], [15 min], [1.5 to 4.0 kW],
    [United States], [10], [104,792], [103,451], [15 min], [15 min], [1.8 to 408 kW],
  )],
  caption: [The two data tracks. Native cadences are retained.]
  , kind: table
)

The tracks should not be pooled casually. A twelve-step horizon represents six hours in the UK and three hours in the United States. The common corpus allows windows to be defined in physical time and resolved at native cadence. It does not erase differences between the underlying processes.

#figure(
  image("figures/week_uk_pv.png", width: 92%),
  caption: [A representative week from the UK track. The daily envelope is clear, while departures from it vary sharply from one day to the next.]
)

#figure(
  image("figures/week_goes_pvdaq.png", width: 92%),
  caption: [A representative week from the United States track. The different cadence, capacity range, and data span make this a complementary transfer setting.]
)

== Numerical observations

Each numerical record identifies a dataset, site, timestamp, geographical location, and station or camera identifiers when they are available. It stores power in watts and normalized power, calculated by dividing measured power by audited installed capacity. Outage and stuck-signal observations retain their quality labels and have missing normalized targets rather than fabricated values.

Weather covariates include temperature, shortwave radiation, direct and diffuse radiation, direct normal irradiance, cloud cover, wind speed, and precipitation. Solar and calendar fields include zenith, azimuth, a clear-sky estimate, seasonal sine and cosine terms, and solar time. These inputs have different availability at prediction time. Calendar and solar geometry are known ahead. Measured weather is an observation and must be treated according to the evaluation protocol.

#figure(
  align(center)[#table(
    columns: 2,
    align: (left, left),
    table.header([Record component], [Contents]),
    table.hline(),
    [Identity], [Dataset, site, UTC time, location, station and camera identifiers],
    [Target], [Measured power and capacity-normalized power],
    [Atmospheric inputs], [Temperature, radiation components, cloud cover, wind, precipitation],
    [Solar and calendar fields], [Solar geometry, clear-sky estimate, seasonal and solar-time terms],
    [Quality information], [Outage, stuck-value, night, capacity, and site-quality flags],
    [Visual reference], [Timestamp-exact index into the image archive],
  )],
  caption: [Logical structure of a numerical record.]
  , kind: table
)

The native time grids are preserved. Gaps are not interpolated to create a superficially regular series. Window construction uses masks to decide whether a history and its future target segment contain enough observations to be scored. This keeps missingness visible to the model and to the evaluation.

The numerical table is deliberately richer than a target-and-weather benchmark. It records the provenance needed to reconstruct a sample, the physical scale needed to normalize it, the quality state needed to decide whether it is valid, and the image reference needed to recover the visual context. This makes the corpus usable for a simple numerical baseline and for a multimodal model without creating two incompatible versions of the same forecast task.

== Image archive and alignment

The HDF5 archive stores one image and timestamp array for each site. UK frames decode to 128 by 128 RGB images; US frames decode to 256 by 256 RGB images. Both tracks have a 15-minute image cadence. A timestamp-exact index in the numerical table points to the local frame position within the appropriate site group.

Forecasting needs more than one image associated with one table row. The loader therefore builds its visual clip from the archive timestamps. This is important when frames are denser than power observations, as they are in the UK track. It also makes a visual window an explicit sequence selected around the forecast origin rather than an incidental attachment to a numerical row.

The current UK archive has three distinct non-HRV channels. It is not a grayscale image copied into three channels. The archive also retains structured infrared information before daylight. This change is important because an earlier daylight-only image source would have left many morning forecasts without meaningful visual input.

Frame coverage is still diurnal. UK images are available from 02:00 to 16:00 UTC and US images from 10:00 to 23:00 UTC. A model cannot treat a missing frame as a clear-sky frame. The data contract therefore exposes a visual validity mask and allows a model to recognize that no visual evidence is available.

== Visual coverage in the primary UK windows

The primary UK loader selects eight recent frames, spaced by 45 minutes with a tolerance of 22.5 minutes. Rebuilding the test windows gives 165,295 valid scored steps. The evaluation stride and daylight rule yield two major origin times.

#figure(
  align(center)[#table(
    columns: 5,
    align: (center, center, center, center, center),
    table.header([Origin], [Windows], [Mean frames filled], [Zero-frame windows], [Share of scored steps]),
    table.hline(),
    [07:30 UTC], [10,004], [7.68 of 8], [3.9%], [70.7%],
    [13:30 UTC], [9,960], [7.67 of 8], [3.8%], [29.3%],
  )],
  caption: [Coverage of the visual input in primary UK test windows.]
  , kind: table
)

Overall, 96.1 percent of scored windows have all eight selected frames and 3.7 percent of scored steps have no visual input. This rules out a simple explanation in which visual models are tested only on a small, unusually complete subset. It also sets a limit: a robust forecaster must still handle a small share of windows using numerical context alone.

Frame differences characterize the available visual signal. Mean absolute change is 8.3 digital numbers over 15 minutes and 35.3 over two hours, while the average within-frame standard deviation is 38.6. The observed cloud state therefore changes substantially over a few hours. That supports a short visual context for an intra-day forecast and cautions against claims that a single observed clip remains equally informative for a day-ahead horizon.

== Quality control

Quality flags remain part of the record. Two UK sites and two US sites are marked as bad sites. The corpus also identifies 15,486 outage observations, 1,318 stuck readings, and 1,535 night-clamped rows. These are not interchangeable forms of low output. An outage is not cloudy weather, a stuck reading is not persistence, and a night-time zero should not dominate a daylight metric.

The primary UK partition excludes its two bad sites before splitting plants. The current United States split predates the new bad-site flags and still includes them in training or validation assignments. The US track is therefore documented as a component of the corpus but is not used for the headline experiment until its split is regenerated.

The standard dataset does not fill missing frames with a mean image or interpolate missing power values. Those operations may be studied as explicit robustness choices, but they would change what has been observed. The common contract preserves masks and flags so every experiment can state how it treats missing data.

#figure(
  image("figures/site_availability_timeline.png", width: 94%),
  caption: [Availability across UK sites. The corpus keeps discontinuities visible so that window construction can exclude or mask invalid observations.]
)

== Dataset characteristics

Normalized power is zero-inflated because generation is zero at night and close to zero near sunrise and sunset. During the day it is right-skewed and bounded by installed capacity. Percentage error metrics are poorly suited to this distribution because very small observed values make small absolute deviations appear arbitrarily large.

The mean daily profile is smooth. The spread around it is not. At a fixed solar time, cloud conditions can move normalized output from near zero to a large fraction of capacity. The difficult part of the forecasting task is therefore not reproducing the daily mean, but accounting for departures from it. A strong reference must account for predictable solar geometry without being mistaken for a solution to cloud-driven uncertainty.

#figure(
  image("figures/diurnal_profile.png", width: 90%),
  caption: [Mean normalized power and its spread through the day. The broad envelope is predictable; the spread around it is the forecasting problem.]
)

Step-to-step changes are concentrated near zero with heavy tails. Most windows evolve smoothly; a smaller group of ramps changes output quickly. The ramp subset uses the upper decile of valid observed changes separately for each plant. It is the regime in which an external visual observation has the clearest possible role.

#figure(
  image("figures/ramp_rates.png", width: 90%),
  caption: [Distribution of step-to-step changes in normalized power. Most changes are small, while the tails contain the cloud-driven ramps used in the focused evaluation regime.]
)

Plants differ in capacity factor, coverage, and the relation between local output and weather covariates. These differences are part of the intended transfer problem. Macro-averaging metrics per plant ensures that each held-out installation contributes one unit of evidence rather than allowing the largest or most complete series to determine the final result.

#figure(
  image("figures/capacity_distribution.png", width: 88%),
  caption: [Installed capacities across the corpus. Capacity normalization allows sites with very different physical scales to enter one transfer evaluation.]
)

#figure(
  image("figures/site_map.png", width: 88%),
  caption: [Locations of the UK systems. Holding out plants prevents direct access to their histories, although geographic proximity remains an important limitation when interpreting transfer.]
)

== Split construction and limits

The UK split is generated once with seed 42. It contains 69 training plants, 15 validation plants, and 14 test plants after the bad sites are excluded. The data loader asserts that these sets are disjoint. The test partition contains 172,656 rows and 171,543 valid power steps before forecast windows are built.

Fourteen days of history and six hours of future power are defined in physical time. This makes the task interpretable across cadences. The local dataset mirror at /Volumes/dataset/dataset reproduces the 165,295 scored UK test steps, while the experiment dataset remains read-only.

The corpus supports a demanding cross-plant comparison within the UK fleet and prepares later cross-dataset evaluation. It does not prove generalization across every climate, sensor, orientation, or deployment type. The UK systems are mostly residential and geographically concentrated. The US track is smaller, covers a shorter period, and awaits a corrected split. These limits belong to the interpretation of every later result.

#include "part-one-extended.typ"
#include "part-one-data-contract.typ"

#pagebreak()
#bibliography("biblio.bib", style: "ieee")
