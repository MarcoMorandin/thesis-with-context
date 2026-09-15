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

= Related work and conceptual background
<cha:related>

== Plant-specific and cross-plant forecasting

Many photovoltaic forecasting studies fit a model to a site's own history. That approach is appropriate when the site has a long archive and the task is to improve one familiar forecast. It does not answer how a system should behave when an installation joins a fleet with little historical data. Cross-plant forecasting changes the unit of separation from timestamps to entities.

This change affects preprocessing as well as evaluation. Per-series standardization uses the mean and variance of each held-out plant. In a transfer setting, those statistics reveal the distribution of the test entity before forecasting begins. The corpus uses audited installed capacity instead. Capacity is installation metadata available at commissioning, so normalizing power by capacity does not leak information from a test time series.

The task is not cold start in the strictest sense. At inference time a model receives fourteen days of recent power history and related observations. This gives it several daily cycles and a local operating context. The model lacks only the long plant-specific training history that conventional approaches depend on.

== Time-series foundation models

Time-series foundation models are pretrained on diverse series before they are used zero-shot or adapted to a target task. Their promise is that a representation learned across many temporal processes can transfer to an unseen series. Chronos-2~@chronos2, TimesFM~@timesfm, TiRex~@tirex, and Tiny Time Mixer~@ttm take different approaches to this goal.

Pretraining does not guarantee success on photovoltaic generation. The target combines a hard night-time zero, strong daily and seasonal structure, abrupt ramps, and differences among plants. A zero-shot model must determine which aspects of its learned temporal vocabulary apply to this setting. Fine-tuning may help, but it also makes it necessary to distinguish the value of pretraining from the value of target-task data.

Chronos-2 is especially relevant because it represents long histories as patches and predicts quantiles for the future horizon. A patch-based sequence also provides a natural place to introduce information from another modality. The methodology chapter will use this property to compare late fusion with shared token-level processing.

== Video foundation models

Cloud fields have spatial extent and change over time. A visual model that reads a clip can represent more than isolated images. V-JEPA 2.1~@vjepa2 is a self-supervised video encoder that learns relationships among latent spatiotemporal regions. In this thesis it is used as a general feature extractor rather than as a photovoltaic model.

A frozen visual encoder is helpful for a controlled study. It reduces the chance that a large visual network learns target-specific shortcuts during training. It also places the scientific focus on the interface: what the forecasting system retains from the visual field, how it aligns visual and numerical observations, and whether that information improves the relevant forecast regime.

== Multimodal fusion

Late fusion encodes a visual clip, pools it into one vector, and combines that vector with a numerical representation. It is simple and cheap, but pooling makes an early irreversible decision. Once the visual field becomes one vector, a downstream forecaster cannot ask which location in the image contained a relevant cloud boundary.

Prompting and soft-token methods retain a mostly fixed foundation model and expose a small number of auxiliary visual tokens. They test whether a pretrained backbone can absorb a new modality without broad retraining. Token-level fusion retains a set of visual observations and passes them with numerical tokens through common transformer layers. This requires a careful token budget, but it lets a future query relate local visual state to a particular time in the numerical context.

The benchmark later compares reference methods, tabular models, supervised sequence models, time-series foundation models, retrieval methods, and multimodal systems. Solar-VLM~@solarvlm, CrossViVit~@crossvivit, and SUNSET~@sunset use external visual information. Time-VLM~@timevlm and VisionTS++~@visiontspp render the numerical series itself as an image. The latter category can test a visual representation of the same input, but it cannot observe a cloud field approaching the plant.

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
