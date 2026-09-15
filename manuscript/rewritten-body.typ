#heading(level: 1, numbering: none)[Abstract]

Photovoltaic forecasting is often evaluated by withholding later observations from plants that a model has already seen. That setup does not answer a practical transfer question: whether a forecasting system can work for a newly connected plant after a short operating history. This thesis evaluates that question with plants, rather than timestamps, held out from training.

The study uses power measurements, weather covariates, and satellite frames from 110 photovoltaic systems in the United Kingdom and the United States. The primary experiment uses 14 held-out UK systems and 165,295 valid daylight forecast steps. Every model receives the same fourteen-day numerical history and predicts the next six hours. Evaluation uses capacity-normalized errors, a skill score relative to Smart Persistence, and error on the highest decile of observed power changes.

The proposed Interleaved Vision Chronos Forecaster combines Chronos-2 with a frozen V-JEPA 2.1 video encoder. Chronos-2 is first fine-tuned on the training plants. During visual training, the video encoder remains fixed; the upper three Chronos-2 encoder blocks and a new fusion bridge are optimized. The bridge retains spatially addressable cells, ranks them by temporal novelty, and inserts them alongside numerical tokens in time order.

The central result comes from a counterfactual test. The same trained model is evaluated with satellite input present and with the visual path disabled. A late-fusion model based on one pooled visual vector has no measurable visual reliance. The interleaved model reduces ramp NMAE from 0.1506 for the numerical control to 0.1435 and has positive visual reliance of 0.0073 across three seeds. It reaches a skill score of 55.64, ahead of the evaluated multimodal baselines. Stale, cross-plant, and randomly selected frames degrade performance, showing that the model uses recent, local, dynamically changing visual information. The visual model has lower interval coverage than the numerical control, so the point-forecast gain does not extend to every probabilistic criterion.

= Introduction
<cha:intro>

== Cross-plant forecasting

Photovoltaic power follows a daily pattern, but the forecast errors that matter arise when clouds alter generation rapidly. A model trained on years of data from one installation can learn that installation's history. A deployed system also needs to serve recently connected plants with no comparable archive.

This thesis studies cross-plant forecasting. Training, validation, and test partitions contain different installations. A test plant contributes neither fitting examples nor data-derived normalization statistics. The model receives recent observations from the plant at inference time, while its parameters were learned from other plants. This setting measures transfer across a forecasting fleet.

Satellite imagery may help because it can show a cloud field before the cloud changes the local power trace. An accuracy difference between separately trained models does not establish that a model used imagery, however. The difference can also arise from optimization, capacity, or regularization. The experiments therefore measure visual reliance with all model parameters fixed.

== Questions and contributions

The thesis asks whether models trained on one set of plants can forecast disjoint plants from a short history; how supervised, foundation, retrieval, and multimodal approaches compare under one protocol; and whether satellite imagery improves the high-ramp regime.

It contributes a multimodal forecasting corpus with one tensor contract, a cross-plant benchmark that groups methods by model family, and the Interleaved Vision Chronos Forecaster. The model is evaluated with a counterfactual reliance metric and controls designed to distinguish useful visual information from an incidental effect of multimodal training.

= Background
<cha:background>

== Foundation models and visual representations

Time-series foundation models are pretrained on many series before they are applied to a target task. Their promise in this setting is transfer to a series from an unseen plant. Chronos-2~@chronos2 is the numerical backbone used here. The benchmark also includes TimesFM~@timesfm, TiRex~@tirex, and Tiny Time Mixer~@ttm, because zero-shot transfer remains an empirical question rather than a property guaranteed by pretraining.

Satellite frames contain spatial structure absent from a scalar power trace. V-JEPA 2.1~@vjepa2 is used as a fixed self-supervised video encoder. It produces a spatiotemporal field of latent patches. The important design choice is the interface between this field and the forecaster. Late fusion supplies one pooled visual vector as an auxiliary feature. Sequence interleaving preserves several visual tokens and lets the forecasting transformer attend jointly to visual and numerical observations.

The comparison includes Time-VLM~@timevlm and VisionTS++~@visiontspp, which render numerical series as images; UniCast~@unicast, which uses multimodal prompting; and Solar-VLM~@solarvlm, CrossViVit~@crossvivit, and SUNSET~@sunset, which consume external visual information.

= Dataset and task definition
<cha:dataset>

== Corpus and split

The corpus combines two tracks. The UK track contains 100 residential systems sampled at 30-minute power cadence from 2019 through 2020. The United States track contains 10 systems sampled every 15 minutes through September 2019. Together they provide 1,337,654 rows. Each row contains normalized power, site metadata, weather and solar covariates, quality flags, and a pointer to a satellite frame in a packed archive.

#figure(
  align(center)[#table(
    columns: 6,
    align: (left, center, center, center, center, left),
    table.header([Track], [Sites], [Rows], [Power cadence], [Frame size], [Period]),
    table.hline(),
    [`uk_pv`], [100], [1,232,862], [30 min], [128 x 128 RGB], [2019-01 to 2020-12],
    [`goes_pvdaq`], [10], [104,792], [15 min], [256 x 256 RGB], [2019-01 to 2019-09],
  )],
  caption: [Dataset tracks in the consolidated corpus.]
  , kind: table
)

The primary results use the UK track. Two sites marked by quality flags are excluded. The remaining sites are split once into 69 training, 15 validation, and 14 test plants. Capacity normalization uses audited installed capacity, which is known at installation, rather than a statistic derived from a held-out series. Metrics are computed per plant and macro-averaged.

Every primary comparison uses fourteen days of numerical history and a six-hour horizon. At the UK cadence this gives 672 historical power values and 12 future values. The visual input is an eight-frame recent clip. The US track is retained for later cross-dataset work; its split must be reconciled with recent site-quality flags before it can support headline comparisons.

= Evaluation protocol
<cha:protocol>

All models use the same plant split, physical history and horizon, and capacity-normalized target. Learned models do not use hand-coded irradiance or clear-sky transformations. Smart Persistence is the sole exception because it is the physical reference used to define forecast skill. Future observed weather is labelled an oracle condition and excluded from competitive comparisons. Retrieval stores only windows from training plants.

The headline skill score is

$ upright(SS) = 100 dot (1 - "NRMSE"_"model" / "NRMSE"_"Smart Persistence") . $

A score of zero matches Smart Persistence and 100 denotes a perfect forecast. The ramp metric restricts error to the top decile of valid step-to-step changes in normalized power for each plant. The same subset is used for every model.

= Baseline families
<cha:baselines>

The benchmark is organized by model family. This describes the role of each comparison without suggesting an ordering of importance.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left, left, left),
    table.header([Model family], [Examples], [Purpose]),
    table.hline(),
    [Reference methods], [Persistence, Smart Persistence, climatology], [Set practical reference points],
    [Tabular models], [LightGBM, TabPFN, TabFM], [Test flattened numerical features],
    [Supervised sequence models], [MLP, DLinear, PatchTST, iTransformer, TFT], [Test task-trained numerical forecasting],
    [Time-series foundation models], [Chronos-2, TimesFM, TiRex, Tiny Time Mixer], [Test pretrained transfer],
    [Foundation-model adaptation], [TS-RAG, Cross-RAG, CoRA], [Test retrieval and lightweight adaptation],
    [Multimodal models], [Time-VLM, UniCast, Solar-VLM, CrossViVit, SUNSET], [Test visual interfaces],
  )],
  caption: [Model families in the benchmark.]
  , kind: table
)

Some published systems have native data assumptions or evaluation windows that cannot be made identical without changing the method substantially. The causal claims about the proposed model rely on the matched numerical control and its counterfactual tests, which keep data, optimization, and most parameters fixed.

= Interleaved Vision Chronos Forecaster
<cha:method>

== Training policy

Chronos-2 uses its native 768-dimensional, twelve-block encoder configuration. A 16-sample patch turns the UK history into 42 context tokens. The output head produces nine quantiles for the 12 future observations.

Chronos-2 is fine-tuned on training plants before visual fusion is introduced. During multimodal training, V-JEPA is fully frozen. The final three Chronos-2 encoder blocks remain trainable at one tenth of the backbone learning rate, together with the new visual bridge and output components. This is partial adaptation, not a frozen-backbone result.

== Visual tokens and sequence interleaving

V-JEPA receives eight 224 x 224 frames and produces four latent time slices, each with a 14 x 14 field of 1024-dimensional patches. The bridge pools each non-overlapping 2 x 2 neighborhood into 49 spatial cells. A position-wise multilayer perceptron maps each cell to the 768-dimensional Chronos representation without mixing locations.

The bridge selects cells by temporal novelty. It compares the latent feature at each spatial location with its value in the preceding V-JEPA slice. A complete anchor field is retained from the first slice, then the remaining token budget is assigned to locations with the largest changes. This produces 98 visual tokens from the clip.

Visual tokens are inserted into the numerical sequence by observation time. The combined sequence has 42 numerical tokens, 98 visual tokens, and one future query. Chronos-2 processes the whole sequence with its existing attention layers. The design is evaluated as a complete interface; the experiment does not attribute its effect to one internal choice alone.

== Counterfactual visual reliance

Visual reliance is measured after training with all parameters fixed:

$ "Reliance"_"ramp" = "NMAE"_"ramp"^("vision off") - "NMAE"_"ramp"^("vision on") . $

A positive value means the model is more accurate when it receives the visual input. Modality dropout during training exposes the model to missing inputs, so the vision-off condition is within its training distribution.

= Results
<cha:results>

== Controlled fusion comparison

The numerical foundation control reaches a skill score of 52.30 and ramp NMAE of 0.1506. A pooled late-fusion variant reaches 52.58 and 0.1487. Its counterfactual reliance is 0.0000 plus.minus 0.0015, so disabling its visual input does not change ramp accuracy within measured variation.

The Interleaved Vision Chronos Forecaster reaches 55.64 plus.minus 0.18 skill and 0.1435 plus.minus 0.0004 ramp NMAE. Its visual reliance is 0.0073 plus.minus 0.0002. The result supports a narrow conclusion: the complete interleaved interface uses visual input and improves ramp accuracy relative to the numerical control.

#figure(
  align(center)[#table(
    columns: 4,
    align: (left, center, center, center),
    table.header([Configuration], [Skill score], [Ramp NMAE], [Visual reliance]),
    table.hline(),
    [Numerical foundation control], [52.30 plus.minus 0.41], [0.1506 plus.minus 0.0010], [n/a],
    [Pooled late fusion], [52.58 plus.minus 0.43], [0.1487 plus.minus 0.0010], [0.0000 plus.minus 0.0015],
    [Interleaved Vision Chronos Forecaster], [55.64 plus.minus 0.18], [0.1435 plus.minus 0.0004], [0.0073 plus.minus 0.0002],
  )],
  caption: [Controlled fusion comparison over seeds 42, 43, and 44.]
  , kind: table
)

== Falsification controls and benchmark context

Replacing the clip with imagery lagged by six hours increases ramp NMAE by 0.0169. Supplying a contemporaneous sequence from another plant increases it by 0.0176 and reduces skill from 55.10 to 37.00. Uniform random selection at the same token budget increases ramp NMAE by 0.0039. These controls are consistent with reliance on recent, local, changing visual content.

Shuffling frame order changes ramp NMAE by less than 0.0003 across seeds. The current model has therefore not demonstrated fine-grained motion tracking. Its visual benefit is best described as use of the recent spatial state and the locations selected by temporal novelty.

Among evaluated multimodal models, the proposed model has the highest recorded skill score. Time-VLM reaches 54.04 with a rendered-series visual stream. Solar-VLM reaches 43.96 and ramp NMAE 0.1514. In the numerical comparison, iTransformer reaches 52.57 skill and 0.1445 ramp NMAE. Chronos-2 zero-shot reaches 47.37 skill and 0.1544 ramp NMAE, while TS-RAG and Cross-RAG reach 47.79 and 47.68 skill. Generic pretraining alone does not resolve the task.

== Calibration and limits

The visual model improves point accuracy while worsening interval coverage. The numerical control covers 77.2 percent of targets with its nominal 80 percent interval. The interleaved model covers 75.3 percent, and its quantile calibration error rises from 0.0297 to 0.0314. The point-forecast result is therefore not an unconditional improvement in probabilistic forecasting.

The evidence is limited to the current UK evaluation. It does not prove transfer across countries, sensors, or plant scales. Fine distinctions among broadly evaluated baselines also require more repeated runs than are currently available. The three-seed fusion comparison provides the strongest evidence in this thesis.

= Conclusion
<cha:conclusion>

This thesis evaluates multimodal foundation-model fusion under a cross-plant photovoltaic forecasting protocol. The Interleaved Vision Chronos Forecaster retains spatially addressable video features and presents them to the time-series model as temporally ordered tokens. On held-out UK plants, it improves ramp NMAE and skill relative to the numerical foundation control. Its accuracy degrades when recent local imagery or novelty-based token selection is removed, which supports the conclusion that it uses satellite information at inference time.

Future work should reconcile the US split and test cross-dataset transfer. It should also address calibration, image-resolution sensitivity, and architectures that can use motion across a clip rather than primarily its recent spatial state.

#pagebreak()
#bibliography("biblio.bib", style: "ieee")
