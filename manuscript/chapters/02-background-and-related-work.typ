// =============================================================================
//  Chapter 2 — Background and related work
//
//  Included by ../thesis.typ. Not compilable on its own: the page setup, show
//  rules and bibliography live in the main file. Image paths are relative to
//  this file, hence `../figures/`.
// =============================================================================

= Background and related work
<cha:related>

This chapter reviews numerical predictors, time-series foundation models, visual encoders, and multimodal interfaces under the plant-disjoint setting defined in @cha:intro. It asks what each method transfers to a held-out plant and what evidence shows that imagery affects its forecast.

== Numerical forecasting models
<sec:rw-numerical>

The benchmark begins with methods that make different assumptions about temporal structure. Persistence, Smart Persistence, climatology, and seasonal naive forecasts establish how much of the target can be explained by the current state and the daily cycle. They matter because photovoltaic power has a strong astronomical envelope. A method that improves only on this envelope may appear accurate while missing the difficult cloud-driven changes. Among these references, Smart Persistence carries the observed clearness relationship forward while the deterministic solar envelope evolves~@haurwitz. It therefore separates the predictable envelope from the unpredictable sky and serves as the denominator of the skill score used throughout this thesis.

The benchmark also includes tabular and sequence predictors. Some learn from the training fleet; others bring an externally pretrained prior. Tabular models convert a forecasting window into a fixed feature vector and ask whether a structured table already exposes useful information. Supervised sequence models keep the series as a sequence and differ in how they compress it, whether they model variables jointly, and how much nonlinearity they permit. The input regime and adaptation mode of each entry are fixed in @sec:baseline-implementations. Scores follow in @cha:results.

=== LightGBM: gradient boosting over a feature table
<sec:rw-lightgbm>

LightGBM~@lightgbm grows an ensemble of decision trees over lagged normalized power, calendar variables, and permitted meteorological covariates. Each split is an axis-aligned threshold, allowing the model to capture nonlinear cutoffs and interactions. In this benchmark, LightGBM provides a feature-table reference for testing how much predictive information the engineered inputs expose.

The transfer setting strains these assumptions. Temporal relationships are fixed before learning begins. LightGBM can discover that a particular lag matters, but it cannot construct intermediate states as an attention-based or recurrent model does. A learned threshold transfers only if its feature has a similar meaning at a held-out plant. LightGBM tests whether the engineered table is sufficient for accurate transfer.

=== TabPFN: tabular foundation model
<sec:rw-tabpfn>

TabPFN~@tabpfn3 uses the same feature table without fitting a predictor on the target data. It applies a transformer pretrained across synthetic tabular tasks and predicts from a bounded in-context set of example rows. That setup is relevant when a new plant has little history and fitting a separate model is poorly posed.

TabPFN's bounded context means that a long, densely sampled corpus must be summarized into the rows and features the model accepts. The row-selection rule is therefore part of the method. @sec:baseline-implementations specifies how it is applied across plants. The result tests how far a pretrained tabular prior can forecast from engineered numerical features without learning directly from the original sequence.

=== DLinear and PatchTST: linear decomposition against patched attention
<sec:rw-dlinear-patchtst>

DLinear~@dlinear separates each series into trend and residual components and forecasts each with a linear map. It provides a simple test of how far linear forecasting can go after capacity normalization. A score near the transformers would weaken a claim that attention or pretraining is needed for this task.

PatchTST~@patchtst occupies the middle of that spectrum and introduces the two ideas that recur throughout this thesis. It divides each univariate history into temporal patches. This shortens the sequence and encourages the model to learn recurring local motifs. PatchTST processes channels independently, which prevents cross-variable correlations from being baked into the representation. Channel independence is a plausible advantage under plant-disjoint evaluation because the correlation between power and a given covariate need not be stable across installations. The cost is that interactions among power and meteorological variables are never modelled directly. Patching also appears in the numerical backbone described in @sec:rw-chronos2. The same compression makes long histories tractable but blurs rapid behavior inside a patch. That tension becomes material when the quantity of interest is a cloud-driven ramp.

=== The Temporal Fusion Transformer: gating and variable selection
<sec:rw-tft>

In contrast to PatchTST's channel-independent design, the Temporal Fusion Transformer~@tft combines recurrent processing, gated residual connections, explicit variable selection, and interpretable multi-horizon attention. It was designed for supervised settings in which known future inputs, observed past inputs, and static metadata coexist and vary in quality. This setup resembles a photovoltaic fleet, where some covariates are reliable at one site and noisy at another. The model's gates can in principle learn to suppress unhelpful covariates. It is included to test whether explicit input selection and a purpose-built multi-horizon decoder compensate for the absence of broad pretraining.

=== iTransformer: variables as tokens
<sec:rw-itransformer>

iTransformer~@itransformer inverts the usual transformer convention. Each physical variable becomes a token whose embedding spans the whole input window. Self-attention operates across variables rather than across time. In photovoltaic forecasting, generation, solar position, cloud cover, irradiance, temperature, and wind share substantial structure. The inversion lets the model relate these variables explicitly without inferring their relationship through a shared temporal representation. The token layout is shown in @fig:related-itransformer.


#figure(
  image("../figures/related_itransformer_original.png", width: 94%),
  caption: [iTransformer architecture. Each variable is embedded as a token spanning the input window, and self-attention models relations across variables. Reproduced from the source architecture~@itransformer.]
) <fig:related-itransformer>


== Time-series foundation models
<sec:rw-tsfm>

Time-series foundation models are pretrained on many series before zero-shot use or task adaptation. Chronos-2~@chronos2, TimesFM~@timesfm, TiRex~@tirex, and Tiny Time Mixer (TTM)~@ttm cover several architectural families. A model exposed to a broad corpus may bring useful temporal representations to a series it has never seen.

Their treatment of the input differs. Patch-based models compress a long context into blocks of observations. This makes long histories tractable, but can blur rapid behavior within a patch. Covariate support and output representations also differ across families, from point forecasts to quantile distributions.

Pretraining does not remove the need for a controlled comparison. Photovoltaic power has daily zeros, seasonality, cloud-driven ramps, and plant-to-plant variation that may not be represented in a general pretraining corpus. A zero-shot result measures the match between prior knowledge and this task. A fine-tuned result measures both that prior and the information obtained from the training fleet. This distinction is carried into every result table in @cha:results, because the two quantities support different claims.

=== Chronos-2: the shared numerical backbone
<sec:rw-chronos2>

Chronos-2~@chronos2 has a dual role in this thesis and therefore deserves a fuller description than the other checkpoints. It tokenizes a numerical history into fixed-length patches, processes them with a pretrained encoder, and emits quantiles for the forecast horizon in one pass. @fig:chronos2-original-architecture shows the original architecture. Patching keeps the sequence short enough for additional visual tokens, while the quantile head supports accuracy and calibration evaluation. The visual arms insert tokens into a backbone that was originally trained for numerical inputs.

#figure(
  image("../figures/chronos2_original_architecture.png", width: 100%),
  caption: [Original Chronos-2 architecture, showing numerical input processing, the transformer backbone, and quantile forecasting. Reproduced from Figure 1 of the source paper~@chronos2.],
) <fig:chronos2-original-architecture>

The same checkpoint appears in the benchmark twice. Its zero-shot run tests how the released model performs on held-out plants before exposure to the training fleet. Fine-tuning it on the training plants produces the numerical baseline. The difference between these runs measures the benefit of in-domain adaptation from that pretrained starting point; it does not measure the value of pretraining itself. The numerical baseline is also the vision-free anchor of the fusion ladder. Each visual arm starts from its seed-matched adapted checkpoint.

=== TimesFM, TiRex, and Tiny Time Mixer: breadth of temporal pretraining
<sec:rw-tsfm-breadth>

The remaining foundation models provide task-level context across several architectures, showing how far off-the-shelf temporal pretraining reaches on this forecasting problem. TimesFM~@timesfm uses a decoder-oriented design intended for direct zero-shot forecasting at arbitrary horizons. TiRex~@tirex emphasizes in-context learning across varied horizon lengths. Tiny Time Mixer~@ttm replaces attention with lightweight mixing blocks. It targets a much smaller parameter budget, keeping both pretraining and adaptation cheap.

Together they show how released time-series checkpoints behave on plant-disjoint photovoltaic forecasting before fitting to the fleet. Photovoltaic power combines nightly zeros, a strong astronomical envelope, and ramps driven by an external process that the series records only after it reaches the plant. A weak zero-shot result therefore says more about the relation between the pretraining corpus and this task than about one architectural choice.

== Adaptation without a new model
<sec:rw-adaptation>

Some methods adapt a pretrained time-series model without fully retraining it. Fine-tuning updates some or all backbone weights on the target fleet. Retrieval conditions a frozen forecaster on similar historical windows drawn from a datastore. Covariate adapters offer a third route by injecting external inputs into an established representation through a small trainable path.

Their scores provide alternatives against which to judge the complete multimodal system, although the information available to each method differs.


=== TS-RAG and Cross-RAG: retrieval as conditioning
<sec:rw-retrieval>

TS-RAG~@tsrag and Cross-RAG~@crossrag retrieve related historical windows and make them available to a frozen or lightly adapted forecaster. @fig:tsrag-original-architecture shows the external-store interface in TS-RAG. Retrieval can supply examples of rare steep transitions at inference time without requiring the model weights to memorize them. The store can also grow without retraining the forecaster.

#figure(
  image("../figures/tsrag_original_architecture.png", width: 100%),
  caption: [TS-RAG overview from the source paper: a retriever supplies knowledge from an external store to a time-series foundation model while its backbone remains frozen~@tsrag.],
) <fig:tsrag-original-architecture>

Retrieval poses a specific risk for cross-plant evaluation. The method's effective training information is determined by the store's contents. A plant-disjoint protocol remains valid only if the store, the query construction, and the reported aggregate all respect the same entity boundary as the forecasting model. 

== Multimodal forecasting interfaces
<sec:rw-interfaces>

Multimodal forecasters use visual inputs in two ways. Time-series renderings let a vision encoder reinterpret measurements already present in the numerical history. Sky-camera and satellite images can add observations of the atmosphere that have not appeared in the power trace. The distinction matters when a forecast improves: a rendered image tests a new representation, while a real image may add new information whose use still needs to be measured.

=== Time-VLM and VisionTS++: rendered time-series images
<sec:rw-rendered>

Time-VLM~@timevlm and VisionTS++~@visiontspp render numerical history as an image and encode it with a visual backbone. Time-VLM also uses statistical text prompts; its paths appear in @fig:related-time-vlm. The rendered image may expose temporal patterns to a visual encoder, but it contains no observation beyond the numerical history used to construct it.

This distinction matters when interpreting a benchmark. A pseudo-image model can be a strong forecasting baseline because it tests a legitimate representation choice. It evaluates the value of re-encoding a history rather than the value of an independently observed modality.

#figure(
  image("../figures/related_time_vlm_original.png", width: 94%),
  caption: [Time-VLM architecture. Historical time-series values are rendered as an image and combined with statistical text prompts before forecasting. Reproduced from the source architecture~@timevlm.]
) <fig:related-time-vlm>


=== Aurora: pretraining across derived modalities
<sec:rw-aurora>

Aurora~@aurora extends this line of work to multimodal foundation-model pretraining. It combines modality-specific encoders with a cross-modality encoder that uses token distillation and modality-guided self-attention, as shown in @fig:related-aurora. Its training uses numerical series, images derived from those series, and text descriptions. The resulting model tests whether these representations improve general forecasting behavior and transfer.

#figure(
  image("../figures/related_aurora_original.png", width: 94%),
  caption: [Aurora architecture. Modality-specific encoders feed a cross-modality encoder that uses token distillation and modality-guided self-attention. Reproduced from the source architecture~@aurora.]
) <fig:related-aurora>


=== SUNSET: a photovoltaic convolutional reference
<sec:rw-sunset>

SUNSET was an early end-to-end model for forecasting photovoltaic power from ground-based sky-image sequences and recent power output~@sunsetearly. The forecasting architecture stacks successive frames along the image channel dimension and extracts visual features with two convolutional blocks. It inserts the power history after image processing and uses fully connected layers to predict 15-minute-ahead output~@sunset. Its released implementation has made it a recurring reference in image-based forecasting comparisons, including the SKIPP'D benchmark~@skippd.

SUNSET provides a specialized convolutional comparison for the foundation-model interfaces studied here.

=== Solar-VLM: gated late fusion
<sec:rw-solarvlm>

Solar-VLM~@solarvlm combines a visual pathway for satellite crops with meteorological text and a time-series pathway. As shown in @fig:related-solar-vlm, transformer blocks and weighted temporal aggregation summarize the visual sequence. A KNN/GAT pathway models relations among sites, and cross-site attention integrates that spatial context. The resulting visual and meteorological representation is fused with the time-series state through a learned gate. The architecture therefore includes temporal and cross-site processing in addition to station-level fusion. For the present comparison, the relevant distinction is that the forecaster receives an aggregated representation rather than a sequence of individually addressable satellite patches.

@fig:related-solar-vlm shows the pooled station representation and gate in Solar-VLM.

#figure(
  image("../figures/related_solar_vlm_original.png", width: 94%),
  caption: [Solar-VLM architecture. The diagram shows visual temporal aggregation, KNN/GAT cross-site processing, cross-site attention, and gated fusion with the time-series branch. Reproduced from the source architecture~@solarvlm.]
) <fig:related-solar-vlm>


=== PV-LLM: satellite and text prompts inside a language backbone
<sec:rw-pvllm>

PV-LLM~@pvllm carries the prompting idea into a pretrained language model, replacing the time-series backbone. It uses a frozen GPT-2 and three input channels. A temporal-semantic encoder maps power and numerical weather history into embeddings aligned with the word space. A text prompt contains plant coordinates, forecast start time, dataset and task descriptions, and input statistics. A satellite encoder uses a context compressor to reduce the patch grid of each frame to a single vector. @fig:related-pv-llm shows these separate paths. Satellite features enter every self-attention layer through adaptive prompt-tuning blocks, whose gates start at zero. In its reported benchmark, the system is trained and evaluated on MMSP(S), a ten-plant Shandong subset of the same MMSP corpus released with FusionSF, with Himawari imagery and ECMWF forecasts at hourly resolution. It reports lower error than unimodal transformers, MLP mixers, Time-LLM, and FusionSF.

The context compressor leaves one vector per frame, so image locations are no longer available as separate tokens inside GPT-2. The vector may still encode spatial information. The paper's prompt ablation finds larger gains from forecast start time and plant coordinates than from its other text components. Those gains concern metadata conditioning and do not establish which satellite content the model uses. Although coordinates can be available for a held-out plant, PV-LLM's reported evaluation does not test plant-disjoint transfer; its results are for easy and hard test windows. The cross-track result in this thesis evaluates different MMTSFM arms on one US plant. It does not change the scope of PV-LLM's published evidence.

#figure(
  image("../figures/related_pv_llm_original.png", width: 96%),
  caption: [PV-LLM architecture. Satellite frames, power and numerical weather history, and a structured text prompt are encoded separately. The text and time-series embeddings enter a frozen language model directly, while satellite embeddings are injected at each layer through adaptive prompt-tuning blocks. Reproduced from the source architecture~@pvllm.]
) <fig:related-pv-llm>

== Nemotron 3 Nano Omni: visual token selection
<sec:rw-nemotron>

Nemotron 3 Nano Omni~@nemotron is a multimodal model outside photovoltaic forecasting. Its visual pathway reduces features by spatial downsampling and then selects tokens from regions that change over time, as shown in @fig:related-nemotron. Efficient Video Sampling~@evs supplies the selection step by pruning temporally redundant video tokens. This thesis draws on the resulting interface principle: reduce the visual token count while keeping selected locations distinct for the downstream backbone.

#figure(
  image("../figures/related_nemotron_visual_path.png", width: 92%),
  caption: [Nemotron 3 Nano Omni visual pathway. The visual encoder produces tokens that are spatially reduced and then sampled before projection into the shared backbone. Reproduced from the source architecture~@nemotron.]
) <fig:related-nemotron>

== V-JEPA 2.1: dense video representations
<sec:rw-vjepa21>

V-JEPA 2.1~@vjepa21 is a self-supervised visual foundation model trained on images and video. Its dense predictive objective trains on both visible and masked tokens, with additional supervision at intermediate encoder layers. The resulting representations retain spatial structure and temporal information while supporting global scene understanding. Unlike a single embedding for each frame, its patch features leave individual image regions available to a downstream model.

@fig:related-vjepa21 shows the training architecture. Separate image and video tokenizers feed a shared context encoder; a predictor learns from both masked and visible tokens against targets from a teacher encoder.

#figure(
  image("../figures/related_vjepa21_architecture.png", width: 96%),
  caption: [V-JEPA 2.1 training architecture. Image and video inputs use separate tokenizers before the context and target encoders. Multi-level predictions are supervised at masked and visible token positions. Reproduced from the source architecture~@vjepa21.]
) <fig:related-vjepa21>

This thesis pairs a frozen V-JEPA 2.1 encoder with Chronos-2. Satellite clips produce visual latents that the pooled-image baseline summarizes or the spatial-token model selects from before fusion with numerical tokens. Nemotron motivates the selection interface, while V-JEPA 2.1 supplies the visual features.


#pagebreak()
