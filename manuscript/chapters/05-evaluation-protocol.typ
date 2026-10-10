// =============================================================================
//  Chapter 5 — Evaluation protocol
//
//  Included by ../thesis.typ. Not compilable on its own: the page setup, show
//  rules and bibliography live in the main file. Image paths are relative to
//  this file, hence `../figures/`.
// =============================================================================

= Evaluation protocol
<cha:protocol>

The dataset contract in @sec:contract defines the forecast sample. This chapter specifies how those samples are compared. It defines which plants are held out, what each model may observe at the forecast origin, how errors are aggregated, and how visual reliance is tested. Within each dataset track, direct comparisons use the same forecast origins, target masks, and horizon. Learned methods predict power without a hand-coded clear-sky or clearness-index transformation; established reference forecasts remain in the benchmark.

== Plant-disjoint partitions

Plant identity is the unit of partitioning within each dataset track. The fixed United Kingdom assignment contains 69 training plants, 15 validation plants, and 14 test plants. @tbl:uk-partition gives the quality-valid power observations in each group before forecast-window and daylight filtering. Any evaluation on another track requires its own plant-disjoint partition under the same separation rule. The results chapter reports the UK test partition.

#figure(
  align(center)[#table(
    columns: 4,
    align: (left, center, center, left),
    table.header([Partition], [Plants], [Valid power steps], [Use]),
    table.hline(),
    [Training], [69], [846,633], [Parameter fitting and foundation-model adaptation],
    [Validation], [15], [182,809], [Early stopping and protocol-fixed tuning],
    [Test], [14], [171,543], [Final cross-plant reporting],
  )],
  caption: [United Kingdom plant partition after quality-controlled exclusions. Each installation belongs to exactly one partition.]
  , kind: table
) <tbl:uk-partition>

The evaluator advances by one forecast horizon between origins and applies the target-validity and daylight masks to the windows defined in @sec:contract. For the United Kingdom partition, this leaves 19,964 test windows and 165,295 scored target steps. The 171,543 test steps in @tbl:uk-partition count valid power observations before those filters.

A forecast origin is the time at which the history ends and prediction begins. A target step is one future power reading within that forecast window, so a window contributes several target steps. A validity mask is a selection rule indicating which readings are eligible: the target-validity mask excludes unavailable or invalid measurements, and the daylight mask restricts scoring to the daylight subset. Matching these masks and origins gives compared models the same observations, avoiding differences in the mix of easy and difficult periods.

Training plants supply fitted parameters, including adapter weights, and any retrieval datastore. Validation plants determine checkpoints and hyperparameters, including retrieval weights when tuned. Test plants are used only for final reporting. This boundary also applies to learned normalization and feature selection. Each implementation declares its fitted statistics. The loader checks that plant assignments do not overlap.

== Scenarios and input parity

The primary benchmark is the United Kingdom plant-disjoint comparison. The results chapter also reports an inference-only UK-to-US transfer evaluation. UK-fitted MMTSFM checkpoints are applied to one held-out US plant after US power is resampled to 30-minute cadence; no US observations are used for fitting. This single-plant evaluation is separate from the primary benchmark and does not measure average performance across the US track. Within-plant and low-history evaluations are not reported. Across these reported comparisons, each model receives the modalities its design supports. Persistence forecasts use no satellite frames. Numerical forecasters use power history and, where supported, covariates. Multimodal systems can also use recent imagery. @tbl:input-parity summarizes the inputs by model family. Each result identifies the fields used by the individual method.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left, center, center, center, left),
    table.header([Model family], [Power history], [Covariates], [Satellite frames], [Purpose in the comparison]),
    table.hline(),
    [Reference forecasts], [Yes], [Method-specific], [No], [Daily and short-horizon reference],
    [Supervised tabular and sequence models], [Yes], [Model-specific], [No], [Task-specific numerical forecasting],
    [Tabular foundation models], [Yes], [Yes], [No], [Transfer from tabular pretraining],
    [Time-series foundation models], [Yes], [Model-specific], [No], [Transfer from broad time-series pretraining],
    [Retrieval and adapter methods], [Yes], [Where supported], [No], [Alternatives to full backbone adaptation],
    [Multimodal systems], [Yes], [Where supported], [Yes], [External visual context],
  )],
  caption: [Modalities available to each model family at the shared forecast origin.]
  , kind: table
) <tbl:input-parity>

== Baseline implementations and comparison boundaries
<sec:baseline-implementations>

The models reviewed in @cha:related are compared under the track-specific plant split and input rules above. This subsection specifies each model’s inputs, whether it learns from the training plants, and whether its reported score uses the same evaluation set as other methods.

The non-learned references use only permitted observations. Persistence repeats the latest valid power value across the horizon. Seasonal naive reuses the value from the same clock time on the previous day. Hourly climatology estimates a plant's typical output at each hour from its available history. Smart Persistence carries the latest observed clearness relationship forward as the deterministic solar envelope changes~@haurwitz. It supplies the denominator of the skill score.

LightGBM~@lightgbm and TabPFN~@tabpfn3 receive the same fixed table of lagged power, calendar variables, and permitted meteorological observations. LightGBM is fitted on training plants, while TabPFN is used zero-shot. In-context examples are summarized by a rule held fixed across plants. None receives satellite frames or forecast-horizon covariates. The supervised sequence models are also trained on the training plants. The reported iTransformer~@itransformer run uses historical covariates. PatchTST~@patchtst, the Temporal Fusion Transformer~@tft, DLinear~@dlinear, and the MLP control also use observed history. MMTSFM additionally receives horizon-aligned covariates. Its comparison with iTransformer therefore does not isolate the visual pathway.

The time-series foundation-model entries start from released checkpoints. Chronos-2~@chronos2 is evaluated both zero-shot and after adaptation on training plants as the numerical control of the fusion ladder. TimesFM~@timesfm and TiRex~@tirex are used zero-shot. Tiny Time Mixer~@ttm appears in zero-shot and fine-tuned forms. CoRA~@cora adds a covariate adapter to a mostly frozen backbone. TS-RAG~@tsrag and Cross-RAG~@crossrag condition the forecaster on windows retrieved from training plants. These entries receive no imagery. Their architectures, pretraining corpora, and adaptation strategies differ. Their rankings therefore compare benchmark performance across methods and do not constitute an architectural ablation.

The MMTSFM arms and the fine-tuned Aurora run use the common held-out plant partition, forecast windows, target masks, and scoring rules. The available runs of SUNSET~@sunset, UniCast~@unicast, Solar-VLM, Time-VLM, VisionTS++, and zero-shot Aurora use native windows or masks. Some also use a different test-plant set. Their scores are therefore reported separately from protocol-aligned comparisons in @cha:results. These methods differ in whether their visual input comes from satellite imagery or a rendering of numerical history. @cha:related describes their designs.

Input adapters map each implementation to the common dataset window, capacity-normalized target, and physical forecast horizon. A metric that a method does not produce, such as quantile quality or a protocol-aligned ramp score, is reported as unavailable rather than reconstructed after evaluation. Parameter count and pretraining compute are not equalized. Comparability rests on matched targets, plant separation, allowed information, and metrics. Each result discloses its adaptation mode and implementation deviations.

== Scoring and reporting

Within each track, valid daylight targets are scored separately for every held-out plant before the plant scores are averaged. This gives installations equal weight even when their usable records differ in length. The two co-primary measures answer different questions. Skill score measures the reduction in NRMSE relative to Smart Persistence across the scored targets. Ramp NMAE measures error during rapid changes in observed power. A gain in overall skill therefore does not compensate for worse ramp accuracy. The formulas are given in @sec:metric-definitions.

Ramp timestamps are selected from observed power before any model is scored. For each plant, the evaluator calculates the absolute change between consecutive valid readings, using the final history reading for the first step of each forecast window. The largest 10% of valid daylight changes across the test set form a fixed subset used for every model and evaluation pass. The manuscript does not state how ties at the threshold or valid readings separated by a missing interval are handled. Models that produce quantiles also report the CRPS approximation, 80% interval coverage, and quantile expected calibration error. These metrics use the same valid-daylight mask and plant weighting.

Training variation is reported alongside these scores. Learned methods use fixed random seeds for data ordering and parameter initialization. Each result states the number of independently completed runs; replicated results report their mean and standard deviation. Where results show mean ± standard deviation, “±” denotes variation across the reported seeds, not a confidence interval.

For multimodal models, a score advantage alone does not show that the trained forecast uses images. Each trained checkpoint is therefore evaluated with vision active and again with its visual embeddings set to zero. The numerical embeddings and token positions remain in place. Both passes use the same weights, forecast origins, target masks, ramp thresholds, and aggregation. Their differences in normalized error and ramp NMAE measure sensitivity to visual input under that checkpoint. This comparison keeps the training route, capacity, and regularization fixed. A separately trained numerical model would change those factors as well. A negligible vision-on/off difference can therefore coexist with a nominal advantage over a numerical model. Higher error with vision disabled indicates a positive contribution from the visual pathway under this checkpoint and intervention.

The vision-on/off pass cannot identify which part of the clip matters. The cached-latent-order, stale-image, and wrong-plant-image substitution interventions are specified in @cha:ablations. They use the trained weights and, where possible, the same forecast origins. They are reported apart from standard model comparisons because they change the input to an existing model. Reports record the latent-slice permutation or its seed, the stale offset, and the donor mapping. A wrong-plant-image substitution needs a clip from a different test plant at the same time. Shuffled test batches allow the evaluator to find such a donor when series-major batches would contain only one plant.

Each reported result identifies the dataset track and plant partition, physical window lengths, input modalities, exact covariate fields, and forecast-origin rules. It also identifies the random seeds, numbers of windows and scored target steps, checkpoint-selection metric, tuning budget, and whether the comparison was prespecified or exploratory. Foundation-model results identify the pretrained checkpoint and state what is known about its training-data provenance, including any unknown site- or time-level overlap. Results from a different window builder, test subset, or non-disjoint split remain separate from the corresponding within-track cross-plant comparison.

#pagebreak()
