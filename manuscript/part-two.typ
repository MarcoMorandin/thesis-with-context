= Evaluation protocol
<cha:protocol>

This chapter defines the conditions under which models are compared in the thesis. The protocol treats photovoltaic forecasting as an entity-transfer problem: a score is useful only when it measures performance at installations that played no part in fitting the model. It also treats a multimodal score as insufficient evidence of visual use. A model may receive images, improve after multimodal training, and still make the same predictions when the images are absent. The protocol therefore fixes the data partition, forecast windows, inputs, metrics, and counterfactual checks before the results are considered.

== Fair comparison principles

Three rules govern every reported comparison. First, all models forecast the same physical horizon from the same amount of prior information. Second, the training, validation, and test partitions contain disjoint plants. Third, learned methods do not introduce hand-coded photovoltaic transformations as hidden modeling steps. These rules prevent a change in data availability or preprocessing from being mistaken for an architectural improvement.

The first rule concerns more than the horizon length. A model that sees an extra observation at the forecast origin, a denser version of the target history, or a future weather measurement has access to a different task. Likewise, a model evaluated on a narrower daylight subset can appear better simply because difficult examples have been removed. The common window builder and evaluator define these decisions once, then expose the resulting masks and timestamps to every implementation.

The second rule applies to operations that are sometimes treated as routine preprocessing. Test-plant rows cannot affect fitted parameters, normalizers, early stopping, feature selection, retrieval databases, or hyperparameter choices. Capacity is recorded metadata and may be used to express power on a common scale. Statistics estimated from a held-out series are not available. This distinction is essential because per-series scaling can encode much of an installation's operating range before the model has made a forecast.

The third rule keeps the comparison focused on learned transfer. The study forecasts power directly and does not insert a clear-sky or clearness-index transformation into the learned pathways. Reference forecasts remain useful because they describe established operational baselines, while the learned models are compared on the common target and window definition.

== Plant-disjoint partitions

The primary experiment uses the United Kingdom track. It begins with 100 residential photovoltaic systems sampled every 30 minutes from 2019 to 2020. Two systems are excluded by the dataset quality flags, leaving 98 plants. A seed-42 partition assigns 69 plants to training, 15 to validation, and 14 to testing. The partitions remain fixed across the benchmark.

The test set contains whole installations rather than later intervals from familiar installations. During fitting, the model may learn regularities that recur across the training fleet, such as the relationship between solar geometry, recent generation, and weather variables. At evaluation time it receives fourteen days of recent context from a test plant, but it has never received that plant as a fitting example. The test set therefore measures adaptation from short context and shared representation, rather than memorization of site-specific behavior.

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
  caption: [Primary United Kingdom plant partition after quality-controlled exclusions. Each installation belongs to exactly one partition.]
  , kind: table
)

The distinction between validation and test plants is especially important for methods with a compact adapter or a retrieval weight. A value chosen after observing the test set is a fitted value, even when it is only one scalar. Retrieval also remains subject to the split: its datastore may contain historical windows from training plants, but no window from a held-out plant.

== Windows defined in physical time

The primary task uses fourteen days of numerical history and a six-hour forecast horizon. The United Kingdom cadence turns those intervals into 672 history steps and 12 future steps. The time interval, rather than the number of samples, defines the task. This allows a corresponding experiment at another cadence to retain the same physical interpretation without pretending that 672 samples always represent the same history.

The visual stream has a different temporal role. It consists of eight recent satellite frames over a short, three-to-six-hour window. Its cadence is deliberately decoupled from the numerical history. A long power history supplies repeated daily structure and a short visual clip supplies recent cloud evolution. Forcing either source onto the other source's grid would either discard available image timing or shorten the numerical context needed for transfer.

Windows are built at valid daylight forecast origins. A history segment, its future target, and the required timestamps must satisfy the contract before the sample is scored. The primary United Kingdom test protocol contains 165,295 valid daylight forecast steps after this construction. The count is reported with all protocol-aligned results so that a reader can distinguish a full evaluation from one based on a different subset.

#figure(
  align(center)[#table(
    columns: 4,
    align: (left, center, center, left),
    table.header([Input or target], [Physical extent], [United Kingdom representation], [Role]),
    table.hline(),
    [Numerical history], [14 days], [672 steps], [Local power and permitted covariates],
    [Visual context], [Recent 3 to 6 h], [8 frames], [Cloud structure before the forecast origin],
    [Forecast horizon], [6 h], [12 steps], [Future normalized power],
    [Scored evaluation], [Daylight origins], [165,295 steps], [Shared final subset],
  )],
  caption: [Primary cross-plant forecast window. Physical time, rather than a fixed number of samples, defines each element.]
  , kind: table
)

== Scenarios and input parity

All results in the main body use the primary cross-plant scenario on the United Kingdom track. Its test plants are disjoint from both fitting partitions, and its scores include the ramp subset defined below. The chapter does not combine this evaluation with within-plant, low-history, or cross-dataset scenarios. Those tasks answer different questions and require their own declared windows and partitions.

Input parity does not mean that every model consumes every available field. A persistence reference has no visual branch, and a numerical forecaster cannot be required to use satellite frames. It means that each class receives only the information allowed by the same forecast origin and that a reported comparison names its inputs. The benchmark uses the following input matrix.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left, center, center, center, left),
    table.header([Model family], [Power history], [Permitted covariates], [Satellite frames], [Purpose in the comparison]),
    table.hline(),
    [Reference forecasts], [Yes], [Method-specific], [No], [Daily and short-horizon reference],
    [Tabular and supervised sequence models], [Yes], [Yes], [No], [Learned numerical forecasting],
    [Time-series foundation models], [Yes], [Model-specific], [No], [Transfer from broad time-series pretraining],
    [Retrieval and adapter methods], [Yes], [Where supported], [No], [Alternatives to full backbone adaptation],
    [Multimodal systems], [Yes], [Where supported], [Yes], [External visual context],
  )],
  caption: [Input-parity matrix. Each family uses information available at the same forecast origin while retaining its intended modality scope.]
  , kind: table
)

Known-ahead and observed covariates are separated explicitly. Calendar variables and solar geometry can be computed for the forecast horizon. A future weather measurement is not an operational input unless it is supplied as a genuine forecast product. This distinction prevents retrospective measurements from entering a model through the covariate interface.

== Metrics and aggregation

The protocol reports cross-plant skill score and ramp normalized mean absolute error side by side. They are co-primary measures. A lower aggregate error does not establish an improvement when ramp error moves in the opposite direction, because the two measures describe different parts of the forecasting problem.

Normalized mean absolute error and normalized root mean squared error use capacity-normalized power. This keeps a large array from dominating the score solely because its power is expressed in more watts. The skill score compares root mean squared error with Smart Persistence:

$ "SS" = 100 dot (1 - frac("NRMSE"_"model", "NRMSE"_"Smart Persistence")) $.

Smart Persistence has a score of zero by construction. Positive values indicate lower root mean squared error than the reference and negative values indicate higher error. The score is reported as a relative measure, while normalized errors retain an absolute interpretation on the capacity-normalized scale.

Ramp NMAE is calculated on rapid observed changes. For each plant, the absolute difference between consecutive valid target values is computed over the full test set. The first horizon point uses the final history observation as its predecessor. A valid daylight difference in the upper decile of that plant's changes belongs to the ramp subset. The resulting subset is fixed before model scoring and shared by every model and every evaluation pass.

Per-plant macro-averaging gives every held-out installation equal weight. The evaluator first aggregates valid steps within each plant, then averages the plant scores. This avoids allowing one long or unusually complete series to determine the reported result. It also makes the result closer to the deployment question: how does the method perform across unfamiliar installations?

== Variation across seeds and decision floors

Learned methods are evaluated with fixed random seeds, including data ordering and parameter initialization. Reported multi-seed results include their mean and standard deviation. The protocol uses two decision floors when interpreting small differences: 0.0037 for skill score and 0.0011 for ramp NMAE. These values are derived from the three-seed variability of the matched iTransformer reference and act as lower bounds; an arm with larger observed variation is judged against that larger variation.

The floors prevent a small numerical difference from being described as an architectural win. A change that falls within the relevant floor is reported as no measurable difference at the available number of seeds. The same rule applies when two methods trade aggregate skill against ramp accuracy. The chapter records both quantities, rather than selecting the more favorable column after evaluation.

== Counterfactual visual reliance

Multimodal models require an additional evaluation. Each trained set of weights is scored first with the visual pathway active and then with vision forced off. No parameters are updated between passes. The resulting differences are reported as marginal normalized error and marginal ramp NMAE.

This paired evaluation asks whether the visual stream changes the forecast made by the trained model. A comparison between separately trained numerical and multimodal models cannot answer that question because the training route, capacity, and regularization may all differ. The forced-off pass keeps those factors fixed and changes only the availability of visual information. It therefore measures reliance after training rather than the general effect of adding a visual branch to an optimization procedure.

The same target masks, ramp thresholds, forecast origins, and aggregation are used in both passes. A visual model with a nominal score advantage and a negligible forced-off difference has learned a result that is compatible with visual regularization or an unused visual interface. A positive marginal change on the ramp subset is evidence that the visual pathway contributes under the trained parameters.

== Evaluation controls

The counterfactual pass establishes whether vision matters to a prediction. It does not establish what visual property the model uses. Three further controls test the provenance and timing of the visual evidence.

The frame-order control changes the order of frames in the supplied clip. It tests whether the model depends on temporal order rather than on a collection of individual images. The stale-image control substitutes an older image sequence, testing whether the model uses conditions near the forecast origin. The cross-plant image-swap control supplies a contemporaneous sequence from another plant, testing whether the benefit depends on the correct local scene rather than on broad image statistics.

Each control reuses the trained weights and the same forecast origins. The plant-swap control requires a test loader that can identify a different donor plant for the same time; its evaluation configuration enables shuffled test batches for that reason. An ordered, series-major loader may contain only one plant at a particular batch position and cannot construct a valid donor example.

Controls are reported separately from the primary skill and ramp measures. They answer causal questions about an already trained architecture, so they do not replace the standard cross-plant score or create a new training comparison.

== Reproducible reporting

Every result records the dataset track, plant partition, physical window lengths, input modalities, forecast origin rules, random seeds, and number of scored steps. Results based on a different window builder, a different test subset, or a non-disjoint split are identified as such and are not merged into the primary cross-plant leaderboard.

The evaluation code consumes the shared dataset contract described in Chapter 3. That contract carries timestamps, masks, plant identity, image references, and quality information alongside the numerical values. It makes the comparison auditable: a reader can reconstruct the source inputs and determine whether a difference came from model behavior or from a change in the evaluation sample. The next chapter uses this protocol to organize the baseline suite.
