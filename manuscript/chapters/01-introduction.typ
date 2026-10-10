// =============================================================================
//  Chapter 1 — Introduction
//
//  Included by ../thesis.typ. Not compilable on its own: the page setup, show
//  rules and bibliography live in the main file. Image paths are relative to
//  this file, hence `../figures/`.
// =============================================================================

= Introduction
<cha:intro>

Many machine-learning and deep-learning forecasters learn to predict an entity's future samples from its earlier observations. When the same entity supplies a long training history, the model can learn recurring temporal patterns together with characteristics specific to that entity. Testing that model on later dates measures how well it extrapolates the familiar series. By contrast, a newly observed entity must be forecast before a comparable training record exists.

A short local history can show the entity's recent state, but it contains limited evidence about seasonal behavior, unusual events, or its response to changing external conditions. The forecaster must therefore carry useful knowledge from other entities and condition that knowledge on what little is known locally. Pretraining on many time series and adapting a shared model to a related fleet are two ways to obtain such knowledge. Neither guarantees transfer: a model can learn patterns that work at the training entities yet fail when the entity being forecast changes.

Photovoltaic power makes the transfer problem concrete. The sun's position sets a regular daily pattern, but clouds can change output quickly, and the same conditions affect plants differently. Because these cloud-driven changes can occur within the six-hour forecast, recent power shows only what has already happened at the plant. Weather fields describe conditions over a wider area, while satellite images show the nearby cloud field. These sources differ in timing, spatial detail, and format, so the model must combine them for each forecast without a long training history from the target plant.

This thesis studies that challenge by constructing an aligned numerical and satellite corpus, evaluating forecasts on plants excluded from fitting, and testing interfaces between a numerical foundation model and a frozen visual encoder.

== Forecasting from limited history and imagery

Plants can differ in orientation, shading, local weather, and sensor behavior even after power is normalized by capacity. The forecaster must interpret a new plant's short record using patterns learned from other installations. The training, validation, and test sets therefore contain different plants. At each test forecast origin, the model receives fourteen days of recent power and permitted covariates. Models that use vision also receive satellite imagery. The model predicts the next six hours. The test plant supplies observations for that forecast, but no examples for fitting the model or its preprocessing statistics.

Satellite imagery may supply information that has not yet appeared in the plant's power trace. This visual context might help when a cloud system is approaching the array or when the local scene is represented more precisely than in the numerical covariates. The later controls examine these two explanations separately.

The challenge is to turn the visual field into a useful forecast input. A pretrained video encoder produces features over time and image location; the numerical foundation model processes patches of power and covariates. A single summary of the visual field is inexpensive but may discard the location of an informative feature. Spatial tokens preserve location when passed to the forecaster, at greater computational cost. The thesis compares these interfaces while keeping the underlying numerical lineage and visual source as consistent as the experiments permit.

An improved multimodal score would still leave open whether the fitted model used the images. Extra trainable components, dropout, or a different optimization path can change a forecast even when the visual content contributes little. The evaluation therefore includes a numerical control and a pass that disables vision in the same trained weights. It also uses substitutions to test whether the image is recent and belongs to the correct plant. These controls distinguish a score gain associated with multimodal training from evidence that the forecast uses the visual observation. Their implementation and limits are specified in @cha:protocol and @cha:ablations.

== Formal problem setting

The task is a forecast for one plant from information available at a specified origin. The notation below defines the plant split, input window, target, and scoring rules; @cha:protocol gives the experimental implementation.

=== Plants, inputs, and forecast windows

Plant identity and forecast time impose different constraints on a prediction. The split determines which installations can shape the fitted model, while the window determines what the model can observe when forecasting at a particular plant.

Let $i$ identify a plant and $t$ a forecast origin on its native power-sampling grid. Denote measured power at that time by $p_(i,t)$ and the plant's audited installed capacity by $c_i$. Capacity-normalized power is then $y_(i,t)=p_(i,t)/c_i$. Capacity is part of the plant record, so this normalization requires neither future production nor a scale fitted on the test plant. At $t$, the model uses recent normalized power and permitted numerical covariates describing atmospheric conditions, solar geometry, and calendar time. Multimodal methods also use a recent satellite clip when imagery is available.

Let $cal(P)$ denote the complete set of plants, with $cal(P)_"train"$, $cal(P)_"val"$, and $cal(P)_"test"$ denoting the training, validation, and test subsets. These subsets are pairwise disjoint and cover all plants:

$ cal(P) = cal(P)_"train" union cal(P)_"val" union cal(P)_"test", $

In particular,

$ cal(P)_"train" ∩ cal(P)_"test" = ∅ $

and the same condition holds for validation against the other two subsets. Model weights, preprocessing statistics, and retrieval stores are fitted without test-plant examples; @cha:protocol specifies the safeguards.

Let $H$ be the number of past power samples, $h$ the number of future power samples, and $L$ the number of satellite frames in a window. The physical history and forecast durations determine $H$ and $h$ at the plant's cadence; $L$ follows the image cadence. Let $x$ denote permitted numerical covariates, $v$ satellite frames, and $m$ validity masks. For plant $i$ at origin $t$, the window operator $cal(W)(i,t)$ pairs the inputs with the future-power target:

$ cal(W)(i,t) = (y_(i,t-H+1:t), x_(i,t-H+1:t+h), v_(i,t-L+1:t), m_(i,t-H+1:t+h), y_(i,t+1:t+h)). $

The first component is the power history through $t$; the final component is the target from the following $h$ steps and is withheld from the model. The covariate and mask components cover the indicated history and forecast intervals.

The slice of $v$ is shorthand for the $L$ frames selected by UTC time up to the origin; its indices do not imply that images share the power-sampling grid. Keeping the two clocks separate avoids discarding frames or inventing intermediate ones. @cha:method specifies the visual encoder's latent shapes and token selection.

Covariates can extend into the forecast horizon where the method permits them (@tbl:input-parity). Calendar and solar-geometry fields are known at the origin, while archived meteorological measurements stand in for an operational weather forecast. Because those archived observations are more informative than an uncertain forecast, the reported results should be read under this optimistic weather-input regime.

=== Predictions and scoring
<sec:metric-definitions>

For each window, the forecaster predicts quantiles of future capacity-normalized power, following the direct quantile-regression formulation of Chronos-2~@chronos2. This study uses nine levels from 0.1 to 0.9. Pinball loss is suited to these outputs because it weights underprediction by the quantile level and overprediction by its complement. Minimizing the loss at a given level therefore estimates that conditional quantile. For level $tau in (0,1)$, target $y$, and prediction $hat(y)^tau$, the loss is

$ rho_tau(y - hat(y)^tau) = cases(
  tau (y-hat(y)^tau) & "if " y >= hat(y)^tau,
  (1-tau)(hat(y)^tau-y) & "otherwise".
). $

Let $cal(Q)$ be the set of nine quantile levels, $b$ index windows in a batch, and $k$ index forecast steps. The target mask $m_(b,k)$ is one for a valid target and zero otherwise. The training objective combines the pinball losses over these indices:

$ cal(L)_"pin" = frac(sum_(b,k,tau) m_(b,k) rho_tau(y_(b,k)-hat(y)_(b,k)^tau), sum_(b,k) m_(b,k) dot |cal(Q)|). $

As in Chronos-2, unavailable targets contribute no loss. Dividing by the number of valid targets and quantile levels keeps the objective's scale from changing merely because a batch has more missing values. Replacing a missing target by zero would instead teach the model that a logging gap is a night-time observation.

The median quantile supplies the point forecast, denoted $hat(y)$, used by normalized mean absolute error (NMAE) and normalized root mean squared error (NRMSE). The outer quantiles define prediction intervals, while the complete quantile set supports the continuous ranked probability score (CRPS) and calibration analysis. Forecasting the horizon in one pass avoids feeding an earlier prediction back as input to a later step.

For a set of valid forecast points $cal(S)$, normalized mean absolute error and normalized root mean squared error are

$ "NMAE" = frac(1, |cal(S)|) sum_((i,t) in cal(S)) abs(y_(i,t)-hat(y)_(i,t)), $

$ "NRMSE" = sqrt(frac(1, |cal(S)|) sum_((i,t) in cal(S)) (y_(i,t)-hat(y)_(i,t))^2). $

Because $y$ is already capacity normalized, both quantities are dimensionless. @cha:protocol specifies plant weighting, valid-step masking, and the ramp subset used for the co-primary comparison.

Skill score is commonly used in PV forecasting to express improvement over a reference method~@singh2026pvforecastreview. Here Smart Persistence is the reference, so the score shows how much NRMSE the model removes beyond a simple forecast evaluated on the same targets:

$ "SS" = 100 (1-frac("NRMSE"_"model", "NRMSE"_"SP")). $

A score of zero matches Smart Persistence; positive values improve on it, and negative values perform worse. The subscript SP identifies that reference forecast. The factor of 100 makes a value of 55.10 a 55.10 percent reduction in NRMSE.

The nine quantiles also allow the forecast's uncertainty to be assessed. The reported CRPS approximation combines their pinball losses into one score, equal to twice the mean loss across quantile levels; lower is better. Interval coverage at nominal level $alpha$ is the fraction of observed targets that fall between the corresponding lower and upper quantiles. At each quantile level, the gap is between its stated probability and the fraction of targets that actually fall below it. Quantile expected calibration error averages these gaps across levels.

== Research questions

Under the plant-disjoint split and scoring rules above, the central question is whether combining a time-series foundation model with a visual foundation model improves six-hour forecasts at unseen plants. The visual arms are fitted on other UK plants but receive no fitting on test plants; the Chronos-2 direct zero-shot baseline receives no UK task adaptation. The numerical comparison tests whether adapting the backbone on training plants improves on that baseline. The visual comparisons ask whether recent, plant-aligned imagery adds information beyond power history and permitted covariates, and whether the fusion interface affects how much of that information reaches the forecast.

The benchmark also tests retrieval and lightweight adaptation as alternatives to full backbone adaptation or visual fusion. It compares overall accuracy and ramp error because rapid changes are difficult to infer from past output alone and may reveal differences between transfer methods. The vision-off and image-substitution controls address a separate question: whether a trained multimodal forecaster depends on the visual observation when making its prediction.

== Contributions

The thesis makes three contributions:

1. #strong[Corpus.] It aligns plant power, numerical context, and satellite imagery across UK and US tracks while retaining sample-level provenance.
2. #strong[Benchmark.] It defines a plant-disjoint evaluation under explicit input and scoring rules. The benchmark compares reference forecasts, supervised models, time-series foundation models, retrieval and adaptation methods, and multimodal systems.
3. #strong[Fusion analysis.] It compares pooled and spatially addressable visual interfaces. On the UK test plants, spatial-token fusion improves the reported skill and ramp NMAE over the adapted numerical control. Same-weights and content-substitution controls show reliance on recent, site-specific imagery. The discussion limits the mechanism and transfer claims to the evidence presented.

== Scope and wider relevance

Photovoltaic power is the testbed for a broader transfer problem: forecasting for an entity absent from training using a short local history and a spatial observation of its environment. A newly commissioned wind farm presents a similar combination of turbine records and approaching weather. An ungauged basin can pair a short streamflow record with satellite observations of upstream conditions. New fields, monitored districts, substations, and traffic links may also have thin local histories and spatial context. These examples share the information structure; the thesis does not measure transfer to those domains.

The thesis studies power forecasting, not irradiance reconstruction, electricity markets, dispatch optimization, or plant control. Proposed mechanisms are distinguished from experiments completed under the protocol.

== Thesis structure

The next chapter positions the transfer problem among numerical and multimodal forecasting methods. The dataset chapter documents the aligned corpus, and the methods chapter develops the fusion interfaces. The evaluation protocol fixes plant separation and scoring before the benchmark and ablation chapters report the results. The discussion interprets the visual evidence and its limits. The conclusion sets out the path from this two-track corpus to a broader transfer benchmark.

#pagebreak()
