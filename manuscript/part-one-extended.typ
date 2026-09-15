
= Formal problem setting
<cha:problem>

== Forecast objects and information sets

Let a plant be indexed by i and let time follow that plant's native sampling cadence. At a forecast origin, a model receives a history of normalized power, a set of numerical covariates, and, when available, a recent sequence of satellite frames. It predicts a future horizon for the same plant.

The power target is measured power divided by installed capacity. This places a small residential installation and a larger array on a common scale while retaining the fact that their operating patterns can differ. The numerical covariates describe atmospheric conditions, solar geometry, and calendar time. The visual sequence is shorter and sampled more densely than the numerical history.

The distinction between observed and known-ahead inputs is essential. Calendar and solar geometry can be calculated for a future timestamp. A weather measurement made during the future horizon cannot be supplied to an operational forecast unless it represents a genuine weather forecast. The experimental protocol separates these cases so that privileged future observations do not appear as ordinary covariates.

== Cross-plant generalization

The cross-plant constraint applies to every fitted operation. If training, validation, and test plant sets are disjoint, no test-plant row may affect parameter fitting, feature scaling, retrieval indexes, or early stopping. This includes apparently routine operations such as estimating a per-series mean or variance.

A temporal split answers a different question. It allows a model to absorb a plant's orientation, shading pattern, inverter behavior, and bias relative to a nearby weather cell. Those are useful regularities for a site-specific system. They are not evidence that the same model will transfer to a new plant. Cross-plant evaluation asks the model to infer what it needs from a short recent context and from the relationships it learned across the training fleet.

The split does not eliminate every source of similarity. UK plants may be geographically close and may share broad weather systems. The task remains harder than a temporal split because the held-out plant has no training history, but it should not be described as a test of geographic extrapolation. That claim requires a deliberately spatial split or an external dataset.

== Forecast horizon and information regimes

The importance of available information changes with lead time. At very short horizons, persistence and local observations are often strong. Within a few hours, cloud structure in a satellite image can be informative. At longer horizons, the observed cloud field becomes stale and numerical weather forecasts become more important.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left, center, left),
    table.header([Forecast regime], [Typical lead time], [Main information source]),
    table.hline(),
    [Nowcasting], [Up to 30 min], [Recent local observations and sky imagery],
    [Intra-hour], [30 min to 2 h], [Satellite imagery and cloud evolution],
    [Intra-day], [2 to 6 h], [Satellite imagery with numerical context],
    [Day-ahead], [6 to 48 h], [Numerical weather prediction and calendar information],
    [Seasonal], [Weeks to months], [Climatology and long-term structure],
  )],
  caption: [Forecasting regimes by lead time. The six-hour horizon sits at the transition between image-led and weather-led information.]
  , kind: table
)

The six-hour horizon used in the primary study is therefore a useful fusion setting. Numerical history alone has already observed some local behavior. Imagery may still reveal recent cloud structure. Neither stream is expected to determine the whole horizon by itself.

= Dataset provenance and harmonization
<cha:provenance>

== Public sources

The corpus combines four public source families. UK power measurements originate from the domestic generation collection used in the UK track. Satellite imagery for that track comes from EUMETSAT products. The US track combines NREL PVDAQ records with GOES crops. Atmospheric covariates come from the Open-Meteo archive. Each source contributes a different aspect of the same forecast sample.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left, left, left),
    table.header([Track], [Source], [Contribution]),
    table.hline(),
    [UK power], [UK domestic generation collection], [Thirty-minute residential power and site metadata],
    [UK imagery], [EUMETSAT satellite products], [Satellite crops centered on each site],
    [US power and imagery], [NREL PVDAQ and GOES], [Fifteen-minute production records and RGB crops],
    [Atmospheric context], [Open-Meteo archive], [Temperature, radiation, cloud, wind, and precipitation variables],
  )],
  caption: [Public sources combined in the experiment dataset.]
  , kind: table
)

The sources do not arrive on a common clock, grid, or storage format. Dataset construction resolves those differences while preserving what was actually observed. Power remains at native cadence. Images remain in a separate packed archive. The numerical record stores the metadata needed to select an aligned visual clip.

== Power and capacity harmonization

The UK source reports energy over its sampling interval. For the forecasting record, this is converted to power in watts using the interval duration. The United States data already provide a compatible power measure. Both tracks retain measured power and a capacity-normalized target.

Installed capacity is audited per site and stored with the numerical record. The normalized target is bounded in the intended range for valid observations, yet the data still contain the real effects of clipping, missing values, and reporting differences. Normalization does not make all plants physically identical. It only removes the most direct scale difference before a shared model is fitted.

Long gaps are not transformed into smooth synthetic targets. The record retains flags and masks. Short, documented processing steps are part of construction, while uncertain or extended gaps remain unavailable. This avoids scoring a forecaster against an interpolation that may be easier to predict than the true process.

== Time, location, and image alignment

All timestamps are represented in UTC. The numerical table contains plant coordinates, timestamp, source identifiers, and a direct image reference. The archive stores each site's image sequence with its own timestamps. A forecasting window begins with the numerical forecast origin and selects visual slots backward from that point.

This design handles the fact that the UK images arrive more frequently than power values. A frame sequence does not have to be forced onto the 30-minute power grid. The loader can take the closest valid frames inside its stated tolerance. The resulting visual timestamps are part of the sample and can be audited alongside the numeric history.

Alignment is more than a file-management detail. A future frame would create a direct leakage channel. A frame from the wrong site would change the geographic observation. A missing frame affects what the model knows. The corpus represents each of these conditions explicitly so that the forecasting task remains definable.

= Exploratory characterization
<cha:characterization>

== Daily envelope and residual uncertainty

The average daily profile confirms an intuitive point: photovoltaic power is not equally difficult to predict throughout the day. Production is near zero at night, rises after sunrise, peaks around the middle of the day, and declines toward sunset. A calendar-aware or clear-sky reference can reproduce much of this envelope.

The conditional spread is the harder quantity. At the same solar hour, some plants experience high output while others are strongly attenuated by clouds. The model must distinguish a predictable astronomical trend from conditions that change the realized power around that trend. This is why a benchmark based only on average daily error can conceal an important failure mode.

#figure(
  image("figures/month_hour_heatmap.png", width: 90%),
  caption: [Normalized generation by month and hour. Seasonal and diurnal structure is strong, while weather-driven departures remain visible within each period.]
)

#figure(
  image("figures/feature_vs_power_hexbin.png", width: 90%),
  caption: [Relationship between selected numerical features and normalized power. Correlation provides useful context but does not remove residual local uncertainty.]
)

== Ramps and the role of visual context

Ramps are changes between consecutive valid power observations. They are uncommon compared with stable intervals, but they matter because they create the largest near-term errors. The ramp subset is defined from the true target for each plant, then reused across all models. This ensures that methods are scored on the same events.

A satellite frame does not make all ramps predictable. A cloud may form outside the crop, move in a direction the model cannot infer, or affect a panel differently from its surroundings. The data nevertheless make a narrow hypothesis test possible: if recent visual structure adds information beyond numerical history, its value should be most visible during these rapid transitions.

== Spatial and operational heterogeneity

The UK plants are not interchangeable even after capacity normalization. They have different median production, missingness, and relationships with covariates. Their locations are also dense enough that some sites share broad cloud fields. The US track adds a much larger capacity range and another image source.

This heterogeneity is useful because it prevents a model from treating the task as a single standardized curve. It also limits interpretation. A model that transfers across the UK fleet may still depend on regional regularities. A later cross-dataset experiment is needed to assess whether the learned representation survives a shift in sensor product, cadence, scale, and climate.

#figure(
  image("figures/site_norm_power_boxplot.png", width: 94%),
  caption: [Distribution of normalized power by UK site. Site-level differences remain after capacity normalization and motivate plant-disjoint evaluation.]
)

#figure(
  image("figures/correlation_matrices.png", width: 92%),
  caption: [Covariate relationships across the dataset. Several variables are strongly related, yet their shared information does not fully describe local cloud effects.]
)
