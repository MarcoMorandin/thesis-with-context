
== Dataset provenance and harmonization
<cha:provenance>

=== Public sources

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

=== Power and capacity harmonization

The UK source reports energy over its sampling interval. For the forecasting record, this is converted to power in watts using the interval duration. The United States data already provide a compatible power measure. Both tracks retain measured power and a capacity-normalized target.

Installed capacity is audited per site and stored with the numerical record. The normalized target is bounded in the intended range for valid observations, yet the data still contain the real effects of clipping, missing values, and reporting differences. Normalization does not make all plants physically identical. It only removes the most direct scale difference before a shared model is fitted.

Long gaps are not transformed into smooth synthetic targets. The record retains flags and masks. Short, documented processing steps are part of construction, while uncertain or extended gaps remain unavailable. This avoids scoring a forecaster against an interpolation that may be easier to predict than the true process.

=== Time, location, and image alignment

All timestamps are represented in UTC. The numerical table contains plant coordinates, timestamp, source identifiers, and a direct image reference. The archive stores each site's image sequence with its own timestamps. A forecasting window begins with the numerical forecast origin and selects visual slots backward from that point.

This design handles the fact that the UK images arrive more frequently than power values. A frame sequence does not have to be forced onto the 30-minute power grid. The loader can take the closest valid frames inside its stated tolerance. The resulting visual timestamps are part of the sample and can be audited alongside the numeric history.

Alignment is more than a file-management detail. A future frame would create a direct leakage channel. A frame from the wrong site would change the geographic observation. A missing frame affects what the model knows. The corpus represents each of these conditions explicitly so that the forecasting task remains definable.

== Exploratory characterization
<cha:characterization>

=== Daily envelope and residual uncertainty

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

=== Ramps and the role of visual context

Ramps are changes between consecutive valid power observations. They are uncommon compared with stable intervals, but they matter because they create the largest near-term errors. The ramp subset is defined from the true target for each plant, then reused across all models. This ensures that methods are scored on the same events.

A satellite frame does not make all ramps predictable. A cloud may form outside the crop, move in a direction the model cannot infer, or affect a panel differently from its surroundings. The data nevertheless make a narrow hypothesis test possible: if recent visual structure adds information beyond numerical history, its value should be most visible during these rapid transitions.

=== Spatial and operational heterogeneity

The UK plants are not interchangeable even after capacity normalization. They have different median production, missingness, and relationships with covariates. Their locations are also dense enough that some sites share broad cloud fields. The US track adds a much larger capacity range and another image source. This heterogeneity prevents a model from treating the task as a single standardized curve.

#figure(
  image("figures/site_norm_power_boxplot.png", width: 94%),
  caption: [Distribution of normalized power by UK site. Site-level differences remain after capacity normalization and motivate plant-disjoint evaluation.]
)

#figure(
  image("figures/correlation_matrices.png", width: 92%),
  caption: [Covariate relationships across the dataset. Several variables are strongly related, yet their shared information does not fully describe local cloud effects.]
)
