
= The dataset contract
<cha:contract>

== A common sample definition

Every forecasting implementation receives the same logical sample. It contains a history of normalized power, a future target horizon, numerical covariates, visual frames, timestamps, plant identity, and masks. The contract does not force all models to use every field. A persistence baseline may read only the target history. A numerical transformer may use covariates. A multimodal model may consume the recent visual clip. The shared definition ensures that these choices are explicit.

The contract also separates values from their validity. A missing power reading, a missing visual slot, and a future target that should not be scored are represented by different masks. This distinction matters because zero is a legitimate photovoltaic value at night, while an absent value is not an observation. Conflating the two would bias both fitting and evaluation.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left, left, left),
    table.header([Sample element], [Temporal extent], [Purpose]),
    table.hline(),
    [Power history], [Fourteen days], [Local context for the held-out plant],
    [Future target], [Six hours], [Forecast value used for scoring],
    [Numerical covariates], [History and permitted future fields], [Atmospheric, solar, and calendar context],
    [Visual clip], [Recent short window], [Independent observation of cloud structure],
    [Masks and timestamps], [Whole sample], [Validity, alignment, and reproducibility],
  )],
  caption: [Logical elements of a forecasting sample.]
  , kind: table
)

== Why physical time matters

The dataset has two power cadences. A fixed number of observations therefore does not mean the same physical history in each track. Defining the task in physical time avoids an accidental advantage for a model evaluated at a coarser cadence. Fourteen days remain fourteen days whether the track has 672 or 1,344 power steps. A six-hour horizon remains six hours whether it has 12 or 24 steps.

This rule also limits aggregation. Raw step-level errors cannot be pooled across the two tracks when their steps represent different lead times. Cross-dataset summaries must use quantities that preserve the interpretation of the task, such as ranks, win rates, or scale-free skill measures. The rule is simple, but it prevents a numerical convenience from changing the scientific question.

== Missingness as part of the data-generating process

Missingness has several sources. A power value may be absent because a logger failed, because a system was offline, or because a quality check identified an implausible reading. A frame may be absent because the image source has limited diurnal coverage or because no frame falls within the requested time tolerance. These causes affect what a model can know and should not be collapsed into one generic missing token.

The primary evaluation requires valid daylight targets. It does not score night-time zero output as an easy success. History masks allow the model to distinguish observed context from gaps. Visual masks allow a multimodal architecture to fall back on numerical evidence when a clip is incomplete. A later robustness study can change those choices, but the base corpus records the evidence before any such intervention.

== Reproducible storage

The numerical table and image archive are stored separately because they have very different access patterns. A training process scans numerical columns repeatedly while it loads a small number of image clips for each batch. Packing millions of frames into site groups avoids the filesystem overhead of opening individual files, and the integer reference in the table makes each lookup auditable.

The experiment dataset is read-only. A local mirror permits inspection and reconstruction of windows without changing the data used by scheduled experiments. The primary UK window count was rebuilt from the local mirror and matches the count used by the protocol. This is a small but important reproducibility check: document claims about coverage and sample count can be verified independently of a training run.

= Dataset limitations and research opportunities
<cha:dataset-limits>

The corpus has a broad multimodal contract, but it is not a universal photovoltaic benchmark. The UK track is dominated by residential systems and one regional satellite product. The United States track is smaller and shorter. Their combination supports a meaningful next step in cross-dataset transfer, but it cannot be used to claim coverage of all deployment types.

The visual archive supplies recent cloud structure rather than a complete meteorological state. It does not directly encode panel orientation, local obstructions, or future cloud formation outside the crop. A successful visual model may learn useful correlations under these limits, while a failed model may be limited by the observation rather than by the idea of multimodal forecasting.

These limitations motivate a careful sequence of future studies. The first is a corrected US split and a cross-dataset evaluation. The second is controlled variation in visual resolution, crop size, and coverage. The third is a missing-modality study that identifies when numerical context alone is sufficient. Each of these studies depends on the data contract established in this chapter.
