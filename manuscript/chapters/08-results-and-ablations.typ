// =============================================================================
//  Chapter 8 — Results and ablations
//
//  Included by ../thesis.typ. Not compilable on its own: the page setup, show
//  rules and bibliography live in the main file. Image paths are relative to
//  this file, hence `../figures/`.
// =============================================================================

= Results and ablations
<cha:ablations>

The benchmark in @cha:results establishes how the forecasting methods compare. This chapter tests whether trained visual forecasts rely on images and how scores respond to changes in image time, location, token selection, and fusion design. Unless stated otherwise, results use the UK cross-plant test, the self-attention temporal mixer, and seeds 42, 43, and 44. The practical reference floors are 0.37 percentage points in skill and 0.0011 in ramp NMAE. They are descriptive thresholds, not significance tests. A smaller difference does not establish equivalence. The main fusion result is reported in @tbl:ablation-ladder. The controls that qualify its interpretation follow.

== Interface and content controls

@tbl:ablation-registry, @tbl:ablation-content-grounding, and @tbl:ablation-registry-continued list the controls and interventions. The following paragraphs report their results and clarify which interpretation each comparison supports. The main arms are defined in @cha:method. Controls preserve the plant split, seed-matched numerical warm start, and scoring rules unless a difference is stated.

#figure(
  align(center)[#set text(size: 8.2pt)
  #set par(justify: false)
  #table(
    columns: (1.65fr, 3.35fr),
    align: (left, left),
    table.header([Control], [Intervention]),
    table.hline(),
    [Anchor-placement check], [Spatial-token payload at one shared position],
    [Paired-novelty selector check], [Paired-novelty selector in the multi-anchor layout],
  )],
  caption: [Fusion-interface controls. Outcomes are reported in the paragraphs that follow.],
  kind: table,
) <tbl:ablation-registry>

To probe fractional positioning, the anchor-placement check moves the same 98 visual tokens to one shared integer position. Across three paired seeds, mean skill changes by −0.02 percentage points and ramp NMAE by +0.0004. Both differences change direction across seeds. Neither clears its practical reference floor. The result gives no support to fractional placement as the source of the gain. It does not test other ways of combining visual and numerical tokens.

With the multi-anchor layout fixed, paired-novelty selection yields skill of 54.16 ± 0.31 percent across three seeds, versus 54.09 ± 0.15 percent with the original selector. Each ± value is the across-seed standard deviation. The 0.07-point difference is below the 0.37-point reference floor. This comparison does not show that the selector change resolves the performance gap. Per-anchor payload, daily spacing, and anchor count remain coupled in the comparison with spatial-token fusion.

#figure(
  align(center)[#set text(size: 8.2pt)
  #set par(justify: false)
  #table(
    columns: (1.65fr, 3.35fr),
    align: (left, left),
    table.header([Control], [Intervention]),
    table.hline(),
    [Stale-image check], [Visual clip shifted six hours into the past],
    [Wrong-plant-image check], [Contemporaneous clip from another plant],
    [Random-token check], [Random token choice at the same 98-token budget],
  )],
  caption: [Content-grounding ablations.],
  kind: table,
) <tbl:ablation-content-grounding>

Shifting the clip six hours into the past raises ramp NMAE by 0.0156–0.0182 across three seeds and lowers skill by about 13.5 percentage points. Ramp degradation is far above the 0.0011 floor. The stale clip performs worse than disabling vision. The fitted forecast therefore relies on timely visual evidence under this substitution.

Substituting another plant's contemporaneous clip raises ramp NMAE by 0.0163–0.0189 across three seeds. This substitution lowers skill from about 55 to 37 percent. Every seed degrades, supporting reliance on site-matched imagery. Together with the stale-image check, this establishes content grounding under the two substitutions. It does not identify which visual features carry the signal.

At the fixed 98-token budget, random selection raises ramp NMAE by 0.0039 and lowers skill by 1.52 percentage points across two paired seeds. Vision-off scores stay unchanged. The weights were trained with novelty-selected tokens, so this inference-time substitution does not separate the selector's training effect from the value of the selected content.

#figure(
  align(center)[#set text(size: 8.2pt)
  #set par(justify: false)
  #table(
    columns: (1.65fr, 3.35fr),
    align: (left, left),
    table.header([Control], [Intervention]),
    table.hline(),
    [Cached-latent-order check], [Permute cached temporal latents after visual encoding],
    [Token-budget sweep], [Sweep token count without retraining],
    [Spatial-resolution probe], [Probe mean-pooled 4×4 and 7×7 grids],
    [Mean-pooling check], [Mean pooling instead of pixel-shuffle concatenation],
    [Backbone-freeze check], [Freeze the adapted encoder blocks during visual-arm training],
  )],
  caption: [Mechanism, representation, and backbone controls.],
  kind: table,
) <tbl:ablation-registry-continued>

Permuting V-JEPA's cached temporal latents after visual encoding changes ramp NMAE by no more than 0.00030 in any seed. This is below the 0.0011 floor. The forecast is insensitive to this particular downstream reordering. The intervention cannot rule out motion information already mixed within each latent. It does not change frame order before encoding.

Evaluating without pruning the 98 visual tokens raises ramp NMAE by 0.0102 across three seeds. Keep settings of 0.3 and 0.7 raise it by 0.0022 and 0.0069. Error is lowest near the training count. Every alternate setting changes sequence length without retraining. The sweep therefore shows sensitivity to this evaluation change, rather than an independently established optimum at 98 tokens.

Under mean pooling in the spatial-resolution probe, a 4×4 grid has higher ramp R² than 7×7 at 30, 60, and 120 minutes. This challenges the 7×7 rationale under the probe's representation. The probe cannot rank the deployed pixel-shuffle pathway against a 4×4 alternative because it pools both grids.

With the number and positions of selected cells fixed, mean pooling raises three-seed skill from 55.10 to 55.64 percent. Ramp NMAE falls from 0.1440 to 0.1435. 80 percent coverage rises from 0.731 to 0.753. The skill gain appears in every paired seed. The ramp difference stays below its reference floor. Preserved sub-patch detail does not explain the original arm's gain. Spatially distinct cells remain available under mean pooling.

Freezing the adapted numerical encoder's blocks during visual-arm training yields three-seed skill of 55.29 percent and ramp NMAE of 0.1441, close to spatial-token fusion's 55.10 percent and 0.1440. Coverage falls from 0.731 to 0.710. Quantile ECE rises from 0.0359 to 0.0444. The point scores give no clear evidence that trainable upper blocks are required. The uncertainty summaries suggest a calibration cost to freezing. This check does not isolate which block changes cause it.

== Fusion results and vision-off controls

@tbl:ablation-ladder compares point forecasts across the fusion configurations and reports the change in error when vision is disabled in the same fitted weights. The numerical baseline provides the image-free reference. Values for that baseline and the pooled-image, spatial-token, and reduced-payload multi-anchor arms summarize three seeds. Full-payload multi-anchor fusion has one completed seed on a smaller eligible target set.

#figure(
  align(center)[#set text(size: 8.4pt)
  #table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    table.header([Arm], [Skill score], [Ramp NMAE], [ΔNMAE, vision off], [ΔRamp NMAE, vision off]),
    table.hline(),
    [Numerical baseline], [52.30], [0.1506], [Not applicable], [Not applicable],
    [Pooled-image baseline], [52.58], [0.1487], [0.0008], [0.0000],
    [Spatial-token fusion], [55.10], [0.1440], [0.0047], [0.0063],
    [Multi-anchor fusion], [54.09], [0.1463], [0.0031], [0.0045],
    [Full-payload multi-anchor fusion], [54.49], [0.1458], [0.0040], [0.0041],
  )],
  caption: [Fusion results on held-out UK plants. Skill is percent. Lower ramp NMAE is better. The Δ columns are error increases after vision is disabled in the same trained weights. Positive values indicate degradation.],
  kind: table,
) <tbl:ablation-ladder>

Among the main fusion configurations in @tbl:ablation-ladder, spatial-token fusion has the highest skill, 55.10 percent, and the lowest ramp NMAE, 0.1440. Relative to pooled-image fusion, its skill is higher by 2.52 percentage points and its ramp NMAE is lower by 0.0047. Both differences exceed the practical reference floors. Pooled-image fusion itself is close to the numerical baseline: skill rises from 52.30 to 52.58 percent and ramp NMAE falls from 0.1506 to 0.1487. These between-arm comparisons reflect full configurations, including differences in fusion path and training recipe. The effect of spatial tokenization remains entangled with those factors.

The vision-off columns give a more direct test of whether each fitted forecast uses its image input. Disabling vision in pooled-image fusion changes ramp NMAE by 0.0000 on average. There is no consistent ramp benefit across its three seeds. The same intervention raises spatial-token fusion's NMAE by 0.0047 on all valid daylight targets and its ramp NMAE by 0.0063. This exceeds the 0.0011 ramp floor and is consistent across seeds. Its predictions therefore rely on vision under this intervention. The larger absolute change on ramps does not establish that the benefit is concentrated there. Ramp and all-daylight errors have different baselines. The pooled-image result leaves its recipe and capacity differences unresolved. Its small between-arm advantage cannot be attributed to fusion placement alone.

Multi-anchor fusion scores 54.09 percent skill and 0.1463 ramp NMAE, below spatial-token fusion on both point measures. Its mean vision-off increases of 0.0031 NMAE and 0.0045 ramp NMAE show that the fitted model uses images. This comparison changes anchor count, daily spacing, latent depth, and per-anchor token budget together. It cannot assign the point-score gap to any one choice. The full-payload feasibility run retains four latent slices and 98 tokens at each of five anchors. It reaches 54.49 percent skill and 0.1458 ramp NMAE. Its vision-off increases are 0.0040 and 0.0041. Its 533-position sequence and smaller eligible target set distinguish it from the 141-position single-anchor arm. Further seeds and common-window rescoring are needed to interpret its placement against the shorter interface.

== Variation across held-out plants

@tbl:per-plant-ladder separates the fourteen held-out UK plants to show the breadth and limits of the aggregate gain. The first four arm columns are three-seed means. Full-payload multi-anchor values come from seed 42 on a smaller set of scored targets.

#figure(
  align(center)[#set text(size: 8.2pt)
  #table(
    columns: (0.65fr, 0.85fr, 0.85fr, 0.85fr, 0.9fr, 1fr),
    align: (left, center, center, center, center, center),
    table.header([Plant], [Numerical], [Pooled], [Spatial], [Multi-anchor], [Full payload]),
    table.hline(),
    [3333], [47.12], [47.64], [51.67], [50.89], [51.79],
    [6648], [50.03], [49.82], [52.28], [50.47], [50.84],
    [7648], [42.88], [43.22], [49.06], [47.72], [48.50],
    [7836], [44.22], [44.89], [49.37], [49.30], [49.70],
    [10793], [62.87], [63.43], [65.93], [65.09], [65.35],
    [11176], [36.29], [36.39], [39.01], [36.12], [38.54],
    [11287], [48.89], [49.34], [49.80], [49.44], [49.36],
    [12642], [44.03], [44.26], [43.76], [42.62], [42.48],
    [13388], [63.56], [64.00], [64.87], [64.34], [64.52],
    [13817], [58.61], [58.91], [61.12], [60.37], [61.31],
    [18989], [61.81], [62.65], [66.96], [66.17], [66.69],
    [26854], [41.31], [41.07], [43.11], [42.01], [42.04],
    [26933], [58.03], [57.95], [60.88], [59.79], [59.14],
    [27020], [53.94], [53.42], [54.61], [53.25], [53.11],
  )],
  caption: [NRMSE-based skill score in percent for each held-out UK plant. The first four arm columns are three-seed means. Full payload is seed 42 on fewer scored targets.],
  kind: table,
) <tbl:per-plant-ladder>

Spatial-token fusion exceeds the numerical baseline in the three-seed mean at thirteen of fourteen plants. Its largest visible gains occur where the pooled-image arm stays close to the numerical control, particularly plant 7648. The result therefore extends beyond a small set of plants already well served by pooled fusion. Plant 12642 is the exception. Several small plant-level differences could change direction across initializations because the table reports means without paired spread. The evidence supports a broad average gain within this UK fleet. It does not guarantee an improvement for every unseen plant.

Reduced-payload multi-anchor fusion trails spatial-token fusion in mean skill at every held-out plant. This consistency makes a single difficult site an unlikely explanation for the aggregate gap. It does not identify the cause. The multi-anchor arm changes daily spacing, temporal depth, and token allocation along with anchor count. The full-payload column moves above the spatial-token column at some plants and below it at others. It uses one seed and a different set of forecast origins. Those entries cannot establish that restoring payload recovers the loss at any individual plant.

Skill measures improvement relative to Smart Persistence at each plant. Its range across plants does not by itself rank their absolute forecast difficulty. Equal plant weighting also prevents sites with more valid windows from determining the chapter's headline score. The table reveals heterogeneity that the aggregate conceals. It contains neither plant-level uncertainty nor a matched analysis of the weather or image-availability conditions that might explain it.

== Skill across the six-hour horizon

@tbl:horizon-skill reports NRMSE-based skill against Smart Persistence at each half-hour lead on valid daylight targets. For each arm, plant NRMSE is macro-averaged at each lead before comparison with the corresponding Smart Persistence value. The first four arms are then averaged across three seeds. These lead-specific values are computed from the saved in-process prediction exports. Their aggregate reconstruction matches the recorded overall skill scores. The full-payload column has one seed and fewer eligible targets.

#figure(
  align(center)[#table(
    columns: (0.65fr, 0.85fr, 0.85fr, 0.85fr, 0.9fr, 1fr),
    align: (center, center, center, center, center, center),
    table.header([Lead], [Numerical], [Pooled], [Spatial], [Multi-anchor], [Full payload]),
    table.hline(),
    [0.5 h], [43.25], [43.74], [43.79], [42.11], [45.19],
    [1.0 h], [46.33], [47.08], [49.07], [47.48], [48.96],
    [1.5 h], [49.14], [50.04], [53.07], [51.51], [52.41],
    [2.0 h], [50.20], [51.38], [54.43], [53.27], [53.88],
    [2.5 h], [52.04], [53.00], [56.12], [55.23], [55.59],
    [3.0 h], [59.18], [59.33], [62.16], [61.49], [61.85],
    [3.5 h], [58.29], [58.25], [60.99], [60.45], [60.50],
    [4.0 h], [57.06], [56.91], [59.60], [59.01], [58.74],
    [4.5 h], [55.34], [55.22], [57.62], [56.94], [56.78],
    [5.0 h], [53.47], [53.14], [55.80], [54.90], [54.84],
    [5.5 h], [50.21], [49.90], [52.53], [51.45], [51.25],
    [6.0 h], [47.98], [47.85], [50.16], [49.19], [49.06],
  )],
  caption: [NRMSE-based skill score in percent by lead time on the fourteen held-out UK plants. The first four arm columns are three-seed means. Full payload is seed 42 on fewer scored targets.],
  kind: table,
) <tbl:horizon-skill>

Every arm's skill rises toward three hours and then declines. Skill is a ratio to Smart Persistence at the same lead. The mid-horizon peak means the fitted models gain the most ground relative to that reference there. The curve describes relative forecast value across lead times. It does not show that absolute forecast error falls toward three hours.

Spatial-token fusion is nearly tied with pooled-image fusion at the first half-hour step. It separates from both pooled-image fusion and the numerical baseline from one hour onward. Its advantage persists at six hours, so it is not confined to the comparison next to the most recent image. The pooled-image arm remains close to the numerical control throughout. This lead-time pattern is compatible with an added benefit from the spatial-token interface. The table compares separately trained configurations. A per-lead, same-weights vision-off analysis would be needed to attribute the shape of the gap to visual input itself.

The reduced-payload multi-anchor arm falls below spatial-token fusion at every lead. Its lower aggregate score is not confined to one part of the horizon. The gap is smallest around the middle of the horizon. These lead-specific means have no paired uncertainty estimate, and the arm changes several design choices together. The full-payload arm scores above spatial-token fusion at the first half-hour step and below it at the later steps. Its single seed and smaller eligible set make that shape provisional. It cannot establish an early-lead advantage or a payload effect.

== Seed-level stability

@tbl:seed-level-ladder reports three matched seeds for the numerical, pooled-image, spatial-token, and reduced-payload multi-anchor arms. Full-payload multi-anchor fusion has one completed seed on a smaller eligible target set.

#figure(
  align(center)[#table(
    columns: (auto, 0.5fr, 0.7fr, 0.72fr, 0.68fr, 0.72fr),
    align: (left, center, center, center, center, center),
    table.header([Arm], [Seed], [SS (%)], [Ramp NMAE], [Coverage 80], [Quantile ECE]),
    table.hline(),
    [Numerical baseline], [42], [52.75], [0.1494], [0.770], [0.0277],
    [Numerical baseline], [43], [51.97], [0.1514], [0.766], [0.0317],
    [Numerical baseline], [44], [52.17], [0.1509], [0.779], [0.0296],
    [Pooled-image baseline], [42], [52.53], [0.1483], [0.754], [0.0324],
    [Pooled-image baseline], [43], [52.18], [0.1498], [0.757], [0.0331],
    [Pooled-image baseline], [44], [53.03], [0.1478], [0.808], [0.0243],
    [Spatial-token fusion], [42], [55.00], [0.1433], [0.731], [0.0328],
    [Spatial-token fusion], [43], [54.97], [0.1448], [0.723], [0.0416],
    [Spatial-token fusion], [44], [55.32], [0.1441], [0.738], [0.0333],
    [Multi-anchor fusion], [42], [53.92], [0.1458], [0.730], [0.0371],
    [Multi-anchor fusion], [43], [54.22], [0.1456], [0.711], [0.0438],
    [Multi-anchor fusion], [44], [54.14], [0.1474], [0.738], [0.0346],
    [Full-payload multi-anchor fusion], [42], [54.49], [0.1458], [0.742], [0.0355],
  )],
  caption: [Seed-level fusion results on held-out UK plants. Skill is in percent. Lower ramp NMAE and quantile ECE are better. 80 percent coverage targets 0.80. The full-payload run scores 163,559 targets versus 165,295 for the other arms.],
  kind: table,
) <tbl:seed-level-ladder>

Across the three recorded seeds, the point metrics show limited variation within each arm. Pairing means that visual arms with the same seed begin from the same adapted numerical checkpoint. This controls the starting numerical model when comparing those runs. This does not make their later training trajectories identical.

Spatial-token fusion improves both skill and ramp NMAE over the numerical baseline in every paired seed. Reduced-payload multi-anchor fusion has lower skill and higher ramp NMAE than spatial-token fusion in all three pairs. The direction of these differences therefore persists across the recorded initializations. Three runs provide limited evidence about training variability. They do not isolate why the multi-anchor design performs worse. Anchor count, temporal depth, token allocation, and daily spacing change together.

The probabilistic measures assess different properties. CRPS summarizes quantile forecast error across the predicted distribution. Coverage counts how often observations fall between the predicted 10th and 90th percentiles; that interval should contain about 80 percent of observations. Quantile ECE measures departures between each quantile's stated probability and its observed frequency.

Calibration varies more clearly across the recorded configurations. Spatial-token coverage is below the nominal 80 percent in every seed. Its quantile ECE exceeds the numerical baseline's in every seed. Reduced-payload multi-anchor coverage is also below nominal in all three runs. Its ECE is higher than spatial-token fusion's in each matched seed. Spatial-token fusion has lower CRPS than the numerical baseline in each recorded run. The three-seed means and standard deviations are 0.0544 ± 0.0003 and 0.0574 ± 0.0004, respectively.

Lower CRPS can coexist with worse coverage or ECE. An improvement in distributional error does not guarantee correctly calibrated probabilities. Interval widths and individual quantile predictions are unavailable for this analysis. The analysis therefore cannot determine whether undercoverage comes from intervals that are too narrow, systematic displacement of those intervals, or both.

The full-payload seed has higher skill and coverage than the reduced-payload multi-anchor seed 42. Their ramp NMAE is identical when rounded to four decimals. The full-payload seed also scores fewer forecast targets. The table offers no second or third seed to show whether the observed differences persist. It documents one larger multi-anchor run; whether its per-anchor payload recovers the spatial-token result or improves calibration remains uncertain. Separately, pooled-image fusion illustrates run variation: its coverage is below nominal in two seeds and slightly above it in the third, while its point scores stay near the numerical baseline.

#pagebreak()
