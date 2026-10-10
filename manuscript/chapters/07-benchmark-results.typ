// =============================================================================
//  Chapter 6 — Benchmark results and baseline comparison
//
//  Included by ../thesis.typ. Not compilable on its own: the page setup, show
//  rules and bibliography live in the main file. Image paths are relative to
//  this file, hence `../figures/`.
// =============================================================================

= Benchmark results
<cha:results>

This chapter reports benchmark performance for the forecasting methods considered in the thesis. The evaluation follows the protocol defined in @cha:protocol and includes both aggregate scores and examples of forecast trajectories.

== Numerical results

This section compares skill score and ramp NMAE across reference forecasts, numerical models, time-series foundation models, and multimodal forecasters.

=== Reference forecasts

@tbl:results-reference reports three rule-based reference forecasts under the common UK protocol. They show how much skill follows from carrying forward recent power or exploiting its recurring daily pattern.

#figure(
  align(center)[#set text(size: 8.4pt)
  #table(
    columns: (1.6fr, 0.9fr, 0.95fr),
    align: (left, center, center),
    table.header([Method], [Skill score], [Ramp NMAE]),
    table.hline(),
    [Persistence], [1.41], [0.2550],
    [Seasonal naive], [10.68], [0.2056],
    [Hourly climatology], [23.37], [0.1665],
  )],
  caption: [Rule-based reference forecasts on the primary cross-plant test set. Higher skill and lower ramp NMAE are better.]
  , kind: table
) <tbl:results-reference>

The scores rise from persistence through seasonal naive to hourly climatology, showing the value of recurring daily structure. Hourly climatology also lowers ramp error. Its fixed per-hour mean cannot adapt to local conditions that change from one forecast day to the next.

=== Tabular models

The models in @tbl:results-tabular use engineered lagged numerical features. They test how far learned tabular representations can improve on the rule-based references without modeling the input as a sequence.

#figure(
  align(center)[#set text(size:  8.4pt)
  #table(
    columns: (1.7fr, 0.95fr, 1fr),
    align: (left, center, center),
    table.header([Method], [Skill score], [Ramp NMAE]),
    table.hline(),
    [LightGBM], [38.54], [0.1672],
    [TabPFN], [40.63], [0.1631],
  )],
  caption: [Tabular models on the primary cross-plant test set. Higher skill and lower ramp NMAE are better.]
  , kind: table
) <tbl:results-tabular>

=== Supervised sequence models

@tbl:results-supervised lists numerical sequence models trained on the benchmark training plants.

#figure(
  align(center)[#set text(size: 8.4pt)
  #table(
    columns: (1.7fr, 0.95fr, 1fr),
    align: (left, center, center),
    table.header([Method], [Skill score], [Ramp NMAE]),
    table.hline(),
    [iTransformer], [52.57], [0.1445],
    [PatchTST], [45.81], [0.1543],
    [Temporal Fusion Transformer], [42.64], [0.1605],
    [MLP baseline], [42.19], [0.1624],
    [DLinear], [32.31], [0.1746],
  )],
  caption: [Supervised numerical sequence baselines.]
  , kind: table
) <tbl:results-supervised>

=== Time-series foundation models zero-shot

@tbl:results-foundation-transfer lists time-series foundation models evaluated without task-specific fitting.

#figure(
  align(center)[#set text(size: 8.4pt)
  #table(
    columns: (1.7fr, 0.95fr, 1fr),
    align: (left, center, center),
    table.header([Method], [Skill score], [Ramp NMAE]),
    table.hline(),
    [Chronos-2 zero-shot], [47.37], [0.1544],
    [TiRex zero-shot], [28.73], [0.1826],
    [TimesFM zero-shot], [27.08], [0.1902],
    [TTM zero-shot], [-8.07], [0.2922],
  )],
  caption: [Protocol-aligned zero-shot time-series foundation models.]
  , kind: table
) <tbl:results-foundation-transfer>

Chronos-2 has the strongest zero-shot result in this table, with skill 47.37 and ramp NMAE 0.1544.
TTM's negative zero-shot skill shows that pretraining alone does not guarantee transfer to this plant-disjoint task.

=== Time-series foundation models adaptation models

@tbl:results-foundation-adaptation compares models that adapt a pretrained time-series foundation model to the training plants, either by fine-tuning or with a covariate adapter.

#figure(
  align(center)[#set text(size: 8.4pt)
  #table(
    columns: (1.7fr, 0.95fr, 1fr),
    align: (left, center, center),
    table.header([Method], [Skill score], [Ramp NMAE]),
    table.hline(),
    [Chronos-2 fine-tuned], [52.30], [0.1506],
    [CoRA], [37.98], [0.1624],
    [TTM fine-tuned], [36.01], [0.1716],
  )],
  caption: [Protocol-aligned time-series foundation-model adaptation results.]
  , kind: table
) <tbl:results-foundation-adaptation>

The fine-tuned Chronos-2 control has skill 52.30 and ramp NMAE 0.1506. Its score is 4.93 points above the zero-shot Chronos-2 result, and its ramp NMAE is lower by 0.0038. By comparison, iTransformer has similar skill but lower ramp NMAE.

=== Retrieval methods

TS-RAG and Cross-RAG are retrieval-augmented forecasters. Their recorded scores use forecast windows selected by their own evaluation procedures and a proxy daylight mask. These differ from the common window and target-selection rules in @cha:protocol. The resulting averages can include different target timestamps. Their scores provide context and are not protocol-aligned comparisons. @tbl:retrieval-native reports these scores.

#figure(
  align(center)[#set text(size: 8.2pt)
  #table(
    columns: (1.3fr, 0.85fr, 0.65fr),
    align: (left, center, center),
    table.header([Method], [Skill score], [Ramp NMAE]),
    table.hline(),
    [TS-RAG], [47.79], [0.1568],
    [Cross-RAG], [47.68], [0.1545],
  )],
  caption: [Recorded retrieval-model results.]
  , kind: table
) <tbl:retrieval-native>

Ramp error depends on the observed changes present in the selected windows, so it also depends on this evaluation boundary. A numerical difference from a protocol-aligned model cannot be attributed to the forecasting method unless both are scored on the same targets with the same ramp-selection rule.

=== Multimodal baselines

The following tables separate protocol-aligned results from contextual scores recorded with native windows or masks. Only rows within the protocol-aligned table share the plant split, forecast origins, target masks, and scoring rules defined in @cha:protocol.

@tbl:results-multimodal reports protocol-aligned benchmark results for MMTSFM and fine-tuned Aurora.

#figure(
  align(center)[#set text(size: 8.4pt)
  #table(
    columns: (1.7fr, 0.95fr, 1fr),
    align: (left, center, center),
    table.header([Method], [Skill score], [Ramp NMAE]),
    table.hline(),
    [MMTSFM spatial-token fusion], [55.10], [0.1440],
    [MMTSFM multi-anchor fusion], [54.09], [0.1463],
    [MMTSFM pooled-image baseline], [52.58], [0.1487],
    [Aurora fine-tuned @aurora], [40.52], [0.1555],
  )],
  caption: [Protocol-aligned multimodal results. Higher skill and lower ramp NMAE are better.]
  , kind: table
) <tbl:results-multimodal>

The matched vision-off controls in @cha:ablations test whether the fitted models depend on their visual pathways.

Spatial-token fusion has the highest skill score among these protocol-aligned multimodal rows. Within the MMTSFM ladder, its ramp NMAE is 0.1440, compared with 0.1487 for pooled-image fusion and 0.1506 for the numerical control. The separately trained fine-tuned Aurora row uses the same test protocol but a different model family.

=== Full-payload multi-anchor feasibility result

Full-payload multi-anchor fusion has one completed seed and a smaller eligible target set, with 163,559 scored target steps compared with 165,295 for the single-anchor arm. Because it requires cached imagery at all five daily anchors, origins missing a required entry are excluded. This smaller scored set can change the average error independently of the model. Its recorded score is therefore descriptive pending additional seeds and scoring both interfaces on their shared eligible windows.

#figure(
  align(center)[#set text(size: 8.4pt)
  #table(
    columns: (1.7fr, 0.95fr, 1fr),
    align: (left, center, center),
    table.header([Method], [Skill score], [Ramp NMAE]),
    table.hline(),
    [MMTSFM full-payload multi-anchor fusion], [54.49], [0.1458],
  )],
  caption: [Full-payload multi-anchor feasibility result on its smaller eligible target set.],
  kind: table,
) <tbl:results-full-payload>

=== Contextual multimodal scores on native windows or masks

The following methods use native evaluation windows or masks, and their test populations may differ. The ramp NMAE values were recomputed from saved predictions using a per-plant top-decile threshold. The archives omit the last history value, so their ramp subsets exclude the first forecast step. These values describe each model's recorded evaluation set and do not support direct ranking against the protocol-aligned results.

#figure(
  align(center)[#set text(size: 8.4pt)
  #table(
    columns: (1.7fr, 0.95fr, 1fr),
    align: (left, center, center),
    table.header([Method], [Skill score], [Ramp NMAE]),
    table.hline(),
    [Time-VLM @timevlm], [54.04], [0.1374],
    [Solar-VLM @solarvlm], [43.96], [0.1514],
    [Aurora zero-shot @aurora], [23.24], [0.2051],
    [SUNSET @sunset], [21.62], [0.1757],
    [UniCast @unicast], [12.11], [0.2302],
  )],
  caption: [Contextual multimodal scores on native evaluation windows or masks.],
  kind: table,
) <tbl:results-multimodal-context>

== Zero-shot transfer from the UK to one US plant

The preceding tables report the primary UK plant-disjoint benchmark. @tbl:us-mmtsfm-transfer adds an inference-only transfer evaluation. Checkpoints fitted on UK training plants are applied to the held-out US plant without fitting on US observations. The US power series is resampled to 30-minute cadence, so its twelve-step horizon also spans six hours. Each row uses the same test plant and 6,053 valid target steps. The table reports one run per arm with seed 42. The single US test plant and single seed make these descriptive measurements, not estimates of performance across US installations.

#figure(
  align(center)[#set text(size: 8.2pt)
  #table(
    columns: (1.65fr, 0.72fr, 0.78fr, 0.82fr, 1.0fr),
    align: (left, center, center, center, center),
    table.header([UK-trained MMTSFM arm], [Skill (%)], [NMAE], [Ramp NMAE], [Vision-off change in NMAE (overall / ramp)]),
    table.hline(),
    [Numerical control], [16.45], [0.2048], [0.2317], [—],
    [Pooled-image fusion], [7.96], [0.2208], [0.2380], [+0.0048 / −0.0066],
    [Spatial-token fusion], [19.60], [0.1962], [0.2301], [−0.0006 / −0.0032],
    [Reduced-payload multi-anchor fusion], [20.06], [0.1960], [0.2192], [−0.0001 / +0.0114],
    [Higher-payload multi-anchor fusion (70 tokens per daily anchor)], [5.62], [0.2311], [0.2355], [−0.0397 / −0.0382],
  )],
  caption: [Inference-only UK-to-US transfer on one held-out US plant (seed 42; 6,053 valid target steps). Skill is relative to Smart Persistence on this US test set. The final column gives NMAE with vision disabled minus NMAE with vision enabled for overall and ramp targets, respectively. Positive values indicate that disabling vision worsens error.],
  kind: table,
) <tbl:us-mmtsfm-transfer>

Spatial-token and reduced-payload multi-anchor fusion have the highest skill scores in this one-plant evaluation, at 19.60 and 20.06 percent. These two arms also have ramp NMAE values below the numerical control's 0.2317, with the largest reduction for reduced-payload multi-anchor fusion. By contrast, pooled-image and the higher-payload multi-anchor variant score below the numerical control on both reported point measures. This interface-dependent ranking shows that adding imagery alone does not ensure better transfer. The higher-payload variant uses 70 selected tokens at each of five daily anchors, fewer than the 98 tokens per anchor used by the UK full-payload experiment. The two configurations should therefore not be conflated.

The vision-off comparison sharpens this pattern. For pooled-image fusion, vision improves overall NMAE but worsens ramp NMAE; for spatial-token fusion, disabling vision slightly lowers both errors. Reduced-payload multi-anchor fusion shows the reverse pattern: disabling vision changes overall NMAE little but worsens ramp NMAE by 0.0114. For the higher-payload multi-anchor variant, disabling vision lowers both errors. Thus, only the reduced-payload multi-anchor run shows a clear positive ramp contribution from vision on this US plant, while the other arms show little benefit or worse ramp error. With one seed and one held-out US plant, these counterfactuals leave the interface ranking and any general mechanism unresolved. Broader conclusions require additional US plants and repeated seeds.

== Visual inspection of the forecasts

The result tables show overall performance, but they do not show where errors occur. @fig:baseline-scatters compares observed and predicted normalized power for fine-tuned Chronos-2, iTransformer, LightGBM, and fine-tuned Aurora. In each panel, observed power is on the horizontal axis and predicted power is on the vertical axis. Points on the dashed diagonal are accurate predictions. Points above it are overpredictions, and points below it are underpredictions. Colour shows where observations are most concentrated. The plots pool valid daylight observations from the 14 held-out plants. Benchmark scores weight plants equally, so the densest parts of a plot do not necessarily have the greatest influence on its score.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 0.55em,
    image("../figures/scatter_mmtsfm_s1.png", width: 100%),
    image("../figures/scatter_itransformer.png", width: 100%),
    image("../figures/scatter_lightgbm.png", width: 100%),
    image("../figures/scatter_aurora_ft.png", width: 100%),
  ),
  caption: [Predicted versus observed normalized power for four benchmark models. Colour shows observation density on a logarithmic scale. The dashed line marks perfect predictions.],
) <fig:baseline-scatters>

In all four panels, the densest observations follow the diagonal at low and moderate output. Predictions spread more widely as observed power rises. High observed values are often predicted too low. This is clearest for LightGBM, whose predictions level off before the observed range ends. iTransformer reaches higher predicted values, although its scatter remains broad. Fine-tuned Chronos-2 and Aurora also show underprediction near the upper end. Each panel contains positive predictions when observed output is close to zero. These patterns show where the models make errors that their aggregate scores cannot locate.

@fig:mmtsfm-scatters compares the three visual fusion arms on the same daylight observations and axes. It shows how the distribution of prediction errors changes across the fusion interfaces.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.4em,
    image("../figures/scatter_mmtsfm_s2a.png", width: 100%),
    image("../figures/scatter_mmtsfm_s2d.png", width: 100%),
    image("../figures/scatter_mmtsfm_s2e.png", width: 100%),
  ),
  caption: [Predicted versus observed normalized power for the three visual fusion arms, shown with identical axes and daylight masking.],
) <fig:mmtsfm-scatters>

All three arms follow the overall rise in observed power. Their predictions spread apart at higher output. Pooled-image fusion has a broad cloud below the diagonal when observed power is high. Spatial-token fusion places more predictions near the diagonal in this range, although it still underpredicts some large values. The multi-anchor panel is more dispersed than the spatial-token panel and shows no comparable tightening around the diagonal. Each arm also makes some positive predictions when observed power is close to zero. 

@fig:tier-traces follows three ramp windows from an evaluation plant. Each row shows the same window in both columns. The left column compares fine-tuned Chronos-2, iTransformer, LightGBM, and fine-tuned Aurora. The right compares pooled-image, spatial-token, and multi-anchor fusion. The pale grey line at negative hours is observed power from the previous day. The break in the horizontal axis skips the intervening night. The dotted line at hour zero marks the forecast origin. To its right, the dark grey line is the observed outcome, and the coloured dashed lines are six-hour forecasts. Their direction, timing, and height show which changes each model follows or misses.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 0.7em,
    image("../figures/traces_tsfm.png", width: 100%),
    image("../figures/traces_multimodal.png", width: 100%),
  ),
  caption: [Forecast trajectories for three ramp windows from plant 7648. The same windows appear in both columns.],
) <fig:tier-traces>

The three windows expose different errors. In window 218, the observed output rises above most forecasts. iTransformer comes closest to the peak among the benchmark models. Spatial-token fusion is closest among the visual arms. In window 1259, output remains low after a brief rise. LightGBM and spatial-token fusion follow that lower level more closely. Several other forecasts stay too high. In window 556, spatial-token and multi-anchor fusion capture the sharp rise but still miss the peak. The pooled-image baseline and fine-tuned Chronos-2 produce flatter, lower forecasts for that window.

@fig:worked-trace shows a pronounced upward ramp within one six-hour forecast. It uses the same broken time axis as @fig:tier-traces. The pale grey segment shows observations from the previous day. The dark grey line on the right shows the outcome after the forecast origin. The coloured dashed lines show the model forecasts.

#figure(
  image("../figures/traces_comparison_11287_w1021.png", width: 94%),
  caption: [Six-hour forecast across an upward ramp at plant 11287, window 1021.]
) <fig:worked-trace>

Observed power rises sharply after the forecast origin, with brief peaks around hours 2.5 and 5. Spatial-token fusion follows the overall climb but smooths those peaks; multi-anchor fusion also rises, though at a lower level. The numerical control and pooled-image baseline flatten earlier. This trace shows how the forecasts differ during one ramp event.

#pagebreak()
