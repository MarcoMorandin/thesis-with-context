// =============================================================================
//  Chapter 8 — Discussion
//
//  Included by ../thesis.typ. Not compilable on its own: the page setup, show
//  rules and bibliography live in the main file. Image paths are relative to
//  this file, hence `../figures/`.
// =============================================================================

= Discussion
<cha:discussion>

The experiments show visual reliance in the numerical-baseline-adapted Chronos-2 forecaster when selected spatial tokens enter its forecasting sequence directly. The tested pooled late-fusion interface shows no measurable reliance. This chapter interprets that contrast across the benchmark and ablation evidence.

== The interface is an information bottleneck

The two interfaces expose different information to Chronos-2. The pooled-image baseline compresses 784 patch cells into one learned summary, so later layers cannot address individual locations as separate tokens. Spatial-token fusion preserves selected cells and their observation positions in the forecasting sequence. This gives the forecaster direct access to spatially distinct evidence, although the architecture alone does not show which visual information it uses.

Compression does not necessarily erase all spatial information: a summary vector can encode relationships between regions in its feature dimensions. The bridge must, however, decide what to retain before the numerical forecaster processes that evidence. Separate spatial tokens let later layers select among locations while also attending to the numerical state. One possible explanation for the pooled result is that its summary overlaps with information already available in power and weather inputs, allowing the model to minimize its loss with little dependence on imagery. The vision-off result is consistent with that explanation but does not identify what the summary contains.

The same-weights vision-off controls test whether the trained forecasts use that evidence. Results in @cha:ablations show a change for spatial-token fusion and little measured change for the pooled-image model. Stale-image and wrong-plant-image substitutions also support the value of recent, site-specific content. These controls show reliance and grounding under the tested inputs; they do not identify a particular visual feature or mechanism.

Mean-pooling and anchor-placement controls narrow the interpretation. Preserving spatial cells under mean pooling retains the measured gain, while fractional positions show no measurable benefit. These results support addressable spatial information as a property of the tested interface, but they do not isolate the operation that extracts useful content or remove the recipe differences between separately trained arms.

The plant and lead breakdowns in @cha:ablations also constrain the interpretation. The spatial-token advantage appears at thirteen of the fourteen test plants, and its skill separates from the numerical baseline after the first half-hour step and persists through the six-hour horizon. This pattern extends beyond any one installation or forecast lead. Identifying the image features behind each lead's gain will require lead-specific input interventions and uncertainty estimates for the paired differences.

== Why multi-anchor fusion did not improve the forecast

Across three seeds, reduced-payload multi-anchor fusion uses images in its fitted forecast but has lower point scores than spatial-token fusion at each plant and reported lead. The comparison does not identify one cause because the multi-anchor design changes evidence age, temporal depth, and token allocation together.

The allocation differs substantially even though the total visual budgets are similar. Spatial-token fusion gives the most recent view 98 selected spatial tokens from four cached temporal latents. The reduced-payload multi-anchor arm gives each daily view 20 tokens from two latents, assigning four fifths of its visual tokens to older views. It therefore adds days of visual history while reducing detail at the current origin. The paired-novelty selector check yields no meaningful recovery at the same multi-anchor budget, so changing the selection rule alone is not a demonstrated remedy.

Full-payload multi-anchor fusion is a separate feasibility case with one completed seed and a smaller eligible target set. It retains 98 tokens and four latents at each anchor, but its descriptive score does not establish whether the larger per-anchor payload improves the forecast. The 533-position sequence may also change optimization and attention relative to the shorter interfaces; no matched-length control establishes this as the cause of the observed result.

Payload dilution, sequence length, and daily spacing are hypotheses for the reduced-payload result. Dilution means that the fixed visual budget represents the most recent scene less fully as older views are added. Those views would help only if the model learns a relation between past scenes and future power that remains useful across days. Some daily information may already be supplied by the fourteen-day numerical history. Daily anchors also describe separate scenes; they do not form a continuous clip tracking a cloud through the intervening hours. The current controls do not test the multi-anchor pathway's use of older views. Matched runs must vary per-anchor detail and anchor spacing independently before these explanations can be distinguished.

== Content grounding without a confirmed motion mechanism

The content controls support recent, site-specific visual information. They do not show downstream use of temporal order: permuting cached latents has little measured effect, and that intervention cannot test temporal information already mixed inside V-JEPA before caching. The tested interface therefore supports visual grounding, while a cloud-motion mechanism remains unestablished.

The selected cells may describe the current sky well enough that their order adds little, or V-JEPA may already combine temporal structure within each latent. The present experiments do not distinguish these explanations, so downstream sensitivity to latent order cannot determine whether the original clip contains useful motion information.

This ambiguity reflects where the intervention occurs. V-JEPA processes the eight-frame clip jointly before its four temporal latent slices are passed to the forecasting model. Reordering those slices changes what the downstream model sees, but cannot undo temporal relationships already encoded in their features. A small effect from that reordering therefore gives evidence about the downstream interface's sensitivity to latent order, not about whether the original clip contained useful motion information.

== Calibration and operational meaning

Spatial-token fusion improves point and ramp forecasts while its nominal 80 percent interval covers fewer observations than the numerical baseline's. Its CRPS improves, but coverage and quantile ECE worsen. As explained in @cha:ablations, distributional error and probability calibration are distinct: a model can reduce quantile errors while assigning probabilities that disagree with observed frequencies. Interval widths and a reliability curve showing observed frequency against predicted probability are unavailable for this analysis. The undercoverage therefore cannot be attributed to excessive concentration, bias, or both.

The reported average improvements in point and ramp scores do not determine the value of imagery at an individual plant or establish an economic threshold. Site-level variation, missed abrupt peaks, and reduced interval coverage remain important operational limits. The experiments do not measure monitoring, recalibration, or fallback behavior when imagery is absent or stale.

== Evidence layers in a composed foundation model

The fusion ladder separates four questions that are often collapsed into one leaderboard score. @tbl:evidence-layers records which of them the current experiments answer.

#figure(
  align(center)[#table(
    columns: (1.0fr, 1.45fr, 1.55fr),
    align: (left, left, left),
    table.header([Question], [Evidence in this thesis], [What remains unresolved]),
    table.hline(),
    [Representation], [Wrong-time and wrong-plant clips are harmful], [Which V-JEPA features encode the useful distinction],
    [Transport], [The raw-token arm responds to vision; the anchor-placement check finds little effect], [Pooling, dropout, and mask effects are not fully factorial],
    [Use], [Same-weights vision-off degradation in spatial-token fusion], [How reliance changes across weather regimes and datasets],
    [Mechanism], [Recent, site-specific content matters; mean pooling does not reduce the gain], [Motion before encoding and the benefit of novelty selection under matched training remain unresolved],
  )],
  caption: [Layers of evidence required for a multimodal contribution. Positive evidence at one layer does not establish the unresolved layers.],
  kind: table,
) <tbl:evidence-layers>

Representation asks whether the visual encoder distinguishes relevant states. Transport asks whether the bridge preserves those distinctions. Use asks whether the trained forecast depends on the pathway. Mechanism asks how the model extracts the useful signal. The same-weights intervention isolates use, and content substitutions test grounding; neither replaces a mechanism-specific control. This layered analysis matters when pretrained components are joined through a new interface. A favorable final score can coexist with a failed transport layer, an unused modality, or an unsupported explanation of how the modality was used.

== What cross-plant transfer establishes

The visual arms are zero-shot with respect to the identity of the fourteen test plants, not with respect to the photovoltaic domain. They are fitted on other UK plants and conditioned on fourteen days from the new plant without updating their parameters. Chronos-2 zero-shot is a different setting because it receives no UK fitting at all. Using one label for both would conceal the contribution of in-domain adaptation.

#figure(
  align(center)[#table(
    columns: (1.0fr, 1.45fr, 1.6fr),
    align: (left, left, left),
    table.header([Transfer setting], [What is held out], [Claim it can support]),
    table.hline(),
    [Temporal split], [Future intervals of known plants], [Forecasting familiar entities through time],
    [Plant-disjoint UK], [Complete plant identities], [Transfer to unseen installations within the UK corpus],
    [Region-disjoint UK], [Plants in coherent geographic regions], [Geographic extrapolation within one sensor context],
    [UK-trained inference on US], [One US test plant; region, capacity range, and imagery source differ; power is resampled to 30 min], [Initial evidence of inference across dataset tracks; not a population-level estimate],
  )],
  caption: [Transfer settings and the scope of the claim supported by each.],
  kind: table,
) <tbl:transfer-claims>

The thesis establishes plant-disjoint transfer within the UK corpus and reports a separate inference-only evaluation on one held-out US plant. That US result tests a checkpoint fitted on UK plants under a change in region, capacity range, and imagery source, with US power resampled to the UK cadence. It is evidence that cross-track inference was evaluated, but one US plant cannot establish generalization across the US fleet. Transfer to a new region with multiple test plants, or to additional sensor sources, remains unmeasured.

== Limitations

The main multi-plant evidence is limited to plant-disjoint transfer within the UK track. The supplementary US results use one seed per arm, so they do not estimate training variation. Several external comparator scores use different windows, masks, inputs, or test populations; they provide context rather than a common ranking. Spatial-token fusion remains under-calibrated at the nominal 80 percent level, and full-payload multi-anchor fusion has one seed on fewer eligible UK targets. The experiments also do not establish downstream use of cloud-motion information.

== Reproducibility, resources, and deployment boundaries

The resolved configuration is part of the evidence for each run, since it fixes the dataset track, visual cache, fusion arm, seed, and training schedule. A sequence-length budget establishes a computational boundary only; it says nothing about forecasting quality.

The reported scores are the measurements recorded during the original evaluation runs. Later evaluations of a loaded checkpoint are not substituted for those measurements, because they are separate evaluations and have not been established to reproduce the original scores. The experimental description must therefore distinguish the fitted model, the inputs and targets used for scoring, and the evaluation that produced each reported number.

For the visual arms, the relevant choices include the frozen encoder and clip construction, the plant partition, the seed-specific adapted numerical initialization, the trainable layers, and the checkpoint-selection rule. These choices determine which information can enter the forecast and what is learned from the training plants. The method and protocol chapters describe them alongside the target counts and comparison boundaries in the results chapters. To test visual reliance as well as forecast accuracy, the evaluation must repeat the same-weights vision-off pass.

Operational use introduces further requirements that the experiment does not measure: timely image retrieval and encoding, validation of capacity and location metadata, freshness gates for the visual input, and decision-specific costs for over- and under-prediction. Public source data also retain licensing and provenance obligations. Site coordinates should be exposed only to the extent required for alignment and reproducibility.

The 533-position full-payload sequence required a smaller microbatch on the available 64 GB A100 than the shorter visual arms. This compute constraint accompanies its one-seed result and limits runtime comparison; the method chapter records the run settings and available timing evidence.

#pagebreak()
