// =============================================================================
//  Chapter 9 — Conclusions
//
//  Included by ../thesis.typ. Not compilable on its own: the page setup, show
//  rules and bibliography live in the main file. Image paths are relative to
//  this file, hence `../figures/`.
// =============================================================================

= Conclusions

The thesis establishes a reproducible plant-disjoint setting for testing whether visual foundation-model fusion improves forecasts for unseen plants.

== Answer to the research question

The central question has a positive answer on the United Kingdom plant-disjoint test. Spatial-token fusion combines an adapted Chronos-2 forecaster with cached V-JEPA 2.1 representations and reaches 55.10 percent skill and 0.1440 ramp NMAE on plants excluded from fitting. The numerical Chronos-2 control reaches 52.30 percent skill and 0.1506 ramp NMAE under the same test protocol. Disabling the visual pathway in the trained spatial-token model raises ramp NMAE by 0.0063, showing that the forecast uses satellite information at inference time. Because this model already receives power history and horizon-aligned weather covariates, the measured visual gain is additional to those inputs.

The content controls make that result more specific. Replacing the current clip with a stale one or imagery from another plant degrades the forecast substantially. The model therefore benefits from recent, site-specific visual evidence, rather than merely from adding visual parameters to the architecture. The tested permutation of cached temporal latents has a small effect; it does not resolve how much motion information the visual encoder captured before caching.

The result also clarifies the value of the interface. A pooled visual summary has little measured effect when switched off in the trained model, whereas selected, spatially addressable tokens remain useful inside the numerical forecasting sequence. This is an observed advantage of the tested systems; the separate controls in @cha:ablations identify which design explanations remain open.

Full-payload multi-anchor fusion is a completed feasibility case with one seed on a smaller eligible target set. It reaches 54.49 percent skill and 0.1458 ramp NMAE; these measurements are descriptive and do not establish a performance recovery over the shorter interfaces.

== What the results establish

The aligned corpus and plant-disjoint protocol make transfer to unseen plants measurable across the UK and US tracks. The results clarify the scope of success: the UK evaluation measures transfer within the photovoltaic domain after fitting on other plants, while the single-plant US test provides an initial cross-track check.

The fusion experiments also separate forecast quality from modality use. Spatial-token fusion improves on the adapted numerical control, and the same-weights and content-substitution controls indicate reliance on recent, plant-specific imagery. The pooled summary shows little measured reliance. These results favor the tested spatial-token interface; the visual features and any motion mechanism remain unresolved.

== Future work

The present experiments establish a strong within-UK transfer result and report initial inference on one held-out US plant. Because the corpus and model design leave room to test a broader claim, the next phase should enlarge the geographic evidence and the set of model comparisons without losing the alignment, plant separation, and visual-use controls that make the current results interpretable.

=== From two tracks to a worldwide benchmark

The worldwide benchmark should be designed around shifts in climate, installation type, and imaging system, with entire regions or sensor families held out from training. This would test whether a model transfers beyond familiar sources while drawing on a much larger corpus.

The UK and US tracks establish the dataset contract, but their geographic range is only a beginning. The current cross-track evaluation includes one US test plant; additional held-out US plants and repeated runs are needed to estimate how consistently the models transfer. A larger release should add plants and longer observation periods in both countries, then incorporate public numerical power records from further regions and pair them with the appropriate satellite products. The same schema can retain each source's native cadence, capacity metadata, quality flags, image timing, and provenance. Coverage across climates, installation scales, and imaging systems would make the corpus useful for studying which representations transfer and where they fail.

The aim is a first worldwide-scale multimodal benchmark for forecasting at unseen plants. It would extend the present plant-disjoint test with held-out regions and satellite sources, so within-region transfer can be distinguished from adaptation to a new climate or sensor. Common input contracts, origin definitions, masks, and plant-weighted metrics would let numerical and multimodal methods be compared as the corpus grows. A systematic survey of released collections should accompany the benchmark so that its claim of priority rests on documented coverage.

=== An agent-assisted dataset construction pipeline

The dataset can grow only if each new source can be checked as carefully as the present tracks. Repeating every discovery, mapping, and alignment step by hand would make a worldwide release difficult to maintain. Automation should therefore produce inspectable proposals that curators approve before records enter the corpus.

An agent-assisted pipeline could discover candidate public archives, extract their documentation and licenses, propose mappings from local fields to the common schema, and flag missing capacity, timestamps, coordinates, or quality information. A Pydantic data contract could validate each proposed mapping against the common schema, while deterministic checks verify units, time zones, quality flags, and timestamp consistency. An Airflow DAG could sequence normalization, gap masking, and frame matching. Each addition would produce a versioned report of source files, transformations, exclusions, and visual-alignment checks for curator review before admission to the corpus.

This workflow could be rerun for new sources under the same checks. Human review would remain necessary for licensing, ambiguous metadata, and failures that automated checks cannot resolve. A final validation stage should test temporal and spatial alignment, duplicate plants, split leakage, and whether the resulting windows can be reconstructed from the stored provenance. This pipeline is a route to finishing a large-scale corpus while keeping the benchmark auditable.

=== Stronger multi-anchor forecasting

More visual history creates a question about access as well as capacity. The five-anchor arm gives the forecaster much more spatial evidence, but its horizon-wide query must still turn that evidence into forecasts for different lead times. Future experiments should study both how much visual context to retain and how each forecast lead retrieves it.

The full-payload multi-anchor arm shows that five detailed daily visual observations can be processed in one forecast, although its 533-position sequence makes training substantially more expensive than the single-anchor arm. The first step is to complete the remaining seeds and compare both arms on the same forecast origins. Larger-memory accelerators could then support systematic study of longer visual histories, finer spatial payloads, and visual-encoder adaptation without the present microbatch constraint. The effects of these extensions on accuracy and calibration must be measured; additional compute alone is not evidence of a gain.

A more targeted extension would change how the forecast reads the five anchors. The current model uses one future-query token to request the whole horizon after visual and numerical tokens have been combined. A new arm could retain that interleaved multi-anchor context while introducing separate, target-free queries for different forecast leads. Cross-attention in the forecasting layers would let each query select relevant visual cells from the five anchors; spatial coordinates and anchor-time embeddings would preserve where and when those cells were observed. An early lead and a later lead could then draw on different locations and days. Query construction would use only information available at the forecast origin, such as lead position and permitted future covariates. Its value should be tested against the existing full-payload arm with the same frames, plant split, scored origins, seeds, and visual-off control; runtime and memory should be reported alongside forecast quality.

=== Extending the external comparison

A benchmark centered on unseen plants should continue to admit architectures developed outside the present fusion ladder. Systems that encode imagery or join modalities differently are especially informative: they test whether the transfer result depends on one particular interface.

PV-LLM~@pvllm is a particularly informative comparison because it fuses satellite information with numerical history and metadata-derived text prompts inside a language-model backbone. Its publication does not link an official implementation, so an evaluation would require a released implementation or a clearly documented reproduction. FusionSF~@fusionsf has an official implementation and supplies a complementary test of vector-quantized multimodal fusion. Both should be adapted to the same plant-disjoint windows and allowed inputs before their results are compared with the present models. As more methods become available, the benchmark can test whether their reported gains persist when the target is transfer to a new plant rather than prediction at a familiar one.
