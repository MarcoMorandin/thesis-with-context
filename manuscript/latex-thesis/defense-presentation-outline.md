# Master's thesis defense — 15-minute presentation

## Approach

Use **24 slides with one specific concept per slide**, little written content, and one main visual. The talk lasts **15 minutes, excluding questions**. Dataset construction receives **4 minutes 5 seconds**, reflecting its scientific role in making the transfer evaluation credible. Slides support the spoken explanation rather than reproduce thesis paragraphs.

The timing table, outline, ASCII layouts, and spoken scripts use the same slide numbers, from 1 to 24. Dataset construction occupies slides 5–9.

**Research question:** Can combining time-series and visual foundation models improve forecasting at plants unseen during training—and can we verify that the model actually uses the imagery?

PV forecasting is the testbed. The scientific contribution concerns multimodal fusion and cross-plant generalization.

Sources: [Introduction](chapter1.tex), [Dataset](chapter3.tex), [Method](chapter4.tex), [Experiments](chapter5.tex), [Discussion](chapter6.tex), and [Conclusions](chapter7.tex). Result provenance: [baseline test report](../../report/BASELINE_TEST_REPORT.md) and [generated results record](../../baselines/results/ALL_RESULTS.md).

## Visual design

- One message title, one visual, and at most two short supporting statements per slide.
- Put explanations and qualifications in speaker notes.
- Use a 16:9 canvas, light background, dark text, and generous whitespace.
- Aim for 32–36 pt titles and 24–28 pt labels. Avoid screenshots of complete thesis tables.
- Assign consistent colors to the numerical control, spatial-token, and multi-anchor methods; label methods directly.
- Preserve units, metric direction, evaluation population, and seed aggregation on result charts.
- Reuse thesis figures, enlarging labels and simplifying panels while preserving scientific content.
- Keep source references and slide numbers discreet. Leave detailed equations and secondary ablations in backup slides.

## Timing

| Slide | Concept | Duration | End |
|---|---|---:|---:|
| 1 | Research question | 0:15 | 0:15 |
| 2 | Unseen-plant forecasting | 0:30 | 0:45 |
| 3 | Complementary imagery | 0:30 | 1:15 |
| 4 | Thesis contributions | 0:20 | 1:35 |
| 5 | A corpus designed for the scientific question | 0:45 | 2:20 |
| 6 | Harmonizing measurements and cadences | 0:50 | 3:10 |
| 7 | Causal time and location alignment | 0:50 | 4:00 |
| 8 | Missingness and measurement quality | 0:50 | 4:50 |
| 9 | Auditable sample contract | 0:50 | 5:40 |
| 10 | Forecast window | 0:30 | 6:10 |
| 11 | Plant-disjoint evaluation | 0:40 | 6:50 |
| 12 | Performance measures | 0:30 | 7:20 |
| 13 | Foundation-model backbones | 0:40 | 8:00 |
| 14 | Pooled-image fusion | 0:20 | 8:20 |
| 15 | Spatial-token fusion | 0:35 | 8:55 |
| 16 | Multi-anchor fusion | 0:30 | 9:25 |
| 17 | UK point skill | 0:45 | 10:10 |
| 18 | UK ramp error | 0:35 | 10:45 |
| 19 | Vision-off control | 0:40 | 11:25 |
| 20 | Image-alignment controls | 0:40 | 12:05 |
| 21 | US point skill | 0:45 | 12:50 |
| 22 | US ramp error | 0:30 | 13:20 |
| 23 | Calibration limitation | 1:05 | 14:25 |
| 24 | Final answer | 0:35 | 15:00 |

## Slide skeleton and high-level design

### 1. Forecasting at unseen plants with multimodal foundation models

**On slide:** Official title, candidate and institutional information, and the research-question subtitle.
**Design:** Large title on the left; one satellite crop on the right; small institutional logo.
**Speaker focus:** Introduce the question in one sentence. Explain that the thesis evaluates fusion quality and evidence of visual reliance.

### 2. A new plant must be forecast without target-plant fitting

**On slide:** “Short local history. Fixed model parameters.”
**Design:** Training plants on one side and the unseen target on the other. Highlight the target and the training boundary.
**Visual candidate:** [Cross-plant context](figures/cross-plant-forecast-context.drawio.png).
**Speaker focus:** Explain the transfer challenge. Models train on other plants and receive no parameter fitting at evaluation plants.

### 3. Satellite imagery offers context beyond past production

**On slide:** Numerical-history and satellite-context labels.
**Design:** Two large aligned panels, with one annotated production transition.
**Visual candidate:** [Power and satellite context](figures/pv-power-satellite-context.png).
**Speaker focus:** Explain the complementary information sources. Whether imagery improves forecasts is an empirical question.

### 4. The thesis contributes a corpus, a benchmark, and a fusion analysis

**On slide:** “Corpus · Benchmark · Fusion analysis,” with one short phrase beneath each.
**Design:** Three spacious columns; no literature table.
**Speaker focus:** State the three contributions as one overview of the research deliverables, following [chapter 1](chapter1.tex).

### 5. The research question determines the dataset requirements

**On slide:** “Measured power + numerical context + aligned satellite observations.” UK: 100 systems; US: 10 systems in the corpus.
**Design:** One large plant map with two source-track labels: UK PV/EUMETSAT and NREL PVDAQ/GOES. A small footer identifies Open-Meteo atmospheric covariates. Avoid a full source table.
**Visual candidate:** [Plant map](figures/site_map.png).
**Speaker focus:** Explain why cross-plant multimodal evaluation requires this combination. Among the collections surveyed in the thesis, none supplied the combination of cross-plant evaluation, native-cadence power and imagery, and sample-level provenance used here. Restrict this claim to the surveyed collections. Eight US plants are eligible for the transfer evaluation, distinct from the ten-site corpus.

### 6. Harmonization preserves the meaning of the observations

**On slide:** “Common physical meaning; preserved native observations.”
**Design:** Two source rows: UK energy → power → capacity-normalized target; US power → capacity-normalized target. Below, show 30-minute UK power and 15-minute US power on the same physical-time axis.
**Speaker focus:** Explain interval-energy conversion, audited installed capacity, and physical-time windows. Capacity normalization enables comparison across plant sizes without eliminating plant-specific behavior. Corpus storage preserves native cadence; US benchmark power is subsequently resampled to 30 minutes for evaluation parity. Do not imply interpolation creates measurements.

### 7. Each image must match the plant and precede the forecast origin

**On slide:** “Correct site. Exact timestamp. Before the origin.”
**Design:** Enlarge a timeline with the forecast origin as a vertical boundary. Show valid historical frames on the left and an excluded future frame on the right. Use a small location label to indicate the site-centered crop.
**Speaker focus:** Explain UTC timestamps, coordinates, archive references, and original frame timestamps. In the UK, select the closest valid preceding frame within a 15-minute window. This prevents future-image leakage and makes alignment inspectable. Distinguish image availability from horizon covariates, whose reanalysis limitation is discussed later.

### 8. Missing data must remain distinct from measured low power

**On slide:** “Observed zero ≠ missing observation.”
**Design:** A single illustrative trace with three labeled cases: valid low/zero production, missing interval, and flagged faulty signal. Mark gaps explicitly rather than drawing an interpolated curve. Label any schematic as illustrative.
**Speaker focus:** Explain why outages, stuck signals, cloud-related low generation, and absent measurements cannot be treated as interchangeable. Preserve quality labels and separate masks. Extended gaps remain unobserved instead of becoming synthetic targets.

### 9. The sample contract makes the benchmark auditable

**On slide:** “Every forecast is traceable to its plant, observations, and validity.”
**Design:** One sample card with three groups: values (history, covariates, imagery, targets), identity/time (site, source, UTC timestamps), and validity (distinct masks). Avoid implementation field names. Use arrows from source observations to the sample card.
**Speaker focus:** Explain model-independent sample reconstruction and plant identity preservation. Separate input availability from target scoring validity. Numerical records and imagery are stored separately but linked through timestamp-exact references. The contribution is a reproducible experimental basis, not merely a collection of files.

**Source for slides 5–9:** [Dataset construction, provenance, harmonization, and contract](chapter3.tex).

### 10. Fourteen days of numerical history support a six-hour forecast

**On slide:** “14 days of numerical context → 6-hour forecast.”
**Design:** One timeline with a clear forecast origin. Put visual observations on a separate row so the slide does not imply fourteen days of dense imagery.
**Visual candidate:** [Forecast window](figures/forecast-window.drawio.png).
**Speaker focus:** Explain one input/output example without listing the full data schema.

### 11. Entire plants are held out from training

**On slide:** “69 training · 15 validation · 14 test plants.” “No fitting on test plants.”
**Design:** Three visibly separated plant groups; emphasize the test boundary.
**Speaker focus:** Explain the generalization claim. Distinguish cross-plant zero-shot evaluation after training elsewhere from an entirely unadapted pretrained model.

### 12. Forecast quality includes overall error and ramp error

**On slide:** “Skill: higher is better” and “Ramp NMAE: lower is better.”
**Design:** Two large metric cards with simple trajectory/transition illustrations. Add a small daylight-scoring and plant-weighting footer.
**Speaker focus:** Explain skill relative to Smart Persistence and error on the largest observed transitions as complementary measures.

### 13. MMTSFM combines temporal and visual foundation models

**On slide:** Chronos-2 and V-JEPA 2.1.
**Design:** Two input paths joining at the forecaster. Label the adapted temporal backbone and frozen visual encoder. Omit token-allocation details here.
**Speaker focus:** Explain each backbone's role and the numerical-only control. Source: [Method](chapter4.tex).

### 14. Pooled-image fusion compresses the scene into one global token

**On slide:** “One global visual token.”
**Design:** Imagery compresses into a single token before entering the forecaster.
**Visual candidate:** [Pooled-image architecture](figures/mmtsfm_s1.png).
**Speaker focus:** Explain the global visual bottleneck and the question it tests.

### 15. Spatial-token fusion preserves spatially distinct information

**On slide:** “Spatially addressable tokens at the latest anchor.”
**Design:** A visual grid becomes multiple tokens entering the numerical sequence.
**Visual candidate:** [Spatial-token architecture](figures/mmtsfm_s2a.png).
**Speaker focus:** Explain the contrast with global pooling and how spatial information becomes accessible.

### 16. Multi-anchor fusion distributes context across five daily anchors

**On slide:** “Five daily anchors.”
**Design:** Five positions along a timeline, each supplying tokens. Reuse slide 15's visual language.
**Visual candidate:** [Multi-anchor architecture](figures/mmtsfm_s2d.png).
**Speaker focus:** Explain visual-history allocation and the trade-off with latest-anchor detail. Keep full payload in backup slides.

### 17. Spatial-token fusion improves UK point-forecast skill

**On slide:** Numerical control **52.30% → 55.10%** spatial-token skill.
**Design:** One directly labeled comparison chart and a small “13 of 14 plants improve” callout. Identify the three matched seeds and show available variation.
**Speaker focus:** Explain the 2.80 percentage-point improvement and consistency. Spatial-token leads the main fusion configurations; avoid a universal best-model claim.
**Source:** [UK results](chapter5.tex).

### 18. Spatial-token fusion also lowers UK ramp error

**On slide:** Ramp NMAE **0.1506 → 0.1440**.
**Design:** One metric comparison beside a readable illustrative forecast transition.
**Visual candidate:** [Forecast comparison](figures/traces_comparison_11287_w1021.png).
**Speaker focus:** Explain the improvement over the adapted numerical control. Acknowledge that Time-VLM has lower ramp NMAE in the external benchmark.

### 19. Disabling vision tests reliance on visual inputs

**On slide:** “Same weights; visual embeddings set to zero.”
**Design:** Intact versus vision-off conditions above one change-in-error chart. Highlight spatial-token ramp NMAE degradation of **0.0063**.
**Speaker focus:** Explain why a same-weights intervention differs from comparing separately trained models. Pooled-image shows negligible mean ramp response.

### 20. Image substitutions test whether time and place matter

**On slide:** Correct imagery versus six-hour stale imagery and wrong-plant imagery.
**Design:** Three labeled conditions with one performance-degradation chart.
**Speaker focus:** The UK model depends on aligned visual content. These controls do not isolate specific visual features or establish a motion mechanism. Keep cached-latent permutation in backup slides.

### 21. Multi-anchor leads point skill under UK-to-US transfer

**On slide:** “Fixed UK-trained checkpoints → eight US plants.” Multi-anchor **10.46%** versus numerical **8.90%** skill.
**Design:** Small UK-to-US label above one point-skill comparison. Identify the satellite-source change.
**Speaker focus:** Explain the ranking change. Region and sensor change together; greater sensor robustness is a possible explanation rather than an isolated causal finding.

### 22. Spatial-token retains the lowest US ramp error among main fusion arms

**On slide:** Numerical **0.1864**, spatial-token **0.1692**, multi-anchor **0.1702** ramp NMAE.
**Design:** One comparison chart, directly labeled, with “Lower is better.” Use the same US-setting footer as slide 21.
**Speaker focus:** The preferred interface depends on the performance measure. Restrict the claim to the tested main arms.

### 23. Better point accuracy does not guarantee calibrated uncertainty

**On slide:** “Prediction intervals under-cover.”
**Design:** One large nominal-versus-observed coverage visual using the thesis results. Keep scope limitations in spoken notes rather than adding a dense table.
**Speaker focus:** Explain calibration as a separate outcome. Briefly state the remaining scope boundaries: limited regions/periods, joint region-and-sensor shift, and idealized reanalysis covariates. Prioritize calibration, broader testing, and operational-input evaluation.
**Source:** [Discussion](chapter6.tex) and [Future work](chapter7.tex).

### 24. Structured fusion improves transfer with verified visual reliance

**On slide:** One large answer: “Structured fusion improves unseen-plant forecasting.” Small qualifier: “Interface choice and calibration remain setting-dependent.”
**Design:** Central statement with generous whitespace. Keep this slide visible during questions.
**Speaker focus:** “The results show that structured fusion improves forecasting at unseen UK plants, and controlled interventions confirm reliance on recent, site-specific imagery. Cross-region transfer changes which fusion strategy performs best, while uncertainty calibration remains an open problem.”

## ASCII slide wireframes

These are layout sketches, not finished charts. Plot marks and traces are illustrative placeholders; use the actual thesis data and figures in the deck. Each sketch uses the same slide number as the outline and script. All slides share a small source footer and a page number. Whitespace is intentional.

### 01 — Research question

```text
+------------------------------------------------------------+
|                                                            |
|  MULTIMODAL FOUNDATION          +-----------------------+   |
|  MODELS FOR FORECASTING         |                       |   |
|  AT UNSEEN PLANTS               |   SATELLITE IMAGE     |   |
|                                |                       |   |
|  Can imagery improve transfer?  +-----------------------+   |
|                                                            |
|  Marco Morandin                                            |
|  Supervisor / degree / date                                |
|  [small institutional logo]                            01  |
+------------------------------------------------------------+
```

### 02 — Unseen-plant forecasting

```text
+------------------------------------------------------------+
|  A new plant must be forecast without local fitting        |
|                                                            |
|  TRAINING PLANTS              UNSEEN TARGET PLANT          |
|  +----+ +----+ +----+          +------------------------+   |
|  | A  | | B  | | C  |          | Short local history    |   |
|  +----+ +----+ +----+          |       /\__/\__         |   |
|           |                   +-----------+------------+   |
|           v                               |                |
|     [learned model] -- fixed parameters --+--> forecast    |
|                                                            |
|  Thesis: Introduction                                  02  |
+------------------------------------------------------------+
```

### 03 — Complementary imagery

```text
+------------------------------------------------------------+
|  Satellite imagery adds spatial context                    |
|                                                            |
|  NUMERICAL HISTORY              SATELLITE CONTEXT          |
|  +------------------------+     +----------------------+   |
|  | Power                  |     |                      |   |
|  |   /\     /\_           |     |   site-centered      |   |
|  | _/  \___/   \__        |     |   cloud-field image   |   |
|  |          ^ transition  |     |         [plant]      |   |
|  +------------------------+     +----------------------+   |
|  Observation at the plant       Environment around it      |
|  Thesis: Introduction                                  03  |
+------------------------------------------------------------+
```

### 04 — Contributions

```text
+------------------------------------------------------------+
|  Three contributions support the research question         |
|                                                            |
|                                                            |
|       CORPUS          BENCHMARK        FUSION ANALYSIS      |
|                                                            |
|    Aligned sources   Held-out plants   Interfaces and      |
|    with provenance   and fair scoring  visual-use controls |
|                                                            |
|                                                            |
|  Thesis: Contributions                                 04  |
+------------------------------------------------------------+
```

### 05 — Dataset requirements

```text
+------------------------------------------------------------+
|  The research question determines the corpus               |
|                                                            |
|  +----------------------------------------------------+    |
|  |                  GEOGRAPHIC MAP                    |    |
|  |                                                    |    |
|  |  UK: 100 systems             US: 10 systems         |    |
|  |  UK PV + EUMETSAT             PVDAQ + GOES          |    |
|  +----------------------------------------------------+    |
|  Measured power + numerical context + satellite imagery    |
|  Atmospheric covariates: Open-Meteo                         |
|  Thesis: Dataset construction                          05  |
+------------------------------------------------------------+
```

### 06 — Harmonization

```text
+------------------------------------------------------------+
|  Harmonization preserves measurement meaning               |
|                                                            |
|  UK energy -- interval duration --> power -- capacity --+  |
|                                                        |  |
|  US power ------------------------------- capacity ----+  |
|                                                        v  |
|                                  Capacity-normalized target|
|                                                            |
|  UK native power:  o---------o---------o   30 minutes       |
|  US native power:  o----o----o----o----o   15 minutes       |
|  Native storage; US evaluation resampled to 30 minutes      |
|  Thesis: Harmonization                                 06  |
+------------------------------------------------------------+
```

### 07 — Time/location alignment

```text
+------------------------------------------------------------+
|  Each image matches the site and precedes the origin        |
|                                                            |
|  Site ID + coordinates          Forecast origin            |
|                                        |                   |
|  Historical imagery                    |  Future image     |
|     [frame]       [frame]    [selected] |    [excluded]     |
|  ------o-------------o-----------o-----|--------x--> time   |
|                             < 15 min >|                   |
|                                        |                   |
|  Exact UTC capture times make alignment auditable          |
|  Thesis: Image alignment                               07  |
+------------------------------------------------------------+
```

### 08 — Quality and missingness

```text
+------------------------------------------------------------+
|  Observed zero is different from missing data              |
|                                                            |
|  Power                                                     |
|    |        /\                     __________               |
|    |       /  \     [gap]         | flagged stuck signal   |
|    |______/    \__          ______|                        |
|    +------------------------------------------------ time  |
|       valid zero       unavailable       faulty reading    |
|                                                            |
|  Preserve quality labels and separate validity masks       |
|  Illustrative trace / Thesis: Quality                  08  |
+------------------------------------------------------------+
```

### 09 — Sample contract

```text
+------------------------------------------------------------+
|  Every forecast sample is traceable                        |
|                                                            |
|  Power / covariates --+   +-----------------------------+   |
|                      +-->| ONE FORECAST SAMPLE         |   |
|  Image archive ------+   |                             |   |
|                          | Values: history / images /  |   |
|                          |         covariates / target |   |
|                          | Identity: plant / source /  |   |
|                          |           UTC timestamps    |   |
|                          | Validity: separate masks    |   |
|                          +-----------------------------+   |
|  Thesis: Dataset contract                              09  |
+------------------------------------------------------------+
```

### 10 — Forecast window

```text
+------------------------------------------------------------+
|  Fourteen days of context support a six-hour forecast       |
|                                                            |
|  NUMERICAL HISTORY                ORIGIN     PREDICTION     |
|  <----------- 14 days ------------>|<------ 6 hours ------> |
|    /\__/\__/\__/\__/\__/\__        |    /\___               |
|                                    |                       |
|  VISUAL OBSERVATIONS               |                       |
|       [older anchors]   [recent]   |                       |
|                                    |                       |
|  Visual allocation varies by fusion interface               |
|  Thesis: Forecast window                               10  |
+------------------------------------------------------------+
```

### 11 — Plant-disjoint evaluation

```text
+------------------------------------------------------------+
|  Entire plant identities are held out                      |
|                                                            |
|  +----------------+ +----------------+ +----------------+  |
|  | TRAINING       | | VALIDATION     | | TEST           |  |
|  |                | |                | |                |  |
|  |      69        | |      15        | |      14        |  |
|  |    plants      | |    plants      | |    plants      |  |
|  +----------------+ +----------------+ +----------------+  |
|                                                            |
|                   No fitting on test plants                |
|  Thesis: Evaluation protocol                           11  |
+------------------------------------------------------------+
```

### 12 — Metrics

```text
+------------------------------------------------------------+
|  Overall quality and ramp quality are complementary        |
|                                                            |
|  +-------------------------+ +-------------------------+   |
|  | SKILL                   | | RAMP NMAE               |   |
|  |                         | |                         |   |
|  | Higher is better        | | Lower is better         |   |
|  | Relative to             | | Error during largest    |   |
|  | Smart Persistence       | | observed transitions    |   |
|  +-------------------------+ +-------------------------+   |
|                                                            |
|  Valid daylight targets / plant-weighted aggregation        |
|  Thesis: Scoring                                       12  |
+------------------------------------------------------------+
```

### 13 — Model backbones

```text
+------------------------------------------------------------+
|  MMTSFM joins temporal and visual foundation models        |
|                                                            |
|  Numerical inputs --> [Chronos-2: adapted] -----+           |
|                                                |           |
|                                                v           |
|                                         [Fusion interface] |
|                                                |           |
|  Satellite clips --> [V-JEPA 2.1: frozen] ------+           |
|                                                v           |
|                                            Forecast        |
|  Thesis: Method                                        13  |
+------------------------------------------------------------+
```

### 14 — Pooled-image fusion

```text
+------------------------------------------------------------+
|  Pooled-image fusion supplies one global token             |
|                                                            |
|  +-------------+                                           |
|  | Visual      |   global                                  |
|  | features    | -- pooling --> [ ONE TOKEN ] --+           |
|  +-------------+                               |           |
|                                                v           |
|  Numerical context ----------------------> [Forecaster]    |
|                                                            |
|                     Compact scene summary                  |
|  Thesis: Pooled-image fusion                           14  |
+------------------------------------------------------------+
```

### 15 — Spatial-token fusion

```text
+------------------------------------------------------------+
|  Spatial tokens preserve separately addressable locations  |
|                                                            |
|  Latest visual anchor                                      |
|     +--+--+--+          Spatial tokens                     |
|     |  |  |  |       --> [v1][v2][v3] ... [v98]            |
|     +--+--+--+                       |                     |
|     |  |  |  |                       v                     |
|     +--+--+--+          [numerical + visual sequence]       |
|                                      |                     |
|                                  Forecast                  |
|  Thesis: Spatial-token fusion                          15  |
+------------------------------------------------------------+
```

### 16 — Multi-anchor fusion

```text
+------------------------------------------------------------+
|  Multi-anchor distributes visual context across days       |
|                                                            |
|     [A1]      [A2]      [A3]      [A4]      [A5]            |
|  ------o---------o---------o---------o---------o--> time    |
|        |         |         |         |         |           |
|      tokens    tokens    tokens    tokens    tokens        |
|        +---------+---------+---------+---------+           |
|                            |                               |
|                     [Fusion / forecast]                    |
|                                                            |
|  Thesis: Multi-anchor                                  16  |
+------------------------------------------------------------+
```

### 17 — UK point skill

```text
+------------------------------------------------------------+
|  Spatial-token improves held-out UK point skill            |
|                                                            |
|  Skill (%)                                      Higher ^   |
|  +----------------------------------------------------+    |
|  |  Numerical control     [bar + seed variation] 52.30 |    |
|  |  Spatial-token         [bar + seed variation] 55.10 |    |
|  +----------------------------------------------------+    |
|                                                            |
|       +2.80 percentage points                              |
|       Mean skill improves at 13 of 14 plants                |
|  UK / three matched seeds / Thesis: Results             17  |
+------------------------------------------------------------+
```

### 18 — UK ramp error

```text
+------------------------------------------------------------+
|  Spatial-token also reduces UK ramp error                  |
|                                                            |
|  RAMP NMAE                 +---------------------------+   |
|                            | Representative transition |   |
|  Numerical     0.1506       |                           |   |
|       |                    | Observed + two forecasts  |   |
|       v                    | [actual thesis trace]     |   |
|  Spatial-token 0.1440       +---------------------------+   |
|                                                            |
|  Lower is better          Trace is illustrative            |
|  UK / three matched seeds / Thesis: Results             18  |
+------------------------------------------------------------+
```

### 19 — Vision-off control

```text
+------------------------------------------------------------+
|  The trained spatial-token model relies on vision          |
|                                                            |
|  [Same weights + intact vision]   [Same weights + zeros]    |
|                 \                   /                      |
|                  \                 /                       |
|                   CHANGE IN RAMP NMAE                      |
|               +--------------------------+                 |
|               | Spatial-token: +0.0063   |                 |
|               | Pooled-image: negligible |                 |
|               +--------------------------+                 |
|  UK / same-weights controls / Thesis: Ablations         19  |
+------------------------------------------------------------+
```

### 20 — Alignment controls

```text
+------------------------------------------------------------+
|  Correct image time and place matter to the forecast       |
|                                                            |
|     CORRECT           SIX-HOUR STALE       WRONG PLANT      |
|  [local clip]         [older local clip]   [distant clip]   |
|                                                            |
|  +----------------------------------------------------+    |
|  |         ACTUAL PERFORMANCE COMPARISON              |    |
|  |  intact reference / stale / substituted site       |    |
|  +----------------------------------------------------+    |
|                                                            |
|  UK / matched substitutions / Thesis: Ablations         20  |
+------------------------------------------------------------+
```

### 21 — US point skill

```text
+------------------------------------------------------------+
|  Multi-anchor leads point skill under US transfer          |
|                                                            |
|  UK-trained checkpoints --> Eight unseen US plants         |
|  SEVIRI                 --> GOES-16                        |
|                                                            |
|  +----------------------------------------------------+    |
|  | Numerical control     [actual chart]          8.90% |    |
|  | Multi-anchor          [actual chart]         10.46% |    |
|  +----------------------------------------------------+    |
|  Fixed parameters / joint geographic and sensor shift      |
|  US / three matched seeds / Thesis: Transfer            21  |
+------------------------------------------------------------+
```

### 22 — US ramp error

```text
+------------------------------------------------------------+
|  Spatial-token leads ramp accuracy among main fusion arms  |
|                                                            |
|  RAMP NMAE                                      Lower v    |
|  +----------------------------------------------------+    |
|  | Numerical control     [actual chart]         0.1864 |    |
|  | Spatial-token         [actual chart]         0.1692 |    |
|  | Multi-anchor          [actual chart]         0.1702 |    |
|  +----------------------------------------------------+    |
|                                                            |
|  The preferred interface depends on the metric             |
|  US / three matched seeds / Thesis: Transfer            22  |
+------------------------------------------------------------+
```

### 23 — Calibration

```text
+------------------------------------------------------------+
|  Better point accuracy does not guarantee calibration      |
|                                                            |
|  Interval coverage (%)                                     |
|  +----------------------------------------------------+    |
|  | - - - - - nominal coverage reference - - - - - - - |    |
|  |                                                    |    |
|  | [UK observed coverage]   [US observed coverage]     |    |
|  | Use thesis values and matched interval definition  |    |
|  +----------------------------------------------------+    |
|  Prediction intervals under-cover                          |
|  Thesis: Calibration and limitations                    23  |
+------------------------------------------------------------+
```

### 24 — Final answer

```text
+------------------------------------------------------------+
|                                                            |
|                                                            |
|           STRUCTURED FUSION IMPROVES                       |
|             UNSEEN-PLANT FORECASTING                       |
|                                                            |
|         Auditable corpus + verified visual reliance        |
|                                                            |
|     Interface choice and calibration remain open issues    |
|                                                            |
|                 Thank you / Questions                      |
|  Thesis: Conclusions                                   24  |
+------------------------------------------------------------+
```

## Spoken script

Use the following as speaker notes, not as text to place on the slides. The script allows time for pointing to figures, pauses, and slide changes. Rehearse aloud against the timing table; speaking speed and the pronunciation of technical names and numbers affect the duration.

### Slide 1 — Research question

“Good morning. My thesis tests whether combining time-series and visual foundation models improves forecasting at unseen solar plants, and whether those improvements depend on the satellite imagery.”

### Slide 2 — Unseen-plant forecasting

“The target plant is excluded from training. We have a short local history, but we do not fit parameters there. Because that history cannot cover every operating condition, the model must transfer patterns learned elsewhere. Solar power is the testbed for this broader problem: forecasting for a new entity with limited observations.”

### Slide 3 — Complementary imagery

“Numerical history describes production and atmospheric covariates. Imagery adds a spatial observation of the surrounding environment. The trace shows what happened at the plant; the image provides complementary cloud context. The scientific question is whether the model can use that extra information to improve forecasts, particularly during abrupt transitions.”

### Slide 4 — Contributions

“There are three contributions: an aligned multimodal corpus with provenance, a plant-disjoint benchmark, and a controlled fusion analysis. I will give particular attention to dataset construction, because its design determines whether the transfer experiment is credible.”

### Slide 5 — A corpus designed for the research question

“Dataset construction is a scientific contribution of this thesis. To test multimodal transfer, I needed measured plant power, numerical context, and satellite observations connected to the same forecast examples.

Among the datasets surveyed in the thesis, none provided the combination of cross-plant evaluation, native-cadence observations, and sample-level provenance used here.

I therefore assembled two tracks: one hundred UK systems paired with EUMETSAT imagery, and ten US systems paired with GOES imagery. Atmospheric covariates come from Open-Meteo. Eight US plants are eligible for the transfer experiment. The construction choices determine whether the resulting comparisons are scientifically interpretable.”

### Slide 6 — Harmonization

“The sources do not initially measure or sample everything in the same way. UK generation is reported as energy over an interval, so I convert it to power using the interval duration. Both tracks retain measured power and a target normalized by audited installed capacity.

This makes plants of different sizes comparable while retaining differences in their observed response.

The corpus also preserves native cadence: UK power is half-hourly and US power is fifteen-minute. Windows are defined in physical time, so fourteen days means the same duration in either track. For the transfer benchmark, US power is resampled to the UK cadence. Stored observations and evaluation grids are therefore distinct.”

### Slide 7 — Time and location alignment

“A numerical record and an image are useful together only if they refer to the correct site and time. Each record preserves UTC timestamps, plant coordinates, source identity, and a reference to the image archive. Frames retain their original capture times.

For UK samples, the closest valid satellite frame preceding the forecast origin is selected within a fifteen-minute window. A frame after the origin cannot enter the historical visual input, because it would reveal future conditions.

These links make alignment auditable. They let us check what the model actually observed rather than assume that two streams were synchronized simply because they share an approximate time label.”

### Slide 8 — Quality and missingness

“A low power reading can have several meanings. It may reflect cloud cover, a valid nighttime zero, an equipment problem, or an absent measurement. Treating these cases identically would change the learning task.

The corpus therefore preserves quality labels for faulty observations and distinguishes values from their validity. Extended gaps remain missing; they are not filled with invented targets. Missing visual slots and invalid forecast targets also have separate masks.

This matters scientifically because otherwise a model could be rewarded for predicting preprocessing artifacts. The benchmark should measure forecasts against valid observations, while retaining the information needed to understand what was excluded.”

### Slide 9 — Sample contract and provenance

“The final step is a common contract for one forecasting sample. It joins numerical history, covariates, imagery, and future targets with the plant identity, timestamps, and validity masks.

Numerical records and image payloads are stored separately, but exact references connect them. This makes it possible to reconstruct the observations belonging to a forecast and verify their temporal alignment.

Preserving plant identity also enables strict separation between training and test plants. The contract is independent of the model, while the protocol specifies which inputs each method may use.

The result is an auditable basis for comparing forecasting systems. The dataset design is part of the evidence supporting the generalization claims.”

### Slide 10 — Forecast window

“Here is one sample from that contract. The model receives fourteen days of numerical history and predicts the next six hours. Imagery appears on a separate timeline because the fusion strategies allocate it differently. Fourteen days of numerical context does not imply dense video over that period. The local history conditions the forecast while model parameters remain fixed.”

### Slide 11 — Plant-disjoint evaluation

“The UK benchmark separates entire plant identities: sixty-nine plants for training, fifteen for validation, and fourteen for testing.

This means that the test plants are excluded from model fitting. We are testing whether learning on other plants helps forecast a new one.

I use zero-shot cross-plant transfer in this specific sense: no parameter fitting at the evaluation plant. This differs from evaluating a completely unadapted foundation model. The multimodal models have been trained on UK training plants, but their parameters stay fixed when applied to the held-out plants.”

### Slide 12 — Performance measures

“I report two complementary measures. Skill expresses the percentage reduction in normalized root mean squared error relative to Smart Persistence. Higher skill is better; zero means matching that reference.

Ramp NMAE measures normalized absolute error on the largest observed power transitions. Lower values are better.

This distinction matters because a model can predict the overall trajectory reasonably well while missing abrupt changes. Evaluation uses valid daylight targets and plant-weighted aggregation, so the headline result represents performance across installations.”

### Slide 13 — Foundation-model backbones

“The proposed system, MMTSFM, combines two pretrained backbones. Chronos-2 processes the numerical context, and V-JEPA 2.1 encodes the satellite clips.

The temporal model is adapted using the training plants. The visual encoder remains frozen, and its representations are used by the fusion interface. Caching those representations also avoids repeatedly encoding the same clips during training.

A numerical-only model provides the central control. This lets us ask whether adding the visual pathway improves on an already adapted temporal forecaster. The next three slides explain how the visual information enters the model.”

### Slide 14 — Pooled-image fusion

“Pooled-image fusion compresses the scene into one global visual token. This is a compact summary without separately addressable spatial locations. It tests whether that summary is sufficient, or whether preserving more structure helps.”

### Slide 15 — Spatial-token fusion

“Spatial-token fusion retains multiple spatially distinct visual tokens from the latest anchor. In the main configuration, it supplies ninety-eight tokens rather than one global summary.

The important difference is that the forecasting model can access different parts of the visual representation separately. It does not have to recover all the relevant spatial information from a single compressed token.

This interface concentrates its visual context at the latest observation. The experiment tests whether preserving this structure produces better forecasts and stronger dependence on the imagery.”

### Slide 16 — Multi-anchor fusion

“The multi-anchor configuration changes how visual context is distributed over time. It supplies tokens from five daily observation windows.

This creates a trade-off: allocating context across several anchors changes the detail available at the most recent observation. The model receives a broader visual history rather than concentrating the main visual budget at one time.

The comparison therefore asks how visual context should be allocated. The answer may depend on the evaluation setting and on whether we prioritize overall error or abrupt transitions.”

### Slide 17 — UK point skill

“On the held-out UK plants, spatial-token fusion improves mean skill from fifty-two point three percent for the numerical control to fifty-five point one percent.

That is an increase of two point eight percentage points. The point-forecast improvement appears in every paired training seed, and mean skill improves at thirteen of the fourteen test plants.

The result therefore extends across most of the tested fleet, although it does not guarantee improvement at every new plant.

Spatial-token fusion leads the main fusion configurations in this setting. This is a scoped comparison, rather than a claim that the model is best on every metric against every external baseline.”

### Slide 18 — UK ramp error

“Ramp NMAE falls from zero point one five zero six to zero point one four four zero. The trace illustrates a transition; the aggregate metric provides the fleet-wide evidence. This is an improvement over the adapted numerical control. Time-VLM has lower ramp error in the external benchmark, so the result does not establish superiority over every baseline.”

### Slide 19 — Vision-off control

“A better score alone does not establish that a model uses its images. To test reliance, I evaluate the same trained weights with the visual embeddings set to zero.

For spatial-token fusion, ramp NMAE increases by zero point zero zero six three. The model performs worse when its visual input is disabled.

Pooled-image fusion, by contrast, has negligible mean ramp degradation in this control.

This separates two questions: how well the complete model forecasts, and how much that fitted model depends on its visual pathway. The same-weights intervention provides direct evidence for the second question.”

### Slide 20 — Image-alignment controls

“The next controls keep a visual input but replace its content. I use imagery from six hours earlier, or imagery from another plant at the corresponding time.

Both substitutions degrade spatial-token forecasts. The model is sensitive to whether the visual observation matches the time and place of the prediction.

These results support reliance on recent, site-specific content in the UK experiment. They do not identify exactly which visual features produce the gain.

They also do not establish an explicit motion mechanism. In particular, the separate cached-latent permutation test leaves open whether temporal information has already been mixed inside the visual encoder.”

### Slide 21 — US point skill

“I then apply the fixed UK-trained checkpoints to eight US plants, without additional fitting. This changes both the geographic region and the satellite source, from SEVIRI to GOES-16.

The ordering of the fusion interfaces changes. Multi-anchor achieves the highest point skill among the main arms, at ten point four six percent, compared with eight point nine percent for the numerical control.

This supports an advantage for distributing visual context across multiple anchors in the tested transfer setting.

However, the experiment does not isolate the reason. Greater robustness to sensor differences is a possible explanation, but region and sensor change together. Additional controlled evaluations are needed to separate those effects.”

### Slide 22 — US ramp error

“For US ramps, spatial-token has the lowest error among the main fusion arms: approximately zero point one six nine, with multi-anchor close behind. Both improve on the numerical control. The important finding is the different ordering: multi-anchor leads overall point skill, while spatial-token leads ramp accuracy.”

### Slide 23 — Calibration and limitations

“Better point predictions do not guarantee reliable uncertainty estimates. The models also produce probabilistic forecasts, but their prediction intervals cover fewer observations than their nominal coverage would suggest. This limitation becomes more pronounced under US transfer.

Calibration therefore needs to be evaluated separately from point accuracy. A forecast can improve its central prediction while still giving unreliable intervals.

There are also boundaries to the transfer evidence. The evaluation covers a restricted set of regions and observation periods. In the US experiment, geography and satellite source change together. Horizon atmospheric covariates come from archived reanalysis, so the benchmark does not fully reproduce the noise of operational weather forecasts.

The next steps are to improve calibration, expand geographic and multi-year testing, and evaluate operational weather inputs. Those extensions would establish how broadly the measured gains carry over beyond the present benchmark.”

### Slide 24 — Final answer

“The thesis contributes an auditable multimodal corpus and evidence that structured fusion improves unseen-plant forecasting. Matched controls confirm reliance on recent, site-specific imagery. Regional transfer changes the preferred interface, and calibration remains unresolved. The broader lesson is that dataset validity, fusion design, and controlled evaluation must be considered together. Thank you. I am happy to take your questions.”

## Backup slides

Keep one concept per backup slide as well:

- Related-work positioning and complete baseline leaderboard.
- Exact skill and ramp-target definitions on separate slides.
- Dataset filtering, eligibility, and provenance.
- Per-plant and per-seed results.
- Cached-latent permutation and the unresolved motion mechanism.
- Token selection, pooling, placement, and encoder-freezing controls on separate slides.
- Full-payload feasibility and its single-seed limitation.
- Trainable parameters, caching, and compute.
- Detailed calibration and future-work plans.

## Rehearsal

Finish the expanded dataset section at 5:40, reach the architecture at 7:20, finish UK results at 10:45, and begin the conclusion at 14:25. Practice transitions so the higher slide count forms one continuous argument. Protect the dataset explanation when shortening the talk: reduce introductory and architectural detail first. Verify every displayed number against its source and check readability at projection size.
