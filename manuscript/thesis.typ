// =============================================================================
//  Interleaved Vision Fusion for Time-Series Foundation Models
//  Measuring What Vision Contributes to Photovoltaic Ramp Forecasting
//
//  Master's thesis — University of Trento, DISI
//  Marco Morandin — Academic year 2025/2026
//
//  Build:  typst compile thesis.typ
//  Watch:  typst watch thesis.typ
//
//  Ported from the DISI LaTeX template (update 2020-08-30). The page geometry,
//  type sizes and heading formats below reproduce that template; see the notes
//  inline where a LaTeX construct has a deliberate Typst equivalent.
// =============================================================================

#let title = "Interleaved Vision Fusion for Time-Series Foundation Models"
#let subtitle = "Measuring What Vision Contributes to Photovoltaic Ramp Forecasting"
#let author = "Marco Morandin"
#let supervisor = "Prof. Elisa Ricci"
#let cosupervisor = "Francesco Gentile"
#let department = "Department of Information Engineering and Computer Science"
#let programme = "Artificial Intelligence Systems"
#let academic-year = "Academic year 2025/2026"

// --- LaTeX size commands at an 11pt base, so the port can quote them ---------
#let sz-large = 14.4pt // \Large
#let sz-larger = 17.28pt // \LARGE
#let sz-huge = 24.88pt // \Huge

// --- Page ---------------------------------------------------------------------
// \usepackage[paperheight=29.7cm,paperwidth=21cm,
//             outer=1.5cm,inner=2.5cm,top=2cm,bottom=2cm]{geometry}
#set page(
  paper: "a4",
  margin: (inside: 2.5cm, outside: 1.5cm, top: 2cm, bottom: 2cm),
  binding: left,
  numbering: none,
)

// book.cls at 11pt sets Computer Modern on a 13.6pt baseline. "New Computer
// Modern" is the same typeface; `leading` is the gap between lines rather than
// the baseline distance, so it is tuned rather than copied.
// \parindent is 17pt and \parskip is 0pt, so paragraphs are marked by an indent
// and not by a blank line; `spacing` therefore matches `leading`, and `all:
// false` leaves the first paragraph after a heading flush, as LaTeX does.
#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set par(
  justify: true,
  leading: 0.72em,
  spacing: 0.72em,
  first-line-indent: (amount: 17pt, all: false),
)

// --- Headings -----------------------------------------------------------------
// \titleformat{\chapter}{\normalfont\Huge\bfseries}{\thechapter}{1em}{}
//   renders "3<1em>Title" — the word "Chapter" is deliberately absent.
// \titlespacing gives 0.59in above a chapter, 0.20in above a section,
//   0.10in above a subsection, 0.02in below each. Those LaTeX skips are *added
//   to* the normal baseline advance, whereas a Typst block's `above`/`below` is
//   the whole gap, so one line's worth is added back here; the two constants are
//   calibrated against the LaTeX build rather than derived.
#let lead-in = 11pt // makes up the advance LaTeX keeps before a title
#let lead-out = 8.5pt // and after it
#set heading(numbering: "1.1.1")

#show heading.where(level: 1): it => {
  // book.cls restarts figure and table numbering at every chapter.
  counter(figure.where(kind: image)).update(0)
  counter(figure.where(kind: table)).update(0)
  block(above: 0.59in + lead-in, below: 0.02in + lead-out, breakable: false)[
    #set text(size: sz-huge, weight: "bold")
    #set par(justify: false)
    #if it.numbering != none [
      #counter(heading).display(it.numbering)#h(1em)
    ]
    #it.body
  ]
}

#show heading.where(level: 2): it => block(
  above: 0.20in + lead-in,
  below: 0.02in + lead-out,
)[
  #set text(size: sz-large, weight: "bold")
  #set par(justify: false)
  #if it.numbering != none [
    #counter(heading).display(it.numbering)#h(0.6em)
  ]
  #it.body
]

#show heading.where(level: 3): it => block(
  above: 0.10in + lead-in,
  below: 0.02in + lead-out,
)[
  #set text(size: 12pt, weight: "bold")
  #set par(justify: false)
  #if it.numbering != none [
    #counter(heading).display(it.numbering)#h(0.6em)
  ]
  #it.body
]

// \paragraph{...} was a run-in bold lead-in, unnumbered and out of the contents;
// the port folds it into the paragraph it introduces, so no rule is needed here.

// --- Cross-references ---------------------------------------------------------
// The LaTeX source wrote "Chapter~\ref{...}" by hand; the port drops the literal
// word and lets Typst supply it, so a level-1 heading reference reads "Chapter 3",
// an appendix reference "Appendix B", and a deeper one "Section 3.2".
// Appendix headings are told apart by their numbering pattern, set in the
// appendix preamble further down.
#set ref(supplement: it => {
  if it.func() == heading {
    if it.level > 1 { "Section" } else if it.numbering == "A.1.1" {
      "Appendix"
    } else { "Chapter" }
  } else {
    it.supplement
  }
})

// --- Figures and tables -------------------------------------------------------
// LaTeX's book class numbers both per chapter (Figure 3.1, Table 7.2).
#set figure(numbering: n => {
  let ch = counter(heading).get().first()
  numbering("1.1", ch, n)
})
#show figure.caption: set text(size: 10pt) // \small
#show figure: set block(above: 1.2em, below: 1.2em)

// The LaTeX tables are \hline-ruled top, under the header and bottom; the Typst
// equivalent is a horizontal-only stroke driven from the table itself.
#set table(
  stroke: (x, y) => (
    top: if y <= 1 { 0.5pt } else { 0pt },
    bottom: 0.5pt,
  ),
  inset: (x: 6pt, y: 4pt),
)
#show table: set text(size: 10pt) // \small

#set list(indent: 1em)
#set enum(indent: 1em)
#show raw: set text(size: 0.92em)

// =============================================================================
// Title page  (was front.tex)
// =============================================================================
#[
  #set align(center)
  #set par(justify: false)

  #image("figures/unitrento-logo.svg", width: 60%)

  #v(2cm)
  #text(size: sz-larger)[#department]

  #v(1cm)
  #text(size: sz-large)[
    Master's Degree in \
    #programme
  ]

  #v(2cm)
  #text(size: sz-large)[#smallcaps[Final Dissertation]]

  #v(1cm)
  #text(size: sz-huge)[#smallcaps[#title]]

  #v(0.5cm)
  #text(size: sz-large, style: "italic")[#subtitle]

  #v(2cm)
  #text(size: sz-large)[
    #grid(
      columns: (1fr, 1fr),
      row-gutter: 0.6em,
      [Supervisor], [Student],
      [#supervisor], [#author],
      [Co-Supervisor], [],
      [#cosupervisor], [],
    )
  ]

  #v(2cm)
  #text(size: sz-large)[#academic-year]
]

#pagebreak()

// =============================================================================
// Acknowledgements  (was acknowledgements.tex)
// =============================================================================
#[
  #set align(center)
  #text(size: sz-huge, weight: "bold")[Acknowledgements]
]

#v(4cm)

#emph[...thanks to...]

#pagebreak()

// =============================================================================
// Contents
// =============================================================================
#set page(numbering: "1")
#counter(page).update(1)

#outline(title: "Contents", indent: auto, depth: 3)

#pagebreak()


// ==========================================================================
// Abstract
// ==========================================================================


#heading(level: 1, numbering: none)[Abstract]
Photovoltaic power forecasting is usually posed as a per-plant supervised problem: one
model is fitted to the history of one installation. That formulation does not survive
contact with deployment, where a newly connected plant arrives with a couple of weeks of
measurements and no training set of its own. This thesis studies the alternative
formulation --- #emph[cross-plant zero-shot forecasting];, in which a model trained on one
fleet must forecast plants it has never observed --- and asks whether frozen foundation
models, fused across modalities, are the right instrument for it.

The work makes three contributions. First, it constructs a dataset of record for the
problem: 1.34 million rows over 110 photovoltaic sites in two regions, pairing measured
power and fourteen weather and solar-geometry covariates with co-registered satellite
imagery on a single timestamp grid, together with a plant-disjoint split and a tensor
contract that every model in the study consumes unchanged. Second, it defines a fairness
protocol --- identical physical-time windows, identical splits, identical
capacity-normalised metrics, no domain physics inside models --- and runs #emph[28]
forecasters under it, spanning statistical references, classical machine learning,
supervised deep networks, four families of time-series foundation model, retrieval and
adapter-based adaptation of frozen backbones, and both endogenous and exogenous multimodal
systems. Third, it proposes MMTSFM, an architecture that fuses a mostly-fixed Chronos-2
time-series backbone with a frozen V-JEPA~2.1 video encoder through two mechanisms:
#emph[selective temporal interleaving];, which weaves visual tokens into the time-series
sequence only inside a short refinement window at roughly two percent token overhead, and
#emph[causal Grassmann mixing];, an $O (L)$ temporal operator that encodes the transition
between consecutive hidden states as a subspace on a Grassmann manifold rather than as a
dot-product similarity.

The benchmark yields four findings that reframe the problem. A well-tuned supervised
transformer (iTransformer, skill score $0.552$) remains the model to beat, ahead of every
foundation model evaluated. Zero-shot time-series foundation models underdeliver on
photovoltaic dynamics, one of them falling below the naive reference. Retrieval over a
frozen backbone reaches skill $0.478$ against $0.331$ for fine-tuning the same backbone,
making it the most effective and cheapest adaptation tested. And the best multimodal model
in the suite synthesises its images #emph[from the series itself];, while every system
consuming genuine satellite frames sits mid-pack --- evidence that visual inductive bias,
not visual information, is what currently converts into forecast skill. MMTSFM is presented
in full, with its training curriculum, its controls and its evaluation plan; the
experimental campaign for the model was still running at the time of writing, and the
chapter that reports it states the measurement protocol and the placeholders rather than
provisional numbers.


// ==========================================================================
// Chapters
// ==========================================================================


= Introduction
<cha:intro>
== Forecasting a plant you have never seen
<sec:motivation>
A photovoltaic installation converts a stochastic, weather-driven input into an
electrical output that a grid operator has to schedule around. The forecasting problem
that follows is a short-horizon regression: given the recent power history of a plant and
whatever exogenous information is available, predict the next few hours of generation.
Framed that way it looks like a solved family of problems, and for a #emph[single,
instrumented, long-lived] plant it very nearly is --- a gradient-boosted model over lagged
power and weather covariates is a strong forecaster, and a supervised transformer trained
on years of that plant's own history is stronger still.

The formulation breaks down at the point where it matters operationally. Photovoltaic
capacity is added continuously and in small increments: rooftop systems, community
installations, new inverters on an existing site. A plant that was energised last month has
no multi-year history to fit a model to. If every installation requires its own training
run, the forecasting stack scales linearly in the number of assets and degrades exactly
where the fleet is growing fastest. The deployment-realistic question is therefore not
"how well can we forecast a plant we have studied" but

#quote(block: true)[
#emph[how well can we forecast a plant we have never seen, given two weeks of its history
and nothing else?]
]

This is a generalisation question, not a photovoltaic-engineering question, and it is the
question this thesis is built around. The distinction matters for how the work is
evaluated. Under the usual protocol a model is trained and tested on the same plants,
separated only in time, and reported accuracy conflates two very different abilities:
learning the idiosyncrasies of a specific installation, and learning something transferable
about how irradiance, cloud cover and time of day map onto power. Splitting by #emph[plant
identity] rather than by time isolates the second ability, and it is the only split under
which the deployment question can be answered honestly. Throughout this thesis, train,
validation and test sets are disjoint sets of #emph[plants];; no test plant contributes a
single row, a single frame or a single normalisation statistic to anything that is fitted.

=== What makes the signal hard
It is worth being precise about the difficulty, because "predict solar power" sounds easier
than it is. The signal decomposes into two parts with completely different character.

The first is deterministic: the position of the sun, and therefore the clear-sky irradiance
reaching a fixed panel, is known analytically for all future time. Any model with a calendar
and a latitude reproduces it. It also accounts for the majority of the raw variance, which
means that a forecast evaluated against a reference that does not know it will look excellent
while having learned nothing.

The second is the modulation of that envelope by the atmosphere --- overwhelmingly by cloud.
It is intermittent, spatially structured, advects at wind speed, and changes the output of an
array by most of its range within minutes. It is not well captured by hourly reanalysis
weather, which is what numerical covariates provide: an hourly areal cloud fraction says
little about whether the sun is occluded at one rooftop at 14:20. This is the part a
forecaster is actually paid to predict, and it is the part for which the numerical track is
an impoverished description.

That gap is the opening for a visual modality. Cloud fields are directly observable in
satellite imagery, at a spatial and temporal resolution that reanalysis does not approach,
and they advect predictably enough over a few hours that what is visible upwind now is
overhead later. @cha:dataset quantifies the residual information available:
conditioning on cloud state changes median output by roughly 0.09 in normalised power
#emph[after] conditioning on irradiance. Whether any current architecture converts that
availability into forecast skill is, in the end, the question this thesis is about.

== Why foundation models, and why multimodal
<sec:why-fm>
Two developments make the cross-plant question tractable in a way it was not a few years
ago.

The first is the arrival of #emph[time-series foundation models];: transformers pretrained
on large, heterogeneous corpora of time series that forecast unseen series zero-shot, with
no per-series fitting~@chronos2@timesfm@tirex. Their promise is precisely the
promise this problem needs --- transfer to a series that was not in the training set. The
second is the maturation of #emph[video foundation models] trained by self-supervised
prediction rather than reconstruction~@vjepa2. This matters because the dominant
source of short-horizon photovoltaic uncertainty is cloud advection, a spatiotemporal
process that is visible in satellite imagery hours before it registers in the power signal.
A model that reads only the power history is, in the ramp regime, structurally blind: the
information that would let it anticipate a cloud edge exists, but not in the modality it
consumes.

Combining the two is not novel in itself; a growing literature attaches visual encoders to
time-series forecasters~@timevlm@unicast@solarvlm@crossvivit. What is contested is
#emph[how deeply] the two modalities should be coupled. The prevailing designs are shallow:
a visual encoder produces a summary vector which is concatenated to the series
representation (late fusion), or injected as a soft prompt~@unicast, or used to
condition an adapter on an otherwise frozen backbone~@cora. In each case the two
modalities meet once, after each has been processed independently. The alternative
investigated here is to let them meet #emph[inside] the temporal operator, as neighbouring
tokens in one sequence, so that the model computes cross-modal interactions at every layer
rather than a single cross-modal readout at the end.

The claim under test is therefore:

#quote(block: true)[
A frozen multimodal foundation model stack (Chronos-2 and V-JEPA~2.1) with deep
token-level fusion achieves cross-plant photovoltaic power forecasting on disjoint test
plants, beating late fusion, unimodal foundation models, and domain-specific
architectures.
]

Every baseline in @cha:baselines exists to falsify one part of that sentence.
The design principle behind the benchmark is that a reader must not be able to say "the
gain could come from $X$, and $X$ was never tested".

== Research question and hypotheses
<sec:rq>
The research question, stated in the terms the thesis is evaluated in:

#quote(block: true)[
Can a frozen multimodal foundation model stack (a time-series foundation model and a
vision foundation model) achieve cross-plant photovoltaic power forecasting on disjoint
test plants by deep token-level fusion, rather than by late fusion or by domain-specific
architectures?
]

It decomposes into a ladder of hypotheses, each of which is answered by a specific
experiment, and each of which is falsifiable independently of the ones above it.

- #strong[H0 --- Foundation-model anchor.] A time-series foundation model applied
  zero-shot is a meaningful reference point for this task. Tested in
  @cha:baselines against classical and supervised alternatives.

- #strong[H1 --- Vision helps.] Adding a visual stream through a late-fusion adapter
  improves over an otherwise identical time-series-only model. Tested by the modality
  ablation of @cha:expdesign.

- #strong[H2 --- Fusion depth matters.] Selective temporal interleaving of visual
  tokens improves over late fusion, with the same backbones and the same data. This is
  the architectural contribution of @cha:method.

- #strong[H3 --- Cross-plant transfer is achievable.] A model can forecast completely
  disjoint plants from a short history, relying on structure learned from other plants.
  This is the evaluation protocol of @cha:protocol, not an ablation.

- #strong[H4 --- Retrieval is a competitive alternative.] Retrieval augmentation on a
  frozen backbone closes the gap to full fine-tuning. Tested in
  @cha:baselines.

== Contributions
<sec:contributions>
#[
#set par(first-line-indent: 0pt)
*A dataset of record for multimodal cross-plant photovoltaic forecasting.* Four public sources --- UK domestic generation~@ukpv, EUMETSAT SEVIRI high-resolution
visible imagery~@eumetsat, NREL PVDAQ systems with GOES-16 crops~@pvdaq, and
Open-Meteo reanalysis covariates~@openmeteo --- are fused into one flat table of
$1 \, 337 \, 654$ rows over 110 sites, plus a packed image archive addressed by a single
timestamp-exact pointer. The construction, the quality flags and the exact plant partition
are given in @cha:dataset.
]

#[
#set par(first-line-indent: 0pt)
*A 28-model benchmark under one fairness contract.* All models consume the same tensor contract, the same 14-day history and 6-hour horizon
defined in #emph[physical] time, the same plant-disjoint split, and the same
capacity-normalised metrics. The suite spans seven tiers, from naive persistence to
photovoltaic-specialised multimodal networks. To the best of this work's knowledge it is
the largest single-protocol comparison of time-series foundation models,
frozen-backbone adaptation methods and multimodal forecasters on photovoltaic data.
]

#[
#set par(first-line-indent: 0pt)
*MMTSFM, an architecture for deep multimodal fusion on frozen backbones.* Two mechanisms are contributed. #emph[Selective temporal interleaving] inserts visual
summary tokens into the time-series sequence only inside a short refinement window, leaving
the macro-context purely numerical; the token overhead is proportional to the number of
visual tokens rather than to context length, on the order of two percent. #emph[Causal
Grassmann mixing] replaces $O (L^2)$ temporal attention with an $O (L)$ operator that encodes
the transition between hidden states as a two-dimensional subspace via a Plücker embedding,
aggregated over multiple temporal offsets. A matched diagnostic twin using ordinary causal
self-attention is retained so that any measured difference is attributable to the inductive
bias and not to the training schedule.
]

#[
#set par(first-line-indent: 0pt)
*A diagnosis of where the headroom actually is.* Two measurements from the benchmark reorient the problem. The gap between a
privileged-information oracle variant of Chronos-2 and its honest fine-tuned counterpart is
$0.173$ skill --- larger than any architectural difference observed in the entire suite,
and evidence that #emph[conditioning];, not architecture, is the binding constraint.
Separately, the strongest multimodal model in the suite uses images synthesised from the
series itself, while every model consuming real satellite frames ranks mid-pack; the
distinction between visual #emph[inductive bias] and visual #emph[information] is developed
in @cha:baselines and~@cha:conclusion.
]

== Scope and non-goals
<sec:scope>
The framing is deliberately that of a machine-learning contribution with photovoltaics as
the testbed, not a photovoltaic-engineering contribution. Several things are consequently
out of scope, and are listed here so the reader does not go looking for them.

- #strong[No domain physics inside models.] Clear-sky-index conversions and
  irradiance-to-power physical models are excluded from every learned model. The single
  exception is smart persistence, which #emph[is] the physics reference and serves as the
  denominator of the skill score.

- #strong[No few-shot in-context adaptation curves.] Accuracy as a function of $K$
  support days per test plant is a different protocol --- context matching rather than
  zero-shot transfer --- and was excluded by an explicit design decision. The evaluation
  harness supports it as an appendix experiment should a reader require it.

- #strong[No grid-operations or electricity-market analysis.] The output of this work
  is a forecast and its error, not a dispatch decision or an economic valuation.

- #strong[No pre-2025 methods as primary comparators.] Older architectures appear
  only where they are still the community reference (persistence, gradient boosting,
  DLinear as the simplicity check).

- #strong[No dataset construction beyond what is required.] The consolidated dataset
  is treated as read-only once built; no result in this thesis depends on re-processing
  it.

== A note on what this thesis reports
<sec:reporting-stance>
Two editorial decisions shape how the results are presented, and both are worth declaring at
the outset because they are unusual enough to be mistaken for omissions.

#emph[Negative results are reported as results.] Several of the most informative measurements
in this work are things that did not happen: foundation models did not transfer zero-shot,
real satellite imagery did not outperform synthesised imagery, and pretraining did not beat a
well-matched supervised architecture. Each is stated in the chapter where it was measured,
with the control that establishes it, and none is relegated to a limitations paragraph.

#emph[Provisional numbers are not reported at all.] The experimental campaign for the proposed
architecture was still running at the time of writing. @cha:expdesign therefore
states the arms, the controls, the interpretation rules and the reporting template in full,
and leaves the result cells empty. A number that later moves is worse than no number, because
it propagates silently into every comparison that cites it --- and because a design whose
interpretation is fixed only after the result is known is not a test of anything.

The second decision has a consequence for how the thesis should be read: its empirical weight
sits in @cha:dataset through~@cha:benchmark, which are complete, rather
than in @cha:expdesign, which is a specification.

== Structure of the thesis
<sec:structure>
@cha:background positions the work against the 2025--2026 literature on
time-series foundation models, frozen-backbone adaptation, video foundation models and
multimodal forecasting, and states the gap that motivates the benchmark.
@cha:dataset describes the construction of the dataset of record, its schema,
its quality flags and its plant-disjoint splits. @cha:protocol defines the
evaluation protocol: the fairness rules, the physical-time windows, the metrics and the
scenario battery. @cha:baselines presents the baseline suite, its
implementation, the full leaderboard and the four findings that follow from it.
@cha:method presents MMTSFM in full: both branches, the fusion mechanisms, the
Grassmann operator, the output head and the four-stage training curriculum.
@cha:expdesign gives the experimental design for the model, its ablation and
control battery, and the reporting protocol under which its numbers will be recorded; the
campaign was still running at the time of writing and no provisional figures are stated.
@cha:conclusion answers the research question as far as the current evidence
permits, states the limitations, and lays out the next experiments in priority order.


= Background and Related Work
<cha:background>
This chapter fixes the notation used throughout the thesis, then surveys the four
literatures the work sits between: time-series foundation models, methods for adapting
frozen backbones, video foundation models, and multimodal time-series forecasting. It
closes with the photovoltaic-specific literature, a comparison against the two other public
multimodal photovoltaic corpora, and the gap statement that motivates
@cha:dataset#[]--@cha:baselines.

== Problem formalisation and notation
<sec:formalisation>
Let a #emph[plant] (equivalently, an #emph[entity];) be indexed by $i$, and let time be
discretised on the plant's native cadence. At forecast origin $t_0$ a model observes a
history window of length $T$ and must produce a horizon of length $H$.

- $upright(bold(Y)) in bb(R)^(N times T times 1)$ --- normalised historical power
  for $N$ entities. The target is #emph[capacity-normalised] power, $y = P \/ P_(upright("installed")) in [0 \, 1]$, so that plants of different nameplate size are
  directly comparable.

- $upright(bold(X))_(upright("cov")) in bb(R)^(N times (T + H) times C)$ --- covariates
  over history #emph[and] horizon. The horizon portion contains only quantities that are
  genuinely known ahead of time (solar geometry, calendar encodings) unless a variant is
  explicitly labelled as using privileged information.

- $upright(bold(V)) in bb(R)^(N times T_v times C_(upright("img")) times H_(upright("img")) times W_(upright("img")))$ --- $T_v$ satellite frames over a short window ending at $t_0$.

- $hat(upright(bold(Y)))_(upright("fut")) in bb(R)^(N times H times Q)$ --- the
  prediction, as $Q$ quantiles per horizon step rather than a point estimate.

The #emph[cross-plant] constraint is a property of the index set, not of the tensors: if
$cal(P)_(upright("train"))$, $cal(P)_(upright("val"))$ and $cal(P)_(upright("test"))$ are
the plant sets, then $cal(P)_(upright("train")) inter cal(P)_(upright("test")) = nothing$, and no statistic computed on $cal(P)_(upright("test"))$ --- including
normalisation constants --- may enter any fitted component. This rules out the standard
practice of per-series standardisation using test-series statistics, which is why the
target is normalised by nameplate capacity, a quantity known at installation time and
independent of the measured signal.

=== A taxonomy of the forecasting task
<sec:taxonomy>
Solar forecasting is not one problem but a family indexed by lead time, and the dominant
information source changes with it. @tab:leadtime states the regimes and where
this thesis sits.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Regime];], [#strong[Lead time];], [#strong[Dominant information source];],),
    table.hline(),
    [Nowcasting], [0--30 min], [Sky-camera imagery; cloud advection within the local field of view],
    [Intra-hour], [30 min -- 2 h], [Satellite imagery; advection at mesoscale],
    [#strong[Intra-day];], [#strong[2--6 h];], [#strong[Satellite imagery plus numerical weather; this
    thesis];],
    [Day-ahead], [6--48 h], [Numerical weather prediction; imagery is stale],
    [Seasonal], [weeks--months], [Climatology; individual weather is irrelevant],
  )]
  , caption: [Forecasting regimes by lead time. The six-hour horizon studied here spans the
  boundary at which satellite imagery ceases to dominate and numerical weather takes over,
  which is why the visual window is short and the numerical context long.]
  , kind: table
  )
<tab:leadtime>
The six-hour horizon is a deliberate choice of the hardest interesting regime. Below two
hours the problem is largely advection and a good image model wins; beyond twelve hours the
observed cloud field has left the domain and only numerical weather matters. Between them
both sources are partially informative and neither is sufficient, which is exactly the
setting in which a fusion architecture should earn its complexity. It is also the regime that
matters operationally for intra-day market participation and reserve scheduling.

=== Why cross-plant differs from cross-time
<sec:cross-plant-theory>
The distinction between splitting by time and splitting by entity is worth stating precisely,
because it determines what a reported number means.

Under a #emph[temporal] split, a model is fitted on ${ (x_t \, y_t) : t < t_0 }$ for plant $i$
and evaluated on later observations of the same plant. The model may --- and in practice
does --- absorb plant-specific constants into its parameters: panel orientation, local
shading geometry, inverter clipping thresholds, the systematic bias of the nearest weather
grid cell. These are genuine, stable, exploitable regularities, and learning them is not
cheating. But they are not #emph[transferable];, and a benchmark that rewards them measures
memorisation of a plant alongside understanding of the physics.

Under a #emph[cross-plant] split, plants in the test set contribute nothing to fitting. The
model receives, at inference, a short history from which everything plant-specific must be
inferred on the fly. Formally the model must learn a mapping conditioned on context rather
than a per-entity parameter vector: $hat(y)_(i \, t + h) = f (upright(bold(Y))_(i \, < t) \, upright(bold(X))_i \, upright(bold(V))_(i \, < t))$ with $f$ shared across all $i$ and no $i$-indexed
parameters anywhere. This is the same requirement that in-context learning imposes on
language models, and it is why foundation models are a natural hypothesis for the problem.

@fig:siteboxplot quantifies what has to be inferred: after capacity
normalisation, median daytime output still ranges from below 0.10 to above 0.55 across
plants. That residual spread is exactly the per-plant information a temporal split would hand
the model for free.

== Time-series foundation models
<sec:tsfm>
A time-series foundation model is a sequence model pretrained on a large corpus of
heterogeneous series and applied to unseen series without per-series fitting. The design
space has converged on a few recurring choices: patch-based tokenisation, in which
contiguous blocks of timesteps become tokens; a normalisation scheme that makes series of
arbitrary scale comparable; and a probabilistic output head, either quantile-based or
sampling-based.

#strong[Chronos-2];~@chronos2 is the backbone used in this thesis. It is an
encoder-only transformer operating on patches of size 16, with an arcsinh-based
normalisation, a nine-quantile output head, and --- the property that matters most here ---
a #emph[group attention] mechanism that attends across the batch axis over rows sharing a
group identifier. This is what allows covariates, and in principle other entities, to be
supplied as additional token rows rather than as extra channels, and it is the hook that
the multimodal design of @cha:method exploits.

#strong[TimesFM~2.5];~@timesfm is a decoder-only model from an independent lineage.
#strong[TiRex];~@tirex is built on the xLSTM family rather than on attention and is
among the strongest zero-shot performers on general-purpose leaderboards. #strong[Tiny Time
Mixers];~@ttm occupy the opposite end of the scale axis: a few million parameters,
mixer-based rather than attention-based, intended to test whether size is what matters.

=== Shared design elements
Four design choices recur across the families and are worth stating once, because they
determine what these models can and cannot do on photovoltaic data.

#[
#set par(first-line-indent: 0pt)
*Patch tokenisation.* Rather than one token per timestep, contiguous blocks of $p$
steps become one token. This buys a factor of $p$ in sequence length --- decisive when the
context is 672 steps --- and imposes a mild smoothing prior, since sub-patch structure must
survive a linear projection to be represented at all. For photovoltaic data with $p = 16$ at a
30-minute cadence, one token spans eight hours: an entire daylight period is roughly one and
a half tokens. Fine-grained ramp structure is therefore compressed aggressively, which is one
plausible mechanism behind the zero-shot results of @cha:benchmark.
]

#[
#set par(first-line-indent: 0pt)
*Scale-free normalisation.* A model pretrained across corpora cannot assume a
scale. Solutions range from instance normalisation with stored statistics to
inverse-hyperbolic-sine transforms that compress large magnitudes while remaining
approximately linear near zero. The choice interacts badly with a target that is exactly zero
for half of every day: a bounded, heavily zero-inflated signal is not what these transforms
were designed for.
]

#[
#set par(first-line-indent: 0pt)
*Probabilistic heads.* Output is a predictive distribution, most commonly a fixed
quantile grid trained with pinball loss, occasionally a sampling-based head. Quantile heads
are cheap and directly scoreable but cannot represent multi-modality --- and a forecast under
broken cloud is genuinely bimodal (the sun is either occluded at the array or it is not).
]

#[
#set par(first-line-indent: 0pt)
*Covariate handling.* This is where the families differ most, and it matters for
this task more than any other design axis. Chronos-2 admits covariates as additional token
rows fused by attention across the batch axis; TimesFM and TiRex in the configurations
evaluated here are target-only; TTM accepts exogenous channels. Given that the covariate
track carries most of the exploitable signal (@fig:correlations), a model that
cannot ingest it is competing with one hand tied --- a caveat that must be attached to the
zero-shot comparison.
]

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,left,left,center,left,),
    table.header([#strong[Model];], [#strong[Family];], [#strong[Backbone];], [#strong[Covariates];], [#strong[Output];],),
    table.hline(),
    [Chronos-2], [encoder transformer], [attention + group attention], [yes], [9 quantiles],
    [TimesFM 2.5], [decoder transformer], [causal attention], [no], [point / quantile],
    [TiRex], [xLSTM], [recurrent state], [no], [quantile],
    [TTM-R2], [mixer], [MLP-mixer blocks], [yes], [point],
  )]
  , caption: [The four time-series foundation model families evaluated, in the configurations used
  here. Covering three distinct architectural paradigms is what makes a negative zero-shot
  result a statement about the task rather than about a model.]
  , kind: table
  )
<tab:tsfm-families>
Including at least three distinct families is a methodological requirement rather than a
completeness gesture. A result obtained with one backbone can always be attributed to that
backbone's pretraining corpus; a result that replicates across an encoder transformer, a
decoder transformer and an xLSTM is a property of the #emph[task];. @cha:baselines
reports all four families under one protocol, and the conclusion --- that zero-shot
time-series foundation models underdeliver on photovoltaic dynamics --- is consistent
across them.

== Adapting frozen backbones
<sec:adaptation>
If a pretrained backbone is not good enough zero-shot, three strategies are available that
do not require training a new model from scratch.

#[
#set par(first-line-indent: 0pt)
*Fine-tuning.* Update some or all backbone weights on in-domain data. Effective,
but it produces a task-specific copy of a large model and forfeits the generality that
motivated pretraining.
]

#[
#set par(first-line-indent: 0pt)
*Retrieval augmentation.* Keep the backbone frozen and condition it on similar
windows retrieved from a datastore. #strong[TS-RAG];~@tsrag retrieves analogous
historical windows and fuses them into the forecast; #strong[Cross-RAG];~@crossrag
extends this across series with clear-sky-aware keys and per-step mixing weights. The
appeal is that adaptation cost is a nearest-neighbour lookup rather than a gradient step.
The fairness question it raises is central to this thesis and is answered in
@sec:parity: the datastore may contain training plants only, since populating
it with test-plant history is transduction, not zero-shot forecasting.
]

#[
#set par(first-line-indent: 0pt)
*Adapters and covariate injection.* #strong[CoRA];~@cora injects exogenous
covariates --- of any modality, in principle including image features --- into a frozen
time-series backbone through a zero-initialised residual adapter. CoRA is the closest
published competitor to the architecture proposed here: it occupies the same design space
(frozen backbone, external information injected through a learned interface) but couples
the modalities once, at a single insertion point, rather than throughout the temporal
operator. If deep token-level fusion cannot beat CoRA supplied with image features, the
fusion-depth claim of this thesis does not survive. Parametric-memory
alternatives~@memts occupy the same niche and are positioned but not run.
]

== Video foundation models
<sec:vision-fm>
The visual branch of the proposed model uses #strong[V-JEPA~2.1];~@vjepa2, a
ViT-L/16 video encoder trained by joint-embedding predictive self-supervision: it learns to
predict the #emph[representation] of masked spatiotemporal regions rather than their pixels.
Three properties motivate the choice over reconstruction-based alternatives such as
VQ-VAE video tokenisers.

First, the objective is predictive rather than reconstructive. A reconstruction loss spends
capacity on pixel-level detail that is irrelevant --- and, for satellite imagery, largely
sensor noise. A predictive objective in representation space yields spatially structured,
semantically coherent features, which is what cloud-structure detection requires. Second,
temporal modelling is native: V-JEPA consumes an entire clip and encodes motion internally
through tubelet embedding with temporal stride, so a downstream summariser only needs to
perform #emph[spatial] compression. A per-frame image encoder would force the temporal
modelling back onto the fusion layer. Third, the model is built for frozen use, remaining
strong under linear probing and adapter tuning --- the operating regime of the staged
curriculum in @sec:curriculum.

=== The joint-embedding predictive objective
The distinction between predictive and reconstructive self-supervision is worth stating
concretely, because it is the reason this branch of the architecture uses a video model
rather than an image tokeniser.

A reconstructive objective --- a masked autoencoder, or a vector-quantised autoencoder ---
asks the network to output the missing #emph[pixels];. The loss is therefore computed in pixel
space, and every source of pixel variance contributes to it, including sensor noise, sub-pixel
registration error and radiometric calibration drift. For satellite crops these are
substantial and carry no meteorological content whatsoever.

A joint-embedding predictive objective asks the network to output the missing
#emph[representation];, as computed by a slowly updated copy of itself. The loss lives in
representation space, so any variation the encoder chooses not to represent costs nothing.
The network is free to discard exactly the nuisance variance a reconstruction loss forces it
to model. What survives is structure that is predictable across space and time --- which, in a
sequence of satellite crops, is cloud morphology and its motion.

Two properties follow that matter for this application. The representation is spatially
structured rather than globally pooled, so downstream modules can attend to regions rather
than to a single vector. And the model is robust under frozen use: because the objective never
required the encoder to serve a decoder, its features are not entangled with a reconstruction
head, and linear probes and light adapters recover most of their utility. That is precisely
the operating regime of the curriculum in @sec:curriculum.

=== What retrieval augmentation actually does
Since retrieval turns out to be the strongest frozen-backbone strategy in
@cha:benchmark, its mechanism deserves more than a citation.

Given a query context $upright(bold(Y))_(i \, < t)$, a retrieval-augmented forecaster embeds it, finds
the $k$ nearest neighbours in a datastore of historical windows under some key function, and
combines the neighbours' #emph[continuations] with the backbone's own forecast --- typically
as a convex combination whose weight $alpha$ is tuned on held-out data, sometimes per horizon
step. The backbone is untouched.

Two design choices distinguish the variants. The #emph[key] determines what counts as
similar: raw values, a learned embedding, or --- in the clear-sky-aware variant --- a
representation that has already divided out the deterministic diurnal component, so that
neighbours match on cloud regime rather than on time of day. The #emph[scope] determines
where neighbours may come from: within the same series, or across series. Cross-series
retrieval is the more powerful option and is exactly what a cross-plant protocol makes
legitimate --- a test plant with two weeks of history has no useful within-series precedent
for an unusual regime, but the training fleet does.

This also makes the fairness rule of @sec:parity non-negotiable. If the datastore
contained test-plant history, retrieval would be doing in-context memorisation of the target
plant, and the resulting number would not be a zero-shot result at all.

== Multimodal time-series forecasting
<sec:multimodal>
The multimodal forecasting literature divides along an axis that turns out to be
empirically decisive, and this thesis adopts the distinction as a reporting category.

#[
#set par(first-line-indent: 0pt)
*Endogenous multimodal.* The second modality is #emph[synthesised from the
numerical series itself];. #strong[Time-VLM];~@timevlm renders the series as an image
and feeds it, with a text description, to pretrained vision-language backbones.
#strong[VisionTS++];~@visiontspp maps series to images to exploit a frozen masked
autoencoder. #strong[Aurora];~@aurora templates weather text from covariates. No
external sensor is involved; what these methods exploit is the #emph[inductive bias] of a
visual backbone --- its priors over locality, periodicity and texture --- applied to a
re-rendering of data the model already had.
]

#[
#set par(first-line-indent: 0pt)
*Exogenous multimodal.* The second modality is a genuine external observation.
#strong[UniCast];~@unicast prompts a frozen time-series foundation model with vision
and text embeddings. #strong[CrossViVit];~@crossvivit is the reference deep
satellite-plus-series model, cross-attending video tokens into a temporal transformer, and
remains the strongest non-foundation-model multimodal competitor.
#strong[Solar-VLM];~@solarvlm and #strong[SUNSET];~@sunset are the
photovoltaic-domain instances.
]

The proposed architecture is exogenous by construction, and the benchmark of
@cha:baselines evaluates both classes side by side. The result --- that the
best endogenous model outperforms every exogenous one --- is the sharpest empirical finding
in this thesis and is what @cha:conclusion argues should reorient the design.

== Photovoltaic forecasting and its conventions
<sec:pv-lit>
The solar forecasting community has its own reference model and its own headline metric,
and both are adopted here to keep the results legible to that community.

#emph[Smart persistence] carries the last observed #emph[clearness index] --- the ratio of
measured power to a clear-sky model~@haurwitz --- forward across the horizon, rather
than carrying power itself forward. It therefore respects the deterministic diurnal shape
and only assumes that cloud conditions persist. It is a much harder reference than naive
persistence, and it is the denominator of the #emph[skill score]
$upright(S S) = 1 - upright(N R M S E)_(upright("model")) \/ upright(N R M S E)_(upright("smart persistence"))$,
the number by which the field reports progress.

Domain-specific architectures follow a common pattern: a convolutional or transformer
encoder over sky or satellite imagery, fused with a recurrent or attention-based encoder
over the power series. SUNSET~@sunset is the canonical convolutional instance;
CrossViVit~@crossvivit the deep attention-based one; Solar-VLM~@solarvlm and
PV-VLM~@pvvlm add a language backbone; FusionSF~@fusionsf and
M3S-Net~@m3snet pursue vector-quantised and multi-scale fusion respectively.

=== Four generations of method
The domain literature has moved through four broad phases, and the assumptions of each are
still visible in the baselines that survive from it.

#emph[Physical and statistical models] came first: clear-sky models with empirical cloud
attenuation, persistence of the clearness index, and autoregressive corrections. They are
interpretable, require no training data, and remain the reference against which everything
else is measured --- smart persistence in this benchmark is a direct descendant.

#emph[Feature-engineered machine learning] followed: gradient boosting and support-vector
regression over lagged power and numerical weather. This generation established that
exogenous weather is worth more than a longer power history, a finding this benchmark
reproduces in the gap between DLinear (target only) and every covariate-consuming model.

#emph[Deep sequence and image models] came next: convolutional encoders over sky or satellite
imagery combined with recurrent power encoders, of which SUNSET is the canonical example, and
later attention-based spatiotemporal architectures such as CrossViVit. This is the generation
that established the multimodal premise --- that imagery should help --- largely on
nowcasting horizons where it demonstrably does.

#emph[Foundation-model adaptation] is the current phase: frozen pretrained backbones adapted
by fine-tuning, retrieval, adapters or prompting, with vision-language models entering the
domain through Solar-VLM and PV-VLM. The premise inherited from the previous generation ---
that real imagery is the payload --- has not been re-tested under the new one, which is part
of what this thesis does.

What these works do not share is a protocol. Horizons range from minutes to a day ahead,
history windows from one hour to a year, splits are usually temporal rather than
plant-disjoint, and normalisation conventions differ enough that reported errors are not
comparable across papers. This is the practical reason a new benchmark was necessary:
without one, the question of whether foundation-model fusion beats domain-specific fusion
cannot be answered from the published numbers.

== Public multimodal photovoltaic datasets
<sec:other-datasets>
Two public corpora pair photovoltaic generation with imagery, and both were considered
before the dataset of @cha:dataset was built.

#strong[ClimateHack.AI 2023];~@climatehackai covers Great Britain at 5-minute
resolution with generation as a fraction of installed capacity, two EUMETSAT satellite
streams (high-resolution visible and an 11-channel non-HRV product) as
$[12 \, 128 \, 128]$ crops, DWD ICON-EU numerical weather forecasts over 38 variables, and ECMWF
CAMS air-quality forecasts. It is organised as a #emph[nowcasting] task: one hour of
history in, four hours out at 5-minute granularity. The layout is per-window tensors rather
than a tall table, which targets minutes-to-hours nowcasting and would require complete
re-windowing to serve a 14-day-history cross-plant protocol. It is also a competition
release with no accompanying peer-reviewed publication, so there is no citable account of
its construction.

#strong[MMSP];, released with FusionSF~@fusionsf, covers 88 plants across a Chinese
province at hourly resolution from January 2021 to June 2022, fusing power, Himawari-8/9
crops at $64 times 64$ pixels (one channel in the 10-plant subset, four in the full set)
and 17 ECMWF weather variables. The task is day-ahead: 24 hours in, 24 hours out. Again the
structure --- hourly cadence, fixed day-ahead windows, per-window forecast tensors --- is a
poor fit for a longer-history intra-day cross-plant protocol.

Neither corpus is deficient; they are built for different questions. But neither supports
the combination this thesis requires: a long numerical history, a short high-cadence visual
window, native per-plant cadences, plant-disjoint splits across two regions, and real
satellite frames rather than nowcast tensors.

== Fusion depth as a design axis
<sec:fusion-depth>
The multimodal methods above differ along an axis that the literature rarely names
explicitly: #emph[where];, in the computation, the two modalities first interact.
@tab:fusion-depth orders the design space by that criterion.

#figure(
  align(center)[#table(
    columns: 4,
    align: (left,left,left,left,),
    table.header([#strong[Depth];], [#strong[Mechanism];], [#strong[Interactions];], [#strong[Examples];],),
    table.hline(),
    [Post-hoc], [Average or stack two independent forecasts], [0], [Ensembles],
    [Late / readout], [Concatenate encoder outputs before the head], [1, at the end], [SUNSET, adapter-style fusion],
    [Prompt], [Project visual features into the input embedding space], [1, at the input], [UniCast],
    [Adapter], [Residual injection at one or more insertion points], [few], [CoRA],
    [Cross-attention], [Visual tokens attended from the temporal stream at every layer], [$L$], [CrossViVit, Solar-VLM],
    [#strong[Token interleaving];], [Visual tokens placed #emph[inside] the temporal sequence], [$L$, and inside the temporal operator itself], [#strong[this work];],
  )]
  , caption: [Fusion depth, ordered by where the modalities first interact and how many times.
  The proposed mechanism differs from cross-attention in that visual tokens are not a separate
  stream being attended #emph[to];; they occupy positions in the sequence the temporal operator
  runs over, so cross-modal pairs are formed by the same mechanism that forms temporal pairs.]
  , kind: table
  )
<tab:fusion-depth>
The implicit assumption in most of this literature is that deeper is better --- more
interaction points give the model more opportunity to condition one modality on the other.
That assumption is rarely tested, because comparisons across the rows of
@tab:fusion-depth are almost always confounded: different papers, different
backbones, different data, different horizons. Isolating the axis requires holding everything
else fixed, which is what the late-fusion and interleaved arms of
@cha:expdesign are constructed to do.

There is also a cost argument that cuts the other way. Cross-attention between a temporal
sequence of length $L_t$ and a visual sequence of length $L_v$ costs $O (L_t L_v)$ per layer.
For a long numerical context this becomes the dominant term, which is why cross-attention
multimodal forecasters typically use short numerical contexts --- and short contexts are
precisely what harmed the foundation models in preliminary experiments. Interleaving, by
contrast, costs $O (L_t + L_v)$ under a linear temporal operator, which is what makes a
fourteen-day context and deep fusion simultaneously affordable.

== Evaluation methodology
<sec:eval-lit>
A forecasting result is a claim about a metric, and the metrics used here carry assumptions
worth making explicit.

#[
#set par(first-line-indent: 0pt)
*The skill score and its reference.* Skill scores of the form
$1 - cal(L)_(upright("model")) \/ cal(L)_(upright("ref"))$ are standard in the atmospheric
sciences, and their interpretation depends entirely on the reference. A skill score against
climatology answers "does the model know anything about today"; against persistence, "does
it know anything beyond the present"; against smart persistence, "does it know anything
beyond the present that is not implied by solar geometry". Only the third is a statement
about forecasting skill in the sense this thesis cares about, and it is a substantially
harder bar --- the difference is visible in @cha:benchmark, where several models
that comfortably beat naive persistence sit near zero against the smart variant.
]

#[
#set par(first-line-indent: 0pt)
*Proper scoring for distributions.* The continuous ranked probability score is a
strictly proper scoring rule~@gneiting: its expectation is uniquely minimised by the
true predictive distribution, so a model cannot improve it by misrepresenting its own
uncertainty. Approximating it by pinball loss over a finite quantile grid preserves
propriety up to discretisation. Reporting it alongside point metrics is what distinguishes a
forecast that is useful for decision-making from one that is merely accurate on average.
]

#[
#set par(first-line-indent: 0pt)
*Significance.* Comparing forecast accuracy on overlapping windows violates the
independence assumptions of naive tests, since consecutive forecast origins share most of
their context. The standard remedy is the Diebold--Mariano test~@diebold on the loss
differential series with an autocorrelation-consistent variance estimator. This thesis does
not report such tests --- a limitation stated plainly in @sec:limitations --- and
consequently treats small differences between models as ties rather than as orderings.
]

#[
#set par(first-line-indent: 0pt)
*Macro-averaging.* Metrics are computed per plant and then averaged, rather than
pooled over all observations. Pooling would weight plants by their number of valid steps and
by their capacity factor, letting a few data-rich, high-output sites dominate. Since the
research question is about generalisation #emph[across] plants, each plant is one unit of
evidence.
]

== Positioning
<sec:positioning>
@tab:positioning places this work against its closest neighbours on the four axes
that matter: whether the backbone is frozen, whether real external imagery is consumed, how
deeply the modalities are fused, and whether evaluation is cross-plant.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,center,center,left,center,),
    table.header([#strong[Work];], [#strong[Frozen];], [#strong[Real imagery];], [#strong[Fusion depth];], [#strong[Cross-plant];],),
    table.hline(),
    [Chronos-2~@chronos2], [---], [no], [--- (unimodal)], [n/a],
    [TS-RAG~@tsrag], [yes], [no], [--- (retrieval)], [varies],
    [CoRA~@cora], [yes], [optional], [adapter], [no],
    [UniCast~@unicast], [yes], [yes], [prompt], [no],
    [Time-VLM~@timevlm], [yes], [no (rendered)], [readout], [no],
    [CrossViVit~@crossvivit], [no], [yes], [cross-attention], [partial],
    [Solar-VLM~@solarvlm], [partly], [yes], [cross-attention], [no],
    [#strong[This work];], [#strong[yes];], [#strong[yes];], [#strong[token interleaving];], [#strong[yes];],
  )]
  , caption: [Positioning against the closest published methods. No prior work occupies all four
  columns simultaneously, and --- more importantly --- no prior work evaluates these
  alternatives against each other under one protocol.]
  , kind: table
  )
<tab:positioning>
The last column is the one most often left empty. Cross-plant evaluation is expensive: it
requires a multi-plant corpus, disjoint splits and a normalisation scheme that does not leak,
and it produces worse numbers than a temporal split on the same data. That combination is a
strong incentive to report the easier protocol, and it is why the benchmark of
@cha:benchmark is a contribution independent of the architecture it was built to
evaluate.

== Gap statement
<sec:gap>
Three gaps follow from the survey above.

+ #strong[No single-protocol comparison exists] across time-series foundation models,
  frozen-backbone adaptation methods and multimodal forecasters on photovoltaic data.
  Published numbers are not comparable, so the relative merit of the three strategies is
  unknown.

+ #strong[Fusion depth has not been isolated.] Late fusion, prompting, adapters and
  deep cross-attention have each been proposed, but rarely against each other with the
  same backbones, the same data and the same schedule --- so "deep fusion is better" is
  an assumption rather than a measurement.

+ #strong[The endogenous/exogenous distinction is not controlled for.] Methods that
  render the series as an image and methods that consume real imagery are reported in the
  same tables, although they exploit entirely different resources.

@cha:dataset and~@cha:protocol build the instrument that closes the
first gap, @cha:baselines runs it, and @cha:method
and~@cha:expdesign address the second and third.

=== Why these gaps have persisted
It is worth asking why, given how active this literature is, the gaps above are still open.
Three structural reasons, none of them a criticism of any individual paper.

#emph[Protocols are expensive and unrewarded.] Building a cross-plant benchmark requires a
multi-plant corpus, a leak-free normalisation scheme, disjoint splits and a fairness contract
that every model must be ported to. The work is substantial, it produces worse headline
numbers than an easier protocol on the same data, and it is not what a methods paper is
reviewed on. The incentives favour reporting a new architecture under a convenient protocol.

#emph[Comparisons cross paper boundaries.] Fusion depth, adaptation strategy and modality
type are each explored within papers, but the comparison that matters --- late against deep,
retrieval against fine-tuning, endogenous against exogenous --- falls between them. Nobody
owns the cross-cutting question, and citation-based comparison cannot answer it because the
protocols differ.

#emph[Negative controls are not publication-shaped.] A modality-off arm run under an
identical recipe is the single most informative experiment a multimodal paper can report, and
it is the one most likely to weaken the paper's headline. That asymmetry is enough to explain
its rarity, and it is the reason this thesis treats that arm as the primary result rather than
as an appendix.


= Dataset Construction
<cha:dataset>
The benchmark of this thesis rests on a dataset that did not previously exist in the form
required: a single table pairing measured photovoltaic power, weather and solar-geometry
covariates, and co-registered satellite imagery, across two regions, at native cadence,
with plant identity as the split key. This chapter states the requirements that shaped it,
the sources fused, the harmonisation performed, the resulting schema, the quality flags,
and the plant-disjoint partition used everywhere in @cha:baselines
and~@cha:expdesign.

== Design requirements
<sec:data-requirements>
Five requirements were fixed before construction, each of which excludes a common shortcut.

+ #strong[One row per (plant, timestamp).] Power, covariates and a frame pointer live
  on the same row. This makes windowing a slicing operation and removes any possibility
  of misalignment between modalities.

+ #strong[Native cadence, no resampling.] The two source datasets sample at 30 and 15
  minutes. Resampling to a common grid would either discard information or fabricate it;
  instead, windows are defined in #emph[physical time] and resolved to a different number
  of steps per dataset (@sec:windows).

+ #strong[No gap interpolation beyond a strict threshold.] Long gaps are dropped
  rather than filled, so that a model is never scored against invented targets.

+ #strong[Plant identity as the split key.] Every row carries `site_id`, and
  splits partition plants, never timestamps.

+ #strong[Frames addressable in constant time.] Millions of individual image files
  are impractical on a shared cluster filesystem; frames are packed into one HDF5 archive
  with a direct integer pointer from each table row.

== Sources
<sec:sources>
Four public sources are fused. @tab:sources summarises what each contributes.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Track];], [#strong[Source];], [#strong[Contribution];],),
    table.hline(),
    [PV power, UK], [`openclimatefix/uk_pv`~@ukpv], [30-minutely domestic
    generation, 2019--2020, with per-site metadata: rounded latitude/longitude and nameplate
    capacity in kWp.],
    [Imagery, UK], [EUMETSAT SEVIRI RSS, high-resolution visible channel~@eumetsat], [Approximately 1 km per pixel, reprojected to each site and cropped to
    $128 times 128$ pixels ($approx$128 km), sized for cloud-advection context.],
    [PV power and imagery, US], [NREL PVDAQ via the OEDI data lake~@pvdaq, paired with
    GOES-16 crops], [10 sites with site metadata (coordinates, nameplate power) and
    $256 times 256$ RGB satellite crops.],
    [Weather], [Open-Meteo Archive API~@openmeteo], [Eight hourly variables per site:
    2 m temperature, shortwave / direct / diffuse radiation, direct normal irradiance, cloud
    cover, 10 m wind speed, precipitation.],
  )]
  , caption: [The four public sources fused into the dataset of record.]
  , kind: table
  )
<tab:sources>
== Harmonisation
<sec:harmonisation>
#[
#set par(first-line-indent: 0pt)
*Power.* UK generation is published as energy over a 30-minute interval
(`generation_Wh`) and is converted to instantaneous power in watts by a factor of
two. Gaps of at most three steps are linearly interpolated; longer gaps are dropped rather
than filled. Nameplate capacity is audited per site and stored as
`installed_power_w`; the modelling target is
$ mono("norm_power") = mono("power_w") / mono("installed_power_w") in [0 \, 1] \, $
set to `NaN` on rows flagged as outage or stuck. Capacity normalisation is what makes
a 1.5 kW rooftop and a 408 kW utility array commensurable, and --- critically for the
cross-plant protocol --- it uses a quantity known at installation rather than a statistic
estimated from the measured signal.
]

#[
#set par(first-line-indent: 0pt)
*Weather.* The eight Open-Meteo variables are hourly, while the power tracks are
sub-hourly; they are joined to each row by nearest timestamp using an as-of merge, so a
30-minute row inherits the enclosing hour's reanalysis values.
]

#[
#set par(first-line-indent: 0pt)
*Solar geometry and clear-sky quantities.* Deterministic quantities are derived
per row from coordinates and timestamp: solar zenith and azimuth, a Haurwitz clear-sky
global horizontal irradiance model~@haurwitz, the clearness index $k_t$, the clear-sky
index (undefined and left as `NaN` below 50 W/m$""^2$ of clear-sky irradiance),
sine and cosine day-of-year encodings, and local apparent solar time. These are the only
covariates that are legitimately #emph[known in advance] over the forecast horizon; the
distinction is what separates the honest and oracle variants of the foundation-model
baselines in @sec:parity.
]

#[
#set par(first-line-indent: 0pt)
*Frames.* Per frame, missing values are set to zero, values are clipped to
$[0 \, 1]$, and empty crops are discarded. Frames are written as a single `uint8` HDF5
archive rather than as individual files.
]

== Storage layout and the frame pointer
<sec:layout>
The result is two artefacts:

- `dataset_all.parquet` --- $1 \, 337 \, 654$ rows $times$ 35 columns, covering
  both datasets;

- `images_all.h5` --- 27 GB, 110 per-site HDF5 groups named
  `<dataset>_<site>`, each holding an `images` array and a
  `timestamps` array of ISO-8601 byte strings.

Alignment between the two is carried by a single canonical column,
`image_h5_index`. It is a #emph[group-local] index into
`images_all.h5[<dataset>_<site>]["images"]` --- not a global row number --- and it
is timestamp-exact, verified row by row for both datasets. This deserves emphasis because
the table retains two legacy pointer columns: `image_index`, which approximates the
canonical pointer, and `image_uk128_index`, which is dead and points into a file
that no longer exists. Any re-use of this dataset should read frames exclusively through
`image_h5_index`.

A visual step is considered valid --- the mask $mono("mask_visual") = 1$ --- if and only if
the row has a matching pointer and timestamp. Night and outage steps have no frame, which
means the visual mask is structurally correlated with the diurnal cycle; models must handle
this rather than assume a dense visual stream.

== Schema
<sec:schema>
@tab:schema groups the 35 columns by role.

#figure(
  align(center)[#table(
    columns: 2,
    align: (left,left,),
    table.header([#strong[Group];], [#strong[Columns];],),
    table.hline(),
    [Identity], [`dataset`, `site_id`, `station_id`,
    `camera_id`, `timestamp_utc`, `latitude`, `longitude`],
    [Target], [`power_w`, `installed_power_w`, `norm_power`],
    [Weather covariates], [`temperature_2m`, `shortwave_radiation`,
    `direct_radiation`, `diffuse_radiation`,
    `direct_normal_irradiance`, `cloudcover`, `windspeed_10m`,
    `precipitation`],
    [Solar geometry and clear-sky], [`solar_zenith`, `solar_azimuth`,
    `clearsky_ghi`, `kt`, `csi`, `doy_sin`, `doy_cos`,
    `solar_time`],
    [Quality flags], [`capacity_fixed`, `outage_flag`, `stuck_flag`,
    `night_clamped`, `bad_site_flag`],
    [Frame pointers], [`image_h5_index` (canonical), `image_index`,
    `image_path`],
  )]
  , caption: [Column groups of `dataset_all.parquet`.]
  , kind: table
  )
<tab:schema>
== Quality flags and curation
<sec:quality>
Curation is recorded as flags rather than applied destructively, so that any downstream
decision to include or exclude a row is visible and reversible.

- `bad_site_flag` --- whole sites whose series is unusable: `uk_pv`
  sites 7239 and 8587, `goes_pvdaq` sites 1283 and 51.

- `outage_flag` --- 15,486 rows with zero or anomalous output during
  daylight.

- `stuck_flag` --- 1,318 rows where the sensor reports a constant value
  across a period during which it should have varied.

- `night_clamped` --- 1,535 rows where night-time output was clamped to
  zero.

One inconsistency is recorded here rather than concealed: the committed
`goes_pvdaq` split predates the introduction of `bad_site_flag` and still
assigns sites 1283 and 51 to train and validation respectively. That split must be
regenerated, excluding the two flagged sites and leaving eight usable plants, before the
`goes_pvdaq` track is run. No result reported in this thesis is affected, because
every reported run is restricted to `uk_pv`; but the limitation is real and is
restated in @sec:limitations.

== Statistics
<sec:data-stats>
#figure(
  align(center)[#table(
    columns: 6,
    align: (left,center,center,center,left,left,),
    table.header([#strong[Dataset];], [#strong[Sites];], [#strong[Rows];], [#strong[Cadence];], [#strong[Span (UTC)];], [#strong[Frames];],),
    table.hline(),
    [`uk_pv`], [100], [1,232,862], [30 min], [2019-01-01 -- 2020-12-31], [$(N \, 128 \, 128)$ uint8 grey],
    [`goes_pvdaq`], [10], [104,792], [15 min], [2019-01-01 -- 2019-09-30], [$(N \, 256 \, 256 \, 3)$ uint8 RGB],
  )]
  , caption: [Per-dataset specification. Valid power steps: $1 \, 217 \, 399$ and $103 \, 451$
  respectively.]
  , kind: table
  )
<tab:dataset-specs>
#figure(image("figures/diurnal_profile.png"),
  caption: [
    Mean capacity-normalised power against solar hour, with the 10th--90th percentile
    band, for both tracks. The mean is a smooth deterministic arc --- which a clear-sky model
    captures for free --- while the band is enormous: at solar noon the tenth percentile of
    `uk_pv` output is near zero and the ninetieth is above 0.65. Essentially all of the
    forecasting difficulty lives in that band, not in the mean, which is precisely why smart
    persistence rather than climatology is the reference the skill score is defined against.
  ]
)
<fig:diurnal>

#figure(image("figures/ramp_rates.png"),
  caption: [
    Distribution of step-to-step changes in normalised power (logarithmic counts). The
    distribution is sharply peaked at zero with heavy, near-exponential tails --- the signature
    of a signal that is usually smooth and occasionally discontinuous. The ramp subset of
    @sec:metrics isolates the tails, which is where a visual modality is expected
    to contribute and where persistence-like models fail.
  ]
)
<fig:ramps>

#figure(image("figures/capacity_distribution.png"),
  caption: [
    Installed capacity per site. `uk_pv` is a narrow residential band clustered
    at round values between 1.5 and 4.0 kW; `goes_pvdaq` spans more than two orders of
    magnitude, from small residential systems to a 408 kW array. Capacity normalisation is what
    makes the two commensurable.
  ]
)
<fig:capacity>

#figure(image("figures/site_map.png"),
  caption: [
    Site locations, sized by capacity and coloured by mean capacity factor. The
    `uk_pv` fleet is spatially dense: many sites are within tens of kilometres of each
    other and therefore share a cloud field. This is the geometry behind the spatial-leakage
    caveat of @sec:splits --- a random plant split does not guarantee spatial
    separation. Note also the systematic north--south gradient in capacity factor, a
    generalisation axis a cross-plant model must absorb.
  ]
)
<fig:sitemap>

The two tracks differ in almost every respect that matters for generalisation, which is
what makes their combination valuable. `uk_pv` is a dense fleet of 100 residential
rooftop systems between 1.5 and 4.0 kW, confined to a small geographic envelope (latitude
50.7--57.8, longitude $- 5.6$--$0.5$) and observed through a single-channel high-resolution
visible product. `goes_pvdaq` spans capacities from 1.8 to 408 kW --- residential
through utility scale --- across a continental spread of the United States (latitude
36.0--39.9, longitude $- 115.2$--$- 75.0$), observed in three-channel RGB at twice the
temporal cadence but over nine months rather than two years. A model that transfers between
them is transferring across sensor, scale, climate and cadence simultaneously.

== Exploratory analysis
<sec:eda>
This section characterises the signal a forecaster has to model. It is not a formality: three
of the design decisions defended elsewhere in this thesis --- the choice of smart persistence
as the reference, the decision to keep a short visual window, and the expectation that vision
should matter in the ramp regime --- rest on properties visible here and nowhere else.

=== The signal is a deterministic envelope plus a stochastic modulation
@fig:weektrace makes the structure of the problem concrete over a single week.
Shortwave radiation (blue) traces smooth, nearly symmetric daily arcs whose amplitude varies
with season and synoptic conditions. Measured power (red) follows the same envelope but is
persistently rougher: on 05 and 07 January the power curve collapses to a fraction of what
the radiation curve implies; on 10 and 19 January the two agree closely; and on several days
the power series oscillates at high frequency within a single afternoon while the radiation
series --- which is hourly reanalysis interpolated to the power grid --- stays smooth.

That mismatch is the forecasting problem in one picture. The deterministic part of the signal
is nearly free: solar geometry gives it analytically. The residual is what the model must
predict, it lives at a temporal scale finer than the reanalysis resolves, and it is driven by
cloud structure that is not represented in any numerical channel at that resolution.

#figure(image("figures/week_uk_pv.png"),
  caption: [
    One week at a single `uk_pv` site: normalised power (red, left axis) against
    shortwave radiation (blue, right axis). The radiation channel is hourly reanalysis and is
    consequently smooth; power varies at the native 30-minute cadence and frequently departs from
    the envelope the radiation implies. The gap between the two curves is what a forecaster must
    model, and it is finer-grained than the covariate track can express.
  ]
)
<fig:weektrace>

=== The mean is trivial; the spread is the problem
@fig:diurnal already showed that the conditional mean is a smooth arc with an
enormous percentile band around it. @fig:monthhour extends this to the seasonal
axis. For `uk_pv` the mean normalised power is an almost perfectly separable function
of month and solar hour: a bright core around April to July at solar noon, decaying
symmetrically in both directions, with the winter months (November to January) contributing
almost nothing at any hour. The pattern is so regular that a lookup table indexed by month
and hour --- the hourly climatology baseline --- reaches a skill score of 0.234 without
observing the current day at all.

The consequence for evaluation is direct. A metric computed against a reference that does
#emph[not] know the diurnal and seasonal structure rewards a model mostly for rediscovering
astronomy. This is precisely why the skill score in @sec:metrics is defined
against smart persistence, which is given the clear-sky curve, rather than against naive
persistence or against zero.

#figure(image("figures/month_hour_heatmap.png"),
  caption: [
    Mean normalised power by month and solar hour. The `uk_pv` panel (right) is
    close to separable in its two arguments --- a strong, entirely deterministic seasonal and
    diurnal structure that any competent model captures and that therefore carries no
    discriminative information between models. The `goes_pvdaq` panel (left) covers a
    single month and shows the diurnal axis only.
  ]
)
<fig:monthhour>

=== Predictability decays within the forecast horizon
@fig:acf plots the autocorrelation of normalised power against lag in hours. Two
readings matter. First, correlation falls below 0.5 within roughly three hours on
`uk_pv` and within two on `goes_pvdaq` --- comfortably inside the six-hour
horizon this thesis forecasts. The task is genuinely hard at the horizon chosen: half the
linear predictability is gone before the horizon is half over. Second, the two datasets
behave very differently at longer lags. `uk_pv` autocorrelation flattens near 0.35
and then #emph[rises] again towards 0.45 around eight hours, the signature of the diurnal
cycle folding back; `goes_pvdaq` plunges to $- 0.75$ at six hours, which is what a
single-peaked daily arc produces when the lag reaches the half-period. The difference is a
cadence and latitude effect, and it is a concrete illustration of why metrics are never
pooled across the two tracks (@sec:windows).

#figure(image("figures/power_acf.png", width: 85.0%),
  caption: [
    Autocorrelation of normalised power against lag in hours, busiest site per dataset.
    Linear predictability halves within two to three hours --- well inside the six-hour forecast
    horizon. The negative excursion on `goes_pvdaq` is the half-period of the diurnal
    arc, not a modelling artefact.
  ]
)
<fig:acf>

=== Ramps are rare, large, and where the difficulty concentrates
@fig:ramps showed the step-to-step change distribution: sharply peaked at zero
with heavy, near-exponential tails on a logarithmic count axis. Most steps are nearly
constant; a small minority move by a third of nameplate capacity in a single interval. A
model optimised for average error can ignore the tails almost entirely and still score well,
which is exactly why the ramp subset of @sec:metrics exists as a separate
reporting column. The spikes at the extremes of both panels are the clipping bounds of the
histogram, not a physical accumulation.

=== Covariate structure: strong, redundant, and incomplete
@fig:correlations gives the Pearson and Spearman correlation structure over
daytime rows. Three observations shape the modelling.

The covariates are #emph[strong];: normalised power correlates at 0.74 with shortwave
radiation, 0.69 with direct radiation and 0.60 with direct normal irradiance. A model with
access to the covariate track starts from a substantial advantage over one restricted to the
power history --- which is visible in the leaderboard, where every covariate-consuming model
outranks DLinear.

The covariates are #emph[redundant];: shortwave and direct radiation correlate at 0.93 with
each other, and direct normal irradiance with cloud cover at $- 0.70$. There are not eight
independent weather channels here; there are perhaps three degrees of freedom presented eight
ways. This is the structure that variate-attention architectures exploit and that
channel-independent ones cannot, and it is the most plausible explanation for the ordering
between iTransformer and PatchTST in @cha:benchmark.

The covariates are #emph[incomplete];, which is the point that motivates the entire visual
branch of this thesis. @fig:hexbin shows the joint density of power against each
driver. The relationship with shortwave radiation is broad rather than tight: at
400 W/m$""^2$ observed power ranges over most of the unit interval. Cloud cover, remarkably,
shows almost no marginal relationship at all --- the density is nearly uniform in the
horizontal direction --- because cloud cover as an hourly areal fraction says little about
whether the sun is occluded at a specific site at a specific minute.

#figure(image("figures/correlation_matrices.png"),
  caption: [
    Pearson (left) and Spearman (right) correlation over daytime rows. The covariate
    block is strongly inter-correlated --- 0.93 between shortwave and direct radiation --- so the
    effective dimensionality of the weather track is far below its eight channels.
  ]
)
<fig:correlations>

#figure(image("figures/feature_vs_power_hexbin.png"),
  caption: [
    Joint density of normalised power against each weather driver (logarithmic density,
    150k daytime sample). The irradiance panels show broad, cone-shaped relationships rather than
    tight ones; the cloud-cover panel is nearly uninformative marginally, because an hourly areal
    cloud fraction does not determine occlusion at a point.
  ]
)
<fig:hexbin>

=== There is residual cloud information beyond irradiance
@fig:cloud-residual isolates the effect that motivates a visual modality. Daytime
observations are binned by shortwave radiation, and within each bin further split by cloud
cover. If the reanalysis irradiance already captured the cloud state, the three boxes within
a bin would coincide. They do not. In the 250--450 W/m$""^2$ bin the median falls from 0.33
under clear skies to 0.27 under mixed conditions to 0.24 under overcast, and the same
monotone ordering holds in every bin. Conditioning on cloud state adds information
#emph[after] conditioning on irradiance.

This is the empirical case for the thesis. There exists cloud-state information not carried
by the numerical channels, it is exactly the kind of information a satellite crop observes
directly, and the size of the effect --- roughly 0.09 in normalised power at fixed irradiance
--- is of the same order as the differences separating models in the leaderboard.
@cha:benchmark reports that no model consuming real frames converts this into
skill, which makes the figure the sharpest statement of the gap between what is available and
what is exploited.

#figure(image("figures/cloudcover_residual_effect.png"),
  caption: [
    Normalised power by irradiance bin, split by cloud cover within each bin. The
    monotone separation of the three boxes at fixed irradiance is residual cloud-state
    information that the numerical covariate track does not carry --- and that a satellite frame
    observes directly.
  ]
)
<fig:cloud-residual>

=== The second track behaves differently
@fig:weekgoes shows the same week-long view for a `goes_pvdaq` site. The
contrast with @fig:weektrace is the reason the two tracks are kept separate in
every table. The cadence is twice as fast, the capacity factor is markedly higher, the daily
arcs are broader --- a lower-latitude summer --- and the power curve tracks the radiation
envelope more closely, with fewer of the deep intra-day collapses that characterise the UK
fleet.

A model that transfers between these tracks is transferring across sensor (greyscale
high-resolution visible against three-channel GOES), spatial scale (a 128 km crop at 1 km
resolution against a 256-pixel RGB frame), capacity class (kilowatts against hundreds of
kilowatts), climate, latitude and sampling cadence simultaneously. That is a genuine
distribution shift and the reason scenario S3 is the strongest available test of
generalisation --- and the reason its absence from the executed scenarios is the principal
limitation of this work.

#figure(image("figures/week_goes_pvdaq.png"),
  caption: [
    One week at a `goes_pvdaq` site, at 15-minute cadence. Compare
    @fig:weektrace: higher capacity factor, broader arcs, closer tracking of the
    radiation envelope. The two tracks are different forecasting problems sharing a schema.
  ]
)
<fig:weekgoes>

=== Between-plant heterogeneity
@fig:siteboxplot shows the distribution of daytime normalised power per site.
Even after capacity normalisation, medians range from below 0.10 to above 0.55, and the
inter-quartile ranges differ by a factor of three. Part of this is geography --- the
north--south gradient visible in @fig:sitemap --- and part is per-installation
factors that the dataset does not record: tilt, azimuth, shading, module technology,
inverter clipping behaviour.

This heterogeneity is what makes cross-plant transfer non-trivial and what a temporal split
would hide entirely. A model fitted on one plant and tested on the same plant absorbs its
orientation and shading into its parameters; a model that must generalise across plants
cannot, and has to recover whatever is recoverable from a two-week history at inference
time. The `goes_pvdaq` sites (orange) sit systematically higher, reflecting both
lower latitude and a different mix of installation types --- which is why cross-dataset
transfer is a genuine distribution shift and not a formality.

#figure(image("figures/site_norm_power_boxplot.png"),
  caption: [
    Daytime normalised power by site, random sample of sites from both tracks. Capacity
    normalisation removes the scale difference but not the difference in #emph[capacity factor];;
    the residual spread is the between-plant variation a cross-plant model must absorb.
  ]
)
<fig:siteboxplot>

=== The target distribution is zero-inflated and right-skewed
@fig:powerdist shows why the target is awkward for a general-purpose sequence
model. Over all rows the distribution has a dominant spike at zero --- night, outage and
deep-overcast steps --- and a long right tail that never approaches the theoretical maximum
of one. Restricting to production steps removes the spike but leaves a strongly right-skewed
distribution with most mass below 0.3. In absolute units the picture is different again:
log-power is roughly bimodal, reflecting the two capacity populations of the two tracks.

Three modelling consequences follow. A model whose output head assumes an unbounded,
approximately symmetric predictive distribution --- as most general-purpose foundation model
heads do --- is mismatched at both ends: it can predict negative power, and it wastes
probability mass above the observed ceiling. The zero spike means that a large fraction of
the loss is contributed by observations that are trivially predictable given solar geometry,
which inflates apparent accuracy unless evaluation is restricted to daylight steps --- as it
is here. And the right skew means mean-squared objectives are dominated by a minority of
high-output steps, which is one reason NMAE and NRMSE order models differently in
@cha:benchmark.

#figure(image("figures/power_distributions.png"),
  caption: [
    Target distribution. Left: all rows, showing the dominant zero spike. Centre:
    production steps only, right-skewed with most mass below 0.3 and an effective ceiling near
    0.85 rather than 1.0. Right: $log_10$ absolute power, whose bimodality reflects the two
    capacity populations.
  ]
)
<fig:powerdist>

=== Per-plant driver relationships are stable
@fig:capfactor decomposes the between-plant variation of
@fig:siteboxplot into two histograms. Daytime mean capacity factor is tightly
concentrated for `uk_pv` --- most sites between 0.20 and 0.32 --- with the
`goes_pvdaq` sites spread much wider and reaching 0.54. But the per-site correlation
between normalised power and shortwave radiation is remarkably consistent: nearly every
`uk_pv` site falls between 0.70 and 0.87, and the `goes_pvdaq` sites reach
0.94.

This is the empirical basis for believing cross-plant transfer is possible at all. Plants
differ substantially in their #emph[level] --- how much they produce --- but very little in
the #emph[structure] of their response to the dominant driver. A model that learns the
irradiance-to-power relationship on one fleet has learned something that holds on another;
what it cannot carry over is the plant-specific scaling, which is exactly what the short
inference history is there to supply.

#figure(image("figures/site_capacity_factor_and_corr.png"),
  caption: [
    Left: daytime mean capacity factor per site. Right: per-site Pearson correlation
    between normalised power and shortwave radiation. Levels differ substantially across plants;
    the structure of the driver relationship barely does. Cross-plant transfer is a problem of
    recovering level, not of relearning structure.
  ]
)
<fig:capfactor>

=== Coverage and missingness
@fig:availability shows per-site data availability by month. Coverage is dense
but not complete: a minority of sites begin reporting only in mid-2019, and a few have
interior gaps. Nothing is imputed across these gaps --- windows that would span one are
simply not generated --- which keeps the evaluation honest at the cost of a slightly uneven
sampling of forecast origins across sites.

The structural missingness matters more than the incidental kind. Night steps have no
frames at all, and outage and stuck-sensor rows are masked out of the target. The visual
mask is therefore correlated with time of day by construction, and any model consuming the
visual stream must handle a systematically absent modality rather than a randomly absent
one. This is the reason modality dropout is part of the tensor contract of
@sec:contract rather than a training-time detail.

#figure(image("figures/site_availability_timeline.png", width: 78.0%),
  caption: [
    Per-site availability by month (blue indicates data present). Coverage is broadly
    dense; a minority of sites start mid-2019 or carry interior gaps. No values are imputed
    across gaps --- affected windows are not generated.
  ]
)
<fig:availability>

Quantitatively, coverage is better than the timeline suggests. @fig:coverage
shows the fraction of days within each site's span that carry data: every `uk_pv` site
exceeds 99.3 percent, and forty-four of them are complete. The gaps visible in
@fig:availability are therefore late #emph[starts] rather than interruptions ---
a set of sites that joined the fleet mid-2019 --- which affects how many windows a site
contributes but not the continuity of the windows it does contribute.

#figure(image("figures/coverage_per_site.png"),
  caption: [
    Percentage of days within each site's span that carry data. Coverage exceeds 99.3
    percent for every `uk_pv` site. The apparent gaps in @fig:availability are
    late starts, not interruptions.
  ]
)
<fig:coverage>

=== Seasonality and the two-year span
@fig:monthly shows mean normalised power by calendar month across the full
`uk_pv` span. The seasonal cycle is large --- a factor of eight between December
(0.05) and May (0.42) --- and it repeats closely across the two years, with 2020 running
slightly higher in spring than 2019.

Two consequences for evaluation follow. Because the split is by plant rather than by time,
every plant contributes windows from every season, so no model is advantaged by a favourable
temporal slice; the seasonal cycle is common-mode across train, validation and test. And
because the span covers two full years, the seasonal cycle is observed twice, which is the
minimum required for a fourteen-day context window to be interpretable: with a single year, a
model could not distinguish a seasonal trend from a secular one. The single-month
`goes_pvdaq` overlap point is a reminder that the second track covers only nine
months and cannot support the same seasonal analysis.

#figure(image("figures/monthly_production.png"),
  caption: [
    Mean normalised power by month over the full span. The seasonal cycle spans a factor
    of eight and repeats closely across the two years. Plant-based splitting makes this cycle
    common-mode across train, validation and test.
  ]
)
<fig:monthly>

== Splits
<sec:splits>
Splits partition #emph[plants];, are generated once with seed 42 at a per-dataset
70/15/15 ratio, exclude sites carrying `bad_site_flag`, and are committed to a
version-controlled file. Disjointness is asserted programmatically at every data load
rather than trusted; a leak would raise rather than silently improve a score.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,center,center,center,left,),
    table.header([#strong[Role];], [#strong[Plants];], [#strong[Rows];], [#strong[Valid steps];], [#strong[Purpose];],),
    table.hline(),
    [Train], [69], [850,654], [846,633], [Fit parameters, fine-tune backbones],
    [Validation], [15], [184,899], [182,809], [Early stopping, hyperparameter and mixing-weight
    tuning],
    [Test], [14], [172,656], [171,543], [Final reporting only],
    [Excluded], [2], [---], [---], [`bad_site_flag`: sites 7239, 8587],
  )]
  , caption: [The `uk_pv` plant partition. Exact site membership is listed in
  @app:splits.]
  , kind: table
  )
<tab:splits>
The companion `goes_pvdaq` split is 7 train / 2 validation / 1 test. With ten
plants, a fixed 15% test share is one or two plants, and per-plant variance would dominate
any comparison; that track is therefore additionally evaluated #emph[leave-one-plant-out];,
reporting mean and standard deviation over folds.

#[
#set par(first-line-indent: 0pt)
*A limitation of random plant assignment.* Assigning plants to splits at random
does not guarantee #emph[spatial] disjointness. Two `uk_pv` rooftops a few
kilometres apart experience nearly the same cloud field, so a random split can place a test
plant's near-twin in the training set and leak spatial information that a
distance-based or region-based split would withhold. The split used here is random because
it is the convention the baseline literature follows and because changing it mid-study
would invalidate cross-comparison; a region-based variant is identified as future work in
@sec:future. The effect is expected to inflate all models equally rather than
to favour any particular one, but it inflates the absolute numbers.
]

== The tensor contract
<sec:contract>
Every model in this thesis --- baselines and proposed architecture alike --- consumes the
same dictionary emitted by the dataset adapter. Fixing this contract is what makes the
comparison of @cha:baselines a comparison of #emph[models] rather than of data
pipelines. With $N$ entities, history $T$, horizon $H$ and $T_v$ frames:

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Key];], [#strong[Shape];], [#strong[Content];],),
    table.hline(),
    [`Y`], [$(N \, T \, C_(upright("tgt")))$], [Normalised historical target],
    [`Y_future`], [$(N \, H \, C_(upright("tgt")))$], [Ground-truth horizon],
    [`X_cov`], [$(N \, T + H \, C_(upright("cov")))$], [Historical and future covariates],
    [`V`], [$(N \, T_v \, C_(upright("img")) \, H_(upright("img")) \, W_(upright("img")))$], [Frames normalised
    to $[0 \, 1]$],
    [`timestamps`], [$(T + H \,)$], [Unix epoch over the full window],
    [`timestamps_v`], [$(T_v \,)$], [Unix epoch of the frames],
    [`entity_ids`], [$(N \,)$], [Plant identifiers],
    [`mask_target`], [$(N \, T \, C_(upright("tgt")))$], [Validity of historical targets],
    [`mask_future`], [$(N \, H \, C_(upright("tgt")))$], [Evaluation mask],
    [`mask_visual`], [$(N \, T_v)$], [Validity of each frame],
    [`mask_modality_dropout`], [$(N \, 2)$], [Numeric / visual dropout],
    [`adj_matrix`], [$(N \, N)$], [Precomputed spatial adjacency],
  )]
  , caption: [The canonical batch dictionary. All models read this and nothing else.]
  , kind: table
  )
<tab:contract>
=== Notes for re-use
Six points would save a future user of this dataset a considerable amount of time, and they
are collected here rather than scattered through the chapter.

+ #strong[Read frames only through `image_h5_index`];, and remember it is
  group-local. `image_uk128_index` is dead; `image_index` is an
  approximation retained for compatibility.

+ #strong[Respect `bad_site_flag`] before generating splits. The committed
  `goes_pvdaq` partition predates the flags and must be regenerated.

+ #strong[Do not standardise per series.] Use capacity normalisation; anything
  estimated from the series leaks under a cross-plant protocol.

+ #strong[Distinguish zero from missing.] Night power is genuinely zero; outage power is
  absent. The masks carry the distinction and the target column does not.

+ #strong[Do not resample across tracks.] The two cadences are native and physical-time
  windows resolve to different step counts; pooling raw step-horizon metrics across them is
  a category error.

+ #strong[Expect a structurally missing visual stream.] There are no night frames, so
  visual availability is correlated with time of day by construction.

Two design decisions in this contract are worth stating explicitly. Masks are carried
alongside the data rather than encoded as sentinel values, so that a model can distinguish
"zero power" from "no measurement" --- a distinction that matters enormously at dawn and
dusk. And `mask_modality_dropout` is part of the contract rather than a training
detail, so that robustness to missing frames can be evaluated under exactly the mechanism
used during training.


= Evaluation Protocol
<cha:protocol>
This chapter is short and load-bearing. Every number in @cha:baselines
and~@cha:expdesign is only as credible as the protocol that produced it, and the
protocol is where a multimodal cross-plant study is most easily compromised --- by a
normaliser fitted on test plants, by a horizon that differs between models, by a retrieval
datastore that quietly contains the answer.

== Three fairness principles
<sec:fairness>
+ #strong[Same horizon and granularity for every model.] All models forecast the same
  future window from the same history cadence. Context length is itself a fairness
  variable: a model given more history is not a better model, so no comparison is made
  across differing $T$.

+ #strong[Disjoint test plants, with no statistic leakage.] No model may train on, or
  derive statistics from, held-out plants. This includes normalisers --- hence
  capacity normalisation (@sec:harmonisation) rather than per-series
  standardisation.

+ #strong[No domain physics inside models.] Clear-sky-index conversions and
  irradiance-to-power formulas are excluded from every learned model, so that measured
  improvements are attributable to modelling rather than to a hand-coded solar prior. The
  sole exemption is smart persistence, which is the physics reference itself.

=== The failure modes these principles prevent
Each principle exists because of a specific way a cross-plant multimodal benchmark can be
compromised, usually without anyone noticing. Naming the failure mode makes the rule
checkable.

#[
#set par(first-line-indent: 0pt)
*Normaliser leakage.* The most common and least visible failure. Standardising each
series by its own mean and standard deviation is routine in forecasting, and under a
cross-plant protocol it silently passes test-plant statistics into the model. The symptom is a
benchmark on which every method looks unusually strong and no method generalises in
deployment. Capacity normalisation eliminates the channel entirely, because nameplate capacity
is metadata, not a measurement.
]

#[
#set par(first-line-indent: 0pt)
*Context-length mismatch.* If one model receives fourteen days and another six
hours, the comparison measures history length rather than architecture. The failure is easy to
introduce accidentally, because foundation models have fixed maximum contexts and quietly
truncate. The rule here is that context length is a fairness variable: models that cannot
consume the full window are documented as such rather than silently compared.
]

#[
#set par(first-line-indent: 0pt)
*Transductive retrieval.* A retrieval baseline whose datastore contains test-plant
history is performing in-context memorisation of the evaluation target. The resulting number
can be excellent and means nothing about zero-shot transfer. The datastore rule closes this,
and the transductive condition --- which is a legitimate thing to study --- is reported
separately if at all.
]

#[
#set par(first-line-indent: 0pt)
*Privileged covariates.* Supplying observed future weather over the horizon is not
forecasting; it is forecasting given a perfect weather forecast. Because covariate arrays
naturally span history and horizon, this is a single indexing decision away at all times. The
protocol partitions covariates into deterministic and observed classes
(@app:covariates), and models receiving the observed class over the horizon are
labelled oracle and excluded from competitive comparison.
]

#[
#set par(first-line-indent: 0pt)
*Physics smuggled into a model.* A clear-sky conversion inside a learned model
converts a modelling result into a statement about how good the clear-sky model is. Since
smart persistence --- which is that physics --- is the reference, a model that internalises it
is partly competing against itself. The exemption is granted to exactly one model, and it is
the reference.
]

#[
#set par(first-line-indent: 0pt)
*Metric-window mismatch.* If two models are scored on different evaluation windows,
their metrics are not comparable even under an identical protocol. This is the defect behind
the flagged Time-VLM row of @cha:baselines, and it is the reason skill scores are
always computed against the reference evaluated on the same windows as the model being
scored, rather than against a single global reference value.
]

== Windows in physical time
<sec:windows>
Windows are defined in physical time and resolved to steps per dataset, rather than being
defined in steps.

- #strong[History $T$ = 14 days] --- 672 steps at the `uk_pv` 30-minute
  cadence, 1344 at the `goes_pvdaq` 15-minute cadence. Two weeks is
  deployment-realistic (a newly connected plant will have that much history) and gives
  every model multiple diurnal cycles, which shorter contexts denied the foundation
  models in preliminary work. Foundation models with a fixed maximum context consume
  their most recent supported window.

- #strong[Horizon $H$ = 6 hours] --- 12 and 24 steps respectively. Skill-decay curves
  are additionally reported at 1, 6 and 24 hours, the last covering the day-ahead case.

- #strong[Visual window $T_v$ = 8 frames] over a short recent window of three to six
  hours, #emph[decoupled] from the numerical history. Vision carries information only over
  the cloud-advection horizon; widening the visual window past a few hours adds compute,
  not signal. This asymmetry --- long numerical context, short visual context --- is a
  design commitment of the architecture in @cha:method, not merely a budget
  decision.

#[
#set par(first-line-indent: 0pt)
*The cadence rule.* Because windows are physical, step counts differ across
datasets: 672 against 1344 for the same fourteen days. A step-horizon of 12 therefore means
six hours in one dataset and three in the other. Two hard rules follow. Physical lead time
is reported beside every per-dataset table; and metrics are never pooled across cadences in
raw form --- cross-dataset aggregation uses only the scale-free statistics of
@sec:aggregation.
]

== Metrics
<sec:metrics>
All metrics are computed on the capacity-normalised target over daylight, valid steps, per
plant, and then macro-averaged over plants so that large or data-rich plants do not
dominate.

#[
#set par(first-line-indent: 0pt)
*Point accuracy.* With $M$ evaluation samples, horizon $H$ and plant capacity
$C_i$:
$ upright(N M A E) = frac(1, M H) sum_(i = 1)^M sum_(h = 1)^H lr(|hat(y)_(i \, h) - y_(i \, h)|) / C_i \, #h(2em) upright(N R M S E) = sqrt(frac(1, M H) sum_(i = 1)^M sum_(h = 1)^H (frac(hat(y)_(i \, h) - y_(i \, h), C_i))^(#h(-1em) 2)) . $
]

#[
#set par(first-line-indent: 0pt)
*Skill score.* The headline number, defined against smart persistence and
computed from NRMSE:
$ upright(S S) = 1 - upright(N R M S E)_(upright("model")) / upright(N R M S E)_(upright("smart persistence")) . $
A score of zero means no better than the reference; one means perfect; negative means
worse than a model that assumes cloud conditions persist. An NMAE-based variant may be
reported as a secondary column but must be labelled as such, since the two rank models
differently.
]

#[
#set par(first-line-indent: 0pt)
*Probabilistic accuracy.* For models with a distributional output, the continuous
ranked probability score is approximated by the pinball loss averaged over the quantile
grid~@gneiting. It rewards forecasts that are both accurate and honestly calibrated,
and it is the only metric here that penalises overconfidence. Calibration is additionally
reported as empirical coverage of the nominal 80% interval.
]

#[
#set par(first-line-indent: 0pt)
*Variance tracking.* The squared Pearson correlation $R^2$ between prediction and
truth measures how much of the #emph[shape] of the day a forecast tracks, independently of
bias or scale.
]

#[
#set par(first-line-indent: 0pt)
*Reading $R^2$ and skill together.* These two must be read jointly, because their
disagreement is diagnostic. A model with high $R^2$ and low skill tracks the timing of the
day correctly but sits off the diagonal --- a bias or scale problem, fixable by
recalibration. A model with low $R^2$ and moderate skill is doing the opposite. In
@cha:baselines several multimodal baselines exhibit exactly the first pattern,
and the distinction changes the interpretation of their ranking entirely.
]

#[
#set par(first-line-indent: 0pt)
*Ramp regime.* Point metrics are additionally reported on a #emph[ramp subset];:
the top decile of $lr(|Delta y|)$ per site, that is, the cloud-transition periods. This is
where a visual modality should earn its tokens --- clear-sky periods are won by persistence
--- so the ramp columns, not the aggregate ones, are the sharpest test of the multimodal
hypothesis.
]

=== Metrics considered and rejected
Three alternatives were considered and are recorded here with the reason for rejection,
because a metric choice that is not justified is a metric choice a reader has to take on
trust.

#[
#set par(first-line-indent: 0pt)
*Mean absolute percentage error.* Undefined at zero and unbounded near it. The
target is exactly zero for roughly half of every day and arbitrarily close to it at dawn and
dusk, so any percentage-of-actual metric is dominated by the least interesting observations
in the dataset.
]

#[
#set par(first-line-indent: 0pt)
*Per-series standardised error.* Normalising each plant's series by its own mean and
standard deviation is standard in general-purpose forecasting benchmarks. It is prohibited
here: those statistics would have to be computed on test-plant data, which is precisely the
leakage the cross-plant protocol exists to prevent. Capacity normalisation uses a quantity
known at installation and is leak-free by construction.
]

#[
#set par(first-line-indent: 0pt)
*Energy-weighted error.* Weighting errors by absolute energy would emphasise
high-output periods, which is arguably what a grid operator cares about. It was rejected
because it would make the metric depend on the capacity mix of the test set, undermining
comparability across the two datasets --- and because the ramp subset already isolates the
high-consequence regime in a way that is transparent rather than baked into the headline
number.
]

=== Aggregation order matters
Metrics are computed #emph[per plant and then averaged];, not pooled over all observations.
The two differ whenever plants contribute unequal numbers of valid steps or have different
capacity factors, which they do (@fig:siteboxplot, @fig:availability).
Pooling would weight a data-rich, high-output plant more heavily than a sparse one, and the
resulting number would answer "how well does the model do on the average #emph[observation];"
rather than "how well does it transfer to the average #emph[plant];". Since the research
question is about generalisation across plants, each plant is one unit of evidence.

The choice has a practical consequence worth noting: macro-averaged metrics are noisier than
pooled ones, because a single difficult plant is not diluted. With fourteen test plants the
standard error of the mean is not negligible, which is a further reason to treat small
differences between models as ties.

== Scale-free aggregation
<sec:aggregation>
When results must be summarised across datasets of different cadence, or across scenarios,
only rank-preserving and scale-free statistics are used: win rate against the reference,
geometric-mean skill, and average rank. Raw step-horizon metrics are never pooled across
cadences.

== Scenario battery
<sec:scenarios>
Seven scenarios were specified. @tab:scenarios states each and, in the last
column, whether it was executed for this thesis --- an honest accounting matters more than
a complete-looking table.

#figure(
  align(center)[#table(
    columns: 4,
    align: (left,left,left,left,),
    table.header([#strong[ID];], [#strong[Split];], [#strong[Question];], [#strong[Run];],),
    table.hline(),
    [S1], [Train plants, held-out time range], [Sanity / upper bound], [no],
    [S2], [#strong[Disjoint test plants (primary)];], [Headline result (H3)], [#strong[yes];],
    [S3], [Train `uk_pv`, test `goes_pvdaq` and reverse], [Distribution-shift
    generalisation], [no],
    [S4], [S2 with $H in { 1 \, 6 \, 24 }$ h], [Skill-decay curves], [partial],
    [S5], [S2 with 10/25/50/100% of train plants], [Sample efficiency], [no],
    [S6], [S2 restricted to top-decile $lr(|Delta y|)$ windows], [Where vision should win], [#strong[yes];],
    [S7], [Train on a month subset, test unseen season], [Temporal shift], [no],
  )]
  , caption: [Scenario battery. Everything reported in this thesis is S2 on `uk_pv`,
  plus the S6 ramp subset and a partial S4 decay profile.]
  , kind: table
  )
<tab:scenarios>
=== What each scenario is for
#[
#set par(first-line-indent: 0pt)
*S1, in-domain.* Train plants, held-out time range. This is the protocol most of the
literature reports, included here only as a sanity ceiling: the gap between S1 and S2 measures
how much of a model's apparent accuracy is plant-specific memorisation.
]

#[
#set par(first-line-indent: 0pt)
*S2, cross-plant.* The headline. Disjoint test plants, short inference history, no
test-plant statistic anywhere. Everything reported in this thesis is S2.
]

#[
#set par(first-line-indent: 0pt)
*S3, cross-dataset.* Train on one track, test on the other. This is the strongest
generalisation test available in the data, because it shifts sensor, scale, climate, latitude
and cadence at once (@sec:eda). It also involves a cadence change, so results
must be reported with the physical lead time attached and aggregated only through the
scale-free statistics of @sec:aggregation. A robustness variant resamples the
faster track to the slower cadence so that step-horizon and physical horizon coincide.
]

#[
#set par(first-line-indent: 0pt)
*S4, long horizon.* The same protocol at one, six and twenty-four hours. Skill decay
curves are how a forecaster's useful range is communicated, and they separate models that are
good at short lead times from models that are good at persistence.
]

#[
#set par(first-line-indent: 0pt)
*S5, data efficiency.* S2 with 10, 25, 50 and 100 percent of training plants. This
is the canonical foundation-model claim --- that pretraining buys sample efficiency --- and it
is the scenario in which a pretrained backbone should beat a supervised one even if it does
not at full data. Its absence is why the comparison in @cha:benchmark between
supervised and pretrained models must be read as holding #emph[at 69 training plants] and not
in general.
]

#[
#set par(first-line-indent: 0pt)
*S6, ramp subset.* S2 restricted to the top decile of $lr(|Delta y|)$. The regime where
a visual modality should win, and therefore the sharpest test of the thesis.
]

#[
#set par(first-line-indent: 0pt)
*S7, seasonal transfer.* Train on a subset of months, test on an unseen season. A
temporal analogue of the cross-plant shift, included for completeness and not executed.
]

The unexecuted scenarios are not decoration. S3 and S5 are the standard evidence a
foundation-model claim is expected to provide --- transfer under distribution shift, and
accuracy as a function of training data --- and their absence is the principal limitation
of this work, restated in @sec:limitations.

== Input parity
<sec:parity>
@tab:parity states which inputs each tier may consume. A baseline never receives
information the models it is compared against cannot access.

#figure(
  align(center)[#table(
    columns: 6,
    align: (left,center,center,center,center,center,),
    table.header([#strong[Tier];], [$upright(bold(Y))$], [$upright(bold(X))_(upright("cov"))$], [$upright(bold(V))$], [#strong[Retrieval];], [#strong[Text];],),
    table.hline(),
    [T0 Reference], [yes], [clear-sky only], [---], [---], [---],
    [T1/T2 Supervised], [yes], [yes], [---], [---], [---],
    [T3 Foundation models], [yes], [where native], [---], [---], [---],
    [T4 Adaptation], [yes], [yes (CoRA)], [---], [yes (RAG)], [---],
    [T5/T6 Multimodal], [yes], [yes], [yes], [model-specific], [model-specific],
    [MMTSFM (ours)], [yes], [yes], [yes], [optional (H4)], [---],
  )]
  , caption: [Input-parity matrix.]
  , kind: table
  )
<tab:parity>
Two rules within this matrix are the ones a critical reader attacks first, and both are
enforced.

#[
#set par(first-line-indent: 0pt)
*The retrieval datastore rule.* Retrieval and memory baselines may populate their
datastore with #emph[training-plant data only];. Allowing test-plant history into the
datastore converts zero-shot forecasting into transduction, and would make the retrieval
tier incomparable with everything else. Transductive retrieval is a legitimate but
#emph[separate] condition and never enters the headline table.
]

#[
#set par(first-line-indent: 0pt)
*The oracle-covariate rule.* Two Chronos-2 variants are reported that consume
#emph[observed future weather] over the forecast horizon --- temperature, cloud cover,
irradiance that would not be known at forecast time. They are labelled #emph[oracle] and
are upper bounds, not competitors. The honest variants receive only deterministic future
covariates: solar geometry and calendar encodings. The gap between the two is one of the
most informative measurements in this thesis (@sec:bar), but it must
never be read as a competitive result.
]

Two further parity constraints apply to the multimodal tier. Models that natively want text
(Solar-VLM) may generate weather descriptions only from covariates available to every other
model --- no external weather service. And no model may consume a normaliser, calibration
constant or context-selection heuristic derived from the test plants.

== Reproducibility
<sec:reproducibility>
Reproducibility here means something stronger than "the code is available". It means that a
number in this thesis can be traced to the job that produced it, that the job can be
re-specified exactly, and that the data it consumed cannot have changed underneath it. Three
mechanisms enforce that.

#emph[The data is immutable.] The dataset of record is treated as read-only, and no result
depends on re-processing it. Splits are committed to version control rather than regenerated
at run time, so a later run cannot silently receive a different partition; their disjointness
is asserted at load, so a leak raises rather than quietly improving a score.

#emph[The configuration is declarative and complete.] Every hyperparameter is composed by
Hydra from configuration groups, and none is hardcoded in model code. Each baseline's
configuration is self-contained in its own directory, so no model's behaviour depends on a
global setting that another model also reads --- a failure mode that is invisible until two
models disagree for reasons unrelated to their architectures.

#emph[The experiment is registered before it runs.] Each experiment declares a one-sentence
hypothesis, a configuration diff, a branch and a nominated comparison baseline before the job
is submitted. This is a guard against a specific and common failure: reframing an experiment
after the fact as a test of whatever it happened to show.

A fixed global seed of 42 governs data loaders, weight initialisation and dropout.
Configuration is Hydra-only~@hydra, with each baseline's hyperparameters
self-contained in its own directory, so that no model's behaviour depends on a global
setting another model also reads. Dependencies are managed with a single lockfile. Training
runs on the Leonardo cluster under SLURM, with standard output and error retained per job,
so that any reported number can be traced to the job that produced it. Splits are committed
to version control rather than regenerated, and their disjointness is asserted at load time.


= The Baseline Suite
<cha:baselines>
A benchmark is only as informative as the alternatives it contains. This chapter describes
the twenty-eight forecasters evaluated under the protocol of @cha:protocol:
why each is present, what it consumes, how it was implemented, and where the
implementation departs from the published method. @cha:benchmark reports what
they scored.

The organising principle is adversarial rather than encyclopaedic. Each model exists to
close one escape route --- one alternative explanation of a multimodal gain that a
sceptical reader would otherwise be entitled to propose. A suite assembled this way is
smaller than a survey and more useful than one, because every row has a job.

== Suite design
<sec:suite-design>
#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Tier];], [#strong[Models];], [#strong[Question it answers];],),
    table.hline(),
    [T0], [Persistence, #strong[smart persistence];, hourly climatology, seasonal naive], [Is the task trivial? What is the reference floor?],
    [T1], [LightGBM (one model per quantile), TabPFN-3], [Does a tabular model already suffice?],
    [T2], [MLP, DLinear, PatchTST, iTransformer, TFT], [Is a simpler or a purely supervised model
    better?],
    [T3], [Chronos-2 zero-shot and fine-tuned (plus oracle variants), TimesFM~2.5, TiRex, TTM-R2
    zero-shot and fine-tuned], [Do foundation models transfer? Is any finding
    backbone-specific?],
    [T4], [TS-RAG, Cross-RAG, CoRA], [Would retrieval or a covariate adapter close the gap without
    vision?],
    [T5], [Time-VLM, Aurora, VisionTS++, UniCast], [Do pseudo-images or prompting suffice?],
    [T6], [Solar-VLM, CrossViVit, SUNSET], [Does the domain state of the art already win?],
  )]
  , caption: [The baseline suite and the question each tier answers.]
  , kind: table
  )
<tab:suite>
The tiers span the three adaptation strategies of @sec:adaptation --- full
supervision (T1--T2), zero-shot transfer (T3), and frozen-backbone adaptation (T4) --- which
is what allows @cha:benchmark to compare them directly rather than by
citation. They also span the endogenous/exogenous distinction of
@sec:multimodal: T5 contains models whose second modality is synthesised from
the numerical series, T6 models that consume genuine satellite frames. Holding the protocol
fixed across that boundary is what makes the comparison of the two classes meaningful, and
it is the comparison that produces the sharpest result in this thesis.

#[
#set par(first-line-indent: 0pt)
*The rebuttal matrix.* @tab:rebuttal states the mapping from objection
to evidence explicitly. It is reproduced here because it is the actual specification the
suite was built against.
]

#figure(
  align(center)[#table(
    columns: 2,
    align: (left,left,),
    table.header([#strong[Objection];], [#strong[Baseline that answers it];],),
    table.hline(),
    ["A simpler model would do as well"], [DLinear, MLP, persistence family],
    ["The gain is just from covariates"], [CoRA, TFT, LightGBM (all covariate-rich, no vision)],
    ["Retrieval would give you the same thing"], [TS-RAG, Cross-RAG],
    ["The result is specific to your backbone"], [TimesFM 2.5, TiRex, TTM-R2 (three other
    families)],
    ["You never showed the model reads the images"], [Shuffled-frames and mismatched-plant
    controls (@cha:expdesign)],
    ["Domain-specific models already do this"], [Solar-VLM, CrossViVit, SUNSET],
    ["Fusion depth does not matter"], [Late-fusion arm; UniCast as published shallow prompting],
    ["Pseudo-images would be cheaper"], [Time-VLM, VisionTS++],
  )]
  , caption: [Objection-to-evidence mapping. Every cell must be occupied before the multimodal
  claim can be defended.]
  , kind: table
  )
<tab:rebuttal>
== Tier 0 --- reference models
<sec:tier0>
Reference models establish the floor and calibrate the reader's sense of how much of the
signal is trivially predictable. On solar data this matters more than in most domains,
because the diurnal cycle alone accounts for a very large fraction of the variance
(@fig:diurnal), and a metric that rewards predicting the sun's position is
uninformative about forecasting skill.

#[
#set par(first-line-indent: 0pt)
*Persistence.* The last observed value is carried flat across the horizon. This is
the absolute floor. At very short lead times it is surprisingly hard to beat --- power
changes slowly under stable conditions --- and its failure mode is exactly the diurnal
ramp: at 08:00 it predicts that 08:00 output will persist until 14:00.
]

#[
#set par(first-line-indent: 0pt)
*Smart persistence.* The clearness index --- the ratio of observed power to a
clear-sky model --- is carried forward instead of the power itself, and the forecast is
reconstructed by multiplying that index by the clear-sky curve over the horizon. This
respects the deterministic solar geometry and assumes only that #emph[cloud conditions]
persist. It is dramatically stronger than naive persistence and is the community's standard
reference; it is the denominator of the skill score in @sec:metrics. It is also
the only model in the study permitted to use clear-sky physics, because it is the physics
reference.
]

#[
#set par(first-line-indent: 0pt)
*Hourly climatology.* The mean normalised power for each month, hour and minute
bin, computed on training plants. It carries no information about the current day at all,
so it detects two pathologies: a model that fails to beat it is not using its inputs, and a
suspiciously strong climatology result indicates that the evaluation windows are dominated
by average conditions rather than by the transitions that matter.
]

#[
#set par(first-line-indent: 0pt)
*Seasonal naive.* The value observed at the same clock time on the previous day.
This is the standard reference in general-purpose forecasting benchmarks and is included so
that results here can be positioned against that literature.
]

== Tier 1 --- classical machine learning
<sec:tier1>
#[
#set par(first-line-indent: 0pt)
*LightGBM.* Gradient-boosted decision trees~@lightgbm over a flattened
feature vector of lagged power and covariates. A separate model is fitted per quantile with
the pinball objective, which yields a genuine probabilistic forecast rather than a point
estimate with error bars bolted on. Gradient boosting is the honest strong baseline for
tabular problems, and a deep model that cannot beat it has not justified its cost. In the
cross-plant setting it has a specific advantage: trees are insensitive to feature scaling
and therefore transfer across plants of different capacity without any special handling.
]

#[
#set par(first-line-indent: 0pt)
*TabPFN-3.* A tabular foundation model~@tabpfn3 in its time-series mode:
prior-fitted, performing inference in a single forward pass with no gradient-based fitting
on the target task. It is the tabular counterpart to the time-series foundation models of
Tier~3, and it tests whether the "foundation model" advantage, if any, is specific to
sequence architectures or is a property of large-scale prior fitting in general.
]

== Tier 2 --- supervised deep learning
<sec:tier2>
All Tier-2 models are trained from scratch on the training plants and must generalise to
disjoint test plants. This is the strongest form of the "do you actually need pretraining"
question: these models see the same data as the fine-tuned foundation models and have no
pretraining at all.

#[
#set par(first-line-indent: 0pt)
*MLP.* A multilayer perceptron over the flattened history window and covariates.
The simplest learned baseline, present to bound how much of the achievable skill requires
any temporal structure at all.
]

#[
#set par(first-line-indent: 0pt)
*DLinear.* <dlinear.>
A single linear layer applied to a trend/seasonal
decomposition~@dlinear. This is the model that embarrassed a generation of
forecasting transformers, and its presence is non-negotiable: reviewers ask for it, and on
strongly periodic signals it is frequently competitive. It consumes the target only, no
covariates, which makes it the cleanest measure of how much information is in the power
history alone.
]

#[
#set par(first-line-indent: 0pt)
*PatchTST.* A channel-independent patch transformer~@patchtst: the series is
divided into patches that act as tokens, and channels are processed independently with
shared weights. It is the strongest conventional supervised patch transformer and is
architecturally the closest supervised analogue to the Chronos-2 backbone, which makes the
comparison between them informative about pretraining specifically rather than about
architecture.
]

#[
#set par(first-line-indent: 0pt)
*iTransformer.* <itransformer.>
Attention is applied across #emph[variates] rather than across
time: each channel becomes a token and the attention matrix models inter-variable
structure~@itransformer. On a problem where the target is driven by a handful of
strongly correlated exogenous quantities --- irradiance, cloud cover, temperature
(@fig:correlations) --- this is an unusually well-matched inductive bias, and
@cha:benchmark shows the consequence.
]

#[
#set par(first-line-indent: 0pt)
*TFT.* The temporal fusion transformer~@tft combines gated residual
processing, variable selection and a quantile output head. It is included because it is
quantile-native, so it supplies a probabilistic comparison point outside the foundation-model
tier. The implementation here is a reduced variant, described in
@sec:implementation.
]

== Tier 3 --- time-series foundation models
<sec:tier3>
#[
#set par(first-line-indent: 0pt)
*Chronos-2, zero-shot.* The backbone of the proposed architecture, evaluated with
no adaptation~@chronos2. This row is the anchor for hypothesis H0 and the honest
starting point of the whole thesis: whatever MMTSFM achieves must be measured against what
its own backbone does untouched.
]

#[
#set par(first-line-indent: 0pt)
*Chronos-2, fine-tuned.* The same model fine-tuned on the training plants. This is
the single most important comparison row for the proposed architecture, because it holds the
backbone, the data and the protocol fixed and varies only the adaptation strategy. A
multimodal architecture that does not beat its own fine-tuned backbone has not demonstrated
anything about multimodality.
]

#[
#set par(first-line-indent: 0pt)
*Chronos-2 oracle variants.* Two additional rows in which the model receives
#emph[observed] future weather over the forecast horizon rather than only the deterministic
covariates known in advance. These are not competitors and are labelled as such throughout;
they quantify the headroom available from perfect conditioning, which turns out to be the
largest single measurement in the study (@sec:bar).
]

#[
#set par(first-line-indent: 0pt)
*TimesFM 2.5.* A decoder-only foundation model from an independent
lineage~@timesfm, present so that any conclusion about zero-shot transfer is not an
artefact of one pretraining corpus.
]

#[
#set par(first-line-indent: 0pt)
*TiRex.* <tirex.>
Built on the xLSTM family rather than on attention~@tirex, and among
the strongest zero-shot performers on general-purpose leaderboards. It is the third distinct
architecture family, which is the threshold at which a negative zero-shot result becomes a
statement about the task rather than about a model.
]

#[
#set par(first-line-indent: 0pt)
*TTM-R2, zero-shot and fine-tuned.* A compact mixer-based
model~@ttm of a few million parameters. It probes the scale axis directly: if
zero-shot performance is a function of capacity, a tiny model should fail and a large one
should not. The fine-tuned variant additionally shows how much of a weak zero-shot result
adaptation can recover.
]

== Tier 4 --- frozen-backbone adaptation
<sec:tier4>
This tier contains the methods that compete with the thesis directly. Each keeps a
pretrained backbone frozen and adds an external mechanism for supplying task-specific
information --- exactly the strategy of the proposed architecture, differing only in what is
supplied and how deeply it is coupled.

#[
#set par(first-line-indent: 0pt)
*TS-RAG.* Retrieval-augmented forecasting on a frozen backbone~@tsrag:
windows similar to the current context are retrieved from a datastore and fused into the
forecast. Per @sec:parity the datastore contains training-plant windows only,
and the mixing weight is tuned on validation plants. This is the "you did not need vision,
you needed precedent" objection in runnable form.
]

#[
#set par(first-line-indent: 0pt)
*Cross-RAG.* A stronger retrieval variant~@crossrag with clear-sky-aware
retrieval keys and per-step mixing weights, retrieving across series rather than only within
one. It is included so that the retrieval tier is represented by a tuned modern method
rather than by a strawman.
]

#[
#set par(first-line-indent: 0pt)
*CoRA.* <cora.>
Covariate-aware adaptation of a frozen backbone through a zero-initialised
residual adapter~@cora. This is the closest published competitor to the proposed
architecture: same frozen-backbone premise, same idea of injecting external information
through a learned interface, but coupling the modalities at a single insertion point rather
than throughout the temporal operator. If deep token-level fusion cannot beat CoRA, the
fusion-depth claim does not survive.
]

== Tier 5 --- endogenous multimodal
<sec:tier5>
Models whose second modality is synthesised from the numerical series or its covariates. No
external sensor is consumed. Their presence is what allows the thesis to separate visual
#emph[inductive bias] from visual #emph[information];.

#[
#set par(first-line-indent: 0pt)
*Time-VLM.* The series is rendered as an image and passed, with a textual
description, to pretrained vision-language backbones~@timevlm. Nothing enters the
model that was not already in the numerical track; what is exploited is the prior structure
of a visual encoder --- locality, periodicity, multi-scale texture --- applied to a
re-rendering of the input.
]

#[
#set par(first-line-indent: 0pt)
*VisionTS++.* A related approach mapping series to images to exploit a frozen
masked autoencoder backbone~@visiontspp, included to show that the endogenous strategy
is not a single lucky implementation.
]

#[
#set par(first-line-indent: 0pt)
*Aurora.* A multimodal foundation model evaluated zero-shot~@aurora, with
weather text templated from the covariate track. It answers the question "why not simply use
an off-the-shelf multimodal foundation model".
]

#[
#set par(first-line-indent: 0pt)
*UniCast.* <unicast.>
Vision and text embeddings injected as soft prompts into a frozen
time-series foundation model~@unicast. Architecturally this is the shallowest point on
the fusion-depth axis while still being a genuine multimodal method, which makes it the
natural contrast for the interleaving hypothesis. It is placed in this tier because in the
configuration run here no external text encoder is available and the visual path operates on
generic features.
]

== Tier 6 --- exogenous multimodal, photovoltaic domain
<sec:tier6>
Models that consume genuine satellite imagery, all of them from the photovoltaic
forecasting literature.

#[
#set par(first-line-indent: 0pt)
*Solar-VLM.* The primary domain state of the art~@solarvlm: satellite
imagery, covariates and generated weather text combined through a vision-language backbone.
Per the parity rules its text is generated only from covariates available to every other
model; no external weather service is consulted.
]

#[
#set par(first-line-indent: 0pt)
*CrossViVit.* <crossvivit.>
The reference deep satellite-plus-series architecture, cross-attending
video tokens into a temporal transformer~@crossvivit. It is the strongest non-foundation
multimodal competitor and the closest published relative of the proposed fusion mechanism ---
both couple modalities inside the encoder rather than at a readout, differing in whether the
coupling is cross-attention or token interleaving.
]

#[
#set par(first-line-indent: 0pt)
*SUNSET.* The canonical convolutional solar baseline~@sunset, a hybrid of an
image branch and a power branch. Nearly every paper in the domain compares against it, so its
presence anchors this benchmark to that literature.
]

== What was deliberately excluded
<sec:exclusions>
A benchmark's exclusions are as much a design decision as its inclusions, and leaving them
unstated invites the suspicion that they were chosen after the results were known. The
following were considered and left out, each for a stated reason.

#[
#set par(first-line-indent: 0pt)
*SPIRIT and PV-VLM.* Both are photovoltaic-domain multimodal methods that would
have strengthened Tier~6. They were excluded for effort budget rather than on principle:
Tier~6 already contains three domain models spanning convolutional, cross-attention and
vision-language designs, and a fourth would have added breadth without opening a new axis.
They remain the first additions a follow-up should make.
]

#[
#set par(first-line-indent: 0pt)
*FusionSF and M3S-Net.* Non-foundation photovoltaic fusion architectures. Both are
released against corpora structured for different tasks (hourly day-ahead windows), and
porting them to this protocol would have required changes substantial enough that the
resulting row would measure the port rather than the method.
]

#[
#set par(first-line-indent: 0pt)
*MEMTS and parametric-memory adapters.* These occupy the same niche as the
retrieval baselines --- external information supplied to a frozen backbone --- which is
already represented by two tuned methods. The marginal information from a third mechanism in
the same family was judged low.
]

#[
#set par(first-line-indent: 0pt)
*Additional foundation-model families.* Toto and Moirai were available and were not
run. The methodological requirement was three distinct architectural families evaluated
zero-shot, which Chronos-2, TimesFM, TiRex and TTM already satisfy; a fifth transformer would
not have changed the conclusion of @sec:finding2.
]

#[
#set par(first-line-indent: 0pt)
*Few-shot in-context adaptation curves.* Accuracy as a function of $K$ support days
per test plant is a different protocol --- context matching rather than zero-shot transfer ---
and mixing the two in one table would obscure both. The harness supports it, and it is
available as an appendix experiment on request, with $K in { 0 \, 1 \, 3 \, 7 }$ days used either as a
retrieval datastore or as a linear-probe update.
]

#[
#set par(first-line-indent: 0pt)
*Post-hoc recalibration.* Several models exhibit systematic scale bias
(@sec:qualitative) that an affine correction would substantially repair. It was
not applied, because the correction would have to be fitted somewhere --- on validation plants
at best --- and the resulting table would compare forecasters-plus-post-processing rather than
forecasters. The bias is reported instead.
]

== Implementation and deviations
<sec:implementation>
Third-party models were vendored and adapted to the tensor contract of
@sec:contract rather than reimplemented from their papers. The deviations
below are recorded because they bound how strongly the corresponding rows may be read; a
benchmark that hides its integration compromises is not reproducible.

- #strong[Tier 2] is served by a single port of a common deep time-series
  library~@tslib, so PatchTST, iTransformer, DLinear and MLP share one trainer and
  differ only in architecture. This is a deliberate fairness choice: differences between
  those four rows cannot be attributed to training-loop details.

- #strong[TFT] is a reduced variant retaining the quantile head and gated residual
  processing but not the full variable-selection stack. Its row should be read as a
  quantile-native supervised reference rather than as a faithful reproduction.

- #strong[TTM] uses the R2 checkpoint, not R3. The R3 release is a trend-plus-residual
  decomposition model whose parameter keys do not load into the available library version;
  loading it silently produces a randomly initialised network that trains and evaluates
  without error. Histories shorter than the model's context are edge-padded with an
  observed mask rather than zero-padded, since zero is a meaningful power value; fine-tuned
  variants receive a frequency token.

- #strong[CrossViVit] runs with single-channel imagery and synthesised coordinate
  encodings, because the `uk_pv` product is greyscale and the original model expects
  multi-channel input with precise geospatial coordinates. Its row is a conservative
  estimate of the published architecture.

- #strong[Time-VLM] runs on evaluation windows not bit-aligned with the rest of the
  suite. The number is reported and flagged rather than omitted; an aligned re-run is
  required before it is used as a precise comparison point.

- #strong[Retrieval baselines] populate datastores from training plants only, with
  mixing weights tuned on validation plants.

- #strong[Foundation models with fixed maximum context] consume their most recent
  supported window rather than the full fourteen days. This is stated rather than silently
  tolerated, because context length is a fairness variable
  (@sec:fairness).

== On tuning budgets
<sec:tuning>
The hardest fairness question in a benchmark of this kind is not which inputs a model
receives --- that is settled by the parity matrix --- but how much effort went into making
each one work. A model can lose because its architecture is unsuited to the task or because
nobody tuned it, and no protocol fully separates the two.

Three practices bound the problem here without solving it. Within Tier~2, four models share a
single trainer and a single hyperparameter search procedure, so differences #emph[within] that
tier are architectural rather than effort-driven --- which is what makes the iTransformer
against PatchTST comparison in @cha:benchmark interpretable. Across tiers,
vendored models retain their authors' published defaults wherever those defaults are
applicable, on the principle that a method's own authors are better placed to configure it
than a third party with a different dataset. And the methods most threatening to the thesis
--- the retrieval baselines and the covariate adapter --- were tuned rather than accepted at
defaults, because a benchmark whose strongest competitors are strawmen proves nothing.

The residual risk runs in a specific direction, and it should be stated: the proposed
architecture received far more attention from its author than any baseline did. Where that
asymmetry would matter most --- the comparison against the same backbone fine-tuned, and
against the vision-free arm of the same model --- the comparisons are internal, sharing the
recipe, the schedule and the data, which is precisely why those two rows and not an external
baseline are treated as the decisive ones in @cha:expdesign.

#[
#set par(first-line-indent: 0pt)
*A correction that changed a conclusion.* One row in an earlier revision of this
benchmark reported a catastrophic score for Time-VLM. The cause was a stale artefact: the
prediction file had been re-saved after inverse scaling, but the metric file was never
regenerated, so the table showed an error computed on standardised rather than physical
units. Re-scoring from the corrected predictions --- with no retraining --- moved the model
from last place to second. All other externally integrated models re-score bit-identically to
their recorded rows, so the defect was isolated; but the episode is recorded here because it
is the reason every row in @cha:benchmark was re-derived from saved predictions
rather than trusted from its original metric file.
]


= Benchmark Results
<cha:benchmark>
This chapter reports what the twenty-eight forecasters of @cha:baselines
scored under the protocol of @cha:protocol, and draws four conclusions from
the result. All numbers are cross-plant: fourteen disjoint test plants, fourteen days of
history, a six-hour horizon, capacity-normalised metrics macro-averaged over plants. No
test-plant data entered any fitted component of any model.

== Leaderboard
<sec:leaderboard>
#figure(
  align(center)[#table(
    columns: 7,
    align: (right,left,left,center,center,center,center,),
    table.header([#strong[\#];], [#strong[Tier];], [#strong[Model];], [#strong[NMAE] $arrow.b$], [#strong[NRMSE] $arrow.b$], [#strong[SS] $arrow.t$], [#strong[$R^2$] $arrow.t$],),
    table.hline(),
    [1], [T2], [#strong[iTransformer];], [0.0699], [0.1032], [#strong[0.552];], [0.765],
    [2], [T5], [#strong[Time-VLM];$""^dagger$], [0.0692], [0.1061], [#strong[0.540];], [0.487],
    [3], [T3], [Chronos-2 oracle FT$""^dagger.double$], [0.0808], [0.1142], [0.504], [0.706],
    [4], [T4], [TS-RAG], [0.0705], [0.1203], [0.478], [0.331],
    [5], [T4], [Cross-RAG], [0.0726], [0.1206], [0.477], [0.375],
    [6], [T3], [Chronos-2 oracle$""^dagger.double$], [0.0817], [0.1213], [0.474], [0.679],
    [7], [T2], [PatchTST], [0.0886], [0.1249], [0.458], [0.651],
    [8], [T6], [Solar-VLM], [0.0955], [0.1283], [0.443], [0.660],
    [9], [T2], [TFT], [0.0889], [0.1330], [0.423], [0.626],
    [10], [T2], [MLP], [0.0958], [0.1352], [0.413], [0.600],
    [11], [T1], [LightGBM], [0.1000], [0.1419], [0.384], [0.555],
    [12], [T4], [CoRA], [0.1025], [0.1444], [0.374], [0.550],
    [13], [T3], [TTM-R2 fine-tuned], [0.1029], [0.1465], [0.364], [0.520],
    [14], [T6], [CrossViVit], [0.1112], [0.1500], [0.349], [0.565],
    [15], [T1], [TabPFN-3], [0.1076], [0.1524], [0.339], [0.499],
    [16], [T3], [Chronos-2 fine-tuned], [0.1120], [0.1543], [0.331], [0.475],
    [17], [T2], [DLinear], [0.1131], [0.1556], [0.325], [0.462],
    [18], [T3], [TiRex zero-shot], [0.1145], [0.1642], [0.287], [0.440],
    [19], [T3], [TimesFM 2.5 zero-shot], [0.1172], [0.1680], [0.271], [0.429],
    [20], [T0], [Hourly climatology], [0.1353], [0.1766], [0.234], [0.314],
    [21], [T5], [Aurora], [0.1280], [0.1769], [0.232], [0.440],
    [22], [T6], [SUNSET], [0.1384], [0.1806], [0.216], [0.359],
    [23], [T3], [Chronos-2 zero-shot], [0.1376], [0.1873], [0.187], [0.277],
    [24], [T5], [UniCast], [0.1433], [0.2025], [0.121], [0.371],
    [25], [T0], [Seasonal naive], [0.1419], [0.2058], [0.107], [0.276],
    [26], [T5], [VisionTS++], [0.1690], [0.2266], [0.017], [0.397],
    [27], [T0], [Persistence], [0.1643], [0.2272], [0.014], [0.171],
    [---], [T0], [#emph[Smart persistence] (reference)], [0.1593], [0.2304], [0.000], [0.210],
    [28], [T3], [TTM-R2 zero-shot], [0.1704], [0.2490], [$- 0.081$], [0.155],
  )]
  , caption: [Cross-plant leaderboard on `uk_pv` (scenario S2). $""^dagger$ non-aligned
  evaluation windows, see @sec:implementation. $""^dagger.double$ oracle variants
  consume observed future weather and are upper bounds, not competitors.]
  , kind: table
  )
<tab:leaderboard>
#[
#set par(first-line-indent: 0pt)
*Probabilistic scores.* Continuous ranked probability scores are available only
for quantile-native models: Chronos-2 oracle fine-tuned 0.0630, Chronos-2 oracle 0.0635,
TFT 0.0689, LightGBM 0.0768, TabPFN-3 0.0815, CoRA 0.0816, Chronos-2 fine-tuned 0.0855,
Chronos-2 zero-shot 0.1072. The ordering broadly tracks the point metrics with one
instructive exception: TFT ranks ninth on skill but third on calibration. A model can be
mediocre at predicting the conditional mean and excellent at describing its own uncertainty,
and for a grid operator sizing reserve capacity the second property is not a consolation
prize.
]

#[
#set par(first-line-indent: 0pt)
*Metric disagreement.* NMAE and NRMSE do not induce the same ordering, and the
disagreements are diagnostic. TS-RAG has the second-best NMAE of the whole suite (0.0705)
but only the fourth-best NRMSE, and an $R^2$ of 0.331 --- far below models with similar
skill. It is very accurate on typical steps and poor on the large deviations that squared
error punishes. Solar-VLM shows the opposite profile: a mediocre NMAE with a high $R^2$ of
0.660, meaning it follows the shape of the day closely while being systematically
mis-scaled. The headline skill score, being NRMSE-based, rewards the first pattern less
than a mean-absolute framing would and the second more.
]

#[
#set par(first-line-indent: 0pt)
*Summary of what follows.* @tab:findings states the four conclusions
before they are argued, so that a reader can check each against the leaderboard directly.
]

#figure(
  align(center)[#table(
    columns: 2,
    align: (left,left,),
    table.header([#strong[Finding];], [#strong[Evidence];],),
    table.hline(),
    [A well-tuned supervised transformer is the model to beat], [iTransformer 0.552, ahead of every foundation model, retrieval scheme and multimodal system],
    [Zero-shot foundation models underdeliver], [0.187, 0.271, 0.287 and $- 0.081$ across three architecture families; three of four below
    climatology],
    [Retrieval beats fine-tuning on a frozen backbone], [0.478 against 0.331 for the same backbone, with no weight update],
    [Genuine exogenous imagery has not paid off], [Best multimodal model (0.540) uses rendered images; every satellite-consuming model ranks
    eighth or below],
  )]
  , caption: [The four findings and the evidence for each, stated in advance of the argument.]
  , kind: table
  )
<tab:findings>
== Reading by tier
<sec:by-tier>
#[
#set par(first-line-indent: 0pt)
*Tier 0.* The reference tier spans a wide range, which is itself informative.
Hourly climatology reaches 0.234 --- a model with no knowledge of the current day beats
naive persistence (0.014) by a wide margin, and beats three of the four zero-shot foundation
models. The gap between persistence and smart persistence is instructive in the other
direction: they score 0.014 and 0.000 on skill respectively, yet smart persistence is the
harder reference in every regime that matters, because the skill score is defined relative
to it and persistence's apparent edge comes from the trivially predictable diurnal
component.
]

#[
#set par(first-line-indent: 0pt)
*Tiers 1 and 2.* Supervised models dominate the upper half of the table.
iTransformer leads outright; PatchTST is seventh; TFT, MLP and LightGBM occupy ninth
through eleventh. DLinear at 0.325 is the informative low end: a single linear layer on a
trend/seasonal decomposition, consuming no covariates, essentially matches fine-tuned
Chronos-2 (0.331). Whatever the pretrained model contributes on this task, it is worth
approximately as much as a linear layer with the right decomposition.
]

#[
#set par(first-line-indent: 0pt)
*Tier 3.* The foundation-model tier is bimodal. Zero-shot rows occupy positions
18, 19, 23 and 28; adapted rows occupy 13 and 16; oracle rows sit at 3 and 6. The spread
within a single backbone --- Chronos-2 ranges from 0.187 zero-shot to 0.504 with oracle
covariates --- is larger than the spread across the entire supervised tier. On this task,
what a foundation model is #emph[conditioned on] matters more than which foundation model it
is.
]

#[
#set par(first-line-indent: 0pt)
*Tier 4.* The two retrieval methods are nearly tied at 0.478 and 0.477 and are the
best non-oracle results built on a foundation model. CoRA trails at 0.374. The ordering
within the tier is the interesting part and is developed in @sec:finding3.
]

#[
#set par(first-line-indent: 0pt)
*Tiers 5 and 6.* The multimodal tiers straddle the whole table. Time-VLM is second
overall; every other multimodal model is eighth or below, and four of the seven fall in the
bottom third. The split does not follow tier boundaries --- it follows the
endogenous/exogenous boundary, which is the subject of @sec:finding4.
]

== Tier-level summary
<sec:tier-summary>
@tab:tier-summary aggregates the leaderboard by tier. With four to eight models per
tier these are small samples and the spread matters more than the mean, but the pattern is
stable enough to be worth stating.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,center,center,center,center,),
    table.header([#strong[Tier];], [#strong[Models];], [#strong[Best SS];], [#strong[Median SS];], [#strong[Worst SS];],),
    table.hline(),
    [T0 Reference], [4], [0.234], [0.061], [0.000],
    [T1 Classical ML], [2], [0.384], [0.362], [0.339],
    [T2 Supervised DL], [5], [#strong[0.552];], [0.423], [0.325],
    [T3 Foundation (honest)], [6], [0.364], [0.301], [$- 0.081$],
    [T3 Foundation (oracle)], [2], [0.504], [0.489], [0.474],
    [T4 Frozen adaptation], [3], [0.478], [0.477], [0.374],
    [T5 Endogenous multimodal], [4], [0.540], [0.177], [0.017],
    [T6 Exogenous multimodal], [3], [0.443], [0.349], [0.216],
  )]
  , caption: [Leaderboard aggregated by tier. Median is over the models within the tier. Oracle
  rows are separated from the honest foundation-model rows because they are upper bounds.]
  , kind: table
  )
<tab:tier-summary>
Three readings. The supervised tier has both the best model and a high median, so its strength
is not one lucky architecture. The honest foundation-model tier has the worst median of any
learned tier and by far the widest spread, which is the bimodality of
@sec:by-tier showing up as a statistic. And the two multimodal tiers differ in a
revealing way: the endogenous tier has the higher ceiling (0.540 against 0.443) and the lower
median (0.177 against 0.349), meaning it contains both the best multimodal model and most of
the worst ones. Rendering a series as an image is a high-variance strategy; consuming
satellite frames is a consistently mediocre one.

== Finding 1: a well-tuned supervised transformer is the model to beat
<sec:finding1>
iTransformer tops every metric --- skill 0.552, $R^2$ 0.765, best NRMSE --- ahead of every
foundation model, every retrieval scheme and every multimodal system in the suite. The lead
is not marginal: 0.074 skill clear of the best non-oracle foundation-model result and 0.221
clear of honest Chronos-2 fine-tuning.

This inverts the usual framing and deserves to be stated without softening. On this task,
architecture plus in-domain supervised training on 69 plants beats large-scale pretraining.
The comparison is fair in the strongest sense: the supervised models also see only training
plants, are also evaluated on disjoint test plants, and receive the same windows and the same
covariates. They have no pretraining and no privileged information.

Why iTransformer specifically? Its inductive bias matches the structure of the problem
unusually well. It applies attention across #emph[variates] rather than across time, treating
each channel as a token, and @fig:correlations shows why that pays: normalised
power correlates at 0.74 with shortwave radiation, 0.69 with direct radiation and 0.60 with
direct normal irradiance, while the covariates are heavily correlated with each other
(0.93 between shortwave and direct radiation). The forecasting problem is substantially a
problem of resolving a small set of strongly inter-correlated exogenous drivers, and
variate-attention models exactly that. PatchTST, which is channel-independent, ranks seventh;
the difference between the two is a clean demonstration that on this task the cross-channel
structure is where the signal is.

The finding does not show that pretraining is worthless. It shows that on a task with 69
training plants and a strongly structured covariate track, the value of pretraining is
smaller than the value of a well-matched inductive bias --- and that a paper reporting only
foundation-model comparisons would have reached the opposite conclusion by omission.

== Finding 2: zero-shot foundation models underdeliver
<sec:finding2>
The zero-shot tier is the weakest part of the leaderboard. Chronos-2 scores 0.187, TimesFM
0.271, TiRex 0.287, and TTM-R2 reaches $- 0.081$ --- worse than assuming cloud conditions
persist. All four trail LightGBM; three of the four trail hourly climatology, a model that
has never seen the current day.

Because the result replicates across an encoder transformer, a decoder transformer and a
mixer architecture, it is a property of the task rather than of any one pretraining corpus.
The natural explanation is distributional. Photovoltaic power is bounded below by zero and
above by capacity, is gated by a deterministic astronomical cycle, is exactly zero for a
large and predictable fraction of every day, and transitions abruptly when a cloud edge
crosses the array (@fig:ramps). General-purpose pretraining corpora are dominated
by smooth economic, demand and sensor series with none of those properties. A model whose
prior is "series are smooth and roughly stationary in the short run" is systematically
wrong here in a way that no amount of context length repairs.

Two secondary observations support the reading. First, the failure is worst for the smallest
model: TTM-R2 zero-shot is the only entry below the reference floor, and fine-tuning the same
architecture recovers it to 0.364 --- a swing of 0.445 skill from adaptation alone. The
zero-shot deficit is therefore not a capacity ceiling but a prior mismatch. Second, the
ordering among zero-shot models does not track their general-purpose leaderboard ordering,
which is what one expects if the limiting factor is domain mismatch rather than model
quality.

Adaptation is not optional on this task. That conclusion is what makes Tier~4 the interesting
tier rather than a completeness exercise.

== Finding 3: retrieval beats fine-tuning on a frozen backbone
<sec:finding3>
Every adaptation step helps --- Chronos-2 moves from 0.187 to 0.331, TTM-R2 from $- 0.081$ to
0.364 --- but the ordering #emph[among] adaptation strategies is the result worth carrying
forward. Retrieval over a frozen backbone reaches 0.478 (TS-RAG) and 0.477 (Cross-RAG)
against 0.331 for fine-tuning the same backbone: 0.147 skill, obtained without a single
gradient update to the model.

The covariate adapter sits between them at 0.374, which sharpens the interpretation. What
retrieval supplies is not simply "more input" --- CoRA also supplies more input, in the form
of the full covariate track, and gains less. What retrieval supplies is #emph[relevant
precedent];: windows whose dynamics resemble the current one. For a problem whose central
difficulty is regime identification --- is this a clear day, a broken-cloud day, or a front
passing? --- an example of how a similar situation evolved is worth more than either a
re-fitted weight matrix or an additional covariate channel.

There is a caveat that keeps this from being a stronger claim than it is. The two retrieval
models have low $R^2$ (0.331 and 0.375) relative to their skill, and @sec:leaderboard
noted the same pattern in their metric disagreement. They are excellent on typical steps and
comparatively weak at tracking the day's shape, which suggests they are exploiting the
datastore for level calibration at least as much as for dynamics. Whether the gain would
survive a harder ramp-weighted evaluation is visible in @app:ramp, where both
sit mid-table on ramp NRMSE despite their aggregate rank.

== Finding 4: genuine exogenous imagery has not paid off
<sec:finding4>
The best multimodal model in the suite is Time-VLM at 0.540 --- second overall,
statistically level with iTransformer, and with the lowest ramp NRMSE of the entire suite
(0.172, @app:ramp), which is precisely where a visual model is supposed to win.
And Time-VLM consumes #emph[no external imagery at all];: it renders the numerical series as
an image and passes it to pretrained vision-language backbones.

Every model that ingests real satellite frames ranks eighth or below: Solar-VLM 0.443,
CrossViVit 0.349, SUNSET 0.216, UniCast 0.121. The domain state of the art, with genuine
satellite input and a vision-language backbone, trails a model that only ever saw a plot of
its own input by 0.097 skill.

Three explanations are available, and they have very different consequences for the design of
a multimodal forecaster.

+ #strong[Redundancy.] The information in a satellite crop is already present in the
  covariate track. Cloud cover, shortwave radiation and direct normal irradiance are
  supplied numerically to every Tier-2 and Tier-4 model, so the frames may add nothing that
  the reanalysis has not already summarised.

+ #strong[Interface loss.] The information is present in the frames but does not survive
  the encoder-to-forecaster interface. A visual encoder trained on natural video produces
  representations optimised for object and motion semantics, not for the quantity a
  forecaster needs, and the learned adapter may be too weak a bridge.

+ #strong[Inductive bias, not information.] What converts into skill in Time-VLM is the
  #emph[prior structure] of a visual backbone applied to data the model already had --- and
  that resource is available without any external sensor.

@fig:cloud-residual is evidence against the first explanation, and it is worth
dwelling on. It bins daytime observations by shortwave radiation and, within each bin, by
cloud cover. If the reanalysis irradiance already captured everything, the three cloud-cover
boxes within an irradiance bin would coincide. They do not: in the 250--450 W/m$""^2$ bin the
median normalised power falls from 0.33 under clear skies to 0.27 under mixed conditions to
0.24 under overcast, a monotone residual effect that persists across every irradiance bin.
There is genuine cloud-state information beyond what the numerical irradiance channel carries.
That is exactly the information a satellite frame observes directly, and exactly the
information the exogenous models are failing to convert.

The remaining explanations are therefore the second and third, and they are distinguishable
only by controlled experiment --- specifically by the shuffled-frames and modality-grid
controls specified in @cha:expdesign. This is the central open question the
benchmark leaves behind, and it reframes the thesis: the interesting question is no longer
"does deep fusion beat late fusion" but "does any current visual interface transmit what
the frames contain".

== Qualitative behaviour
<sec:qualitative>
Aggregate metrics hide failure modes that a plot exposes immediately. @fig:traces
shows one forecast window on a representative test plant; @fig:scatter contrasts
four models' calibration pooled over all of that plant's windows.

#figure(image("figures/traces_comparison.png", width: 92.0%),
  caption: [
    One forecast window on test site 10793: the best model of each cluster against the
    truth. Time-VLM is the only model that anticipates elevated output at the start of the
    horizon; the others decay smoothly from the forecast origin in the manner of persistence,
    and all of them miss the sharp drop at step~2. Legend metrics are window-local and are not
    the pooled numbers of @tab:leaderboard.
  ]
)
<fig:traces>

The trace exhibits the characteristic failure of the whole suite. The truth rises above the
forecast origin, holds, then falls by three quarters within two steps. Every model except
Time-VLM produces a monotone decay from the origin --- the shape a model outputs when it has
no information about what is about to happen and falls back on the conditional mean. Even
Time-VLM, which correctly holds output high for two steps, misses the drop entirely and ends
the horizon far above the truth. The models are not disagreeing about the ramp; they are
collectively blind to it.

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 6pt,
    row-gutter: 6pt,
      image("figures/scatter_itransformer.png"), image("figures/scatter_time_vlm.png"),
      image("figures/scatter_solar_vlm.png"), image("figures/scatter_visionts_pp.png"),
  ),
  caption: [
    Actual against predicted power on test site 10793, pooled over all windows. Top:
    iTransformer and Time-VLM, the two leaders, both tightly distributed about the diagonal.
    Bottom: Solar-VLM, which tracks shape but sits below the diagonal at high output, and
    VisionTS++, whose cloud is over-dispersed and biased upward at low power. Per-panel metrics
    are site-local and are #emph[not] the pooled fourteen-plant numbers of
    @tab:leaderboard.
  ]
)
<fig:scatter>

Two distinct pathologies appear in the scatter plots, and neither is visible in a skill
column. Several exogenous multimodal models --- Solar-VLM, SUNSET, CrossViVit ---
#emph[systematically under-predict];: they track the shape of the day well ($R^2$ between 0.36
and 0.66) yet sit below the diagonal at high output. VisionTS++ fails differently, with an
over-dispersed cloud biased upward at low power: respectable variance tracking ($R^2$ 0.397)
and almost no skill (0.017), because it is wrong about magnitude in both directions rather
than consistently in one. UniCast is the cleanest instance of the first pathology, with
predictions of approximately $0.61 thin y + 0.063$ at correlation 0.61, compounded by a late-day
skew in its native evaluation windows.

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 6pt,
    row-gutter: 6pt,
      image("figures/scatter_chronos2_zs.png"), image("figures/scatter_ts_rag_orig.png"),
  ),
  caption: [
    The effect of retrieval on a frozen backbone, same test site. Left: Chronos-2
    zero-shot. Right: TS-RAG over the same frozen backbone. Retrieval tightens the distribution
    substantially without any weight update.
  ]
)
<fig:scatter-rag>

Both pathologies are #emph[bias and scale] failures rather than timing failures, and both are
in principle recoverable by recalibration. Recalibration was not attempted, for two reasons.
It would have to be fitted on validation plants only, to avoid the leakage the protocol
forbids; and more fundamentally it would change what the tier measures, converting a
comparison of forecasters into a comparison of forecasters-plus-post-processing. The honest
reading is that several published multimodal architectures transfer their #emph[shape]
knowledge across plants but not their #emph[scale] knowledge --- which is itself a finding
about cross-plant generalisation, and one that capacity normalisation was supposed to prevent.

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 6pt,
    row-gutter: 6pt,
      image("figures/scatter_lightgbm.png"), image("figures/scatter_dlinear.png"),
      image("figures/scatter_crossvivit.png"), image("figures/scatter_unicast.png"),
  ),
  caption: [
    Calibration across tiers, same test site. Top: LightGBM and DLinear, the classical
    and linear references --- both well centred, DLinear visibly more dispersed. Bottom:
    CrossViVit and UniCast, two exogenous multimodal models, both showing the compression toward
    the mean that produces high $R^2$ with low skill.
  ]
)
<fig:scatter-tiers>

#figure(image("figures/traces_multimodal.png", width: 92.0%),
  caption: [
    The multimodal tier alone, same window as @fig:traces. The seven models
    disagree by more than the whole range of the truth. Time-VLM and Solar-VLM hold output high;
    SUNSET collapses immediately to 0.10 and stays there; VisionTS++ and CrossViVit decay to near
    zero. All of them consume --- or in Time-VLM's case simulate --- a visual modality, and no two
    of them read this window the same way.
  ]
)
<fig:traces-mm>

@fig:traces-mm is worth more attention than its aggregate row. The spread among
multimodal models on a single window exceeds the spread of the entire leaderboard: at the
final horizon step, predictions range from 0.00 (VisionTS++) to 0.26 (Time-VLM) against a
truth of 0.11. Two models overshoot by more than a factor of two, two undershoot to nothing,
and one --- SUNSET --- ignores the horizon entirely and outputs a near-constant. A visual
modality is evidently not a stabilising influence; it is a source of variance that each
architecture converts differently, which is consistent with the interface-loss explanation of
@sec:finding4.

== The ramp regime
<sec:ramp-results>
@app:ramp reports every model on the top-decile $lr(|Delta y|)$ subset. Three
observations follow, and they matter more for the thesis than the aggregate table does,
because the ramp regime is where the multimodal hypothesis was supposed to pay.

First, the ordering is broadly preserved but compressed. Time-VLM leads at 0.172 ramp NRMSE,
iTransformer follows at 0.177, and the reference floor sits at 0.281 --- a range of 0.109
against a range of 0.147 in the aggregate NRMSE column. Ramps are hard for everyone, and the
spread between a good model and a bad one narrows exactly where the value of forecasting is
highest.

Second, the retrieval models lose their advantage. TS-RAG and Cross-RAG rank fourth and fifth
overall but seventh and eighth on ramp NRMSE, behind Solar-VLM and PatchTST. This supports the
reading of @sec:finding3: retrieval supplies level calibration and typical-case
accuracy more than it supplies transition dynamics. Retrieving a window that looks like the
current one is most useful when the current one is unremarkable.

Third --- and this is the observation that shapes @cha:expdesign --- the model
with the best ramp accuracy in the entire suite consumes no external imagery. If satellite
frames were being exploited for advection, the exogenous tier should own this column. It does
not: Solar-VLM is fourth, CrossViVit sixteenth, SUNSET nineteenth. Whatever the cloud-state
information demonstrated in @fig:cloud-residual is worth, no current architecture
in this suite is extracting it in the regime where it should matter most.

== Practical considerations
<sec:practical>
Accuracy is not the only axis on which these models differ, and a benchmark that ignores cost
gives misleading guidance to anyone who has to deploy something.

#[
#set par(first-line-indent: 0pt)
*Cost of adaptation.* The three adaptation strategies differ by orders of magnitude
in what they require. Retrieval needs an index over training-plant windows and a
nearest-neighbour lookup at inference; no gradients, no accelerator, and a new plant can be
served the moment its history arrives. Adapter tuning needs a short training run over a small
parameter set. Full fine-tuning needs a training run over the whole backbone and produces a
task-specific copy of a large model that must then be stored and served. That the cheapest of
the three is also the most accurate on this task (@sec:finding3) is the most
practically consequential finding in the chapter.
]

#[
#set par(first-line-indent: 0pt)
*Cost of the visual branch.* Every exogenous multimodal model in the suite requires
a satellite ingest pipeline: reprojection, per-site cropping, storage of terabytes of frames,
and a vision encoder forward pass per forecast. In this study that pipeline is the single
largest engineering component of the dataset (@sec:sources) and the vision
encoder dominates inference cost. The endogenous alternative --- rendering the series as an
image --- requires none of it. Against that background, Time-VLM outranking every
satellite-consuming model is not merely a scientific result but an economic one.
]

#[
#set par(first-line-indent: 0pt)
*Serving a growing fleet.* The deployment scenario that motivated this thesis ---
new plants arriving continuously --- favours methods whose marginal cost per plant is near
zero. Supervised in-domain models, including the leader, must be retrained as the fleet
grows if they are to exploit new plants; zero-shot and retrieval methods absorb a new plant
by adding rows to an index. The leaderboard ordering and the operational ordering are
therefore not the same, and a practitioner reading only @tab:leaderboard would
draw the wrong conclusion.
]

== What the benchmark cannot answer
<sec:benchmark-limits>
Four questions are outside what these numbers support, and stating them prevents
over-reading.

#[
#set par(first-line-indent: 0pt)
*Whether the results transfer to another fleet.* Everything here is
`uk_pv`: one country, one satellite product, one capacity class, two years. The
`goes_pvdaq` track exists in the dataset and differs in sensor, scale, latitude and
cadence, but was not run. Cross-dataset generalisation is unproven --- not disproven.
]

#[
#set par(first-line-indent: 0pt)
*Whether the ordering is statistically significant.* Each model was run once, and
no significance testing accompanies the table. Differences of a few thousandths of skill ---
TS-RAG at 0.478 against Cross-RAG at 0.477, for instance --- are ties, and are read as such
throughout.
]

#[
#set par(first-line-indent: 0pt)
*Whether the models were equally well tuned.* Hyperparameter budgets were
comparable but not identical, and vendored implementations carry their authors' defaults.
A model can lose a benchmark because its architecture is unsuited or because nobody tuned it,
and this study cannot fully separate the two. The tier structure mitigates rather than
eliminates the concern: within Tier~2, four models share one trainer, so differences there are
architectural.
]

#[
#set par(first-line-indent: 0pt)
*Whether the models would rank the same at another horizon.* Every number here is a
six-hour forecast. At one hour persistence-like methods are far stronger and the spread
compresses; at twenty-four hours the observed cloud field is irrelevant and numerical weather
dominates, which would penalise every imagery-consuming model and favour those that ingest
the covariate track well. The skill-decay scenario that would settle this was only partially
run, so the ordering reported should be read as an ordering #emph[at six hours];.
]

#[
#set par(first-line-indent: 0pt)
*Whether better conditioning would change the ordering.* The oracle rows show that
the same backbone gains 0.173 skill from privileged future weather. It is entirely possible
that the ranking of architectures would reorder under better --- but still legitimate ---
conditioning, and nothing here rules that out.
]

== What the benchmark establishes
<sec:bar>
Three numbers define the target for any proposed architecture on this task.

- #strong[0.552] --- the overall bar (iTransformer). A model that does not approach it
  has not beaten supervised in-domain training.

- #strong[0.540] --- the multimodal bar (Time-VLM), achieved with no external imagery.

- #strong[0.331] --- the honest same-backbone reference (Chronos-2 fine-tuned). Any
  architecture built on Chronos-2 must clear this to show that its additions, rather than
  its backbone, are doing the work.

A fourth number is not a bar but a measurement of available headroom. The oracle variants,
which see observed future weather, reach 0.474 and 0.504 against 0.331 for the honest
fine-tuned model --- a gap of #strong[0.173 skill] attributable purely to better conditioning
over the horizon. That is larger than the difference between any two architectures in the
entire suite. It is the single strongest indication in this work of where effort should be
directed, and @cha:conclusion returns to it.


= MMTSFM: Deep Multimodal Fusion on Frozen Backbones
<cha:method>
This chapter presents the proposed architecture. MMTSFM couples a largely fixed time-series
foundation model with a frozen video foundation model through two mechanisms that, to the
best of this work's knowledge, have not previously been combined: #emph[selective temporal
interleaving];, which places visual tokens inside the time-series sequence but only within a
short refinement window, and #emph[causal Grassmann mixing];, an $O (L)$ temporal operator
that encodes the transition between hidden states as a subspace on a Grassmann manifold.

The chapter describes the system as built and measured. A subsequent redesign exists on
paper and is discussed as future work in @sec:future; nothing in this chapter
depends on it.

== Overview
<sec:overview>
The model consumes, per entity row, a 672-step normalised power history, fourteen covariate
channels over history and horizon, and eight satellite frames covering a short recent
window. It emits the full six-hour horizon as nine quantiles in a single forward pass.

The data flow is:

+ The power history is patch-tokenised by the #emph[pretrained] Chronos-2 patch
  embedding into $T_(upright("ctx")) = ceil.l 672 \/ 16 ceil.r = 42$ context tokens, plus
  $T_(upright("fut")) = ceil.l 12 \/ 16 ceil.r = 1$ future token. Each covariate channel becomes
  its own token row on the batch axis.

+ The eight frames are encoded as a #emph[clip] by frozen V-JEPA~2.1 into
  $[B \, 4 \, 196 \, 1024]$ --- four latent time steps, 196 spatial patches --- and compressed
  by a learned latent summariser into $n_(upright("vis"))$ tokens of width 768
  ($n_(upright("vis")) = 1$ for `uk_pv`, 2 for `goes_pvdaq`).

+ A multimodal embedding layer adds modality, segment, token-type, entity and rotary
  positional information.

+ The fusion mode determines where visual tokens enter: as additional batch rows
  (late fusion) or woven into the temporal sequence (interleaved).

+ Twelve Chronos-2 encoder blocks process the result, each applying temporal mixing,
  group self-attention across the batch axis, and a feed-forward layer.

+ The output head decodes the final future positions into nine quantiles per horizon
  step and inverts the normalisation.

@fig:architecture shows the assembled system.

#figure(image("figures/mmtsfm_architecture.svg"),
  caption: [
    MMTSFM as assembled, for `uk_pv`. Cool boxes are pretrained or frozen;
    warm boxes are trained from scratch --- the latent summariser, the fusion interface, the
    Grassmann mixer and the quantile head are the only newly initialised components. Tensor
    shapes are annotated on the edges.
  ]
)
<fig:architecture>

=== Shapes, worked through
@tab:shapes traces one `uk_pv` sample from input to output. Tracing shapes
explicitly is worth a page here because two of the design's least obvious properties --- the
aggressive patch compression of the numerical stream, and the extreme compression of the
visual stream --- are invisible in prose and immediate in a shape table.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Stage];], [#strong[Shape];], [#strong[Note];],),
    table.hline(),
    [Power history], [$[B \, 672]$], [14 days at 30-min cadence],
    [After patch embedding], [$[B \, 42 \, 768]$], [$ceil.l 672 \/ 16 ceil.r$ tokens; one token
    $approx$ 8 h],
    [Future positions], [$[B \, 1 \, 768]$], [$ceil.l 12 \/ 16 ceil.r$],
    [Covariates], [$[B \, 684 \, 14]$], [history $+$ horizon],
    [Covariate token rows], [$14 times [B \, 1 \, 768]$], [one batch row per channel],
    [Frames], [$[B \, 3 \, 8 \, 224 \, 224]$], [8 frames over a 6-hour window],
    [After V-JEPA], [$[B \, 4 \, 196 \, 1024]$], [tubelet stride 2; 4 latent steps],
    [After latent summariser], [$[B \, 1 \, 768]$], [$n_(upright("vis")) = 1$],
    [Interleaved sequence], [$[B \, 44 \, 768]$], [$42 + n_(upright("vis")) + T_(upright("fut"))$],
    [After 12 encoder blocks], [$[B \, 44 \, 768]$], [shape preserved],
    [Output head], [$[B \, 12 \, 9]$], [horizon $times$ quantiles],
  )]
  , caption: [One `uk_pv` sample end to end. Two compressions dominate: 672 timesteps
  become 42 tokens, and $4 times 196 times 1024$ visual activations become a single
  768-dimensional vector.]
  , kind: table
  )
<tab:shapes>
The second compression is worth dwelling on, because it is a candidate explanation for any
weak visual contribution. The V-JEPA output contains $4 times 196 = 784$ patch embeddings of
width 1024; the summariser reduces this to one vector of width 768, a reduction of roughly
three orders of magnitude. Whatever spatial structure the cloud field has --- a front
approaching from the west, a break opening to the south --- must survive that bottleneck to
influence the forecast. The visual token budget sweep specified in
@sec:sensitivity exists precisely to test whether it does.

== Design principle: decoupled resolution
<sec:decoupled>
The two modalities are kept at #emph[different temporal resolutions on purpose];. The
numerical stream carries a long macro-context --- two weeks, spanning many diurnal cycles,
enough to expose seasonality and regime structure. The visual stream carries a short
micro-context at high cadence, covering only the last few hours.

Both alternatives are worse. Resampling a fortnight of satellite frames onto the numerical
grid is computationally prohibitive and mostly redundant, since frames more than a few hours
old carry no information about the coming six hours: cloud fields advect out of the crop.
Resampling the numerical series up to the frame cadence discards the macro-context that
makes the foundation-model backbone useful in the first place. Decoupling keeps each modality
at the resolution at which it is informative, and the cost is that fusion must handle
sequences in which the two streams have incommensurable time bases --- which is exactly what
@sec:interleaving addresses.

== Numerical branch
<sec:numeric-branch>
The backbone is Chronos-2~@chronos2 at its native checkpoint size: $d_(upright("model"))$
768, twelve layers, twelve heads with head dimension 64, feed-forward width 3072, input
patch size 16, nine protocol quantiles, and an inverse-hyperbolic-sine normalisation that
makes series of arbitrary magnitude comparable without per-series statistics.

Because the configured input patch size matches the checkpoint, the patch embedding loads
#emph[pretrained] and transfers; only the nine-quantile output head is reinitialised. This is
a deliberate design constraint rather than a happy accident --- a mismatched patch size would
discard the tokeniser and with it most of the value of the pretrained backbone.

#[
#set par(first-line-indent: 0pt)
*A reproducibility trap worth recording.* The configuration files in this project
nominally specify a smaller model ($d_(upright("model"))$ 512, six layers, feed-forward 2048).
These values are #emph[silently ignored];: loading from a pretrained checkpoint preserves the
checkpoint's architecture, and only the Grassmann-specific fields and the tokeniser
configuration propagate. Any report of this system that quotes the configured dimensions
rather than the checkpoint's is wrong. The failure is silent --- no warning, no shape error
--- and it is the kind of discrepancy that survives a long time in a research codebase.
]

#[
#set par(first-line-indent: 0pt)
*Covariates.* The fourteen covariate channels enter as separate token rows on the
batch axis, sharing a group identifier with the target row, with their masks set so that
values survive to the encoder. That they genuinely reach the model in both fusion modes was
verified by perturbation: changing a covariate changes the forecast. This is worth
stating because a covariate path that is silently masked out is a common and invisible
failure, and because it is what makes the honest-versus-oracle distinction of
@sec:parity meaningful.
]

== Visual branch
<sec:visual-branch>
The visual encoder is a frozen V-JEPA~2.1 ViT-L/16~@vjepa2, applied to the entire
eight-frame clip rather than frame by frame. Tubelet embedding with temporal stride two
yields four latent time steps of 196 spatial patches each, at width 1024. Motion is therefore
encoded #emph[inside] the backbone, and the downstream module only has to perform spatial
compression.

That module is a learned #strong[latent summariser];: a Perceiver-style cross-attention
block~@perceiver with a small set of learned queries, applied causally across latent
time, and equipped with a #emph[null token] so that the summariser can represent "no useful
visual evidence" rather than being forced to emit a confident summary of an uninformative
sky. It maps $[B \, 4 \, 196 \, 1024]$ to $[B \, n_(upright("vis")) \, 768]$.

All domain specialisation lives in this summariser. There is no per-sensor input projection:
the data loader delivers uniformly shaped frames for every plant and source, so the queries
--- and only the queries --- learn what to extract from a satellite crop for the purpose of
forecasting power. This mirrors the adapter-recycling strategy used to specialise frozen
vision-language backbones, and it keeps the number of newly initialised parameters small,
which matters for the curriculum of @sec:curriculum.

== Multimodal embedding
<sec:embedding>
Before fusion, every token receives an additive composition of five embeddings: a
#emph[modality] embedding (numeric or visual), a #emph[segment] embedding (context or
future), a #emph[token-type] embedding (target, covariate or visual), an #emph[entity]
embedding, and rotary positional encoding~@rope over the temporal axis. Additive
composition keeps the embedding table small and lets the encoder disentangle the factors
linearly.

== Fusion
<sec:interleaving>
Two fusion modes are implemented, and the difference between them is the architectural
hypothesis under test.

#[
#set par(first-line-indent: 0pt)
*Late fusion.* The visual summary is passed through a cross-modal adapter and
appended as $N_(upright("soft")) = 1$ additional #emph[batch rows] sharing the target's group
identifier. The two modalities therefore meet only in the group-attention operator, once per
block, and never interact along the time axis. This is the standard design and serves as the
control.
]

#[
#set par(first-line-indent: 0pt)
*Selective temporal interleaving.* Let $T_M = T_(upright("ctx")) - n_(upright("vis"))$. The
encoder input sequence is
$ upright(bold(S)) = underbrace([thin upright(t s)_0 \, dots.h \, upright(t s)_(T_M - 1) thin], upright("macro: pure time series")) #h(0em) bar.v.double #h(0em) underbrace([thin upright(t s)_(T_M) \, v_(T_M) \, dots.h \, upright(t s)_(T_(upright("ctx")) - 1) \, v_(T_(upright("ctx")) - 1) thin], upright("refinement: interleaved")) #h(0em) bar.v.double #h(0em) underbrace([thin upright(f u t)_0 \, dots.h thin], upright("future")) . $
Visual tokens are inserted #emph[only] inside the refinement window. The macro region stays
purely numerical, so its temporal geometry --- the structure the pretrained backbone knows
how to exploit --- is left undisturbed; in the refinement region the two modalities alternate,
so that the temporal operator computes genuinely cross-modal pairs.
]

The causal access pattern follows from the ordering. A time-series token at position
$T_M + 2 k$ attends to the full macro history and all previous refinement pairs, but not to the
visual token that follows it; the visual token at $T_M + 2 k + 1$ attends to everything above plus
its own time-series partner. Causality is preserved by construction and requires no masking
special case.

#[
#set par(first-line-indent: 0pt)
*Cost.* Total sequence length is $T_(upright("ctx")) + n_(upright("vis"))$: the overhead is
$n_(upright("vis")) \/ T_(upright("ctx"))$, proportional to the number of visual tokens rather than to
context length. For `uk_pv` this is $42 + 1 + 1 = 44$ tokens, roughly two percent. Longer
numerical contexts make the ratio smaller, not larger --- which is the opposite of the scaling
behaviour of full cross-attention between two long sequences, and is the practical argument
for the design.
]

== Temporal mixing: causal Grassmann flow
<sec:grassmann>
Each encoder block begins with a temporal mixing operator. Two are implemented and are
selected by a single configuration switch, with everything else --- including the training
schedule --- held identical, so that any measured difference is attributable to the operator
alone.

=== The operator
Ordinary self-attention scores a pair of positions by the dot product of their
representations: a measure of #emph[similarity];. Grassmann mixing instead encodes the
#emph[transition] between two positions as an oriented plane. For each position $i$ and each
offset $delta$ in a fixed set:

+ #strong[Reduction.] $upright(bold(z))_i = W_(upright("red")) upright(bold(h))_i in bb(R)^r$
  with $r$ even (default 32), so that rotary embedding applies.

+ #strong[Phase injection.] Rotary positional encoding is applied to $upright(bold(z))$.

+ #strong[Plücker encoding.] The wedge product of the two reduced states, normalised,
  $ upright(bold(p))_(i \, delta) = frac(upright(bold(z))_(i - delta) and upright(bold(z))_i, lr(bar.v.double upright(bold(z))_(i - delta) and upright(bold(z))_i bar.v.double) + epsilon) in G (2 \, r) \, #h(2em) dim (upright(bold(p))) = binom(r, 2) \, $
  is a point on the Grassmann manifold of two-dimensional subspaces: the plane spanned by
  the two states, independent of their magnitudes.

+ #strong[Projection.] $upright(bold(g))_(i \, delta) = W_(upright("plu")) thin upright(bold(p))_(i \, delta) in bb(R)^d$.

+ #strong[Multi-scale aggregation.] A softmax over the offsets
  $delta in { 1 \, 2 \, 4 \, 8 \, 12 \, 16 }$ mixes the projections.

+ #strong[Gated fusion.] $upright(bold(h))'_i = alpha_i ⊙ upright(bold(h))_i + (1 - alpha_i) ⊙ upright(bold(g))_i$ with
  $alpha_i = sigma #h(-1em) (W_(upright("gate")) [upright(bold(h))_i bar.v.double upright(bold(g))_i])$,
  so the operator can learn to pass the residual stream through unchanged.

=== Why this inductive bias
Three arguments motivate the construction. #emph[Geometry over magnitude];: the Plücker
normalisation makes the representation scale-invariant, so what is encoded is the
#emph[direction] of state evolution. In a physical system the direction of change ---
"rapidly clouding" as opposed to "stable" --- is frequently more predictive than the
level, and photovoltaic power is such a system. #emph[Efficiency];: the operator is $O (L)$ in
sequence length, since each position pairs with a fixed number of offsets rather than with
all predecessors, which admits contexts of a thousand or more steps within a single device's
memory where quadratic attention does not. #emph[Multi-scale by construction];: the offset set
spans half an hour to eight hours at the `uk_pv` cadence, so several temporal
frequencies are tracked simultaneously without stacking layers.

=== Modality-aware offset gating
Interleaving interacts with the offset structure in a way that requires a correction. Inside
the refinement window, with the pattern
$[upright(t s)_k \, v_k \, upright(t s)_(k + 1) \, v_(k + 1) \, dots.h]$, the modality composition of a
pair depends on the parity of $delta$: offset~1 produces genuinely cross-modal pairs
(time series with visual), while offsets 2, 4 and beyond produce same-modality pairs
(time series with time series, or visual with visual), and large offsets reach back across
the boundary into the pure-macro region and mix interleaved with non-interleaved cadences.

@tab:offset-parity works this out explicitly for the interleaved pattern.

#figure(
  align(center)[#table(
    columns: 4,
    align: (center,left,left,left,),
    table.header([$delta$], [Pair at a time-series query (position $T_M + 2 k$)], [Pair at a visual query
      (position $T_M + 2 k + 1$)], [Class],),
    table.hline(),
    [1], [$(v_(T_M + k - 1) \, thin upright(t s)_(T_M + k))$], [$(upright(t s)_(T_M + k) \, thin v_(T_M + k))$], [cross-modal],
    [2], [$(upright(t s)_(T_M + k - 1) \, thin upright(t s)_(T_M + k))$], [$(v_(T_M + k - 1) \, thin v_(T_M + k))$], [unimodal],
    [4], [time series with time series, stride 2], [visual with visual, stride 2], [unimodal],
    [8, 12, 16], [reaches back into the pure-macro region], [likewise], [unimodal / boundary],
  )]
  , caption: [Modality composition of Grassmann pairs inside the refinement window. Offset~1 is
  the #emph[only] offset producing genuinely cross-modal subspaces; even offsets produce
  same-modality pairs, and large offsets cross the boundary between interleaved and pure
  cadence.]
  , kind: table
  )
<tab:offset-parity>
These pair classes live on different statistical manifolds and should not share a single
projection unmodified. The mitigation is a set of four learned scalar biases, one per
modality-pair class --- time/time, time/visual, visual/time, visual/visual --- added to the
offset logit before the aggregation softmax:
$ ell_(i \, delta) = ⟨ upright(bold(q))_i \, upright(bold(k))_delta ⟩ + b_(m_(delta \, i)) . $
The model can thereby down-weight offsets that produce semantically incoherent pairs inside
the refinement window without disturbing behaviour in the macro region. The cost is four
parameters and no change in asymptotic complexity. Whether the correction is necessary is an
ablation in its own right (@sec:ablation-design).

=== The diagnostic twin
The alternative operator is ordinary causal self-attention over the time axis with rotary
embeddings --- $O (L^2)$, no geometric structure. It is retained not as a weaker baseline but
as a #emph[matched diagnostic];: identical stage schedule, identical data, identical
everything except the mixer. Without it, any advantage of the full system could be attributed
to the training recipe rather than to the inductive bias. With it, the comparison is clean.

== Complexity and parameter budget
<sec:complexity>
=== Asymptotic cost
Let $L$ be the temporal sequence length in tokens, $d$ the model width, $r$ the Grassmann
reduced dimension, $lr(|Delta|)$ the number of offsets, $N$ the number of token rows sharing a
group, and $L_v$ the number of visual tokens.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Operator];], [#strong[Time];], [#strong[Activation memory];],),
    table.hline(),
    [Temporal self-attention], [$O (L^2 d)$], [$O (L^2)$ per head],
    [Causal Grassmann mixing], [$O #h(-1em) (L thin lr(|Delta|) thin (d r + r^2))$], [$O (L thin lr(|Delta|) thin r^2)$],
    [Group self-attention (batch axis)], [$O (L N^2 d)$], [$O (L N^2)$],
    [Feed-forward], [$O (L d thin d_(upright("ff")))$], [$O (L d_(upright("ff")))$],
    [Cross-attention to a visual stream], [$O (L L_v d)$], [$O (L L_v)$],
    [Token interleaving (this work)], [absorbed into the temporal operator], [$O (L_v)$ extra
    tokens],
  )]
  , caption: [Per-layer cost. The Grassmann mixer is linear in $L$; the offset set and the reduced
  dimension are constants ($lr(|Delta|) = 6$, $r = 32$), so the operator is $O (L)$ in the quantity
  that grows.]
  , kind: table
  )
<tab:complexity>
Two consequences follow. First, at the context lengths this thesis uses the asymptotic
advantage is real but modest: $T_(upright("ctx")) = 42$ patches is short enough that quadratic
attention is not a bottleneck, so a null accuracy result would #emph[not] be evidence against
the operator's value at scale. The regime where it matters is patch-level or unpatched
context in the thousands, which is why the efficiency comparison in
@sec:efficiency sweeps context length rather than reporting a single number.

Second, the interleaving mechanism is cheap in a way cross-attention is not. Adding $L_v$
visual tokens to a sequence of length $L$ costs $O (L_v)$ additional positions in a linear
operator, against $O (L L_v)$ for cross-attention. With $L = 42$ and $L_v = 1$ the difference is
negligible; with a thousand-step context and eight visual tokens it is three orders of
magnitude. The design is built for the second regime.

=== What is actually trained
#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Component];], [#strong[Origin];], [#strong[Status by stage];],),
    table.hline(),
    [Chronos-2 patch embedding], [#strong[reinitialised];], [trainable throughout],
    [Group self-attention], [pretrained], [last 3 of 12 blocks trainable at $0.1 times$ LR],
    [Feed-forward], [pretrained], [last 3 of 12 blocks trainable at $0.1 times$ LR],
    [Temporal mixer ($times 12$)], [#strong[from scratch] (Grassmann)], [trainable throughout],
    [V-JEPA 2.1 ViT-L/16], [pretrained], [frozen; last four blocks unfrozen in S2a],
    [Latent summariser], [#strong[from scratch];], [trainable from S2a],
    [Cross-modal adapter (late only)], [#strong[from scratch];], [trainable in S2a],
    [Quantile output head], [#strong[reinitialised];], [trainable throughout],
    [Modality-pair biases (4 scalars)], [#strong[from scratch];], [trainable from S2b],
  )]
  , caption: [Provenance and training status of each component. Only four components are newly
  initialised, and the curriculum introduces them one at a time.]
  , kind: table
  )
<tab:params>
The design intent is visible in the table: the overwhelming majority of parameters are
pretrained, and the newly initialised set is small, structurally isolated, and introduced in
a controlled order. This is the concrete meaning of "frozen multimodal foundation model
stack" in the research question --- the claim is not that no parameter is updated, but that
the pretrained representations are preserved and the new machinery is confined to the
interfaces between them.

Stated precisely, so that the word "frozen" is not read more strongly than the
implementation supports: Chronos-2 is #emph[fine-tuned on the train plants in Stage~1];, and
thereafter held fixed #emph[except] its last three of twelve encoder blocks, which continue to
train at $0.1 times$ the base learning rate together with the re-initialised patch
embeddings. Two consequences follow, and both are stated rather than elided. The weights
held fixed are not the pretrained ones but a PV-adapted checkpoint; and a quarter of the
encoder is still adapting while the visual path is learned. Whether the visual path in fact
#emph[needs] that adaptation capacity --- that is, whether a genuinely frozen backbone
($n_(upright("unfreeze")) = 0$) reads the sky as well --- is measured directly by the ablation
study and is not assumed here. V-JEPA~2.1, by contrast, is frozen in
the literal sense: it is consumed from a precomputed latent cache and receives no gradient
in any stage.

=== The failure mode this guards against
An untrained module inserted into a pretrained residual stream produces two problems at once.
Its outputs are noise, which corrupts the representation the downstream pretrained layers
expect; and its gradients are large, which drives those layers away from their pretrained
solution before the new module has learned anything worth preserving. The result is a model
that trains stably, reports plausible losses, and has quietly destroyed the pretraining it was
built to exploit.

Three mechanisms guard against it here. The gated fusion in the Grassmann mixer
(@sec:grassmann) allows the operator to behave as the identity, so the block can
pass the residual stream through unchanged until the mixer is useful. The warmup schedule in
Stage~1 anneals that gate over two thousand steps. And the curriculum holds the pretrained
backbone fixed apart from its top three encoder blocks, which train at $0.1 times$ LR, while
the visual interface is learned --- so gradients from the newly initialised summariser reach
at most the top half of the encoder, and never at full step size.

== Group attention and the encoder block
<sec:block>
Each of the twelve blocks applies, in order: temporal mixing (@sec:grassmann);
#strong[group self-attention];, which attends across the #emph[batch] axis at each sequence
position over all rows sharing a group identifier, fusing the target row with the fourteen
covariate rows and, in late fusion, the visual rows; and a position-wise feed-forward layer
with residual connection. Rotary encoding is not applied in group attention, as there is no
natural ordering along the entity axis. Group attention and the feed-forward layer are
pretrained; the temporal mixer, when Grassmann, is trained from scratch.

Under interleaving, group attention at refinement positions additionally fuses visual tokens
#emph[across entities];, which is the mechanism by which a plant can, in principle, benefit
from a neighbour's cloud observation --- the cross-plant analogue of retrieval.

== Output head
<sec:head>
The whole horizon is predicted in one forward pass. The final hidden states at the future
positions of the target rows are projected to $Q times p_(upright("out"))$ values, reshaped into
$H$ horizon steps by $Q$ quantiles, and rescaled through the inverse of the input
normalisation.

Formally, with $p_(upright("out"))$ timesteps decoded per output embedding and $T_(upright("fut"))$
future positions, the head produces
$ hat(Y)_(b \, h \, q) = #scale(x: 120%, y: 120%)[\[] W_(upright("out")) thin upright(bold(h))_(b \, thin T_(upright("ctx")) + k)^((L)) #scale(x: 120%, y: 120%)[\]]_(q \, j) \, #h(2em) h = k thin p_(upright("out")) + j \, $
for $k in \[ 0 \, T_(upright("fut")) \)$ and $j in \[ 0 \, p_(upright("out")) \)$, so the horizon coverage is
$H = T_(upright("fut")) thin p_(upright("out"))$. Training minimises the masked pinball loss
$ cal(L) = 1 / lr(|cal(M)|) sum_((b \, h) in cal(M)) sum_q max #scale(x: 120%, y: 120%)[\(] tau_q (y_(b \, h) - hat(y)_(b \, h \, q)) \, #h(0em) (tau_q - 1) (y_(b \, h) - hat(y)_(b \, h \, q)) #scale(x: 120%, y: 120%)[\)] \, $
where $cal(M)$ is the set of positions with a valid future target. Masking matters more
here than it might appear: without it the model would be trained to predict zero at night
and at outage steps indiscriminately, learning to reproduce the sensor's failure modes
alongside the physics.

Two properties follow. There is #emph[no error accumulation];: an autoregressive decoder that
errs at $t + 1$ poisons $t + 10$, whereas here each horizon step is decoded from context rather
than from previous predictions. And the output is #emph[distributional by construction] ---
nine quantiles, not a point --- because calibrated uncertainty is a first-class requirement
of the protocol (@sec:metrics) rather than an afterthought. Training minimises
the masked pinball loss over the quantile grid.

== Training curriculum
<sec:curriculum>
The central risk in this architecture is that randomly initialised modules --- the latent
summariser, the cross-modal adapter, the Grassmann mixer --- corrupt the pretrained residual
streams before they learn anything useful. Gradients from an untrained mixer flowing into a
pretrained backbone can destroy in a few hundred steps what pretraining provided.

The curriculum addresses this by introducing new components one at a time, each stage
warm-starting from the previous stage's best checkpoint (weights only, not optimiser state).

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,left,left,left,left,),
    table.header([#strong[Stage];], [#strong[Fusion];], [#strong[Vision];], [#strong[Backbone];], [#strong[Purpose];],),
    table.hline(),
    [S1], [---], [off], [trainable, mixer warmup], [Time-series pretraining; anchor the
    from-scratch mixer before it can corrupt the residual stream],
    [S2a], [late], [last four blocks unfrozen], [frozen], [Learn the vision-to-backbone mapping
    against a stable, low-variance target],
    [S2b], [interleaved], [re-frozen], [frozen except mixer], [Teach the mixer cross-modal geometry
    in isolation],
    [S3], [interleaved], [progressive unfreeze], [all trainable], [Joint fine-tuning],
  )]
  , caption: [The four-stage curriculum. Each stage warm-starts from the previous stage's best
  checkpoint.]
  , kind: table
  )
<tab:curriculum>
Stage~1 includes an explicit warmup of two thousand steps during which the Grassmann mixer's
gate is annealed, so that the block behaves close to the identity until the mixer has learned
something. Stage~2a deliberately uses #emph[late] fusion even though interleaving is the
contribution: late fusion presents the visual branch with a stable target, and only once that
mapping is learned does Stage~2b introduce cross-modal pairs into the temporal operator.

=== Why the stages are ordered this way
The ordering is not arbitrary, and each transition is motivated by a specific failure it
avoids.

#emph[Stage 1 before anything visual] because the Grassmann mixer is the only from-scratch
component inside the pretrained residual stream. Introducing it simultaneously with a
from-scratch visual interface would make an unstable result impossible to attribute. Training
it alone, on the numerical task, with an annealed gate, establishes a mixer that at worst
behaves as the identity before any other novelty is added.

#emph[Stage 2a with late fusion, not interleaving];, even though interleaving is the
contribution. The visual branch must learn a mapping from V-JEPA representations into the
backbone's embedding space, and that is far easier to learn against a stable target. Late
fusion provides one: the visual tokens enter through group attention, which is pretrained and
frozen, so the summariser is optimising against a fixed function. Introducing interleaving at
this point would make the target move --- the mixer's behaviour on cross-modal pairs would be
changing while the summariser was still learning what to emit.

#emph[Stage 2b with the backbone frozen except the mixer];, so that the only thing learning
from cross-modal Plücker pairs is the operator that computes them. If the whole backbone were
trainable here, an improvement could be attributed to general adaptation rather than to the
mixer learning cross-modal geometry.

#emph[Stage 3 last];, when every interface has been learned and full joint training can refine
rather than establish.

#[
#set par(first-line-indent: 0pt)
*Warm-starting, not resuming.* Each stage loads the previous stage's weights but
not its optimiser state. This is deliberate: the parameter set changes between stages, and
carrying momentum estimates for a parameter whose role has just changed --- from frozen to
trainable, or from late-fusion to interleaved context --- is more likely to destabilise than
to help.
]

#[
#set par(first-line-indent: 0pt)
*Modality dropout.* Throughout training, modalities are dropped independently with
asymmetric Bernoulli probabilities, using the same mask that appears in the tensor contract.
This serves two purposes: robustness to genuinely missing frames at inference (night,
outage, sensor gaps), and prevention of the degenerate solution in which the model ignores one
modality entirely because the other is easier to exploit early in training.
]

The asymmetry matters. The visual stream is dropped more aggressively than the numerical one,
for the simple reason that it is genuinely absent far more often --- there are no night frames
--- and a model trained under symmetric dropout would be calibrated for a missingness pattern
that does not occur. The numerical stream is dropped only lightly, because a forecaster with
no power history at all is not a configuration that needs to work.

== Alternatives considered and rejected
<sec:alternatives>
Several designs were evaluated on paper and not adopted. Recording them makes the design
space visible and prevents a reader from assuming an obvious option was overlooked.

#[
#set par(first-line-indent: 0pt)
*A reconstructive video tokeniser instead of V-JEPA.* A vector-quantised video
autoencoder was the initial candidate for the visual branch. It was rejected because its
training objective spends representational capacity on pixel-level fidelity, and for
satellite crops much of that detail is sensor noise. A predictive objective in representation
space yields features organised by what is semantically coherent across space and time,
which is what cloud-structure detection needs. The distinction matters more here than in
natural video: two satellite crops that differ substantially in pixels may be
meteorologically identical, and a reconstruction loss cannot express that.
]

#[
#set par(first-line-indent: 0pt)
*Cross-attention between the two streams.* The obvious deep-fusion mechanism, and
the one CrossViVit and Solar-VLM use. It was rejected on cost grounds rather than on
principle: cross-attention between a temporal sequence of length $L$ and $L_v$ visual tokens
costs $O (L L_v)$ per layer, which forces either a short numerical context or a small visual
budget. Preliminary work established that short numerical contexts are exactly what harms
foundation-model performance on this task, so the design that preserves a fourteen-day
context was preferred. Interleaving achieves layer-wise cross-modal interaction at
$O (L + L_v)$.
]

#[
#set par(first-line-indent: 0pt)
*Full fine-tuning of both backbones.* The simplest way to adapt two pretrained
models is to unfreeze both and train end to end. It was rejected for two reasons. It forfeits
the property the research question is about --- whether #emph[frozen] foundation models can
be composed --- and it is the configuration most likely to destroy pretrained representations
early in training, since gradients from a randomly initialised fusion interface reach both
backbones from step one. Full joint training does occur, but only in the final stage, after
the interfaces have been learned against frozen targets.
]

#[
#set par(first-line-indent: 0pt)
*A per-sensor input projection.* An earlier design included a learned $1 times 1$
convolution per imagery source, to accommodate a "sensor zoo" of heterogeneous satellite
products. It was removed once the data loader was standardised to deliver uniformly shaped
three-channel frames for every plant and source: the projection had nothing left to do, and
its parameters were a free path for the visual branch to overfit source identity rather than
cloud structure.
]

#[
#set par(first-line-indent: 0pt)
*Autoregressive horizon decoding.* Decoding the horizon step by step, conditioning
each prediction on the last, is the default in much of the forecasting literature. It was
rejected because errors compound --- a mistake at $t + 1$ propagates through the remaining
eleven steps --- and because it is incompatible with the quantile output: sampling from a
quantile head to feed the next step either collapses the distribution or requires an
ensemble. Predicting the whole horizon in one pass avoids both, at the cost of not modelling
dependence #emph[between] horizon steps, which the evaluation metrics do not measure.
]

#[
#set par(first-line-indent: 0pt)
*Interleaving across the full context.* Placing visual tokens throughout the
sequence rather than only in a refinement window was considered and rejected on two grounds.
Frames more than a few hours old carry no information about the coming six hours, so the
tokens would be uninformative; and inserting them into the macro region would disturb the
uniform temporal spacing that the pretrained backbone's positional structure assumes. The
refinement window confines the disturbance to the region where the visual signal is live.
]

== Implementation
<sec:impl>
The system is built on PyTorch Lightning~@lightning, which separates the training
lifecycle from the architecture and supplies distributed training, mixed precision and
checkpointing; and on Hydra~@hydra, which composes model, data, trainer and stage
configuration groups so that no hyperparameter is hardcoded in model code. Training runs on
the Leonardo cluster under SLURM.

#[
#set par(first-line-indent: 0pt)
*An engineering finding worth reporting.* The time-series-only ablation arm ---
the same model with the vision stack disabled --- consumes #emph[more] memory than the full
multimodal arm, which is counter-intuitive enough to have cost two failed runs. With vision
off, samples fall through the standard multivariate code path in which every covariate
channel becomes its own series row: fifteen rows per sample rather than one, producing a
tensor of $[30 \, 86 \, 768]$ at batch size two against $[2 \, 89 \, 768]$ for the interleaved path
--- a fifteenfold expansion along the batch dimension entering group attention. The fix was
to reduce batch size and compensate with gradient accumulation. The observation is recorded
because it is the sort of detail that makes an ablation appear to fail for architectural
reasons when the cause is a code path.
]


= Experimental Design and Results
<cha:expdesign>
The training campaign for MMTSFM was still running at the time of writing. This chapter
therefore states, in full, #emph[how] the model is measured --- the arms, the controls, the
reporting rules and the integrity constraints --- and leaves the result tables as
placeholders to be filled from the completed runs. Earlier development runs exist but are
superseded by the current campaign and are deliberately not reported here: a provisional
number that later moves is worse than no number, because it propagates into every
downstream comparison.

Everything below is on the primary scenario of @sec:scenarios: cross-plant
(S2) on `uk_pv`, fourteen disjoint test plants, fourteen days of history, six-hour
horizon, evaluated against the bars established in @sec:bar.

== Arms
<sec:arms>
Four arms constitute the main experiment. They are designed so that each pair differs in
exactly one factor, which is what makes the differences interpretable.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,left,left,left,left,),
    table.header([#strong[Arm];], [#strong[Vision];], [#strong[Mixer];], [#strong[Fusion];], [#strong[Isolates];],),
    table.hline(),
    [A --- numeric], [off], [Grassmann], [---], [The vision-lift lower bound],
    [B --- late], [on], [self-attention], [late], [The conventional multimodal design],
    [C --- interleaved], [on], [Grassmann], [interleaved], [The proposed model],
    [D --- no pair bias], [on], [Grassmann], [interleaved], [The modality-pair correction],
  )]
  , caption: [The four main arms. All share the curriculum, data, seed and schedule of
  @cha:method; they differ only in the stated factor.]
  , kind: table
  )
<tab:arms>
Three comparisons follow directly:

- #strong[C against A] answers H1 --- does the visual stream contribute measurable
  skill at all? Arm~A is the lower bound: the identical recipe with the vision stack
  disabled. If C does not exceed A, no amount of fusion sophistication is doing work, and
  any advantage over an external baseline comes from the training recipe rather than from
  multimodality.

- #strong[C against B] answers H2 --- does fusion #emph[depth] matter? Same backbones,
  same data, same schedule; only the point at which the modalities meet differs.

- #strong[C against D] isolates the modality-aware offset gating of
  @sec:grassmann, and is expected to show up in calibration at least as much
  as in point accuracy.

Arm~A deserves emphasis because it is the arm most often omitted in multimodal work. Without
a modality-off control run under the #emph[identical] recipe, a multimodal model that beats
an external unimodal baseline proves nothing about its second modality: the gain may come
entirely from the curriculum, the normalisation or the horizon head. Arm~A is what converts
the comparison from suggestive to conclusive, in whichever direction it falls.

== Reporting template
<sec:reporting>
Results will be recorded in the form of @tab:mmtsfm-results, with each arm's
in-process test metrics, and read against the three bars of @sec:bar: 0.552
overall, 0.540 multimodal, and 0.331 for the same backbone fine-tuned.

#figure(
  align(center)[#table(
    columns: 6,
    align: (left,center,center,center,center,center,),
    table.header([#strong[Arm];], [#strong[NMAE] $arrow.b$], [#strong[NRMSE] $arrow.b$], [#strong[SS] $arrow.t$], [#strong[CRPS] $arrow.b$], [#strong[Cov\@80];],),
    table.hline(),
    [A --- numeric (vision off)], [---], [---], [---], [---], [---],
    [B --- late fusion], [---], [---], [---], [---], [---],
    [C --- interleaved (proposed)], [---], [---], [---], [---], [---],
    [D --- no pair bias], [---], [---], [---], [---], [---],
    table.cell(align: left, colspan: 6)[#emph[Reference points];],
    [Chronos-2 fine-tuned (same backbone)], [0.1120], [0.1543], [0.331], [0.0855], [---],
    [TS-RAG (best frozen-backbone adaptation)], [0.0705], [0.1203], [0.478], [---], [---],
    [iTransformer (overall bar)], [0.0699], [0.1032], [0.552], [---], [---],
  )]
  , caption: [Reporting template for the MMTSFM campaign. Reference rows are from
  @tab:leaderboard; the arm rows are pending completion of the runs.]
  , kind: table
  )
<tab:mmtsfm-results>
Three secondary views are reported alongside the aggregate table, because the aggregate
alone cannot distinguish the hypotheses of interest.

#[
#set par(first-line-indent: 0pt)
*Per-horizon profile.* Normalised mean absolute error as a function of lead time,
per arm. If the visual stream contributes, its contribution should be concentrated at short
to medium lead times, where advection is predictable from the observed cloud field, and
should decay as the field leaves the crop. An arm whose profile is indistinguishable from
the vision-free arm at #emph[every] lead time has not used its frames, regardless of what
the aggregate says.
]

#[
#set par(first-line-indent: 0pt)
*Ramp subset.* The same metrics restricted to the top decile of $lr(|Delta y|)$ per
site (scenario S6). This is where the multimodal hypothesis lives or dies: clear-sky periods
are won by persistence, and vision earns its tokens only during cloud transitions. The
reference to beat here is Time-VLM's ramp NRMSE of 0.172, the best in the suite.
]

#[
#set par(first-line-indent: 0pt)
*Calibration.* Empirical coverage of the nominal 80% interval alongside the
continuous ranked probability score. The modality-pair-bias ablation in particular is
expected to affect the predictive distribution more than the conditional mean.
]

== The run matrix
<sec:run-matrix>
Each arm is not one job but a chain of four, one per curriculum stage, each warm-starting from
the previous stage's best checkpoint. @tab:runmatrix states the full matrix, which
is what determines the compute budget and the order in which results become available.

#figure(
  align(center)[#table(
    columns: 6,
    align: (left,center,center,center,center,left,),
    table.header([#strong[Arm];], [#strong[S1];], [#strong[S2a];], [#strong[S2b];], [#strong[S3];], [#strong[Notes];],),
    table.hline(),
    [A --- numeric (vision off)], [yes], [---], [---], [yes], [S2a/S2b are vision stages and do not
    apply],
    [B --- late fusion, self-attention], [yes], [yes], [---], [yes], [stops at late fusion by
    construction],
    [C --- interleaved, Grassmann], [yes], [yes], [yes], [yes], [the full curriculum],
    [D --- interleaved, no pair bias], [yes], [yes], [yes], [yes], [branches from C at S2b],
  )]
  , caption: [Curriculum stages required per arm. Arms C and D share their S1 and S2a
  checkpoints, so the marginal cost of D is two stages rather than four.]
  , kind: table
  )
<tab:runmatrix>
Two properties of this matrix affect interpretation. Arms C and D share their earlier stages,
so the comparison between them isolates the modality-pair bias cleanly --- they are identical
up to the point where the bias is introduced. Arm A, by contrast, skips the vision stages
entirely and therefore differs from the others in #emph[schedule] as well as in modality; its
S1 and S3 are longer to compensate for the stages it does not run. That asymmetry is
unavoidable --- a vision-free model cannot run a visual alignment stage --- and it is a
caveat on the vision-lift comparison that must be stated whenever that comparison is reported.

== Controls
<sec:ablation-design>
Aggregate metrics cannot establish that a model #emph[reads] its second modality; they can
only establish that having it does not hurt. Four controls are specified for that purpose,
and their status is stated honestly.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Control];], [#strong[What it establishes];], [#strong[Status];],),
    table.hline(),
    [Shuffled frames], [The frames are read #emph[temporally];. A random permutation of the visual
    window should degrade a model that uses advection; a model whose gain is regularisation is
    unaffected.], [specified],
    [Mismatched-plant frames], [The frames are #emph[spatially grounded];. Substituting another
    plant's frames should degrade a model using local cloud structure; a model relying on a
    generic cloudiness prior is unaffected.], [specified],
    [Modality grid], [A four-way decomposition --- target only, target with covariates, target
    with vision, all three --- attributing the gain to a specific input rather than to their
    combination.], [specified],
    [Vision-only], [An upper bound on the information content of the visual stream alone.], [specified],
  )]
  , caption: [Control battery. "Specified" means designed and registered but not yet executed.]
  , kind: table
  )
<tab:controls>
The shuffled-frames and modality-grid controls are the highest-value of the four, because
between them they distinguish the three explanations of any observed outcome: the frames
carry usable signal and the model extracts it; the frames carry signal that the encoder
interface fails to transmit; or the frames are redundant with the covariate track, which
already contains cloud cover and irradiance. The benchmark of @cha:baselines
--- where the best multimodal model uses no external imagery --- makes the second and third
explanations live possibilities rather than pedantic alternatives, which raises the value of
running these controls considerably.

Beyond the four above, three sensitivity studies are specified: the visual window length
(three, six and twelve hours), the visual token budget after the latent summariser, and
frozen against partially unfrozen backbones. The last separates the contribution of the
fusion architecture from the contribution of raw adaptation capacity, and complements the
Chronos-2 fine-tuned row already in @tab:leaderboard.

== How each control would be interpreted
<sec:control-logic>
A control is only useful if its possible outcomes have been assigned meanings #emph[before]
it runs. @tab:control-logic does that for the two decisive controls, so that the
result cannot be rationalised after the fact.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Control];], [#strong[Outcome];], [#strong[Conclusion];],),
    table.hline(),
    [Shuffled frames], [Skill degrades materially], [The model reads temporal structure in the
    visual stream; the visual path is working],
    [], [Skill unchanged], [The frames contribute nothing temporal --- at best a static prior, at
    worst nothing],
    [], [Skill improves], [The visual path is actively harmful and the shuffle acts as
    regularisation],
    [Modality grid], [Target $<$ target$+$cov $<$ target$+$cov$+$vis], [Both modalities contribute;
    the architecture works as intended],
    [], [target$+$cov $=$ target$+$cov$+$vis], [Vision is redundant with the covariate track],
    [], [target$+$vis $>$ target$+$cov], [Vision carries information the covariates do not, but
    fusion of all three fails to combine them],
  )]
  , caption: [Pre-registered interpretation of the two decisive controls. Assigning meanings in
  advance is what prevents a null result from being explained away.]
  , kind: table
  )
<tab:control-logic>
The third row of each block is the one that would be most informative, and neither is the
outcome the architecture was designed to produce. Evidence from
@cha:benchmark makes them live possibilities rather than pedantic ones: real
satellite frames demonstrably carry cloud-state information beyond the covariate track
(@fig:cloud-residual), and yet no model in the suite that consumes them converts
that into skill.

== Statistical treatment
<sec:statistics>
Three practices govern how differences between arms are read.

#[
#set par(first-line-indent: 0pt)
*Ties.* Differences smaller than the seed-to-seed variance are reported as ties, not
as orderings. Since that variance has not been measured for this architecture --- one seed per
arm --- the operational rule is conservative: no ordering is claimed between arms differing by
less than approximately 0.01 skill, which is the scale on which nearby leaderboard entries
already sit.
]

#[
#set par(first-line-indent: 0pt)
*Overlapping windows.* Consecutive forecast origins share most of their context, so
per-window losses are strongly autocorrelated and naive tests overstate significance. The
appropriate instrument is a Diebold--Mariano test~@diebold on the loss-differential
series with an autocorrelation-consistent variance estimator, computed per plant and combined
across plants. This is specified but not executed, and its absence is a stated limitation.
]

#[
#set par(first-line-indent: 0pt)
*Per-plant reporting.* Alongside every aggregate figure, the per-plant distribution
is retained, so that an improvement driven by one anomalous plant can be distinguished from a
uniform improvement. Win rate against the reference over the fourteen test plants
(@app:aggregation) is the summary statistic for this: a model beating the
reference on every plant by a modest margin is a more trustworthy result than one with the
same mean driven by three plants.
]

== Compute and reproducibility
<sec:compute>
Training runs on the Leonardo cluster under SLURM, with one job per arm and per-job logs
retained so that any reported number traces to the job that produced it. Configuration is
composed by Hydra; the exact override set for each arm is recorded in the experiment registry
(@app:registry) before the job is submitted, not reconstructed from logs
afterwards.

Three practices support reproducibility beyond seed fixing. Splits are committed to version
control rather than regenerated, and their disjointness is asserted at every load, so a
future run cannot silently receive a different partition. Every experiment is registered
before it launches with a hypothesis, a configuration diff and a nominated comparison
baseline, which prevents the retrospective reframing of an experiment as a test of whatever
it happened to show. And the dataset of record is read-only, so no result can depend on an
undocumented change to the data.

== Sensitivity studies
<sec:sensitivity>
Three sweeps are specified beyond the controls. Each varies one design constant that was
fixed by argument rather than by measurement, and each has a prediction attached.

#[
#set par(first-line-indent: 0pt)
*Visual window length: 3, 6 and 12 hours.* @sec:decoupled argues that
vision carries information only over the cloud-advection horizon, and that widening the
window adds cost rather than signal. The prediction is a concave profile: three hours is too
short to capture an approaching field, twelve hours includes frames whose cloud structure has
left the domain, and six hours --- the configuration of record --- is near the optimum. A
monotone increasing profile would falsify the decoupled-resolution argument and indicate that
the visual context should simply be longer.
]

#[
#set par(first-line-indent: 0pt)
*Visual token budget.* The latent summariser compresses $4 times 196$ V-JEPA
patch embeddings into $n_(upright("vis"))$ tokens, with $n_(upright("vis")) = 1$ for `uk_pv`.
That is aggressive compression, and it is a candidate explanation for a weak visual
contribution: a single 768-dimensional vector may simply be too narrow a channel for the
spatial structure of a cloud field. Sweeping the budget separates "the vision path does not
work" from "the vision path is bottlenecked". If skill rises with token count, the
interface is the problem; if it is flat, the interface is not.
]

#[
#set par(first-line-indent: 0pt)
*Frozen against partially unfrozen backbones.* The curriculum unfreezes
progressively, and the final stage trains everything. Comparing arms that stop at each stage
separates the contribution of the fusion architecture from the contribution of raw adaptation
capacity. It also complements the Chronos-2 fine-tuned row of
@tab:leaderboard: if a fully unfrozen MMTSFM merely matches a fine-tuned bare
backbone, the fusion machinery is contributing nothing that adaptation does not.
]

== Horizon-resolved analysis
<sec:horizon-analysis>
Aggregate metrics average over a six-hour horizon in which the information available to the
model changes character completely. At the first step the model is essentially doing
persistence with a correction; by the twelfth, the cloud field observed at the forecast
origin has largely advected out of the crop and only the numerical context and the
deterministic covariates remain informative. Reporting a single number across that range
hides the mechanism.

Three horizon-resolved views are therefore specified. #emph[Per-step NMAE] per arm, which
should separate the arms at short lead times if vision works at all. #emph[Skill decay] at
one, six and twenty-four hours, which positions the model against the regimes of
@tab:leadtime. And #emph[the vision-minus-no-vision difference as a function of
lead time];, which is the sharpest single plot the experiment can produce: a positive
difference concentrated at short horizons decaying to zero is exactly the signature the
architecture predicts, and a flat zero line at every horizon is exactly the signature of a
visual path that is not being read.

== An efficiency claim measured separately
<sec:efficiency>
The Grassmann operator's asymptotic advantage is a claim about #emph[cost];, and it must be
measured as one rather than inferred from an accuracy table. The comparison against the
matched self-attention twin is therefore reported on two axes: peak memory and step latency
as a function of context length, at $T in { 128 \, 512 \, 1024 \, 2048 }$ patches, with all else
held fixed.

This matters for how a null accuracy result would be interpreted. If the two mixers score
identically but one is linear in sequence length, the geometric operator is still the better
engineering choice at long context, and the honest conclusion is that its inductive bias is
neutral rather than beneficial at the context lengths tested. Reporting only accuracy would
lose that distinction.

== Integrity constraint on reported numbers
<sec:integrity>
One methodological constraint governs every number this chapter will contain, and it is
stated here rather than in a footnote because it restricts what may be claimed.

Saved checkpoints of this model #strong[do not reproduce their training-time scores when
re-scored in a fresh process];. The discrepancy is large --- validation loss moving from
approximately 2.84 to 4.08, with a corresponding collapse in test skill --- and it affects
both temporal-mixer variants, so it is not specific to the Grassmann operator.

The obvious explanations were eliminated experimentally. Weight restoration is
tensor-faithful: all non-vision parameter keys apply and their values were verified
element-wise after loading. The configurations are bit-identical between the saving and
loading processes. The visual feature cache is unchanged. The dataset iteration is
deterministic under the fixed seed, and the derived buffers are deterministic. Restoring
through the training entry point rather than the evaluation entry point fails identically.
The conclusion that survives is that the checkpoint #emph[content] does not correspond to the
model that produced the scores, despite capturing every registered parameter --- and that
apparently seamless resumed runs are consistent with one or two epochs of retraining
repairing the damage rather than with a faithful restore.

Two consequences follow, and both are binding:

+ #strong[In-process numbers are the record.] Metrics computed within the training
  process that produced the model are internally consistent and are what will be reported.

+ #strong[No post-hoc re-scoring.] No number in this thesis may be derived by reloading
  a checkpoint and evaluating it, and no analysis requiring that path --- including
  activation-level inspection of a trained model --- can currently be performed.

The next diagnostic step is an activation fingerprint bisection: dumping per-module outputs
on a fixed batch at save time and again after reload, and locating the first module whose
output diverges. Until that is resolved, the constraint above stands, and the reader should
treat any claim in this thesis that would require re-scoring a checkpoint as unsupported.

== What each outcome would mean for the thesis
<sec:outcomes>
The campaign has three broad possible outcomes, and the thesis is written so that each of
them is reportable. Stating this in advance is not hedging; it is what prevents the result
from being framed after the fact to suit whatever happened.

#[
#set par(first-line-indent: 0pt)
*If interleaving beats late fusion, and both beat the vision-free arm.* The
architectural hypotheses H1 and H2 are supported. The contribution is then the mechanism
itself, and the interesting follow-up questions are about scale: does the advantage grow with
context length, where the $O (L)$ mixer's asymptotics start to matter, and does it survive the
cross-dataset shift of scenario S3.
]

#[
#set par(first-line-indent: 0pt)
*If the arms are indistinguishable.* The vision-free arm matches the fusion arms,
and interleaving matches late fusion. H1 and H2 are unsupported at the scale tested, and the
thesis's contribution shifts to the benchmark and the controls --- which is why
@cha:benchmark is written to stand on its own. This outcome is not a failure of
the experiment; it is a measurement, and given the benchmark evidence that no
satellite-consuming model in the suite converts imagery into skill, it is the outcome the
surrounding evidence currently favours. The controls of @sec:ablation-design
would then become the primary result, because they distinguish a redundant modality from a
mis-interfaced one.
]

#[
#set par(first-line-indent: 0pt)
*If the vision-free arm #emph[beats] the fusion arms.* The visual path is actively
harmful --- most likely because the newly initialised summariser and fusion interface consume
capacity and inject noise without supplying information. This would be the most informative
outcome of the three for the field, and the diagnostic path is clear: the token-budget sweep
(@sec:sensitivity) separates a bottleneck from a nuisance, and the shuffled-frames
control establishes whether anything temporal is being read at all.
]

In all three cases the reporting rule is the same. The vision-free arm is reported as a
headline row, not as an appendix ablation, and no comparison against an external baseline is
presented without it.

== Threats to validity
<sec:threats>
+ #strong[A single dataset.] Every number reported in this thesis is
  `uk_pv`. Cross-dataset transfer to `goes_pvdaq` (scenario S3) is
  #emph[unproven];, not disproven, and no claim about generalisation beyond a single
  fleet, sensor and climate is supported.

+ #strong[A single seed per arm.] Differences smaller than the seed-to-seed variance
  cannot be distinguished, and that variance has not been measured. No significance test
  --- Diebold--Mariano~@diebold or otherwise --- accompanies the comparisons, so
  small differences between arms must be read as ties.

+ #strong[The checkpoint integrity issue] of @sec:integrity.

+ #strong[Ramp subsets are not bit-aligned across tiers.] For the externally
  integrated baselines the ramp mask is recomputed from saved predictions on their native
  windows, so ramp columns are comparable #emph[within] a tier or by rank, not
  element-wise across the whole leaderboard.

+ #strong[Two harness-limited baseline rows.] Time-VLM's evaluation windows are not
  aligned with the rest of the suite, and UniCast's weakness is partly attributable to a
  window skew in its native evaluation regime. Both are flagged in
  @sec:implementation; neither number should be used as a precise comparison
  point.

+ #strong[Oracle rows are upper bounds.] The two privileged-information variants
  quantify available headroom and are never competitors.

+ #strong[Random rather than region-based plant splits.] As discussed in
  @sec:splits, spatially adjacent plants may straddle the split boundary,
  which inflates absolute numbers across the board.


= Conclusions
<cha:conclusion>
== Summary
<sec:summary>
This thesis set out to determine whether frozen multimodal foundation models, fused deeply,
can forecast photovoltaic plants they have never observed. Answering that required building
three things: a dataset, a protocol, and a model.

The #strong[dataset] fuses four public sources into $1 \, 337 \, 654$ rows over 110 sites in
two regions, pairing measured power and fourteen covariates with co-registered satellite
imagery on one timestamp grid, addressed by a single verified frame pointer, split by plant
identity with disjointness asserted at load time. The #strong[protocol] fixes windows in
physical time, normalises by nameplate capacity so that no test-plant statistic enters any
fitted component, forbids domain physics inside models, and constrains retrieval datastores
to training plants. The #strong[benchmark] runs 28 forecasters under that protocol, spanning
statistical references, classical machine learning, supervised deep networks, four
time-series foundation-model families, retrieval and adapter adaptation, and both endogenous
and exogenous multimodal systems. The #strong[model];, MMTSFM, fuses frozen Chronos-2 and
frozen V-JEPA~2.1 through selective temporal interleaving and causal Grassmann mixing, under
a four-stage curriculum designed to keep randomly initialised modules from corrupting
pretrained residual streams.

== Answering the research question
<sec:answer>
The research question asked whether deep token-level fusion of frozen foundation models
achieves cross-plant forecasting better than late fusion or domain-specific architectures.
The evidence currently available supports a partial answer, and the parts differ in how firm
they are.

#[
#set par(first-line-indent: 0pt)
*H3 --- cross-plant transfer --- is established as a workable evaluation regime.* Models trained on 69 plants forecast fourteen unseen plants from two weeks of history at
skill levels up to 0.552. The protocol discriminates sharply: the spread between the best and
worst models is far larger than any plausible measurement noise, so the benchmark is
informative rather than saturated.
]

#[
#set par(first-line-indent: 0pt)
*H0 --- foundation models as anchors --- resolves negatively in its strong form.* Zero-shot time-series foundation models do not merely underperform; three of four fall below
hourly climatology, and one falls below the naive floor. The result replicates across three
architecture families, so it is a property of photovoltaic dynamics rather than of a
particular pretraining corpus. Adaptation is mandatory on this task.
]

#[
#set par(first-line-indent: 0pt)
*H4 --- retrieval --- is supported, and strongly.* Retrieval over a frozen backbone
reaches 0.478 against 0.331 for fine-tuning the same backbone: 0.147 skill, with no weight
updates. Among the frozen-backbone strategies tested, supplying relevant precedent beats both
re-fitting the model and injecting covariates through an adapter.
]

#[
#set par(first-line-indent: 0pt)
*H1 and H2 --- vision, and fusion depth --- remain open, and the benchmark makes* them harder rather than easier.
The proposed model's own arms had not completed at the time
of writing, so the direct tests are pending. But the surrounding evidence is not neutral. The
best multimodal model in the suite consumes no external imagery, and every system that does
consume genuine satellite frames sits mid-pack. Any claim that deep fusion of real imagery
helps must now clear a bar set by a model that renders its own input as a picture.
]

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[Hypothesis];], [#strong[Claim];], [#strong[Verdict];],),
    table.hline(),
    [H0], [A time-series foundation model applied zero-shot is a meaningful reference], [#strong[Rejected in its strong form];: three of four fall below climatology],
    [H1], [A visual stream improves over an identical time-series-only model], [#strong[Open];;
    the direct control was still running],
    [H2], [Selective interleaving improves over late fusion], [#strong[Open];; same reason],
    [H3], [Cross-plant zero-shot forecasting from a short history is achievable], [#strong[Supported];: up to 0.552 skill on disjoint plants],
    [H4], [Retrieval on a frozen backbone closes the gap to fine-tuning], [#strong[Supported, and
    then some];: 0.478 against 0.331],
  )]
  , caption: [The hypothesis ladder of @sec:rq against the evidence obtained.]
  , kind: table
  )
<tab:verdicts>
== What the benchmark teaches
<sec:lessons>
Three conclusions generalise beyond this application.

#[
#set par(first-line-indent: 0pt)
*Visual inductive bias and visual information are different resources, and the* literature conflates them.
Endogenous multimodal methods --- those that render a series as
an image --- exploit the #emph[priors] of a visual backbone: locality, periodicity, texture
sensitivity, multi-scale structure. Exogenous methods supply genuinely new observations. On
this benchmark the first converts into skill and the second, so far, does not. Reporting them
in one table, as is standard, obscures the finding entirely. At minimum, multimodal
forecasting results should state which resource is being exploited.
]

#[
#set par(first-line-indent: 0pt)
*The modality-off arm is not optional.* A multimodal model that beats an external
unimodal baseline has demonstrated nothing about its second modality, because the comparison
confounds modality with training recipe, normalisation, horizon head and a dozen other
choices. Only an arm that is identical except for the modality isolates the contribution.
This thesis treats that arm as the primary result rather than as an appendix ablation, and
the design of @cha:expdesign is built around it.
]

#[
#set par(first-line-indent: 0pt)
*Conditioning may matter more than architecture.* The single largest measured
quantity in the entire study is not an architectural difference. It is the 0.173 skill gap
between a Chronos-2 variant that sees observed future weather and the same model that does
not. No pair of architectures in the 28-model suite differs by that much. If a fraction of
that gap can be recovered by better conditioning --- improved covariate forecasts, retrieved
analogous regimes, or a visual stream that actually informs the horizon --- it would exceed
what any architectural change in this study achieved.
]

#[
#set par(first-line-indent: 0pt)
*Negative and null results carry the information here.* Three of the most useful
outputs of this work are things that did not happen: zero-shot foundation models did not
transfer, exogenous imagery did not convert into skill, and pretraining did not beat a
well-matched supervised architecture. None of these would have appeared in a study designed
only to demonstrate a proposed model, because none of them requires the proposed model to
observe. They required a benchmark with the alternatives in it. That is an argument for
building the instrument before building the method, and it is the methodological position
this thesis ended up defending.
]

#[
#set par(first-line-indent: 0pt)
*A note on what "foundation model" bought.* It is worth separating two claims that
are often merged. Pretraining did not give zero-shot competence on this task --- that is
settled by @sec:finding2. But pretraining did give a usable
#emph[initialisation];: fine-tuned Chronos-2 reaches 0.331 and adapted TTM-R2 0.364, both well
above their zero-shot counterparts and above the naive references. The honest summary is that
on photovoltaic data these models are good starting points and poor finished products, which
is a considerably weaker claim than the one the term "foundation model" invites.
]

== Limitations
<sec:limitations>
- #strong[One dataset.] All results are `uk_pv`: one region, one sensor, one
  capacity class, two years. Cross-dataset transfer is untested.

- #strong[One seed per configuration];, no significance testing, and no measured
  seed-to-seed variance; small differences must be read as ties.

- #strong[Half the control battery unexecuted];, including the two controls
  (shuffled frames, modality grid) that would most directly settle the vision question.

- #strong[An open checkpoint-integrity defect] that blocks all post-hoc analysis of
  trained models (@sec:integrity).

- #strong[Random rather than region-based plant splits];, which allows spatially
  adjacent plants to straddle the split boundary.

- #strong[An inconsistent `goes_pvdaq` split] that predates the bad-site flags
  and must be regenerated before that track is used.

- #strong[Two harness-limited baseline rows] whose numbers are indicative rather than
  precise.

=== Which limitations actually threaten the conclusions
The list above is not uniformly serious, and treating it as such would be its own form of
dishonesty. Ranked by how much they threaten what this thesis claims:

#emph[The single dataset is the real threat.] Every conclusion in
@cha:benchmark is a conclusion about a fleet of 98 UK residential rooftops
observed through one satellite product over two years. The finding that supervised training
beats pretraining is exactly the kind of finding that could invert on a fleet with fewer
training plants or greater heterogeneity --- and the sample-efficiency scenario that would
test it was not run. This limitation bounds the scope of every headline in the thesis.

#emph[The missing significance testing is moderate.] It affects the reading of small
differences --- whether TS-RAG genuinely beats Cross-RAG --- but not the reading of large
ones. A 0.221 gap between iTransformer and fine-tuned Chronos-2 is not a statistical artefact
under any plausible variance estimate.

#emph[The checkpoint defect is severe but narrow.] It forecloses post-hoc analysis of trained
models entirely, which is a serious methodological handicap for the architecture work. It
does not affect any baseline number, since those were re-derived from saved predictions.

#emph[Random rather than region-based splits inflate all models roughly equally.] The concern
is real for the absolute values and weak for the ordering, which is what the thesis argues
from.

#emph[The unexecuted controls do not threaten a conclusion --- they withhold one.] Nothing
claimed here depends on them; they are what would be needed to claim more.

== Future work
<sec:future>
The following are ordered by expected information gained per unit of compute, which is not
the order in which they are most tempting.

+ #strong[Run the two decisive controls.] Shuffled frames and the four-way modality
  grid are cheap relative to a training run and would determine whether the visual path is
  uninformative or merely mis-interfaced. Every subsequent decision about the vision branch
  depends on that answer, so it should be bought first.

+ #strong[Attack conditioning rather than architecture.] The 0.173 skill oracle gap is
  the largest measured headroom in this work. Concretely: better future-covariate
  estimates, regime retrieval as conditioning, or a visual stream whose output is injected
  at the horizon rather than only in the context.

+ #strong[Combine retrieval with fusion.] The two most effective levers identified ---
  retrieval over a frozen backbone, and multimodal conditioning --- have never been tested
  together. Retrieval supplies precedent, vision supplies the current state of the sky, and
  there is no obvious reason the two should be redundant.

+ #strong[Close the checkpoint-integrity defect] by activation-fingerprint bisection.
  Until this is fixed, no trained model in this line of work can be analysed after the fact,
  which forecloses interpretability work entirely.

+ #strong[Extend to the second dataset and to region-based splits.] Cross-dataset
  transfer (scenario S3) is the standard evidence a foundation-model claim is expected to
  provide, and region-based splits would remove the spatial-leakage concern in one step.

+ #strong[Reconsider the visual interface.] If the controls show that the frames carry
  signal the current encoder does not transmit, the summariser --- not the fusion operator
  --- is the component to redesign. A structured state formulation, in which the visual
  stream updates an explicit latent state rather than contributing tokens, is the natural
  next design and exists in draft.

=== The three experiments a follow-up should run first
The list above is ordered by expected information per unit of compute. The top three deserve
concrete specification, because "future work" that is not specified does not get done.

#[
#set par(first-line-indent: 0pt)
*Experiment 1: shuffled frames, one training run.* Take the interleaved arm, apply a
random temporal permutation to the visual window at evaluation only, and re-score. No
retraining is required, so the cost is one evaluation pass. Under the interpretation table of
@cha:expdesign, an unchanged score is decisive: the model is not reading temporal
structure in the frames, and every downstream design decision about the visual branch changes.
]

#[
#set par(first-line-indent: 0pt)
*Experiment 2: the modality grid, four training runs.* Target only; target plus
covariates; target plus vision; all three. Identical recipe, identical schedule, one seed each
--- ideally three. This is the experiment that attributes any gain to a specific input rather
than to their combination, and it subsumes several of the single zero-out ablations that would
otherwise be run separately. Four runs is the largest single item on this list and the one most
worth paying for.
]

#[
#set par(first-line-indent: 0pt)
*Experiment 3: retrieval on top of the fusion model.* Add a TS-RAG-style retrieval
head over the frozen backbone of the interleaved arm, with the datastore restricted to
training plants. The two most effective levers identified in this work --- retrieval, worth
0.147 skill over fine-tuning, and multimodal conditioning --- have never been tested together,
and there is no obvious reason they should be redundant: retrieval supplies precedent, vision
supplies the current state of the sky. If they compose additively, the combination would clear
the 0.55 bar.
]

=== If the visual interface is the problem
Should the controls show that the frames carry signal the current encoder does not transmit,
the component to redesign is the summariser, not the fusion operator --- and three directions
are worth naming.

#emph[Widen the bottleneck before redesigning it.] A single 768-dimensional token stands in
for $4 times 196$ patch embeddings (@tab:shapes). Increasing the budget is the
cheapest possible test and should precede any architectural change.

#emph[Give the visual stream a task.] The summariser is currently trained only by the
forecasting loss, which is a weak and distant signal for learning what to extract from a
satellite crop. An auxiliary objective with a direct target --- predicting the next frame's
representation, or the observed clearness index at the site --- would supervise the interface
directly rather than through twelve encoder blocks and a quantile head.

#emph[Condition the horizon, not just the context.] Visual tokens currently enter only the
context region, so their influence on a horizon step is mediated entirely by the temporal
operator. Since cloud advection is a statement about the #emph[future] --- what is upwind now
will be overhead in two hours --- injecting a visual-derived representation at the future
positions is arguably the more natural design, and it is closer to how the oracle-covariate
variants achieve their 0.173 skill advantage. A structured state formulation, in which the
visual stream updates an explicit latent state carried forward across the horizon rather than
contributing context tokens, is the systematic version of this idea and exists in draft.

== Closing remark
<sec:closing>
A thesis is expected to end with the model it proposed. This one ends somewhere else, and the
displacement is the honest summary of what was learned.

The most useful output of this work is not the proposed architecture but the instrument built
to evaluate it. A benchmark that ranks 28 forecasters under one protocol on disjoint plants
makes several comfortable assumptions in the multimodal forecasting literature testable, and
two of them do not survive contact: that pretrained foundation models transfer better than
supervised training on a well-matched architecture, and that consuming real imagery is what
makes a multimodal forecaster work. Whichever way the proposed model's own arms fall, those
two results stand, and they are what a subsequent effort should be designed around.


// =============================================================================
// Bibliography
// =============================================================================
// The LaTeX source used \bibliographystyle{plain}: numeric labels, entries
// sorted alphabetically. Typst's nearest built-in numeric style is "ieee".
#bibliography("biblio.bib", style: "ieee")

// =============================================================================
// Appendices  (was attachments.tex, under \appendix)
// =============================================================================
// \appendix switches chapter numbering to A, B, C… and the template prints
// "Appendix A  Title" rather than the bare number used for chapters.
//
// The appendices are spaced differently from the body, and not by accident: the
// template's `\begingroup` — the one that suppresses `\clearpage` and installs
// the tight `\titlespacing` — closes before `\appendix`. So the appendices get
// book.cls's own chapter layout back: each starts on a fresh page, with 50pt of
// space above the title and 40pt below, and their sections use book's wider
// afterskip rather than the 0.02in the body chapters use. The constants below
// are calibrated against the LaTeX build the same way the body ones are.
#counter(heading).update(0)
#set heading(numbering: "A.1.1")

#show heading.where(level: 1): it => {
  counter(figure.where(kind: image)).update(0)
  counter(figure.where(kind: table)).update(0)
  pagebreak(weak: true)
  // Typst drops a block's `above` at the top of a page; book.cls keeps its
  // \vspace*{50pt} there, so the space is placed explicitly.
  v(76pt, weak: false)
  block(below: 54pt, breakable: false)[
    #set text(size: sz-huge, weight: "bold")
    #set par(justify: false)
    Appendix #counter(heading).display(it.numbering)#h(1em)
    #it.body
  ]
}

#show heading.where(level: 2): it => block(above: 0.20in + lead-in, below: 18.3pt)[
  #set text(size: sz-large, weight: "bold")
  #set par(justify: false)
  #counter(heading).display(it.numbering)#h(0.6em)
  #it.body
]

#set figure(numbering: n => {
  let ch = counter(heading).get().first()
  numbering("A.1", ch, n)
})


= Notation and Glossary
<app:notation>
== Symbols
#figure(
  align(center)[#table(
    columns: 2,
    align: (left,left,),
    table.header([#strong[Symbol];], [#strong[Meaning];],),
    table.hline(),
    [$i$], [Plant (entity) index],
    [$N$], [Number of entities in a batch group],
    [$T$], [History window length in steps (672 for `uk_pv`)],
    [$H$], [Forecast horizon in steps (12 for `uk_pv`)],
    [$T_v$], [Number of visual frames in a window (8)],
    [$T_(upright("ctx"))$], [History length in #emph[tokens] after patching ($ceil.l T \/ 16 ceil.r = 42$)],
    [$T_(upright("fut"))$], [Horizon length in tokens ($ceil.l H \/ 16 ceil.r = 1$)],
    [$T_M$], [Length of the pure-numerical macro region, $T_(upright("ctx")) - n_(upright("vis"))$],
    [$n_(upright("vis"))$], [Visual summary tokens per sample (1 for `uk_pv`)],
    [$L$], [Generic sequence length in tokens],
    [$Q$], [Number of output quantiles (9)],
    [$C_i$], [Installed capacity of plant $i$],
    [$d$], [Model width (768)],
    [$r$], [Grassmann reduced dimension (32)],
    [$delta$], [Grassmann temporal offset, $delta in { 1 \, 2 \, 4 \, 8 \, 12 \, 16 }$],
    [$upright(bold(Y))$], [Historical normalised power, $(N \, T \, 1)$],
    [$upright(bold(X))_(upright("cov"))$], [Covariates over history and horizon, $(N \, T + H \, C)$],
    [$upright(bold(V))$], [Satellite frames, $(N \, T_v \, C_(upright("img")) \, H_(upright("img")) \, W_(upright("img")))$],
    [$hat(upright(bold(Y)))_(upright("fut"))$], [Predicted quantiles, $(N \, H \, Q)$],
    [$upright(bold(z))_i$], [Reduced hidden state at position $i$],
    [$upright(bold(p))_(i \, delta)$], [Normalised Plücker coordinates of the pair $(i - delta \, i)$],
    [$alpha_i$], [Grassmann fusion gate at position $i$],
    [$cal(P)_(\*)$], [Plant sets: $cal(P)_(upright("train"))$,
    $cal(P)_(upright("val"))$, $cal(P)_(upright("test"))$],
  )]
  , caption: [Symbols used throughout. Numerical values in parentheses are the
  `uk_pv` configuration of record.]
  , kind: table
  )
<tab:notation>
== Terms
/ Capacity factor: #block[
Mean output as a fraction of nameplate capacity. Differs across
plants even after normalisation, because of orientation, shading and climate.
]

/ Clearness index ($k_t$): #block[
Ratio of observed to clear-sky irradiance. The quantity
smart persistence carries forward.
]

/ Cross-plant: #block[
An evaluation protocol in which test plants contribute nothing --- not a
row, not a frame, not a normalisation statistic --- to anything fitted.
]

/ Endogenous multimodal: #block[
A method whose second modality is synthesised from the
numerical series or its covariates; no external sensor.
]

/ Exogenous multimodal: #block[
A method that consumes a genuine external observation, here
satellite imagery.
]

/ Group attention: #block[
Attention across the batch axis over token rows sharing a group
identifier; the mechanism by which Chronos-2 ingests covariates.
]

/ In-process metric: #block[
A metric computed inside the training process that produced the
model, as opposed to one derived by reloading a checkpoint.
]

/ Macro region: #block[
The portion of the token sequence that remains purely numerical under
selective interleaving.
]

/ Oracle variant: #block[
A model given observed future weather over the horizon. An upper
bound, never a competitor.
]

/ Ramp: #block[
A large step-to-step change in power, operationally the top decile of
$lr(|Delta y|)$ per site.
]

/ Refinement window: #block[
The recent portion of the sequence into which visual tokens are
interleaved.
]

/ Skill score: #block[
$1 - upright(N R M S E)_(upright("model")) \/ upright(N R M S E)_(upright("ref"))$ with
smart persistence as the reference.
]

/ Smart persistence: #block[
Persistence of the clearness index rather than of power. The
reference model of the solar forecasting community.
]

/ Transductive retrieval: #block[
Retrieval whose datastore contains test-entity history.
Prohibited in the headline protocol.
]

= Plant Splits
<app:splits>
The `uk_pv` partition is generated once with seed 42 at a per-dataset 70/15/15
ratio, excludes sites carrying `bad_site_flag`, and is committed to version control
as the source of truth. Disjointness is asserted programmatically at every data load.

== Train (69 plants)
#quote(block: true)[
, 10367, 10512, 10589, 10630, 10702, 10840, 10843, 11042, 11287, 12495, 12642, 13309,
13311, 13390, 13773, 14394, 14467, 14531, 14649, 14859, 14924, 16216, 16921, 18161, 26811,
26831, 26844, 26846, 26848, 26869, 26879, 26904, 26919, 27054, 3149, 3175, 3333, 3770, 3872,
4090, 6427, 6493, 6618, 6669, 6675, 6827, 6838, 6892, 6966, 6975, 7017, 7051, 7088, 7338,
7359, 7378, 7401, 7412, 7498, 7521, 7533, 7547, 7608, 7651, 7674, 7834, 7836, 9153
]

== Validation (15 plants)
#quote(block: true)[
, 6481, 6498, 6732, 7019, 7356, 7648, 10973, 12826, 13057, 16474, 16769, 18249, 26901,
26970
]

== Test (14 plants)
#quote(block: true)[
, 6648, 7315, 7756, 8066, 9191, 10793, 11176, 13388, 13817, 18989, 26854, 26933, 27020
]

== Excluded
Sites `7239` and `8587` carry `bad_site_flag` and are dropped,
leaving 98 usable plants of 100.

== The `goes_pvdaq` companion split
The committed partition is 7 train (`1202, 1277, 1278, 1283, 1289, 1367, 1420`),
2 validation (`1203, 51`) and 1 test (`36`), additionally rotated
leave-one-plant-out. As noted in @sec:quality, this split predates the
`bad_site_flag` annotation and still assigns flagged sites `1283` and
`51`; it must be regenerated over the eight usable plants before the track is run.

= Scale-Free Aggregation
<app:aggregation>
Cross-plant results are additionally summarised with the rank-preserving statistics of
@sec:aggregation: win rate against smart persistence over the fourteen test
plants, geometric-mean skill, and average rank. These are the only statistics permitted for
aggregation across datasets of differing cadence.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,left,center,center,center,),
    table.header([#strong[Group];], [#strong[Model];], [#strong[Win rate] $arrow.t$], [#strong[SS$""_(upright("geo"))$] $arrow.t$], [#strong[Avg.~rank] $arrow.b$],),
    table.hline(),
    [Reference], [Hourly climatology], [1.000], [0.225], [2.07],
    [], [Seasonal naive], [0.714], [0.100], [3.50],
    [], [Persistence], [0.429], [0.004], [4.29],
    [], [Smart persistence], [0.000], [0.000], [4.14],
    [], [LightGBM], [1.000], [0.379], [1.00],
    [Foundation], [Chronos-2 zero-shot], [0.929], [0.179], [1.07],
    [], [Chronos-2 fine-tuned], [1.000], [0.324], [1.00],
    [], [Chronos-2 oracle], [1.000], [0.469], [1.00],
    [], [Chronos-2 oracle FT], [1.000], [0.498], [1.00],
    [], [TimesFM 2.5 zero-shot], [1.000], [0.265], [1.00],
    [], [TiRex zero-shot], [1.000], [0.282], [1.00],
    [], [TTM-R2 fine-tuned], [1.000], [0.358], [1.00],
    [], [TTM-R2 zero-shot], [0.286], [$- 0.095$], [1.71],
    [Supervised], [iTransformer], [1.000], [0.550], [1.00],
    [], [PatchTST], [1.000], [0.453], [1.00],
    [], [TFT], [1.000], [0.419], [1.00],
    [], [MLP], [1.000], [0.407], [1.00],
    [], [DLinear], [1.000], [0.318], [1.00],
    [], [TabPFN-3], [1.000], [0.332], [1.00],
    [], [CoRA], [1.000], [0.368], [1.00],
    [Multimodal], [Time-VLM], [1.000], [0.562], [1.00],
    [], [Solar-VLM], [1.000], [0.435], [1.07],
    [], [CrossViVit], [1.000], [0.349], [1.87],
    [], [SUNSET], [1.000], [0.236], [3.07],
    [], [Aurora], [1.000], [0.223], [2.00],
    [], [UniCast], [0.786], [0.114], [4.00],
    [], [VisionTS++], [0.429], [0.004], [3.57],
  )]
  , caption: [Win rate, geometric-mean skill and average rank against smart persistence over the
  fourteen `uk_pv` test plants. Ranks are computed within each model's own comparison
  group, so they are comparable within a block and not across blocks.]
  , kind: table
  )
<tab:aggregation>
A win rate of 1.000 with a modest geometric-mean skill --- Chronos-2 fine-tuned, for
instance --- indicates a model that beats the reference on #emph[every] plant by a moderate
margin, which is a different and more trustworthy profile than a high mean driven by a few
easy plants. TTM-R2 zero-shot is the only model to lose to the reference on a majority of
plants.

= Ramp-Regime Results
<app:ramp>
Ramp metrics restrict evaluation to the top decile of $lr(|Delta y|)$ per site --- the
cloud-transition periods where a multimodal model should have its advantage. As noted in
@sec:threats, the ramp mask for externally integrated baselines is recomputed
from saved predictions on their native windows and is therefore not bit-aligned with the
tier~0--3 subset; read within-tier or by rank.

#figure(
  align(center)[#table(
    columns: 4,
    align: (left,left,center,center,),
    table.header([#strong[Tier];], [#strong[Model];], [#strong[Ramp NMAE] $arrow.b$], [#strong[Ramp NRMSE] $arrow.b$],),
    table.hline(),
    [T5], [Time-VLM], [0.1374], [#strong[0.1720];],
    [T2], [iTransformer], [0.1432], [0.1769],
    [T3], [Chronos-2 oracle FT], [0.1494], [0.1824],
    [T6], [Solar-VLM], [0.1512], [0.1849],
    [T2], [PatchTST], [0.1543], [0.1888],
    [T3], [Chronos-2 oracle], [0.1544], [0.1915],
    [T4], [Cross-RAG], [0.1545], [0.1969],
    [T4], [TS-RAG], [0.1568], [0.2004],
    [T2], [TFT], [0.1620], [0.2001],
    [T2], [MLP], [0.1627], [0.1981],
    [T4], [CoRA], [0.1653], [0.2021],
    [T0], [Hourly climatology], [0.1665], [0.2037],
    [T1], [LightGBM], [0.1668], [0.2024],
    [T1], [TabPFN-3], [0.1694], [0.2050],
    [T3], [TTM-R2 fine-tuned], [0.1702], [0.2062],
    [T6], [CrossViVit], [0.1720], [0.2100],
    [T2], [DLinear], [0.1740], [0.2092],
    [T3], [Chronos-2 fine-tuned], [0.1759], [0.2115],
    [T6], [SUNSET], [0.1757], [0.2177],
    [T3], [TiRex zero-shot], [0.1826], [0.2233],
    [T3], [Chronos-2 zero-shot], [0.1902], [0.2335],
    [T3], [TimesFM 2.5 zero-shot], [0.1902], [0.2319],
    [T5], [VisionTS++], [0.2008], [0.2515],
    [T5], [Aurora], [0.2091], [0.2516],
    [T0], [Seasonal naive], [0.2056], [0.2575],
    [T5], [UniCast], [0.2302], [0.2814],
    [T0], [Smart persistence], [0.2317], [0.2806],
    [T0], [Persistence], [0.2550], [0.2990],
  )]
  , caption: [Ramp-subset accuracy, sorted by ramp NMAE. Time-VLM --- which consumes no external
  imagery --- attains the best ramp NRMSE in the suite.]
  , kind: table
  )
<tab:ramp>
= Configuration of Record
<app:config>
All hyperparameters are composed by Hydra from four groups --- `model`, `data`,
`trainer` and `stage` --- and none is hardcoded in model code. The following are
the settings that define the reported configuration.

== Backbone
Chronos-2 at native checkpoint size: $d_(upright("model")) = 768$, twelve layers, twelve heads,
head dimension 64, feed-forward width 3072, input patch size 16, nine quantiles
$[0.1 \, 0.2 \, dots.h \, 0.9]$, inverse-hyperbolic-sine normalisation. As stated in
@sec:numeric-branch, configured dimensions smaller than the checkpoint's are
silently ignored on load.

== Grassmann mixer
Reduced dimension 32 (must be even for rotary embedding); offsets ${ 1 \, 2 \, 4 \, 8 \, 12 \, 16 }$;
Plücker normalisation epsilon $10^(- 8)$; modality-pair bias enabled by default when the
fusion mode is interleaved.

== Vision
V-JEPA~2.1 ViT-L/16, frozen; eight frames per window; tubelet temporal stride two, giving
four latent time steps of 196 patches at width 1024; latent summariser output
$n_(upright("vis")) = 1$ token for `uk_pv` and 2 for `goes_pvdaq`, derived from
cadence and asserted at model initialisation to be bounded by the context length.

== Windows
History fourteen days; horizon six hours; decay horizons reported at one, six and
twenty-four hours; visual window six hours ending at the forecast origin.

== Optimisation
Masked pinball loss over the nine quantiles; asymmetric Bernoulli modality dropout; global
seed 42; bfloat16 mixed precision; distributed data parallel; early stopping on validation
loss. The vision-free arm uses batch size 2 with gradient accumulation 8 for the reason given
in @sec:impl.

= Baseline Integration Notes
<app:integration>
Third-party models were vendored and adapted to the tensor contract of
@sec:contract. The deviations below bound how strongly each row of
@tab:leaderboard may be read.

- #strong[Tier 2] (MLP, DLinear, PatchTST, iTransformer, TFT) is served by one port of
  a common deep time-series library~@tslib; the four transformer-family models share
  a trainer and differ only in architecture. TFT is a reduced variant retaining the
  quantile head and gating but not the full variable-selection stack.

- #strong[TTM] uses the R2 checkpoint. The R3 release is a trend-plus-residual
  decomposition model whose parameter keys do not load into the available library version,
  yielding silently randomised weights. Histories shorter than the model's context are
  edge-padded with an observed mask; fine-tuned variants receive a frequency token.

- #strong[Chronos-2 oracle variants] differ from the honest variants only in which
  covariates are exposed over the horizon: the honest variants receive deterministic
  quantities (solar geometry, calendar), the oracle variants additionally receive observed
  future weather. They are upper bounds.

- #strong[Retrieval baselines] (TS-RAG, Cross-RAG) populate their datastores with
  training-plant windows only; mixing weights are tuned on validation plants. Cross-RAG
  additionally uses clear-sky-aware retrieval keys and per-step mixing.

- #strong[CoRA] is implemented as a zero-initialised residual adapter over the frozen
  backbone.

- #strong[CrossViVit] runs with single-channel imagery and synthesised coordinate
  encodings, since `uk_pv` frames are greyscale and the original model expects
  multi-channel input with precise geospatial coordinates.

- #strong[SUNSET] is run from the public implementation against the packed frame
  archive.

- #strong[Solar-VLM] generates its weather text only from covariates available to every
  other model; no external weather service is consulted.

- #strong[Time-VLM] runs on evaluation windows not bit-aligned with the rest of the
  suite. An earlier reported catastrophic score for this model was traced to a stale
  artefact --- de-standardised predictions saved without regenerating the metric file ---
  and corrected by re-scoring, without retraining. The other externally integrated models
  re-score bit-identically to their recorded rows, so the correction is isolated.

- #strong[UniCast] re-scores bit-identically to its record. Its weakness is genuine:
  amplitude under-prediction of approximately $0.61 thin y + 0.063$ with correlation 0.61, plus
  a late-day skew in its native evaluation windows. Neither is recoverable without
  retraining or an illegitimate recalibration fitted on targets.

= Experiment Registry
<app:registry>
Every experiment is registered before it launches, with a one-sentence hypothesis, a
configuration diff, a branch and a nominated comparison baseline. The registry is the
authoritative record of what was run; training logs are never used to reconstruct history.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,left,left,),
    table.header([#strong[ID];], [#strong[Hypothesis];], [#strong[Status];],),
    table.hline(),
    [A00], [Chronos-2 zero-shot establishes the foundation-model anchor], [done],
    [A01], [Late fusion improves over time-series only (H1)], [done],
    [A02], [Interleaved fusion improves over late fusion (H2)], [in progress],
    [A03], [Grassmann mixing improves over matched self-attention], [in progress],
    [A04], [Visual window length: 3 vs 6 vs 12 hours], [specified],
    [A05], [Cross-plant held-out --- folded into the protocol, not an ablation], [folded],
    [A06], [Few-shot context matching], [deprecated by design decision],
    [A07], [TS-RAG on a frozen backbone (H4)], [done],
    [A08], [Cross-RAG against TS-RAG], [done],
    [A09], [Shuffled-frames control: the frames are read temporally], [specified],
    [A10], [Mismatched-plant frames control: spatial grounding], [specified],
    [A11], [Vision-only upper bound], [specified],
    [A12], [Modality-contribution grid (target / +covariates / +vision / all)], [specified],
    [A13], [Visual token budget sweep], [specified],
    [A14], [Frozen against partially unfrozen backbone], [specified],
    [A15], [Retrieval datastore size and top-$k$ sweep], [specified],
    [W4], [Cross-plant group batching activates group attention at train time], [code complete],
    [W5], [Bounding the visual window and supplying true frame intervals improves sub-hourly skill], [code complete],
    [W6], [Dual-pass marginal-gain measurement confirms the visual stream is used], [code complete],
    [W7], [The visual token count is derived from cadence and bounded by context length], [done],
  )]
  , caption: [Experiment registry. "Specified" means designed and registered but not executed.]
  , kind: table
  )
<tab:registry>
= Covariate Reference
<app:covariates>
The fourteen covariate channels supplied to models fall into two classes with very different
status at forecast time. #emph[Observed] covariates are measurements or reanalysis products
that are known only up to the forecast origin; over the horizon they must themselves be
forecast, and supplying their true future values constitutes privileged information.
#emph[Deterministic] covariates are computable analytically for arbitrary future times from
coordinates and a calendar, and may therefore be supplied over the horizon without
compromising the protocol.

#figure(
  align(center)[#table(
    columns: 4,
    align: (left,left,left,left,),
    table.header([#strong[Channel];], [#strong[Unit];], [#strong[Meaning];], [#strong[Class];],),
    table.hline(),
    [`temperature_2m`], [$""^circle.stroked.tiny$C], [Air temperature at 2 m; affects module efficiency], [observed],
    [`shortwave_radiation`], [W/m$""^2$], [Global horizontal irradiance at the surface], [observed],
    [`direct_radiation`], [W/m$""^2$], [Beam component of surface irradiance], [observed],
    [`diffuse_radiation`], [W/m$""^2$], [Scattered component], [observed],
    [`direct_normal_irradiance`], [W/m$""^2$], [Beam irradiance normal to the sun], [observed],
    [`cloudcover`], [%], [Total areal cloud fraction], [observed],
    [`windspeed_10m`], [m/s], [Wind at 10 m; module cooling and advection speed], [observed],
    [`precipitation`], [mm], [Accumulated precipitation], [observed],
    [`solar_zenith`], [deg], [Angle between the sun and the vertical], [deterministic],
    [`solar_azimuth`], [deg], [Compass bearing of the sun], [deterministic],
    [`clearsky_ghi`], [W/m$""^2$], [Haurwitz clear-sky global horizontal
    irradiance~@haurwitz], [deterministic],
    [`doy_sin`, `doy_cos`], [---], [Cyclic day-of-year encoding], [deterministic],
    [`solar_time`], [h], [Local apparent solar time], [deterministic],
  )]
  , caption: [The covariate track. Only the deterministic class may be supplied over the forecast
  horizon; the honest and oracle variants of the foundation-model baselines differ precisely in
  whether the observed class is also supplied.]
  , kind: table
  )
<tab:covariates>
Two derived quantities appear in the table but are not fed to models as covariates. The
clearness index $k_t$ is the ratio of observed to clear-sky irradiance, and the clear-sky
index is its power analogue; both are used by smart persistence, which is the physics
reference and is exempt from the no-domain-physics rule, and by Cross-RAG for retrieval keys.
The clear-sky index is left undefined below 50 W/m$""^2$ of clear-sky irradiance, where the
ratio is numerically unstable.

= Metric Definitions
<app:metrics>
All metrics are computed on the capacity-normalised target over daylight, valid steps, per
plant, then macro-averaged. Let $y_(i \, h)$ and $hat(y)_(i \, h)$ be truth and prediction for
sample $i$ at horizon step $h$, $C_i$ the plant capacity, $M$ the number of samples and $H$
the horizon.

#[
#set par(first-line-indent: 0pt)
*Normalised mean absolute error.* $ upright(N M A E) = frac(1, M H) sum_(i = 1)^M sum_(h = 1)^H frac(\| hat(y)_(i \, h) - y_(i \, h) \|, C_i) . $
]

#[
#set par(first-line-indent: 0pt)
*Normalised root mean squared error.* $ upright(N R M S E) = sqrt(frac(1, M H) sum_(i = 1)^M sum_(h = 1)^H (frac(hat(y)_(i \, h) - y_(i \, h), C_i))^(#h(-1em) 2)) . $
]

#[
#set par(first-line-indent: 0pt)
*Skill score.* $ upright(S S) = 1 - upright(N R M S E)_(upright("model")) / upright(N R M S E)_(upright("smart persistence")) \, $
computed within the same scenario tag, so that a model is always compared against the
reference evaluated on exactly the windows it was itself evaluated on.
]

#[
#set par(first-line-indent: 0pt)
*Pinball loss and CRPS.* For quantile level $tau$ and prediction
$hat(y)^((tau))$,
$ cal(L)_tau (y \, hat(y)^((tau))) = cases(delim: "{", tau thin (y - hat(y)^((tau))) & upright("if ") y gt.eq hat(y)^((tau)) \,, (1 - tau) thin (hat(y)^((tau)) - y) & upright("otherwise.")) $
The continuous ranked probability score is approximated by averaging over the nine-point
quantile grid $tau in { 0.1 \, dots.h \, 0.9 }$. This is a strictly proper scoring
rule~@gneiting up to the discretisation of the grid.
]

#[
#set par(first-line-indent: 0pt)
*Coverage.* The empirical fraction of observations falling inside the nominal
central 80% interval $[hat(y)^((0.1)) \, hat(y)^((0.9))]$. A well-calibrated model reports
0.80; below indicates overconfidence, above indicates excessive width.
]

#[
#set par(first-line-indent: 0pt)
*Variance tracking.* $R^2 = upright(c o r r) (y \, hat(y))^2$ pooled over daylight,
valid steps. Note that this is the squared correlation, not the coefficient of
determination: it is invariant to affine rescaling of the forecast and therefore isolates
#emph[shape] tracking from bias and scale, which is exactly the separation
@sec:qualitative relies on.
]

#[
#set par(first-line-indent: 0pt)
*Ramp metrics.* NMAE and NRMSE restricted to the top decile of $\| Delta y \|$ per site, computed on the site's native windows.
]

#[
#set par(first-line-indent: 0pt)
*Scale-free aggregates.* Win rate is the fraction of test plants on which a model
beats the reference. Geometric-mean skill is
$1 - (product_j upright(N R M S E)_(upright("model") \, j) \/ upright(N R M S E)_(upright("ref") \, j))^(1 \/ J)$ over $J$ plants. Average rank is the mean position across plants within a
comparison group.
]

= Compute Environment
<app:compute>
== Hardware and scheduling
Training and evaluation run on the Leonardo supercomputer under SLURM. Jobs request GPU nodes;
standard output and error are redirected per job and retained, so that every number in this
thesis traces to a job identifier. Long training runs are checkpointed and resumed on
walltime expiry, a practice that interacts with the integrity defect of
@sec:integrity and is discussed there.

Data lives on the project's fast scratch filesystem as a single Parquet table plus a single
27 GB HDF5 archive. The choice of one packed archive over per-frame files is not incidental:
a shared parallel filesystem degrades badly under millions of small-file opens, and a
per-frame layout made the visual data loader the training bottleneck in early experiments.

== Practices imposed by the environment
- #strong[Storage discipline.] Project scratch is quota-bound, and an unattended
  cache can exhaust it. Visual feature caches are written once and reused across runs
  rather than regenerated per job, and intermediate artefacts are pruned on a schedule.

- #strong[Stride subsampling.] Training windows are drawn at a stride rather than at
  every timestep. Consecutive origins share almost all of their context, so the marginal
  information of a stride-1 sample is small while its cost is not.

- #strong[Gradient accumulation over batch size.] Where the memory profile forbids a
  useful batch --- notably the vision-free arm, for the reason given in
  @sec:impl --- the effective batch is recovered by accumulation rather than
  by reducing the token budget.

- #strong[Mixed precision.] Training uses bfloat16 with distributed data parallelism.

- #strong[No data in version control.] The repository contains code, configuration and
  prose; datasets, checkpoints and logs live outside it and are referenced by path.

== Software
Dependencies are managed with a single lockfile and a single resolver, so that an environment
can be reconstructed exactly. Configuration is composed by Hydra~@hydra; the training
lifecycle is handled by PyTorch Lightning~@lightning, which supplies distributed
training, mixed precision and checkpointing without entangling them with the model definition.
Tier-2 baselines derive from a common deep time-series library~@tslib; all other
third-party models are vendored at a pinned revision.

= Reproducibility Checklist
<app:repro>
== Data
- The dataset of record is a single Parquet table plus a single HDF5 archive,
  treated as read-only. No result depends on re-processing it.

- Splits are generated once with seed 42 and #emph[committed to version control];;
  they are never regenerated at run time.

- Plant disjointness between train, validation and test is asserted
  programmatically at every data load. A leak raises rather than silently improving a
  score.

- Frames are addressed exclusively through `image_h5_index`, a group-local,
  timestamp-exact pointer verified row by row.

- The target is normalised by audited nameplate capacity --- a quantity known at
  installation --- so no normalisation statistic is estimated from evaluation data.

== Models
- Global seed 42 for data loaders, weight initialisation and dropout.

- Configuration is Hydra-only; each baseline's hyperparameters are self-contained in
  its own directory, so no model's behaviour depends on a global setting another model
  also reads.

- Third-party models are vendored at a pinned revision and adapted to the tensor
  contract rather than reimplemented. Every deviation from the published method is
  recorded in @app:integration.

- Foundation models with a fixed maximum context consume their most recent supported
  window; this is stated per model rather than silently tolerated.

== Experiments
- Every experiment is registered before it launches, with a one-sentence hypothesis,
  a configuration diff, a branch and a nominated comparison baseline
  (@app:registry).

- Training runs under SLURM with per-job logs retained, so any reported number traces
  to the job that produced it.

- Only in-process metrics are reported. Post-hoc re-scoring from checkpoints is
  blocked by the defect of @sec:integrity and no number in this thesis is
  derived that way.

- Interpretations of control outcomes are fixed in advance
  (@tab:control-logic).

== Known gaps
- One seed per configuration; seed-to-seed variance is unmeasured.

- No Diebold--Mariano or equivalent significance testing.

- The `goes_pvdaq` split predates the bad-site flags and must be regenerated
  before that track is run.

- Ramp masks for externally integrated baselines are recomputed on their native
  windows and are not bit-aligned with the tier~0--3 subset.

- Two baseline rows are harness-limited and flagged as indicative
  (@app:integration).
