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

TBD

#pagebreak()

// =============================================================================
// Contents
// =============================================================================
#set page(numbering: "i")
#counter(page).update(1)

#outline(title: "Contents", indent: auto, depth: 3)

#pagebreak()



// ==========================================================================
// Abstract
// ==========================================================================

#heading(level: 1, numbering: none)[Abstract]

This thesis studies forecasting for photovoltaic plants excluded from model fitting, where prediction must rely on a short local history and knowledge transferred from other plants. It constructs an aligned corpus of power, numerical context, and satellite imagery from UK and US sites, then compares a numerical time-series foundation model with visual fusion interfaces. On the UK test plants, spatial-token fusion reaches 55.10% skill and 0.1440 ramp NMAE, compared with 52.30% skill and 0.1506 ramp NMAE for the adapted numerical control. Disabling vision in the trained spatial-token model raises ramp NMAE by 0.0063; stale and wrong-plant image substitutions also degrade performance. An inference-only evaluation on one held-out US plant offers an initial cross-track test. Spatial-token and reduced-payload multi-anchor fusion have the highest skill scores among evaluated arms, while visual reliance varies by interface. With one test plant, the results provide no population-level estimate for US installations, and a cloud-motion mechanism remains unestablished. Spatial-token fusion's 80% interval coverage is below nominal despite its point-forecast gains.

#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)

// =============================================================================
// Chapters
//
// Each chapter is one file under chapters/. Order here is the order in the
// document; heading, figure, table and citation numbering all continue across
// includes exactly as they did in the single-file source.
// =============================================================================

#include "chapters/01-introduction.typ"
#include "chapters/02-background-and-related-work.typ"
#include "chapters/03-dataset-construction.typ"
#include "chapters/04-method-fusion-ladder.typ"
#include "chapters/05-evaluation-protocol.typ"
#include "chapters/07-benchmark-results.typ"
#include "chapters/08-results-and-ablations.typ"
#include "chapters/09-discussion.typ"
#include "chapters/10-conclusions.typ"



#pagebreak()
#bibliography("biblio.bib", style: "ieee")
