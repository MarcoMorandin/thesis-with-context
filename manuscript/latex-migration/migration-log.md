# Migration Log

## Template inspection

- Root entry point: `latex-thesis/surname_name_mastercourse_ay.tex` (the requested filename is nested in the department template directory).
- Document class: `book`, A4, 11 pt, two-sided; template packages and geometry retained.
- Front matter: `front.tex`, followed by optional `acknowledgements.tex`; abstract is included through `abstract.tex` after the table of contents.
- Chapter insertion: sequential `\input{chapterN}` commands inside the template's existing chapter-formatting group.
- Bibliography: BibTeX with `plain`; root now points to the authoritative `../biblio.bib`.
- Figures: `epsfig` is supplied by the template; thesis raster/vector assets remain in `../figures/` and are included from chapter files.
- Tables: standard LaTeX `tabular` is available; no table-format package added.
- Mathematics: `amsmath` is added for the source cases equation; other display math uses standard LaTeX environments.
- Appendices: the template's optional appendix mechanism remains; no appendix is present in the current Typst thesis.
- Build instructions: `pdflatex`, `bibtex`, then two further `pdflatex` runs, as stated in the root template. Available local tools include `pdflatex`, `bibtex`, and `latexmk`.
- Existing sample chapter, abstract, acknowledgement, bibliography, and attachment text are template placeholders; thesis content replaces applicable placeholders.
- Package added: `amsmath`, required to represent the source pinball-loss cases equation and its text conditions; no visual styling packages were added.

## Chapter-by-chapter fidelity review

| Chapter | Typst source | LaTeX destination | Status | Prose and structure | Content blocks | References |
|---|---|---|---|---|---|---|
| CH01 | `chapters/01-introduction.typ` | `latex-thesis/chapter1.tex` | VERIFIED | PASS | Equations PASS; no tables/figures | PASS |
| CH02 | `chapters/02-background-and-related-work.typ` | `latex-thesis/chapter2.tex` | VERIFIED | PASS | Figures/captions PASS | PASS |
| CH03 | `chapters/03-dataset-construction.typ` | `latex-thesis/chapter3.tex` | VERIFIED | PASS | Tables, figures/captions PASS | PASS |
| CH04 | `chapters/04-method-fusion-ladder.typ` | `latex-thesis/chapter4.tex` | VERIFIED | PASS | Equations, tables, figures/captions PASS | PASS |
| CH05 | `chapters/05-evaluation-protocol.typ` | `latex-thesis/chapter5.tex` | VERIFIED | PASS | Tables, figures/captions PASS | PASS |
| CH06 | `chapters/07-benchmark-results.typ` | `latex-thesis/chapter6.tex` | VERIFIED | PASS | Tables, figures/captions PASS | PASS |
| CH07 | `chapters/08-results-and-ablations.typ` | `latex-thesis/chapter7.tex` | VERIFIED | PASS | Tables, figures/captions PASS | PASS |
| CH08 | `chapters/09-discussion.typ` | `latex-thesis/chapter8.tex` | VERIFIED | PASS | Tables/captions PASS | PASS |
| CH09 | `chapters/10-conclusions.typ` | `latex-thesis/chapter9.tex` | VERIFIED | PASS | No tables/figures | PASS |

Review findings: prose wording, sentence order, paragraph order and heading hierarchy match after markup conversion. Two LaTeX paragraph breaks incorrectly split source paragraphs; these were corrected in CH03 and CH06. Equations, table contents and ordering, figure assets and captions, citations, labels, and logical reference targets were checked against the Typst sources. Front-matter title fields, abstract, and acknowledgement placeholder match the source.

Compilation: PASS; `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/thesis-latex-standalone-build surname_name_mastercourse_ay.tex` completed after review fixes. Final PDF: 64 pages, A4. The final log has no undefined references/citations, missing-file errors, or fatal LaTeX errors.

## Migration Summary

- Root LaTeX file: `latex-thesis/surname_name_mastercourse_ay.tex`.
- Typst sources represented: nine current chapter files, in their source order.
- LaTeX content destinations: `chapter1.tex` through `chapter9.tex`.
- Equations, tables, figures, citations, and references: migrated and reviewed against the current Typst sources.
- Package added: `amsmath` for equation syntax.
- Build: PASS using `latexmk`; 64-page A4 PDF staged outside the repository.
- Unresolved: none in migration syntax, references, citations, or assets. Existing bibliography metadata warnings are noted above; source records were left unchanged.
- Fidelity status: PASS; `steps.md` marks the migration COMPLETE.

## Standalone LaTeX assets

- Copied the complete `figures/` directory into `latex-thesis/figures/`, preserving its contents and subdirectories.
- Updated all 38 figure inclusions to use `figures/<asset>` paths relative to the LaTeX project directory.
- Confirmed the root uses the local `biblio.bib` file.
- Rebuilt from `latex-thesis/` into `/tmp/thesis-latex-standalone-build`; build passed and produced a 64-page A4 PDF with no unresolved references, citations, missing-file errors, or LaTeX errors in the final log.

## Chapter page-break behavior

- Removed manual `\\pagebreak` and `\\par` commands between chapter inputs and restored the `book` class's normal chapter page break (`openany` keeps chapters from forcing blank verso pages).
- Added `\\raggedbottom` so LaTeX leaves natural white space at the bottom of pages instead of stretching paragraph spacing to align page bottoms.
- Rebuilt successfully to `/tmp/thesis-latex-standalone-build`; the PDF is 70 A4 pages. Confirmed all nine chapter headings begin on separate pages. No unresolved citations/references, missing assets, or fatal LaTeX errors; 18 existing BibTeX metadata warnings remain.

## Figure Placement and Non-floating Fix

- Added `float` package to `latex-thesis/surname_name_mastercourse_ay.tex` to provide the `[H]` placement specifier.
- Updated all figure environments in `front.tex`, `chapter2.tex`, `chapter3.tex`, `chapter4.tex`, and `chapter6.tex` from `[htbp]` / `[h!]` to `[H]`.
- Content preservation: no text, captions, labels, equations, numbers, or figure assets were modified.
- All figures are now fixed in the text flow at their exact narrative locations, preventing them from floating around.
- Rebuilt successfully: 71-page A4 PDF with 0 errors, all citations and references resolved.
- Explicitly enabled `\raggedbottom` in `latex-thesis/surname_name_mastercourse_ay.tex` (preamble, `\mainmatter`, and chapter group) to prevent LaTeX from stretching vertical inter-paragraph spacing to satisfy `\flushbottom`, keeping paragraph spacing compressed and leaving empty space at the bottom of pages before figures.

## Caption Styling Enhancement

- Added `caption` package with options `[font=small,labelfont=bf,margin=0.8cm]` to `latex-thesis/surname_name_mastercourse_ay.tex`.
- Figures and tables now feature bold labels (e.g., **Figure 6.1:**, **Table 6.10:**), a smaller font size (`small` / 10 pt vs. 11 pt body prose), and 0.8 cm left/right margins, making them immediately distinguishable from surrounding paragraphs.
- Clean build: 71-page A4 PDF with 0 errors.

