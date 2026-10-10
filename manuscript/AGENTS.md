# Typst-to-LaTeX Thesis Migration

The repository contains an existing scientific thesis written in Typst and a department-provided LaTeX thesis template.

When the requested task is thesis migration, follow this section.

---

## Required files

Before starting or resuming:

1. Read `AGENTS.md`.
2. Read `latex-migration/steps.md`.
3. Inspect `surname_name_mastercourse_ay.tex`.
4. Inspect the existing department template structure.
5. Read the relevant Typst source files.
6. Use `latex-migration/steps.md` as the authoritative progress tracker.
7. Write operational information to:
   `latex-migration/migration-log.md`.

---

## Authority hierarchy

### Content authority

The Typst thesis is authoritative for:

- prose;
- section titles;
- paragraph order;
- claims;
- terminology;
- equations;
- table contents;
- captions;
- citations;
- scientific structure.

### Format authority

The department LaTeX template is authoritative for:

- page layout;
- document class;
- fonts;
- spacing;
- margins;
- headings;
- chapter appearance;
- title pages;
- headers/footers;
- captions;
- bibliography style;
- table presentation;
- figure presentation;
- front matter;
- back matter.

Never import Typst visual style into the LaTeX thesis.

---

## Root document

The only thesis entry point is:

`surname_name_mastercourse_ay.tex`

Do not create a replacement root file.

Do not create an alternative LaTeX template.

Integrate all migrated content into the project rooted at this file.

---

## Content preservation

Migration must not modify thesis content.

Do not:

- rewrite;
- paraphrase;
- proofread;
- improve grammar;
- shorten;
- expand;
- reorder;
- humanize;
- simplify;
- normalize terminology;
- change claim strength;
- change technical wording.

Only syntax changes required for valid LaTeX are allowed.

---

## No converter scripts

Do not create conversion scripts or automated translation pipelines.

Forbidden approaches include:

- Python converter scripts;
- regex migration programs;
- Pandoc pipelines;
- shell-based document translators;
- custom Typst parsers;
- generated intermediate translators.

Directly migrate the content into LaTeX.

---

## Structural migration

Preserve the semantic hierarchy of:

- chapters;
- sections;
- subsections;
- subsubsections.

Use the hierarchy supported by the template.

Do not change hierarchy for visual reasons.

---

## Paragraph fidelity

Preserve:

- paragraph boundaries;
- sentence order;
- wording;
- punctuation where representable;
- technical terms;
- numerical values;
- citations.

LaTeX escaping is not a content modification.

---

## Typst formatting

Do not reproduce Typst formatting unless it encodes semantic content.

Ignore Typst-specific:

- fonts;
- margins;
- heading styling;
- spacing;
- color;
- page-layout logic;
- decorative elements.

Use the LaTeX template instead.

---

## Equations

Translate Typst math syntax into equivalent LaTeX math syntax.

Preserve mathematical meaning exactly.

Never simplify or reformulate equations.

Check:

- operators;
- subscripts;
- superscripts;
- indices;
- vectors;
- sets;
- quantiles;
- conditions;
- alignment;
- equation labels.

---

## Tables

Recreate tables using LaTeX constructs compatible with the department template.

Preserve exactly:

- values;
- rows;
- columns;
- labels;
- units;
- captions;
- scientific ordering.

Do not reproduce Typst table styling.

---

## Figures

Reuse existing figure assets.

Use LaTeX figure syntax and the department template's conventions.

Preserve:

- figure identity;
- caption content;
- logical placement;
- references.

Do not edit image content.

---

## Citations

Use the bibliography mechanism already configured by the LaTeX template.

Do not replace it unnecessarily.

Preserve:

- cited source;
- citation location;
- citation scope;
- citation key where feasible.

Do not add or remove sources.

---

## Cross-references

Convert references semantically.

All references must continue to point to the same:

- section;
- figure;
- table;
- equation;
- appendix;
- chapter.

---

## Special characters

Escape LaTeX-special characters in prose as needed.

Do not apply blind escaping to:

- mathematics;
- LaTeX commands;
- citation keys;
- labels;
- paths.

---

## Package rule

Use existing template packages whenever possible.

Do not add packages to imitate Typst appearance.

New packages require a functional need for representing existing thesis content.

Any added package must be documented in the migration log.

---

## Direct-edit rule

The migration result must exist in actual `.tex` files used by:

`surname_name_mastercourse_ay.tex`

Do not merely produce converted content in notes, chat, or temporary files.

---

## Typst source protection

Typst source files are READ ONLY.

Never modify the original thesis during migration.

---

## Verification rule

After every migrated structural unit, compare source and destination.

Check:

- no text omitted;
- no text added;
- no text rewritten;
- sentence order preserved;
- paragraph order preserved;
- numbers preserved;
- citations preserved;
- equations equivalent;
- tables equivalent;
- references equivalent.

---

## Chapter completion rule

A chapter is complete only when:

- all prose is migrated;
- all headings are migrated;
- all equations are migrated;
- all tables are migrated;
- all figures are migrated;
- all citations are migrated;
- all cross-references are checked;
- content fidelity passes;
- template conventions are respected;
- compilation succeeds where available.

---

## Build failures

Fix build failures through LaTeX syntax and template-compatible structure.

Never delete or rewrite thesis content merely to make the document compile.

---

## Operational persistence

Use:

`latex-migration/steps.md`

as the source of truth.

Update it after meaningful completed units.

A checkpoint is a recovery mechanism.

Do not stop because progress was saved.

Continue until the complete migration scope is finished.