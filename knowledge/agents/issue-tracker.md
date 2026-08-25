# Issue tracker — local markdown

**Canonical for**: where the `mattpocock-skills` engineering skills read and write
issues. Consumer rules for domain docs live in [domain.md](domain.md).

Issues and specs live as markdown files in `.scratch/`. This repo has a GitHub remote
(`MarcoMorandin/thesis-with-context`) but **no `gh` CLI installed** and no history of
tracking work in GitHub Issues, so the GitHub path is not usable here.

`.scratch/` is working state, not prose of record — it is exempt from the `knowledge/`
single-source rule ([../../AGENTS.md](../../AGENTS.md) §5). A finding that outlives its
ticket gets promoted into `knowledge/`, and the ticket links to it.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01` — never a single
  combined tickets file
- Triage state is a `Status:` line near the top of each issue file
- Comments and conversation history append to the bottom under a `## Comments` heading

Triage labels are **not configured**: the `triage` skill is not installed. If it is
installed later, re-run `/mattpocock-skills:setup-matt-pocock-skills` to add
`knowledge/agents/triage-labels.md`.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`, creating the directory if needed.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue
number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body)
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with
  the question in the body. A `Type:` line records the ticket type
  (`research` / `prototype` / `grilling` / `task`); a `Status:` line records
  `claimed` / `resolved`
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when
  every file it lists is `resolved`
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked and
  unclaimed; first by number wins
- **Claim**: set `Status: claimed` and save before any work
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`,
  then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`
