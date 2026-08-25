# Triage labels

**Canonical for**: the label vocabulary the `triage` skill uses. Where issues live is in
[issue-tracker.md](issue-tracker.md); domain-doc rules are in [domain.md](domain.md).

The skills speak in terms of five canonical triage roles. This file maps those roles to
the strings actually used in this repo's tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Maintainer needs to evaluate this issue |
| `needs-info` | `needs-info` | Waiting on reporter for more information |
| `ready-for-agent` | `ready-for-agent` | Fully specified, ready for an AFK agent |
| `ready-for-human` | `ready-for-human` | Requires human implementation |
| `wontfix` | `wontfix` | Will not be actioned |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the
corresponding string from the right-hand column.

## How these are applied here

The tracker is local markdown, not a labelled system — there is nothing to create and no
`gh label` to call. A role is recorded as a `Status:` line near the top of the ticket
file, per [issue-tracker.md](issue-tracker.md):

```markdown
# 03 — Store the vision-off pass in ProtocolEvaluator

Status: ready-for-agent
Type: task
Blocked by: 01
```

`Status:` carries the triage role. `wayfinder` additionally uses `claimed` / `resolved`
on its decision tickets; the two vocabularies do not collide, because a wayfinder ticket
is claimed or resolved and a triaged issue carries one of the five roles above.

Edit the right-hand column if the vocabulary ever changes.
