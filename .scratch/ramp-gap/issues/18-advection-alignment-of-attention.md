# 18 — Does per-tau attention move with the cloud field?

Type: task
Status: blocked
Blocked by: 16

## Question

If the three horizon queries attend to different regions, do those regions displace in a
direction consistent with observed cloud motion?

## Why it is worth a ticket

This is the difference between "the queries differ" and "the queries are doing advection".
If the attention centroid for +120 min sits further upwind than the centroid for +30 min,
along the motion estimated from the frames themselves, that is direct mechanistic evidence
that the architecture learned the physical structure of the problem — the strongest possible
version of this chapter's claim.

If the maps differ wildly but bear no relation to cloud structure, that is also informative:
the queries specialised on something other than advection.

## Method

Estimate cloud motion vectors from the same 8 frames the latents were extracted from
(optical flow, e.g. Farneback, on the 128 px crops — CPU-only, no GPU, no encoder). Compare
the displacement of the per-tau attention centroid against the CMV-implied displacement at
each lead time. Report correlation, not a binary verdict.

The model is **not** forced to obey optical flow; this is a read-out, not a loss.

## Note on the wider CMV question

Several reviewers argued explicit CMV should replace the learned pathway. That pivot is
recorded in the map's fog, not here: this ticket uses optical flow only as a *measuring
instrument* for the learned model, which is cheap and commits to nothing.
