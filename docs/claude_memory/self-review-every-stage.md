---
name: self-review-every-stage
description: "Before and during every stage, self-review the plan and the earlier code against the original project so its mistakes are not repeated"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d327e5e5-b1d8-472a-8bda-4cf94ddd836e
  modified: 2026-10-25T07:23:47.177Z
---

The user asked (2026-10-25) that I "always perform self-review of what was done, of what I am
about to do, and against the original project; I will not repeat mistakes that were made".

**Why:** the rebuild exists because the original pipeline had silent errors (time zone, clipped
target, leaky rolling, unpaired route comparison...). Repeating one would undo the project's
purpose.

**How to apply:**
- Every stage plan (docs/stageN_plan.md) gets a self-review section before the code:
  1. the original scripts, notebooks and report claims for that stage, with each mistake and the
     guard that prevents it (stage 8B: E1–E11);
  2. a review of my own earlier code for that layer (stage 8B: A1–A6).
- During the runs, check every result against theory and sanity (stage 8B: persistence beat the
  CLIPER reference, which is impossible, so the definition was fixed and a test was added).
- Each finding gets an ID (C-xx or F-xx), an errata row, a test and a run-journal note.
- Report the self-review findings to the user in the stage summary.

Related: [[hebrew-rtl-display-rules]].
