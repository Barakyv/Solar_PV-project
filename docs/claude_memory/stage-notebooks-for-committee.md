---
name: stage-notebooks-for-committee
description: "Every completed stage needs a live \"process notebook\" in the original's numbering, built from outputs, for the committee"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d327e5e5-b1d8-472a-8bda-4cf94ddd836e
  modified: 2026-11-01T11:32:24.103Z
---

The user will present the project to the committee from live notebooks ("תיעוד בלייב להראות"),
one per stage, in the numbering of the original project (01_data_ingest … 07_evaluation).
Each notebook has a builder `scripts/make_<name>_notebook.py` using `scripts/_notebook.py`:
notes formatted from the stage outputs (never typed), an MD note after every code cell, sources'
sha256 in the metadata, checked by `tests/contracts/test_stage_notebooks.py` (add new names there).

**Why:** asked on 2026-11-01 together with the review of the old backup; the committee knows the
original project, so each notebook also states what differed in the original.

**How to apply:** finish every stage (7, 9, 10, 11) with its notebook and rebuild it after reruns.
Related: [[self-review-every-stage]], [[hebrew-rtl-display-rules]].
