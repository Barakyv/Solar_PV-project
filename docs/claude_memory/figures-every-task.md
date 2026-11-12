---
name: figures-every-stage
description: Every task ends with figures (skill task-figures) - look back over earlier tasks, keep models uniform, write the sheet, show every figure in the app
metadata:
  node_type: memory
  type: feedback
  originSessionId: 617ff654-16a0-43ed-a08c-aae51392d374
  modified: 2026-11-12T09:39:29.189Z
---

From 2026-11-11 every stage adds figures to the figure book (`solarpv.figures`,
`docs/figures_plan.md`). On 2026-11-12 the user widened this three times:
1. **Every task** (a work item such as W5, not only a stage) ends with figures that cover
   everything it did and found, and the figures are **shown** in the app. W4 and W5 had built
   figures but never shown them, and the user thought none had been made.
2. **Look back** at every figure production: review every earlier task and stage for figures the
   new outputs make possible, earlier figures that miss a model or carry a typed number; build
   what existing outputs allow, list the rest for the user, and log it (§7 of the plan).
3. **Uniform across the models**: every model-level figure covers every model in one layout,
   one colour and marker per model (`MODEL_STYLE`), coverage matrix in §8 of the plan.

**Why:** the user writes the process reports from these figures and wants everything
documented; the report presents the method and plays down that the submitted project was wrong,
while `docs/defense_notes.md` keeps the corrections for defense questions.

**How to apply:** follow the project skill `.claude/skills/task-figures/SKILL.md` at the end of
every task: list figures in `WORK_ITEMS`, `python -m solarpv.figures task <key>`, look at every
PNG, raise `REQUIRED`/`REQUIRED_TASKS` (T-72), add the look-back row, then `SendUserFile`
(render) with the PNGs and the sheet, and list ID + takeaway in the closing message. A figure
that reveals something is a finding (X-ID, errata, journal; a decision for the user if it
touches a result). Related: [[stage-notebooks-for-committee]], [[self-review-every-stage]].
