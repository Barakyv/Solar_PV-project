---
name: task-figures
description: Locally authored for this project (not downloaded). Use at the end of every task of the Ktora PV rebuild (a work item such as W6, or a stage without work items), before the closing message - builds the task's figures for the process report, looks back over every earlier task and stage for figures that can now be made, keeps the figures uniform across the models, writes the task's figure sheet and shows every figure to the user.
---

# Task figures (Ktora rebuild)

The user writes the process reports from these figures, so a task is not complete until its
figures are built, documented and shown. Rules 6–9 of `docs/figures_plan.md` and the
`CLAUDE.md` rule "Figures for every task, shown to the user" are the source; this file is the
checklist. `PROJECT_MASTER.md` and `CLAUDE.md` win where they differ.

## When to use

At the end of every task, after its results are accepted and before the closing message:
every work item (W6 and on), every stage, and every redo of a task's figures. Not for an
exploratory plot that decides nothing.

## 1. What did the task do and find?

List every new step, check, decision, run and finding of the task (the plan's self-review
section, the run journal, the errata rows of the task). Each one needs a figure that shows what
was done, what was found and what it led to, unless it is already drawn. A finding of the
self-review (M-xx, X-xx) that changed the code usually deserves a before-and-after figure.

## 2. Look back (rule 8)

Review every earlier task and stage (`docs/figure_sheets/README.md`, `docs/figure_catalog.md`):

- **New outputs, old tasks.** Does this task's output (a run, a table with all models, a report,
  a check file) make a figure possible for an earlier task? Put it under the task it documents
  (`WORK_ITEMS`), not under the current one.
- **Incomplete figures.** Does an earlier figure miss a model or a route that now exists, or
  carry a number typed in the code that an output can now give? Redraw it from the output.
- **Producer figures.** Is a figure made by a stage producer (the `EXISTING` list) needed in the
  book, in the book's style?
- **Not possible.** What needs new computation, a new run or a user decision is listed, not
  built.

Record one row in the look-back log, `docs/figures_plan.md` §7: date, the task, what was
reviewed, what was built, what was not built and why. T-72 fails without it.

## 3. Uniform across the models (rule 9)

Every model-level kind of figure covers every model in one layout: inputs and budgets, score
against the references, significance, routes, example days, error breakdown, learning curves,
search history, hyper-parameter importance, bias, time, intervals and explanations where the
model has them. Each model keeps its colour and marker (`MODEL_STYLE` in
`solarpv.figures.s09_models`; T-72). A model without the output is named in the figure with the
reason. Update the coverage matrix, `docs/figures_plan.md` §8.

## 4. Build

1. Register each figure (`solarpv.figures`, one module per stage or theme): question,
   description without typed counts, a takeaway formatted from the builder's facts, every
   non-numeric claim asserted in the builder, the step it led to, its inputs. Process figures
   never compare with the submitted project; corrections are kind `correction` and go to
   `docs/defense_notes.md`. Validation and train only; never the test split before stage 10.
2. List the figures under the task in `WORK_ITEMS` (`solarpv/figures/registry.py`).
3. `python -m solarpv.figures task <key>` (writes `docs/figure_sheets/<key>.md`); rebuild
   redrawn figures of other tasks with `build --only`.
4. **Look at every PNG** (Read the image): overlapping legends, cut text, empty panels, colours
   against `MODEL_STYLE`, a y axis that does not start at zero must say so. Fix and rebuild.
5. A figure that reveals something new is a finding: an X-ID, an errata row, a run-journal note,
   and a decision for the user if it touches a result (never act on it alone).

## 5. Record

- `tests/contracts/test_figures.py`: raise `REQUIRED` and `REQUIRED_TASKS`.
- `docs/figures_plan.md`: the figure rows of §5, a dated note, the §7 row, the §8 matrix.
- `docs/run_journal.md`, `docs/data_processing_record.md` (OP-56 additions),
  `PROJECT_MASTER.md` (§17 line).
- `python -m solarpv.figures catalog`, the contract tests, ruff; commit; then rebuild the task's
  figures at the clean commit so the sheet records it, and commit again.

## 6. Show

`SendUserFile` with display `render`: every PNG of the task (in groups of up to seven), the
figures redrawn by the look-back, and the sheet. The closing message lists each figure's ID with
its takeaway, the look-back's additions and what was not built, in the user's language and
layout rules.
