# Figures plan: a figure book for every stage

User request (2026-11-11): every stage of the project gets figures that explain the procedure
of that stage, the findings that led to the next step, and how the work progressed. The main
figure set focuses on the method. The corrections of the submitted project go into a separate
set that feeds a side report for the defense (`docs/defense_notes.md`). From now on every
stage adds its figures as it is done. At the end the user chooses the figures for the report.

The report itself keeps the short correction note and the errata table of D-02; this plan only
decides where each figure lives.

## 1. What exists before this plan

| Stage | Producer | Figures | Kind |
|---|---|---|---|
| 3 | `solarpv.eda` | 10 (`outputs/03_eda/figures/`) | data overview; fig02 is a before/after of the time fix |
| 3b | `solarpv.eda.quality` | 30 (`outputs/03b_quality/figures/`) | evidence for every cleaning threshold |
| 6 | `solarpv.features.decide_k` | 1 (`outputs/06_features/d11_l04.png`) | D-11 criterion 2 and L-04 |
| 6b | `solarpv.nwp.check` | 1 (`outputs/06b_gfs/value_check.png`) | D-12 |
| 7 | `solarpv.selection.run` | 7 (`outputs/07_feature_selection/figures/`, v6 figures 19–25) | selection protocol |
| 9 | `solarpv.models.xgb_sota` | 2 per run (`outputs/runs/<run>/xgb/figures/`) | Optuna history, learning curves |

Stages 0, 1, 2, 4, 5, 8 and the W3 part of stage 9 have no figure. No catalog lists the
figures, says what each one shows or what it led to, or checks that a figure is current.

## 2. Self-review before the code

### 2.1 The original project's figures (v6)

| ID | Finding | Effect | Handling here |
|---|---|---|---|
| G1 | Every v6 figure was drawn from the shifted data (F-01) and the clipped target (F-06) | none of them can be reused | all figures are drawn from the rebuild's outputs; the v6 figures are regenerated in stage 11 (`docs/scope_register.md`) |
| G2 | No figure was linked to a code version or to its inputs (C-38) | a figure could not be traced or shown to be current | a manifest records the commit, the sha256 of every input and of every output file; a test fails on a stale figure |
| G3 | Figure 17 showed LSTM training-set metrics as seed variability (C-10) | wrong quantity under a correct-looking title | every figure states the split and the quantity on its axis; model figures use validation only |
| G4 | Models were plotted on different hours (C-35) and the route figure was not paired (C-34) | not comparable | evaluation figures read the stage-8 tables, which share the rows |
| G5 | The split figure (Figure 11) had a wrong train size; the report had no data-description figures | the reader could not see the data or the splits | stage 1 draws the split timeline from the manifest; stages 2–5 get data figures |

### 2.2 My own figures so far

| ID | Finding | Handling |
|---|---|---|
| X1 | Five producers, five styles: DPI 110–150, default colour cycles, titles that use internal IDs (D-11, L-04) | one style module; titles in plain words; IDs only in the catalog |
| X2 | Several figures separate series by colour only | every series has colour plus line style or marker (Okabe-Ito palette) |
| X3 | `fig01_overview` uses two y axes | new figures use one axis per panel; fig01 stays as it is and is listed |
| X4 | Only the stage-3 and 3b figures are checked for staleness | the figure manifest covers the new figures; the catalog lists the existing ones with their producer |
| X5 | No figure has a written takeaway | every new figure has a question, a takeaway formatted from computed facts, and the step it led to |
| X6 | Value axes of some line figures do not start at zero without saying so | bar charts start at zero; line charts of errors may not, and the axis label names the quantity |

## 3. Rules

1. **Process and correction sets.** A process figure shows what the rebuild did and found. It
   never compares with the original project. A correction figure shows a difference from the
   submitted project and is used in `docs/defense_notes.md`. The catalog keeps them in two
   sections.
2. **No test rows.** No figure computes a target or model statistic on the test split. Whole-
   period data descriptions (data quality, clocks, flags) may span all splits, with the splits
   marked, as in stage 3.
3. **Read outputs, do not decide.** A figure reads the tables and manifests a stage wrote. Where
   it must read data, it uses the stage's own functions. No figure changes a decision.
4. **Facts, not typed numbers.** Every number in a takeaway is formatted from the facts that the
   builder returns, and the facts are stored in the manifest.
5. **One style.** `solarpv.figures.style`: Okabe-Ito colours, colour plus a second cue, 200 DPI
   PNG, white background, constrained layout, one value axis per panel.
6. **Every stage adds its figures** (standing rule in `CLAUDE.md`): a stage is not complete until
   its figures are registered, built and listed in the catalog.
7. **Every task ends with its figures, a sheet, and the figures shown** (user request,
   2026-11-12). A task is a work item of a stage (W1–W6 of stage 9) or a stage without work
   items. Its figures cover everything the task did and found: every new step, check, decision
   and finding. At the end of the task its figures are listed under the task in `WORK_ITEMS`
   (`registry.py`), built with `python -m solarpv.figures task <key>`, which also writes the
   task's sheet `docs/figure_sheets/<key>.md` (question, what the figure shows, takeaway, what
   it led to, file, inputs, commit; generated, never edited), and every PNG is looked at. The
   figures and the sheet are then shown to the user in the app, and the closing message lists
   each figure's ID with its takeaway. A task is not complete until its figures are shown.
8. **Every task looks back** (user request, 2026-11-12). Before its figures are built, the task
   reviews every earlier task and stage: what the new outputs now make possible (a run, a table
   with all models, a report), which earlier figure is now incomplete (a model missing, a
   number typed that can now be read from an output), and which producer figure has no place
   in the book. What can be drawn from existing outputs is built, under the task it documents;
   what needs new computation or a decision is listed for the user. Each look-back is a row of
   the look-back log (§7); T-72 fails when a completed task has none.
9. **Uniform figures across the models** (user request, 2026-11-12). Every model-level kind of
   figure (score against the references, significance, route comparison, learning curves,
   search history, bias, error breakdown, example days, intervals and explanations where the
   model has them) covers every model in one layout, and each model keeps one colour and
   marker in every figure (`MODEL_STYLE` in `solarpv.figures.s09_models`; T-72). A model
   without the output is named in the figure with the reason. The coverage matrix is §8.

## 4. Design

- `src/solarpv/figures/`: `style.py`, `registry.py` (the figure spec), one module per stage
  (`s00_s02.py`, `s04_s05.py`, `s06.py`, `s07_s09.py`), `correction.py` and `__main__.py`.
- Command: `python -m solarpv.figures build [--stage 6] [--only S06-02]`, `task <key>` (the
  end of a task, rule 7), `list` (with each figure's task), `catalog`.
- Outputs: `outputs/figures/<stage>/<ID>_<slug>.png`, `outputs/figures/figures_manifest.json`
  and a local gallery `outputs/figures/gallery.html`, grouped by task, for choosing figures at
  the end.
- Sheets (rule 7): `docs/figure_sheets/<key>.md`, one per task (`S06`, `S09-W5`, ...;
  `corrections` for the correction figures of the figure book), and an index
  `docs/figure_sheets/README.md`; generated with the catalog.
- Catalog: `docs/figure_catalog.md`, generated: per stage, the ID, title, question, description,
  takeaway, the step it led to, the file, and the existing figures of that stage.
- Test (T-72): `tests/contracts/test_figures.py`: unique IDs; every completed stage has process figures;
  every registered figure is in the manifest and on disk with its recorded hash; inputs
  unchanged since the build; the catalog equals its regeneration; correction figures are not in
  the process section; every figure belongs to exactly one task, every completed task has its
  minimum of process figures (`REQUIRED_TASKS`), and every sheet equals its regeneration (X52).

## 5. The figures

| ID | Stage | Title | Source |
|---|---|---|---|
| S00-01 | 0 | Raw export: rows per monthly file | `00_data_audit/raw_data_inventory.csv` |
| S01-01 | 1 | The pipeline, stage by stage | stage list (drawn) |
| S01-02 | 1 | Train, validation and test periods | constants, stage-6 table |
| S01-03 | 1 | Feature contract and excluded columns | `solarpv.features.contract`, `leakage` |
| S02-01 | 2 | Daylight-saving days on a continuous UTC grid | ingest parquet |
| S02-02 | 2 | Raw value anomalies found at ingestion | `ingest_qa.csv`, manifest |
| S02-03 | 2 | Production centre against solar noon after UTC conversion | `centroid_offsets.csv` |
| S04-01 | 4 | Logger clock offsets applied per day | `configs/clock_offsets.csv` |
| S04-02 | 4 | Clock check after the correction | clock audit monthly tables |
| S04-03 | 4 | The second data source: held values per day | ingest parquet |
| S05-01 | 5 | What each cleaning rule changed | `cleaning_log.csv` |
| S05-02 | 5 | Operationally limited daytime hours per month | stage-5 hourly table |
| S06-01 | 6 | From power to the clear-sky index on two days | stage-6 table (train) |
| S06-02 | 6 | Choosing the clear-sky reference | `d11_criteria.csv` |
| S06-03 | 6 | The clear-sky index on train | stage-6 table (train) |
| S06-04 | 6 | Which past days each horizon may use | contract rule (drawn) |
| S06-05 | 6 | Feature table per horizon | `feature_dictionary.csv` |
| S6b-01 | 6b | Forecast runs public before issuance | GFS feature table |
| S6b-02 | 6b | GFS agreement with the measured clear-sky index | `horizon_skill.csv` |
| S6b-03 | 6b | Value of the GFS features on validation | `value_check.csv` |
| S07-01 | 7 | Selected sets against the full pool inside train | `inner_train_check.csv` |
| S07-02 | 7 | Model inputs per route and horizon | `model_inputs.json`, dictionary |
| S08-01 | 8 | Evaluation hours on validation | stage-6 table (validation) |
| S08-02 | 8 | Evaluation code checked on the surviving forecasts | fixture tables |
| S08-03 | 8 | Reference forecasts on validation | stage-8 tables |
| S09-01 | 9 | Run modes of the pipeline runner | drawn |
| S09-02 | 9 | XGB development run against the references | dev-run tables |
| S09-03 | 9 | Monotone constraint check | `d15_check.csv` |
| S09-04 | 9 | Linear correction of the indirect route | `l09_check.csv` |
| S09-05 | 9 | Significance of the differences in kW | dev-run `dm_tests_kw.csv` |
| S09-06 | 9 | Sequence datasets: one origin per local day | `checks_dev.csv` |
| S09-07 | 9 | Smoke runs of the five neural models | smoke and dev `learning_curves.csv`, `run.json` |
| S09-08 | 9 | Learning curves of the neural models | dev `learning_curves.csv`, `best_params.json` |
| S09-09 | 9 | Neural models in the development run | dev-run tables (neural and XGB) |
| S09-10 | 9 | Optuna searches of the neural models | dev `trials.csv`, `best_params.json` |
| S09-11 | 9 | Where the TFT models look in their encoder | dev `attention_*.csv` |
| S09-12 | 9 | Mean bias of the neural models | dev `mbe.csv`, `r14_bias.csv` |
| S09-13 | 9 | The two phases of the final run | the runner's step order and rules (code) |
| S09-14 | 9 | What the development XGBoost relies on | W5 XGB dev `explain/xgb/*` |
| S09-15 | 9 | Attribution of the feature groups | W5 XGB dev `xgb_shap_groups.csv` |
| S09-16 | 9 | TFT attention: same hour or fixed clock hours? | neural dev `explain/tft/*` |
| S09-17 | 9 | Hyper-parameter importance of the development studies | both dev runs `hparam_importance.csv` |
| S09-18 | 9 | Expected duration of the final run | the dev `step.json`, curves and trials |
| S09-19 | 9 | Automatic warnings of the development runs | both dev runs `report_warnings.csv` |
| S09-20 | 9 | What the select step verifies | both dev runs `select/select.json` |
| S09-21 | 9 | SHAP explains the booster of the forecast | W5 XGB dev `xgb_shap_checks.csv` |
| S09-22 | 9 | Checks of the XGB explanation | W5 XGB dev `xgb_shap_checks.csv`, `xgb_shap_rank_stability.csv` |
| S09-23 | 9 | What the TFT models select | neural dev `tft_variable_importance.csv` |
| S09-24 | 9 | Points the run reports raise for G-1 | both dev runs `report_g1_points.csv` |
| S09-25 | 9 | The six models side by side | the run configurations, `select.json` |
| S09-26 | 9 | Every model against persistence: Diebold-Mariano in kW | both dev runs `dm_tests_kw.csv` |
| S09-27 | 9 | The two routes, model by model | both dev runs `route_comparison.csv` |
| S09-28 | 9 | Forecasts on three validation days | the dev prediction files, stage-6 truth (validation) |
| S09-29 | 9 | The quantile forecasts of the sequence models | neural dev `quantiles_*.csv`, predictions |
| S09-30 | 9 | Where the error comes from, model by model | both dev runs `seasonal_metrics.csv`, `benchmark_views.csv` |
| S09-31 | 9 | Learning curves of the XGB final fits (look-back, W2) | W5 XGB dev `learning_curves.csv`, `best_params.json` |
| S09-32 | 9 | Optuna searches of XGB (look-back, W2) | W5 XGB dev `trials.csv` |
| S09-33 | 9 | The runs of stage 9 so far (look-back, W1) | every stage-9 `run.json` |
| S08-04 | 8 | The secondary views of the evaluation (look-back) | stage-8 `benchmark_views.csv` |
| C-01 | correction | Daily profile read as UTC and as local time | stage-6 table (train), shifted |
| C-02 | correction | Zero-output daytime hours by label hour | submission forecasts |
| C-03 | correction | Clear-sky index of the submission and of the rebuild | submission, stage-6 table |
| C-04 | correction | Rolling windows per horizon, original and fixed | drawn |
| C-05 | correction | Periods read by the feature selection; DC dominance | backup evidence |
| C-06 | correction | Original weather series: correlation by time shift | backup evidence |
| C-07 | correction | Seed table of the submitted BiLSTM | submission seed table and benchmark |
| C-08 | correction | Scored hours per model, submitted and rebuilt | submission benchmark, dev-run tables |
| C-09 | correction | Which lags a TFT encoder can hold | dev `attention_*.csv`, v6 §11.2 claim |
| C-10 | correction | SHAP groups: the submission against the rebuild | `07_shap/shap_group_importance.csv`, W5 XGB dev groups |

*Added 2026-11-12 (W4 figures review):* S09-07–S09-12 were built with W4 but were missing from
this table; C-07–C-09 draw the W4 differences from the submission (C-10, C-35, C-54, C-55). The
review also found X36 (a typed, wrong hour count in a description) and X37 (the seed-42 row of the
submitted seed table); see `docs/errata.md`.

*Added 2026-11-12 (W5):* S09-13–S09-19 (module `solarpv.figures.s09_w5`) show the final protocol,
the explanations and the run reports of the two development runs; C-10 draws the difference of
the SHAP groups from the submission (C-59, C-60). One pre-written claim was refuted by the data
before the build: S09-16 was drafted to assert that every TFT prefers the same hour of earlier
days; the dev run shows a ratio of 0.82 to 1.07, so the figure now states what the data show (a
preference for fixed clock hours) and asserts only the absence of a clear same-hour preference.

*Added 2026-11-12 (user request after W5, rule 7):* the W4 and W5 sessions built their figures
but never showed them to the user, and nothing grouped the figures by task (X52). The registry
now lists the figures of each work item (`WORK_ITEMS`), `python -m solarpv.figures task <key>`
builds a task's figures and writes its sheet in `docs/figure_sheets/`, the gallery is grouped by
task, and T-72 checks the tasks and the sheets. A review of what W5 did against its figures
found five outputs without one, now S09-20 (the select step's checks), S09-21 (SHAP of the
saved booster against the booster cut at the forecast's rounds, X46), S09-22 (dominance and
rank stability of the XGB explanation), S09-23 (the TFT variable-selection weights by group,
X48) and S09-24 (the points the run reports list for G-1).

*Added 2026-11-12 (user request: W4 redone, rules 8 and 9):* the W4 figures were reviewed the
same way, with a look-back over every earlier task and stage and a check that every model gets
the same kinds of figure. W4 gained S09-25–S09-30 (the models side by side, significance against
persistence, the two routes, three example days, the quantile forecasts, the error by season and
by operation); the look-back added S09-31 and S09-32 to W2 (XGB learning curves and searches in
the layout of the neural ones), S09-33 to W1 (the runs of stage 9), S08-04 to stage 8 (the
secondary views), redrew S09-06 with all five encoders read from the W3 report (it showed three,
with typed lengths), and gave XGB one colour in every figure (it was blue in S09-02 and black in
S09-09). S09-29 found X53: the quantiles of two TFTs are out of order on most direct-route hours.
Findings: X53, X54 (`docs/errata.md`).

## 7. Look-back log (rule 8)

One row per look-back. *At the end of* is the task that ran it; *Reviewed* the earlier tasks and
stages; *Built* the figures it added, under the task each documents; *Not built* what needs new
computation or a decision, for the user.

| Date | At the end of | Reviewed | Built | Not built, and why |
|---|---|---|---|---|
| 2026-11-12 | S09-W5 | the W5 outputs against the W5 figures (rule 7; rule 8 came after) | S09-20–S09-24 | none |
| 2026-11-12 | S09-W4 (redone after W5) | stages 0–8; S09-W1, S09-W2, S09-W3 and S09-W5; every model-level figure against rule 9 | W4: S09-25–S09-30; W2: S09-31, S09-32; W1: S09-33; W3: S09-06 redrawn with all five encoders; stage 8: S08-04; S09-02 redrawn in XGB's colour | explanations of the BiLSTM and PatchTST (no such output; new computation and a change of D-31, the user's decision); seed spread (one seed in the dev runs; the final run has three); intervals of XGB and PatchTST (point forecasts only); test-split figures (stage 10, after G-1); stages 3 and 3b keep their producer figures (complete, listed in the catalog) |

## 8. Coverage of the models (rule 9)

| Kind of figure | XGB | BiLSTM | TFT-96 | TFT-168 | TFT baseline | PatchTST | Figures |
|---|---|---|---|---|---|---|---|
| Inputs, outputs and budgets | yes | yes | yes | yes | yes | yes | S09-25 |
| Score against the references | yes | yes | yes | yes | yes | yes | S09-02, S09-09 |
| Significance in kW | yes | yes | yes | yes | yes | yes | S09-26 (S09-05: XGB against each reference) |
| The two routes | yes | yes | yes | yes | yes | yes | S09-27 |
| Example days | yes | yes | yes | yes | yes | yes | S09-28 |
| Error by season and by operation | yes | yes | yes | yes | yes | yes | S09-30 |
| Learning curves | yes | yes | yes | yes | yes | yes | S09-31, S09-08 |
| Search history | yes | yes | yes | yes | fixed settings | fixed settings | S09-32, S09-10 |
| Hyper-parameter importance | yes | yes | yes | yes | fixed settings | fixed settings | S09-17 |
| Mean bias | validation line | yes | yes | yes | yes | yes | S09-12 |
| Smoke runs and time | yes | yes | yes | yes | yes | yes | S09-33, S09-07, S09-18 |
| Intervals | point only | yes | yes | yes | yes | point only | S09-29 |
| Explanation | SHAP | none (D-31) | attention, selection weights | attention, selection weights | attention, selection weights | none (D-31) | S09-14, S09-15, S09-21, S09-22; S09-11, S09-16, S09-23 |
