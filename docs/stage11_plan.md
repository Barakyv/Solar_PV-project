# Stage 11, part A: tables, figures and the first errata from the frozen final run

Run `20261114T083930Z_final`, tag `g1-frozen` (`10611a6`), HEAD `a92756b`. No model is trained, and no
file of the pipeline, the models or the configs changes (D-36, part b: only `src/solarpv/figures/`,
`scripts/make_*`, `notebooks/`, `docs/`, `tests/`, `outputs/` and the two root documents).
The test split was scored once, by the step `test` (D-32). Everything below reads the saved
prediction files, the tables of the run and the stage-6 truth table; it never re-runs `test` and never
passes `--allow-test`. Every read of the test rows is appended to `outputs/07_evaluation/test_access_log.csv`.

## 1. What exists, and what is missing (PROJECT_MASTER §7, stage 11)

| Item of §7 | Exists in the run | Missing before this part |
|---|---|---|
| benchmark, DM, MDE, route, seasonal, common mask (S-23–S-28, S-31) | `report/*.csv`, `evaluate/val/*`, `test/test/*` (all nine models and references, same rows) | CI in kW; a figure from the final run |
| SHAP, rank stability (S-32) | `explain/xgb/*`, `test/shap_test/*` | the figure |
| TFT attention (S-33), hyper-parameter importance (S-36) | `explain/tft/*`, `explain/hparam_importance.csv`, `bilstm|tft*/trials.csv` | the figures |
| ablation, CV (S-38, S-41) | `report_ablation*.csv`, `report_cv*.csv`; S10-06 draws the CV | the ablation figure |
| bias² + variance (S-39) | prediction files of three seeds | the analysis |
| production quintiles (S-40) | prediction files | the analysis |
| hour × season (S-30) | prediction files | the analysis |
| encoder length 96 vs 168 (S-34) | one forecast file per TFT, the DM only against XGB | the paired test |
| PatchTST structure (S-35) | its forecast file | the gap table |
| errors by season (L-20) | `seasonal_metrics.csv` | the figure, and the like-for-like reading |
| errors on dust days (L-20) | nothing | a definition of a dust day (Q-29) |
| literature comparison (L-20) | `nRMSE_mean_pct` (L-16) in the benchmark | the aligned table |

Not done in this part (it needs training, a code change outside the D-36 list, or a decision):
the dust-day split (Q-29), `reproduce_paper_results.py --xgb-only` and `--skip-train` (S-45: the
script is outside the D-36 list; Q-30), the report text (S-01, S-05, S-42–S-44) and the final
errata table of old against new values (part B).

## 2. Self-review before the code

**The original project** (`submission_evidence/13_report`, §9.2, §11.3, §11.4, §12.3, §12.4):

- **M45, production quintiles (§12.4).** The bins are quintiles of the *observed* power, with edges
  taken from the test sample itself. Binning by the outcome mechanically puts positive errors in the
  low bins and negative errors in the high bins (the forecast regresses to the mean), so the
  "W-shaped profile", the "+50 kW at near-zero output because of inverter downtime" and the
  "systematic peak underestimation" were read as physics without a control. Here: edges are fixed
  on validation, two binnings are reported side by side (observed power, and the forecast itself, which is
  known at issuance), and the share of operation-limited hours (D-22) is reported per bin, so the
  downtime reading is tested instead of assumed.
- **M46, bias² + variance (§12.3).** The table has no definition. "Variance" is the share of the
  MSE that is not the squared mean error, which is the variance of the error, not the variance of
  the model; the single mean-prediction sentence about the BiLSTM was not tested. Here: the identity
  MSE = ME² + Var(e) is asserted per row of the table, the seed-ensemble split (bias² of the
  seed mean and variance across seeds) is a second, separately named table, and the conditional
  diagnostics of L-15 (correlation, variance ratio) are beside them.
- **M47, hour × season (§9.2).** "The top four models" were chosen by their score, which selects the
  panels by the result; night cells were greyed by hand; no count per cell. Here: all six models and
  the references in one layout, the number of hours per cell in the table, and cells with fewer than
  20 hours masked. The test months (October to May) hold no summer; the validation months (February to
  September) hold one hour-of-season mix that cannot be compared cell by cell with test.
- **M48, encoder length (§11.3).** The "consistent advantage of the shorter window" is a gap of 1.4 pp
  read without a paired test, and the attention picture was interpreted as "noise time steps". The
  two models also have different hyper-parameters found by Optuna. Here: a paired Diebold-Mariano
  test in kW (HLN, Bonferroni over the six cells), the attention numbers as description only, and the
  limit stated: one seed each, and tuned separately.
- **M49, PatchTST (§11.4).** "Within 1.2 pp of XGB" was turned into an upper bound on the marginal
  value of feature engineering, although the XGB of v6 had NWP inputs and the PatchTST had 243
  origins. Here: the gap to XGB and to climatology with the DM, and no bound is stated.
- **M50, comparison with the literature.** The submitted report quotes nRMSE normalised by the plant
  capacity; the papers of `docs/literature_review.md` normalise by the mean daytime power. A
  comparison of the two numbers is wrong by a factor of about three. Here: `nRMSE_mean_pct` (L-16),
  the differences of climate, NWP and resolution in the table, and no ranking against the papers.

**My earlier code and the new code:**

- **X100.** A second implementation of the metrics could disagree with the run's own tables. The new
  module asserts that its nRMSE in kW equals `report_benchmark_val.csv` and `report_benchmark_test.csv`
  to 1e-6 for every model, seed and cell before it writes anything.
- **X101.** The new analyses read the test truth. Every read is logged; nothing is selected, tuned or
  decided from it; comparisons of test use the validation of the same view (S10-03 rule).
- **X102.** Local hour and season must use the local clock (F-01), with DST (hours 23 and 25 of a
  day); the module takes both from `tz_convert("Asia/Jerusalem")`, and a test covers a DST day.
- **X103.** Quintile edges with ties (many zero or near-zero hours) can give empty or unequal bins;
  the table carries `n` per bin and a test checks that the bins partition the rows.
- **X104.** A comparison of test against validation across seasons confounds the season with the
  calendar months (validation autumn is September, test autumn is October and November); every such
  table names the months.

## 3. Rules for the numbers

- No winner is declared (D-37, S10-04). Four of the five neural models have one seed.
- The Diebold-Mariano test against climatology is significant in one cell of six; no sentence says
  that XGB is better than climatology.
- The seed-mean rows of a model are used in every table; a figure shows the range over seeds where
  the model has three.
- Every number of a document comes from a file of `outputs/11_analyses/` or of the run; the
  takeaways of the figures are formatted from the builder's facts.

## 4. Decisions asked of the user (not decided here)

- **Q-29, dust days (L-20).** The project holds no aerosol record. Options: (a) CAMS AOD, which needs
  a free ADS account (user action, Q-14); (b) a label from the site's own measured clear-hour deficit
  of day D, with a threshold fixed on train only (a label made from the outcome, so the errors on those
  days are conditional on the outcome); (c) drop the dust-day half of L-20 and keep the season half.
- **Q-30, the options `--xgb-only` and `--skip-train` of `scripts/reproduce_paper_results.py` (S-45).**
  The script is outside the D-36 list, so editing it reopens G-1. Options: (a) a new `scripts/make_*`
  script that rebuilds the tables and figures from the frozen predictions (allowed), and leave
  `--xgb-only` (a training run) for the end; (b) edit the script and reopen G-1.

## 5. Part B: the report sections, the traceability and the old against new results (2026-11-14)

Same frame as part A: run `20261114T083930Z_final`, tag `g1-frozen`, no model trained, the step `test` not run
again, no file outside the D-36 list edited. Before the work: the tag is present, `git log` shows `a5167f0` and
`fff0e5c` above `a92756b`, no tracked change, `python -m solarpv.pipeline status` lists 14 steps `ok` with outputs
as recorded, and `tests/contracts/test_after_tag.py` passed (3).

**Decisions of the user (2026-11-14):**
- **Q-29, dust days: option (c).** The dust-day half of L-20 is dropped and the season half stays (S11-04, S11-05,
  S11-17). The project holds no aerosol record; no label is made from the outcome. The report says so under
  future work. L-20 is marked done for its season half in §16b.
- **Q-30, S-45: option (a).** A new script, `scripts/make_tables_from_frozen.py`, rebuilds the tables, the figures and
  the report sections from the frozen predictions without training (the counterpart of `--skip-train`).
  `scripts/reproduce_paper_results.py` is outside the D-36 list and is not edited, so G-1 stays closed. `--xgb-only`,
  a training run, is left for the end of the stage.
- **Backup of the repository and the archive outside the VDI:** not done yet (open, the user's action).

**What part B writes:** `docs/report_v7.md` (S-01, S-05, S-42 to S-44 and the correction note of D-02), generated by
`scripts/make_report_v7.py`; `docs/traceability.csv` (one row per number of the text: claim, value_old, value_new,
file, script, verified_on); `docs/old_vs_new.csv` and the table "Old vs new results" of `docs/errata.md`; the figures
S11-18, S11-19, C-11 and C-12 (task S11-B); the notebook `10_stage11_part_b.ipynb`; OP-67; T-178 to T-184.

**Self-review before the code.** The original project (`submission_evidence/13_report`, sections 3.1, 3.5, 13 to 16):
- **M51, section 3.1:** the plant is described as two inverters (C-30), a section of a 40 MW park.
- **M52, sections 13.2 and 14 (C2):** the indirect route is called the most actionable finding from a truth rebuilt
  from clipped k (C-16); here both routes are scored against the actual AC and the recommendation is withdrawn.
- **M53, sections 13.1 and 13.5:** "would likely re-establish a larger gap" and "a summer lowers all errors by 5 to 8
  points" are statements with no computation; the first is now tied to the ablation and the SHAP shares, the second is
  withdrawn.
- **M54, sections 13.3 and 14 (C6):** the "independent" agreement of SHAP and attention on three lags was not tested;
  attention is a description, not evidence of use.
- **M55, section 15:** D1 repeats the 13.5 percent artefact of the time base (F-01c); D2 reads a search-space edge as a
  finding; D3 to D5 name metrics that are not the scored ones (C-07).
- **M56, section 13.6:** a sensitivity analysis for anonymised coordinates that the rebuild no longer uses (D-05).
- **M57, section 14 (recommendations):** use the indirect route, do not use TFT-168, build an ensemble for the peak
  bias that came from the binning (M45).

My own earlier work and the new code (X107 to X112, listed in `docs/errata.md`): a typed number could drift from the
outputs (X107); old and new numbers could be presented as a difference although the unit or the definition differs
(X108); the placeholder of this file split a table (X109); three first-draft errors found by reading the generated
report, a different aggregation of the rank correlation (X110), a count of test scorings that was 0 (X111), and a
range of the feature counts built from two horizons only (X112).

**Rules for the numbers (checked by T-178 to T-184).** Every number of the report is formatted from a file and
registered; every test number sits beside the validation number of the same view; no winner is declared (D-37; four of
the five neural models have one seed); the Diebold-Mariano test against climatology is significant in one cell of six
and the report says so; the positive bias is made in the operation-limited hours (F-30); no budget, seed or rule
changes because of a test number; the quintile profile of v6 is not written as physics (M45).
Not done in part B: the dust-day split (Q-29, dropped), the radar (C-84) and `--xgb-only` (training).

## 6. Part C: the final report v7 with its figures (2026-11-15)

Same frame as parts A and B: run `20261114T083930Z_final`, tag `g1-frozen`, no model trained, the step `test` not run
again, no file outside the D-36 list edited (the changes are in `scripts/make_report_v7.py`, `src/solarpv/figures/`,
`tests/` and `docs/`). Before the work: the tag is present, `git log` shows `a407f85` and `ee4c08b` above `a5167f0`, no
tracked change, `python -m solarpv.pipeline status` lists 14 steps `ok` with outputs as recorded, and
`tests/contracts/test_after_tag.py` passed (3).

**Decision of the user (2026-11-15).** The figures of the report follow the figures of the submitted report as closely as
possible, with figures added where they complete the understanding of what was done; the report presents the work as
closed, without raising open questions. The choice was delegated to the session, with that rule.

**How the figures were chosen.** Each of the 25 figures of v6 (`submission_evidence/13_report`, the figure column of
`docs/scope_register.md`) was mapped to its counterpart in the figure book: 19 have one and are in the report. Six are not
redrawn: five drew the three-method feature selection of v6, which read the validation and test periods and was replaced
(F-26, D-23), and one repeated the seasonal numbers as a radar (C-84). Twelve process figures are added where a section
of v6 had a sentence and no figure, or where the method needs one (pipeline, operation-limited hours, target, GFS
availability, the selection check, issuance windows, lags, scored hours, validation against test, normal operation,
encoder length, PatchTST, the rebuild chain). Not chosen: the correction figures (C-xx, defense notes only), figures whose
takeaway describes a state before the final run (S09-37, S09-38), and the comparison with the literature (S11-15, a
comparison across normalisations and climates that the report does not need). The map is `docs/report_v7_figures.csv`
and figure S11-20.

**What part C writes.** `docs/report_v7.md`, the whole report in the chapter order of v6 (abstract, chapters 1 to 17,
appendices) with 31 figures, each placed in the section it supports, referenced in the text, and captioned with the
takeaway of its builder; `docs/report_v7_README.md` (the correction note of D-02, data and code availability, the
appendices, and the decisions still open: Q-09, references for SHAP and GFS, `--xgb-only`, the backup);
`docs/report_v7_figures.csv`; figure S11-20 (task S11-C); notebook `11_stage11_part_c.ipynb`; OP-68; T-185 to T-189.

**Self-review before the text.** The original (`submission_evidence/13_report`): **M58**, section 2.1 cites Ding et al.
(2011) with no entry in the reference list and attaches Mellit & Pavan (2010), a neural-network paper, to support vector
regression (C-92). The abstract and introduction of v6 state the route advantage and the agreement of SHAP and attention
as contributions (C2, C6 of C-11); v7 states the contributions that the rebuild supports. My own earlier work: S01-02
counted calendar hours (X113), S01-01 drew the status of stage 9 (X114), the bias shares of section 12.3 used another
aggregation than S11-11 (X115), and first-draft claims of the new chapters were not asserted (X116).

**Rules for the numbers.** As in part B (T-178 to T-184), and: every figure of the report is placed once, numbered in
order and referenced (T-185); every caption is the current takeaway of its builder (T-186); the 25 figures of v6 are each
accounted for (T-187); the README holds the note, the availability, the appendices and the open decisions, and the report
holds no coordinates (T-188); the text and S11-11 give the same shares (T-189).

## 7. Part D: the quantile crossing in the report, with its sources (2026-11-16)

**Request of the user (2026-11-16).** Document the sorting of the quantiles (D-33) in the report, with its sources; follow
the session's recommendations for the rest (close L-14 as future work; close Q-03, Q-11 to Q-14 with the answers given).
Same frame as parts A to C: run `20261114T083930Z_final`, tag `g1-frozen`, no model trained, the step `test` not run, only
paths of the D-36 list edited (`src/solarpv/figures/`, `scripts/make_report_v7.py`, `tests/`, `docs/`).

**Self-review before the code.** The original: the scope register of v6 (S-01 to S-45) lists the quantile loss of the
BiLSTM-Opt (S-17) and of the TFTs, and no interval or quantile result, so there is no number of v6 to correct here; the
order of the quantiles was never checked in v6, and the point forecast of its neural models was the 0.5 head as here.
My own earlier work: **X117**, D-33 (2026-11-12) committed the report to state the share of hours with crossed quantiles
and its cost (on those hours the point forecast is not the median of the sorted distribution); parts B and C wrote only
a line of future work, and the share was measured only on the validation hours of the development run (S09-29), never on
the final run and never on test. **X118**, L-14 (the potential skill of the quantile-loss models) still read "stage 11"
with no decision, and the demonstration in the defence list of PROJECT_MASTER §11 still named
`reproduce_paper_results.py --skip-train`, which Q-30 replaced by `scripts/make_tables_from_frozen.py`.

**What part D computes.** `solarpv.figures.analyses11` writes `quantile_crossing.csv`: for every model with quantiles
(BiLSTM-Opt, three seeds; TFT-96, TFT-168, TFT baseline), route, horizon and split, on the scored hours of `load_frame`
(daytime with truth; the counts equal those of `metrics_kw.csv`): the share of hours whose quantiles are out of order;
the share on which the 0.5 head is not the median of the sorted quantiles, and the mean and largest distance between the
two in kW; and the mean quantile loss in kW, raw and sorted. XGB-Opt and PatchTST forecast a point only.

**Rules for the numbers.** The order is read on the model's own output (k on the indirect route, kW on the direct one),
as in S09-29; the distances and the quantile loss are in kW through the evaluation's `to_kw`, which is monotone, so
sorting and converting commute (asserted). The 0.5 head equals `y_pred` of the prediction file on every row (asserted).
The quantile files are checked against the SHA-256 of their step records, and their test rows are read from the saved
files only and logged (X101). Sorting never raises the quantile loss (asserted per cell). Nothing is decided from the
table: no score, no point forecast and no test changes. The sources are verified against Crossref and OpenAlex with the
user's approval (2026-11-16): Chernozhukov, Fernández-Val & Galichon (2010), Econometrica 78(3), 1093–1125,
doi:10.3982/ECTA7880, for the monotone rearrangement; Koenker & Bassett (1978), Econometrica 46(1), 33–50,
doi:10.2307/1913643, for the quantile loss. T-191 checks the arithmetic on a synthetic frame, T-192 the table, the
report text and the two references. Figure S11-21 (task S11-D) draws the table.

