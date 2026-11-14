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
