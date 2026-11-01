---
name: forecasting-validity-review
description: Locally authored for this project (not downloaded). Use before accepting any evaluation, selection or model result of the Ktora day-ahead PV forecasting rebuild - checks issuance-time availability, leakage, split isolation, identical evaluation windows, baselines, uncertainty and provenance against the project's own pipeline.
metadata:
  origin: locally authored, 2026-11-01, for C:/Users/baraky/Desktop/solar (not from any skill repository)
  source-of-truth: PROJECT_MASTER.md (spec §6, decisions §3, tests §8)
---

# Forecasting validity review (Ktora rebuild)

A checklist to run **before** a number is accepted, written to a record, or shown to the user.
It points at the code and tests that already enforce each rule; it does not replace them.
PROJECT_MASTER.md wins over this file wherever they differ.

## When to use

- A new evaluation, selection, value check or model result has just been produced.
- A result looks better than expected, or horizons are in an unexpected order.
- Before an OP entry, a run-journal entry, a notebook note or a report sentence quotes a number.

Not for exploratory plots that decide nothing.

## 1. Information available at issuance

- T0 = 00:00 local (Asia/Jerusalem) of the first forecast day; rows are left-labelled hours in
  true UTC (F-01). `solarpv.features.build.issuance_utc`.
- Every past-day value must end by T0: `build.past_value` (DST guard), lags per horizon
  (`build.kept_lags`: no lag_1d at D+2; no lag_1d or lag_2d at D+3), rolling windows h … h+N−1
  days back (F-07). Tests T-09, T-34.
- Imputation uses only earlier days (`build.impute_same_hour`, F-24); no bfill, no whole-column
  statistics.
- Weather: only GFS runs whose every file was uploaded before T0 (`nwp.download.late_files`,
  T-17 row by row). Observed or archive values at the target hour are never inputs (F-11, F-28).
- Inputs come from an allow-list and pass `features.leakage.assert_no_leakage` (no same-hour
  AC/DC power, no theoretical power, no `fcst_*` without the `fcst_gfs_` prefix). F-27.

## 2. Split isolation

- Splits: train ≤ 2024-01-31; validation 2024-02 to 2024-09; test 2024-10 to 2025-05
  (`configs/split_manifest.yaml`).
- Everything fitted or chosen (thresholds, imputers, feature selection, hyper-parameters,
  model choice) uses train, or train folds; validation only where a decision rule says so.
  Selection reads train rows only (`selection.common.train_table`, F-26, T-47).
- The test split is evaluated only with `allow_test=True` after `g1-frozen`; every access is
  logged in `outputs/07_evaluation/test_access_log.csv`. Any other test access is a defect.
- Inner validation is chronological: expanding date folds with a gap
  (`selection.common.date_folds`), never shuffled rows.

## 3. Comparable evaluation

- Truth from the stage-6 feature table, never from a prediction file
  (`evaluation.truth.load_truth`).
- Same hours for every model: all daytime hours with truth (C-35); the two routes paired on the
  same hours with actual AC as truth (C-34). `evaluate.check_alignment`, `check_coverage` must
  be empty.
- One metric definition: nRMSE in % of 3,449 kW; for k, prediction × theo_kw first (and
  nRMSE_mean as a second normalisation, L-16). Never compare k-unit and kW-unit errors.
- Report against the references on the same rows: persistence (DST-safe), climatology, clear
  sky × MEAN_DAYTIME_K, and CLIPER skill.
- Views: all hours, normal operation, data source (D-21, D-22).

## 4. Uncertainty and decisions

- A gain counts only if it beats a margin fixed in advance: max(0.05 pp, 2 × pooled seed std)
  (the D-12 rule, `nwp.check.decide`). Three seeds (42, 123, 777) for anything compared.
- Model comparisons: Diebold–Mariano with HLN and Bonferroni, block-bootstrap intervals
  (`evaluation.stats_tests`, `metrics.bootstrap_mean_ci`). A small metric difference alone is
  not significance.
- Decisions on validation are taken once, with the rule written before the run; a rerun after a
  bug fix is recorded as such in the run journal.

## 5. Red flags (stop and investigate with systematic-debugging)

- One feature dominating importance by a large factor (the original DC power: 17 times the
  third feature).
- Error falling with the horizon, or D+3 better than D+1 (check the mapping, then an inner
  train period and a paired bootstrap, as in R6b-07).
- A model beating the references by an implausible margin, or matching a perfect-information
  benchmark.
- Performance changing sharply between inner-train folds and validation.
- A selected set clearly worse than the full pool on validation (stage 7, S8).

## 6. Provenance

Before the result is used: RunLog file under the stage's `runs/`; manifest with git commit,
input hashes and seeds; OP entry in `docs/data_processing_record.md`; run-journal entry; numbers
regenerated with `python scripts/build_data_record.py` from the project root; notebook rebuilt by
its builder. No number is
typed by hand into a note or document.
