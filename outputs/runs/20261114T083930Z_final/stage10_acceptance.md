# Stage-10 acceptance, run 20261114T083930Z_final

Date: 2026-11-14. Written after the last phase-1 step (`report`, ok) and after reading
`report/report.md` in full. No test number exists: phase 1 scored validation only.
Reviewed with the `forecasting-validity-review` skill (sections 1-6).

## What was read

- `report/report.md` (509 lines), `evaluate/val/dm_tests_kw.csv` (48 rows), `config.yaml`,
  `outputs/07_evaluation/test_access_log.csv`, and `python -m solarpv.pipeline list`.

## Checks (§4 of the runbook)

1. **All steps complete, one commit.** The 12 phase-1 steps (select, baselines, xgb, cv, ablation,
   bilstm, tft96, tft168, tftbase, patchtst, evaluate, explain, report) are `ok`, all at commit
   `10611a6`, which carries `g1-frozen`. `cv` and `ablation` ran. `n_resumed` is 0 for every step:
   no checkpoint restore, no interruption. `test` is the only step of `list` not run (phase 2).
2. **Warnings (41) read.** 12 are `test_access`: forecasting the test window with the encoder series
   and features as inputs (D-32); none scores a test row. The access log has 12 rows, all
   forecasts, none a score. In all 13 prediction files, `y_true` of test-window rows is blank.
   The rest are matched facts that are already G-1 list points (D-37): edge of the search space (M23),
   13 of 1,080 XGB trials at their cap, two XGB best iterations at the cap (direct D1 seed 123, direct D2
   seed 42), BiLSTM indirect seed 123 best epoch at 60, XGB validation/train RMSE gap 2.1-2.7 on
   indirect D1 and D3. None is a failure.
3. **XGB seed spread in kW:** maximum 0.079 pp (indirect D+2), against the limit of 0.15 pp. Passed.
   BiLSTM maximum 0.074 pp.
4. **kW DM (D-14, Q-19):** XGB is significantly better (HLN, Bonferroni alpha 0.00625) than clear sky x k,
   climatology and persistence at all 6 target x horizon cells (largest HLN p for climatology 0.0029).
   Against PatchTST significant in all 6. Against BiLSTM and the TFTs it is not significant in most
   cells (XGB is nominally best, never significantly worse). Stated limitation (D-37): at D+2 the
   climatology margin is 91-93 % of the minimum detectable effect.
5. **Budgets and time per model** are those of v6 (`config.yaml`). Seconds: xgb 3,214, cv 121,
   ablation 389, bilstm 4,706, tft96 8,701, tft168 12,210, tftbase 722, patchtst 899, explain 333.
   GPU steps (bilstm, tft96, tft168, tftbase, patchtst) 27,238 s in total (7.6 h), on GRID P40-24Q.

## Validity review

- No leakage indicator: the top SHAP features are `fcst_gfs_*` (issued before T0) and clear-sky;
  no same-hour power, no dominating single feature by a large factor.
- Red flag examined: error falls from D+1 to D+3 for XGB and TFT on validation. This is the known
  validation-sample effect of R6b-07/R7-05; the 5-fold CV inside train shows the expected order
  (indirect: 7.4, 7.7-7.8, 7.9-8.0 for D+1, D+2, D+3), and validation is better than CV by more at D+3
  (-1.4 to -1.7 pp). No action.
- Same rows for every model (n = 2,901 daytime hours with truth in each cell).
- Validation numbers are tuning-split numbers (D-10); they decide nothing about the test split.
- Positive bias of the neural forecasts, BiLSTM final-fit gap to the best trial (up to 0.25 pp) and the PatchTST
  best step near the budget are G-1 list points (D-37), reported as limitations.

## Verdict

**Accepted.** Phase 2 (`--allow-test`) may run on a separate decision of the user. Nothing was
committed or edited in a tracked file during the run.
