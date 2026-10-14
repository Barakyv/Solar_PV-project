# Revision Action Checklist
**Solar PV Forecasting Capstone — Ktora Plant**
**Audit date:** 2026-05-11 | **Auditor:** Claude Sonnet 4.6

All items are sourced from `REPORT_CLAIM_EVIDENCE_MATRIX.csv`, `NUMERICAL_CONSISTENCY_AUDIT.md`, and `MISSING_PROJECT_WORK_AUDIT.md`. Each item is traceable to a project artifact.

---

## CRITICAL — Must Fix Before Any Submission

These items contain factual errors verified by direct CSV evidence. No submission should proceed without these corrections.

- [ ] **C1. Correct XGB-Opt direct D+1 nRMSE: 12.66% → 12.06%**
  - Find all occurrences (abstract, benchmark table, Fig 1, Conclusion C1, recommendation section)
  - Source: `outputs/07_evaluation/benchmark_summary.csv` row `xgb_sota,direct,1` → 12.0595%

- [ ] **C2. Correct TFT-96 direct D+1 nRMSE: 12.66% → 12.11%**
  - Same fix pattern as C1; verify in benchmark table
  - Source: `benchmark_summary.csv` row `tft_96,direct,1` → 12.1132%

- [ ] **C3. Correct TFT baseline direct D+1 nRMSE: 13.54% → 12.80%**
  - TFT baseline and TFT-168 direct-target values appear SWAPPED
  - Source: `benchmark_summary.csv` row `tft,direct,1` → 12.7982%

- [ ] **C4. Correct TFT-168 direct D+1 nRMSE: 12.8% → 13.26%**
  - Part of the same swap as C3
  - Source: `benchmark_summary.csv` row `tft_168,direct,1` → 13.2643%

- [ ] **C5. Correct XGB Winter indirect D+1 seasonal nRMSE: 27.0% → 32.22%**
  - Most numerically severe error (5.2 pp); update in text, Table/Fig 3 heatmap, and seasonal section
  - Source: `outputs/07_evaluation/seasonal_metrics.csv` row `xgb_sota,indirect,1,Winter` → 32.221%

- [ ] **C6. Correct leakage feature count: 17 → 19**
  - `docs/leakage_audit.md` lists: Category 1 = 7, Category 2 = 10, Category 3 = 2 → total = 19
  - Update all occurrences of "17 leakage features" to "19 leakage features"

- [ ] **C7. Add mandatory 05e vs 05f distinction statement**
  - Insert: "Stage 05e produced the official modelling feature sets. Stage 05f is a post-hoc audit artifact and did not alter the modelling inputs."
  - Source: `artifacts/05_feature_selection/refined_features_indirect_D1_05f.json` — disclaimer field

- [ ] **C8. Add leakage–feature-selection interaction disclosure**
  - Insert: "The 05e feature JSONs include same-timestamp NWP features subsequently classified as leakage; these were excluded at training time via the EXCL list in `xgb_sota.py` lines 51–73."
  - Source: `final_features_indirect_D1.json` contains fcst_ghi/dni/dhi in recommended set despite leakage_audit.md classifying them as leakage

---

## MAJOR — Fix Before Submission

These items involve meaningful inaccuracies or disclosure gaps that could mislead reviewers.

- [ ] **M1. Correct XGB seed max spread (indirect): 0.095 pp → 0.111 pp**
  - Source: `seed_variability_summary.csv` → nRMSE_max=29.095 nRMSE_min=28.984 → range=0.111 pp

- [ ] **M2. Correct XGB seed max spread (direct): 0.125 pp → 0.092 pp**
  - Source: `seed_variability_summary.csv` → D+3 range = 12.185−12.093 = 0.092 pp

- [ ] **M3. Add BiLSTM seed variability disclosure**
  - Clarify: LSTM seed metrics in seed table are training-set metrics, NOT daytime-evaluated day-ahead nRMSE
  - Authoritative LSTM daytime eval = 30.24% (seed 42 only, from `benchmark_summary.csv`)
  - Ready-to-paste text in `PROPOSED_REPORT_ADDITIONS.md` Section E

- [ ] **M4. Quantify LSTM timestamp bug with before/after values**
  - Read `outputs/06_lstm_sota/align_seed42.log` to confirm pre-correction D+1 ≈ 17%
  - If confirmed: add disclosure with exact formula and before/after nRMSE
  - Ready-to-paste text in `PROPOSED_REPORT_ADDITIONS.md` Section D

- [ ] **M5. Verify and clarify DM test n_pairs range**
  - Report states n=2,554; actual range is 2,510 (LSTM) to 2,580 (PatchTST) depending on model
  - Add: "n_pairs ranges from 2,510 to 2,580 depending on model input window"
  - Source: `dm_tests.csv` n_pairs column

- [ ] **M6. Correct route comparison lower bound precision**
  - Current: "1.7–2.1 pp" → Correct: "1.66–2.10 pp"
  - LSTM D+1 advantage = 1.655 pp (route_comparison.csv); rounds to 1.7 only with aggressive rounding
  - Source: `route_comparison.csv`

- [ ] **M7. Expand power analysis range to full horizon set**
  - Current: "38–39% of MDE" (D+1 only) → Full range: "29–50% of MDE across all horizons"
  - Source: `dm_tests_extended.csv` pct_of_mde column (indirect: 39.3/44.4/50.0%; direct: 38.1/35.2/29.0%)

- [ ] **M8. Find and cite source for inverter downtime 13.5% claim**
  - This value was not found in any of the standard evaluation CSVs
  - Either add source citation or soften: "approximately X% of daytime test hours [SOURCE]"

---

## MODERATE IMPROVEMENTS — Complete Before Final Grading Submission

- [ ] **Mo1. Add Data Science Milestone Table**
  - Ready-to-paste in `PROPOSED_REPORT_ADDITIONS.md` Section B
  - Insert after Introduction or at start of Methodology

- [ ] **Mo2. Add feature count table by selection stage**
  - Shows pipeline: 86 → 42 → 40 (05e) → 24 (05f recommended) → ~31–33 (effective after EXCL)
  - Source: `final_features_indirect_D1.json` (n_starting_features=42, consensus n=40)

- [ ] **Mo3. Clarify starting pool 86 vs 42 discrepancy**
  - Report says "~86 engineered variables"; 05e JSON shows 42 entering the consensus step
  - Add: "The initial pool of ~86 features was reduced to 42 following Stage 1 domain-knowledge filtering before the consensus step"

- [ ] **Mo4. Add SHAP cross-horizon rank stability values**
  - Cite Spearman ρ = 0.51–0.80 from `shap_horizon_corr.csv`
  - Supports claim that feature importance is consistent across D+1/D+2/D+3

- [ ] **Mo5. Add common-mask n values explicitly**
  - State n=2,116 (direct D+1), n=2,109 (indirect D+1) from `common_mask_summary.csv`
  - Ready-to-paste in `PROPOSED_REPORT_ADDITIONS.md` Section H

- [ ] **Mo6. Verify coordinate sensitivity source**
  - Claims Δ < 2% on all metrics — find the script/CSV that generated these values
  - If verified, add source citation

- [ ] **Mo7. Regenerate any figure containing incorrect direct-target values**
  - Any figure with XGB direct D+1 = 12.66% must be regenerated with 12.06%
  - Especially: Fig 1 (benchmark overview), any scatter plots for direct target

- [ ] **Mo8. Add 05f tier counts per horizon**
  - At minimum for indirect D+1: CORE=15, SUPPORTED=8, CONDITIONAL=1, EXCLUDED=18
  - Source: `refined_features_indirect_D1_05f.json`

- [ ] **Mo9. Add permutation importance role clarification**
  - 05e JSON includes permutation_importance as a 4th method but voting is 3-method (SHAP+Stability+MI)
  - Clarify in Feature Selection section

- [ ] **Mo10. Verify CV output value (16.08% for XGB indirect)**
  - Not found in standard CSVs — must come from 07_evaluation notebook or separate CV output
  - If correct, cite the source file

---

## OPTIONAL APPENDIX ADDITIONS

- [ ] **Op1. Append full feature lists per target/horizon (05e official sets)**
  - 6 JSON files × 40 features each — too long for body; suitable as appendix table
  - Note which features appear in all 6 sets (FORECAST_SAFE, horizon-independent)

- [ ] **Op2. Optuna hyperparameter search space table**
  - Move from body text to appendix if present in main body
  - Confirm search space values against training scripts

- [ ] **Op3. PatchTST rolling window protocol detail**
  - 243 daily forecast origins; last 2×INPUT_SIZE historical observations; neuralforecast 3.1.7

- [ ] **Op4. Error decomposition results table**
  - XGB bias² < 0.15% of MSE; LSTM ~1.5–2% positive bias
  - Source: `scripts/run_error_decomposition.py` output (not yet read)

- [ ] **Op5. Production quintile analysis table**
  - Q1–Q5 nRMSE and bias values from quintile analysis output (not yet read)
  - If figures are present (Fig 16) the table can be appendix

- [ ] **Op6. Add Spearman correlation matrix for SHAP horizon stability**
  - Full 6×6 matrix from `shap_horizon_corr.csv`

- [ ] **Op7. Leakage audit feature table**
  - Full table of all 19 leakage features with category and reason for exclusion
  - Ready reference: `docs/leakage_audit.md` can be reproduced as appendix table

---

## ITEMS REQUIRING MANUAL HUMAN CONFIRMATION

These items cannot be verified by automated CSV comparison and require the researcher to check source material directly.

| Item | Question | Risk if unconfirmed |
|------|----------|-------------------|
| LSTM pre-correction D+1 nRMSE | Read `align_seed42.log`: was pre-correction D+1 ≈ 17%? Exact value needed. | Report claim "~17%" unverifiable |
| Inverter downtime 13.5% | What is the source calculation? Which script/log produces this count? | Unsupported disclosure |
| Summer nRMSE estimate 5–8 pp | Is this estimate from a model run or an analytical extrapolation? Add hedge if estimate only. | Overclaim |
| TFT-96 Optuna importance 46.7%/64.8% | Find the Optuna hyperparameter importance output (likely in TFT training notebook). | Specific unverifiable numbers |
| CV output XGB indirect = 16.08% | Find the cross-validation output file (not in benchmark_summary.csv). | Train-test gap claim unverifiable |
| Quintile nRMSE values (Q1=7.85%, Q2=20.71%, Q4=6.88%, Q5=8.78%) | Find the quintile analysis CSV or figure source. | All quintile claims unverifiable |
| Q5 bias = -258 kW | Same — find quintile analysis source. | |
| TFT attention figure shows -24h/-48h/-168h peaks | Visual inspection of Fig 7 / fig19_tft_attention.png needed. | Cross-architectural convergence claim partially unverifiable |
| 05e JSON leakage features classified as CORE in 05f | Check all 6 05f JSON files — do fcst_ghi/dni/dhi appear in CORE tier for any of them? | Misleading tier assignments for leakage features |

---

## Quick Priority Summary

| Priority | Count | Examples |
|----------|-------|---------|
| CRITICAL (must fix) | 8 | XGB direct 12.06%, Winter 32.22%, leakage count 19, 05e vs 05f distinction |
| MAJOR (fix before submission) | 8 | Seed spread correction, BiLSTM disclosure, LSTM bug quantification |
| MODERATE (improve before grading) | 10 | DS milestone table, feature count table, SHAP ρ values |
| OPTIONAL appendix | 7 | Feature lists, Optuna table, error decomposition |
| Manual confirmation | 9 | LSTM log, inverter downtime, Summer estimate, quintile source |
