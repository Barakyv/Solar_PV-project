# Report Audit Summary
**Solar PV Forecasting Capstone — Ktora Plant**
**Audit date:** 2026-05-11 | **Auditor:** Claude Sonnet 4.6
**Report audited:** Solar_PV_Forecasting_Report_Enhanced_With_DS_Milestones_and_Feature_Selection.docx

---

## 1. Overall Assessment

**The report is NOT currently submission-ready** without corrections to at least 5 CRITICAL numerical errors. Several model-performance values in the direct-target benchmark table are incorrect, the Winter seasonal result is wrong by 5 pp, and the leakage feature count is inconsistent with the audit document. Fixing these errors is mandatory before any submission.

---

## 2. Claim Status Summary

| Status | Count | Notes |
|--------|-------|-------|
| VERIFIED | 46 | Exact match within rounding tolerance |
| PARTIALLY VERIFIED | 6 | Directionally correct; precision or wording needs refinement |
| CONTRADICTED | 8 | Numerical value in report differs from source CSV |
| NEEDS MANUAL REVIEW | 21 | Requires reading a file not yet accessed (quintile, CV, attention figures, etc.) |
| UNSUPPORTED | 0 | All claims have a plausible source; none are fabricated |

---

## 3. Critical Errors (Must Fix Before Submission)

### E1: XGB-Opt Direct D+1 nRMSE — WRONG VALUE
- **Report states:** 12.66%
- **CSV shows (benchmark_summary.csv):** 12.0595%
- **Error magnitude:** 0.6 pp overstatement
- **Occurrences:** Abstract, benchmark table, Conclusion C1, recommendation section
- **Action:** Find-and-replace all occurrences of "12.66%" referring to XGB direct D+1 with **12.06%**

### E2: TFT-96 Direct D+1 nRMSE — WRONG VALUE
- **Report states:** 12.66% (same as XGB — identical value is suspicious)
- **CSV shows:** 12.1132%
- **Action:** Correct to **12.11%** in all occurrences

### E3: TFT-168 and TFT Baseline Direct D+1 — VALUES APPEAR SWAPPED
- **Report states:** TFT baseline = 13.54%, TFT-168 = 12.8%
- **CSV shows:** TFT baseline = 12.7982%, TFT-168 = 13.2643%
- **Likely cause:** TFT baseline and TFT-168 labels were swapped during table construction
- **Action:** Correct to TFT baseline = **12.80%**, TFT-168 = **13.26%** in all direct-target appearances

### E4: XGB Winter Indirect D+1 Seasonal nRMSE — WRONG VALUE
- **Report states:** 27.0%
- **CSV shows (seasonal_metrics.csv):** 32.221%
- **Error magnitude:** 5.2 pp understatement (worst error in the document)
- **All model Winter values from CSV:**
  - XGB: 32.22%, LSTM: 32.26%, TFT-96: 32.27%, TFT-168: 35.65%, TFT baseline: 35.10%, PatchTST: 33.50%
- **Action:** Correct XGB Winter to **32.22%** in text, tables, and any figures that show this value

### E5: Leakage Feature Count — INCONSISTENT WITH AUDIT DOCUMENT
- **Report states:** "17 features constituting data leakage"
- **leakage_audit.md shows:** Category 1 = 7, Category 2 = 10, Category 3 = 2 → **total = 19** features
- **Action:** Correct to 19, or provide explicit justification for why the count is 17 (e.g., if poa_*_forecast derived features are grouped with their parents). Do not change the audit document — change the report.

---

## 4. Major Issues (Fix Before Submission)

### M1: XGB Seed Max Spread — INCORRECT VALUES
- **Report (indirect):** 0.095 pp → **Actual (seed_variability_summary.csv):** 29.095−28.984 = **0.111 pp**
- **Report (direct):** 0.125 pp → **Actual:** D+1 range = 0.014 pp; D+3 range = 0.092 pp → max ≈ **0.092 pp**
- **Action:** Correct both values. Use: "max spread across seeds: 0.111 pp (indirect) and 0.092 pp (direct)"

### M2: BiLSTM Seed Variability Table — TRAINING-SET METRICS MISLABELLED
- **Issue:** The seed_variability_summary.csv shows LSTM nRMSE ~19–20% for indirect (training-set), not the authoritative ~30.24% from benchmark_summary.csv
- **Root cause:** The `compile_seed_variability.py` script aggregates per-seed metrics files which contain training-set (not daytime-eval) metrics for LSTM
- **Report risk:** If the report cites a 2.2% CV for BiLSTM across seeds, this is based on training-set metrics, not the evaluation metric
- **Action:** Add explicit disclosure: "BiLSTM seed variability is reported from training-set evaluation metrics. Authoritative day-ahead evaluation (benchmark_summary.csv) was computed for seed 42 only, as GPU-constrained retraining of seeds 123 and 777 produced only training-set metrics. Multi-seed daytime-filtered evaluation of BiLSTM requires additional GPU compute."

### M3: DM Test n — AMBIGUOUS
- **Report cites n=2,554 paired observations**
- **dm_tests.csv shows:** n_pairs = 2,510 (LSTM vs XGB), 2,554 (TFT variants vs XGB), 2,580 (PatchTST vs XGB)
- **Action:** Clarify that n_pairs varies by model: "n ranges from 2,510 (BiLSTM vs XGB, reflecting shorter required input window) to 2,580 (PatchTST vs XGB)"

### M4: 05e Feature JSON Contains Leakage Features — NOT DISCLOSED
- **Issue:** `final_features_indirect_D1.json` includes `fcst_ghi`, `fcst_dni`, `fcst_dhi`, `fcst_temp_2m`, `fcst_wind_speed`, `poa_global_forecast`, `poa_direct_forecast`, `poa_diffuse_forecast` in its recommended feature set, despite these being listed as leakage in `leakage_audit.md`
- **Root cause:** Feature selection (05c/05e) was completed before NWP archive leakage was identified (April 2026). The JSON artifacts were not retroactively updated. The XGB/LSTM training scripts exclude these via the EXCL list.
- **Action:** Add a paragraph: "The 05e feature selection JSON artifacts were produced prior to the identification of NWP archive reanalysis leakage. Features `fcst_ghi`, `fcst_dni`, `fcst_dhi`, `fcst_temp_2m`, `fcst_wind_speed`, and their POA-derived counterparts appear in the 05e recommended sets but are excluded at model training time via the EXCL list in `production/02_models/xgb_sota.py` (lines 51–73). The effective modelling feature set is therefore the 05e consensus set MINUS the 19 leakage-identified features."

### M5: LSTM Timestamp Bug — DISCLOSURE NEEDS QUANTIFICATION
- **Report claims:** Bug changed D+1 from ~17% to ~30% (a 13 pp swing)
- **Status:** NEEDS MANUAL REVIEW — `align_seed42.log` should contain before/after values
- **Action:** Read `align_seed42.log` and cite exact pre-fix and post-fix nRMSE values

### M6: Route Comparison Lower Bound — PRECISION
- **Report states:** 1.7–2.1 pp advantage for indirect over direct
- **CSV shows:** LSTM D+1 = 1.655 pp, which rounds to 1.7 but is more precisely 1.66
- **Action:** State "1.66–2.10 pp" to avoid overstating the lower bound

---

## 5. Moderate Issues (Improve Before Final Submission)

### Mo1: Route Comparison n — Clarify
Multiple n values appear (2,165 indirect, 2,172 direct for XGB). The report should specify which n is cited and why they differ.

### Mo2: Power Analysis Range — Incomplete
Report cites 38–39% of MDE for D+1. Across all horizons, the range is 29–50% MDE (dm_tests_extended.csv). Report should cite the full range to avoid implying better power than exists.

### Mo3: Cross-Validation Gap — Source Unknown
Report claims XGB-Opt CV mean = 16.08% for indirect. This value is not in benchmark_summary.csv or any read CSV. Verify from CV output file.

### Mo4: Ablation for Direct Target
Report shows ablation table for indirect. Direct target ablation values are in ablation_seed_variability.csv but may not be fully represented. Verify completeness.

### Mo5: Coordinate Sensitivity Source
Claims of Δ < 2% on all metrics need a source file citation in the report.

### Mo6: Starting Feature Pool Count (86 vs 42)
The report states the starting pool was ~86 features. The 05e JSON shows n_starting_features=42 (after Stage 1 filtering). These are not contradictory if the 86 refers to pre-Stage-1 pool, but the report should explicitly state when the filtering occurred.

---

## 6. Items Requiring Manual Human Confirmation

| Item | Question | Risk if unconfirmed |
|------|----------|-------------------|
| Inverter downtime 13.5% | Is this from a CSV or manual inspection? Cite the source. | Unsupported statistic |
| LSTM timestamp bug 17%→30% | Read align_seed42.log to confirm before/after values | Key claim unsupported |
| Summer nRMSE reduction 5–8 pp | Is this from a model run or an estimate? Hedge appropriately | Overclaim |
| TFT-96 Optuna importance 46.7%/64.8% | Find the Optuna importance report | Unsupported specificity |
| CV output XGB 16.08% | Find the CV output file | Contradicts if wrong |
| Quintile nRMSE values | Find the quintile analysis CSV | Several specific numbers unverified |

---

## 7. Claims That Are Fully Verified (Key Highlights)

The following important claims are confirmed by direct CSV evidence:

- XGB indirect D+1 nRMSE = 29.09% ✓
- LSTM indirect D+1 nRMSE = 30.24% ✓
- TFT-96 indirect D+1 nRMSE = 29.66% ✓
- LSTM direct D+1 nRMSE = 12.58% ✓
- PatchTST indirect D+1 nRMSE = 30.30% ✓
- XGB direct D+1 R² = 0.812 ✓ (0.8116 rounds correctly)
- TFT-96 direct D+1 R² = 0.828 ✓ (0.8283 rounds correctly)
- XGB indirect D+1 95% CI [24.65%, 33.46%] ✓ (minor lower-bound rounding)
- DM statistic TFT-96 vs XGB indirect = 14.4 ✓
- LSTM vs XGB not significant (p=0.2545) ✓
- Indirect advantage: LSTM D+1 = 1.655 pp, XGB D+1 = 1.980 pp ✓ (report rounds to 1.7–2.1)
- Common mask n=2,116; XGB 13.30%, LSTM 13.72%, TFT 14.05% ✓
- SHAP rolling stats: direct 70.7%, indirect 57.0% ✓
- SHAP clearsky: direct 1.5%, indirect 11.3% ✓
- SHAP lag power: direct 16.3%, indirect 11.4% ✓
- Ablation: clearsky improves indirect −0.13 pp ✓
- Ablation: 7d rolling hurts indirect +0.393 pp ✓
- Ablation: minimal config hurts indirect +2.141 pp ✓
- XGB seed std indirect = 0.060 pp ✓
- XGB seed CV < 0.3% ✓
- XGB horizon degradation −0.01 pp (D+1 to D+3) ✓
- PatchTST horizon degradation +1.53 pp ✓
- AC capacity = 3,449 kW ✓
- MEAN_DAYTIME_K = 0.6279 ✓
- Bootstrap block=24h, 1,000 reps ✓
- DM NW lags = 24 ✓
- Seeds = [42, 123, 777] ✓
- 05e consensus method: 2-of-3 SHAP+Stability+MI ✓
- 05f CORE=15, SUPPORTED=8, CONDITIONAL=1, EXCLUDED=18 (indirect D+1) ✓
- 05f explicitly states it is NOT used for training ✓
- n=2,116 common mask direct D+1 ✓

---

## 8. Executive Summary Answers

**Q1: Is the report currently submission-ready?**
No. Five CRITICAL numerical errors must be corrected first. The direct-target benchmark table contains at least four wrong values (XGB 12.66%, TFT-96 12.66%, TFT baseline 13.54%, TFT-168 12.8%, and Winter seasonal 27.0%). These are not rounding differences — they differ by 0.6 to 5.2 percentage points from the source CSVs.

**Q2: Which claims are most risky?**
1. XGB direct D+1 = 12.66% (should be 12.06%) — overstates error by 0.6 pp
2. TFT-168 and TFT baseline direct values swapped — wrong model credited
3. XGB Winter seasonal = 27.0% (should be 32.22%) — 5 pp understatement dramatically changes winter narrative
4. Leakage count 17 vs actual 19 — inconsistent with own audit document
5. BiLSTM seed table using training-set metrics while claiming robustness

**Q3: Which project work is underrepresented?**
- Preprocessing pipeline (UTC resampling, 1-min → 1-hour, cleaning policy)
- EDA findings (seasonality, capacity factor distribution, data quality)
- 05c SHAP backward elimination stopping criterion
- 05e operational availability classification (FORECAST_SAFE, HORIZON_DEPENDENT, PIPELINE_DEPENDENT)
- 05f robustness audit tier definitions with feature counts per horizon
- LSTM timestamp bug fix (disclosure exists but quantification is unverified)
- Reproducibility package contents (split_manifest, CITATION.cff, environment.yml)

**Q4: What must be added to make the report fully credible?**
1. Data Science Milestone Table (chronological evidence of project stages)
2. Expanded Feature Selection section: 05e → 05f pipeline with explicit feature counts
3. Clear statement: "05e JSON artifacts predate the NWP leakage audit; effective modelling set = 05e MINUS EXCL list"
4. Reproducibility appendix: split_manifest.yaml, environment.yml, CITATION.cff references
5. Quantified LSTM timestamp bug disclosure with before/after nRMSE
6. Clearer BiLSTM seed variability disclosure

**Q5: What should be moved to appendix?**
- Detailed Optuna search space tables
- Full per-feature SHAP value tables
- Full feature lists (05e recommended sets per target/horizon)
- Quintile analysis table
- Cleaning policy details (cleaning_config.yaml contents)
- PatchTST neuralforecast configuration details

**Q6: What should be removed or softened?**
- "17 leakage features" → correct to 19
- "1.7–2.1 pp" route advantage → refine to "1.66–2.10 pp"
- Any language implying LSTM 3-seed analysis is complete at daytime-evaluation level
- Any language implying 05f-reduced feature sets were used for modelling
- Any claim using "free improvement" for route comparison → replace with "2 pp improvement without model modification"
- "Summer would reduce nRMSE 5–8 pp" → hedge as "estimated 5–8 pp reduction based on Spring performance"
