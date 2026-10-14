# Numerical Consistency Audit
**Solar PV Forecasting Capstone — Ktora Plant**
**Audit date:** 2026-05-11 | **Auditor:** Claude Sonnet 4.6

This document checks for internal contradictions within the report, independent of CSV verification. All discrepancies are classified by location, conflicting values, likely cause, and required fix.

---

## 1. Direct-Target Benchmark Table — Multiple Errors

### 1.1 XGB-Opt Direct D+1 nRMSE
- **Locations in report:** Abstract, benchmark table (Fig 1/Table), Conclusion C1, recommendation
- **Value stated:** 12.66%
- **Correct value (benchmark_summary.csv):** 12.06%
- **Magnitude:** 0.6 pp overstatement
- **Likely cause:** Carry-over from an older evaluation run (pre-NWP-leakage-fix or different normalization)
- **Acceptable with explanation:** No — this is not a rounding difference; the value is factually wrong
- **Required fix:** Replace "12.66%" with "12.06%" in all occurrences (confirmed in benchmark_summary.csv row: `xgb_sota,direct,1,2580,12.0595`)

### 1.2 TFT-96 Direct D+1 nRMSE
- **Locations:** Benchmark table
- **Value stated:** 12.66% (same as XGB — identical values are a red flag)
- **Correct value:** 12.11%
- **Magnitude:** 0.55 pp overstatement
- **Likely cause:** Same stale evaluation run; or copy-paste of XGB value
- **Required fix:** Replace with "12.11%" (`tft_96,direct,1,2554,12.1132`)

### 1.3 TFT Baseline Direct D+1 nRMSE
- **Locations:** Benchmark table, discussion (TFT baseline underperformance claim)
- **Value stated:** 13.54%
- **Correct value:** 12.80%
- **Magnitude:** 0.74 pp overstatement
- **Likely cause:** TFT baseline and TFT-168 direct-target values appear SWAPPED in the report table
- **Required fix:** Replace with "12.80%" (`tft,direct,1,2554,12.7982`)

### 1.4 TFT-168 Direct D+1 nRMSE
- **Locations:** Benchmark table
- **Value stated:** 12.8% (this matches TFT baseline, not TFT-168)
- **Correct value:** 13.26%
- **Magnitude:** 0.46 pp understatement
- **Likely cause:** Swap with TFT baseline (see 1.3 above)
- **Required fix:** Replace with "13.26%" (`tft_168,direct,1,2554,13.2643`)

**Net effect of 1.2–1.4:** The direct-target ranking changes. Corrected ranking for direct D+1:
1. XGB-Opt: 12.06%
2. TFT-96: 12.11%
3. LSTM: 12.58%
4. TFT baseline: 12.80%
5. TFT-168: 13.26%
6. PatchTST: 13.84%

The report implies TFT-96 and XGB are tied at 12.66% — this is false. XGB is worse than TFT-96 by 0.05 pp on direct D+1, and the correct ranking preserves XGB as best only on indirect.

---

## 2. Winter Seasonal nRMSE — Major Error

### 2.1 XGB-Opt Winter Indirect D+1
- **Locations:** Seasonal section, seasonal heatmap (Fig 3), any sentence about winter performance
- **Value stated:** 27.0%
- **Correct value (seasonal_metrics.csv):** 32.22%
- **Magnitude:** 5.2 pp understatement (most severe single error in the document)
- **Likely cause:** The 27.0% may be from an earlier evaluation run on a different winter subset, or it represents internal validation (not test-set) winter performance
- **Acceptable:** No — 5 pp is not a rounding issue
- **Required fix:** Replace "27.0%" with "32.22%" for XGB winter indirect D+1

All model Winter indirect D+1 values (from seasonal_metrics.csv) for reference:
| Model | Winter indirect D+1 nRMSE |
|-------|--------------------------|
| XGB SOTA | 32.22% |
| LSTM SOTA | 32.26% |
| TFT-96 | 32.27% |
| TFT baseline | 35.10% |
| TFT-168 | 35.65% |
| PatchTST | 33.50% |

Note: If the report claims "Winter performance [27.0%] is better than Fall [34.7%]" — this ordering is CORRECT (winter is better than fall), but the magnitudes are wrong. The corrected narrative: Winter (32.22%) is better than Fall (34.74%) but worse than Spring (22.33%).

---

## 3. Model Naming Inconsistencies

### 3.1 Report names vs CSV names
| Report name | CSV name | Consistent in report? |
|-------------|----------|----------------------|
| XGB-Opt | xgb_sota | Generally consistent; verify "XGBoost Optimized" vs "XGB-Opt" in all sections |
| BiLSTM-Opt | lstm_sota | Report uses "BiLSTM-Opt" throughout; CSV uses "lstm_sota" — acceptable divergence |
| TFT-96 | tft_96 | Consistent |
| TFT-168 | tft_168 | Consistent |
| TFT baseline | tft | Verify "TFT baseline" is never called "TFT" alone in tables (could be confused with TFT-96/168) |
| PatchTST | patchtst | Consistent |

**Action:** Ensure every table header and figure caption uses the exact same name as the body text. The term "TFT" alone is ambiguous — always use "TFT baseline" to distinguish from TFT-96 and TFT-168.

---

## 4. nRMSE Denominator Ambiguity

### 4.1 Indirect vs Direct normalization
- Indirect nRMSE: denominator = MEAN_DAYTIME_K = 0.6279
- Direct nRMSE: denominator = AC_CAPACITY_KW = 3,449 kW
- Route comparison: both targets re-expressed at kW scale (denominator = 3,449 kW for both)

**Risk:** The report must not compare indirect nRMSE (0.6279-normalized) directly to route-comparison nRMSE (3449-normalized) without noting the denominator shift.

From route_comparison.csv:
- XGB indirect→power D+1 at kW scale = **11.16%** (different from indirect nRMSE = 29.09%)
- XGB direct D+1 at kW scale = **13.14%** (similar to direct nRMSE = 12.06%)

The difference: indirect nRMSE 29.09% is normalized by 0.6279; the same predictions normalized by 3,449 kW give 11.16%. The report must clarify which normalization is used in each table.

**Action:** Every table presenting nRMSE values must label which denominator is used. Route comparison section must explicitly state: "Both targets re-expressed at AC power scale (3,449 kW denominator) for fair comparison."

---

## 5. Sample Size n — Inconsistencies

### 5.1 n varies by model — not always stated
| Model | n (benchmark_summary.csv D+1 indirect) |
|-------|----------------------------------------|
| XGB SOTA | 2,580 |
| LSTM SOTA | 2,510 |
| TFT variants | 2,554 |
| PatchTST | 2,580 |

Report claims: "~2,580 daytime test hours" — this is ONLY correct for XGB and PatchTST. LSTM has 2,510 and TFT has 2,554.

**Action:** State: "XGB-Opt and PatchTST evaluate on n=2,580 daytime hours; BiLSTM-Opt on n=2,510; TFT variants on n=2,554. Differences reflect input window lengths required by each model architecture."

### 5.2 DM test n varies
- Report: "n=2,554 paired observations for DM tests"
- Actual: LSTM vs XGB n_pairs = 2,510; TFT vs XGB = 2,554; PatchTST vs XGB = 2,580

**Action:** Report range: n_pairs varies from 2,510 to 2,580.

### 5.3 Route comparison n — ambiguous
- XGB indirect→power n=2,165; XGB direct n=2,172 (different because of pair intersection)
- Report states "n=2,172" without specifying this is the direct-route subset

**Action:** Clarify which n is cited for each route.

---

## 6. Overlapping CI vs Significant DM Test — Language Check

**Report claim:** "XGB-Opt and TFT-96 have fully overlapping bootstrap CIs but TFT-96 is significantly worse by paired DM test"

**Verification:**
- XGB CI: [24.76%, 33.46%] (benchmark_summary.csv)
- TFT-96 CI: [24.71%, 34.78%] (benchmark_summary.csv)
- These CIs DO fully overlap ✓
- DM test TFT-96 vs XGB: p=0.0, significant ✓

**Assessment:** This claim is internally consistent and educationally valuable. The explanation (marginal CI vs paired directional test) is scientifically correct. **No change needed**, but the report must ensure the explanation is clear enough that readers do not conclude the models are statistically equivalent.

---

## 7. "All Models" Claims — Scope Check

### 7.1 Route comparison covers only XGB and BiLSTM
- route_comparison.csv contains only xgb_sota and lstm_sota
- TFT-96, TFT-168, PatchTST are NOT in the route comparison
- **Action:** Every "all models prefer indirect" or "indirect consistently outperforms" claim must be qualified: "for XGB-Opt and BiLSTM-Opt (route comparison was not computed for TFT variants and PatchTST)"

### 7.2 Seed variability covers only XGB (3 seeds with daytime eval) and LSTM (3 seeds training-only)
- **Action:** Do not say "all models were tested across multiple seeds" — only XGB was fully evaluated across seeds at daytime-evaluation level

### 7.3 Common-mask evaluation covers only XGB + LSTM + TFT baseline
- TFT-96, TFT-168, PatchTST are NOT in common_mask_summary.csv
- **Action:** State explicitly which models appear in the common-mask evaluation

---

## 8. SOTA Terminology — Scope Check

### 8.1 Internal vs external SOTA
- The report may use "SOTA" to refer to the best model within this study
- This must not be confused with external state-of-the-art benchmarks (NREL, IEA, etc.)
- **Action:** Verify the report always qualifies "SOTA" as "best-performing model in this study" or "study-best" rather than claiming external benchmark status

---

## 9. Feature Selection Set Consistency

### 9.1 05e vs 05f — must not be conflated
- Report must state: 05e = official modelling input (40 features for indirect D+1, before leakage EXCL)
- Report must state: 05f = post-hoc audit (CORE=15, SUPPORTED=8, CONDITIONAL=1 → 24 recommended)
- **Risk:** Any sentence that says "we trained models on CORE features" or "05f features were used" is WRONG unless explicitly verified by reading model training scripts

### 9.2 Feature count evolution
- Pre-Stage-1 pool: ~86 features (report claim — needs verification)
- Post-Stage-1 / entering 05e: n_starting_features = 42 (from 05e JSON)
- After 05e consensus (2-of-3 vote): 40 features
- After 05f audit (CORE+SUPPORTED+CONDITIONAL): 24 features (not used for training)
- Effective modelling set: 40 (05e) MINUS leakage features excluded by EXCL list

**Action:** Present this as a clear pipeline table in the Feature Selection section.

---

## 10. Language and Claims to Soften or Remove

| Current language | Problem | Proposed replacement |
|-----------------|---------|---------------------|
| "17 features constituting data leakage" | Contradicts audit doc count of 19 | "19 features constituting data leakage" |
| "1.7–2.1 pp improvement" (route) | LSTM D+1 = 1.655 pp, rounds to 1.7 aggressively | "1.66–2.10 pp improvement" |
| "free improvement" (route) | Overclaims — implies zero cost | "2 pp nRMSE improvement without model modification" |
| "essentially unbiased" (XGB bias²) | Strong claim requiring source | "bias² contributes < 0.15% of MSE" with citation |
| "dominant" (rolling stats) | Can be kept with % quantification: "dominant (70.7%)" | Add percentage |
| XGB/BiLSTM "SOTA" | Verify context — internal SOTA only | Add qualifier: "within this study" |
| "Summer would reduce nRMSE 5–8 pp" | Estimate, not verified | "estimated 5–8 pp reduction, based on Spring (22%) being the most amenable season" |
| "single-site studies cannot distinguish models within 1–2 pp" | Broad generalization | "studies with n~2,500 daytime test hours are substantially underpowered for detecting differences below ~1 pp" |
| BiLSTM seeds std "~2.2%" | Based on training-set metrics not daytime-eval | Add caveat about training-set origin or remove |

---

## 11. Figure and Table Reference Consistency

The report references Figures 1–25 and multiple tables. The following should be verified during final editing:

- Figure numbers must be sequential and consistent with the actual figures in the document
- Every figure caption must match the figure content (especially after correcting direct-target nRMSE values)
- Table entries in benchmark tables must match corrected CSV values
- Any figure that shows direct-target performance must be regenerated if the incorrect 12.66% appears in its labels or axes

**High-risk figures after corrections:**
- Fig 1: Benchmark overview — contains direct-target nRMSE values that are wrong
- Fig 3: Seasonal heatmap — may show Winter value of 27% which is wrong
- Any "radar chart" or "spider chart" showing seasonal performance

---

## 12. Conclusion Claims vs Table Values

| Conclusion | Values stated | Source verification |
|------------|--------------|---------------------|
| C1: XGB best indirect D+1 at 29.09% | ✓ VERIFIED | benchmark_summary.csv |
| C1: TFT-96 second at 29.66% | ✓ VERIFIED | benchmark_summary.csv |
| C1: XGB direct D+1 = 12.66% (in C1) | ✗ WRONG — correct to 12.06% | benchmark_summary.csv |
| C2: Indirect 1.7–2.1 pp advantage | ≈ VERIFIED with minor precision issue | route_comparison.csv |
| C3: XGB horizon −0.01 pp | ✓ VERIFIED | benchmark_summary.csv |
| C4: PatchTST 30.30% with zero features | ✓ VERIFIED | benchmark_summary.csv |
| C5: TFT-96 outperforms TFT-168 by 1.2–1.4 pp | Need to verify: 31.047−29.664=1.38pp indirect | ✓ benchmark_summary.csv |
| C7: Top-4 within 1.2 pp | ✓ VERIFIED: 30.30−29.09=1.21pp | benchmark_summary.csv |
| C8: Q2 ramp 20.7% Q4 mid 6.9% | NEEDS MANUAL REVIEW | Quintile output not found |
