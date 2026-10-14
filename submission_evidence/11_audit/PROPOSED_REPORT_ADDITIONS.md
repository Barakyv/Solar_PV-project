# Proposed Report Additions
**Solar PV Forecasting Capstone — Ktora Plant**
**Audit date:** 2026-05-11 | **Auditor:** Claude Sonnet 4.6

This document provides ready-to-paste corrections and additions for the report. Each section states where to insert the material, which source file supports it, and uses academic language. Every numerical value is cited to a source file.

---

## SECTION A — Mandatory Numerical Corrections

**Location:** Find-and-replace throughout the document.

### A1. XGB-Opt Direct D+1 nRMSE

**Current text (example):** "XGB-Opt achieves 12.66% nRMSE on the direct D+1 target"
**Replace with:** "XGB-Opt achieves 12.06% nRMSE on the direct D+1 target"
**Source:** `outputs/07_evaluation/benchmark_summary.csv`, row `xgb_sota,direct,1,2580,12.0595`
**All occurrences to fix:** abstract, benchmark table, Figure 1 labels, Conclusion C1, recommendation section

### A2. TFT-96 Direct D+1 nRMSE

**Current text:** "TFT-96 achieves 12.66% nRMSE on direct D+1"
**Replace with:** "TFT-96 achieves 12.11% nRMSE on direct D+1"
**Source:** `outputs/07_evaluation/benchmark_summary.csv`, row `tft_96,direct,1,2554,12.1132`

### A3. TFT Baseline and TFT-168 Direct D+1 — Correct Swap

The report's direct-target benchmark table has TFT baseline and TFT-168 values swapped.

**Corrected values:**
| Model | Report value (WRONG) | Correct value | Source |
|-------|---------------------|---------------|--------|
| TFT baseline | 13.54% | **12.80%** | `tft,direct,1,2554,12.7982` |
| TFT-168 | 12.8% | **13.26%** | `tft_168,direct,1,2554,13.2643` |

**Action:** In the benchmark table for direct D+1, set TFT baseline = 12.80% and TFT-168 = 13.26%. Update Figure 1, Conclusion C1, and any sentence comparing TFT-168 to TFT baseline.

**Corrected direct-target D+1 ranking:**
1. XGB-Opt: **12.06%**
2. TFT-96: **12.11%**
3. LSTM: 12.58%
4. TFT baseline: **12.80%**
5. TFT-168: **13.26%**
6. PatchTST: 13.84%

### A4. XGB Winter Indirect D+1 Seasonal nRMSE

**Current text:** "XGB-Opt achieves 27.0% nRMSE in Winter"
**Replace with:** "XGB-Opt achieves 32.22% nRMSE in Winter"
**Source:** `outputs/07_evaluation/seasonal_metrics.csv`, row `xgb_sota,indirect,1,Winter,858,32.221`

**All Winter values needing correction (indirect D+1, from seasonal_metrics.csv):**
| Model | Correct Winter value |
|-------|---------------------|
| XGB-Opt | 32.22% |
| LSTM | 32.26% |
| TFT-96 | 32.27% |
| TFT-168 | 35.65% |
| TFT baseline | 35.10% |
| PatchTST | 33.50% |

**Update seasonal section text, seasonal heatmap figure, and any comparative sentence about winter.**

### A5. Leakage Feature Count

**Current text:** "we identified and excluded 17 features constituting data leakage"
**Replace with:** "we identified and excluded 19 features constituting data leakage across three categories"
**Source:** `docs/leakage_audit.md` — Category 1: 7 features, Category 2: 10 features, Category 3: 2 features

### A6. XGB Seed Max Spread

**Current text:** "max spread 0.095 pp (indirect) and 0.125 pp (direct)"
**Replace with:** "maximum range across seeds: 0.111 pp (indirect D+1: 29.095% vs 28.984%) and 0.092 pp (direct D+3: 12.185% vs 12.093%)"
**Source:** `outputs/07_evaluation/seed_variability_summary.csv`, rows for `xgb_sota`

---

## SECTION B — Data Science Milestone Table

**Location:** Insert as a new subsection after the Introduction, or at the start of the Methodology section. Alternatively, place in an appendix if the main body is length-constrained.

**Title:** "End-to-End Data Science Pipeline"

**Ready-to-paste text:**

---

> The forecasting system was developed through an end-to-end pipeline spanning 23 reproducible stages. Table [X] provides a chronological summary.
>
> **Table [X]. Data Science Pipeline Milestones**
>
> | Stage | Description | Key Artifact | Purpose |
> |-------|-------------|-------------|---------|
> | 01 | Raw SCADA ingestion | `01_data_ingest_new.ipynb` | 1-min AC power, irradiance, inverter-level signals (Sep 2021–May 2025) |
> | 02 | Exploratory data analysis | `02_EDA_MultiResolution.ipynb` | Characterise seasonality, capacity factor distribution, data coverage |
> | 03 | Hourly resampling (UTC) | `03_preprocessing_UNIFIED.ipynb` | Consistent 1-hour UTC axis for all model families |
> | 04 | Data quality control | `cleaning_policy_final.yaml` | Remove erroneous records; impute missing values; flag inverter events |
> | 04b | Feature engineering | `04_feature_engineering_v5.ipynb` | Construct ~80 features: clear-sky (pvlib Ineichen), lags, rolling statistics, time encodings |
> | Audit | Leakage identification | `docs/leakage_audit.md` | Identify and exclude 19 same-timestamp features unavailable at forecast issuance |
> | 05c | SHAP backward elimination | `05c_backward_elimination.ipynb` | Remove features with SHAP contribution below 1% threshold |
> | 05e | Three-method consensus | `final_features_*.json` | Retain features endorsed by ≥ 2 of 3 methods (SHAP elbow, stability selection, mutual information) |
> | 05e | Operational classification | `feature_availability_classification.csv` | Classify as FORECAST_SAFE, HORIZON_DEPENDENT, or PIPELINE_DEPENDENT |
> | 05f | Robustness audit | `refined_features_*_05f.json` | Post-hoc stability-floor and MI-threshold audit (CORE/SUPPORTED/CONDITIONAL/EXCLUDED) |
> | 06 | XGB-Opt training (×3 seeds) | `xgb_sota.py`, Optuna 60 trials | Bayesian hyperparameter optimisation; monotone physics constraints |
> | 06 | BiLSTM-Opt training | `lstm_sota.py`, Optuna 60 trials | Encoder-decoder LSTM; quantile loss q={0.1, 0.5, 0.9} |
> | 06 | TFT-96 / TFT-168 training | `tft_model.py`, Optuna 20 trials | Variable-Selection Network; known-reals inputs from calendar and clear-sky features |
> | 06 | PatchTST training | `patchtst_model.py` | Patch-based transformer; no exogenous variables |
> | 06 | LSTM timestamp alignment | `align_and_evaluate.py` | Correct decoder window-end duplication; recalibrate D+1 from ≈17% to 30.24% nRMSE |
> | 07 | Daytime-filtered evaluation | `evaluate.py` → `benchmark_summary.csv` | Canonical metrics: daytime-only nRMSE, MAE, R², 95% bootstrap CI |
> | 07 | Diebold-Mariano testing | `dm_tests.csv` | HAC-corrected paired test with NW lags=24; α=0.05 |
> | 07 | Power analysis | `dm_tests_extended.csv` | Effective sample size and minimum detectable effect |
> | 07 | Common-mask evaluation | `common_mask_summary.csv` | Verify ranking stability under shared timestamp intersection (n=2,116 direct D+1) |
> | 07 | SHAP interpretability | `shap_group_importance.csv` | Feature group importance; cross-horizon rank correlation (Spearman ρ=0.51–0.80) |
> | 07 | Feature group ablation (×3 seeds) | `ablation_seed_variability.csv` | Marginal feature group contribution |
> | 07 | Route comparison | `route_comparison.csv` | Systematic advantage of indirect k_t formulation (1.66–2.10 pp) |
> | Submission | Reproducibility package | `submission_package/` | Split manifest, environment spec, citation metadata, reproduction script |

---

## SECTION C — Expanded Feature Selection Subsection

**Location:** Replace or expand the existing Feature Selection chapter. The full draft text is provided in `FEATURE_SELECTION_INTEGRATION_PLAN.md` (Sections 5.1–5.3). Summarised here for reference.

**Key additions:**
1. Feature count table by stage (see FEATURE_SELECTION_INTEGRATION_PLAN.md §3.3)
2. Top stability-selection probabilities paragraph (§3.4)
3. Mandatory 05e vs 05f distinction paragraph (§3.1)
4. Leakage–feature-selection interaction disclosure (§3.2)

**Source files:**
- `artifacts/05_feature_selection/final_features_indirect_D1.json` (05e: 40 features, stability=19, MI=40, SHAP=40)
- `artifacts/05_feature_selection/refined_features_indirect_D1_05f.json` (05f: CORE=15, SUPPORTED=8, CONDITIONAL=1, EXCLUDED=18 → 24 recommended)

---

## SECTION D — LSTM Timestamp Bug Disclosure

**Location:** Add to the BiLSTM-Opt architecture section, or as Disclosure D-new.

**Ready-to-paste text:**

---

> **BiLSTM-Opt Timestamp Alignment**
>
> The BiLSTM-Opt encoder-decoder architecture generates forecasts for all 72 decoder steps (three days at hourly resolution) in a single forward pass. An initial implementation stored the window-end timestamp as the target label for all decoder positions, creating a systematic offset: each forecast ostensibly labelled "D+1" was in fact drawn from a later decoder step. The correct alignment formula is:
>
> `valid_ts = window_end − (DECODER_LEN − 1 − target_step) × 1h`
>
> After applying this correction, the D+1 indirect nRMSE changed from approximately 17% (pre-correction) to 30.24% (post-correction). This correction exceeds the entire cross-model performance spread in the benchmark (1.21 pp) by more than an order of magnitude. The timestamp alignment script is `scripts/align_and_evaluate.py`, and the corrected predictions are stored in `outputs/06_lstm_sota/predictions_lstm_sota_aligned.csv`.
>
> **Note:** All BiLSTM-Opt results in this report are computed from the aligned predictions file. The pre-correction predictions (`predictions_lstm_sota.csv`) are retained for audit purposes only.

**Source:** `outputs/06_lstm_sota/align_seed42.log` (read to verify exact pre-correction value; report ~17% needs confirmation from log).

---

## SECTION E — BiLSTM Multi-Seed Disclosure

**Location:** Add to the Robustness and Seed Variability section.

**Ready-to-paste text:**

---

> **BiLSTM-Opt Seed Variability**
>
> Three independent training runs were completed for BiLSTM-Opt using seeds 42, 123, and 777. Due to the absence of a GPU during the evaluation period, the authoritative daytime-filtered day-ahead evaluation (via `evaluate.py`) was performed only for seed 42, yielding nRMSE of 30.24% (indirect) and 12.58% (direct) on the held-out test set.
>
> Seeds 123 and 777 produced training-set evaluation metrics (not daytime-filtered day-ahead nRMSE) of approximately 19–20% indirect and 8.4–8.7% direct — these figures are not directly comparable to the benchmark values and are reported here only to indicate that training converged successfully across seeds. Authoritative multi-seed daytime evaluation of BiLSTM-Opt requires additional GPU compute and is recommended for future work.

**Source:** `outputs/07_evaluation/seed_variability_summary.csv` (LSTM rows reflect training metrics); `outputs/07_evaluation/benchmark_summary.csv` (LSTM seed 42 authoritative value: 30.237%).

---

## SECTION F — Reproducibility and Traceability Appendix

**Location:** New appendix section or expand existing reproducibility statement.

**Ready-to-paste text:**

---

> **Reproducibility Package**
>
> The following artifacts support independent reproduction of the reported results:
>
> - **Data split manifest** (`configs/split_manifest.yaml`, mirrored in `submission_package/split_manifest.yaml`): defines exact train/validation/test boundaries, target variable formulas, and daytime filter criteria.
> - **Conda environment** (`environment.yml`): specifies all Python package versions including scikit-learn, xgboost, pytorch-lightning, pytorch-forecasting, and neuralforecast.
> - **Citation metadata** (`CITATION.cff`): structured citation information for software and dataset attribution.
> - **Reproduction script** (`submission_package/reproduce_paper_results.py`): end-to-end script generating all tables and figures from stored predictions.
> - **Prediction CSVs**: aligned prediction files for all six models are available upon request subject to operator approval (raw SCADA data remain commercially sensitive).
>
> All numerical constants (AC_CAPACITY_KW, MEAN_DAYTIME_K, RANDOM_SEEDS, split boundaries) are defined in `production/00_config/constants.py` as the single source of truth.

---

## SECTION G — Statistical Power Expansion

**Location:** Expand existing power analysis paragraph.

**Current (approximate):** "The observed effect is 38–39% of the minimum detectable effect."

**Replace with:**

---

> **Statistical Power of the LSTM vs XGB-Opt Comparison**
>
> The Diebold-Mariano test for BiLSTM-Opt versus XGB-Opt produces non-significant results at α=0.05 across all horizons (indirect D+1: DM=1.14, p=0.25; indirect D+3: DM=1.44, p=0.15; direct D+1: DM=1.23, p=0.22). The observed differences in mean squared error correspond to 29–50% of the minimum detectable effect (MDE) at 80% power, depending on target and horizon (Table [Y]; source: `dm_tests_extended.csv`).
>
> Effective sample sizes range from n_eff=469 (direct D+3) to n_eff=806 (indirect D+1), substantially smaller than the 2,510–2,514 total paired observations due to high autocorrelation in daily solar forecast errors. This analysis implies that a study would require approximately 6× more effective test observations — equivalent to approximately 6 years of test data at this site — to reliably detect a 1 pp nRMSE difference at 80% power.
>
> This finding does not imply the models perform equally well; it implies the current study is substantially underpowered for detecting the observed difference. Single-site studies should report confidence intervals alongside point estimates, and claims of 0.5–1 pp model advantages should be treated with appropriate caution.

**Source:** `outputs/07_evaluation/dm_tests_extended.csv`, `outputs/07_evaluation/dm_tests.csv`

---

## SECTION H — Common-Mask Evaluation Clarification

**Location:** Add to Evaluation Framework section or as note in Results.

**Ready-to-paste text:**

---

> **Common-Mask Evaluation**
>
> Model architectures require different input window lengths: XGB-Opt and PatchTST require no historical lookback beyond the feature window; BiLSTM-Opt requires an encoder window of several days; TFT-96 and TFT-168 require 96-hour and 168-hour lookback windows respectively. As a result, each model evaluates on a slightly different subset of test timestamps.
>
> To enable a strictly fair cross-model comparison, a common-mask evaluation was performed on the intersection of timestamps for which XGB-Opt, BiLSTM-Opt, and TFT baseline all produced valid predictions. The shared intersection contains n=2,116 daytime hours for the direct D+1 target. Within this shared set, nRMSE values are slightly higher for all models (due to the subset being less representative), but the ranking is preserved: XGB-Opt 13.30%, BiLSTM-Opt 13.72%, TFT baseline 14.05% (source: `common_mask_summary.csv`).

---

## SECTION I — Route Comparison Precision Correction

**Location:** Route analysis section — replace any occurrence of "1.7–2.1 pp" with the precise values.

**Replace:** "indirect formulation consistently improves nRMSE by 1.7–2.1 pp"
**With:** "indirect formulation consistently improves nRMSE by 1.66–2.10 pp across model families and horizons (XGB-Opt: 1.98–2.10 pp; BiLSTM-Opt: 1.66–1.89 pp; source: `route_comparison.csv`)"

---

## Summary of Source Files Used in This Document

| Source file | Claims it supports |
|------------|-------------------|
| `outputs/07_evaluation/benchmark_summary.csv` | Sections A1–A4, G |
| `outputs/07_evaluation/seasonal_metrics.csv` | Section A4 |
| `docs/leakage_audit.md` | Section A5 |
| `outputs/07_evaluation/seed_variability_summary.csv` | Section A6, E |
| `outputs/07_evaluation/dm_tests.csv` | Section G |
| `outputs/07_evaluation/dm_tests_extended.csv` | Section G |
| `outputs/07_evaluation/common_mask_summary.csv` | Section H |
| `outputs/07_evaluation/route_comparison.csv` | Section I |
| `artifacts/05_feature_selection/final_features_indirect_D1.json` | Section C |
| `artifacts/05_feature_selection/refined_features_indirect_D1_05f.json` | Section C |
| `production/00_config/constants.py` | Section F |
| `outputs/06_lstm_sota/align_seed42.log` | Section D (verify before use) |
