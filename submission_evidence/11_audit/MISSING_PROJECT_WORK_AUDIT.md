# Missing Project Work Audit
**Solar PV Forecasting Capstone — Ktora Plant**
**Audit date:** 2026-05-11 | **Auditor:** Claude Sonnet 4.6

This document identifies project work that was completed but is absent, insufficiently described, or unsupported by artifact citations in the main report.

---

## Data Science Milestone Table (Proposed)

| # | Milestone | Stage | Key Artifact | Methodological Purpose | Report Coverage | Recommendation |
|---|-----------|-------|-------------|----------------------|-----------------|----------------|
| 1 | Raw SCADA ingestion | 01 | `01_data_ingest/01_data_ingest_new.ipynb` | Establish data provenance; raw 1-minute SCADA + Open-Meteo API | Weak — data source mentioned but ingestion pipeline not described | Add paragraph describing raw format, temporal extent, API source for NWP |
| 2 | Hourly resampling and UTC handling | 03 | `03_preprocessing/03_preprocessing_UNIFIED.ipynb` | Consistent time axis for all model families | Absent | Add: "Raw 1-minute SCADA data were resampled to hourly means. All timestamps are UTC; local reporting uses UTC+2." |
| 3 | Data quality control | 04 | `04_data_cleaning/cleaning_policy_final.yaml`, `data/cleaned/cleaning_logs/` | Remove erroneous records (curtailments, sensor failures, inverter trips) | Partial — mentioned but no row-count evidence | Add: rows removed, imputation strategy, cleaning rule count |
| 4 | Exploratory data analysis | 02 | `02_eda/EDA_1H/02_EDA_MultiResolution.ipynb`, `outputs/EDA/` | Understand seasonality, capacity factor distribution, outlier patterns | Weak — cited but key EDA findings not quantified | Add one paragraph with EDA key findings (capacity factor range, seasonal patterns, data coverage) |
| 5 | Clear-sky irradiance modelling | 04b | `04_feature_engineering/04_feature_engineering_v5.ipynb` | Deterministic solar geometry baseline (pvlib Ineichen model) | Present — pvlib cited | Add: pvlib version, Ineichen model choice rationale, site coordinates used |
| 6 | Capacity estimation | 00 | `production/00_config/constants.py` (AC_CAPACITY_KW=3449) | Normalization denominator for direct target and route comparison | Present — 3,449 kW stated | Verify how 3,449 kW was derived (nameplate vs measured peak) |
| 7 | Target construction (k_t formula) | 04b | `configs/split_manifest.yaml` | Define indirect target as performance index; remove deterministic solar geometry component | Present — formula stated | Verified OK |
| 8 | NWP leakage identification | audit | `docs/leakage_audit.md`, `archive/root_scripts/investigate_leakage.py` | Prevent using same-timestamp reanalysis as forecast input | Present — audit cited | Add: date leakage was identified (2026-04-08), specific API clarification (Open-Meteo archive vs forecast endpoint) |
| 9 | SHAP backward elimination (05c) | 05c | `05_feature_selection/05c_backward_elimination.ipynb`, `artifacts/05_feature_selection/backward_elimination_results.json` | Empirically identify redundant features; establish nRMSE baseline for selection | Partial — exists in Feature Selection Report but underdescribed in main report | Add elbow threshold description and baseline nRMSE value (16.033% for indirect D+1 from 05e JSON) |
| 10 | Three-method feature consensus (05e) | 05e | `05_feature_selection/05e_final_feature_selection.ipynb`, `artifacts/05_feature_selection/final_features_*.json` | Produce stable, operationally available feature sets using SHAP + Stability + MI | Present — described but missing feature counts per method | Add: SHAP elbow selected 40, Stability selected 19, MI selected 40; consensus (2-of-3) = 40 for indirect D+1 |
| 11 | Feature robustness audit (05f) | 05f | `05_feature_selection/05f_feature_analysis.ipynb`, `artifacts/05_feature_selection/refined_features_*.json` | Post-hoc audit of 05e set using stability floor (0.30) and MI threshold (75th pct) | Partial — mentioned in Feature Selection Report | Expand in main report with tier counts; add explicit disclaimer that 05f was not used for training |
| 12 | Operational availability classification | 05e/05f | `artifacts/05_feature_selection/feature_availability_classification.csv` (expected) | Classify features as FORECAST_SAFE, HORIZON_DEPENDENT, PIPELINE_DEPENDENT | Absent from main report | Add paragraph with classification counts; cite classification CSV |
| 13 | XGB hyperparameter optimization | 06 | Optuna study (60 trials), training logs | Tune XGB on validation daytime nRMSE; monotone constraints for physical plausibility | Present — search space stated | Verified OK |
| 14 | LSTM hyperparameter optimization | 06 | `outputs/06_lstm_sota/train_seed42.log`, 60 trials | Same objective; encoder-decoder Bayesian optimization | Present | Add: quantile loss function q=0.1/0.5/0.9; 25 quick-train vs 60 final epochs |
| 15 | TFT variants optimization | 06 | `lightning_logs/version_*/hparams.yaml` (38 runs) | 20-trial Optuna for TFT-96 and TFT-168 | Present — noted as GPU-limited | Verified OK |
| 16 | PatchTST training | 06 | `outputs/06_patchtst/`, `production/02_models/patchtst_model.py` | Patch-based transformer as exogenous-free baseline | Present | Add: neuralforecast 3.1.7, rolling 243 origins, patch_len=16 stride=8 |
| 17 | LSTM timestamp alignment bug fix | 06 | `outputs/06_lstm_sota/align_seed42.log`, `scripts/align_and_evaluate.py` | Correct window-end label duplication that inflated D+1 performance | Partial — bug mentioned; before/after values need quantification | Add: exact formula `valid_ts = window_end − (DECODER_LEN − 1 − target_step)h`; before = ~17% (verify from log); after = 30.24% |
| 18 | Multi-seed training (XGB) | 06 | `outputs/06_xgb_sota/metrics_xgb_sota_seed*.csv`, `outputs/07_evaluation/seed_variability_summary.csv` | Quantify stochastic variability in XGB optimization | Present | Correct max-spread values in report |
| 19 | Multi-seed training (LSTM GPU-limited) | 06 | `seed_variability_summary.csv` LSTM rows | Same goal as XGB multi-seed — blocked by GPU unavailability | Present with disclosure | Add explicit caveat: LSTM seed metrics in seed table are training-set, not daytime-eval |
| 20 | Daytime-filtered evaluation (evaluate.py) | 07 | `production/03_evaluation/evaluate.py` | Authoritative evaluation: daytime filter, normalization, bootstrap CI, DM test | Present | Add: cite evaluate.py as authoritative; note it is distinct from model-internal metrics |
| 21 | Common-mask evaluation | 07 | `outputs/07_evaluation/common_mask_summary.csv`, `scripts/common_mask_analysis.py` | Ensure ranking comparisons are valid despite differing input windows | Present | Add n counts explicitly in text (D+1 direct: n=2,116 shared) |
| 22 | DM test with NW lag fix | 07 | `outputs/07_evaluation/dm_tests.csv`, `scripts/detectable_effect.py` | Correct statistical test for autocorrelated forecast errors | Present | Verified OK — NW lags=24 confirmed |
| 23 | Statistical power analysis | 07 | `outputs/07_evaluation/dm_tests_extended.csv` | Quantify what effect size the study can detect | Present | Broaden reported range from 38–39% to 29–50% MDE across horizons |
| 24 | SHAP group importance | 07 | `outputs/07_evaluation/shap_group_importance.csv`, `scripts/recompute_shap.py` | Explain predictions by feature group | Present | Verified OK |
| 25 | SHAP rank stability (cross-horizon) | 07 | `outputs/07_evaluation/shap_horizon_corr.csv`, `scripts/shap_rank_stability.py` | Confirm feature importance is consistent across forecast horizons | Partial — mentioned but Spearman ρ values not cited in main body | Add: Spearman ρ 0.51–0.80 across horizons; cite shap_horizon_corr.csv |
| 26 | Feature group ablation | 07 | `outputs/07_evaluation/ablation_seed_variability.csv`, `scripts/run_ablation_study.py` | Quantify marginal value of each feature group | Present | Verified OK |
| 27 | Route comparison analysis | 07 | `outputs/07_evaluation/route_comparison.csv`, `scripts/compare_routes.py` | Compare indirect→power conversion vs direct kW forecasting | Present | Correct lower bound to 1.66 pp |
| 28 | Error decomposition (bias² + variance) | 07 | `scripts/run_error_decomposition.py` | Detect systematic over/under-prediction | Partial — values stated but source not cited | Find output CSV; cite in report |
| 29 | Production quintile analysis | 07 | Quintile output (file not found) | Error behavior by production regime (near-zero, ramp, midday, peak) | Present | Find quintile CSV; add source citation |
| 30 | Reproducibility package | submission | `submission_package/README.md`, `split_manifest.yaml`, `CITATION.cff`, `environment.yml` | Enable independent reproduction of results | Partial — mentioned but artifacts not fully cited | Add appendix listing all submission package contents |

---

## Coverage Assessment by Category

### STRONG (adequately covered):
- Model architectures and hyperparameter search spaces
- DM test methodology
- Primary metrics (nRMSE, denominator)
- Route comparison results
- SHAP group importance
- Feature ablation
- Reproducibility statement (partial)

### PARTIAL (present but needs expansion):
- Data ingestion and preprocessing pipeline
- Leakage audit (count error, missing category breakdown)
- 05c backward elimination (results present in Feature Selection Report but not integrated in main)
- 05e feature selection (method trio described but feature counts per method missing)
- 05f robustness audit (tier definitions present but feature counts missing from main body)
- LSTM timestamp bug fix (disclosed but not quantified with before/after values)
- SHAP rank stability (analysis done but Spearman ρ values not reported)
- Statistical power analysis (range incomplete: 38–39% should be 29–50%)

### ABSENT (needs to be added):
- Hourly resampling and UTC handling (preprocessing stage)
- Data quality control row-count evidence
- EDA key quantitative findings
- Operational availability classification (FORECAST_SAFE, HORIZON_DEPENDENT, PIPELINE_DEPENDENT)
- Data Science Milestone Table (proposed in plan, not yet written)
- Source citations for CV output, error decomposition, quintile analysis
- Explanation that 05e JSON contains leakage features (excluded via EXCL list at training)

---

## Proposed Data Science Milestone Table for Report

Insert as a new subsection titled **"Data Science Pipeline: End-to-End Milestones"** after the Introduction or at the start of the Methodology section.

| Milestone | Stage | Primary Artifact | Methodological Purpose |
|-----------|-------|-----------------|----------------------|
| Raw SCADA ingestion | 01 | `01_data_ingest_new.ipynb` | Collect 1-min AC power, irradiance, SCADA signals from two inverters |
| Hourly resampling (UTC) | 03 | `03_preprocessing_UNIFIED.ipynb` | Align all signals to consistent 1-hour UTC timestamps |
| Quality control and cleaning | 04 | `cleaning_policy_final.yaml` | Remove erroneous records; impute missing values; flag inverter events |
| Exploratory data analysis | 02 | `02_EDA_MultiResolution.ipynb` | Characterize seasonality, capacity factor distribution, outlier patterns |
| Feature engineering | 04b | `04_feature_engineering_v5.ipynb` | Construct 80+ engineered features: clear-sky, lags, rolling stats, time encodings |
| Leakage audit | audit | `docs/leakage_audit.md` | Identify and exclude 19 same-timestamp features unavailable at forecast issuance |
| SHAP backward elimination (05c) | 05c | `05c_backward_elimination.ipynb` | Empirically remove features with SHAP contribution < 1% of total |
| Three-method feature consensus (05e) | 05e | `final_features_*.json` | Retain features endorsed by ≥ 2 of 3 independent methods (SHAP elbow, stability selection, mutual information) |
| Operational availability classification | 05e | `feature_availability_classification.csv` | Classify each feature as FORECAST_SAFE, HORIZON_DEPENDENT, or PIPELINE_DEPENDENT |
| Post-hoc feature robustness audit (05f) | 05f | `refined_features_*_05f.json` | Apply stability floor (0.30) and MI threshold (75th percentile) as sensitivity analysis; tier features as CORE/SUPPORTED/CONDITIONAL/EXCLUDED |
| XGB-Opt and BiLSTM-Opt training | 06 | Optuna study (60 trials/target) | Bayesian hyperparameter search; monotone physics constraints for XGB |
| TFT-96 and TFT-168 training | 06 | `lightning_logs/*/hparams.yaml` | Temporal Fusion Transformer; 20 Optuna trials; GPU-limited |
| PatchTST training | 06 | `patchtst_model.py` | Patch-based transformer without exogenous features (temporal baseline) |
| LSTM timestamp alignment fix | 06 | `align_seed42.log` | Correct window-end duplication bug; recalculate D+1 from ≈17% to 30.24% |
| Multi-seed robustness (XGB ×3) | 06 | `seed_variability_summary.csv` | Quantify stochastic variability across seeds 42/123/777 |
| Authoritative evaluation (evaluate.py) | 07 | `benchmark_summary.csv` | Daytime-filtered, properly normalized nRMSE with 95% bootstrap CI |
| Diebold-Mariano significance testing | 07 | `dm_tests.csv` | HAC-corrected paired DM test with NW lags=24; α=0.05 two-sided |
| Statistical power analysis | 07 | `dm_tests_extended.csv` | Quantify minimum detectable effect and effective sample size |
| Common-mask evaluation | 07 | `common_mask_summary.csv` | Verify ranking stability under shared timestamp intersection |
| SHAP interpretability | 07 | `shap_group_importance.csv` | Feature group importance; cross-horizon rank stability |
| Feature group ablation (×3 seeds) | 07 | `ablation_seed_variability.csv` | Marginal contribution of each feature group to model performance |
| Route comparison (indirect vs direct) | 07 | `route_comparison.csv` | Systematic advantage of k_t formulation over direct kW forecasting |
| Reproducibility packaging | submission | `submission_package/` | Split manifest, citation metadata, environment spec, reproduction script |
