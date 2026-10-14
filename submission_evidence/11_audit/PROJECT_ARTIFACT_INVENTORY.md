# Project Artifact Inventory
**Solar PV Forecasting Capstone — Ktora Plant**
**Audit date:** 2026-05-11
**Auditor:** Claude Sonnet 4.6

This inventory catalogues every relevant file in the repository, organized by category. Files in `archive/` or `old/` subdirectories are marked as LEGACY. Files verified as referenced in the main report are marked REPORT_CITED. Files that directly support a report claim are marked EVIDENCE.

---

## I. Key Evaluation Outputs (Primary Evidence)

| File | Stage | Purpose | Current? | EVIDENCE for |
|------|-------|---------|----------|-------------|
| `outputs/07_evaluation/benchmark_summary.csv` | 07 | All-model nRMSE, MAE, R², RMSE, CI | ✅ | All performance claims |
| `outputs/07_evaluation/dm_tests.csv` | 07 | Diebold-Mariano test statistics, p-values | ✅ | DM claims, significance claims |
| `outputs/07_evaluation/dm_tests_extended.csv` | 07 | Extended DM with n_eff, CI, %MDE | ✅ | Power analysis claims |
| `outputs/07_evaluation/route_comparison.csv` | 07 | Indirect→power vs direct at kW scale | ✅ | Route comparison claims |
| `outputs/07_evaluation/common_mask_summary.csv` | 07 | Shared-timestamp nRMSE and n | ✅ | Common-mask claims |
| `outputs/07_evaluation/seed_variability_summary.csv` | 07 | XGB and LSTM nRMSE across seeds 42/123/777 | ✅ | Seed variability claims |
| `outputs/07_evaluation/seasonal_metrics.csv` | 07 | Per-model seasonal nRMSE | ✅ | Seasonal performance claims |
| `outputs/07_evaluation/shap_group_importance.csv` | 07 | SHAP importance by feature group | ✅ | SHAP claims |
| `outputs/07_evaluation/shap_horizon_corr.csv` | 07 | Cross-horizon SHAP rank correlation | ✅ | SHAP stability claims |
| `outputs/07_evaluation/shap_rank_stability.csv` | 07 | Per-feature SHAP values by horizon | ✅ | Feature importance claims |
| `outputs/07_evaluation/ablation_seed_variability.csv` | 07 | Ablation delta nRMSE across seeds | ✅ | Ablation claims |
| `outputs/07_evaluation/FINDINGS.md` | 07 | Narrative findings summary | ✅ | Context / corrections log |
| `outputs/07_evaluation/benchmark_final.md` | 07 | Formatted benchmark results | ✅ | Supplementary reference |

---

## II. Model Prediction Files

| File | Model | Seeds | Aligned? | Notes |
|------|-------|-------|----------|-------|
| `outputs/06_lstm_sota/predictions_lstm_sota_aligned.csv` | LSTM | 42 | ✅ YES | AUTHORITATIVE — use for eval |
| `outputs/06_lstm_sota/predictions_lstm_sota_aligned_seed123.csv` | LSTM | 123 | ✅ YES | Aligned |
| `outputs/06_lstm_sota/predictions_lstm_sota_aligned_seed777.csv` | LSTM | 777 | ✅ YES | Aligned |
| `outputs/06_lstm_sota/predictions_lstm_sota.csv` | LSTM | 42 | ❌ NO | Unaligned — NOT for primary eval |
| `outputs/06_xgb_sota/predictions_xgb_sota.csv` | XGB | 42 | ✅ N/A | No alignment issue |
| `outputs/06_xgb_sota/predictions_xgb_sota_seed123.csv` | XGB | 123 | ✅ N/A | |
| `outputs/06_xgb_sota/predictions_xgb_sota_seed777.csv` | XGB | 777 | ✅ N/A | |
| `outputs/06_tft/predictions_tft_aligned.csv` | TFT baseline | — | ✅ YES | |
| `outputs/06_tft_96/predictions_tft_96_aligned.csv` | TFT-96 | — | ✅ YES | |
| `outputs/06_tft_168/predictions_tft_168_aligned.csv` | TFT-168 | — | ✅ YES | |
| `outputs/06_patchtst/predictions_patchtst_aligned.csv` | PatchTST | — | ✅ YES | |

---

## III. Model Metrics Files

> ⚠️ WARNING: `metrics_lstm_sota.csv` files contain **training-set** metrics, NOT daytime-filtered day-ahead evaluation metrics. Do NOT use these to verify report claims. Use `benchmark_summary.csv` instead.

| File | Source | Comparable to report? |
|------|--------|-----------------------|
| `outputs/07_evaluation/benchmark_summary.csv` | evaluate.py | ✅ YES — authoritative |
| `outputs/06_lstm_sota/metrics_lstm_sota.csv` | LSTM training loop | ❌ NO — training-set only |
| `outputs/06_lstm_sota/metrics_lstm_sota_seed123.csv` | LSTM training loop | ❌ NO |
| `outputs/06_lstm_sota/metrics_lstm_sota_seed777.csv` | LSTM training loop | ❌ NO |
| `outputs/06_xgb_sota/metrics_xgb_sota.csv` | XGB training | ✅ Partial — seed 42 metrics |
| `outputs/06_xgb_sota/metrics_xgb_sota_seed123.csv` | XGB training | ✅ Partial |
| `outputs/06_xgb_sota/metrics_xgb_sota_seed777.csv` | XGB training | ✅ Partial |
| `outputs/07_evaluation/seed_variability_summary.csv` | compile_seed_variability.py | ⚠️ LSTM rows = training metrics; XGB rows = authoritative |

---

## IV. Feature Selection Artifacts

### 05e Official Feature Sets (Modelling Inputs)

> ✅ These are the official modelling feature sets. XGB and LSTM models trained using these sets MINUS leakage-excluded features (EXCL list in training scripts).

| File | Target | Horizon | 05e consensus n | Contains leakage features? |
|------|--------|---------|-----------------|---------------------------|
| `artifacts/05_feature_selection/final_features_indirect_D1.json` | indirect | 1 | 40 | ⚠️ YES (fcst_ghi/dni/dhi etc.) — excluded at training |
| `artifacts/05_feature_selection/final_features_indirect_D2.json` | indirect | 2 | TBD | ⚠️ Likely |
| `artifacts/05_feature_selection/final_features_indirect_D3.json` | indirect | 3 | TBD | ⚠️ Likely |
| `artifacts/05_feature_selection/final_features_direct_D1.json` | direct | 1 | TBD | ⚠️ Likely |
| `artifacts/05_feature_selection/final_features_direct_D2.json` | direct | 2 | TBD | ⚠️ Likely |
| `artifacts/05_feature_selection/final_features_direct_D3.json` | direct | 3 | TBD | ⚠️ Likely |

**Critical note:** The 05e indirect D+1 JSON (`final_features_indirect_D1.json`) shows `"_leakage_fix_removed": []` and still includes `fcst_ghi`, `fcst_dni`, `fcst_dhi`, `fcst_temp_2m`, `fcst_wind_speed`, `poa_global_forecast`, `poa_direct_forecast`, `poa_diffuse_forecast` — all of which appear in `docs/leakage_audit.md`. The JSON artifacts were NOT updated post-leakage-fix. The XGB/LSTM training scripts exclude these via EXCL list. The report must make this clear.

### 05f Audit/Sensitivity Sets (Post-Hoc Only)

> ❌ NOT used for model training. Post-hoc robustness and interpretability audit only.

| File | Target | Horizon | 05f recommended n | Note |
|------|--------|---------|-------------------|------|
| `artifacts/05_feature_selection/refined_features_indirect_D1_05f.json` | indirect | 1 | 24 | CORE=15, SUPPORTED=8, CONDITIONAL=1, EXCLUDED=18 |
| `artifacts/05_feature_selection/refined_features_indirect_D2_05f.json` | indirect | 2 | TBD | |
| `artifacts/05_feature_selection/refined_features_indirect_D3_05f.json` | indirect | 3 | TBD | |
| `artifacts/05_feature_selection/refined_features_direct_D1_05f.json` | direct | 1 | TBD | |
| `artifacts/05_feature_selection/refined_features_direct_D2_05f.json` | direct | 2 | TBD | |
| `artifacts/05_feature_selection/refined_features_direct_D3_05f.json` | direct | 3 | TBD | |

### Other Feature Selection Artifacts

| File | Purpose |
|------|---------|
| `artifacts/05_feature_selection/stage1_filtered_features.json` | Post-stage-1 pool |
| `artifacts/05_feature_selection/stage2_rfe_selected_features.json` | RFE results |
| `artifacts/05_feature_selection/backward_elimination_results.json` | 05c output |
| `artifacts/05_feature_selection/selected_features_direct.json` | Aggregate direct selection |
| `artifacts/05_feature_selection/selected_features_indirect.json` | Aggregate indirect selection |
| `artifacts/05_feature_selection/_pre_leakage_fix_backup/` | PRE-FIX BACKUP — superseded, do not cite |
| `artifacts/05_feature_selection/feature_list_simple.txt` | Human-readable feature list |

---

## V. Configuration Files

| File | Purpose | Used in report? |
|------|---------|-----------------|
| `production/00_config/constants.py` | AC_CAPACITY_KW=3449, MEAN_DAYTIME_K=0.6279, seeds, splits | ✅ |
| `configs/split_manifest.yaml` | Official train/val/test date definitions, target formulas | ✅ |
| `submission_package/split_manifest.yaml` | Submission copy — verify matches configs/ version | ✅ |
| `04_data_cleaning/cleaning_config.yaml` | Data cleaning rules | Partial |
| `04_data_cleaning/cleaning_policy_final.yaml` | Final cleaning policy | Partial |
| `environment.yml` | Conda environment spec | Reproducibility |
| `.zenodo.json` | Zenodo citation metadata | Reproducibility |

---

## VI. Leakage and Audit Documents

| File | Purpose |
|------|---------|
| `docs/leakage_audit.md` | Lists all 19 excluded leakage features (Cat 1: 7, Cat 2: 10, Cat 3: 2, Cat 4: 1) |
| `docs/reproducibility_statement.md` | Reproducibility guide |
| `outputs/07_evaluation/FINDINGS.md` | Full findings document with all corrections |

---

## VII. Notebooks (Active Stages)

| Notebook | Stage | Purpose |
|----------|-------|---------|
| `01_data_ingest/01_data_ingest_new.ipynb` | 01 | Raw SCADA ingestion |
| `02_eda/EDA_1H/02_EDA_MultiResolution.ipynb` | 02 | 1-hour EDA |
| `03_preprocessing/03_preprocessing_UNIFIED.ipynb` | 03 | UTC resampling, timezone handling |
| `04_data_cleaning/01_data_cleaning.ipynb` | 04 | Quality control, imputation |
| `04_feature_engineering/04_feature_engineering_v5.ipynb` | 04b | Feature construction |
| `05_feature_selection/05c_backward_elimination.ipynb` | 05c | SHAP backward elimination |
| `05_feature_selection/05e_final_feature_selection.ipynb` | 05e | Three-method consensus |
| `05_feature_selection/05f_feature_analysis.ipynb` | 05f | Post-hoc audit |
| `06_modeling/06_xgb_sota.ipynb` | 06 | XGB SOTA training |
| `06_modeling/06_lstm_sota.ipynb` | 06 | BiLSTM SOTA training |
| `06_modeling/06_tft_model.ipynb` | 06 | TFT training |
| `07_evaluation/07_evaluation.ipynb` | 07 | Evaluation |
| `08_reporting/08_reporting.ipynb` | 08 | Report generation |

---

## VIII. Production Scripts

| Script | Purpose |
|--------|---------|
| `production/run_pipeline.py` | Orchestration |
| `production/02_models/xgb_sota.py` | XGB SOTA — contains EXCL list (leakage exclusions, lines 51-73) |
| `production/02_models/lstm_sota.py` | LSTM SOTA |
| `production/02_models/tft_model.py` | TFT |
| `production/02_models/patchtst_model.py` | PatchTST |
| `production/03_evaluation/evaluate.py` | Authoritative evaluation (daytime filter, normalization) |
| `scripts/align_and_evaluate.py` | LSTM timestamp alignment |
| `scripts/compile_seed_variability.py` | Seed variability compilation |
| `scripts/detectable_effect.py` | MDE analysis |
| `scripts/compare_routes.py` | Route comparison |
| `scripts/run_ablation_study.py` | Feature ablation |
| `scripts/common_mask_analysis.py` | Common mask intersection |
| `scripts/recompute_shap.py` | SHAP recomputation |
| `scripts/shap_rank_stability.py` | SHAP rank stability |

---

## IX. Report Documents

| File | Version | Notes |
|------|---------|-------|
| `C:\Users\204323141\Downloads\Solar_PV_Forecasting_Report_Enhanced_With_DS_Milestones_and_Feature_Selection.docx` | Under audit | Latest version being audited |
| `docs/Solar_PV_Forecasting_Report.docx` | Project copy | |
| `submission_package/Solar_Forecasting_Report_Sharon.docx` | Submission copy | Verify vs. audited version |
| `reports/Feature_Selection_Report.docx` | Feature selection detail | Supplementary report |
| `docs/full_report.md` | Markdown source | Check if matches DOCX |

---

## X. Figures

| Directory | Content | Count |
|-----------|---------|-------|
| `reports/figures_academic/` | Final academic figures (fig01–fig25+) | 20+ |
| `outputs/06_xgb_sota/shap/` | SHAP value files and plots | 12+ |
| `outputs/06_xgb_sota/plots/` | SHAP bar charts | 6 |
| `submission_package/figures/` | Submission figures (fig05, fig06, fig17) | 3 |
| `outputs/EDA/` | EDA figures | 200+ |

---

## XI. Training Logs

| File | Content |
|------|---------|
| `outputs/06_lstm_sota/master_training.log` | Master LSTM training log |
| `outputs/06_lstm_sota/align_seed42.log` | Timestamp alignment fix log (LSTM bug evidence) |
| `outputs/06_lstm_sota/train_seed42.log` | Seed 42 training |
| `outputs/06_lstm_sota/train_seed123.log` | Seed 123 training |
| `outputs/06_lstm_sota/train_seed777.log` | Seed 777 training |
| `lightning_logs/version_*/hparams.yaml` | 38 hyperparameter configurations from TFT/LSTM training |

---

## XII. Submission Package

| File | Purpose |
|------|---------|
| `submission_package/README.md` | Instructions |
| `submission_package/split_manifest.yaml` | Data splits |
| `submission_package/reproduce_paper_results.py` | Reproduction script |
| `submission_package/evaluation_outputs/benchmark_summary.csv` | Submission copy |
| `submission_package/evaluation_outputs/dm_tests.csv` | Submission copy |
| `submission_package/evaluation_outputs/common_mask_summary.csv` | Submission copy |
| `submission_package/figures/` | Submission figures |
| `CITATION.cff` | Citation metadata |

---

## Summary Statistics

| Category | Count | Notes |
|----------|-------|-------|
| Notebooks (active) | ~15 | 05c, 05e, 05f, 06×4, 07, 08 most critical |
| Notebooks (legacy/archive) | 55+ | In old/ or archive/ directories |
| Production scripts | 10 | production/ directory |
| Analysis scripts | 15 | scripts/ directory |
| Evaluation CSVs (current) | 13 | outputs/07_evaluation/ |
| Prediction CSVs (current) | ~15 | outputs/06_*/ aligned versions |
| Feature JSONs (05e) | 6 | final_features_{target}_D{h}.json |
| Feature JSONs (05f) | 6 | refined_features_{target}_{h}_05f.json |
| Config files | 10 | Including split_manifest.yaml |
| Figures (academic) | 25+ | reports/figures_academic/ |
