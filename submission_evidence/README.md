# Submission Evidence Package
**Solar PV Forecasting Capstone — Ktora Plant**
**Prepared:** 2026-05-12

This folder contains the complete evidence package supporting every numerical and methodological claim in the report. Use it in parallel with `REPORT_EVIDENCE_MAP.md` (folder `11_audit/`) to locate the source of any specific value.

---

## Folder Structure

| Folder | Contents | Primary report sections |
|--------|----------|------------------------|
| `01_benchmark/` | Canonical model metrics: nRMSE, MAE, R², 95% CI for all 6 models × 3 horizons × 2 targets | §7 Results tables, Abstract, Conclusions C1–C8 |
| `02_statistical_tests/` | Diebold-Mariano tests, block bootstrap CIs, statistical power / MDE analysis | §8.1 Statistical Significance, §8.2 Power Analysis |
| `03_route_comparison/` | Indirect vs direct route nRMSE advantage per model and horizon | §9 Route Comparison |
| `04_seasonal/` | Seasonal nRMSE breakdown by model, season, horizon, and target | §10 Seasonal Analysis |
| `05_seed_variability/` | XGB-Opt seed spread (seeds 42/123/777); feature ablation seed variability | §11 Robustness, §12 Ablation |
| `06_common_mask/` | Common-timestamp evaluation (intersection of all model evaluation sets) | §11 Common-Mask Evaluation |
| `07_shap/` | SHAP feature group importance; cross-horizon Spearman rank correlation | §12 Interpretability |
| `08_lstm_alignment/` | LSTM timestamp alignment log; aligned predictions; training-set metrics (NOT daytime-eval) | §3.8 LSTM Disclosure, §12 LSTM Alignment |
| `09_feature_selection/` | 05e official feature sets (6 JSONs, used for modelling); 05f post-hoc audit sets (6 JSONs, interpretability only) | §3.6 Feature Selection |
| `10_config/` | Split manifest (train/val/test boundaries), constants (capacity, seeds, normalisation), leakage audit | §2 Data, §3.7 Leakage |
| `11_audit/` | Full audit trail: evidence matrix, action checklist, proposed additions, evidence map | Internal audit |
| `12_scripts/` | Reproducibility scripts: alignment, route comparison, power analysis, seed variability, SHAP stability, ablation | Appendix / reproducibility |
| `13_report/` | Final corrected report DOCX, conda environment spec, CITATION.cff | Submission |

---

## Key File Reference

### For verifying benchmark table values
→ `01_benchmark/benchmark_summary.csv`
- Columns: `model, route, horizon, n_pairs, nRMSE, MAE, R2, RMSE`
- Filter by `route` (indirect/direct) and `horizon` (1/2/3)

### For verifying DM test statistics and p-values
→ `02_statistical_tests/dm_tests.csv`
- Columns: `model, route, horizon, dm_stat, p_value, n_pairs`
- All vs XGB-Opt baseline; NW lags=24

### For verifying statistical power / MDE
→ `02_statistical_tests/dm_tests_extended.csv`
- Adds `n_eff`, `ci_lower`, `ci_upper`, `pct_of_mde`
- Report states 29–50% of MDE across horizons

### For verifying route comparison (1.66–2.10 pp)
→ `03_route_comparison/route_comparison.csv`
- XGB-Opt: 1.98–2.10 pp; BiLSTM-Opt: 1.66–1.89 pp

### For verifying seasonal values (Winter, Spring, Fall)
→ `04_seasonal/seasonal_metrics.csv`
- Columns: `model, route, horizon, season, n, nRMSE`

### For verifying XGB seed spread
→ `05_seed_variability/seed_variability_summary.csv`
- XGB indirect D+1: max range 0.111 pp (seeds 42 vs 777)
- XGB direct D+3: max range 0.092 pp

### For verifying LSTM D+1 nRMSE (authoritative daytime-eval)
→ `01_benchmark/benchmark_summary.csv`, row `lstm_sota, indirect, 1` → 30.237%
→ `08_lstm_alignment/metrics_lstm_sota.csv` contains TRAINING-SET metrics only (~19-20%), NOT comparable to benchmark

### For verifying leakage feature count (19 features in 3 categories)
→ `10_config/leakage_audit.md`

### For verifying 05e feature sets (official modelling inputs)
→ `09_feature_selection/final_features_indirect_D1.json` (and other 5 target×horizon combinations)
- Note: these JSONs include leakage features (fcst_ghi/dni/dhi etc.) that were excluded at training time via EXCL list

### For the complete claim→source mapping
→ `11_audit/REPORT_EVIDENCE_MAP.md`

---

## Important Notes

1. **LSTM seed metrics in `seed_variability_summary.csv` are training-set metrics** (~19–20% indirect, ~8.5% direct). The authoritative LSTM daytime-filtered nRMSE is in `benchmark_summary.csv` (seed 42 only: 30.24% indirect, 12.58% direct).

2. **05e JSONs contain leakage features** — the 05e feature sets predate the April 2026 leakage identification. Features fcst_ghi, fcst_dni, fcst_dhi, fcst_temp_2m, fcst_wind_speed and derived features appear in 05e JSONs but were excluded at training time. The 05f JSONs (`refined_features_*_05f.json`) are post-hoc audit artifacts and did NOT change modelling inputs.

3. **nRMSE denominators differ by target**: indirect nRMSE uses MEAN_DAYTIME_K=0.6279; direct nRMSE uses AC_CAPACITY_KW=3,449 kW. Do not compare indirect and direct nRMSE values directly.

4. **The corrected report** is `13_report/Solar_PV_Forecasting_Report_CORRECTED_v5.docx`. Earlier versions in Downloads are superseded.
