# Report Evidence Map
**Solar PV Forecasting Capstone — Ktora Plant**
**Generated:** 2026-05-12 | **How to use:** Open this file alongside the report. Each section matches the report section number. Every numerical claim links to the exact CSV file, column, and row filter.

---

## Legend
- **File** — relative path from project root (`C:\Users\204323141\Desktop\solar_ds_capstone\`)
- **Row filter** — how to find the exact row (e.g. `model=xgb_sota, target=indirect, horizon=1`)
- **Column** — the specific column that holds the value
- **Value** — the exact number as it appears in the CSV

---

## Abstract

| Claim in report | Value | File | Row filter | Column |
|----------------|-------|------|------------|--------|
| XGB-Opt indirect D+1 nRMSE = 29.09% | 29.0949 | `outputs/07_evaluation/benchmark_summary.csv` | model=xgb_sota, target=indirect, horizon=1 | nRMSE_pct |
| TFT-96 indirect D+1 nRMSE = 29.66% | 29.6641 | `outputs/07_evaluation/benchmark_summary.csv` | model=tft_96, target=indirect, horizon=1 | nRMSE_pct |
| PatchTST indirect D+1 = 30.30% | 30.3005 | `outputs/07_evaluation/benchmark_summary.csv` | model=patchtst, target=indirect, horizon=1 | nRMSE_pct |
| Top-4 within 1.2 pp of each other | 30.30 − 29.09 = 1.21 pp | `outputs/07_evaluation/benchmark_summary.csv` | all models, indirect, horizon=1 | nRMSE_pct |
| Indirect route advantage 1.66–2.10 pp | XGB: 13.14−11.16=1.98 pp; LSTM: 12.95−11.30=1.65 pp | `outputs/07_evaluation/route_comparison.csv` | all rows | nRMSE_kW_pct (indirect−direct) |
| Statistical power: most differences below MDE | LSTM vs XGB: 29–50% of MDE | `outputs/07_evaluation/dm_tests_extended.csv` | model_a=lstm_sota, model_b=xgb_sota | pct_of_mde |

---

## §1.1 — Contributions (Introduction)

| Claim in report | Value | File | Row filter | Column |
|----------------|-------|------|------------|--------|
| 19 leakage features identified and excluded | 19 (Cat1=7, Cat2=10, Cat3=2) | `docs/leakage_audit.md` | Sections 1–3 | — |
| Feature-selection: 3-method consensus (SHAP + Stability + MI) | n_starting=42, consensus=40 | `artifacts/05_feature_selection/final_features_indirect_D1.json` | keys: n_starting_features, methods | — |
| Six models on identical splits | Split dates defined | `configs/split_manifest.yaml` | — | train/val/test boundaries |
| Indirect route 1.66–2.10 pp advantage | See Abstract row above | `outputs/07_evaluation/route_comparison.csv` | — | — |
| SHAP & TFT attention: same lag structure (−24h, −48h, −168h) | Visual + SHAP top features | `outputs/06_xgb_sota/shap/shap_indirect_D1.csv` | — | — |
| Power analysis: comparisons below MDE at n≈2,500 | 29–50% of MDE | `outputs/07_evaluation/dm_tests_extended.csv` | model_a=lstm_sota | pct_of_mde |

---

## §3.6 — Feature Selection Pipeline

| Claim in report | Value | File | Row filter / Key | Column / Field |
|----------------|-------|------|-----------------|----------------|
| Starting pool ~86 features → 42 entering consensus | n_starting_features=42 | `artifacts/05_feature_selection/final_features_indirect_D1.json` | key: n_starting_features | — |
| SHAP backward elimination baseline nRMSE = 16.03% | 16.0327 | `artifacts/05_feature_selection/final_features_indirect_D1.json` | key: baseline_nrmse | — |
| SHAP elbow retained 40 features | 40 | `artifacts/05_feature_selection/final_features_indirect_D1.json` | key: methods → shap_elbow | n_selected |
| Stability selection retained 19 features | 19 | `artifacts/05_feature_selection/final_features_indirect_D1.json` | key: methods → stability_selection | n_selected |
| MI retained 40 features | 40 | `artifacts/05_feature_selection/final_features_indirect_D1.json` | key: methods → mutual_information | n_selected |
| 2-of-3 consensus = 40 features (indirect D+1) | 40 | `artifacts/05_feature_selection/final_features_indirect_D1.json` | key: n_consensus_features | — |
| 05f CORE = 15, SUPPORTED = 8, CONDITIONAL = 1, EXCLUDED = 18 | per tier | `artifacts/05_feature_selection/refined_features_indirect_D1_05f.json` | tiers dict | counts |
| 05f recommended = 24 features | 24 | `artifacts/05_feature_selection/refined_features_indirect_D1_05f.json` | key: recommended_05f | count |
| 05e produced BEFORE leakage audit (fcst_ghi/dni/dhi in JSON) | present in recommended set | `artifacts/05_feature_selection/final_features_indirect_D1.json` | recommended_features list | fcst_ghi, fcst_dni, fcst_dhi visible |
| Models exclude leakage via EXCL list (lines 51–73) | — | `production/02_models/xgb_sota.py` | lines 51–73 | EXCL variable definition |
| Backup of pre-leakage JSONs | — | `artifacts/05_feature_selection/_pre_leakage_fix_backup/` | — | — |

---

## §3.7 — Anti-Leakage Protocol

| Claim in report | Value | File | Detail |
|----------------|-------|------|--------|
| 19 features excluded as leakage | 19 total | `docs/leakage_audit.md` | Category 1: 7 target-duplicate sensors; Category 2: 10 same-timestamp NWP archive features; Category 3: 2 same-timestamp SCADA sensors |
| Category 1 — target duplicates (7) | actual_power_kW, performance_ratio, site_total_AC_power_kW, inv1_AC_power_kW, inv2_AC_power_kW, ghi_measured, irradiance_proxy | `docs/leakage_audit.md` | Section 1 |
| Category 2 — NWP archive (10) | fcst_ghi, fcst_dni, fcst_dhi, fcst_temp_2m, fcst_cloud_cover, fcst_wind_speed, poa_global_forecast, poa_direct_forecast, poa_diffuse_forecast, mod_temp_forecast | `docs/leakage_audit.md` | Section 2 |
| Category 3 — SCADA sensors (2) | mod_temp1, ambient_temp | `docs/leakage_audit.md` | Section 3 |

---

## §3.8 — Data Split

| Claim in report | Value | File | Field |
|----------------|-------|------|-------|
| Train ≤ 2024-01-31 | 2024-01-31 | `configs/split_manifest.yaml` | train_end |
| Validation 2024-02-01 → 2024-09-30 | — | `configs/split_manifest.yaml` | val_start, val_end |
| Test 2024-10-01 → 2025-05-31 | — | `configs/split_manifest.yaml` | test_start, test_end |
| AC capacity = 3,449 kW (nRMSE denominator, direct) | 3449.0 | `production/00_config/constants.py` | AC_CAPACITY_KW |
| MEAN_DAYTIME_K = 0.6279 (nRMSE denominator, indirect) | 0.6279 | `production/00_config/constants.py` | MEAN_DAYTIME_K |
| Seeds = [42, 123, 777] | [42, 123, 777] | `production/00_config/constants.py` | RANDOM_SEEDS |

---

## §7 — Main Benchmark Results (Tables 2 & 3)

### Indirect Target D+1 nRMSE — all rows from `outputs/07_evaluation/benchmark_summary.csv`

| Model | nRMSE % | 95% CI | R² | n | MAE | Row filter |
|-------|---------|--------|-----|---|-----|------------|
| XGB-Opt | **29.09%** | [24.76, 33.46] | 0.847 | 2,580 | 0.0939 | model=xgb_sota, target=indirect, h=1 |
| TFT-96 | 29.66% | [24.71, 34.78] | 0.847 | 2,554 | 0.0750 | model=tft_96, target=indirect, h=1 |
| PatchTST | 30.30% | [26.02, 34.53] | 0.834 | 2,580 | 0.0975 | model=patchtst, target=indirect, h=1 |
| BiLSTM-Opt | 30.24% | [25.22, 35.20] | 0.834 | 2,510 | 0.0812 | model=lstm_sota, target=indirect, h=1 |
| TFT-168 | 31.05% | [26.49, 35.91] | 0.833 | 2,554 | 0.0834 | model=tft_168, target=indirect, h=1 |
| TFT baseline | 31.76% | [27.83, 36.48] | 0.817 | 2,554 | 0.1163 | model=tft, target=indirect, h=1 |

### Direct Target D+1 nRMSE — all rows from `outputs/07_evaluation/benchmark_summary.csv`

| Model | nRMSE % | 95% CI | R² | n | Row filter |
|-------|---------|--------|-----|---|------------|
| XGB-Opt | **12.06%** | [10.47, 13.62] | 0.812 | 2,580 | model=xgb_sota, target=direct, h=1 |
| TFT-96 | 12.11% | [10.41, 13.79] | 0.828 | 2,554 | model=tft_96, target=direct, h=1 |
| BiLSTM-Opt | 12.58% | [10.82, 14.40] | 0.794 | 2,510 | model=lstm_sota, target=direct, h=1 |
| TFT baseline | 12.80% | [11.50, 14.38] | 0.788 | 2,554 | model=tft, target=direct, h=1 |
| TFT-168 | 13.26% | [11.80, 14.85] | 0.794 | 2,554 | model=tft_168, target=direct, h=1 |
| PatchTST | 13.84% | [12.37, 15.32] | 0.752 | 2,580 | model=patchtst, target=direct, h=1 |

> **Note on R²:** XGB-Opt direct D+1 R²=0.8116 from CSV → rounds to **0.812** (3 dp). The report (v1–v5) incorrectly showed 0.808, which is the D+3 value (0.8076). Corrected to 0.812 in v6.

### n Varies by Model — Source: benchmark_summary.csv, column `n`

| Model | n (indirect D+1) | n (direct D+1) | Reason |
|-------|-----------------|----------------|--------|
| XGB-Opt | 2,580 | 2,580 | No lookback window required |
| PatchTST | 2,580 | 2,580 | Autoregressive, no exogenous window |
| TFT-96 | 2,554 | 2,554 | 96-hour lookback drops first 4 days |
| TFT-168 | 2,554 | 2,554 | 168-hour lookback |
| TFT baseline | 2,554 | 2,554 | Same fixed architecture |
| BiLSTM-Opt | 2,510 | 2,510 | Encoder window requires more history |

---

## §8.1 — Diebold-Mariano Tests

**Source file:** `outputs/07_evaluation/dm_tests.csv`
**Methodology:** NW lags=24, two-sided, α=0.05 — column `nw_lags`

### Key DM results (indirect D+1 vs XGB-Opt)

| Comparison | n_pairs | DM stat | p-value | Significant? | Row filter |
|-----------|---------|---------|---------|-------------|------------|
| BiLSTM-Opt vs XGB | 2,510 | 1.14 | 0.255 | **No** | target=indirect, h=1, model_a=lstm_sota |
| TFT baseline vs XGB | 2,554 | 2.45 | 0.014 | Yes | target=indirect, h=1, model_a=tft |
| TFT-96 vs XGB | 2,554 | **14.44** | 0.000 | Yes | target=indirect, h=1, model_a=tft_96 |
| TFT-168 vs XGB | 2,554 | 9.37 | 0.000 | Yes | target=indirect, h=1, model_a=tft_168 |
| PatchTST vs XGB | 2,580 | 2.52 | 0.012 | Yes | target=indirect, h=1, model_a=patchtst |

### Key DM results (direct D+1 vs XGB-Opt)

| Comparison | n_pairs | DM stat | p-value | Significant? | Row filter |
|-----------|---------|---------|---------|-------------|------------|
| BiLSTM-Opt vs XGB | 2,510 | 1.23 | 0.218 | **No** | target=direct, h=1, model_a=lstm_sota |
| TFT baseline vs XGB | 2,554 | 1.40 | 0.161 | **No** | target=direct, h=1, model_a=tft |
| TFT-96 vs XGB | 2,554 | 11.25 | 0.000 | Yes | target=direct, h=1, model_a=tft_96 |
| TFT-168 vs XGB | 2,554 | 10.01 | 0.000 | Yes | target=direct, h=1, model_a=tft_168 |
| PatchTST vs XGB | 2,580 | 6.80 | 0.000 | Yes | target=direct, h=1, model_a=patchtst |

> **n_pairs varies by model** — 2,510 (BiLSTM), 2,554 (TFT variants), 2,580 (PatchTST). The sentence in §8.1 citing n=2,554 refers specifically to the TFT-96 comparison.

---

## §8.2 — Statistical Power Analysis

**Source file:** `outputs/07_evaluation/dm_tests_extended.csv`

### BiLSTM-Opt vs XGB-Opt: % of MDE (all horizons)

| Target | Horizon | n_eff | % of MDE | p-value | Row filter |
|--------|---------|-------|---------|---------|------------|
| indirect | D+1 | 806.5 | **39.3%** | 0.255 | target=indirect, h=1, model_a=lstm_sota |
| indirect | D+2 | 791.0 | **44.4%** | 0.213 | target=indirect, h=2, model_a=lstm_sota |
| indirect | D+3 | 703.6 | **50.0%** | 0.149 | target=indirect, h=3, model_a=lstm_sota |
| direct | D+1 | 560.7 | **38.1%** | 0.218 | target=direct, h=1, model_a=lstm_sota |
| direct | D+2 | 552.0 | **35.2%** | 0.248 | target=direct, h=2, model_a=lstm_sota |
| direct | D+3 | 468.3 | **29.0%** | 0.308 | target=direct, h=3, model_a=lstm_sota |

> Range 29–50% cited in §8.2. n_eff range 468–806 (vs nominal n=2,510–2,580 — ratio ≈5–6× due to autocorrelation).

---

## §9 — Route Comparison (Indirect vs Direct)

**Source file:** `outputs/07_evaluation/route_comparison.csv`
**Normalisation:** Both targets expressed at AC capacity scale (3,449 kW denominator)

### XGB-Opt route comparison (all horizons)

| Horizon | Indirect nRMSE% | Direct nRMSE% | Advantage (pp) | Row filter |
|---------|----------------|--------------|---------------|------------|
| D+1 | 11.16% | 13.14% | **1.98 pp** | model=xgb_sota, h=1 |
| D+2 | 11.19% | 13.20% | **2.01 pp** | model=xgb_sota, h=2 |
| D+3 | 11.18% | 13.28% | **2.10 pp** | model=xgb_sota, h=3 |

### BiLSTM-Opt route comparison (all horizons)

| Horizon | Indirect nRMSE% | Direct nRMSE% | Advantage (pp) | Row filter |
|---------|----------------|--------------|---------------|------------|
| D+1 | 11.30% | 12.95% | **1.66 pp** | model=lstm_sota, h=1 |
| D+2 | 11.34% | 13.07% | **1.72 pp** | model=lstm_sota, h=2 |
| D+3 | 11.20% | 13.09% | **1.89 pp** | model=lstm_sota, h=3 |

> Report says "1.66–2.10 pp across XGB and BiLSTM families". Min = 1.655 (LSTM D+1), Max = 2.10 (XGB D+3).

---

## §10 — Seasonal Analysis

**Source file:** `outputs/07_evaluation/seasonal_metrics.csv`
**Filter:** target=indirect, horizon=1

| Model | Winter (Dec–Feb) | Spring (Mar–May) | Fall (Oct–Nov) | Row filter |
|-------|-----------------|-----------------|----------------|------------|
| XGB-Opt | **32.22%** | 22.33% | 34.74% | model=xgb_sota, indirect, h=1 |
| BiLSTM-Opt | 32.26% | — | — | model=lstm_sota |
| TFT-96 | 32.27% | — | 35.07% | model=tft_96 |
| TFT-168 | 35.65% | — | — | model=tft_168 |
| TFT baseline | 35.10% | **20.9%** | **41.3%** | model=tft |
| PatchTST | 33.50% | — | — | model=patchtst |

> Key correction: Report previously stated XGB Winter = 27.0%. Correct value = 32.22% (verified from seasonal_metrics.csv).

---

## §11 — Seed Variability & Robustness

**Source file:** `outputs/07_evaluation/seed_variability_summary.csv`

### XGB-Opt (authoritative daytime-filtered evaluation across 3 seeds)

| Target | Horizon | Mean nRMSE | Std | Min | Max | **Max spread** |
|--------|---------|-----------|-----|-----|-----|---------------|
| indirect | D+1 | 29.053% | 0.060 pp | 28.984% | 29.095% | **0.111 pp** |
| indirect | D+2 | 29.118% | 0.045 pp | 29.079% | 29.167% | 0.088 pp |
| indirect | D+3 | 29.081% | 0.028 pp | 29.050% | 29.106% | 0.056 pp |
| direct | D+1 | 12.058% | 0.007 pp | 12.050% | 12.064% | 0.014 pp |
| direct | D+2 | 12.132% | 0.035 pp | 12.110% | 12.172% | 0.062 pp |
| direct | D+3 | 12.135% | 0.047 pp | 12.093% | 12.185% | **0.092 pp** |

> **Report §11:** "Maximum spread across seeds: 0.111 pp (indirect D+1) and 0.092 pp (direct D+3)"

### ⚠️ BiLSTM-Opt seed rows — TRAINING-SET METRICS (NOT daytime-filtered)

| Target | Horizon | Mean (training) | Note |
|--------|---------|----------------|------|
| indirect | D+1 | ~19.7% | NOT comparable to benchmark 30.24% |
| direct | D+1 | ~8.5% | NOT comparable to benchmark 12.58% |

> BiLSTM authoritative values: indirect D+1 = **30.237%**, direct D+1 = **12.577%** — from `benchmark_summary.csv`, model=lstm_sota, horizon=1.

### Common-Mask Evaluation

**Source:** `outputs/07_evaluation/common_mask_summary.csv`

| Target | n (shared) | XGB nRMSE | BiLSTM nRMSE | TFT baseline nRMSE |
|--------|-----------|-----------|-------------|-------------------|
| direct D+1 | **2,116** | 13.30% | 13.72% | 14.05% |
| indirect D+1 | **2,109** | 32.15% | 33.25% | 34.91% |

> Rankings identical to full-set evaluation. Source rows: target=direct/indirect, horizon=1.

---

## §12 — SHAP Interpretability

### Feature Group Importance

**Source:** `outputs/07_evaluation/shap_group_importance.csv`, target=direct/indirect, horizon=1

| Group | Direct D+1 % | Indirect D+1 % | Column |
|-------|-------------|----------------|--------|
| rolling_stats | **70.7%** | **57.0%** | pct |
| lag_power | 16.3% | 11.4% | pct |
| time_encodings | 6.6% | — | pct |
| clearsky | 1.5% | 11.3% | pct |

### SHAP Cross-Horizon Rank Stability

**Source:** `outputs/07_evaluation/shap_horizon_corr.csv`

| Comparison | Spearman ρ | p-value |
|-----------|-----------|---------|
| indirect D+1 vs D+2 | 0.656 | 0.000 |
| indirect D+1 vs D+3 | **0.512** | 0.000 |
| indirect D+2 vs D+3 | 0.769 | 0.000 |
| direct D+1 vs D+2 | 0.713 | 0.000 |
| direct D+1 vs D+3 | 0.603 | 0.000 |
| direct D+2 vs D+3 | **0.798** | 0.000 |

> Report says "Spearman ρ = 0.51–0.80". Min = 0.512 (indirect D1 vs D3), Max = 0.798 (direct D2 vs D3).

### Feature Group Ablation

**Source:** `outputs/07_evaluation/ablation_seed_variability.csv`, target=indirect, horizon=1

| Ablation | Delta nRMSE (pp) | Interpretation |
|---------|-----------------|----------------|
| Remove rolling_mean_7d | **+0.393 pp** | Most valuable group |
| Remove all features (minimal) | **+2.141 pp** | Full feature set value |
| Remove rolling_mean_3d | **−0.029 pp** | Redundant with 7d |
| Remove clearsky features | −0.130 pp | Partially redundant indirect |

---

## §12 — LSTM Timestamp Alignment (BiLSTM Disclosure)

**Source:** `outputs/06_lstm_sota/align_seed42.log`

| Before correction | After correction | Delta |
|------------------|-----------------|-------|
| ~17% (approximate, read from log) | **30.24%** (indirect D+1) | ~13 pp |

- Correction formula: `valid_ts = window_end − (DECODER_LEN − 1 − target_step) × 1h`
- Aligned predictions: `outputs/06_lstm_sota/predictions_lstm_sota_aligned.csv`
- Pre-correction archive: `outputs/06_lstm_sota/predictions_lstm_sota.csv`

---

## §14 — Conclusions C1–C8

| Conclusion | Claim | Value | Source |
|-----------|-------|-------|--------|
| C1 | XGB-Opt best indirect D+1 | 29.09% | benchmark_summary.csv, xgb_sota, indirect, h=1, nRMSE_pct |
| C1 | XGB-Opt best direct D+1 | 12.06% | benchmark_summary.csv, xgb_sota, direct, h=1, nRMSE_pct |
| C1 | TFT-96 second indirect | 29.66% | benchmark_summary.csv, tft_96, indirect, h=1 |
| C2 | Indirect advantage 1.66–2.10 pp | Full table above | route_comparison.csv |
| C3 | XGB horizon flat (indirect D+3−D+1) | −0.008 pp | benchmark_summary.csv, D+3 vs D+1 for xgb_sota, indirect |
| C4 | PatchTST 30.30% with zero exogenous features | 30.300% | benchmark_summary.csv, patchtst, indirect, h=1 |
| C5 | TFT-96 outperforms TFT-168 by 1.38 pp | 31.05−29.66 = 1.38 pp | benchmark_summary.csv, tft_96 vs tft_168, indirect, h=1 |
| C7 | Top-4 within 1.21 pp | 30.30−29.09 = 1.21 pp | benchmark_summary.csv, indirect, h=1 |
| C8 | BiLSTM vs XGB below MDE | 29–50% of MDE | dm_tests_extended.csv, lstm_sota vs xgb_sota |

---

## Reproducibility Chain

To reproduce any number in the report from scratch:

```
Raw SCADA
  → 03_preprocessing_UNIFIED.ipynb  (hourly resampling, UTC)
  → 04_feature_engineering_v5.ipynb (80+ features)
  → docs/leakage_audit.md           (19 features excluded)
  → 05e consensus JSONs              (official feature sets)
  → xgb_sota.py / lstm_sota.py      (training + Optuna, EXCL list applied)
  → align_and_evaluate.py            (LSTM timestamp fix + evaluate.py)
  → outputs/07_evaluation/           (all CSVs in this map)
```

**Single script to reproduce all tables:** `submission_package/reproduce_paper_results.py`
**Environment:** `environment.yml`
**Split definition:** `configs/split_manifest.yaml`
**Constants:** `production/00_config/constants.py`
