# Benchmark Summary (Canonical)

This file is the single source of truth for benchmark metrics.

## Evaluation Configuration
- Train end: 2024-01-31 23:59
- Validation end: 2024-09-30 23:59
- Test window: 2024-10-01 00:00 to 2025-05-31 23:59
- Daytime filter: sun_elevation > 5.0 degrees
- Direct nRMSE denominator: AC_CAPACITY_KW = 3449.00
- Indirect nRMSE denominator: MEAN_DAYTIME_K = 0.6279

## Missing Prediction Files
- C:\Users\204323141\Desktop\solar_ds_capstone\outputs\06_modeling_v2\predictions_all.csv

## Top Model Per Target/Horizon
### Target: direct
- D+1: xgb_sota (nRMSE=12.059%, CI=[10.467,13.625])
- D+2: xgb_sota (nRMSE=12.110%, CI=[10.628,13.617])
- D+3: xgb_sota (nRMSE=12.185%, CI=[10.700,13.705])

### Target: indirect
- D+1: xgb_sota (nRMSE=29.095%, CI=[24.764,33.458])
- D+2: xgb_sota (nRMSE=29.167%, CI=[25.008,33.422])
- D+3: xgb_sota (nRMSE=29.086%, CI=[24.932,33.418])

## Full Metrics Table (CSV)
- benchmark_summary.csv