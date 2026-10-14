# Leakage Audit — Solar DS Capstone
**Last updated:** 2026-04-23

## Summary

This document records every feature that was removed due to data leakage concerns,
the stage at which it was identified, and the reason for exclusion.

## Leakage Categories

### Category 1: Same-Timestamp Target Duplicates
Features that encode the actual generation at the forecast hour — directly or algebraically
equivalent to the target. Available only in historical records, not at midnight issuance.

| Feature | Reason for Exclusion |
|---------|---------------------|
| `actual_power_kW` | Duplicate of `site_total_AC_kW` under a different name (r ≈ 1.0 with target) |
| `performance_ratio` | ≈ target_k (AC_kW / clearsky_power, r = 0.95 with target) |
| `s01_inv1_AC_kW` | Inverter-level sub-component of the target |
| `s01_inv2_AC_kW` | Inverter-level sub-component of the target |
| `s01_inv1_DC_kW` | Same-timestamp SCADA — not available at midnight issuance |
| `s01_inv2_DC_kW` | Same-timestamp SCADA — not available at midnight issuance |
| `sum_inv_ac_kW` | Algebraic duplicate of site_total_AC_kW |

### Category 2: Same-Timestamp NWP Reanalysis (Archive API)
These columns come from the Open-Meteo archive-api (GFS Seamless reanalysis), merged at the
SAME timestamp as the target. They represent actual atmospheric conditions at the forecast
hour — NOT a true D+1 forecast issued at midnight.

| Feature | Reason for Exclusion |
|---------|---------------------|
| `fcst_ghi` | Archive reanalysis GHI — same timestamp as target |
| `fcst_dni` | Archive reanalysis DNI — same timestamp as target |
| `fcst_dhi` | Archive reanalysis DHI — same timestamp as target |
| `fcst_temp_2m` | Archive reanalysis temperature — same timestamp |
| `fcst_cloud_cover` | Archive reanalysis cloud cover — same timestamp |
| `fcst_wind_speed` | Archive reanalysis wind speed — same timestamp |
| `poa_global_forecast` | Derived from fcst_ghi (same-timestamp NWP) |
| `poa_direct_forecast` | Derived from fcst_dni (same-timestamp NWP) |
| `poa_diffuse_forecast` | Derived from fcst_dhi (same-timestamp NWP) |
| `wind_speed_squared` | Derived from fcst_wind_speed |

### Category 3: Same-Timestamp SCADA Sensors
Physical measurements at the exact forecast hour — not available at midnight issuance.

| Feature | Reason for Exclusion |
|---------|---------------------|
| `mod_temp1` | Module temperature — measured at forecast hour |
| `ambient_temp` | Ambient temperature — measured at forecast hour |

### Category 4: Historical Features (Leakage-Affected Models Only)
Used in XGB Phase-1 and Phase-2 but identified as leaky. These models appear in
the report for evolution context only and are NOT valid baselines.

| Feature | Reason for Exclusion |
|---------|---------------------|
| `theoretical_power` | Derived from clearsky + capacity — structurally encodes target scale |

## Retained Safe Features (Examples)
- **Clear-sky irradiance** (`ghi_clearsky`, `dhi_clearsky`, `dni_clearsky`): Computed
  from pvlib using only site geometry and calendar — no actual measurements.
- **Rolling production statistics** (e.g., `site_total_rolling_mean_7d_same_tod`):
  7-day window ending at `T-24h` (yesterday same time). Lag-safe at all horizons.
- **Horizon-lagged features** (`lag_1d`, `lag_2d`): Yesterday / two-days-ago values.
  Excluded progressively for D+2 (lag_1d excluded) and D+3 (lag_1d + lag_2d excluded).
- **Sun geometry** (`sun_elevation`, `solar_azimuth`, `hour_cos`, `hour_sin`): Purely
  deterministic from calendar and site location.

## Verification
The leakage check was performed in `production/02_models/xgb_sota.py` (EXCL list, lines 51-73)
and `production/01_feature_engineering/build_features.py`.
A y_true consistency assertion in `production/03_evaluation/evaluate.py` (A1.1) confirms
that aligned pairs share identical observed values across models.

## Phase Identification
- **Phase-1/Phase-2 leakage** identified: 2025-10 (model evolution)
- **NWP archive leakage** identified: 2026-04-08 (post-training audit)
- **Timestamp alignment fix** applied: 2026-04 (LSTM decoder deduplication)
