# Leakage audit (v2, rebuild)

Supersedes `submission_evidence/10_config/leakage_audit.md` (2026-04-23).
Code: `src/solarpv/features/leakage.py`, `src/solarpv/config/constants.py`. Tests: T-01, T-09, T-17, T-18.

## Issuance rule

Forecasts are issued at **T0 = 00:00 local time (Asia/Jerusalem)**. D+h is the h-th local day starting at T0.
A value may enter a D+h feature only if it is known at T0. All data are processed in true UTC (fix F-01).

## 1. Same-timestamp information (blocked by name)

| Category | Examples | Why |
|---|---|---|
| Targets and duplicates | `target_k`, `site_total_AC_kW`, `actual_power_kW`, `sum_inv_ac_kW`, inverter AC/DC, `performance_ratio` | Equal or algebraically tied to the target |
| Reanalysis NWP | `fcst_ghi`, `fcst_dni`, `fcst_dhi`, `fcst_temp_2m`, `fcst_cloud_cover`, `fcst_wind_speed`, `poa_*_forecast`, `wind_speed_squared` | Open-Meteo archive = observed atmosphere at the target hour |
| Same-timestamp sensors | `irradiance`, `pyranometer`, `mod_temp1/2`, `ambient_temp`, wind, soiling, precipitation | Measured at the target hour |
| Derived from same-timestamp power | `theoretical_power`*, `site_loading`, efficiencies, `site_power_per_irr`, QA flags, inverter state | Built from target-hour data |

\* `theoretical_power` is a deterministic clear-sky quantity, but it was listed as leakage in the original audit and had zero importance; it is removed for consistency (D-03).

Total: the exact-name list in `leakage.py` plus patterns (`^fcst_` except `fcst_gfs_`, `_forecast$`, `^iec_`, `_kVar$`, `_PF$`, `_state$`, `_flag$`, `^has_`, …).

## 2. Information after T0 (blocked by construction)

| Feature family | D+1 | D+2 | D+3 |
|---|---|---|---|
| `*_lag_1d` | allowed | dropped | dropped |
| `*_lag_2d` | allowed | allowed | dropped |
| `*_lag_3d/7d/14d` | allowed | allowed | allowed |
| Rolling same-tod (N days) | lags 1..N | lags 2..N+1 | lags 3..N+2 |
| Clear-sky, solar geometry, calendar | allowed (deterministic) | allowed | allowed |
| `fcst_gfs_*` (stage 6b) | 12Z run of previous UTC day, init + 6 h ≤ T0 | same run | same run |

The original rolling features used lags 1..N at every horizon (F-07) and are corrected here.
Verified by perturbation tests: changing any value after T0 must not change any D+h feature (T-09, T-18).

## History

- 2025-10: Phase-1/2 leakage (`theoretical_power`, target duplicates) identified.
- 2026-04-08: NWP archive leakage identified.
- 2026-10-13: rebuild audit — time-zone bug (F-01), rolling leakage at D+2/D+3 (F-07).
- 2026-10-15: stage-5 quality columns blocked as model inputs: `n_*`, `src_*`, `rule_*`, `frac_*`, `*_flag`, `hour_source` (T-01; the original used quality flags such as `has_statistical_outlier` as features).
