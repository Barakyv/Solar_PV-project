# Reproduction 20261112T202013Z

- commit `09533a318d`, work folder `C:\Users\204323141\repro_g1\r2`
- options: {'with_eda': True, 'rerun_selection': True, 'neural': True}
- solarpv imported from `C:\Users\204323141\repro_g1\r2\src\solarpv\__init__.py`
- result: **NOT reproduced**

- floating-point noise: every differing cell numeric, relative difference at most 1e-09, nothing missing on one side only (reported, not failed)

| step | exit | seconds | files | equal | float noise | failing | listed |
|---|---|---|---|---|---|---|---|
| checksums | 0 | 2.7 | 0 | 0 | 0 | 0 | 0 |
| ingest | 0 | 50.8 | 4 | 2 | 2 | 0 | 0 |
| clock | 0 | 97.2 | 8 | 4 | 4 | 0 | 0 |
| preprocess | 0 | 173.6 | 3 | 3 | 0 | 0 | 0 |
| clean | 0 | 40.3 | 6 | 6 | 0 | 0 | 0 |
| eda | 1 | 33.9 | 15 | 5 | 0 | 0 | 10 |

## Failing and listed files

- ingest: `data/processed/01_ingest/ingest_qa.csv` (table): 8 numeric cells differ (max abs 7.28e-12, max rel 2.02e-16), 0 other cells, in ['min', 'p50', 'max']; DataFrame.iloc[:, 4] (column name="min") are different DataFrame.iloc[:, 4] (column name="min") values are different (3.7037 %) [index]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
- ingest: `data/processed/01_ingest/scada_1min_utc.parquet` (table): 6669577 numeric cells differ (max abs 7.28e-12, max rel 9.74e-14), 0 other cells, in ['iec_mode', 'iec_sp_w', 'iec_sp_q', 'iec_sp_v', 'iec_sp_pf', 'ambient_temp']; DataFrame.iloc[:, 0] (column name="iec_mode") are different DataFrame.iloc[:, 0] (column name="iec_mode") values are different (0.00061 %) [index]: [2021-08-31 21:00:00+00:00, 2021-08-31 21:01:00+00:0
- clock: `00_data_audit/clock_drift_2026-10-14/daily_estimates_minute_after.csv` (table): 1768 numeric cells differ (max abs 1.84e-12, max rel 3.24e-12), 0 other cells, in ['tau_met', 'clear_rel_rmse', 'clear_ratio', 'tau_dc', 'dc_rel_rmse']; DataFrame.iloc[:, 1] (column name="tau_met") are different DataFrame.iloc[:, 1] (column name="tau_met") values are different (53.64135 %) [index]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
- clock: `00_data_audit/clock_drift_2026-10-14/daily_estimates_minute_before.csv` (table): 1529 numeric cells differ (max abs 2.75e-12, max rel 1.91e-12), 0 other cells, in ['tau_met', 'clear_rel_rmse', 'clear_ratio', 'tau_dc', 'dc_rel_rmse']; DataFrame.iloc[:, 1] (column name="tau_met") are different DataFrame.iloc[:, 1] (column name="tau_met") values are different (40.25059 %) [index]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
- clock: `00_data_audit/clock_drift_2026-10-14/daily_estimates_polled_after.csv` (table): 94 numeric cells differ (max abs 1.88e-13, max rel 3.3e-14), 0 other cells, in ['tau_met', 'clear_rel_rmse', 'clear_ratio', 'tau_dc', 'dc_rel_rmse']; DataFrame.iloc[:, 1] (column name="tau_met") are different DataFrame.iloc[:, 1] (column name="tau_met") values are different (46.83544 %) [index]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
- clock: `00_data_audit/clock_drift_2026-10-14/daily_estimates_polled_before.csv` (table): 94 numeric cells differ (max abs 8.12e-13, max rel 6.56e-14), 0 other cells, in ['tau_met', 'clear_rel_rmse', 'clear_ratio', 'tau_dc', 'dc_rel_rmse']; DataFrame.iloc[:, 1] (column name="tau_met") are different DataFrame.iloc[:, 1] (column name="tau_met") values are different (48.10127 %) [index]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

## Environments (nothing installed)

- solar: Python 3.11.18, 306 pins in `environment.lock.yml`, differing: {'pyside6': {'lock': '6.11.3', 'installed': '6.11.3 '}, 'tzdata': {'lock': '2026e', 'installed': '2026.6'}}
- solar-gpu: 43 pins, differing: {}
