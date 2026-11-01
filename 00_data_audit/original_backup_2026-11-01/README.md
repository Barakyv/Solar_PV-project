# Audit of the original project backup (2026-11-01)

On 2026-11-01 the user placed `old files/solar_ds_capstone/`, an older backup of the original
project, in the project folder. **It is a step on the way, not the final project.** It dates from
about January 2026; the project went through several more iterations before the v6 submission
(`submission_evidence/`). Wherever the backup and the submission disagree, the submission
prevails (PROJECT_MASTER.md §0, rule 5). The backup is read-only, like the rest of `old files/`.

The review notebook `notebooks/00b_original_backup_review.ipynb` shows the evidence below live.

## How to reproduce

```
python 00_data_audit/original_backup_2026-11-01/inventory.py
python 00_data_audit/original_backup_2026-11-01/selection_evidence.py
python 00_data_audit/original_backup_2026-11-01/weather_source.py
```

`inventory.py` reads the partial `.git` folder of the backup with zlib only and writes the blobs
that exist nowhere else to `data/external/original_backup_recovered/` (git-ignored). No git
command, notebook, pickle or model file from the backup is executed or unpickled. The site
coordinates found in the backup are private and are not written out. Run logs: `runs/`.

## What the backup holds

| | Files | Size |
|---|---|---|
| Working copy | 508 | 1.0 GB |
| Git history only (loose blobs that match no working-copy file) | 400 | 1.3 GB |

The git folder is partial: no HEAD, config, index or root tree; the reflog lists commits on
2025-10-17 and 2026-01-20. Groups (`inventory.csv`, `inventory_summary.json`):

- **Feature selection, base stage (13 files):** the direct and indirect selection notebooks of
  January 2026, their reports and their outputs. **Relevant to stage 7.**
- **Feature selection, earlier version (1):** the December 2025 notebook, identical to
  `old files/05_feature_selection.ipynb`.
- **Weather source of the original NWP (3):** the HTTP cache of the feature notebook (two
  versions) and `ketura_gfs_forecast.parquet`. **Relevant to stage 6b and the report.**
- **Modelling, intermediate (56):** XGBoost with Optuna, quantile XGBoost, a Keras BiLSTM with
  attention, PyTorch LSTM-attention and time-series-transformer experiments, checkpoints.
  Context for stage 9; none is the final code.
- **Feature engineering, intermediate (39):** versions 2, 3 and 5 of notebook 04, earlier
  pipelines.
- **EDA, ingest, preprocessing (578), figures (171), raw export copies (21), tables (9),
  configuration (17):** earlier versions of stages 2-5, already rebuilt; no new information.

**Not in the backup:** `xgb_sota.py`, the final PyTorch `lstm_sota.py`, the TFT and PatchTST code,
and the code of the 05c, 05e and 05f selection steps (`final_code_present` in the summary). R-16
stays open.

## Findings

### 1. The base stage of the submitted selection (stage 7)

`selection_evidence.json`

- The direct and indirect base-selection notebooks of January 2026 each end with 42 features.
  **These are exactly the 42 starting features of the submitted D+1 consensus files** (05e), for
  both targets. The D+2 and D+3 starting sets (36 and 34 or 35) are the same list minus the lags
  not available at that horizon: the base stage ran once, not per horizon.
- Their parameters differ from the December notebook that PROJECT_MASTER.md §7 had taken as the
  base: MI threshold 0.02 (not 0.01); a protected core of 12 features (solar zenith, solar
  azimuth, aoi; five weather forecasts; hour and day-of-year encodings), not 17; category minimums
  solar 5, weather 5, temporal 4, lag 6, rolling 4; RFE to 42 features with five time-series
  folds.
- **Both read the whole feature table** and cross-validate over all rows: no filter by split or
  date, so the selection saw the validation and test periods (F-26). Gaps were filled with the
  median of the whole column (indirect) or with zero before SHAP (direct).
- **The direct notebook did not forbid the per-inverter DC power of the same hour** (only the
  site total); the indirect one did. In the direct SHAP ranking the two DC columns are first and
  second, with 95 % of the total importance; the first is 17 times the third feature (F-27).
  The submitted files show the DC columns removed afterwards (`_leakage_fix_removed`), after they
  had shaped the ranking and the elbow.

### 2. The original weather "forecast" (stage 6b, report)

`weather_source.json`, `weather_lag_scan.csv`

- The cached request is the Open-Meteo **archive** endpoint with `models=gfs_seamless`, one
  request for 2021-09-01 to 2025-05-31 in UTC: a single continuous series with no issue time,
  so every value is the one valid at the target hour, unknown at a midnight issuance (F-28).
- It is GFS-based, not a perfect observation: against the measured clear-sky index (train and
  validation), its daily correlation (0.822) is about that of the real D+1 GFS run of stage 6b
  (0.806). The leak is one of timing, not of accuracy.
- It aligns with the measurement only after a one-hour shift (the API labels the mean of the
  preceding hour at the hour end, N7): hourly r 0.631. At the 2-3 h offset of the original join
  (F-01, N3) the hourly r is 0.301 and 0.268. This is a likely reason why the original weather
  features contributed little, while correctly aligned real forecasts now lower the error by
  0.9-1.7 points (D-12).

### 3. Modelling (stage 9, context only)

The intermediate notebooks confirm known issues (k clipped at 1.2, F-06; nRMSE divided by the
largest observed value; chronological 70/15/15 splits; `theoretical_power` among the BiLSTM
inputs, C-18) and give an Optuna search space for XGBoost. The Keras BiLSTM settings (128 and 64
units, four attention heads, 168 h look-back) are identical to the surviving
`07_LSTM_Attention v2.ipynb`.
