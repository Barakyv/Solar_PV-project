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

## 3. Gate G-1: the manual audit (2026-11-12, W6)

Written against the code at commit `09533a3`, row by row, for gate G-1 (`PROJECT_MASTER.md` §7;
M35, X57 of `docs/stage9_plan.md` §20). T0 is 00:00 local of the first forecast day; "s" is the
source hour (left label), which must end by T0 (s + 1 h ≤ T0). Every number comes from
`outputs/g1_audit/audit/` (`python scripts/g1_audit.py`).

### 3.1 Inputs of the models, by feature family

| # | Family | Latest information used, against T0 | Enforced by | Proven by |
|---|---|---|---|---|
| 1 | Measured lags (`*_lag_{1,2,3,7,14}d`, 7 series) | the same UTC hour d days before the target hour; kept lags 1–14 d at D+1, 2–14 d at D+2, 3–14 d at D+3; a value whose source hour ends after T0 (the 25-hour day) is NaN | `features.build.past_value`, `kept_lags` | T-09, T-18 (perturbation after T0, both DST changes, every horizon), T-10 |
| 2 | Rolling same-hour means and stds (12) | days h … h + N − 1 before the target day (D-04, fix of F-07) | `features.build.rolling_features` | T-09 |
| 3 | Imputation of 1–2 | the median of the same feature column at the same UTC hour over the 30 previous days; each of those values was known at its own, earlier T0; no bfill or ffill (F-24) | `features.build.impute_same_hour` | T-34 |
| 4 | D-22 masks of the power series | the plant flags of the source hour itself (minutes of s) | `features.build.masked_sources` | T-35 |
| 5 | Clear sky and solar geometry (9 + aoi, azimuth, zenith, air mass) | deterministic functions of the time and the site | `features.geometry` | T-36 |
| 6 | Calendar (8 time encodings, day of year, day length, time to solar noon, daylight fraction, day of week) | deterministic | `features.build.calendar_features` | T-36 |
| 7 | GFS (22 `fcst_gfs_*`) | the 12Z run of the previous UTC day, every file of it uploaded before T0 (two late runs replaced by 06Z); one run for the three horizons, its lead per horizon. On the 1,127 issuances of train and validation the last file arrived a median 5.1 h and at least 1.1 h before T0 (53 issuances under 5 h, X65); the row-by-row check, not a margin, is what holds | `nwp.download.late_files`, `nwp.gfs` | T-17 (every row, every horizon), T-43, T-44 |
| 8 | Everything measured at the target hour, the targets, reanalysis `fcst_*`, flags and quality columns | blocked by name and pattern | `features.leakage.EXCL_EXACT`, `EXCL_PATTERNS`, `assert_no_leakage` | T-01 |
| 9 | Encoder of the sequence models (BiLSTM, three TFT) | the measured series of [T0 − L, T0) (L = 72, 96 or 168 h), after the D-22 masks, gaps filled by row 3, with their fill flags | `models.sequence_data.build`, `_check_issuance` | T-02, T-66–T-71, T-77 |
| 10 | Decoder of the sequence models | the calendar, clear-sky and geometry inputs of the valid hour, and the GFS columns of the table of the step's own horizon (the same run as row 7); the TFT never reads an unknown input on a decoder step | `models.sequence_data.known_inputs`, `models.tft_model` | T-02, T-78 |
| 11 | PatchTST | the target series of [T0 − 168 h, T0) only; RevIN per window | `models.sequence_data.build` (for PatchTST), `models.patchtst_model` | T-77, T-78 |
| 12 | Scalers | z-score per channel fitted on the train hours (not the warm-up); no library scaler | `models.sequence_data.Scaler` (`fit`) | T-69, T-78 |
| 13 | L-09 correction (indirect XGB) | a, b fitted by OLS on out-of-fold train forecasts (five expanding date folds, 3-day gap), with the number of rounds and the parameters that validation chose (D-10: validation tunes and stops; the coefficients never depend on the test split); applied with the clear-sky power of the target hour (deterministic) | `models.xgb_sota.fit_mos`, `apply_mos` | T-61 (for fixed rounds the coefficients do not change when the validation labels are permuted) |
| 14 | References | persistence: the same hour h days back, DST-safe, filled from earlier days; climatology: train mean by month and hour; clear sky × MEAN_DAYTIME_K (train). The CLIPER reference takes σ and γ from the evaluated split (`docs/deviations.md`, 2026-10-25): a reference statistic, never a model input, and in-sample, so the CLIPER skill is conservative | `models.baselines`, `evaluation.evaluate` | T-38, T-40 |
| 15 | Feature selection | train rows only, one registered validation look (R7-05) | `selection.common.train_table` | T-47, T-50, T-82 |
| 16 | Tuning and early stopping | validation (D-10): validation scores are optimistic by construction; only the test split is held out | `models.xgb_sota`, `models.neural_job` | T-52, T-85 |
| 17 | The k history (D-24, also L-11a) | not adopted (its rule failed by 0.002 points at D+1) | `selection.run` (`configs/k_history_decision.yaml`) | T-49 |
| 18 | `hour_source`, `op_limited_flag` of the target hour | evaluation views only, never inputs | `features.leakage` patterns | T-01 |

No external source was added after stage 6b, so the literature rule of the gate (a new source
passes T-17 and a perturbation check; L-11a only from day D−1) has nothing new to check.

### 3.2 The processing steps before the hourly table (X57)

Rows 1–4 start from the stage-5 hourly table. The steps that build that table were audited here
for the first time, by asking what a step does with data after T0 when it prepares a source hour
before T0.

| # | Step | Look-ahead | Measured on the real data | Reading |
|---|---|---|---|---|
| 19 | Ingest (stage 2) | none: one row at a time, a time-zone conversion | — | — |
| 20 | Clock correction (stage 4, D-19) | the offset of a day is a centred 31-day median (61 days for the inverter delta) of daily estimates, within segments whose resync dates were found on the whole series; the minutes of day d move by an offset estimated from days d − 15 … d + 15 | with a trailing median (days before only) the whole-minute shift would differ on 552 of 1,369 days (40.3 %) for the meteo clock and on 541 (39.5 %) for the inverter clock, by at most 3 minutes (84 and 87 days by 2 minutes or more; 55 warm-up days after a resync) | the estimates measure the logger's clock (the timing of the irradiance curve on clear days), not the weather or the plant: a retrospective correction of a measurement error, as in any quality-controlled archive. Declared (`docs/deviations.md`) and accepted by the user (D-35) |
| 21 | Polled reconstruction (stage 4, F-18) | minutes between polls are interpolated, so those after the last poll before T0 lean towards the first poll after it (at most 15 minutes later) | only on the 94 issuance days of the polled source (from 2025-02-26, all in the test period); bound on the mean of the last source hour (a night hour): 0 kW for power, 0.23 W/m² for irradiance, at most 0.68 °C (median 0.17 °C) for the temperatures | negligible, and only in the test period; declared |
| 22 | Cleaning (stage 5) | two-sided rules: R3 removes a frozen run when it reaches daytime, which may be after T0; margins of 1 or 15 minutes around R1 and R3 hits; F3 uses a centred 31-minute window and 30-minute state runs; F2 judges runs of non-integer set points | every issuance day (1,364): the stage-4 minutes of [T0 − 72 h, T0) cleaned alone against cleaned with the next day; of the 65,472 source cells per series (48 hours before T0), 1 changed for power, 18 for irradiance and 15 for each temperature, on 4 issuance days (2021-09-10, 2022-10-19, 2023-01-13, 2023-12-15; none in validation or test). All are communication freezes from the late afternoon into the next day, which R3 removes only once the run reaches the next morning; 65 of the 66 changed cells are a stale value against a missing one, and the last is an irradiance mean of 0.32 against 0.41 W/m² at 20:00 local in January (the night offset of the sensor). None is a scored daytime hour, and a lag reads the same clock hour, so they reach only night and twilight rows. Method check: the cleaning with the next day equals the stage-5 table on every compared cell | negligible; declared |

### 3.3 The test split

`python scripts/g1_audit.py --only test_access` (2026-11-12, rerun 2026-11-13 after the runs of
that day, X76): the access log `outputs/07_evaluation/test_access_log.csv` does not exist and has
never been in git; 159 step records of 21 runs hold no test access; 149 forecast files
(predictions, uncorrected and ablation forecast files) hold no hour at or after 2024-10-01, the
last one being 2024-09-30 23:00 UTC. Every data-based decision used train or validation
(`PROJECT_MASTER.md` §3; stage plans). The raw data of the test period were read for data
preparation only, never with a model output: the ingest QA (F-14), the clock offsets of the test
months (D-19), the detection and reconstruction of the polled source, which exists only in the
test period (F-18, D-20, D-21: the method was chosen by a simulation on train, N5), the cleaning
(thresholds from physics and train, `configs/cleaning_policy.yaml`) and the descriptive figures
of stage 3 over the whole period. The fixture of T-15 scores the submitted LSTM file with its own
truth, which measures the evaluator, not the rebuild.

## History

- 2025-10: Phase-1/2 leakage (`theoretical_power`, target duplicates) identified.
- 2026-04-08: NWP archive leakage identified.
- 2026-10-13: rebuild audit — time-zone bug (F-01), rolling leakage at D+2/D+3 (F-07).
- 2026-10-15: stage-5 quality columns blocked as model inputs: `n_*`, `src_*`, `rule_*`, `frac_*`, `*_flag`, `hour_source` (T-01; the original used quality flags such as `has_statistical_outlier` as features).
