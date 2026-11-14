# Data-processing record

Every operation performed on the data in the rebuild, in order, with its reason, method,
parameters, result, verification and evidence. Written to be lifted into the final report
(data chapter, "Reproduction and correction note", appendix). All numbers come from
`docs/data_record_numbers.md`, which is generated from the stage manifests (sections N1–N15).

IDs: OP-xx = operation on the data (this file); F-xx finding, D-xx decision, A-xx assumption,
T-xx test, all defined in `PROJECT_MASTER.md`. Corrections relative to the submitted report are
listed in `docs/errata.md`; accepted deviations from a plan in `docs/deviations.md`.

## 0. How this record is kept

- An operation that changes, removes, flags, aggregates, or measures data gets an OP entry
  **in the same commit** as the code that performs it.
- After a stage is (re)run, `python scripts/build_data_record.py` regenerates the numbers file;
  `tests/contracts/test_data_record.py` fails if it is stale or if a stage, a cleaning rule or
  an OP is missing from this record.
- Raw files are never modified. Every processed file is reproducible from the raw files and
  the code at the tag given for its stage.

## 1. Overview

| Stage | Input | What happens | Output | Tag |
|---|---|---|---|---|
| 0 | `Raw data/` (50 files) | inventory, checksums (OP-01) | `00_data_audit/raw_checksums.sha256` | — |
| 2 | 45 monthly CSVs | checksum check, tag mapping, local time → UTC, QA scan (OP-02–OP-05) | `data/processed/01_ingest/` | `stage-02-passed` ¹ |
| — | ingest output | site and panel geometry (OP-06, OP-07) | `configs/site.private.yaml` | — |
| — | ingest output | clock and data-source audits (OP-08–OP-13) | `configs/clock_offsets.csv`, `00_data_audit/` | — |
| 4 | ingest output, clock offsets | source handling, clock correction, hourly means (OP-14–OP-17) | `data/processed/04_preprocess/` | `stage-04-passed` |
| 5 | stage-4 minutes, cleaning policy | value rules, flags, hourly re-aggregation (OP-18–OP-28) | `data/processed/05_clean/`, `outputs/05_cleaning/` | `stage-05-passed` ² |
| 3 | stage-5 output, raw files, audits | figures and tables of the data chapter; findings fed back into stage 5 (OP-29) | `outputs/03_eda/`, `notebooks/02_eda.ipynb` | `stage-03-passed` |
| 8A | old LSTM predictions | evaluation-code check (OP-30) | `outputs/07_evaluation_fixture/` | `stage-08a-passed` |
| 3b | stages 2, 4, 5 | data quality before and after cleaning, evidence for every threshold; findings fed back into stage 5 (OP-31) | `outputs/03b_quality/`, `notebooks/02b_data_quality.ipynb` | `stage-03b-passed` |
| 6 | stage-5 hourly data and minute flags | panel orientation, solar geometry, targets and D-11, D-22 masks, lags and rolling statistics, imputation, calendar (OP-32–OP-39) | `data/processed/06_features/`, `outputs/06_features/`, `configs/target.yaml`, `configs/derived_constants.yaml`, `notebooks/04b_features_target.ipynb` | `stage-06-passed` |
| 8B | stage-6 tables | truth, reference forecasts, evaluation rules and views, on validation only (OP-40, OP-41) | `outputs/07_evaluation/` | `stage-08b-passed` |
| 6b | NOAA GFS archive (AWS), stage-6 tables | download of the runs public before issuance, hourly reconstruction, candidate features per horizon, sanity and value check (OP-42–OP-44) | `data/processed/06b_gfs/`, `outputs/06b_gfs/`, `configs/nwp_decision.yaml` | `stage-06b-passed` |
| 7 | stage-6 and stage-6b tables, train rows | k-history check, selection per target × horizon, inside-train check, one validation look, the input of stage 9 (OP-46–OP-48) | `outputs/07_feature_selection/`, `configs/k_history_decision.yaml` | — |
| 9 (W1) | stage-6 table without the test rows, stage manifests | runner: input check, evaluation window, references as prediction files, evaluation per run (OP-49) | `outputs/runs/<run_id>/` | — |

¹ The stage-2 commit missed `src/solarpv/data/` because of an unanchored `.gitignore` pattern;
the ingestion code entered git with the stage-4 commit (`0b856c9`). The stage-2 output is
unchanged (same sha256, N1).
² Stage 5 was revised during stage 3 (R3 margin, new rule R7, OP-20 and OP-24) and again during
stage 3b (R7 per hour, F2 drift, new flag F3: OP-24, OP-26, OP-27); the current cleaning is the
one at `stage-03b-passed`.

## 2. Raw data (stage 0)

**OP-01 — Inventory and checksums.**
- *What:* SHA-256 of the 50 files in `Raw data/` (45 monthly SCADA exports `2109.CSV` …
  `2505.CSV`, plus `Ktora.csv`, `example to data.xlsx`, two tag files and a tag-path table).
- *Why:* the original source code was lost; the raw files are the only ground truth, so they
  must be provably unchanged.
- *Format found:* one row per minute, all values quoted, first column `t_stamp`
  (`YYYY-MM-DD HH:MM:SS.000`) in **Israel local time with daylight saving**, 27 tag columns.
- *Evidence:* `00_data_audit/raw_checksums.sha256`, `raw_data_inventory.csv`,
  `scripts/make_checksums.py` (`--verify`).
- *Side file:* `Ktora.csv` is a 2024 export at 15-min resolution, in local time, of a different
  inverter of the same park (F-16). Not used.

## 3. Ingestion and time base (stage 2, `solarpv.data.ingest`)

**OP-02 — Checksum verification on every run.** The 45 monthly files are hashed and compared
with the stage-0 list before reading; a mismatch stops the run (N2).

**OP-03 — Tag mapping.** The 27 raw tag paths are mapped to canonical names (PROJECT_MASTER.md
§6.1; `data/processed/01_ingest/tag_map.csv`). Values parsed as numbers; no non-numeric values
were found (N2).

**OP-04 — Local time → true UTC (fixes F-01).**
- *Method:* `tz_localize("Asia/Jerusalem", ambiguous="infer", nonexistent="raise")`, then
  `tz_convert("UTC")`. The repeated October hour is two consecutive blocks with different
  values, decoded by file order; the March hour is absent in the raw files.
- *Result:* one continuous 1-minute UTC grid, 1,971,360 minutes, 2021-08-31 21:00 to
  2025-05-31 20:59 UTC; no duplicates, no gaps (N2).
- *Original:* notebook 03 attached "UTC" to the local timestamps without converting, so
  production was shifted 2 h (winter) or 3 h (summer) against solar geometry (F-01). Its
  consequences: real afternoon production zeroed by a night rule on the wrong geometry (F-01b),
  and the reported "13.5 % inverter downtime" was pre-dawn hours (F-01c).
- *Verification:* T-05 (DST transitions), T-08 (production centred on solar noon in every
  month, ±30 min on the raw stamps).

**OP-05 — QA scan, no values changed.** Per column: missing, non-numeric, exact 32,766
sentinel, range, negatives, inverter AC above 1,897 kW (N2, `ingest_qa.csv`). Found (F-14):
fault code −270 in the three temperature sensors, small negative irradiance at night,
4 power spikes up to 40,825 kW, impossible wind speeds up to 629 m/s, saturated soiling
sensors. One spike is 32,766.26 kW (a sentinel that the exact-value check missed; corrected in
stage 5). These are handled in stage 5, not here.

## 4. Site and panel geometry

**OP-06 — Site location (F-12, D-05).** The two inverters are 2 of the 25 inverters of the
40 MW Ketura Solar park; solar geometry uses the park centre (29.96523 N, 35.07446 E, 101 m),
kept in the git-ignored `configs/site.private.yaml`. The old point lay inside the kibbutz; the
evaluation mask is identical at both points (F-02: 2,580 test daytime hours).

**OP-07 — Panel orientation measured from the data (F-13).**
- *Method:* clear days chosen from the DC curve alone; grid search of tilt 0–45° and azimuth
  130–230° fitting the in-plane sensors and the DC power to modelled clear-sky POA (Ineichen,
  Hay–Davies), sun above 15° (`00_data_audit/orientation_2026-10-14/`).
- *Result:* tilt 20° (range 20–23°), azimuth 180° (N3). The original assumed 30°. Both
  irradiance sensors lie in the panel plane and agree (ratio 1.000); there is no horizontal
  GHI sensor (D-09).
- *Limitation:* measured before the clock correction; after it, the DC centroid sits 1–2.7 min
  after solar noon, which suggests the array faces 2–3° west of the sensor. The array
  orientation is re-measured on corrected data in stage 6 (A-04).

## 5. Logger clocks and data sources (audits feeding stage 4)

**OP-08 — Drift detection (F-15).** Power-weighted centre of the clear-day curves minus solar
noon, per month, for DC and for the irradiance sensor (`clock_drift_2026-10-14/
centroid_offsets.csv`). It showed slow drift and jumps, different for the two loggers.

**OP-09 — Clock-offset estimation (D-19, `scripts/estimate_clock_drift.py`).**
- *Offset convention:* stamped time − true time.
- *Daily estimates, per data source:* `tau_met`, the shift that best aligns the in-plane sensor
  with clear-sky POA at 20°/180° on clear days (relative rmse ≤ 3 %); `delta`, inverter minus
  meteo clock, from the cross-correlation of minute-to-minute changes of DC and irradiance on
  cloudy days (peak correlation ≥ 0.7; precision ~0.1 min; independent of orientation and
  temperature); `tau_dc`, the same fit on DC, used only to compare the two clocks of the
  second source.
- *Steps:* found in delta (dated to the day) and in tau; 2023-05-09 (meteo resynchronised, the
  inverter logger not) and 2024-01-21 (meteo clock set to the drifted inverter clock) (N4).
- *Smoothing:* 31-day rolling median within segments between steps; inverter = meteo + delta
  (61-day window). A joint fit of orientation and shift was rejected: azimuth and a constant
  shift are confounded (8° ≈ 6 min).
- *Absolute frame:* the clear-sky model at 20°/180°. Anchors: −0.56 min at the start of the
  data, −1.98 min in the 30 days after the 2023-05-09 resync (N4). Absolute uncertainty about
  ±2 min, including a seasonal bias of the estimator of about ±1 min.
- *Result:* offsets up to −16 min before correction; residual monthly medians after correction
  ≤ 1.74 min (tau) and ≤ 0.70 min (delta) (N4). Written to `configs/clock_offsets.csv`.

**OP-10 — Detection of a second data source (F-18, `solarpv.data.resolution`).**
- *Observation:* from 2025-02-26 about 87 % of daytime minutes repeat the previous value in
  every tag. Each tag holds an instantaneous snapshot taken every 15 min on its own phase
  (e.g. irradiance at :05/:20/:35, AC at :08/:23, DC at :13/:28); the minute containing a poll
  is a time-weighted blend of the old and new value.
- *Detection per tag and hour:* share of daytime minutes inside constant runs of 10–16 min
  (≥ 0.5 → polled) or change rate ≥ 0.25 per minute (→ minute source). Change rate alone was
  rejected: partial freezes of the old source look polled.
- *Extent:* `first_day` 2025-02-26 (first appearance, connected to the sustained period with
  gaps ≤ 2 days), `era_start` 2025-03-03 (7 consecutive days with polled hours). The old
  minute source still appears for some hours afterwards (e.g. 2025-03-26, 2025-04-29),
  on its old clocks. About 35 % of the test period is polled (1,001 polled and 70 mixed
  daytime test hours, N7).

**OP-11 — Snapshot or average? (A-18).** On cloudy days the change between successive polled
values (p90: 253 W/m², 333 kW) is far larger than between 15-min means (96 W/m², 128 kW) and
similar to 1-minute values 15 min apart (175 W/m², 231 kW): the polled values are
instantaneous snapshots.

**OP-12 — Clock of the second source (A-19).** tau_dc − tau_met in the polled period, minus the
same DC bias measured on the minute source in the same months (+1.40 min), is +0.36 min: one
clock for all tags (N4). The old minute source after the switch keeps drifting
(−17.4 min on 2025-04-29, measured on a full minute-source day).

**OP-13 — Choice of reconstruction by simulation on train (D-21).** The 2025 polling was
applied to real minute data of March–May 2022 and 2023 (random phase per tag and day, hold)
and the hourly mean of site AC rebuilt four ways (N5): hold as in the raw file 2.58 % of
capacity RMSE; **linear interpolation between poll instants 1.62 %** (clear days 0.85 %,
variable days 2.91 %); clear-sky-index interpolation 1.62 %; naive fusion of six tags 3.12 %.
Linear interpolation was kept. Against a day-ahead error of about 12 % of capacity, the
remaining measurement noise adds about 0.1 percentage point (√(12² + 1.6²) ≈ 12.11).

## 6. Hourly preprocessing (stage 4, `solarpv.data.preprocess`)

**OP-14 — Source assignment and reconstruction.** Every tag-minute is assigned to its source
(OP-10). Continuous tags of the polled source are rebuilt: each snapshot is placed at its poll
instant (sub-minute, from the blend minute) and the minutes between are linearly interpolated;
long runs get a point every 15 min. Step-like tags (states, mode, setpoints) are left as they
are. Verification: T-19 (synthetic: the 7.5-min hold lag disappears; on the data, tau moves
from +4.3 to −3.1 min).

**OP-15 — Clock correction (D-19, D-20).** Each tag-minute is re-stamped at true time with the
offset of its source and clock group (`configs/clock_offsets.csv`), rounded to whole minutes and
constant over a local day, so daytime minutes are neither lost nor duplicated except where a
tag changes source. Minutes lost: 188–291 per column, at night or at source switches (N6).
Verification: T-20; T-08 tightened — irradiance centroid within ±2 min of solar noon in all
45 months, inverter-vs-meteo lag (delta) within ±1 min, DC centroid within ±3 min (the DC
centroid carries the array's physical asymmetry, `docs/deviations.md`).

**OP-16 — Site total.** `site_total_AC_kW = s01_inv1_AC_kW + s01_inv2_AC_kW` minute by minute;
NaN if either inverter is missing.

**OP-17 — Hourly means.** Left-labelled (row t covers [t, t+1h)), valid with at least 45 of
60 minutes, otherwise NaN (A-09). Result: 32,856 hours; no hour below coverage before cleaning
(N6). Verification: T-21. Outputs are byte-identical when rerun from the committed code.

## 7. Cleaning (stage 5, `solarpv.data.clean`, policy `configs/cleaning_policy.yaml`)

The policy file holds every threshold; the evidence for each is
`00_data_audit/cleaning_2026-10-15/explore_cleaning.py` and, in full, the stage-3b data-quality
analysis (OP-31, `notebooks/02b_data_quality.ipynb`, Q05–Q17); thresholds come from physics and
the train period. The test plan (`docs/stage5_test_plan.md`) was written before the code.

*Original pipeline:* notebook 03 only added flags on 15-min data; the cleaning that fed
notebook 04 was done in a step that did not survive and applied the night rule on the wrong
geometry (F-01b). Spikes stayed in the target (notebook 04 shows a 1-minute maximum of
12,444 kW), the −270 code was only flagged, curtailment was not handled, quality flags were
used as model features, and gaps were filled with backward fill (future information).

Value rules — a wrong measurement is replaced; the rule is stored per minute in
`rule_<column>`; precedence R1, R3, R6, R7, R5, R2, R4 (counts per column in N7):

| ID | Rule | Condition (policy file) | Action |
|---|---|---|---|
| OP-18 | R1 range | outside the physical range (temperatures −20…60/90 °C, irradiance −20…1,600 W/m², power −20…1,897 kW per inverter, wind 0…60 m/s), plus neighbours: ±1 min on the minute source (blended 1-min averages), ±15 min on the polled source (interpolated towards the bad value) | NaN |
| OP-19 | R2 offset | irradiance between −20 and 0 W/m² (sensor offset at night) | 0 |
| OP-20 | R3 frozen | a run of one repeated value ≥ 30 min (ambient temperatures ≥ 60 min, 0.1 °C steps) that reaches daytime (sun > 10°); power runs of exactly 0 excluded (inverter stops); plus neighbours ±1 min (minute source) / ±15 min (polled), because when data resume the 1-min average blends the stale value with the real one — margin added after stage 3 found a 247 kW blend minute at sunrise on 2021-09-12 | NaN |
| OP-21 | R4 night zero | sun below 0° and 0 < \|power\| ≤ 5 kW. Stage 3b showed there is no night noise (below −6° the inverters read exactly 0); 94 % of the readings R4 changes are tiny twilight production between −3° and 0° (238 kWh in 3¾ years). Kept as a definition — the target is exactly 0 when the sun is down, where k is undefined and nothing is evaluated | 0 |
| OP-22 | R5 night stale | sun below −6° and power > 5 kW (frozen values during communication outages) | NaN |
| OP-23 | R6 dead DC | DC = 0 while the same inverter delivers more than 100 kW AC (inverter 1, March 2023) | NaN |
| OP-24 | R7 scale fault | per local day **and per UTC hour**, median pyranometer / cleaned irradiance (sun > 30°, irradiance > 300 W/m²; at least 20 usable minutes in the period) outside 0.9–1.1 → the whole period. Added after stage 3 found the pyranometer at exactly ×0.75 from 2023-05-28 to 2023-12-31 and on days in Dec 2024 – Apr 2025, while the irradiance sensor stays consistent with clear-sky POA and DC (F-19). The hourly check and the minimum minutes were added in stage 3b: the fault also switches within a day (2021-11-05), and a freeze day had been judged on 2 minutes (F-21) | NaN |

Flag rules — real plant events, values never changed (D-22):

| ID | Flag | Condition | Use (D-22) |
|---|---|---|---|
| OP-25 | F1 `downtime_inv1/2_flag` | sun > 5°, in-plane irradiance > 150 W/m² (if missing: clear-sky POA > 300), inverter AC < 20 kW | out of the training loss; NaN in past-day features; included in the main evaluation, excluded in the secondary one |
| OP-26 | F2 `curtail_flag` | sun > 5° and park setpoint `iec_sp_w` < 100 %, except runs of non-integer setpoint values longer than 10 minutes. A commanded step (0/25/50/75 %) cuts output to about the setpoint plus ten points and shows at most 3 blended minutes; on 2022-02-09…12 the setpoint drifted through non-integer values for 4,309 minutes while output stayed normal (stage 3b, F-20). `iec_mode` is not used: output under modes 0/1 matches mode 2 (F-17) | same |
| OP-27 | F3 `derate_inv1/2_flag` | the inverter produces (AC ≥ 20 kW, sun > 5°) with status code 5 or 6 for at least 30 minutes, or its AC stays at a stable fraction below 0.85 of the other inverter's (centred 31-minute window: median of log(AC/AC_other) < log 0.85 and inter-quartile range < 0.05; sun > 15°, both inverters > 100 kW). Partial outages: one inverter at ~75 % (once ~50 %) of its twin for hours or days, unseen by F1 (stage 3b, F-22) | same |

**OP-28 — Hourly re-aggregation.** Same hourly rule as OP-17 on the cleaned minutes, plus per
hour: share of minutes with each flag (`frac_*`), `op_limited_flag` (≥ 5 % of the hour flagged by
F1, F2 or F3),
`hour_source` (minute / polled / mixed, D-21) and valid-minute counts (`n_*`). Quality columns
can never be model inputs (T-01).

*Result (N7):* hourly maximum of site AC 2,948.4 kW (3,271.9 before; spikes removed);
op-limited daytime hours 379 (train), 146 (val), 186 (test), of which 245 / 43 / 93 with a
derated inverter (F3) — before stage 3b they were 193 / 106 / 106; daytime hours with a missing
target 42 / 1 / 5, all from communication freezes. The pyranometer loses 409,160 minutes to R7
(it is not a model input). Real twilight production (up to ~29 kW with the sun 0–3° below the
horizon) is kept.

*Verification:*
- runtime self-check: no value changes without a rule code;
- T-22: 25 synthetic tests of every rule (including the R3 margin, R7 per day and per hour),
  flag (including the F2 drift and both signs of F3), precedence, idempotence and the hourly
  flags;
- T-23: on the real output, for every cleaned column, a value changed if and only if a rule
  code is set, the new value is the one the rule prescribes, the rule's condition holds on the
  stage-4 value, and no minute meeting a condition was missed — re-derived with independent
  code;
- T-16 (no daytime production > 5 kW ever becomes 0), T-24 (ranges), T-25 (flags never change
  values; flags match their definition), T-26 (hourly output), T-27 (daytime energy unchanged on
  fault-free days, ±0.1 %), T-28 (log and manifest equal the output; policy and file hashes);
- mutation check: 12 faults planted in a copy of the output, all 12 caught
  (`00_data_audit/cleaning_2026-10-15/mutation_check_output.txt`; the last three — an unflagged
  derate, a flagged setpoint drift, a kept pyranometer fault — added in stage 3b).

## 8. Exploratory analysis (stage 3, `solarpv.eda`)

**OP-29 — Figures and tables of the data chapter.** No data are changed. Plan:
`docs/stage3_eda_plan.md`; outputs: `outputs/03_eda/` (10 figures, 5 tables, manifest with input
hashes); narrative: `notebooks/02_eda.ipynb`. Statistics that could inform a modelling choice are
computed on train only. Results (N9):
- exact split sizes (train 21,195 h, val 5,832 h, test 5,829 h; the submitted report gave
  ~21,049 h for train); 2,580 test daytime hours, 186 of them op-limited, 1,001 polled;
- the time-zone bug measured on raw files: clear-day production centre +120.4 min (winter) and
  +173.6 min (summer) after solar noon under the original reading, +1.9 and +2.6 min after the
  rebuild;
- temperature coefficient of AC per unit irradiance −0.42 %/K (train, 700–900 W/m²);
- daily clear-sky index by month and its autocorrelation (0.33 at one day, ~0 from two days);
- the in-plane sensor reads 10–20 % above the Ineichen clear-sky POA at the clearest hours
  (scale question for D-11, stage 6).

*Fed back into stage 5:* the R3 margin (OP-20) and rule R7 (OP-24, F-19). *Noted as a
limitation:* minute-source hours after the 2025 switch use the last estimated legacy offset
(§18). Rerun after the stage-3b changes to cleaning (fig09 now also shows F3). *Verification:* T-30 (every planned output exists, tables agree with the stage-5
manifest and with physics, outputs made from the current inputs).

## 9. Evaluation-code check (stage 8A)

**OP-30 — Reproduction of the old LSTM metrics.** The surviving old LSTM predictions
(`submission_evidence/08_lstm_alignment/`) are evaluated with the new evaluation code: the
daytime mask (sun > 5° at mid-hour) reproduces n = 2,510 / 2,512 / 2,514, and nRMSE matches the
reported values within 1e-4 pp (indirect) and 0.0012 pp (direct) (N8, T-15). This proves that
differences in the corrected results come from data and models, not from a changed metric.
The old predictions are used only to test the code, never as a target.

## 10. Data quality before and after cleaning (stage 3b, `solarpv.eda.quality`)

**OP-31 — Evidence for every threshold, and the effect of cleaning.** The analysis itself
changes no data. Requested by the user on 2026-10-16. Plan written before the code:
`docs/stage3b_quality_eda_plan.md`; outputs: `outputs/03b_quality/` (31 analyses, each with a
table, a figure and the facts its note relies on, in `quality_manifest.json` with input hashes);
narrative: `notebooks/02b_data_quality.ipynb`, a markdown note under every code cell, numbers
formatted from the manifest. Thresholds are judged on train only; analyses over all periods are
descriptive. Parts: the export as delivered (Q01–Q04), the evidence for each cleaning threshold
(Q05–Q17), what cleaning changed (Q18–Q22), the cleaned data for modelling (Q23–Q31). Results
(N10):
- the raw export has no NaN at all: outages appear as repeated values (Q01), and 96 % of frozen
  daytime minute-columns are freezes of the whole logger (Q09) — the basis of R3;
- every R1 limit sits in an empty region (Q05); the R1/R3 margins are justified by blended
  minutes next to faults (Q06, Q10); the R3 duration lies on a plateau of the run-length curve
  (Q08); R5 removes only frozen values (Q11); R6 is one dead sensor (Q12); the R7 band separates
  two populations at 1.00 and 0.75 (Q13); the F1 threshold sits in an empty valley (Q14); the
  hourly thresholds are insensitive (Q17);
- cleaning changes the tails, not the body of the data (Q19); energy changes only on 20 freeze or
  spike days (Q20); nothing frozen is left (Q22).

*Fed back into stage 5 (rerun; all tests and the mutation check green):* F-20 (F2 ignores a
drifting non-integer setpoint, OP-26), F-21 (R7 per hour and with minimum minutes, OP-24),
F-22 (new flag F3 for derated inverters, OP-27); R4's justification corrected (OP-21).
*For later stages:* the clear-sky index is unstable at low sun (D-11); winter-morning row
shading; the observed peak is 85 % of the 3,449 kW capacity (A-08); keep `ambient_temp` (D-09);
past-day lags add little beyond the clear-sky features; no weekday effect; a decline of about
1 % per year with a stable irradiance sensor; test has no summer. *Verification:* T-31 (every
planned output exists and was made from the current inputs, the notebook was built from the
current manifest with a note under every code cell, and the claims of the notes hold).

## 11. Target and features (stage 6, `solarpv.features`)

**Records and conventions.**
- Plan written before the code: `docs/stage6_plan.md`.
- Every run writes a log (`outputs/06_features/runs/`) and is described in `docs/run_journal.md`
  (R6-01 to R6-08).
- Outputs:
  - `data/processed/06_features/features_1H_D{1,2,3}.parquet`, one table per horizon;
  - `outputs/06_features/`: feature dictionary, manifest, D-11 and L-04 evidence;
  - `configs/target.yaml` and `configs/derived_constants.yaml`.
- Narrative: `notebooks/04b_features_target.ipynb`, a note after every code cell.
- Splits: decisions are made on train; D-11's third criterion and L-04 on validation, as
  specified; the test split is used for nothing.
- Numbers: N11.

**OP-32 — Panel orientation re-measured on the corrected data (A-04, F-23).**
- *Why:* OP-07 measured 20°/180° before the clock correction. Afterwards, the DC production
  centre sat 1–2.7 min after solar noon. The plane enters the theoretical power (the denominator
  of k) and the POA features.
- *Data:*
  - 174 clear train days, 6 per month over 29 months: a smooth DC curve, the top 40 % of
    energy, no op-limited minute, DC complete;
  - 10-min means with the sun above 15°;
  - DC corrected for module temperature (−0.421 %/K, fitted on the same days).
- *Method:* grid search of tilt and azimuth against Ineichen clear-sky POA (Hay–Davies), in four
  ways:
  - M1: one scale for all days;
  - M2: a scale per day (the daily shape only);
  - M3: the azimuth at which the DC centre coincides with the centre of the POA_cs curve;
  - M0: the irradiance sensor, fitted like M2.
- *Result (N11):*

  | Method | Tilt | Azimuth |
  |---|---|---|
  | M1 | 21° | 185° |
  | M2 | 15.5° | 186.5° |
  | M3 | — | 184.7° |
  | M0 (sensor) | 16.5° | 180.5° |

  - The DC methods agree on about 185°, and the sensor faces 180°. It is the panels, not the
    clocks, that face west of south.
  - The difference exceeds the 2° rule of the plan, so the **panel plane is 21°/185°**
    (`configs/panel_orientation.yaml`).
  - The clock correction (OP-09) stays framed on the sensor plane 20°/180°, which M0 confirms.
- *Evidence:* `00_data_audit/orientation_2026-10-20/` (script, results, figure); log
  `20261020T094900Z`; R6-01.
- *Original:* 30°/180° assumed.
- *Open:* external confirmation of the row orientation (Q-11).

**OP-33 — Solar geometry and clear-sky values per hour.**
- *Method:* for every minute of the hour [t, t+1h):
  - pvlib solar position (apparent zenith);
  - Ineichen clear sky with pvlib's Linke-turbidity climatology;
  - Hay–Davies transposition to the panel plane (albedo 0.2);
  - angle of incidence;
  - relative air mass (Kasten–Young; NaN with the sun below the horizon).

  The hourly value is the mean of the 60 minutes (a circular mean for the azimuth),
  left-labelled like the data. Sunrise, sunset and transit per local day come from pvlib (SPA).
- *Why:* the data are hourly means, so the reference must be the hourly mean of the same
  quantity, not a value at one instant.
- *Deviations, both in `docs/deviations.md`:*
  - The mean covers the full hour, not only the minutes in which AC is valid (§6.4). The
    forecast converts k to kW with the theoretical power of the full hour, because data gaps are
    not known at T0, so the training target uses the same value. It matters in 26 daytime hours
    whose AC mean rests on 46–59 minutes (7 in train, 19 in test; N11).
  - Hay–Davies instead of isotropic (§6.3): it is the model used for clear-sky POA since
    stage 2, and it was fixed in the plan before the D-11 criteria were computed.
- *Verification:* T-36 (the hourly value equals the mean of the minute values).

**OP-34 — Targets: theoretical power, k, and the exact inversion (fixes F-06).**
- *Method:*
  - `theo_kw = CS × 3,449 / 1,000`, where CS is the hourly mean clear-sky irradiance. Both
    GHI_cs and POA_cs are kept as columns until D-11.
  - `k = AC / theo` where theo > 50 kW and AC is known (`k_valid`); 0 where theo ≤ 50 kW
    (night, deep twilight); NaN where AC is missing.
  - k is not clipped. `k_suspect = k > 1.5` is a flag only.
- *Result (N11):*
  - With POA, 5.4 % of valid daytime train hours are above 1, 0.61 % above 1.2 and 0.04 %
    above 1.5. With GHI: 32.4 / 12.5 / 0.99 %.
  - Every daytime hour (F-02 mask) has theo above 50 kW (smallest 62 kW). All daytime hours
    with AC are therefore valid, as §6.5 assumes.
- *The 23 k_suspect hours, examined (N11):*
  - 9 in train, 7 in val, 7 in test.
  - All are at low sun (hourly mean elevation 1.4–9.7°, theo below 200 kW) and none is
    op-limited.
  - Each is a ratio of two small numbers, sensitive to the clear-sky diffuse at low sun and to
    the ±2 min clock uncertainty. They are kept and flagged.
- *Verification:*
  - T-11: k × theo equals AC within 2.3e-13 kW on 15,809 valid hours.
  - T-13: no value sits at exactly 1.2, values above 1.2 are kept, and k_suspect means k > 1.5.
- *Original:* k was a mean of 15-min ratios (not invertible), on local time labelled UTC, against
  GHI_cs, and clipped at 1.2: 16.8 % of daytime test hours sat at exactly 1.2.

**OP-35 — The k denominator (D-11), the training threshold (L-04) and MEAN_DAYTIME_K.**
- *D-11 criteria, fixed in the plan before the run (majority wins):*
  1. the share of k > 1 on train;
  2. the spread of the monthly median k of clear days on train;
  3. the validation nRMSE in kW of a fixed XGBoost trained on k and converted to kW: 400 trees,
     depth 6, learning rate 0.05, seeds 42/123/777.
- *D-11 result (N11):* POA wins all three:

  | Criterion | POA | GHI |
  |---|---|---|
  | 1. share of k > 1 | 0.054 | 0.324 |
  | 2. clear-day spread | 0.108 | 0.413 |
  | 3. validation nRMSE | 7.94 ± 0.09 % | 8.09 ± 0.06 % |

  **D-11 = POA_cs on the panel plane** (`configs/target.yaml`).
- *L-04 (literature review):*
  - Training thresholds on theo of 50, 100 and 150 kW were tried, three seeds each. A higher
    threshold is adopted only if its gain exceeds max(0.05 pp, 2 × the pooled seed std).
  - The best gain was 0.053 pp against the 0.119 pp needed, so the threshold **stays 50 kW**.
  - The first rule, a gain of 0.05 pp on a single fit, chose 150 kW from noise. It was replaced
    before any use (R6-03, R6-04).
- *MEAN_DAYTIME_K:* the mean k over valid daytime train hours, **0.788157**
  (`configs/derived_constants.yaml`). It is the denominator of the indirect nRMSE (§6.9). The
  original's 0.6279 is kept only for the fixture (T-15).

**OP-36 — Operation masks before the past-day features (D-22).**
- *Method:* before any lag or rolling statistic:
  - each inverter's AC becomes NaN in hours where its own stop (F1) or derate (F3), or the
    curtailment (F2), covers at least 5 % of the minutes;
  - the site total becomes NaN in op-limited hours;
  - weather columns are not masked.
- *Why:* a limitation is not weather. Yesterday's curtailed afternoon must not look like a
  cloudy one.
- *Result (N11):* 474 hours masked for inverter 1, 548 for inverter 2, 733 for the site total.
- *Verification:* T-35.
  - Synthetic: each inverter follows its own flags.
  - Real data: no site-total lag equals the raw value of an op-limited hour.

**OP-37 — Lags and rolling statistics, safe for every horizon (fixes F-07; D-04, D-16).**
- *Issuance:* the forecast for local day D+h is issued at T0 = 00:00 local of day D+1.
- *Lags:* the value at the same UTC hour d days back, d ∈ {1, 2, 3, 7, 14}. lag_1d is dropped
  at D+2, and lag_1d and lag_2d at D+3 (LAG_DROP).
- *Rolling:* the mean and standard deviation (ddof 1, at least 2 values) of the same UTC hour
  over days h … h+N−1 back, for N = 3 and 7.
- *DST guard:*
  - A past value from hour s may feed hour t only if s + 1 h ≤ T0(t).
  - On the 25-hour fall-back day, the 24-hour shift of the last hour reaches T0's own hour.
    The guard turns such values into NaN.
  - Over the four autumns: 28 / 56 / 84 values at D+1 / D+2 / D+3 (N11).
- *Verification:*
  - T-09 and T-18, synthetic and real: changing every source value from T0 on changes no
    feature of the forecast, across both DST transitions and at every horizon.
  - Mutation check: the original's rolling window and lags without the guard are both caught
    (`00_data_audit/features_2026-10-20/`).
- *Original:* the rolling windows started 1 day back at every horizon, so at D+2 and D+3 they
  included days not known at T0 (F-07). DST was not handled.

**OP-38 — Imputation of past-day gaps, and the warm-up.**
- *Method:*
  - A missing past-day value (data gap, OP-36 mask, DST guard) is filled with the median of the
    same feature at the same UTC hour over the previous 30 days: earlier days only, at least
    one value. Never backward fill.
  - The first 14 days (336 hours) have too little history. They carry `warmup = True` and are
    excluded from training (§6.6).
- *Result (N11):* at most 3.56 % of a past-day feature is imputed (site-total lag 14d).
- *Why the change:* the original (`04_feature_engineering_v4.ipynb`):
  - filled with the expanding median of the whole column, nights included, which pulls
    daytime gaps towards zero and mixes the seasons;
  - then applied `ffill().bfill()`, which fills the first rows from later values.
- *Verification:*
  - T-34: only the same hour on earlier days is used, and later values change nothing.
  - Mutation check: backward fill and a median from later days are both caught.

**OP-39 — Calendar features and the contract (74 features; Q-05).**
- *Method:*
  - `hour_sin/cos` from the UTC hour (§6.6).
  - month, season, day of year, day_of_week, hour_local and is_weekend from the local day of
    the target hour.
  - `is_weekend` = Friday and Saturday, the Israeli weekend.
  - `time_to_solar_noon` = pvlib transit − middle of the hour.
  - `day_length` = sunset − sunrise.
  - `daylight_fraction` = day_length / the longest day of the year at the site.
- *Contract:*
  - `sun_azimuth` equalled `solar_azimuth` and is removed (Q-05). That leaves 74 features at
    D+1, 67 at D+2 and 60 at D+3.
  - Every feature has its definition, unit and availability at T0 in
    `outputs/06_features/feature_dictionary.csv`.
- *Original:*
  - weekend Saturday–Sunday (`day_of_week >= 5`);
  - `time_to_solar_noon` = local hour − 12, which overwrote an earlier pvlib transit value;
  - `daylight_fraction` from a cosine approximation of the day length.
- *Verification:*
  - T-10: the contract (74 / 67 / 60).
  - T-36: weekend, solar noon and hourly means.
  - T-33: the tables, i.e. rows, UTC axis, feature sets and no leaky column (T-01).

*Verification of the whole stage:*
- The tests above, plus the mutation check: five planted leaks, all caught.
- The outputs are byte-identical across reruns from clean commits (R6-07, R6-08).

*For later stages:*
- Clear-sky geometry still correlates with k (|r| about 0.4).
- Yesterday's power correlates −0.28 with today's k. Proposal for stage 7, to be decided on
  validation: lags of k for the indirect route.
- k persists only weakly from one day to the next (0.34, Q26), so the GFS forecasts
  (stage 6b) are expected to carry the skill.

## 12. Evaluation layer on the corrected tables (stage 8B, `solarpv.evaluation`)

**Records.**
- Plan written before the code: `docs/stage8b_plan.md`. It includes a self-review against the
  original evaluation scripts (E1–E11) and against the stage-8A code (A1–A6).
- Outputs: `outputs/07_evaluation/val/`, with run logs in `outputs/07_evaluation/runs/`.
- Numbers: N12.
- No model is trained in this stage. The run scores the references on **validation only**, and
  the test split is not opened.

**OP-40 — Reference forecasts (`solarpv.models.baselines`).**
- *Why:* the skill score needs a reference (§6.9). Each reference is computed from the truth
  table by one code path, for the split being scored.
- *Method:*
  - **Persistence (h):** the target at the same UTC hour h days earlier, if that hour ended by
    the issuance time. It uses the stage-6 DST guard (OP-37), so on the 25-hour day it never
    reads the hour that starts at T0. Gaps are filled with the same-hour median of the previous
    30 days, earlier days only (OP-38). It is naive: operational limits are not masked.
  - **Climatology:** the train mean per (UTC month, UTC hour), over hours with a known target
    and normal operation (D-22).
  - **Clear sky × MEAN_K:**
    - direct: `clip(theo_kw × 0.788157, 0, 3,449)`;
    - indirect: 0.788157 where theo > 50 kW, and 0 otherwise.
- *Result (N12, validation D+1):*

  | Reference | nRMSE in kW |
  |---|---|
  | Persistence | 9.68 % |
  | Climatology | 8.04 % |
  | Clear sky × MEAN_K | 8.96 % |

  Persistence degrades from D+1 to D+2, as expected.
- *Verification:*
  - **T-38:** perturbing the truth from T0 on changes no persistence forecast of that day,
    around the fall-back day and for every horizon; the climatology ignores val, test and
    op-limited rows.
  - **T-39:** the clear-sky route advantage is exactly 0.
- *Original:* the skill reference was "same time yesterday" at every horizon (C-08).
  Climatology and clear sky were not computed.

**OP-41 — Evaluation on the corrected tables (`solarpv.evaluation.evaluate`, `truth`).**
- *Truth:* `target_k`, `site_total_AC_kW` and `theo_kw` from the stage-6 table. The y_true
  written in a prediction file is used only as an alignment check, with a tolerance of 1e-6 in
  k and 1e-3 kW. It catches a file shifted by one hour (the v6 BiLSTM bug), swapped targets
  and wrong split labels (E8, E9).
- *Coverage:* every model, seed, target and horizon must predict every daytime hour of the
  split that has truth. All models are therefore scored on the same n (E9).
- *Daytime:* F-02, from geometry only. It is asserted equal to the table's flag, and is never
  taken from the truth (E3).
- *Split guard:* validation by default. The test split needs `--allow-test`, and every such
  run is appended to `outputs/07_evaluation/test_access_log.csv` (E10). *(Superseded by D-32
  and R-G1-05: the command line scores validation only; the step `test` of a final run scores
  the test split.)*
- *New columns and tables:*
  - nRMSE in kW for both targets;
  - nRMSE by the mean actual AC of the evaluated rows (L-16);
  - skill against CLIPER (L-13). Its σ and γ come from the observed daytime series of the
    evaluated split: train statistics let persistence beat the reference (stage-8B plan §7);
  - secondary views: normal operation, and by data source (D-21, D-22);
  - a CI of the mean DM loss differential;
  - common-mask sensitivity;
  - seed variability from evaluated seeds (E5).
- *Result (N12):* 2,901 validation daytime hours with truth for every reference, target and
  horizon; 2,755 of them in normal operation. No prediction file exists yet.
- *Verification:*
  - **T-39–T-41:** split guard, alignment, coverage, zero-output hours kept, paired route
    comparison, the view counts, and the real validation run.
  - **T-15 unchanged:** the old metrics are still reproduced, and the BiLSTM rows of the fixture
    are identical before and after the change.
- *Original:*
  - The route comparison scored each route on its own rows. These were filtered by its own
    truth (k > 0.01, AC > 10 kW), and the "true" kW of the indirect route was rebuilt from the
    clipped k (C-34, C-16).
  - The common mask also filtered daytime by the truth, and used three models (C-35).
  - The models were scored on different n (2,510–2,580).

## 13. GFS forecasts (stage 6b, `solarpv.nwp`)

**Records and conventions.**
- Plan written before the code: `docs/stage6b_plan.md`. It includes the facts checked in the
  archive and a self-review against the original NWP (N1–N8) and against my own spec (B1–B5).
- The user asked for a multi-period sample before the full download
  (`00_data_audit/gfs_sample_2026-10-25/`), and approved the full download after reviewing it.
- Outputs:
  - `data/external/gfs_1p00/`: point values per run, not in git;
  - `data/processed/06b_gfs/gfs_features_D{h}.parquet`;
  - `outputs/06b_gfs/`;
  - `configs/nwp_decision.yaml`.
- Run journal: R6b-01 to R6b-08. Numbers: N13.
- Statistics on train and validation only; the test split informs nothing.

**OP-42 — GFS download and point extraction (`solarpv.nwp.download`, `grib`).**
- *Source:* the NOAA GFS archive on AWS (`noaa-gfs-bdp-pds`), 1°, the 12Z run of each UTC day from
  2021-08-29 to 2025-05-30, files f009–f084 every 3 h.
- *Method:*
  - From each file, 11 messages (DSWRF, four cloud covers, 2-m temperature and humidity,
    precipitable water, 10-m wind, sunshine duration) are fetched with HTTP Range requests from
    the `.idx` offsets.
  - They are decoded in memory with ecCodes; only the 3×3 grid points around 30N 35E are kept.
  - The `Last-Modified` time of every file is recorded.
- *Availability rule (T-17):*
  - A run serves an issuance T0 only if init + 6 h ≤ T0 and every one of its files was uploaded
    before T0.
  - Upload times: median 3.7 h after init, all within 5.2 h, except 7 of 35,646 files. Those
    were re-uploaded about 10.4 h after init, in 2 of the 1,371 runs.
  - The archive keeps only the latest version of a file, so those two runs (2022-01-24,
    2022-07-23) were rejected; their records are kept. The 06Z run of the same day is used,
    flagged as a fallback.
- *Result (N13):* 1,371 runs, 2 of them fallbacks; 26.5 GB transferred, 48 MB stored.
- *Verification:*
  - **T-43:** index parsing, message selection, range merging, the run choice, the
    late-upload rule.
  - **Grid assertions** on every message.
  - **The multi-period sample:** 30 runs, every check passed (R6b-04).
- *Original:* the Open-Meteo **archive** API at the target time: values not known at issuance
  (C-06, F-11). Removed from the final v6 models as leakage.

**OP-43 — Hourly reconstruction and candidate features per horizon (`solarpv.nwp.gfs`).**
- *De-averaging (L-06):* GFS averages run from the start s of each 6-hour window. The 3-hour mean
  is ((n−s)·A(s,n) − (n−3−s)·A(s,n−3))/3. Sunshine duration (SUNSD) is accumulated in the same
  windows, although tagged instantaneous; it is de-accumulated into a sunshine fraction.
- *To hours:*
  - **Irradiance:** the 3-hour mean is distributed in proportion to the clear-sky GHI. Block
    means are kept, and dark hours get 0. An even spread at twilight, the plan's first rule, put
    up to 27 W/m² into dark hours (R6b-03).
  - **Averaged clouds:** constant within the block.
  - **Instantaneous fields:** linear in time at the middle of the hour.
  - **Space:** bilinear to the site.
- *Per horizon:*
  - **Run and rows.** The rows of local day D+h come from the run of the UTC day of T0, and T-17
    is enforced on every row. A row labelled t gets the GFS values for (t, t+1].
  - **Candidate columns** (prefix `fcst_gfs_`, dictionary in
    `outputs/06b_gfs/gfs_feature_dictionary.csv`):
    - GHI, kt, POA and kt_poa (L-01, L-03);
    - the four cloud covers, the 3×3 cloud mean and std (L-07, L-08);
    - temperature, humidity, precipitable water, wind and sunshine (L-07);
    - cell temperature, physical PV power and k_phys (L-01);
    - daily and centred means (L-02).
  - **One non-GFS candidate (L-11a):** yesterday's clear-sky index of the measured irradiance.
- *Result (N13):* every hour of the three horizons has GFS values. Leads: D+1 10–34 h, D+2
  34–58 h, D+3 58–82 h (up to 88 h for the two fallbacks).
- *Verification:*
  - **T-42:** de-averaging, block means kept, dark hours, the label convention.
  - **T-17 on the real tables, row by row**, together with the coverage, horizon differences
    and range checks (T-44).
- *Original:* one NWP series for all horizons (N2), merged on a clock shifted by 2–3 h (N3),
  gaps filled backwards (N4).

**OP-44 — Sanity and value check; decision D-12 (`solarpv.nwp.check`).**
- *Sanity (train and validation, N13):*
  - The GFS clear-sky index on the panel plane against the measured one has a daily
    correlation of 0.82 / 0.79 / 0.74 at D+1 / D+2 / D+3, falling with the horizon. Neither red
    flag of the original's problem is raised.
  - The best hourly lag against the measurement is 0 h.
  - The seasonal bias is small: −0.025 to +0.012.
- *Value check:*
  - The fixed XGBoost of D-11 (3 seeds), trained on train and scored on validation, nRMSE in kW.
  - Adding the GFS candidates to the stage-6 contract lowers the error by 0.9–1.7 pp at every
    horizon and for both targets. The rule's margin is at most 0.13 pp.
  - **D-12: admitted** (`configs/nwp_decision.yaml`).
- *A red flag examined (R6b-07):*
  - With GFS, D+3 scored better on validation than D+1.
  - The mapping is correct (the leads above), and the same order appears with GFS features
    alone.
  - On an inner period of train (fit to 2023-04, scored 2023-05 to 2024-01), the order is the
    expected D+1 < D+2 < D+3.
  - On validation the paired bootstrap CI of RMSE(D+3) − RMSE(D+1) includes 0.
  - Conclusion: a property of the validation sample (D+3 GFS happened to be as good as D+1
    there), not a leak (`00_data_audit/gfs_value_2026-10-25/`).

## 14. Audit of the original backup (2026-11-01)

**OP-45 — Audit of the original backup (2026-11-01, no project data changed).** `00_data_audit/original_backup_2026-11-01/` (README, three scripts, run logs).
- *Why:* the user added an older backup of the original project (about January 2026). It is a step on the way to the v6 submission, never the final pipeline; the submission prevails.
- *Method:* an inventory of the working copy and of the partial `.git` folder (blobs read with zlib only, nothing executed or unpickled); the configurations of the base-selection notebooks parsed with `ast.literal_eval`; the original weather series compared with the measurement and with the real D+1 GFS run on train and validation only.
- *Result (N14):* the base selection fed the submitted 05e files, read the whole table (F-26) and, for the direct target, kept the per-inverter DC power (F-27); the original NWP was the Open-Meteo archive with `gfs_seamless`, of about D+1 skill, joined 2-3 h off the clock (F-28).
- *Verification:* T-45 (the claims follow from the tracked outputs; no coordinate is written out).

## 15. Feature selection (stage 7, `solarpv.selection`)

**Records and conventions.**
- Plan written before the code: `docs/stage7_plan.md`, with a self-review of the original
  selection and of my own spec; changes during the stage in its §7 (R7-01, R7-04).
- Outputs: `outputs/07_feature_selection/` (the twelve 05e and 05f files in the schema of the
  submitted ones, the step tables, figures 19–25, `model_inputs.json`, the manifest) and
  `configs/k_history_decision.yaml`.
- Run journal: R7-01 to R7-05; the canonical run is the college one (R7-04), the laptop outputs
  are kept in `00_data_audit/selection_2026-11-03/laptop_r7-02/`. Numbers: N15.
- Selection reads train rows only; validation is read once, at the end (S8); the test split is
  not read.
- Every score is nRMSE in kW (% of 3,449 kW); for the indirect route k × theoretical power first.
  Models: the fixed XGBoost of D-11 (3 seeds where two sets are compared, 8 threads, RH-03).

**OP-46 — Selection tables and the k-history candidates; decision D-24 (`solarpv.selection.common`).**
- *Why:* the original selection read the whole table, validation and test included (F-26); the
  report shows `target_k_lag_1d` (C-32), which is in no set.
- *Method:*
  - The stage-6 table of each horizon joined with the GFS candidates admitted by D-12, train
    rows only (`train_table` refuses any other split, T-47).
  - k history: `target_k_lag_{d}d` for the lags known at T0 (1, 2, 3, 7, 14; no lag 1 at D+2,
    no lag 1 or 2 at D+3) and the 3- and 7-day same-hour rolling means, from hours with k valid,
    theoretical power > 50 kW and no operational limit; DST guard and past-only imputation as in
    stage 6 (T-49).
  - Admission inside train: the pool with and without the k history, 5 expanding date folds
    with a 3-day gap, the D-12 rule (gain at D+1 above max(0.05 pp, 2 × pooled seed std), D+2
    and D+3 not worse by more than that).
- *Result (N15):* the gain at D+1 is 0.002 pp short of its margin, so the k history is **not
  admitted** (D-24), as in the laptop run.
- *Verification:* T-49; T-50 checks the decision against the rule.
- *Original:* no k history in any set, while the report shows one (C-32): a text correction in
  stage 11.

**OP-47 — Selection per target × horizon, S1–S7 (`solarpv.selection.steps`, `run`).**
- *Method:*
  - **Pool:** an allow-list, the feature contract of the horizon plus the admitted GFS
    candidates, checked by `assert_no_leakage` (F-27).
  - **Filters:** variance < 0.001 and MI < 0.02 (5 neighbours), with a core of 11 exempt. Pairs
    with |r| > 0.98 are reported, not removed (R7-01).
  - **RFE** one feature per step to 42, by mean |SHAP| on held-out folds; the core is kept.
  - **05c:** backward elimination from the 42 with a refit at each size; elbow = the smallest set
    within 1 % of the score of the 42.
  - **05e:** three votes (SHAP elbow; stability, 30 subsamples of 70 % of the train days, a share
    of at least 1 % of the mean |SHAP|, probability ≥ 0.6; MI above a shuffled null) and a
    consensus of 2 of 3. **05f:** tiers, an audit only.
  - Selection model: the fixed XGBoost of D-11 for both routes (R7-04; the submission used the
    base-notebook model of 100 trees).
- *Result (N15):*
  - Pools of 96 / 89 / 82 features at D+1 / D+2 / D+3.
  - 05e sets: indirect 31 / 28 / 32, direct 18 / 19 / 15.
  - GFS features lead every ranking; no dominant feature.
  - The MI vote endorsed all 42 starting features in every set, so the consensus is the union of
    the elbow and stability votes (R7-02, kept as registered).
- *Verification:* T-47, T-48 (synthetic data: date folds, the kW score, the selection model,
  RFE, elbow, stability, MI vote, consensus, tiers), T-50 (the twelve files, the allow-list, no
  leaky name, train only).
- *Original:* the whole table (F-26); for the direct target the per-inverter DC power of the same
  hour stayed in until after the ranking (F-27).

**OP-48 — Inside-train check, the validation look and the input of stage 9 (R7-04).**
- *Why:* the submitted 05e sets went into the models without a comparison with the full pool
  (F-29); on the laptop the direct sets were worse than the full pool inside train and on
  validation (R7-03).
- *Method, with the rule fixed before the run (`docs/stage7_plan.md` §7, R7-04):*
  - Inside train, every target × horizon: the 05e set against the full pool on the date folds
    and on an inner hold-out (fit to 2023-04-30, scored May 2023 to January 2024), 3 seeds. A
    set passes when on both it is worse by at most the D-12 margin.
  - A set that fails takes the full pool, and validation is not used for it. The others get one
    look at validation (train fit, validation scored; the third look of stage 7, counting the
    two laptop runs): the 05e set unless a red flag.
  - `run.decide_input` writes the outcome to `model_inputs.json`, which stage 9 reads.
- *Result (N15):*
  - **Indirect:** passes inside train at every horizon, and validation shows no red flag. Input
    of stage 9: the 05e sets of 31 / 28 / 32 features.
  - **Direct:** fails at every horizon on both samples, by 0.10–0.71 pp. Input of stage 9: the
    full pool of 96 / 89 / 82 features. The importance is concentrated (top three features
    37–65 % of the mean |SHAP|), so few features reach the 1 % share of the stability vote, and
    those left out carry 14–29 % of it together.
  - Validation, reported only for the direct route: red flags at D+2 and D+3, as inside train;
    at D+1 the 05e set is 0.068 pp better than the full pool (margin 0.058 pp), the opposite of
    inside train.
- *Verification:* T-48 (`decide_input`: a failed inside-train check cannot be overturned by
  validation), T-50 (`model_inputs.json` follows both checks and the manifest; no leaky name).
- *Original:* no such comparison; the 05e sets were described as the model input (§3.6), while
  the original XGBoost was trained on all its columns (C-31).

## 16. The stage-9 runner (W1, `solarpv.pipeline`)

**Records.**
- Plan written before the code: `docs/stage9_plan.md`, with a self-review (§10) against the
  original orchestration scripts (M1–M6) and against my earlier code (X1–X8).
- Each run writes `outputs/runs/<run_id>/`: `run.json`, `config.yaml`, invocation records, step
  records (`step.json`) and logs. The prediction CSVs of the runs are not in git.
- No model is trained in W1. Only validation rows are scored; smoke and dev runs never load a
  test row.

**OP-49 — The runner: input check, evaluation window, references as files, evaluation
(`solarpv.pipeline`).**
- *Why:* one command runs every step of stages 9–11 in sequence, with the same rules for every
  model (§3 of the plan). W1 brings the two steps that train nothing: `baselines` and `evaluate`.
- *Method:*
  - **Input check (preflight of every run, and `check`):** every input is compared with the
    SHA-256 recorded by the stage that produced it (the stage-6 and stage-6b manifests: the three
    feature tables, the three GFS tables, the two dictionaries, `configs/derived_constants.yaml`);
    `model_inputs.json` and `configs/split_manifest.yaml` must be tracked and unmodified in git.
    A missing or changed input stops the run (M4).
  - **Reading the truth:** the stage-6 D+1 table with `exclude_splits=("test",)`: the parquet
    reader drops the test rows, so they never reach pandas. The table has one row group, so the
    bytes are decoded by the reader, but no project code sees a test row (X3).
  - **Evaluation window:**
    - smoke: the 28 local days 2024-03-15 to 2024-04-11 of validation (671 hours: the window
      contains the 23-hour day of the spring DST change, 2024-03-29);
    - dev and final: the whole validation split;
    - final with `--allow-test`: also the whole test split (stage 11 only).
    Rows of the split before the window are relabelled `<split>_context` and feed only the
    persistence reference; rows after the window are dropped.
  - **References (`baselines` step):** OP-40 unchanged, for every row of the window, written as
    prediction files in the contract of §6.10 (`predictions_<reference>.csv`, seed −1). They then
    pass the alignment and coverage checks of OP-41 like any model.
  - **Evaluation (`evaluate` step):** OP-41 unchanged, on the prediction files of the completed
    steps of the same run only, each checked against the hash in its step record (M1, X1); the
    references are taken from those files (`with_references=False`); rows duplicated across files
    are an error (X2); in smoke and dev a prediction row in the test split is an error (X7).
- *Result (run journal R9-02):* the numbers of the smoke window are a code check, not a result
  of the study; the dev run equals the stage-8B validation tables (N12).
- *Verification:*
  - **T-55 (real data):** a dev run of `baselines,evaluate` reproduces the eight committed
    stage-8B validation tables (`outputs/07_evaluation/val/`) to the written precision, with no
    test row loaded; the smoke window is covered on every daytime hour; nothing outside the run
    folder changes.
  - **T-51–T-54:** step selection, modes and budgets, the gates, records, resume and the fixes
    of X1–X8 on synthetic tables.
- *Original:* one script trained, evaluated and reported in sequence, scoring whatever files were
  in `outputs/`, overwriting its logs, and evaluating the test split every time (C-36–C-41).

**W2: the `xgb` step and the checks of the stage-9 plan §14.** Self-review before the code
(`docs/stage9_plan.md` §13: M7–M12 against the original, X10–X13 against my code); the decision
rules were committed before any XGB result (§14, commit `0cc8d17`). Only train and validation
rows are read; nothing here reads the test split.

**OP-50 — The `xgb` step: model tables, rows, no imputation, raw predictions
(`solarpv.pipeline.steps.run_xgb`, `solarpv.models.xgb_sota`, `solarpv.pipeline.data`).**
- *Why:* XGB-Opt of §6.11 in the runner, with its training logs (§3.5 of the plan), on the
  inputs of stage 7 (OP-48) and with every rule of the W2 review.
- *Method:*
  - **Reading (X10):** `model_table(h, splits)` joins the stage-6 table of D+h with its GFS table
    as the stage-6b loader does (left join on `timestamp_utc`, the GFS `t0_utc` dropped), but the
    parquet reader filters the rows: the feature table by `split` ∈ {train, val}, the GFS table
    (no split column) by `timestamp_utc` < 2024-10-01 00:00 UTC. It asserts that no row at or after
    the test start was read. Test rows are read only in a final run with `allow_test`, and only
    to be predicted.
  - **Inputs (M12a):** the list of `model_inputs.json` per target × horizon (31/28/32 indirect,
    96/89/82 direct), each feature checked against the stage-7 pool of the horizon and the leakage
    rules, in its recorded order.
  - **Rows (M11, §14.2):**
    - fit: train, not warm-up, not op-limited; indirect also `k_valid` and theo > 50 kW
      (9,575 rows), direct AC present (20,393 rows, nights included); smoke mode the last 56 local
      days of train;
    - early stopping: the rows of the tuning window with the same mask (dev: the whole validation
      split; smoke: the 28-day window of OP-49), RMSE in the training unit (k or kW), 50 rounds;
    - objective: the daytime hours of the tuning window with AC truth, op-limited hours included
      (§14.1; dev: 2,901 hours, 146 op-limited).
  - **No imputation (M8):** NaN is passed to XGBoost; the NaN count of every feature on the fit,
    early-stopping, scoring and predicted rows is written to `xgb_config.json`. On the real
    tables NaN occurs only in `irradiance_clear_index_prevday` (a few train hours) and in the GFS
    ratios at night (direct fit rows).
  - **Monotone constraints (M7):** read from `configs/monotone_decision.yaml` (OP-51); exact
    names of the D-15 list; the map per target × horizon and the string XGBoost applied are
    recorded.
  - **Search and fit:** Optuna TPE, `TPESampler(seed=S)`, XGBoost `random_state=S`, the space of
    §6.11, `hist`, 8 threads, one study per target × horizon × seed in `optuna.db`; the best
    trial's parameters refitted with the same rows and early stopping (no refit on train + val,
    D-10); the refit must equal its trial.
  - **Outputs:** `predictions_xgb_sota_seed<S>.csv` with the raw model output (k never clipped)
    for every hour of every window of the run, in the contract of §6.10; `trials.csv`,
    `learning_curves.csv`, `best_params.json`, `xgb_config.json`, two figures, the boosters in
    `models/` and the study in `optuna.db` (both not in git).
- *Result:* run journal R9-05 (smoke) and R9-06 (dev).
- *Verification:* T-56–T-59 (constraints by exact name, allow-list, rows, no imputation, the
  objective equal to the evaluation's `nRMSE_kW_pct`, raw output, the full record, a reloaded
  booster reproduces the predictions exactly, seeds, the warnings), T-60 (no test row read,
  filters inside the reader), T-61 (validation labels do not change the trees).
- *Original:* `fillna(0)`, substring constraints, training on every hour, clips in the scoring
  scripts, saved parameters that were not the model (C-20, C-42–C-46).

**OP-51 — D-15, monotone constraints for k (`python -m solarpv.models.xgb_checks d15`).**
- *Why:* the original applied +1 on clear-sky irradiance to the k target too, against the sign
  of k in train (C-33); D-15 asks whether to keep that constraint.
- *Method (§14.3, registered before the result):* the fixed XGBoost of D-11 (400 trees, depth 6,
  learning rate 0.05; a new dict per fit, X13), seeds 42, 123, 777, on the indirect fit rows of
  train with the stage-7 inputs; setting A, +1 on the D-15 features present (`ghi_clearsky` at
  every horizon), against setting B, none; scored by §14.1 on the whole validation split.
  gain_h = mean_A − mean_B, margin_h = max(0.05 pp, 2 × pooled seed std); B only if gain_1 >
  margin_1 and gain_h > −margin_h at D+2 and D+3, otherwise A, the v6 setting, stays. The
  marginal Spearman sign of k and of AC against each D-15 feature on the indirect fit rows is
  reported. The direct target keeps the D-15 list with +1 (not part of the question).
- *Result:* `configs/monotone_decision.yaml`, `outputs/09_xgb_checks/d15_check.csv` and
  `d15_marginal_sign.csv`; run journal R9-04.
- *Verification:* T-62 (the decision file is readable by the step; `XGB_PARAMS` unchanged),
  T-63 (the rule).
- *Original:* +1 by substring on both targets, never compared (C-20, C-33).

**OP-52 — L-09 (bias correction) and R-14 (mean bias) on the dev run
(`python -m solarpv.models.xgb_checks l09|r14 --run <id>`).**
- *Why:* L-09 is the literature idea of a final MOS layer (§16b), registered with its rule
  before any model result; R-14 predicts a positive level bias of the direct target from the
  1 %/year decline of output per unit irradiance (Q29).
- *Method (§14.4, §14.5):*
  - **L-09:** per target × horizon the best parameters of the dev study with a fixed number of
    rounds (the dev best iteration + 1, no early stopping), seeds 42, 123, 777; out-of-fold kW
    predictions on train from the five expanding date folds of stage 7 (3-day gap); OLS
    y = a + b × ŷ on the out-of-fold daytime hours with AC truth, not op-limited, not warm-up;
    applied as clip(a + b × ŷ, 0, 3,449) to the validation forecast of the same base model
    trained on all of train; the rule of §14.3 per target. Seed 42 on all of train with the dev
    rounds is the dev final fit; the check logs the largest difference from the dev predictions.
  - **R-14:** mean bias error (forecast − actual, kW) of the dev forecast on the validation
    daytime hours with AC truth, overall and per local month, with a 95 % circular block
    bootstrap interval (block 24 h, 1,000 resamples, seed 42); reported, not decided.
  - The checks only read the dev run (every file checked against its step record) and never
    write a `predictions_*` file or into a run folder.
- *Result:* `configs/mos_decision.yaml`, `outputs/09_xgb_checks/l09_check.csv`, `r14_bias.csv`;
  run journal R9-07.
- *Verification:* T-61 (the MOS coefficients do not depend on the validation labels), T-63 (the
  rules, the outputs, L-09 and R-14 on a synthetic dev run, the run folder unchanged).
- *Original:* no bias correction and no bias check.

**OP-53 — The adopted L-09 correction inside the `xgb` step
(`solarpv.pipeline.steps.run_xgb`, `solarpv.models.xgb_sota.fit_mos` and `apply_mos`).**
- *Why:* §14.4 adopted the correction for the indirect route (OP-52) and requires it to become
  part of the step before G-1. Design and self-review before the code: `docs/stage9_plan.md` §15
  (X14–X16).
- *Method:*
  - **Decision:** `configs/mos_decision.yaml` is an input of the step (in its fingerprint) and a
    tracked input of `check`; per target, `adopted` true gives the corrected forecast. Today:
    indirect corrected, direct raw. A smoke or dev run without the file writes the raw forecast
    with a warning (L-09 is decided on a dev run); a final run refuses.
  - **Refitted in every run:** for each target × horizon × seed with the correction adopted, the
    final fit's model (its searched parameters, seed, constraints and rounds = best iteration + 1,
    fixed, no early stopping) is trained on the five expanding date folds of the run's train
    range (3-day gap; dev and final all of train, smoke its 56 days); OLS y = a + b × ŷ on the
    held-out daytime hours with AC truth, not op-limited, not warm-up. Only train rows enter
    (A-14). The same functions serve the L-09 check (X14).
  - **Applied** to the final fit's forecast of every window of the run, on daytime hours:
    kW' = clip(a + b × ŷ_kW, 0, 3,449); for k the file holds k' = kW' / theo, so that
    `evaluate.to_kw` returns kW' (X15; the smallest daytime theo with AC truth is about 62 kW in
    train and validation). Night hours keep the raw output. The model name stays `xgb_sota`.
  - **Records:** `uncorrected_xgb_sota_seed<S>.csv` (the raw booster output of the corrected
    targets, not scored by `evaluate`; X16), `mos_coefficients.csv`, a `mos` entry per fit in
    `best_params.json` (a, b, out-of-fold hours, fold rows, the uncorrected and corrected value of
    the tuning window), `value_kw_forecast` (the value of the written forecast), the decision and
    its hash in `xgb_config.json`.
  - The search, the objective (the base model is tuned without the correction), early stopping
    and the boosters are unchanged.
- *Result:* run journal R9-09. The dev run reproduces the study of R9-06, and its seed-42
  coefficients and corrected values equal the L-09 check exactly. In kW the indirect route
  improves as decided; its k-unit statistics (nRMSE in k, skill and DM in k) get worse, because
  the kW intercept weighs heavily at low sun; the unit of those statistics is open (Q-19).
- *Verification:* T-64 (the step and the check fit the same coefficients and the evaluation scores
  the corrected value; exact k ↔ kW inversion; nights and the direct target unchanged; the
  uncorrected file; without the decision a raw dev forecast and a refused final run), T-58 and
  T-59 (the written forecast's value equals the evaluation; the booster and the recorded
  coefficients reproduce the written file exactly), T-61 (the coefficients do not depend on the
  validation labels).
- *Original:* no bias correction.

**OP-54 — DM and skill in kW for both routes (`solarpv.evaluation.evaluate`; Q-19, the D-14
amendment of 2026-11-07).**
- *Why:* every decision of the rebuild is taken in kW (stage-9 plan §14.1), but OP-41 computed
  the DM tests and the skill of the indirect route in k units. With the adopted L-09 correction
  the two units disagree (OP-53, R9-09). The user decided (Q-19) to add the kW statistics for
  both routes and keep the target-unit ones as secondary. Design and self-review before the
  code: `docs/stage9_plan.md` §16 (X17–X18).
- *Method:*
  - `evaluate.in_kw` converts the scored rows once: the truth is actual AC for both targets, the
    indirect forecast goes through `to_kw`, the direct forecast is unchanged.
  - `dm_tests_kw.csv`: the same `dm_table` on those rows (the same pairs, reference, HLN,
    Bonferroni families per target × horizon, CI of the mean loss difference and power analysis),
    with a `unit` column. It is the table the stage-10 acceptance reads.
  - `benchmark_summary.csv` gains `skill_score_kw` (against persistence of h days, on the common
    hours) and `skill_cliper_kw` (against the CLIPER RMSE of the AC series).
  - Only with a truth table; the fixture mode of stage 8A is unchanged (T-15). No existing column
    or table changes.
  - The validation command now reads the truth without the test split (X17).
- *Result:* run journal R9-11. The regenerated stage-8B tables equal the committed ones in every
  existing column and table; for the direct target the kW statistics equal the existing ones
  exactly. In the dev run XGB is significantly better than every reference in kW, in both routes
  and at every horizon.
- *Verification:* T-65; T-55 (the dev run of `baselines,evaluate` equals the regenerated
  stage-8B tables, `dm_tests_kw.csv` included).
- *Original:* DM and skill in the target's unit only (the DM of the original did not apply HLN,
  C-15).

**OP-55 — The datasets of the sequence models (`solarpv.models.sequence_data`; W3, D-25–D-28).**
- *Why:* BiLSTM-Opt, the three TFT models and PatchTST need windows: an encoder of past hours, a
  decoder over the forecast days, and a mapping of decoder steps to D+1..D+3. The original built
  them on the local clock read as UTC, by step number, inside each split, with different hours
  per model (C-47–C-52). Design and self-review before the code: `docs/stage9_plan.md` §17
  (E1–E7, M13–M18, X19–X23); the open choices were decided by the user (D-25–D-28) before any
  code.
- *Method:*
  - **Inputs:** the three stage-6 feature tables, the three stage-6b GFS tables, the stage-5
    hourly table and its minute flags, the cleaning policy (the 5 % share of D-22); in smoke and
    dev every file is filtered inside the parquet reader to `timestamp_utc < TEST_START` (X21).
  - **Origins:** one per local day, T0 = 00:00 local in UTC (D-16). For a scope (train after the
    warm-up, or the run's validation window) the origins run from two local days before its first
    hour to the local day of its last hour: 872 training and 246 validation origins in dev, 58 and
    30 in smoke; 245 test origins in a final run with `allow_test` only.
  - **Encoder:** the L hours before T0 (72 h BiLSTM, 96 h TFT-96, 168 h TFT-168 and the TFT
    baseline; PatchTST's input size is set in W4). Series: the seven measured series of stage 6
    with the D-22 masks of `masked_sources` (X22), and `target_k` for the indirect route (NaN in
    op-limited hours); gaps filled once with the stage-6 rule (same UTC hour, previous 30 days,
    older rows only) and flagged by one 0/1 channel per series; the known inputs of the encoder
    hours (D+1 GFS).
  - **Decoder:** the hours of the three local days from T0 (72, or 71 / 73 around a DST day),
    padded to 73 (D-25); the horizon of a step is the local day of its valid hour; the decoder
    stops at the end of its split (training at VAL_START, validation at TEST_START in every
    mode). Inputs: the part of `model_inputs.json` known at T0, the union over the horizons (D-26,
    D-27, D-28: indirect 10 calendar, clear-sky and geometry features + 20 GFS, direct 27 + 22),
    the GFS value from the table of the step's horizon, whose `t0_utc` is the origin's T0 (the
    same 12Z run for the three days; init + 6 h ≤ T0 asserted). No lag, no rolling statistic, no
    measurement of a decoder hour. PatchTST: the target series only (L-19).
  - **Scaling (A-14):** z-scores per channel from the train hours that are not warm-up; targets
    not transformed (k as is, AC / 3,449); a missing input takes the train mean (counted).
  - **Loss and writer:** the loss mask is `xgb_sota.fit_mask` on the kept steps (X23); the writer
    turns window forecasts into the rows of §6.10, one per hour of the scope and horizon, and
    refuses duplicates, gaps and missing forecasts.
  - `python -m solarpv.models.sequence_data [--mode dev|smoke]` builds the train and validation
    datasets of every model × target and checks them against theory; it writes
    `outputs/09_sequence_data/dataset_report_<mode>.json`, `checks_<mode>.csv` and a run log, and
    no array.
- *Result (run journal R9-13, commit `854b0f7`):* every check passes on train and validation for
  the five models and both targets. 872 training and 246 validation origins; three-day decoders
  of 72 steps, and of 71 / 73 on the 6 / 9 training origins and the 3 validation origins around a
  DST day; every D+h block that ends before its limit is its local day (23, 24 or 25 hours);
  every hour of the scope exactly once per horizon (20,859 train hours after the warm-up, 5,832
  validation hours); the loss rows equal the XGB-dev fit and early-stopping rows exactly (9,575 /
  2,825 indirect, 20,393 / 5,685 direct); the encoder series equal the stage-6 `lag_1d` with 0
  mismatches; the scaler is fitted on 20,859 train hours. Missing known inputs take the train
  mean: `airmass` at night (direct route) and `irradiance_clear_index_prevday` on 24 train hours.
- *Verification:* T-66–T-71 (synthetic, both DST days), T-02 (the real train and validation
  tables, 16 cases).
- *Original:* naive local clock, hourly origins with fixed leads (BiLSTM) or steps by number
  (TFT), windows inside each split, a different set of scored hours per model, the 05e sets for
  the BiLSTM and 18 calendar and irradiance known reals for the TFT, no weather forecast.

## 16b. Figure book (all stages, `solarpv.figures`)

**OP-56 — Process and correction figures for every stage (read-only).**
- *Why:* user request (2026-11-11): every stage gets figures that show its procedure, what was
  found and what it led to; differences from the submitted project go to a separate set for the
  defense notes. Plan, self-review (G1–G5, X1–X6) and rules: `docs/figures_plan.md`.
- *Method:* each figure is registered with its question, a takeaway formatted from the facts its
  builder computes, and the step it led to. Builders read stage outputs (tables, manifests,
  configs) and, where needed, the stage-6 and stage-5 tables through the stage functions; no data
  is written or changed. Target and model statistics use train or validation only; whole-period
  data descriptions (clocks, flags, held values) mark the splits. Claims in a takeaway that do not
  come straight from a number are asserted in the builder.
- *Parameters:* Okabe-Ito colours with a second cue, 200 DPI PNG, one value axis per panel.
- *Outputs:* `outputs/figures/<stage>/<ID>_<slug>.png`, `outputs/figures/figures_manifest.json`
  (commit, sha256 of every input and file, facts, takeaway), `outputs/figures/gallery.html`,
  `docs/figure_catalog.md` (generated; it also lists the figures the stage producers made before).
- *Result:* 37 figures: 31 process figures for stages 0, 1, 2, 4, 5, 6, 6b, 7, 8 and 9, and 6
  correction figures used by `docs/defense_notes.md`.
- *Additions:* W4 (2026-11-11) added S09-07–S09-12 (OP-57). The review of the W4 figures
  (2026-11-12, R9-18) added the correction figures C-07 (the submitted BiLSTM seed table), C-08
  (scored hours per model) and C-09 (the attention window of TFT-96), read from the submitted
  files and the dev runs (validation only), and fixed X36 (a typed hour count) and X37 (C-10).
  The book now holds 46 figures: 37 process and 9 correction. W5 (2026-11-12, OP-58) added the
  process figures S09-13–S09-19 and the correction figure C-10: 54 figures, 44 process and 10
  correction. After W5 (2026-11-12, user request, X52, R9-22) every task gets a generated figure
  sheet in `docs/figure_sheets/`, the gallery is grouped by task, and the process figures
  S09-20–S09-24 cover the W5 outputs that had none (the select checks, the booster rounds of
  SHAP, the dominance and rank checks, the TFT selection weights, the G-1 points): 59 figures,
  49 process and 10 correction. The W4 figures were then redone with a look-back and one look
  for every model (2026-11-12, X53, X54, R9-23): S09-25–S09-30 for W4, S09-31 and S09-32 for
  W2, S09-33 for W1 and S08-04 for stage 8, with S09-06 and S09-02 redrawn: 69 figures, 59
  process and 10 correction. S09-28 and S09-29 read the dev prediction and quantile files and
  the stage-6 truth through `load_truth` with the test split excluded in the reader.
- *Verification:* `tests/contracts/test_figures.py` (T-72): every registered figure is built,
  unchanged since the build and drawn from unchanged inputs; every takeaway re-renders from its
  facts; the catalog is current; every completed stage has process figures; process figures do
  not read the submission's evidence except the evaluation-code check; no description carries a
  typed count, and the scored hours and the seed-table facts match their source tables (X36, X37);
  every figure belongs to one task, every completed task has its figures, and every sheet is
  current (X52).
- *Original:* no figure was linked to its inputs or code version; the v6 figures were drawn from
  the shifted, clipped data and are regenerated in stage 11.

## 16c. The neural models (stage 9, W4, `solarpv.pipeline.neural`)

**OP-57 — The neural steps `bilstm`, `tft96`, `tft168`, `tftbase`, `patchtst`: the windows of
W3 to a training worker in `solar-gpu`, and back into prediction files.**
- *Why:* D-08 keeps the five neural models of v6. They train on the datasets of OP-55 so that
  every model sees the same hours, inputs and loss rows as XGB. Plan and self-review before the
  code: `docs/stage9_plan.md` §18 (M19–M24, X25–X33; X34 and X35 found while testing); the GPU
  environment (D-29) and the dev budgets (D-17) decided by the user.
- *Method:*
  - **Data side (env `solar`):** the stage-5, 6 and 6b inputs read before TEST_START (dev); the
    W3 scaler (train hours); per target the training windows (872 origins in dev) and the
    windows of every evaluation window (246 validation origins); `check_dataset` on each; the
    missing known inputs checked on the hours the windows read (X30, X35). The arrays go to the
    worker as compressed files with SHA-256, without any timestamp; the valid hours stay with the
    runner (M13, M18).
  - **Training side (env `solar-gpu`):** FP32 on the P40, deterministic kernels (X32; the TFT's
    interpolation as a matrix product, X34). Losses as a mean over the W3 loss mask only (X31):
    the quantile loss (BiLSTM 0.1/0.5/0.9; TFT the library's seven quantiles), PatchTST
    neuralforecast's MAE with `available_mask`. Early stopping on that loss over the validation
    windows, the best epoch restored. Optuna TPE with seed 42 on §14.1, pooled over the three
    horizons (one model per target). No fitted normaliser in the libraries (X29).
  - **Writing:** the point forecast (the 0.5 quantile; PatchTST's output) through
    `sequence_data.predictions_frame`, one row per hour of the window and horizon; the quantiles
    in `quantiles_*.csv` (not scored); the MBE of every model on its training and validation
    windows (`mbe.csv`, M24); trials, learning curves, the configuration and the warnings.
- *Parameters:* the budgets of D-17 (dev) and D-08 (final) in `pipeline/config.BUDGETS`; the
  settings of §18.4 in `models/neural_job.py` (`SEARCH_SPACES`, `FIXED`, `QUANTILES`).
- *Result (run journal R9-17, run `20261111T160714Z_dev`, 3.2 h on the P40):* every step passed;
  every model was scored on the same 2,901 daytime validation hours per horizon as XGB-dev and the
  references. Validation nRMSE in kW at D+1, indirect / direct: BiLSTM 6.97 / 6.75, TFT-96 6.78
  / 6.75, TFT-168 6.77 / 6.79, TFT baseline 6.83 / 6.87, PatchTST 8.77 / 8.09 (XGB-dev 6.69 /
  6.66; persistence 9.68; climatology 8.06 / 8.04). Every neural model is below persistence at
  every target and horizon; PatchTST is above climatology. Mean bias on validation +0.37 to
  +2.73 % of capacity. These are tuning-split numbers with reduced budgets and one seed.
- *Smoke (R9-16):* the first run found X35 before any training; after the fix every step passed,
  and a repeated run wrote byte-identical forecasts.
- *Verification:* T-73–T-81 (`tests/unit/test_neural_runner.py` with a stub worker;
  `tests/gpu/test_neural_gpu.py` in `solar-gpu`); smoke runs on real data (R9-16): the objective
  equals the evaluation's `nRMSE_kW_pct` within 5e-7, and two runs with one seed write
  byte-identical forecasts.
- *Original:* hourly origins for the BiLSTM and step-number horizons for the TFT (OP-55), seeds
  of the BiLSTM scored by training metrics (C-10), two objectives for the BiLSTM (C-53), the TFT
  baseline from an earlier code version (C-54), the libraries' own normalisers (settings lost,
  R-16), and unexplained neural biases (C-57).

## 16d. The last runner steps (stage 9, W5, `solarpv.pipeline`)

**OP-58 — `select`, `explain`, `report`, the step `test`, `archive` and `promote` (read-only on
the data; they explain and summarise model outputs).**
- *Why:* the runner had to verify the input of every model, explain the trained models, summarise
  each run with its warnings, and close the final gate before W6. Plan and self-review before the
  code: `docs/stage9_plan.md` §19 (M25–M31 in the original, X38–X49 in my code; X50 and X51 found
  while testing and writing the runbook); the user's decisions D-30 (`select` verifies and reuses),
  D-31 (the `explain` scope, with `archive`) and D-32 (two phases of a final run).
- *Method:*
  - **`select`:** no row of any table is read: the stage-7 outputs must be tracked and
    unmodified, the tables named in the stage-7 run logs must be the current ones (by SHA-256
    prefix), the manifest must record selection on train with one validation look, and
    `model_inputs.json` must equal `selection.run.decide_input` on the committed checks (the
    indirect 05e sets; the direct full pool built from the parquet schemas). Its SHA-256 is
    recorded; every model step of the run refuses another file.
  - **`explain`:** XGB: `shap.TreeExplainer` (tree-path-dependent, no background) of each booster
    sliced to its forecast rounds, on the daytime scoring hours of the validation window with AC
    truth (never fit rows); additivity against the uncorrected forecast and agreement with
    XGBoost's `pred_contribs`, both relative to the output scale; the kW view of the indirect
    route (theo × the L-09 slope); group attribution as the mean of |Σφ over the group|; Spearman
    rank stability over common features. TFT: the stored validation attention (encoder share on
    positions whole days before each step's valid hour, against the share of such positions; the
    profile by lag and per horizon block) and the variable-selection weights per channel and per
    variable. Optuna: fANOVA with evaluator seeds 0–9, on copies of the study databases.
  - **`report`:** every table read from a hashed file of the run; the warnings of §3.5 recomputed
    by the steps' own functions and matched with the step logs; the seed spread in kW from the
    benchmark; the facts that inform the G-1 list.
  - **Final gate:** the gate before every final step; no slices in final; every read of a test
    row logged; validation scored by `evaluate`, the test split only by the step `test` after the
    acceptance file. `promote` copies a complete final run to the §6.10 paths; `archive` packs a
    run with the SHA-256 of every file.
- *Parameters:* SHAP tolerances 1e-4 (additivity) and 1e-5 (`pred_contribs`) relative to the
  output scale (1 for k, 3,449 for kW), the kW view within 0.34 kW; the stage-7 dominance factor
  10; ten fANOVA seeds; TPE start-up 10 trials.
- *Result (run journal R9-20, R9-21):* the smoke run of the whole sequence passed at a clean
  commit and repeated the first smoke run byte for byte. The XGB dev run `20261112T063719Z_dev`
  reproduces `20261107T135938Z_dev` byte for byte; its SHAP adds up within 1.4e-6 of the output
  scale on 2,901 hours per horizon, and the saved boosters would be off by up to 199 kW if
  explained whole (X46). A GFS forecast leads every target and horizon; the GFS group holds 56 to
  79 % of the group attribution; no dominance flag (at most 6.7). The neural dev run, resumed with
  the new steps: no TFT prefers the hour it forecasts (same-hour ratio 0.82 to 1.07); the
  attention follows fixed clock hours of earlier days. The reports reproduce 9 of 10 warnings
  exactly; the tenth is X45 (an edge of the effective `hidden_continuous_size` that the W4 step did
  not see). No test row was read.
- *Verification:* T-82–T-91 (`tests/unit/test_w5_steps.py`); the smoke and dev runs above;
  notebook `06_models.ipynb`; figures S09-13–S09-19 and C-10.
- *Original:* no selection step in the reproduction (C-58); SHAP of a 76-feature model with
  leaking rolling groups (C-59), first-match group shares (C-60), no record of the rows or the
  explainer (C-61), attention offered as validation (C-62), one unseeded importance evaluation
  (C-63), numbers without a file (C-64).

## 16e. Quantile forecasts read as a distribution (stage 9, `solarpv.evaluation.quantiles`)

**OP-59 — Quantiles sorted per row wherever they are used as a distribution (D-33, X53; no
stored file changed).**
- *Why:* figure S09-29 showed that the seven quantile heads of the TFTs are separate outputs that
  nothing orders: on the scored validation hours they are out of order on 66 % (TFT-96, direct)
  and 56 % (TFT baseline, direct) of the hours. An interval or a quantile loss read from such
  rows is not a distribution. The decision (user, 2026-11-12) is option A: sort after training for
  every use as a distribution, and leave the point forecast as the model wrote it.
- *Method:* `ordered(frame)` returns a copy in which the columns `q<level>` are sorted per row
  (columns taken in the order of their level, never of their position); every other column, the
  point forecast `y_pred` included, is untouched, and the stored `quantiles_*.csv` files keep the
  raw model output. The sorting is the monotone rearrangement of the quantile curve. It never
  raises the summed quantile loss for any truth value: with ρ_t(u) = t·u + max(−u, 0), the sum
  over the quantiles is Σ t_k (y − q_k) + Σ max(q_k − y, 0); the second sum does not depend on the
  order and the first is smallest when the q_k rise with the t_k (the rearrangement inequality).
  The reference (Chernozhukov, Fernández-Val and Galichon, 2010) was written from memory and is
  to be verified before it enters the report.
- *Parameters:* none; the central interval is the 0.1 to 0.9 quantile (nominal 80 %).
- *Result (run journal R9-24; validation, scored daytime hours, 8,703 per model and route,
  development run, seed 42):* quantiles out of order on 66.4 % (TFT-96 direct), 56.1 % (TFT
  baseline direct), 12.4 % (TFT-96 indirect) and below 2 % elsewhere; never in the BiLSTM. After
  sorting, the quantile loss of the direct route falls from 35.59 to 34.66 (TFT-96) and from
  35.41 to 34.91 kW (TFT baseline), and the 80 % interval covers 63.8 % (was 62.2 %) and 80.5 %
  (was 72.0 %). The written point forecast equals the raw 0.5 head in kW for all eight model and
  route pairs, so no score, no Diebold–Mariano test and no Optuna objective changes. Sorting then
  converting to kW equals converting then sorting (largest difference 0.0): the conversion is a
  product with a non-negative number followed by a clip.
- *Cost, stated in the report:* on a crossed hour the point forecast is not the median of the
  sorted distribution. Sorting the point too would have moved 60.5 % of the TFT-96 direct hours
  (8.7 kW mean, 47.6 kW at most) and raised the nRMSE by 0.016 points. Prevention of the crossing
  in training (a new model and tuning, about 3 h of reruns) is future work.
- *Verification:* T-92 (`tests/unit/test_quantiles.py`: rows in order unchanged, crossed rows
  sorted, point and other columns untouched, the loss never rises row by row for random truth
  values, ordering by level, coverage count, a frame without quantiles refused); figure S09-29
  asserts that the BiLSTM never crosses and that the sorted loss is not above the raw loss.
  `tests/contracts/test_quantile_use.py` keeps the rule in force for later code: every file that
  reads `quantiles_*.csv` or a `q<level>` column must import the sorting module (exceptions: the
  writer in `pipeline/neural.py`, which must keep the raw output, and `pipeline/report.py`, which
  only filters file names); a planted reader is caught, and removing the import from the figure
  module makes the guard fail. On the development run, for the four quantile models, the
  written point equals the 0.5 head, sorting leaves it unchanged and leaves no row out of order,
  and the sorted loss is not above the raw one.
- *Original:* not compared; the surviving submission files were not examined for the order of
  the quantiles.

## 16f. Gate G-1 (W6)

**OP-60 — Stages 4 and 5 rerun by accident on the college machine, and the original files
restored from the hand-off package (2026-11-12; no number changed).**
- *Why:* while writing the reproduction script, `python -m <module> --help` was called on the
  stage modules to read their options. The modules had no argument parsing (X62), so the
  command ran them: `solarpv.data.preprocess` and `solarpv.data.clean` completed and rewrote
  `data/processed/04_preprocess/` (both parquet files and the manifest), `05_clean/` (both
  parquet files and the manifest) and the tracked `outputs/05_cleaning/cleaning_manifest.json`;
  `ingest`, `features.build`, `features.decide_k` and `nwp.gfs` stopped before writing their
  outputs (the shell closed their output pipe; the GFS run was stopped by hand); the selection,
  which would have made a fourth validation look, was never reached.
- *Method:* the nine affected files were extracted from the hand-off package
  (`solar_handoff_2026-11-03_ee976e4.zip`, outside the repository, read-only) into a new, empty
  folder, and each was checked against `HANDOFF_MANIFEST.sha256`. The content of the four
  regenerated parquet files was compared with the originals (`pandas.testing.assert_frame_equal`,
  exact). The originals were then copied back, the regenerated files moved aside (not deleted),
  and the logs of the three aborted runs moved to `outputs/g1_audit/incident_2026-11-12/`.
- *Result:* the regenerated files had the same shape, columns, types, index and every value as
  the originals (1,971,360 × 31 and 32,856 × 35 for stage 4; 1,971,360 × 48 and 32,856 × 43 for
  stage 5); every statistic of the regenerated cleaning manifest equalled the committed one. Only
  the bytes differed, because the parquet writer differs: pyarrow 26.0.0 and pandas 3.1.0 on the
  college machine against 25.0.0 and 3.0.6 on the laptop. So stages 4 and 5 reproduce exactly on
  the college environment, and a reproduction must compare content, not bytes (M33).
- *Verification:* after the restore, 22 of 22 stage files of the hand-off (stages 1, 4, 5, 6, 6b)
  equal their hand-off hashes; the stage-6 and 6b manifests match their tables; the tracked
  cleaning manifest is unmodified (`git status`). Evidence:
  `outputs/g1_audit/incident_2026-11-12/regenerated_vs_original_content.json`.
- *Prevention:* every entry point now parses its arguments before any work (`--help` runs
  nothing, an unknown or abbreviated option is an error): `solarpv.config.cli`, T-98
  (`tests/contracts/test_entry_points.py`).
- *Original:* not applicable.

**OP-61 — The reproduction from the raw data in a fresh checkout, and the G-1 audit (W6; no
stored file changed).**
- *Why:* gate G-1 asks that the pipeline runs from scratch in a clean environment and that the
  test split touched no decision (`PROJECT_MASTER.md` §7). The submitted reproduction started from
  a finished feature table and compared nothing (M32, M33).
- *Method:* `scripts/reproduce_paper_results.py --dev --with-eda --rerun-selection` clones this
  repository at its `HEAD` into an empty folder outside it, with the clone's `src` first on
  `PYTHONPATH` (X55). It is given only the raw files (read in place, verified against their
  checksums), the GFS point extracts of stage 6b and the private site file (removed at the end).
  It runs the stages of §19 in order, compares every written file with this checkout's by content
  (floating-point noise, relative at most `1e-9`, is reported and not failed; D-34), puts each
  stage back to the reference before the next, and then runs the two development runs of the
  runner. `scripts/g1_audit.py` measures the look-ahead of the data preparation (X57) and the
  test-split access.
- *Result (run journal R9-26–R9-30, `outputs/g1_audit/`):* 19 steps, every one exited 0, in 4 h
  37 min. Stage 2: 12.5 % of the values differ from the laptop's file by at most `9.7e-14`
  relative, because the laptop's parser was not correctly rounded (X63); stages 4, 5 and 6 and
  the GFS tables reproduce within floating-point noise; the feature selection (train part, 31
  files) and every forecast file of both development runs are identical in bytes or in content.
  The three decision checks run on the laptop give other numbers on this machine and the same
  decisions (X66). The look-ahead measures: the clock correction would change the whole-minute
  shift on 40 % of the days by at most 3 minutes, the cleaning changes 66 of 458,304 source cells
  (none a scored hour), the polled interpolation is bounded by 0.68 °C at night. No run has read
  or forecast a test hour (86 step records, 81 forecast files).
- *Verification:* T-93–T-99, T-72 (S09-34–S09-39); the full suite at the commit to be tagged.
- *Difference from the original:* the original reproduction started from the finished table.
- *Final reproduction at the candidate (13.11, run journal R9-35,
  `outputs/g1_audit/reproduction_20261113T120823Z/`):* commit `e262fe3`, 19 steps, every one exit 0,
  4.98 h. The forecast files of both development runs (the XGB run with `cv` and `ablation`, the
  neural run) are identical in bytes; the 7 files that fail the comparison are those D-34 explains
  (`decide_k` 3, `nwp_check` 2, the two report tables of the neural run). The two environments were
  the existing ones; nothing was installed.
- *Reproduction at the final candidate after the checkpoints (14.11, run journal R9-38,
  `outputs/g1_audit/reproduction_20261113T202349Z/`):* commit `ac63bb0`, 19 steps, every one exit 0,
  4.84 h; nothing installed. Every forecast file, best-parameter file and result table of both
  development runs is identical in bytes. 17 files fail the comparison and D-34 explains each: the
  same 5 of `decide_k` and `nwp_check` (X66); 3 report tables (the column `n_resumed` of D-38, X45);
  and 9 learning-curve and trial tables of TFT-96 and TFT-168 in which only the column
  `gpu_peak_mb` differs, by 0.00977 MB in the curves and 0.1 MB (the rounding) in the trial tables,
  at most 7.2e-5 relative (X99: the peak of the GPU memory moves with the timing of Python's
  garbage collector while the weights stay equal bit for bit,
  `outputs/g1_audit/gpu_peak_probe.txt`, T-120). No step resumed.

**OP-62 — The steps `cv` (S-38) and `ablation` (S-41) of D-36 (no stored file changed).**
- *Why:* v6 §12.2 (a 5-fold expanding-window CV against the test) and §12.5 (a feature-group
  ablation) are part of the approved scope and train models, so by D-36 they are written before
  the tag, as runner steps. The originals were scored on the test split (the ablation), could
  not be reproduced (the CV script was lost), used groups by substring that left the six rolling
  std features in the model, and read differences below the seed noise as findings (M37–M44).
- *Method:* `solarpv.pipeline.analyses`, after the `xgb` step of the same run. `cv`: the five
  expanding date folds of stage 7 (3-day gap) over the train rows, the model of each `xgb`
  record (searched parameters, seed, constraints, rounds, fixed), scored on the daytime hours with
  AC truth, op-limited hours included, in nRMSE of kW. `ablation`: the final fit of the `xgb`
  step repeated without one feature group of the contract at a time, without all measured
  history and with the time encodings only; early stopping on the validation rows; the raw
  forecast in kW; the delta against the `full` fit of the same seed; the verdict by
  max(0.05 pp, 2 × the pooled seed std). In a final run the test window is forecast (logged) and
  scored only by the step `test`. One fold loop for L-09 and the CV (`xgb_sota.oof_folds`).
- *Parameters:* folds 5, gap 3 days, fold warning below 500 held hours; margin floor 0.05 pp;
  seeds those of the `xgb` step (dev 42; final 42, 123, 777).
- *Result (development run `20261113T075845Z_dev`, one seed; stage-9 plan §21.7):* the mode `full`
  reproduced the `xgb` step in 55 of 55 fits; CV means 7.41–8.11 % against validation 6.43–7.12 %;
  removing GFS raises the error by 0.96–1.48 pp; the xgb outputs equal those of the W5 run byte
  for byte.
- *Verification:* T-100–T-103; smoke `20261113T075016Z_smoke`; figures S09-40 and S09-41.
- *Original:* v6 §12.2 (CV 16.08 % against test 29.09 %) and §12.5 (deltas on the test) are
  withdrawn (C-70–C-77); the rebuild's values are not comparable.

**OP-63 — The code changes after the external review of gate G-1 (W6; no stored data file changed).**
- *Why:* an independent reviewer read the source against the specification of §6 and, with three fix
  checks, wrote 26 comments (1 blocking, 7 should fix, 18 notes; `docs/g1_review_log.md`). The blocking
  one: every neural step of a final run scored the test split in phase 1, before the stage-10 acceptance
  existed (D-32). The should-fix rows: phase 1 wrote the test truth into the run folder; the phase-2
  command ran `report` before `test` (and a forced `report` failed on its own moved-aside folder); the
  acceptance bound to nothing; a retry of phase 2 scored the test split twice; a report older than a
  step it reports on passed; the promoted prediction files had a blank test `y_true`; `promote` did not
  compare what it promotes with what `test` scored (X77-X93).
- *Method:* `y_true` of the test rows is blank in the prediction files of phase 1 and the packed test
  arrays of a neural job hold zeros and NaN (`evaluation.evaluate.withhold_test_truth`); the neural
  steps score train and validation only; `--allow-test` runs `test` and `report` only, `test` first;
  `test` compares the SHA-256 of the files it scores with the inputs of the run's `evaluate`, refuses a
  step finished after the acceptance or a report older than a step, is never replaced by `--force`,
  writes `truth_test.csv`, and writes its benchmark after the SHAP and the analyses; `promote` fills the
  test `y_true` from `truth_test.csv` and refuses files that differ from those `test` scored; an unlisted
  completed step folder of the same run is adopted; the evaluation command line scores validation only;
  the alignment table has `compared_rows`. The audit script reads every kind of forecast file.
- *Parameters:* none changed (budgets, seeds, rules of the models and of the evaluation are as before).
- *Result:* no development output changed: four smoke runs at successive candidates, each compared with
  the previous one, were byte-identical in all 40 forecast files; the reproduction from the raw data at
  the final candidate is in OP-61 and `docs/run_journal.md` R9-35. No final run has been made.
- *Verification:* T-104 to T-112 (the regression tests fail on the old code: checked for T-104); the full
  suite at the final candidate `d126db8` (637 passed, 1 skipped); the audit over 22 runs (172 step records, 160
  forecast files, no test hour); figure S09-42.
- *Original:* v6 scored the test split in every run (C-37, C-40); no comparable protocol existed.

**OP-64 — Checkpoints and resume inside the long steps (W6; no stored data file changed).**
- *Why:* the user asked (13.11) that the code can continue after an interruption, with a checkpoint
  after every epoch and after every model run, although the college IT answered that a disconnected
  session keeps running, that there is no session limit and that the disk is not wiped. Until then
  the unit of `--resume` was the step: an interrupted TFT-168 (3.8 to 7.2 h) started again from its
  first line.
- *Method:* `solarpv.models.checkpoint` (docs/stage9_plan.md §23, D-38). A folder
  `<run folder>/checkpoints/<step>/`, written atomically and bound to the run, the step and its
  fingerprint (code commit, configuration, SHA-256 of every input; for the neural steps also the
  content digest of every target's arrays). The neural worker saves a fit's state about once a minute (at the end of the first epoch that ends 60 s or more after the last write) (model,
  Adam state, the order generator, the random states, the best epoch's weights, the curve), after
  every finished trial (the finished trials and the sampler) and after every finished target x
  seed; PatchTST after every target x seed. The `xgb` step saves after every finished trial and
  every finished target x horizon x seed. A study is replayed from its finished trials and its
  pickled sampler. A fit whose model has a multi-layer LSTM with dropout is brought to its
  checkpoint by running its training passes again (X95). The folder is deleted when the step
  succeeds; a step that was resumed says so in its record (`notes.resumed`) and its log.
- *Parameters:* none changed (budgets, seeds, search spaces, the loss, the early stopping and every
  output are as before, D-37). New: the seconds between two writes of a fit's state, 60 (R9-37,
  `SOLARPV_CHECKPOINT_INTERVAL_S`).
- *Result:* see the run journal R9-36 and R9-37: a fit continued from an epoch, and a worker killed at an
  epoch, a trial and the end of a target x seed and started again, give the forecasts, trials,
  curves and records of an uninterrupted one (bit for bit); the same for the xgb step.
- *Verification:* T-113 to T-119 (the GPU tests run in `solar-gpu`, T-117 and T-118).
- *Original:* none: the original project had no resume at all (M3, M5).

## 17. Differences from the original data pipeline

| Topic | Original | Rebuild | Ref. |
|---|---|---|---|
| Time base | local time labelled UTC | true UTC, DST decoded | OP-04, F-01 |
| Logger clocks | not checked | drift up to −16 min corrected, residual ≤ 2 min | OP-09, OP-15, F-15 |
| 2025 data source | 15-min snapshots used as minute data | detected, rebuilt at poll instants, own clock | OP-10–OP-14, F-18 |
| Night rule | wrong geometry, afternoon production zeroed | true geometry; only readings ≤ 5 kW below the horizon set to 0 (tiny twilight output; no night noise exists) | OP-21, F-01b |
| Spikes, sentinel, −270 | left in the data (flag only) | removed with their blended neighbours | OP-18, F-14 |
| Frozen values | not handled | removed (≥ 30 min repeated, reaching daytime, with a margin for blended minutes) | OP-20, OP-22 |
| Redundant irradiance sensor | ×0.75 calibration fault for 217+ days unnoticed (not a model input) | detected against clear-sky POA and DC; faulty days removed | OP-24, F-19 |
| Curtailment, stops | not handled; "13.5 % downtime" was a time-zone artefact | flagged, used per D-22; a drifting setpoint is not curtailment | OP-25, OP-26 |
| Partial inverter outages | not handled (learned as weather) | flagged (F3), used per D-22 | OP-27, F-22 |
| Quality flags as features | yes | blocked | OP-27, T-01 |
| Panel orientation | 30°/180° assumed | 21°/185° measured on the corrected data; the irradiance sensor at 20°/180° | OP-07, OP-32, F-13, F-23 |
| Gap filling | backward fill; past-day gaps from the expanding median of the whole column (nights included), then ffill/bfill | none in stages 2–5; past-day gaps from the same UTC hour over the previous 30 days, never from later values | OP-38, A-15 |
| Indirect target | k against GHI_cs, a mean of 15-min ratios, clipped at 1.2 | k = AC / theoretical power from POA_cs on the panel plane, hourly ratio of means, not clipped, exact inversion | OP-34, OP-35, F-06 |
| MEAN_DAYTIME_K | 0.6279 | 0.788157 (train, valid daytime hours) | OP-35, §6.9 |
| Rolling features | from 1 day back at every horizon (D+2/D+3 saw unknown days) | days h … h+N−1 back, DST guard | OP-37, F-07 |
| Past-day power in limited hours | used as is | masked per inverter (D-22) | OP-36, D-22 |
| Calendar | weekend Saturday–Sunday; solar noon = local hour − 12; daylight fraction approximated | weekend Friday–Saturday; solar noon from the transit; daylight fraction from sunrise and sunset | OP-39, Q-05 |
| Route comparison | each route on its own rows, filtered by its own truth; indirect truth rebuilt from the clipped k | identical rows for both routes, F-02 daytime, actual AC as truth | OP-41, C-34, C-16 |
| Rows scored per model | 2,510–2,580, by input window; daytime in the common mask taken from the truth | every model on every daytime hour with truth; daytime from geometry only | OP-41, C-35 |
| Skill reference | same time yesterday at every horizon | persistence of h days, DST-safe; climatology and clear sky added; CLIPER as an extra column | OP-40, C-08 |
| Weather forecast | Open-Meteo archive at the target time (not known at issuance), one series for all horizons, backward fill; removed from the final v6 models | GFS runs public before issuance (T-17, actual upload times), own leads per horizon, no backward fill; admitted on validation (D-12) | OP-42–OP-44, C-06, F-11 |
| Feature selection | whole table, validation and test included; per-inverter DC kept for the direct target; the selected sets never compared with the full pool | train only, from an allow-list; each set checked against the full pool inside train and once on validation, with the rule fixed in advance; the direct route keeps its full pool | OP-45–OP-48, F-26, F-27, F-29 |
| Test access | not controlled | validation by default; test only with an explicit flag, logged | OP-41, E10 |
| Model runs | one script; failed steps ignored; whatever files were in `outputs/` scored; logs overwritten; test evaluated in every run | one runner with smoke, dev and final modes; inputs checked against the stage manifests; each run in its own folder with hashes; only the run's own files scored; no test row loaded outside final | OP-49, C-36–C-41 |
| XGB training | `fillna(0)`; every train hour (nights, stops, invalid k); +1 constraints by substring on both targets; clips in the scoring scripts; saved parameters not the model | no imputation; fit masks per target (D-22, L-04); exact D-15 list, decided for k on validation; raw output, one conversion (`evaluate.to_kw`); full record, saved booster, explicit iteration range | OP-50, OP-51, C-20, C-33, C-42–C-46 |
| Bias correction | none | L-09 tested on validation by a registered rule; adopted for the indirect route only, refitted in every run on out-of-fold train predictions | OP-52, OP-53 |
| Sequence-model windows | naive local clock; BiLSTM: an origin every hour, one step kept per horizon (fixed leads of 24 / 48 / 72 h); TFT: steps 1–24 / 25–48 / 49–72 by number; windows inside each split, so each model scored on its own hours; no weather forecast | one origin per local day at T0; the decoder covers the three local days (71–73 h) and each step's horizon is its local day; origins from two days before the scope, so every hour is forecast at every horizon; D-22-masked encoder series; the known-at-T0 part of the stage-7 inputs, GFS from the run public at T0 | OP-55, C-47–C-52 |

## 18. Limitations carried into the report

- Absolute clock frame about ±2 min (depends on the sensor orientation and the clear-sky
  model); residual relative alignment ≤ 2 min (OP-09).
- Polled period (≈ 35 % of test): hourly means rest on four snapshots per tag, so cloudy hours
  are noisier; measured floor 1.62 % of capacity RMSE (OP-13). Evaluation will be reported
  overall, by source, and without polled hours (D-21). A minute-resolution export from the
  operator would remove this limitation (Q-12).
- Inverter state codes are undocumented; stops are detected from power and irradiance (OP-25),
  and only codes 5 and 6 are used, for derates, because the AC ratio confirms their meaning
  (OP-27).
- Output per unit irradiance declines by about 1 % per year over train while the irradiance
  sensor is stable against the clear-sky model (OP-31, Q29); no trend feature is used, so a small
  level bias of the direct target in test is possible.
- nRMSE of the direct target is normalised by 3,449 kW (as in the original); the observed peak is
  85 % of that (OP-31, Q23).
- Minute-source hours after the 2025 switch (a few hours, e.g. 2025-03-26, 2025-04-29) are
  corrected with the last estimated legacy offset (−15.9 min); a direct measurement on
  2025-04-29 gives −17.4 min, so these hours may be about 1.5 min off (within the ±2 min goal).
- The in-plane irradiance reads 10–20 % above the Ineichen clear-sky model at the clearest
  hours. k is defined against the model (OP-34), and k → kW uses the same model, so the
  model's scale does not bias any result in kW; it only sets the level of k (MEAN_DAYTIME_K
  0.788, OP-35).
- The panel plane 21°/185° is measured from the data (OP-32); external confirmation of the
  row orientation is open (Q-11).
- k is least stable at low sun: the 23 hours with k > 1.5 all have the sun below 10°
  (OP-34).
- Hourly clear-sky values cover the full hour while AC covers its valid minutes; 26 daytime
  hours have 46–59 AC minutes (OP-33).
- The test has 2,575 daytime hours with truth, not the 2,580 of the report: five daytime
  hours lost their AC to communication freezes (OP-41, A-09).
- GFS at 1° every 3 h, interpolated to hours through the clear-sky index; the hourly 0.25°
  files would cost about ten times the transfer (L-05, OP-42). The grid point is about 7 km
  from the site.
- Two runs were replaced by the 06Z run of the same day because a file had been re-uploaded
  after the issuance time (OP-42).
- Feature selection (OP-47, OP-48): the MI vote endorsed every starting feature, so in practice
  the consensus rests on the SHAP elbow and the stability vote; the 1 % share of the stability
  vote is relative, so a route with concentrated importance gets small sets (the direct route
  keeps its full pool for this reason); the 05c curve is one seed, and its 1 % tolerance is about
  one seed std of the score.
- The k history missed its D+1 margin by 0.002 pp (OP-46); the rule is applied as registered.
- For the direct route at D+1, inside train and validation disagree on the sign of the 05e
  gap (OP-48); the full pool follows the registered rule, and its cost on validation is
  0.068 pp.

## 19. Reproduction

```
python scripts/make_checksums.py --verify "Raw data"
python -m solarpv.data.ingest
python scripts/estimate_clock_drift.py        # writes configs/clock_offsets.csv (committed)
python -m solarpv.data.preprocess
python -m solarpv.data.clean
python -m solarpv.eda
python scripts/make_eda_notebook.py
python -m solarpv.eda.quality
python scripts/make_quality_notebook.py
python 00_data_audit/orientation_2026-10-20/orientation_check.py  # A-04 evidence; configs/panel_orientation.yaml is committed
python -m solarpv.features.build
python -m solarpv.features.decide_k     # writes configs/target.yaml (committed)
python -m solarpv.features.build        # again, with the D-11 decision
python scripts/make_features_notebook.py
python -m solarpv.nwp.download               # about 26 GB of transfer, resumable; --recheck
python -m solarpv.nwp.gfs
python -m solarpv.nwp.check                  # writes configs/nwp_decision.yaml (committed)
python -m solarpv.evaluation.evaluate        # validation; the test split is scored by the step `test` of a final run (D-32)
python -m solarpv.evaluation.evaluate --fixture
python -m solarpv.selection.run --skip-validation   # about 95 min on 8 threads; writes configs/k_history_decision.yaml
python -m solarpv.selection.run --validation-only   # the one validation look; writes model_inputs.json
python -m solarpv.pipeline check --mode dev
python -m solarpv.pipeline run --mode dev --only baselines,evaluate   # equals the stage-8B tables
python -m solarpv.models.xgb_checks d15      # writes configs/monotone_decision.yaml (committed)
python -m solarpv.pipeline run --mode dev --only baselines,xgb,evaluate
python -m solarpv.models.xgb_checks l09 --run <dev run id>   # writes configs/mos_decision.yaml
python -m solarpv.models.xgb_checks r14 --run <dev run id>
python -m solarpv.pipeline run --mode dev --only baselines,xgb,evaluate   # again, with L-09 (OP-53)
python -m solarpv.models.sequence_data       # the sequence datasets, train and validation (OP-55)
python -m solarpv.figures build              # the figure book, catalog and gallery (OP-56)
python scripts/build_data_record.py
pytest
```

## 20. Pending operations (documented here when performed)

Stage 9 sequence models (W4), on the datasets of OP-55, and the D-21 robustness check on val.
XGB (W2) is OP-50–OP-53; the kW statistics of the evaluation are OP-54; the sequence datasets
(W3) are OP-55.
