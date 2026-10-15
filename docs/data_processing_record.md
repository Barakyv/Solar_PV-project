# Data-processing record

Every operation performed on the data in the rebuild, in order, with its reason, method,
parameters, result, verification and evidence. Written to be lifted into the final report
(data chapter, "Reproduction and correction note", appendix). All numbers come from
`docs/data_record_numbers.md`, which is generated from the stage manifests (sections N1–N9).

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
| 5 | stage-4 minutes, cleaning policy | value rules, flags, hourly re-aggregation (OP-18–OP-27) | `data/processed/05_clean/`, `outputs/05_cleaning/` | `stage-05-passed` ² |
| 3 | stage-5 output, raw files, audits | figures and tables of the data chapter; findings fed back into stage 5 (OP-28) | `outputs/03_eda/`, `notebooks/02_eda.ipynb` | `stage-03-passed` |
| 8A | old LSTM predictions | evaluation-code check (OP-29) | `outputs/07_evaluation_fixture/` | `stage-08a-passed` |

¹ The stage-2 commit missed `src/solarpv/data/` because of an unanchored `.gitignore` pattern;
the ingestion code entered git with the stage-4 commit (`0b856c9`). The stage-2 output is
unchanged (same sha256, N1).
² Stage 5 was revised during stage 3 (R3 margin, new rule R7, OP-20 and OP-24); the current
cleaning is the one at `stage-03-passed`.

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
`00_data_audit/cleaning_2026-10-15/explore_cleaning.py` (thresholds from physics and the train
period). The test plan (`docs/stage5_test_plan.md`) was written before the code.

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
| OP-21 | R4 night zero | sun below 0° and 0 < \|power\| ≤ 5 kW | 0 |
| OP-22 | R5 night stale | sun below −6° and power > 5 kW (frozen values during communication outages) | NaN |
| OP-23 | R6 dead DC | DC = 0 while the same inverter delivers more than 100 kW AC (inverter 1, March 2023) | NaN |
| OP-24 | R7 scale fault | per local day, median pyranometer / cleaned irradiance (sun > 30°, irradiance > 300 W/m²) outside 0.9–1.1 → the whole day. Added after stage 3 found the pyranometer at exactly ×0.75 from 2023-05-28 to 2023-12-31 and on days in Dec 2024 – Apr 2025, while the irradiance sensor stays consistent with clear-sky POA and DC (F-19) | NaN |

Flag rules — real plant events, values never changed (D-22):

| ID | Flag | Condition | Use (D-22) |
|---|---|---|---|
| OP-25 | F1 `downtime_inv1/2_flag` | sun > 5°, in-plane irradiance > 150 W/m² (if missing: clear-sky POA > 300), inverter AC < 20 kW | out of the training loss; NaN in past-day features; included in the main evaluation, excluded in the secondary one |
| OP-26 | F2 `curtail_flag` | sun > 5° and park setpoint `iec_sp_w` < 100 %. `iec_mode` is not used: output under modes 0/1 matches mode 2 (F-17) | same |

**OP-27 — Hourly re-aggregation.** Same hourly rule as OP-17 on the cleaned minutes, plus per
hour: share of minutes with each flag (`frac_*`), `op_limited_flag` (≥ 5 % of the hour flagged),
`hour_source` (minute / polled / mixed, D-21) and valid-minute counts (`n_*`). Quality columns
can never be model inputs (T-01).

*Result (N7):* hourly maximum of site AC 2,948.4 kW (3,271.9 before; spikes removed);
op-limited daytime hours 193 (train), 106 (val), 106 (test); daytime hours with a missing
target 42 / 1 / 5, all from communication freezes. The pyranometer loses 403,409 minutes to R7
(it is not a model input). Real twilight production (up to ~29 kW with
the sun 0–3° below the horizon) is kept.

*Verification:*
- runtime self-check: no value changes without a rule code;
- T-22: 20 synthetic tests of every rule (including the R3 margin and R7), flag, precedence,
  idempotence and the hourly flags;
- T-23: on the real output, for every cleaned column, a value changed if and only if a rule
  code is set, the new value is the one the rule prescribes, the rule's condition holds on the
  stage-4 value, and no minute meeting a condition was missed — re-derived with independent
  code;
- T-16 (no daytime production > 5 kW ever becomes 0), T-24 (ranges), T-25 (flags never change
  values; flags match their definition), T-26 (hourly output), T-27 (daytime energy unchanged on
  fault-free days, ±0.1 %), T-28 (log and manifest equal the output; policy and file hashes);
- mutation check: 9 faults planted in a copy of the output, all 9 caught
  (`00_data_audit/cleaning_2026-10-15/mutation_check_output.txt`).

## 8. Exploratory analysis (stage 3, `solarpv.eda`)

**OP-28 — Figures and tables of the data chapter.** No data are changed. Plan:
`docs/stage3_eda_plan.md`; outputs: `outputs/03_eda/` (10 figures, 5 tables, manifest with input
hashes); narrative: `notebooks/02_eda.ipynb`. Statistics that could inform a modelling choice are
computed on train only. Results (N9):
- exact split sizes (train 21,195 h, val 5,832 h, test 5,829 h; the submitted report gave
  ~21,049 h for train); 2,580 test daytime hours, 106 of them op-limited, 1,001 polled;
- the time-zone bug measured on raw files: clear-day production centre +120.4 min (winter) and
  +173.6 min (summer) after solar noon under the original reading, +1.9 and +2.6 min after the
  rebuild;
- temperature coefficient of AC per unit irradiance −0.42 %/K (train, 700–900 W/m²);
- daily clear-sky index by month and its autocorrelation (0.33 at one day, ~0 from two days);
- the in-plane sensor reads 10–20 % above the Ineichen clear-sky POA at the clearest hours
  (scale question for D-11, stage 6).

*Fed back into stage 5:* the R3 margin (OP-20) and rule R7 (OP-24, F-19). *Noted as a
limitation:* minute-source hours after the 2025 switch use the last estimated legacy offset
(§11). *Verification:* T-30 (every planned output exists, tables agree with the stage-5
manifest and with physics, outputs made from the current inputs).

## 9. Evaluation-code check (stage 8A)

**OP-29 — Reproduction of the old LSTM metrics.** The surviving old LSTM predictions
(`submission_evidence/08_lstm_alignment/`) are evaluated with the new evaluation code: the
daytime mask (sun > 5° at mid-hour) reproduces n = 2,510 / 2,512 / 2,514, and nRMSE matches the
reported values within 1e-4 pp (indirect) and 0.0012 pp (direct) (N8, T-15). This proves that
differences in the corrected results come from data and models, not from a changed metric.
The old predictions are used only to test the code, never as a target.

## 10. Differences from the original data pipeline

| Topic | Original | Rebuild | Ref. |
|---|---|---|---|
| Time base | local time labelled UTC | true UTC, DST decoded | OP-04, F-01 |
| Logger clocks | not checked | drift up to −16 min corrected, residual ≤ 2 min | OP-09, OP-15, F-15 |
| 2025 data source | 15-min snapshots used as minute data | detected, rebuilt at poll instants, own clock | OP-10–OP-14, F-18 |
| Night rule | wrong geometry, afternoon production zeroed | true geometry, only ≤ 5 kW noise zeroed | OP-21, F-01b |
| Spikes, sentinel, −270 | left in the data (flag only) | removed with their blended neighbours | OP-18, F-14 |
| Frozen values | not handled | removed (≥ 30 min repeated, reaching daytime, with a margin for blended minutes) | OP-20, OP-22 |
| Redundant irradiance sensor | ×0.75 calibration fault for 217+ days unnoticed (not a model input) | detected against clear-sky POA and DC; faulty days removed | OP-24, F-19 |
| Curtailment, stops | not handled; "13.5 % downtime" was a time-zone artefact | flagged, used per D-22 | OP-25, OP-26 |
| Quality flags as features | yes | blocked | OP-27, T-01 |
| Panel tilt | 30° assumed | 20° measured | OP-07, F-13 |
| Gap filling | backward fill | none at this stage; no future information later | §6.6 |

## 11. Limitations carried into the report

- Absolute clock frame about ±2 min (depends on the sensor orientation and the clear-sky
  model); residual relative alignment ≤ 2 min (OP-09).
- Polled period (≈ 35 % of test): hourly means rest on four snapshots per tag, so cloudy hours
  are noisier; measured floor 1.62 % of capacity RMSE (OP-13). Evaluation will be reported
  overall, by source, and without polled hours (D-21). A minute-resolution export from the
  operator would remove this limitation (Q-12).
- Inverter state codes are undocumented; stops are detected from power and irradiance (OP-25).
- Minute-source hours after the 2025 switch (a few hours, e.g. 2025-03-26, 2025-04-29) are
  corrected with the last estimated legacy offset (−15.9 min); a direct measurement on
  2025-04-29 gives −17.4 min, so these hours may be about 1.5 min off (within the ±2 min goal).
- The in-plane irradiance reads 10–20 % above the Ineichen clear-sky model at the clearest
  hours; the clear-sky reference scale matters for the k target and is examined in stage 6.
- The array may face 2–3° west of the irradiance sensor; re-measured in stage 6 (A-04).

## 12. Reproduction

```
python scripts/make_checksums.py --verify "Raw data"
python -m solarpv.data.ingest
python scripts/estimate_clock_drift.py        # writes configs/clock_offsets.csv (committed)
python -m solarpv.data.preprocess
python -m solarpv.data.clean
python -m solarpv.eda
python scripts/make_eda_notebook.py
python scripts/build_data_record.py
pytest
```

## 13. Pending operations (documented here when performed)

Stage 6 target `k` (D-11), array orientation re-measurement (A-04),
features with D-22 masking; stage 6b GFS forecasts (D-12); stage 7 feature selection;
stage 9 models and the D-21 robustness check on val.
