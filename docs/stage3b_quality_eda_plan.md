# Stage 3b plan: data-quality EDA before and after cleaning

**Requested on 2026-10-16:** a full exploratory analysis of data quality before and after
cleaning. The analysis must give the evidence behind every cleaning threshold, and every
notebook cell must carry a markdown note that interprets it. This extends stage 3 (which
covers the corrections and the target) to **all 27 tags**, every cleaning rule, and the
properties of the cleaned data that matter for modelling. The plan was written before the
code.

## Rules

- **EDA changes no data.** If the analysis shows that a rule is wrong or missing, the change
  goes back to stage 5 with its own OP entry and tests (as R3-margin and R7 did in stage 3).
- **Thresholds are justified on train only.** Anything that sets or supports a threshold or a
  modelling choice is computed on the train split (UTC ≤ 2024-01-31 23:59) and labelled
  `train`.
- **Descriptive analyses may span the whole period** (`all`), for example rule hits per split
  or missing data over time, but they are never used to choose anything. One analysis (Q29)
  compares the splits and is marked *descriptive only*.
- **"Before cleaning" means the stage-4 output.** These are the clock-corrected true-UTC
  minutes with the polled data rebuilt, i.e. exactly what the cleaning rules see. The raw
  export (stage 2) is used where the export format itself is the subject (Q01, Q03).
- **Daytime** is the policy's sun elevation > 10° at the minute (true UTC), unless a rule
  defines its own (F1/F2: > 5°; evaluation: > 5° at mid-hour).
- **Numbers are never typed by hand.** Every analysis returns a table, a figure and a dict of
  facts, written to `outputs/03b_quality/quality_manifest.json`. The notebook's notes are
  formatted from that manifest, and T-31 checks that the claims the notes make still hold.

## Products

| Item | Path |
|---|---|
| code | `src/solarpv/eda/quality/` (`python -m solarpv.eda.quality`) |
| outputs | `outputs/03b_quality/{figures/qNN_*.png, tables/qNN_*.csv, quality_manifest.json}` |
| notebook | `notebooks/02b_data_quality.ipynb`, built and executed by `scripts/make_quality_notebook.py`. Each code cell runs one analysis on the data and is followed by a markdown note: what is shown, what it means, which rule or decision it supports |
| record | OP-30 in `docs/data_processing_record.md`, N10 in `docs/data_record_numbers.md` |
| test | T-31, `tests/contracts/test_quality_eda_output.py` |

## Analyses

Columns: **ID**; **question**; **data** (stage and scope); **method**; **supports**, the rule,
threshold or decision the analysis justifies; **check (T-31)**, the claim the note makes,
asserted on the manifest facts. "Main columns" are irradiance, pyranometer, the four power
columns, the two module temperatures and the two ambient temperatures.

### Part A: the export as delivered

| ID | Question | Data | Method | Supports | Check (T-31) |
|---|---|---|---|---|---|
| Q01 | What is in the data: units, roles, ranges, gaps? | raw + stage 4, all | Per column: group, unit, logger, role (target / model input via lags / QA only / unused, with reason), min, p1, p50, p99, max, distinct values, non-integer share of coded tags, NaN in raw and in stage 4 | the scope of cleaning (which columns get rules) | 27 tags + site total; raw has 0 NaN in every column (gaps are not NaN in the export) |
| Q02 | When and where are values held (no new data)? | stage 4, all, descriptive | Monthly share of daytime minutes inside a run of ≥ 30 identical non-zero values, per column (heatmap month × column) | R3; dead sensors (Q28) | file exists |
| Q03 | Can the two sampling regimes be told apart per tag-hour? | raw, all (format detection, not a model choice) | Distribution of the stage-4 detection statistics per tag-hour (share of minutes in 10–16 min runs, change rate), before and after 2025-02-26 | D-21 detection thresholds (0.5, 0.25) | < 2 % of classified tag-hours have a poll share in (0.3, 0.7) |
| Q04 | What do the value distributions look like before cleaning? | stage 4, all | Histograms (log count) of every ranged column with the policy limits; fault values listed | R1 limits | file exists |

### Part B: the evidence for each cleaning threshold (stage 4, train)

| ID | Question | Method | Supports | Check (T-31) |
|---|---|---|---|---|
| Q05 | Are the R1 limits placed in empty regions of the data? | Per ranged column: largest value inside the limits, smallest value outside, distinct fault values (−270 and its blends, 32,766, spikes) | R1 limits | power: largest in-range ≤ 0.9 × upper limit and smallest above ≥ 2 × upper limit; temperatures: smallest in-range ≥ lower limit + 15 °C |
| Q06 | Are the neighbours of an R1 value contaminated? | For every R1 minute on the minute source: deviation of the minutes t−1 and t+1 from the local median, against the column's normal 1-min change | R1 margin ±1 (minute) | share of neighbours deviating more than the normal p99.9 change is reported; claim made only if ≥ 20 % |
| Q07 | How long do values naturally stay constant? | Survival of daytime non-zero run lengths per column (share of daytime minutes in runs ≥ k, k = 5…120) | R3: 30 min, ambient 60 min | plateau: share(≥ 15) / share(≥ 60) ≤ 1.3 for every R3 column except the ambient temperatures, and share(≥ 30) / share(≥ 120) ≤ 1.3 for them |
| Q08 | Are long runs sensor faults or communication freezes? | Number of main columns frozen at the same minute; list of freeze events (start, end, columns, split) | R3 is a *communication* rule; margin | ≥ 80 % of frozen daytime minutes have ≥ 5 main columns frozen at once |
| Q09 | What happens when data resume after a freeze? | For each frozen run on the minute source, is the first minute after it a blend (strictly between the stale value and the next value)? Example: 247 kW at sunrise on 2021-09-12 | R3 margin ±1 / ±15 | share of resume minutes that are blends is reported |
| Q10 | What is real at night and in twilight? | AC per inverter by sun-elevation band (< −6, −6…−3, −3…0, 0…5): \|AC\| quantiles, max outside frozen runs | R4 (≤ 5 kW, sun < 0), R5 (> 5 kW, sun < −6) | below −6°, outside frozen runs, max AC ≤ 5 kW; twilight production up to tens of kW exists between −3° and 0° |
| Q11 | Is DC consistent with AC? | Efficiency AC/DC (DC > 100 kW, daytime); minutes with DC = 0 and AC > 100 by month | R6 (DC = 0, AC > 100) | ≥ 95 % of R6 minutes in March 2023; efficiency median in [0.95, 1.0] |
| Q12 | Where does the pyranometer / irradiance ratio fall, day by day? | Daily median ratio (sun > 30°, irradiance > 300); histogram, timeline, ratio vs elevation | R7 band 0.9–1.1; min elevation 30°, min reference 300 | train: no day in [0.80, 0.90) or (1.10, 1.20]; faulty days at 0.75 ± 0.03 |
| Q13 | How does an inverter stop look in the data? | AC per inverter when sun > 5° and irradiance > 150 W/m²: histogram (log), AC vs irradiance | F1 (AC < 20 kW) | minutes with 20 ≤ AC < 100 kW are ≤ 0.5 % of those minutes (the threshold sits in a valley) |
| Q14 | Does the park setpoint really limit output, and does `iec_mode`? | Site AC against setpoint × capacity when setpoint < 100; AC / irradiance by `iec_mode` | F2 (setpoint < 100), `iec_mode` not used | ≥ 99 % of setpoint-limited minutes at or below the cap (+2 %); output under modes 0/1 within 5 % of mode 2 |
| Q15 | Are the hourly thresholds sensitive? | Valid minutes per hour (coverage 45/60); share of flagged minutes per daytime hour, and number of op-limited hours for thresholds 0.01…0.5 | A-09 (45/60), D-22 (0.05) | hours with 1–44 valid target minutes ≤ 1 % of hours |

### Part C: after cleaning (stage 5 against stage 4)

| ID | Question | Data | Method | Check (T-31) |
|---|---|---|---|---|
| Q16 | What did each rule change, where and when? | all, descriptive | Minutes, % of column, daytime minutes, kWh by rule × column × split; monthly timeline of rule hits | sums equal `cleaning_log.csv` |
| Q17 | Did the distributions change only in the tails? | all | Quantiles before and after per cleaned column; overlaid histograms | after: no value outside the limits; p1, p50 and p99 of the daytime values change < 1 % |
| Q18 | Is energy conserved? | all | Daily energy before and after; days changed > 0.1 % with the responsible rule; total change by split | total daytime energy change < 0.5 % in every split |
| Q19 | Where is data missing after cleaning? | all, descriptive | Monthly NaN share per cleaned column; hourly target NaN by split; longest gaps | file exists |
| Q20 | Is anything left? | all, descriptive (a finding goes back to stage 5) | Hourly clear-sky index > 1.5; daytime non-zero runs ≥ 30 left; AC above the per-inverter maximum; robust outliers of AC / irradiance in normal operation | 0 frozen daytime runs ≥ 30 min left; remaining outliers listed |

### Part D: the cleaned data for modelling (train)

| ID | Question | Method | Supports | Check (T-31) |
|---|---|---|---|---|
| Q21 | What is the real ceiling of each inverter? | AC vs DC at the top; distribution of the per-day maximum; clipping plateau | A-08 (3,449 kW), interpretation of nRMSE | file exists |
| Q22 | Which temperature sensors agree, and which is reliable? | ambient_temp vs site_ambient_temp, mod_temp1 vs mod_temp2: differences, correlation, fault rates; physics: module − ambient vs irradiance | D-09 (temperature part, still open) | file exists |
| Q23 | How are the hourly variables related? | Spearman correlation, daytime hours | stage-7 feature screening | file exists |
| Q24 | How much does the same hour on past days tell? | Correlation of the target and of the hourly clear-sky index with lags 1–14 days | lag_power features (lags 1, 2, 3, 7, 14) | file exists |
| Q25 | How cloudy is it, by season? | Hourly clear-sky index distribution; share of clear / partly cloudy / overcast hours by season | D-11, error expectations | file exists |
| Q26 | Is there a weekday effect? | Daily clear-sky index by day of week; Kruskal–Wallis test | `is_weekend`, `day_of_week` in the contract | p-value reported |
| Q27 | Is there a long-term trend (degradation, soiling)? | Monthly temperature-corrected performance index at 700–900 W/m²; slope on train, val/test shown descriptively | A-16, need for a trend feature | file exists |
| Q28 | Why are the other tags not used? | Per unused tag: the reason (measured at the target time → EXCL; dead sensor; constant) with its evidence | §6.7 EXCL, scope of cleaning | precipitation never > 0 is stated only if true |
| Q29 | How do the splits differ? *Descriptive only, used for no decision* | Months per split; daytime target and clear-sky index by split | reading of the test results (no summer in test) | file exists |

## Acceptance

- `python -m solarpv.eda.quality` produces every item above.
- The notebook runs end to end, with a markdown note after every code cell.
- T-31 is green.
- OP-30 and N10 are written, and PROJECT_MASTER.md is updated.
- Findings are recorded below and, if they change cleaning, fed back to stage 5 with tests.

## Findings made during stage 3b

(filled in during the stage)
