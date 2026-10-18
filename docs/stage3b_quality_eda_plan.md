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
| Q02 | When and where are values held (no new data)? | stage 4, all, descriptive | Monthly share of daytime minutes inside a run of ≥ 30 identical values (power: non-zero), per column (heatmap month × column) | R3; dead sensors (Q30) | file exists |
| Q03 | Can the two sampling regimes be told apart per tag-hour? | raw, all (format detection, not a model choice) | Distribution of the stage-4 detection statistics per tag-hour (share of minutes in 10–16 min runs, change rate), before and after 2025-02-26 | D-21 detection thresholds (0.5, 0.25) | < 2 % of classified tag-hours have a poll share in (0.3, 0.7) |
| Q04 | What do the value distributions look like before cleaning? | stage 4, all | Histograms (log count) of every ranged column with the policy limits; fault values listed | R1 limits | file exists |

### Part B: the evidence for each cleaning threshold (stage 4, train)

| ID | Question | Method | Supports | Check (T-31) |
|---|---|---|---|---|
| Q05 | Are the R1 limits placed in empty regions of the data? | Per ranged column: largest value inside the limits, smallest value outside, distinct fault values (−270 and its blends, 32,766, spikes) | R1 limits | power: largest in-range ≤ 0.9 × upper limit and smallest above ≥ 2 × upper limit; temperatures: smallest in-range value not next to a fault ≥ lower limit + 15 °C |
| Q06 | Are the neighbours of an R1 value contaminated? | For every R1 minute on the minute source: deviation of the minutes t−1 and t+1 from the next three minutes, against the column's normal 1-min change | R1 margin ±1 (minute) | ≥ 20 % of temperature-fault neighbours beyond the normal p99.9 change |
| Q07 | Is negative irradiance an offset or a fault? ¹ | Irradiance by sun-elevation band: share negative, minimum | R2 (−20…0 → 0) | minimum ≥ −20 W/m²; no negative value with the sun above 10° |
| Q08 | How long do values naturally stay constant? | Survival of daytime non-zero run lengths per column (share of daytime minutes in runs ≥ k, k = 2…240) | R3: 30 min, ambient 60 min | plateau: share(≥ 15) / share(≥ 60) ≤ 1.3 for every R3 column except the ambient temperatures, and share(≥ 30) / share(≥ 120) ≤ 1.3 for them |
| Q09 | Are long runs sensor faults or communication freezes? | Number of main columns frozen at the same minute; list of freeze events (start, end, columns, split) | R3 is a *communication* rule | ≥ 80 % of frozen daytime minute-columns have ≥ 5 main columns frozen at once |
| Q10 | What happens when data resume after a freeze? | For each frozen run on the minute source, is the first minute after it a blend (strictly between the stale value and the next value)? Example: 247 kW at sunrise on 2021-09-12 | R3 margin ±1 / ±15 | ≥ 50 % of train resumes are blends |
| Q11 | What is real at night and in twilight? | AC per inverter by sun-elevation band (< −6, −6…−3, −3…0, 0…5, 5…10): shares, max outside frozen runs | R4 (≤ 5 kW, sun < 0), R5 (> 5 kW, sun < −6) | below −6°, outside frozen runs, AC is exactly 0; ≥ 99 % of R5 minutes in or next to a freeze; ≥ 90 % of R4 candidates in twilight (−3…0°) ² |
| Q12 | Is DC consistent with AC? | Efficiency AC/DC (DC > 300 kW, daytime), also on steady minutes; minutes with DC = 0 and AC > 100 by month | R6 (DC = 0, AC > 100) | ≥ 95 % of inverter 1's R6 minutes in March 2023; efficiency median in [0.95, 1.0]; AC > DC on < 1 % of steady minutes |
| Q13 | Where does the pyranometer / irradiance ratio fall, by day and by hour? | Daily and hourly median ratio (sun > 30°, irradiance > 300, enough usable minutes); histogram, timeline, an intra-day example, ratio vs elevation | R7 band 0.9–1.1 per day **and per hour**; min elevation 30°, min reference 300 | faulty days at 0.75 ± 0.03, good days at 1.00 ± 0.01; at most one train day in the gap [0.80, 0.90) ∪ (1.10, 1.20] ² |
| Q14 | How does an inverter stop look in the data? | AC per inverter when sun > 5° and irradiance > 150 W/m²: histogram (log), AC vs irradiance | F1 (AC < 20 kW) | minutes with 20 ≤ AC < 100 kW are ≤ 0.5 % of those minutes (the threshold sits in a valley) |
| Q15 | Does the park setpoint really limit output, and does `iec_mode`? ² | Relative output (AC per irradiance over the monthly normal) under each setpoint; runs of non-integer setpoint values; relative output by `iec_mode` | F2 (setpoint < 100, not a drift), `iec_mode` not used | relative output < 0.9 at setpoint 50; ≥ 0.95 on the 2022 drift days; step changes ≤ 10 non-integer minutes, the drift longer; modes 0/1 within 5 % of mode 2 |
| Q16 | Does one inverter run derated, unseen by F1? ¹ | Windowed (31-min) median and IQR of log(AC1/AC2); AC ratio by status code and episode length; an example day | new flag F3 | valley share < 1 %; ≥ 50 % of low windows stable; code 5 sustained ≥ 30 min at ratio < 0.85, code 1 within 0.95–1.05; hours equal the stage-5 manifest |
| Q17 | Are the hourly thresholds sensitive? | Valid minutes per hour (coverage 45/60); share of flagged minutes per daytime hour, and number of op-limited hours for thresholds 0.01…0.5 | A-09 (45/60), D-22 (0.05) | hours with 1–44 valid target minutes ≤ 1 % of hours; op-limited hours at 0.02 and 0.10 within 10 % |

### Part C: after cleaning (stage 5 against stage 4)

| ID | Question | Data | Method | Check (T-31) |
|---|---|---|---|---|
| Q18 | What did each rule change, where and when? | all, descriptive | Minutes, % of split, daytime minutes, kWh by rule × column × split; monthly timeline of rule hits | sums equal `cleaning_log.csv` |
| Q19 | Did the distributions change only in the tails? | all | Quantiles before and after per cleaned column; overlaid histograms | after: no value outside the limits; p1, p50 and p99 of the daytime values move < 1 % of the 1–99 % range for the model-relevant columns ² |
| Q20 | Is energy conserved? | all | Daily energy before and after; days changed > 0.1 % with the responsible rule; total change by split | every day changed > 0.1 % is explained by R1 or R3 ² |
| Q21 | Where is data missing after cleaning? | all, descriptive | Monthly NaN share per cleaned column; hourly target NaN by split; longest gaps | file exists |
| Q22 | Is anything left? | all, descriptive (a finding goes back to stage 5) | Hourly clear-sky index > 1.2; daytime runs ≥ 30 min left; robust outliers of AC / irradiance in normal operation; inverter imbalance without a flag | 0 frozen daytime minutes left; ≥ 90 % of the hours above 1.2 at low sun; F3 explains ≥ 80 % of the imbalance hours |

### Part D: the cleaned data for modelling (train)

| ID | Question | Method | Supports | Check (T-31) |
|---|---|---|---|---|
| Q23 | What is the real ceiling of each inverter? | AC vs DC at the top; distribution of the per-day maximum; clipping plateau | A-08 (3,449 kW), interpretation of nRMSE | file exists |
| Q24 | Which temperature sensors agree, and which is reliable? | ambient_temp vs site_ambient_temp, mod_temp1 vs mod_temp2: differences, correlation, fault rates; physics: module − ambient vs irradiance | D-09 (temperature part, still open) | the ambient sensors correlate ≥ 0.95 |
| Q25 | How are the hourly variables related? | Spearman correlation, daytime hours | stage-7 feature screening | file exists |
| Q26 | How much does the same hour on past days tell? | Correlation of the target and of the hourly clear-sky index with lags 1–14 days | lag_power features (lags 1, 2, 3, 7, 14) | file exists |
| Q27 | How cloudy is it, by season? | Hourly clear-sky index distribution; share of clear / partly cloudy / overcast hours by season | D-11, error expectations | file exists |
| Q28 | Is there a weekday effect? | Daily clear-sky index by day of week; Kruskal–Wallis test | `is_weekend`, `day_of_week` in the contract | p > 0.05 (the note says "no effect") |
| Q29 | Is there a long-term trend (degradation, soiling, sensor drift)? | Monthly temperature-corrected performance index at 700–900 W/m²; clearest-hour plant output and sensor reading against the clear-sky model; slopes on train, val/test descriptive | A-16, need for a trend feature | sensor drift not significant (p > 0.05) |
| Q30 | Why are the other tags not used? | Per unused tag: the reason (measured at the target time → EXCL; dead sensor; constant) with its evidence | §6.7 EXCL, scope of cleaning | precipitation never > 0 |
| Q31 | How do the splits differ? *Descriptive only, used for no decision* | Months per split; daytime target and clear-sky index by split | reading of the test results | no summer hours in test |

¹ Added during the stage (see below). ² Check changed during the stage, with the reason below.

## Acceptance

- `python -m solarpv.eda.quality` produces every item above.
- The notebook runs end to end, with a markdown note after every code cell.
- T-31 is green.
- The data-processing record and PROJECT_MASTER.md are updated (OP entries, numbers section N10).
- Findings are recorded below and, if they change cleaning, fed back to stage 5 with tests.

## Changes made during the stage

The plan above is the version after implementation; the original is in git (commit `e16cd97`).

- **Q07 (R2) added.** The original plan had no analysis for rule R2; the numbering of the later
  analyses moved by one.
- **Q16 (F3) added** after finding F-22 (below); numbering moved by one more.
- **Q11:** the original check ("max AC ≤ 5 kW below −6°") holds more strictly (exactly 0); a
  check that R4's candidates are twilight readings was added, because the data contradict R4's
  stated reason (night noise).
- **Q13:** "no train day in the gap" became "at most one". One gap day was a freeze day judged on
  2 usable minutes (now excluded by the minimum-minutes rule); the other, 2021-11-05, is a day on
  which the fault switched within the day — the evidence for the hourly check (F-21).
- **Q15:** the planned check (output ≤ setpoint × capacity) was wrong for setpoint 0, where the
  plant keeps a residual of about 2.5 % of normal output; it was replaced by the relative output
  under each setpoint. This also exposed the 2022 drift (F-20).
- **Q19:** "< 1 % for every column" became "for the model-relevant columns": the pyranometer (R7)
  and DC of inverter 1 (R6) shift more by design, because those rules remove whole faulty
  populations.
- **Q20:** "total change < 0.5 % in every split" became "every changed day explained by R1 or R3":
  train loses 0.8 % because two long freezes held high values for many hours — energy that was
  never produced. Removing it is the intended effect.

## Findings made during stage 3b (fed back into stage 5)

- **F-20 — a drifting setpoint is not curtailment (Q15).** From 2022-02-09 to 2022-02-12 the park
  setpoint drifted through non-integer values (98 % → 28 %) for 4,309 minutes while output stayed
  normal. A commanded step shows at most 3 non-integer minutes. F2 now ignores runs of non-integer
  values longer than 10 minutes (`curtailment.max_noninteger_run_minutes`); train loses 1,808
  wrongly flagged minutes, val and test are unchanged.
- **F-21 — the pyranometer fault also switches within a day (Q13).** R7 now judges each UTC hour
  as well as each local day, with at least 20 usable minutes per period. A first version asked
  30 for a day and missed 2023-12-11, a freeze day with 24 usable minutes at ×0.75 (caught by
  T-30). The pyranometer is a QA sensor only; no model input changes.
- **F-22 — partial inverter outages (Q16).** One inverter sometimes runs at a stable ~75 % (once
  ~50 %) of its twin for hours or days, with status code 5 or 6. Nothing flagged these hours, so
  the models would have learned an equipment fault as weather. New flag F3 (values unchanged,
  D-22): status 5/6 for ≥ 30 minutes while producing, or a stable windowed AC ratio below 0.85.
  A first version without the stability and duration conditions flagged 400 / 92 / 374 hours
  (start-up episodes, clouds and 2025 polling phases); the final rule flags 245 / 43 / 93 daytime
  hours in train / val / test.
- **R4's justification corrected (Q11).** There is no night noise: below −6° the inverters read
  exactly 0. R4 acts on tiny twilight readings (238 kWh over 3¾ years); it is kept as a
  definition (target exactly 0 when the sun is down) and its comment in the policy is corrected.
- For later stages (no change to cleaning): the clear-sky index is unstable at low sun (D-11);
  winter-morning row shading; the observed peak is 85 % of the 3,449 kW capacity (A-08);
  recommendation to keep `ambient_temp` (D-09); weak lag information; no weekday effect; a decline
  of about 1 % per year with a stable sensor; no summer in test.
