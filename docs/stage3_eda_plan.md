# Stage 3 plan — exploratory data analysis

The submitted report has no EDA chapter: §3 holds one data table (with wrong claims: "two
inverters" as the plant, k ∈ [0, 1.2], ~21,049 train hours) and the split timeline (its
Figure 11); the original EDA notebook (02) did not survive. Stage 3 builds the data chapter of
the corrected report: it describes the data, shows each correction with evidence, and gives the
descriptive basis for the stage-6 decisions.

**Rules.** EDA changes no data. Anything that could inform a modelling choice (seasonal shape,
power–irradiance relation, variability, autocorrelation) is computed on **train only** and
labelled so; descriptive overviews of the whole period mark val and test. Code:
`solarpv.eda` (`python -m solarpv.eda`); outputs in `outputs/03_eda/` (figures, tables,
manifest); narrative in `notebooks/02_eda.ipynb`; operations in the data-processing record
(OP-28). Input: the stage-5 output, plus the raw files for the time-zone figure and the
stage-4/clock audit outputs.

| # | Figure / table | Shows | Data | Check (T-30) |
|---|---|---|---|---|
| T1 | `table_splits.csv` | hours, daytime hours, valid-target and op-limited daytime hours, polled hours per split (replaces report §3.1/§3.8 tables) | all | hours sum to 32,856; test daytime = 2,580; equals the cleaning manifest |
| T2 | `table_time_bug.csv` | clear-day production centroid vs solar noon when the raw local stamps are read as UTC (original) and as local time (rebuild) | raw, one winter and one summer train month | original ≈ +120 / +180 min; rebuild within ±5 min |
| F1 | `fig01_overview` | daily energy 2021–2025, split shading, polled period, clock steps, 30-day max power (no capacity change, A-16) | all | file exists |
| F2 | `fig02_time_bug` | diurnal profile of AC vs clear-sky POA, original vs rebuild (F-01) | raw, train months | file exists |
| F3 | `fig03_clock` | daily clock offsets of both loggers and the polled source, monthly estimates, residual after correction (F-15) | clock audit | file exists |
| F4 | `fig04_polled` | one hour of held 2025 data with the per-tag poll phases, its reconstruction, and the simulation result (F-18) | raw minutes, stage 4, simulation | file exists |
| F5 | `fig05_cleaning` | minutes per cleaning rule, a communication freeze before/after, daily energy before vs after (stage 5) | stages 4–5 | file exists |
| F6 | `fig06_diurnal` | mean AC by local hour × month | train | file exists |
| F7 + T3 | `fig07_power_irradiance`, `table_physics.csv` | hourly AC vs in-plane irradiance, coloured by module temperature; AC/irradiance vs module temperature → temperature coefficient | train | coefficient between −0.6 and −0.2 %/K |
| F8 + T4 | `fig08_variability`, `table_variability.csv` | daily clear-sky index by month; autocorrelation of the daily index, lags 1–14 days | train | file exists |
| F9 | `fig09_operations` | monthly daytime hours with inverter stops, curtailment and missing target, by split | all | file exists |
| F10 + T5 | `fig10_sensors`, `table_sensors.csv` | each in-plane sensor against clear-sky POA, and their ratio before and after cleaning. **Added during the stage:** this check found the pyranometer ×0.75 fault (F-19) → cleaning rule R7 | all | irradiance / clear-sky (p90) in [1.0, 1.3] every month; ratio after R7 in [0.9, 1.1] |

Acceptance: every item above produced by one command, numbers in the manifest, T-30 green,
an OP entry in the data-processing record, figures referenced in the notebook with interpretation.

## Findings made during stage 3 (fed back into stage 5)

- A 247 kW blend minute at sunrise after a communication freeze (fig05) → R3 gained a margin (±1 min minute source, ±15 min polled).
- The pyranometer read ×0.75 for 217 days in 2023 and on days in 2024-12 .. 2025-04 (fig10, F-19) → new rule R7; the irradiance sensor used by the models is consistent with clear-sky POA and DC.
- Minute-source hours after the 2025 switch use the last estimated legacy offset (−15.9 min); a direct measurement on 2025-04-29 gives −17.4 min (fig03) — about 1.5 min on a few hours, listed as a limitation.
