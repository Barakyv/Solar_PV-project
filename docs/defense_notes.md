# Defense notes: what changed from the submitted project, and why

A side report for preparing the defense. The report itself presents the method stage by stage
(the process figures of `docs/figure_catalog.md`) and keeps the short correction note and the
errata table of D-02. These notes are the backing for questions about the differences: for each
one, what the submitted project did, why it looked correct, what revealed it, what the rebuild
does, the evidence, and a short answer.

Every number here comes from a figure manifest (`outputs/figures/figures_manifest.json`) or from
the finding it cites in `PROJECT_MASTER.md` §2 and `docs/errata.md`.

## The one-paragraph answer

"I rebuilt the pipeline from the raw export and checked every step against an independent
reference: the position of the sun, the twin inverter, the twin sensor, and the forecasts that
survived from the submission. That is how the differences surfaced; none of them was visible in
the error metrics, because a consistent error in both train and test still gives plausible
scores. Each one is fixed, has an automatic test that fails if it comes back, and is listed in
the errata with its effect on the numbers."

## Why none of this showed in the results

1. **A consistent error does not break the metrics.** When train and test carry the same shift,
   the model learns it and the error stays in the range of the literature.
2. **Success messages checked that code ran, not that it was right.** A conversion step printed
   a check mark whether or not it converted; the reproduction script printed the report's
   numbers from constants (C-37).
3. **Partial guards felt complete.** A block-list, a midnight-issuance rule and a chronological
   split each covered part of the problem.
4. **Late patches instead of re-runs.** Leaky columns were removed before training, after they
   had shaped the feature selection.
5. **Plausible stories for anomalies.** Zero output was read as inverter faults; the cap at 1.2
   as a routine safety limit.
6. **A missing chain.** The first notebooks and the final XGB script were lost, and the report
   had no data-description chapter, so the one figure that shows the time shift was never drawn.

## 1. The time base

- **Submitted:** local time stamps were labelled UTC without conversion (notebook 03); the
  feature notebook converted only naive indexes, so it skipped the conversion.
- **Why it looked right:** the code says "ensure UTC" and the notebook printed a check mark after
  its conversion step; scores were plausible.
- **What revealed it:** the surviving forecast file holds, at the label 06:00 UTC, exactly the
  raw mean of 06:00 local, and the offset follows daylight-saving time; the production centre
  sits two to three hours after solar noon.
- **Rebuild:** conversion from Asia/Jerusalem at ingestion; a monthly check of the production
  centre against solar noon (T-05, T-08).
- **Figure:** C-01 (`../outputs/figures/correction/C-01_time_shift_profile.png`); process view
  S02-01, S02-03.
- **IDs:** F-01, C-12.
- **Likely question:** "How do you know the old data was shifted?" **Answer:** the clear-day
  production centre was 120 minutes (winter) and 174 minutes (summer) after solar noon; after the
  conversion it is 1.9 and 2.6 minutes.

## 2. The "inverter downtime" disclosure

- **Submitted:** 13.5 % of daytime hours had zero output and were reported as inverter downtime.
- **Why it looked right:** downtime at a desert plant is a plausible story.
- **What revealed it:** all 339 zero hours carry the labels 03:00-06:00, local hours before
  sunrise; there is no whole-day outage. A real fault does not happen at the same three hours
  every morning.
- **Rebuild:** daytime from the sun only; real stops, curtailment and partial outages flagged
  from the inverter data (F1, F2, F3, D-22).
- **Figure:** C-02 (`../outputs/figures/correction/C-02_false_downtime.png`); process view S05-02.
- **IDs:** F-01c, C-13.

## 3. The target and its cap

- **Submitted:** k was a mean of 15-minute ratios against horizontal clear-sky irradiance,
  clipped at 1.2; converting k back to kW did not return the measured power.
- **Why it looked right:** a cap on the clear-sky index looks like a routine safety step.
- **What revealed it:** 16.7 % of daytime test hours sit exactly at 1.2; a cap that touches one
  hour in six is changing the data. Multiplying k by the clear-sky power did not give AC back.
- **Rebuild:** hourly ratio of means against the panel-plane clear-sky power (chosen by three
  pre-registered criteria), no clipping, exact inversion (D-11, T-11); 0.61 % of train hours
  exceed 1.2, all at low sun.
- **Figure:** C-03 (`../outputs/figures/correction/C-03_k_clipping.png`); process view S06-01,
  S06-02, S06-03.
- **IDs:** F-06, C-29, C-16.

## 4. The multi-day averages

- **Submitted:** the 3- and 7-day same-hour averages started one day before the target day at
  every horizon, so at D+2 and D+3 they included days after the issuance midnight.
- **Why it looked right:** the code is correct for D+1, where it was written; a rule removed the
  one-day lag at longer horizons, but not the averages built from it.
- **What revealed it:** asking, feature by feature and horizon by horizon, "is this known at
  midnight?"; the nearly flat error across horizons was a hint, not a proof.
- **Rebuild:** windows from lag h back (F-07), with a test that plants the old window and must
  fail (T-09, T-18).
- **Figure:** C-04 (`../outputs/figures/correction/C-04_rolling_windows.png`); process view
  S06-04.
- **IDs:** F-07, C-17.

## 5. The feature selection

- **Submitted:** the base selection read the whole feature table, test included; in the direct
  route the same-hour DC power of each inverter was allowed and dominated the ranking, and was
  removed only at the end.
- **Why it looked right:** the selection used a time-series split; the final models were trained
  and scored on the right periods; the submitted files no longer list the DC columns.
- **What revealed it:** the January backup: the notebooks read the table without a split filter,
  and their lists equal the submitted ones; two DC columns hold 95 % of the importance, the first
  17 times the third feature.
- **Rebuild:** selection on train only, validation opened once, test never read; an allow-list
  instead of a block-list (D-23).
- **Figure:** C-05 (`../outputs/figures/correction/C-05_selection_leaks.png`); process view
  S07-01, S07-02.
- **IDs:** F-26, F-27, C-09, C-31.

## 6. The weather data

- **Submitted:** an archive series with the value of each target hour, not a forecast issued
  before midnight, joined through the shifted clock.
- **Why it looked right:** the request names a forecast model, and its accuracy was not
  suspiciously high; the leak is in timing, not in accuracy. It was partly caught in April 2026
  and removed before training.
- **What revealed it:** the cached request in the backup (archive end point, one continuous
  series); the correlation with the measured k is 0.63 after a 1-hour shift but 0.30 and 0.27 at
  the 2- and 3-hour shifts of the time bug.
- **Rebuild:** real GFS runs, each file's upload time checked against the issuance time (T-17);
  accepted on validation (D-12).
- **Figure:** C-06 (`../outputs/figures/correction/C-06_weather_alignment.png`); process view
  S6b-01, S6b-02, S6b-03.
- **IDs:** F-28, F-11, D-12.

## 7. Items without a figure

- **Significance tests (F-04, C-15):** the report describes HLN and Bonferroni corrections that
  the code did not apply; recomputing the test showed which formula matched. The rebuild applies
  both, checked against a hand calculation (T-03); process view S09-05.
- **Self-confirming reproduction (C-37, C-40):** the reproduction script printed the expected
  results from constants and scored the test period in every run. The runner reads its summary
  from the run's tables and never loads a test row outside the final run (T-52, T-53); process
  view S09-01.
- **Skill score (C-08):** the reported 90-96 % against persistence was recomputed at about 22 %
  on the same forecasts (F-10).
- **Plant description (F-12, C-30):** the data are 2 of the 25 inverters of the 40 MW park, not
  the whole plant; the report text is corrected in stage 11.
- **The evaluation code itself did not change (T-15):** on the surviving BiLSTM forecasts the new
  code reproduces the submitted nRMSE within 0.0012 points (S08-02), so differences in the
  results come from the data and the models.

## 8. Found in the data, not handled in the submission

These are not mistakes of the submitted code but properties of the data that the rebuild found
by checking against a reference that should agree:

- logger clocks drifting up to 16 minutes, with two re-synchronisations (F-15; S04-01, S04-02);
- a second data source from 2025-02-26 with values held for 15 minutes (F-18; S04-03);
- partial inverter outages lasting days (F-22; S05-02);
- a pyranometer reading 0.75 of the truth for months (F-19), a QA sensor only;
- panels facing 185 degrees, not 180 (F-23).
