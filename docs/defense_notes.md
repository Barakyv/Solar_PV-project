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

## 7. The neural models

The neural models were rebuilt in stage 9, W4 (`docs/stage9_plan.md` §18.2, findings M19-M24).
Their code did not survive, so the differences come from the submitted tables and report.

- **Submitted:** the BiLSTM seed spread rested on the training script's own metrics; the six
  models were scored on different hours; the TFT attention was read at a lag that TFT-96 does
  not have; one TFT version came from earlier code; the neural forecasts carried unexplained
  biases.
- **Why it looked right:** each table had the right columns and plausible values; a seed table
  with three rows looks like a seed study.
- **What revealed it:** comparing each table with the file it should agree with: the seed-42 row
  with the scored seed-42 forecast, the hour counts across models, an attention lag with the
  encoder length.
- **Rebuild:** one dataset per model with one forecast per hour and horizon; every seed scored by
  the run's own evaluation; one tuning objective; attention stored over each model's own encoder.
- **Figures:**
  - C-07 (`../outputs/figures/correction/C-07_bilstm_seed_table.png`), the seed table (C-10,
    X37): all three seed rows, seed 42 included, are the training script's metrics, about ten
    points below the one scored forecast at indirect D+1;
  - C-08 (`../outputs/figures/correction/C-08_scored_hours.png`), the scored hours (C-35,
    C-54): different hours per model in the submission, the same hours for every model and
    reference in the rebuild;
  - C-09 (`../outputs/figures/correction/C-09_attention_window.png`), the attention window
    (C-55): TFT-96 has no position at -168 h, and the rebuild's encoders rise at their first
    position whatever their length.
  - Process views: S09-06 (datasets), S09-09 (results on one set of hours), S09-10 (searches),
    S09-11 (attention), S09-12 (bias).
- **Items without their own figure:**
  - *One objective (C-53):* the report says every model was tuned on daytime nRMSE (§4.1) and,
    for the BiLSTM, on the validation quantile loss (§5.2). The rebuild tunes every model on the
    same daytime nRMSE in kW, with the quantile loss as the training loss; process view S09-10.
  - *The TFT baseline (C-54):* in the submission an earlier code version; in the rebuild the same
    step code and datasets as TFT-168, with the three reported values fixed.
  - *Hidden size 16 "always chosen" (C-56):* 16 is the smallest value of the space, so the result
    says the space may stop too early, not that small models are better. The runner warns about a
    best value on the edge of the space.
  - *Unexplained neural biases (C-57):* the Supplement shows NMBE from −6.0 % to +6.5 % among the
    neural models on the indirect route, normalised by a mean k of 0.6279 rather than the
    capacity, so it is not drawn beside the rebuild's bias in % of capacity. The rebuild has no fitted target transform and identity normalisers, and reports the
    bias of every model; process view S09-12.
- **IDs:** C-10, C-35, C-53-C-57, M19-M24, X37.
- **Likely question:** "Why are the neural results different from the report?" **Answer:** "The
  neural models were rebuilt on the same windows, hours, objective and scoring as XGB, so that a
  difference between families is a difference of the models."

## 8. Items without a figure

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
- **The neural models** have their own section (§7).

## 9. Found in the data, not handled in the submission

These are not mistakes of the submitted code but properties of the data that the rebuild found
by checking against a reference that should agree:

- logger clocks drifting up to 16 minutes, with two re-synchronisations (F-15; S04-01, S04-02);
- a second data source from 2025-02-26 with values held for 15 minutes (F-18; S04-03);
- partial inverter outages lasting days (F-22; S05-02);
- a pyranometer reading 0.75 of the truth for months (F-19), a QA sensor only;
- panels facing 185 degrees, not 180 (F-23).

## 10. The explanations of the models

The explanations were rebuilt in stage 9, W5 (`docs/stage9_plan.md` §19.2, findings M25-M31),
on the development runs; the report's chapter 11 is rewritten from the final run.

- **Submitted:** the SHAP groups of §11.1 came from a 76-feature XGBoost (C-31) and were led by
  rolling statistics that leak at D+2 and D+3 (F-07); the group shares summed the members'
  separate magnitudes, with names assigned by the first matching keyword (C-60); the rows and
  the explainer were not recorded (C-61); the TFT attention, smoothed and read by lag before
  issuance, was offered as validation of the SHAP lags, with a -168 h peak that TFT-96 cannot
  have (C-55, C-62); the hyper-parameter importances came from one unseeded evaluation of 20
  trials (C-63) and had no output file (C-64).
- **Why it looked right:** each table had plausible shares, and the agreement between two
  methods looks like a confirmation.
- **What revealed it:** the input of the explained model against the documented input, the
  feature groups against the leakage findings, the attention lag against the encoder length, and
  every claim against an output file.
- **Rebuild:** exact TreeSHAP of the booster sliced to its forecast rounds (the saved boosters
  hold 50 rounds more; explained whole they would be off by up to 199 kW in development, X46),
  on the scored validation hours, with additivity checked; group attribution as the mean of
  |sum over the group|; attention read per forecast step against the same hour of earlier days;
  fANOVA over ten seeds; every report number read from a hashed file.
- **What the rebuild finds (development runs):** the GFS forecasts lead the SHAP attribution in
  both routes; the TFT models show no preference for the hour they forecast (same-hour ratio
  0.82 to 1.07) but a preference for fixed clock hours of earlier days (noon for TFT-96). A
  preference for a fixed clock hour draws peaks every 24 hours in a profile by lag before
  issuance, like the one in v6; such peaks do not mean "the same time yesterday".
- **Figures:**
  - C-10 (`../outputs/figures/correction/C-10_shap_groups_v6.png`), the SHAP groups (C-59,
    C-60): rolling statistics led the submission; GFS forecasts lead the rebuild;
  - process views: S09-14 and S09-15 (SHAP), S09-16 (attention), S09-17 (importance).
- **IDs:** C-31, C-55, C-58-C-64, M25-M31, X46-X48.
- **Likely question:** "Why does your SHAP chapter differ from the report?" **Answer:** "The
  submitted SHAP explained a model with leaking inputs and 76 features; the rebuild explains the
  model that produced the forecasts, on its own inputs, and checks that the explanation adds up
  to the forecast."

## 11. The cross-validation and the ablation

Rebuilt in stage 9, W6, as the steps `cv` and `ablation` (`docs/stage9_plan.md` §21, findings
M37-M44, decision D-36). They train models, so they were written before the code was frozen.

- **Submitted:** the ablation refit the XGB-Opt without each feature group and scored the test
  split (M37); its "baseline" was not the XGB-Opt it was compared with (default parameters,
  constraints by substring, `fillna(0)`, a clip at 0; M38); the groups were substrings of the
  column names, so "no rolling statistics" left the six rolling-std features in the model (M39);
  the mode `minimal` was described as "just time encodings" but kept three other inputs (M40);
  deltas of -0.03 to +0.39 pp were read as findings although the seed spread is up to 0.11 pp
  (M41); the metric was its own (M42). The cross-validation script is lost, its folds were not
  described (M44), and a CV nRMSE of 16.08 % against a test value of 29.09 % was read as
  seasonality although the two numbers may not share a definition (M43).
- **Why it looked right:** each table had plausible numbers and a sentence explaining it.
- **What revealed it:** reading the ablation script against the objective and the evaluation
  layer, and the requirement that the fit without removal must equal the XGB-Opt.
- **Rebuild:** validation only (the test window is forecast in the final run and scored in
  phase 2, never used to decide); the mode `full` is the same fit as the `xgb` step and must
  reproduce its value to 1e-9 (it did, in 55 of 55 fits of the development run); groups from the
  feature contract, constraints by exact name, one kW metric; a delta counts only beyond
  max(0.05 pp, 2 x the pooled seed std), per seed against the same seed's `full`; a CV of five
  expanding date folds of the train rows with a 3-day gap, the folds' months written next to every
  mean.
- **What the rebuild finds (development run, one seed, verdicts provisional):** the fold means
  are 7.4 to 8.1 % (single folds range from 4.7 to about 11 %) and the validation value is 0.6 to 1.7
  points below the CV mean in every case: reported, not explained. Removing the GFS forecasts
  raises the error by 1.0 to 1.5 points, the largest change of any group; a model with the time
  encodings only is 1.2 to 1.5 points worse than the full one. Removing the clear-sky inputs of the
  indirect route never improves it by more than 0.001 points (v6 read an improvement off the test).
- **Figures:** process views S09-40 (the CV) and S09-41 (the ablation); both are XGB only, as in
  v6, and say so.
- **IDs:** M37-M44, C-70-C-77, X68-X75.
- **Likely question:** "Your CV error is half the report's: did the models get better?" **Answer:**
  "The two numbers do not share a definition: the report's script is lost and its metric differs;
  the rebuild measures one kW metric on the same hours everywhere and reports the gap to the
  validation value without explaining it."
