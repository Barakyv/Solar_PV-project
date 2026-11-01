# Methodology changes since the submitted report (v6)

A section-by-section map from the submitted report
(`submission_evidence/13_report/Solar_PV_Forecasting_Report_CORRECTED_v6.docx`) and its evidence
package to the rebuild. It records:
- what stayed as submitted;
- what changed, and of which kind;
- why it changed, with a pointer to the evidence;
- what is still to be decided.

The scope itself, every v6 deliverable and the stage that produces it, is tracked in
`docs/scope_register.md`.

Written on 2026-10-21, after stage 6, and updated on 2026-10-25 after the user restored the
full approved scope. Stages 0–6 and 8A are done. **No new model results exist yet**, so the
effects on the conclusions in §4 are expectations, not results.

The detailed records are:
- `docs/errata.md`: every bug, its fix and its effect on the numbers;
- `docs/deviations.md`: accepted departures from the plan;
- `docs/data_processing_record.md`: every operation on the data (OP-xx);
- `PROJECT_MASTER.md`: findings F-xx, decisions D-xx and contradictions C-xx.

This document is the overview that links them to the report the supervisor knows.

## 1. The study is the same study

The rebuild reproduces the project at the level of its question, method and structure, not at the
level of its numbers (D-01). These elements are unchanged:

| Element | As submitted (v6) | Rebuild |
|---|---|---|
| Question and title | Day-ahead PV forecasting at one plant; comparison of gradient boosting, recurrent and transformer models | Same |
| Data | Ktora SCADA export, 1-min, 2021-09 to 2025-05 | The same 45 raw files, verified by checksum (OP-01, OP-02) |
| Resolution | Hourly means of 1-min data | Same |
| Splits | Train to 2024-01-31; val 2024-02 to 2024-09; test 2024-10 to 2025-05 | Same dates. Exact counts: 21,195 / 5,832 / 5,829 hours; the report gave ~21,049 / ~5,832 / ~5,832 |
| Horizons and issuance | D+1, D+2, D+3 issued at midnight | Same. Midnight is now defined explicitly as 00:00 local, in UTC (D-16) |
| Lag exclusion | lag_1d dropped at D+2; lag_1d and lag_2d at D+3 | Same rule (LAG_DROP) |
| Two targets | Indirect clear-sky performance index `target_k`; direct AC power in kW | Same two targets. The definition of k is corrected (§2.3) |
| Daytime mask | Solar elevation above 5°, 2,580 test daytime hours | Same mask. 2,580 reproduced exactly (F-02) |
| Primary metric | Daytime nRMSE; direct normalised by 3,449 kW | Same |
| Uncertainty | Block bootstrap, 24-h blocks, 1,000 resamples | Same |
| Significance | Diebold-Mariano with HLN correction, Newey-West with 24 lags | Same; now actually computed as described (§2.6) |
| Power analysis | Observed effect as % of the minimum detectable effect | Same (`detectable_effect` logic kept) |
| Models | XGB-Opt (60 trials, seeds 42/123/777, monotone constraints); BiLSTM-Opt (encoder-decoder, quantile loss, 60 trials); TFT-96 and TFT-168 (20 trials each); TFT baseline (fixed hyperparameters); PatchTST (univariate, rolling origins) | **All six, with the v6 budgets** (D-08, user decision 2026-10-25: the full approved scope), plus reference baselines. Every deliverable of v6 is tracked in `docs/scope_register.md` |
| Feature families | Clear sky, solar geometry, time encodings, lags (1, 2, 3, 7, 14 days), rolling 3 d and 7 d same time of day | The same families and names: 74 of the original 76 XGB inputs (F-05) |
| Feature selection | 05c SHAP elimination, 05e 2-of-3 consensus, 05f robustness audit | The same procedure, rerun on the corrected data (stage 7, D-13) |
| Analyses of chapters 7–12 | Benchmark, common mask, horizons, DM, power, seasonal, hourly, route, SHAP, seeds, CV, error decomposition, quintiles, ablation | All recomputed with the same definitions (stage 11) |
| Stage structure | Notebooks 01 ingest … 07 evaluation | The same numbering: NB01–NB08; NB02b and NB04b are added |

**The evaluation code is proven unchanged.** The rebuilt `evaluate.py` scores the surviving old
BiLSTM predictions:
- n = 2,510 / 2,512 / 2,514 reproduced exactly;
- nRMSE within 1e-4 pp (indirect) and 0.0012 pp (direct) of the report (T-15).

Every difference in the new numbers therefore comes from the data and the models, not from a
changed metric.

## 2. What changed, by report section

**Kinds of change:**
- **Correction:** the code did something other than what the report states, or what is right.
- **Data finding:** a property of the data that was unknown at submission.
- **Completion:** the report describes a step that the code did not do, or did only in part.
- **Decision:** a choice the report left open or made implicitly, now decided with criteria fixed
  in advance, on train or validation.
- **Removal:** something no longer done.
- **Extension:** new and optional; not part of the original design.

### 2.1 Plant and site (§3.1, §13.6)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| "Utility-scale installation, 3.449 MW, two inverters" | A 3.45 MW section (2 of 25 inverters) of the 40 MW Ketura Solar park | Data finding | Operator layout (F-12, C-30). Only the wording of §3.1 changes |
| Anonymised modelling coordinates, sensitivity < 2 % (§13.6) | Solar geometry at the park centre; the coordinates stay private | Correction | F-12, D-05, OP-06. The evaluation mask is identical (2,580 hours). §13.6 is rewritten; disclosure depends on Q-09 |
| Panels at tilt 30°, azimuth 180° (v4 default) | Tilt 21°, azimuth 185°, measured from the clock-corrected data | Data finding | F-13, F-23, OP-32: three DC methods agree. External confirmation is open (Q-11) |

### 2.2 Data processing (§3.2)

The report describes this stage in two sentences: resample to hourly means, compute clear sky. The
original project had more steps (an EDA notebook, a cleaning policy, `03_preprocessing_UNIFIED`),
but the report gave them little space. The 2026-05 audit `11_audit/MISSING_PROJECT_WORK_AUDIT.md`
says so: "Absent" / "Partial". The rebuild found that this stage carried the largest errors.

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| Timestamps treated as UTC | Local Israel time converted to true UTC, DST decoded | Correction | Production was shifted 2–3 h against solar geometry: the production centre was +120 / +174 min after solar noon (winter / summer), against +1.9 / +2.6 min now (F-01, OP-04, T-05, T-08). This changes every number |
| Logger clocks assumed correct | Two logger clocks drifting up to −16 min, corrected per day and per source | Data finding | F-15, D-19, OP-08–OP-09, OP-15. Residual ≤ 2 min |
| One 1-min source assumed | From 2025-02-26 the export holds 15-min snapshots; detected and rebuilt | Data finding | F-18, D-20, D-21, OP-10–OP-14. About 35 % of the test period |
| Cleaning: a policy existed, but the code did not survive and is barely described | An explicit policy with value rules R1–R7 and flags F1–F3; every change is logged and reconciled | Completion + Correction | F-14 (spikes up to 40,825 kW and a 32,766 sentinel had stayed in the target), F-19, OP-18–OP-28, T-22–T-28 |
| Inverter stops and curtailment not handled; "13.5 % downtime" (disclosure D1) | Stops, curtailment and partial outages are flagged, never erased. Used in training and in a secondary evaluation | Data finding + Decision | F-01c (the 13.5 % were pre-dawn hours created by the time bug), F-17, F-22, D-22. Op-limited daytime hours: 379 / 146 / 186 |
| Clear sky "for each hour" | The mean of the 60 minute values of each hour, from the true solar position | Correction | Matches hourly-mean data (OP-33, T-36) |
| — | A data-quality analysis before and after cleaning (31 analyses) | Completion | User request. It gives the evidence for every cleaning threshold (OP-31, notebook 02b) |

### 2.3 Targets (§3.3, §6.1)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| k = AC / (GHI_cs × 3,449/1,000), clipped to [0, 1.2] | k = AC / (POA_cs × 3,449/1,000) on the panel plane, never clipped | Correction + Decision | Clipping put 16.8 % of daytime test hours at exactly 1.2, mostly because of F-01 (F-06). The denominator was inconsistent in the original: the report states GHI_cs, notebook v4 used POA (C-27). It is now decided by three criteria fixed in advance (D-11): POA won 3 of 3. With POA, 0.61 % of train daytime hours exceed 1.2 |
| Hourly k = mean of 15-min ratios | Hourly k = ratio of the hourly means; k × theo = AC exactly | Correction | The mean of ratios could not be converted back to kW (F-06, T-11) |
| MEAN_DAYTIME_K = 0.6279 | 0.788157 (train, valid daytime hours) | Correction | Recomputed from the corrected k (C-28, OP-35). **The indirect nRMSE percentages are therefore not on the old scale:** the same RMSE in k units gives about 20 % lower nRMSE. The errata must say this before any comparison |

### 2.4 Features (§3.4, §3.7, §3.9)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| ~80 features (76 in the XGB input, F-05) | A contract of 74: the same names, minus `theoretical_power` and `sun_azimuth` | Correction | `theoretical_power` is on the leakage list but was an input (C-18); `sun_azimuth` duplicated `solar_azimuth` (Q-05) |
| Rolling 3 d / 7 d from 1 day back at every horizon | Rolling over days h … h+N−1 back for horizon h | Correction | At D+2 and D+3 the window included days not known at issuance (F-07). This is the likely reason for the flat horizon profile in §7.5 |
| — | DST guard: on the 25-hour day, a value from inside the forecast day is removed | Correction | 28 / 56 / 84 values at D+1 / D+2 / D+3 (OP-37, T-18) |
| Gaps: expanding median of the whole column, then ffill/bfill | The median of the same hour over the previous 30 days; never backward fill | Correction | The old fill used nights and later values (F-24, OP-38, T-34) |
| Weekend Saturday–Sunday; solar noon = local hour − 12; approximate day length | Friday–Saturday; solar noon from the solar transit; day length from sunrise and sunset | Correction | F-25, OP-39 |
| Lagged power used even in curtailed or stopped hours | Masked before lags and rolling | Decision | A curtailed hour is not weather (D-22, OP-36) |
| A quality flag (`has_statistical_outlier`) in the candidate pool | Quality columns can never be inputs | Correction | T-01 (F-17) |
| §3.4 lists `target_k_lag_1d` | No lag of k exists in the original 05e sets or in the 76 XGB inputs | Report text | The table does not match the evidence. Lags of k are proposed for stage 7, to be tested on validation; if adopted, the text becomes true |
| 19 leaky features excluded (count varies: 17 / 19 / 20) | One updated list (20) and an automated check | Completion | C-02, T-01. The rebuild also found leaks the original audit missed: rolling at D+2/D+3, `theoretical_power`, and the backward fill. It **extends contribution 1 of the report** (the anti-leakage framework) rather than replacing it |

### 2.5 Models (§4.1, §5)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| Six models: XGB-Opt, BiLSTM-Opt, TFT-96, TFT-168, TFT baseline, PatchTST | The same six models, with the v6 budgets | — | D-08 first excluded the TFTs (2026-10-14). The user restored the full approved scope on 2026-10-25 |
| Sequence models: decoder steps 1–24 / 25–48 / 49–72 taken as D+1 / D+2 / D+3 | Each step is assigned to a horizon by the local day of its valid time | Correction | On the 23- and 25-hour DST days, a fixed step count puts hours into the wrong day (D-16, T-02) |
| Final BiLSTM-Opt and TFT code (did not survive) | Rebuilt from §3.4, §5 and §6 of the report; inputs fixed in the stage-9 plan before training | Completion | R-16. Each gap is recorded in `docs/deviations.md` |
| — | Baselines: persistence matched to the horizon, climatology, clear sky × MEAN_DAYTIME_K | Extension | §6.9. They anchor the skill score and the G-1 gate |
| Monotone constraints "+1 on ghi_clearsky, poa_global_clearsky, sun_elevation" | An explicit list (D-15). **Open: for which target?** | Correction + open question | The surviving ablation script applied +1 by substring, and to both targets. For the direct target it is physical. For k, the train data show the opposite marginal sign: Spearman −0.37 / −0.40 / −0.36 for GHI_cs, POA_cs and sun elevation. To be tested on validation in stage 9 |
| BiLSTM timestamp bug, fixed after the fact (§13.3) | Correct alignment from the start (T-02) | Correction kept | — |
| Final XGB fit: early stopping on validation | Unchanged: no refit on train+val | — | D-10 (2026-10-25): v6 does not describe a refit, so it is not part of the approved scope |

### 2.6 Evaluation (§4.2, §6, chapters 7–12)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| DM "with HLN" | HLN actually applied; Bonferroni per target × horizon, as in the original `constants.py`; the uncorrected value also reported | Completion | The original code did not apply HLN (C-15, D-14, T-03) |
| Skill vs "same time yesterday" at every horizon | Persistence matched to the horizon (the latest value known at issuance) | Correction | At D+2/D+3, yesterday is not known at issuance (C-08) |
| Route comparison: indirect truth built from the clipped k | Actual AC is the truth for both routes | Correction | C-16, T-12. The 1.66–2.10 pp advantage must be recomputed |
| Ablation with leaky NWP in the base model | Recomputed on the leak-free feature set | Correction | C-07: the `minimal` set of the old script even kept `fcst_dni` and `fcst_ghi`. If GFS enters (§2.7), the `no_nwp` row becomes meaningful again |
| Route comparison rows: each route filtered by its own truth (k > 0.01, AC > 10 kW); 2,165 vs 2,172 (XGB) and 2,243 vs 2,520 (BiLSTM) rows, called "paired" in §10 | Identical rows for both routes, daytime from geometry | Correction | C-34; found in the stage-8B self-review of `compare_routes.py` |
| Models scored on different n (2,510–2,580); the common mask took daytime from the truth, over three models | Every model on every daytime hour with truth (2,575 on test); daytime from geometry; all models | Correction | C-35 |
| Test set reachable at any time | Validation by default; test only with an explicit flag, logged | Completion | Enforces "never use the test for decisions" (E10) |
| Persistence the only reference | Persistence (DST-safe), climatology and clear sky × MEAN_K; CLIPER skill and nRMSE by the mean as extra columns | Extension | §6.9, L-13, L-16; persistence stays the main skill reference |
| One evaluation over all daytime hours | The same main evaluation, plus secondary views: without op-limited hours, by data source, and without polled hours | Extension | D-21, D-22. The main number is defined as before |
| 5-fold expanding CV (§12.2): the script did not survive | Rewritten | Completion | Stage 11 |

### 2.7 NWP (§3.6, §3.7, §11.1, §13.1, Future Work)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| NWP from the Open-Meteo archive merged at the target hour; found to be leakage and removed; the final models have no weather forecast. (The backup shows the request: the archive endpoint with `gfs_seamless`, one series valued at the target hour, joined 2–3 h off the clock; F-28) | GFS forecast runs from the NOAA archive (1°), only runs whose every file was public before issuance (T-17, actual upload times); own leads per horizon. **Admitted (D-12, 2026-10-25):** with the fixed XGBoost of D-11, validation nRMSE in kW falls by 0.9–1.7 pp at every horizon and for both targets | Decision (done, stage 6b) | **This is the report's first Future Work item**: "Replace same-timestamp reanalysis proxies with genuine T−24h NWP forecasts (e.g., ECMWF, GFS)". It also tests the report's own hypothesis in §13.1, that real NWP would widen the gap between feature-rich models and PatchTST. It is the same NWP model family as before (GFS), now taken at the right time |

### 2.8 Feature selection (§3.6)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| The 05e consensus sets are "the official modelling inputs" | The 05e sets, regenerated, are the XGB-Opt input per target × horizon, as the report describes. The full contract is compared once on validation, in an appendix | Completion (D-13, 2026-10-25) | The original XGB was in fact trained on all 76 non-excluded columns: the SHAP file has 76 rows (F-05), and the ablation script uses a blocklist (C-31). The errata states this |
| 05e JSONs contain the leaky NWP columns, excluded at training time | Regenerated on the corrected data, with GFS candidates and without leaky columns | Correction | C-09 |
| The base selection (before 05c/05e) read the whole table, validation and test included; the direct one kept the per-inverter DC power of the same hour | Selection on train only, from an allow-list; same steps and thresholds as the January 2026 base notebooks that fed the submission | Correction | F-26, F-27; evidence from the backup (`00_data_audit/original_backup_2026-11-01/`) |

## 3. Report text that is factually wrong and must be corrected in v7

- §3.1: the plant description (C-30).
- §3.3: the k formula, the range [0, 1.2] and the value 0.6279 (F-06, D-11).
- §3.4: `target_k_lag_1d`, which the old inputs did not contain; and the feature count, given as
  60 / 80 / 78 in different places (C-19).
- §3.7: the leakage count (C-02).
- §3.8: the approximate split counts.
- §9.2: the "afternoon shoulder" explained by sea-breeze clouds. To be re-checked: under the time
  bug, local hours were shifted (C-14).
- §10: the mechanism text "range [0, 1.2]" and "normalisation by GHI_cs".
- §13.3: "~17 % before alignment", for which there is no evidence (C-22).
- §13.6: the coordinates.
- §15, D1: "13.5 % inverter downtime" (F-01c, C-13).
- §16 and the README: paths of the old `production/` code. A mapping old path → new path is needed.

## 4. Expected effect on the conclusions (to be confirmed in stage 11)

| Conclusion | Expectation | Reason |
|---|---|---|
| C1 ranking | Recomputed with the same six models; the order may change | All corrections |
| C2 route advantage 1.66–2.10 pp | Unknown; may shrink or grow | Old truth was the clipped k (C-16); k is now on POA and unclipped |
| C3 XGB flat across horizons | **Likely to change**: some degradation with the horizon is expected | The flat profile matches the rolling leak (F-07) |
| C4 PatchTST within 1.2 pp of XGB | Likely to widen if GFS adds skill | The report's own expectation (§13.1) |
| C5 TFT encoder length | Recomputed (TFT-96 vs TFT-168) | Same comparison, corrected data |
| C6 SHAP and TFT attention agree on −24/−48/−168 h | Recomputed. The lag pattern may differ: the rolling windows now follow the horizon, and the time base is corrected | F-01, F-07 |
| C7 comparisons underpowered | Likely still true; recomputed | Same test size |
| C8 errors highest at ramps (Q2) | Recomputed. The Q1 "+50 kW from inverter downtime" is now explained by the flags | D-22 |

## 5. Decisions that affect continuity

**Decided on 2026-10-25.** The user restored the full approved scope: "that was the scope of the
work, and that is what was approved".

1. **The six models (D-08).**
   - XGB-Opt, BiLSTM-Opt, TFT-96, TFT-168, TFT baseline and PatchTST are all retrained, with the
     v6 budgets.
   - Every deliverable of v6 (S-01–S-45, all 25 figures) is tracked in `docs/scope_register.md`.
   - T-37 fails if one of them leaves the plan.
2. **05e as the modelling input (D-13):** the new pipeline does what the report says. The errata
   records that the old code used the full matrix (C-31).
3. **No refit on train+val (D-10):** as in v6.
4. **Literature extensions:** only those that strengthen an existing chapter. Ideas that add model
   families or blending (L-17, L-18) are outside the approved scope; at most a marked appendix,
   after the whole register is done (§16b).

**Still open.**

5. **Monotone constraints on the k target (D-15, C-33):** test on validation in stage 9 whether to
   keep them for the indirect target. If they are dropped there, that is a correction and goes
   into the errata.
6. **Informing the supervisor (Q-02):** the new numbers will differ substantially: the denominator
   of the indirect nRMSE, the route truth, the horizon profile. The plan (§11 of PROJECT_MASTER)
   is to inform her early, before any new number is shown.

## 6. How the changes can be presented in v7 without changing the study

- **Same structure:** keep the chapter structure, numbering, the six model names (XGB-Opt,
  BiLSTM-Opt, TFT-96, TFT-168, TFT baseline, PatchTST), terms (clear-sky performance index, 05e/05f, midnight issuance, lag exclusion) and
  the 25 figures of v6 (`docs/scope_register.md`).
- **Up front:** a one-page "Reproduction and correction note". It explains that the code was
  rebuilt from the raw data, lists the bugs found (time base, k clipping, rolling leakage, DM
  correction, route truth), and states that the evaluation code reproduces the old metrics (T-15).
- **Appendix:** an errata table with old and new values and the reason for each difference
  (`docs/errata.md`). It also separates the change of scale in the indirect nRMSE (MEAN_DAYTIME_K)
  from changes in skill.
- **Data processing:** keep §3.2 short, with one paragraph per new step (time base, clocks,
  second source, cleaning, operational flags). Put the details in a data appendix built from
  `docs/data_processing_record.md`.
- **New elements:** label them as what they are:
  - **completions** of what the report described: DM with HLN, 05e as the modelling input (D-13);
  - **corrections:** the targets, the rolling statistics;
  - **the first Future Work item, carried out:** GFS.
