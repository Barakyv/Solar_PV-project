# Methodology changes since the submitted report (v6)

A section-by-section map from the submitted report
(`submission_evidence/13_report/Solar_PV_Forecasting_Report_CORRECTED_v6.docx`) and its evidence
package to the rebuild. It records:
- what stayed as submitted;
- what changed, and of which kind;
- why it changed, with a pointer to the evidence;
- what is still to be decided.

Written on 2026-10-21, after stage 6. Stages 0–6 and 8A are done. **No new model results exist
yet**, so the effects on the conclusions in §4 are expectations, not results.

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
| Models carried forward | XGB-Opt (Optuna 60 trials, seeds 42/123/777, monotone constraints); BiLSTM-Opt (encoder-decoder, quantile loss); PatchTST (univariate, rolling origins) | Same three models and budgets (D-08). The TFT models are not carried forward (§2.5) |
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
| Six models: XGB-Opt, BiLSTM-Opt, TFT-96, TFT-168, TFT baseline, PatchTST | XGB-Opt, BiLSTM-Opt, PatchTST, plus reference baselines | **Removal** (D-08) | The TFT models are not retrained. **This is the most visible change for a reader of v6** (see §5) |
| — | Baselines: persistence matched to the horizon, climatology, clear sky × MEAN_DAYTIME_K | Extension | §6.9. They anchor the skill score and the G-1 gate |
| Monotone constraints "+1 on ghi_clearsky, poa_global_clearsky, sun_elevation" | An explicit list (D-15). **Open: for which target?** | Correction + open question | The surviving ablation script applied +1 by substring, and to both targets. For the direct target it is physical. For k, the train data show the opposite marginal sign: Spearman −0.37 / −0.40 / −0.36 for GHI_cs, POA_cs and sun elevation. To be tested on validation in stage 9 |
| BiLSTM timestamp bug, fixed after the fact (§13.3) | Correct alignment from the start (T-02) | Correction kept | — |
| Final XGB fit: early stopping on validation | Unchanged unless D-10 (refit on train+val) is adopted | Open | D-10 is only proposed; v6 does not describe a refit |

### 2.6 Evaluation (§4.2, §6, chapters 7–12)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| DM "with HLN" | HLN actually applied; Bonferroni per target × horizon, as in the original `constants.py`; the uncorrected value also reported | Completion | The original code did not apply HLN (C-15, D-14, T-03) |
| Skill vs "same time yesterday" at every horizon | Persistence matched to the horizon (the latest value known at issuance) | Correction | At D+2/D+3, yesterday is not known at issuance (C-08) |
| Route comparison: indirect truth built from the clipped k | Actual AC is the truth for both routes | Correction | C-16, T-12. The 1.66–2.10 pp advantage must be recomputed |
| Ablation with leaky NWP in the base model | Recomputed on the leak-free feature set | Correction | C-07: the `minimal` set of the old script even kept `fcst_dni` and `fcst_ghi`. If GFS enters (§2.7), the `no_nwp` row becomes meaningful again |
| One evaluation over all daytime hours | The same main evaluation, plus secondary views: without op-limited hours, by data source, and without polled hours | Extension | D-21, D-22. The main number is defined as before |
| 5-fold expanding CV (§12.2): the script did not survive | Rewritten | Completion | Stage 11 |

### 2.7 NWP (§3.6, §3.7, §11.1, §13.1, Future Work)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| NWP from the Open-Meteo archive merged at the target hour; found to be leakage and removed; the final models have no weather forecast | GFS forecast runs from the NOAA archive, using only a run published before issuance. GFS enters only if it is leak-free (T-17) and improves validation (D-12) | Decision (to be tested, stage 6b) | **This is the report's first Future Work item**: "Replace same-timestamp reanalysis proxies with genuine T−24h NWP forecasts (e.g., ECMWF, GFS)". It also tests the report's own hypothesis in §13.1, that real NWP would widen the gap between feature-rich models and PatchTST. It is the same NWP model family as before (GFS), now taken at the right time |

### 2.8 Feature selection (§3.6)

| As submitted | Now | Kind | Why / evidence |
|---|---|---|---|
| The 05e consensus sets are "the official modelling inputs" | **Open (D-13).** The proposal makes the full contract the input and feature selection an analysis | Open decision | The evidence shows that the original XGB was trained on all 76 non-excluded columns: the SHAP file has 76 rows (F-05), and the ablation script uses a blocklist, not the 05e lists. So the report's description and the code already differed. See §5 |
| 05e JSONs contain the leaky NWP columns, excluded at training time | Regenerated on the corrected data, with GFS candidates and without leaky columns | Correction | C-09 |

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
| C1 ranking | Recomputed with three models. TFT-96, the 2nd place, is absent unless D-08 changes | D-08 |
| C2 route advantage 1.66–2.10 pp | Unknown; may shrink or grow | Old truth was the clipped k (C-16); k is now on POA and unclipped |
| C3 XGB flat across horizons | **Likely to change**: some degradation with the horizon is expected | The flat profile matches the rolling leak (F-07) |
| C4 PatchTST within 1.2 pp of XGB | Likely to widen if GFS adds skill | The report's own expectation (§13.1) |
| C5 TFT encoder length | Dropped | D-08 |
| C6 SHAP and TFT attention agree on −24/−48/−168 h | Dropped, unless TFT-96 is retrained; SHAP alone remains | D-08 |
| C7 comparisons underpowered | Likely still true; recomputed | Same test size |
| C8 errors highest at ramps (Q2) | Recomputed. The Q1 "+50 kW from inverter downtime" is now explained by the flags | D-22 |

## 5. Decisions that affect continuity (open)

1. **TFT (D-08).**
   - **What disappears without TFT:**
     - half of the model list;
     - the second-best model of v6;
     - §5.3–5.5 and §11.2, 11.3, 11.5;
     - conclusions C5 and C6;
     - two contributions of the introduction: 3, the "six-model benchmark", and 5,
       cross-architectural interpretability.
   - **What survives:** the title stays valid, since PatchTST is a transformer.
   - **Option:** retrain TFT-96 alone, with its v6 budget (20 Optuna trials, GPU). That keeps the
     interpretability thread. TFT-168 and the TFT baseline become a stated limitation.
2. **Feature selection as the modelling input (D-13).**
   - The report presents 05e as the official input and feature selection as a core contribution
     (§3.6, contribution 2).
   - If 05e is used as the input, the new pipeline will do what the report says. This follows the
     logic of D-14: implement as described.
   - Keeping the full contract is closer to what the old code did, but it changes the role of
     the largest methods section.
   - Either way, the errata must state that the old code used the full matrix.
3. **Literature extensions (§16b).**
   - Some ideas strengthen existing chapters without new structure: GFS (L-01–L-11), CLIPER skill
     as an extra reference (L-13), and bias diagnostics folded into §12.3 (L-15).
   - Others would add new model families (L-17: Extra Trees, MLP, physical baseline) or blending
     (L-18). Together with the removal of TFT, they would change the composition of the benchmark.
   - If tested, they belong in a clearly marked extension, not in the main benchmark.
4. **Monotone constraints on the k target (D-15):** test on validation whether to keep them for
   the indirect target.
5. **Refit on train+val (D-10):** not in v6. Keep the v6 practice unless there is a reason
   to change it.
6. **Informing the supervisor (Q-02):** the new numbers will differ substantially: the denominator
   of the indirect nRMSE, the route truth, the horizon profile. The plan (§11 of PROJECT_MASTER)
   is to inform her early, before any new number is shown.

## 6. How the changes can be presented in v7 without changing the study

- **Same structure:** keep the chapter structure, numbering, model names (XGB-Opt, BiLSTM-Opt,
  PatchTST), terms (clear-sky performance index, 05e/05f, midnight issuance, lag exclusion) and
  the figure set of v6.
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
  - **completions** of what the report described: DM with HLN, 05e as input if D-13 so decides;
  - **corrections:** the targets, the rolling statistics;
  - **the first Future Work item, carried out:** GFS.
