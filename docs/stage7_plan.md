# Stage 7 plan: feature selection (05c, 05e, 05f) on the corrected data

Written on 2026-11-01, before the code. Decision D-13: the selection of the submission is rerun
on the corrected data, with the GFS candidates admitted in stage 6b (D-12); the 05e consensus sets
are the input of XGB-Opt for every target × horizon, as the report says (§3.6); 05f stays a
robustness audit; the full contract is compared once on validation, for an appendix. The input
of the sequence models is fixed in the stage-9 plan (L-12: select per model family).

Sources for what the original did: the submitted 05e and 05f files
(`submission_evidence/09_feature_selection/`), the report (§3.6, Figures 19–25), and the January
2026 base-selection notebooks of the backup, whose output is exactly the starting set of the
submitted D+1 files (`00_data_audit/original_backup_2026-11-01/`, F-26, F-27). The code of
05c, 05e and 05f did not survive anywhere; their method is rebuilt from the parameters recorded in
the submitted files and the report, and every definition that had to be chosen is listed in §3.

## 1. Facts checked at the start (2026-11-01)

| Item | Finding |
|---|---|
| Candidate pool per horizon | Contract 74 / 67 / 60 (stage 6) + 22 admitted candidates (21 GFS and `irradiance_clear_index_prevday`) = 96 / 89 / 82. Proposed for the indirect route only: k-history candidates (S0) |
| Selection rows (train, after warm-up, not op-limited) | Indirect: 9,575 hours with k valid and theo > 50 kW. Direct: 20,393 hours with AC (night included, as in training). Scored on 9,132 daytime hours. Train ends 2024-01-31 |
| Missing values in the pool | Only `irradiance_clear_index_prevday`: 6 (D+1) and 10 (D+3) train rows. XGBoost handles them; MI and correlation use complete rows |
| Base stage of the submission (January 2026) | Variance 0.001 and MI 0.02 (sklearn, 5 neighbours), with the core exempt; category minimums protect the best-MI features of under-represented categories; RFE with step 1 to 42 features on the non-protected features, ranked by XGBoost's built-in importance (100 trees, depth 5, learning rate 0.1), fitted once on all rows; the protected features are always kept. Run once, on the D+1 table; D+2 and D+3 started from the same list minus the unavailable lags |
| Core of the base stage (12) | `solar_zenith`, `solar_azimuth`, `aoi`, `fcst_ghi`, `fcst_dni`, `fcst_dhi`, `fcst_temp_2m`, `fcst_cloud_cover`, `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`. Exempt from both filters and kept through the RFE; free to drop in 05c and 05e |
| 05c (submitted files) | SHAP-guided backward elimination from the 42; elbows at 0.5 %, 1 %, 2 % and a second-derivative elbow; the 1 % elbow was used. The exact elbow rule is not recoverable (the recorded numbers fit no single rule); the report says the curve was scored on validation |
| 05e (submitted files) | Three votes: the 05c elbow set; stability selection, SHAP on 30 subsamples of 70 %, threshold 0.6; MI top-k with k equal to the size of the 05c set. Consensus: at least 2 of 3. Permutation importance on a validation fold was reported, not voted |
| 05f (submitted files) | Tiers by stability: CORE ≥ 0.75, SUPPORTED 0.50–0.75, CONDITIONAL 0.30–0.50 with MI above its 75th percentile, otherwise EXCLUDED (also with no vote). Floor sensitivity 0.25–0.40. Audit only: "does NOT change modelling inputs" |
| Availability classes (report §3.6.6) | FORECAST_SAFE (geometry, clear sky, time encodings, rolling statistics), HORIZON_DEPENDENT (lags), PIPELINE_DEPENDENT (NWP) |

## 2. Self-review before the code

### 2a. Mistakes of the original selection that must not come back

| # | In the original | Here | Test |
|---|---|---|---|
| P1 | The whole table was read: filters, RFE and SHAP saw validation and test (F-26). The 05c curve was scored on validation | Every selection statistic is computed on train rows only; the loader refuses any other split. Validation is opened once, in S8 | T-47 |
| P2 | The direct notebook kept the per-inverter DC power of the same hour (F-27); leaky `fcst_*`, `has_statistical_outlier` and `theoretical_power` were candidates; removed only afterwards | The pool is an allow-list (contract + admitted candidates + k-history if admitted) checked by `assert_no_leakage` before any step | T-47 |
| P3 | Rows with any missing value dropped in the RFE; median of the whole column or zero filled in other steps | No filling in stage 7: the stage-6 imputation is already past-only (F-24). XGBoost takes NaN; MI and correlation drop incomplete rows and report how many | T-48 |
| P4 | Quotas per category kept weak lags and rolling features in | No quotas (user, 2026-11-01). The core of the base stage, mapped to the new names, is kept through the base stage as in the original | T-48 |
| P5 | Category by substrings of the name | Groups from the contract (`group_of`) and the GFS dictionary; availability classes from those groups | T-48 |
| P6 | Every step measured something different (built-in importance, R² on all hours, an nRMSE of unknown units) | One score for every decision: nRMSE in kW, % of 3,449 kW, on the held-out daytime hours of time folds inside train; for k, the prediction × theo_kw | T-48 |
| P7 | RFE ranked by XGBoost's built-in importance, biased towards continuous features with many splits | Ranked by mean \|SHAP\| on the held-out folds (user, 2026-11-01) | T-48 |
| P8 | The MI vote took the top k with k = size of the SHAP set, so 2-of-3 was almost the SHAP set alone | A feature gets the MI vote only if its MI exceeds the 95th percentile of the MI of shuffled copies (user, 2026-11-01) | T-48 |
| P9 | Subsamples of the stability step of unknown construction; adjacent hours are near-duplicates | Subsamples of whole days (70 % of train days), SHAP on the held-out 30 % | T-48 |
| P10 | Base stage run once on D+1; D+2 and D+3 inherited its list | Run per target × horizon (user, 2026-11-01): the GFS candidates of each horizon come from different leads | T-49 |
| P11 | Time-series folds over the sorted rows, with no gap | Folds by date, an expanding window over train, a 3-day gap before each held-out block | T-48 |

### 2b. Review of my earlier work for this stage

| # | Point | Consequence |
|---|---|---|
| B1 | Stage 6b already used validation to admit GFS (D-12). Selecting on it again would use it a second time, and stage 9 tunes on it | Selection on train only; one validation comparison in S8 |
| B2 | The stage-6 note proposed testing k lags "on validation" | Tested inside train instead (S0), with the D-12 rule; consistent with B1 |
| B3 | In stage 7 of PROJECT_MASTER.md I had written the December parameters (MI 0.01, depth 6) and, in chat, that the core is protected only at the first filter | The January notebooks fed the submission: MI 0.02, depth 5, core exempt from the filters and kept through the RFE. Corrected here and in §7 |
| B4 | The GFS candidates are highly collinear (POA, kt_POA, physical power, k_phys); SHAP splits importance between duplicates (L-12, C2) | Duplicate removal at \|r\| > 0.98 (the December notebook's step) before the RFE; the member with the highest MI is kept, a core feature wins its group |
| B5 | `irradiance_clear_index_prevday` is not GFS but sits in the GFS tables | Its own group (`history_other`), availability HORIZON_DEPENDENT |
| B6 | The k-history candidates must follow the stage-6 rules exactly | Built with `past_value` (DST guard), `kept_lags` and `impute_same_hour` from `solarpv.features.build`, from k on valid, not op-limited hours |

## 3. Parameters and definitions (decision D-23)

Fixed before any result. "Original" = January base notebooks and the submitted files.

| Item | Value | Origin |
|---|---|---|
| Rows | train, not warm-up, not op-limited; indirect k_valid and theo > 50 kW; direct AC present | as in training (D-11, D-22) |
| Folds | 5 expanding folds by date over train (6 equal blocks of days), 3-day gap | P11 (deviation) |
| Score | nRMSE kW on held-out daytime hours with AC, pooled over the folds | P6 (deviation) |
| Selection model | XGBoost 100 trees, depth 5, learning rate 0.1, subsample 0.8, colsample 0.8, min_child_weight 3, gamma 0.1, alpha 0.1, lambda 1, seed 42 | original |
| Filters | variance < 0.001 removed; MI < 0.02 removed (5 neighbours, seed 42); core exempt | original |
| Core (11) | `solar_zenith`, `solar_azimuth`, `aoi`, `fcst_gfs_ghi`, `fcst_gfs_poa` (for DNI and DHI, which GFS does not give), `fcst_gfs_t2m`, `fcst_gfs_tcc`, `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` | original, mapped |
| Correlated pairs | \|Pearson r\| > 0.98 on selection rows, reported only (R7-01) | L-12 |
| RFE | step 1 to 42 features, mean \|SHAP\| on held-out folds; core kept | original target; ranking P7 (deviation) |
| 05c | backward elimination from the 42 by mean \|SHAP\|, refit each step; elbow = smallest set whose score is at most 1 % (relative) above that of the 42; 0.5 %, 2 % and the second-derivative elbows reported | original thresholds; rule written explicitly (deviation) |
| 05e stability | 30 subsamples of 70 % of train days; a feature is selected in a subsample when its share of the total mean \|SHAP\| on the held-out days is ≥ 1 %; vote at probability ≥ 0.6 | original counts and threshold; selection rule written explicitly (deviation) |
| 05e MI vote | MI > 95th percentile of the MI of 3 shuffled copies of every start feature (pooled null) | P8 (deviation) |
| Consensus | at least 2 of 3 votes | original |
| 05f | tiers CORE ≥ 0.75, SUPPORTED ≥ 0.50, CONDITIONAL ≥ 0.30 and MI > its 75th percentile, else EXCLUDED; floors 0.25–0.40 by 0.05 | original |
| Availability | FORECAST_SAFE: clear sky, geometry, time encodings, rolling statistics; HORIZON_DEPENDENT: lags, k history, yesterday's clear index; PIPELINE_DEPENDENT: GFS | original (§3.6.6) |

## 4. Steps

### S0. k-history candidates for the indirect route (decision D-24)
- Candidates: `target_k_lag_{d}d` for the lags kept at the horizon (1, 2, 3, 7, 14; minus 1 at D+2,
  minus 1 and 2 at D+3) and `target_k_rolling_mean_{3,7}d_same_tod` (days h … h+N−1 back),
  from k on hours with k valid, theo > 50 kW and not op-limited; DST guard and past-only
  imputation as in stage 6.
- Admission, inside train: the fixed XGBoost of D-11 (3 seeds) on the folds of §3, pool with
  and without the k history. The D-12 rule: the gain at D+1 exceeds max(0.05 pp, 2 × pooled seed
  std), and D+2 and D+3 are not worse by more than that margin. If admitted, they join the
  indirect pool; the report's `target_k_lag_1d` (C-32) is then true.

### S1. Pool
Contract(h) + the 22 admitted candidates (+ k history if admitted, indirect only).
`assert_no_leakage`; every name must be in the contract, the GFS dictionary or the k-history list.

### S2. Filters (base stage)
Variance, then MI, with the core exempt. Logged: what each removed.

### S3. Correlated pairs
Pairs of |r| > 0.98 reported, not removed (changed after R7-01, §7).

### S4. RFE to 42
One feature per step; the score curve is kept (Figure 22's base count).

### S5. 05c, SHAP backward elimination
Curve of the score against the number of features (Figure 20), elbows.

### S6. 05e, stability and MI votes, consensus
Selection probabilities (Figure 21), counts per method (Figure 22), MI against stability
(Figure 23).

### S7. 05f audit and availability
Tiers, floor sensitivity (Figure 25), availability classes (Figure 24),
`feature_availability_classification.csv`. The pipeline diagram (Figure 19) is drawn from the
counts.

### S8. One look at validation
The fixed XGBoost of D-11, 3 seeds, validation nRMSE kW: the 05e set against the full pool.
Reported in the appendix (D-13). If the 05e set is worse than the full pool by more than the
margin of the D-12 rule, this is a red flag, examined before stage 9; the 05e set stays the input
(D-13) unless the user decides otherwise.

### Outputs
`outputs/07_feature_selection/`: `final_features_{target}_D{h}.json` (05e, the schema of the
submitted files), `refined_features_{target}_D{h}_05f.json`,
`feature_availability_classification.csv`, the curves and votes as CSV, Figures 19–25, the
k-history decision (`configs/k_history_decision.yaml`), `selection_manifest.json`, run logs.
Notebook `05_feature_selection.ipynb`.

## 5. Tests (IDs continue PROJECT_MASTER.md §8.2)

| ID | What |
|---|---|
| T-47 | Selection rows are train only (the loader raises on any other split); the pool passes the allow-list and `assert_no_leakage`; a planted DC column is refused |
| T-48 | Synthetic: folds by date with the gap and no overlap; score in kW for both routes; duplicates keep the highest MI and the core; RFE never drops the core; the elbow rule; stability on whole days; a noise feature gets no MI vote and a signal does; consensus and tiers follow the table of §3 |
| T-49 | k history: DST guard, availability per horizon, past only (changing a later k changes nothing), the same imputation as stage 6 |
| T-50 | Outputs: 12 JSON files in the submitted schema; 05e ⊂ pool; no test row read; the S8 comparison present; the notebook current (T-46) |

## 6. Acceptance and red flags

Accepted when every combination runs end to end, the tests pass, the numbers are documented
(OP, N15, run journal) and the notebook is built.

Expected (L-12, D-12): GFS irradiance features near the top at every horizon; most lags and
calendar features drop; smaller sets than the original 40 / 23 / 19 (indirect) and 38 / 32 / 25
(direct).

Red flags:
- one feature dominating the SHAP ranking by a large factor (as the DC power did);
- no GFS irradiance feature left in a 05e set (contradicts D-12);
- the inner-train score falling with the horizon;
- the 05e set clearly worse than the full pool on validation (S8).

## 7. Changes during the stage

- **R7-01, the duplicate step removed (2026-11-01).** The first run removed one representative
  per connected group of |r| > 0.98. Groups chain (A ~ B and B ~ C put A and C in one group), and
  on daytime rows every same-time-of-day variable follows the diurnal shape: the rolling means of
  production went out as "duplicates" of `aoi`, and the power lags as duplicates of the
  irradiance lags (33–43 features per set). The direct 05e sets then lost 0.4–0.5 points on
  validation against the full pool. The January base stage, which fed the submission, had no
  such step: its RFE, one feature per step with a refit, removes redundant copies one at a time.
  The step now only reports the pairs above 0.98 (L-12, C2: checked, not removed); B4 is
  answered by the RFE. The first run's validation comparison is void; the rerun is the one look
  of S8, and the journal records both.
