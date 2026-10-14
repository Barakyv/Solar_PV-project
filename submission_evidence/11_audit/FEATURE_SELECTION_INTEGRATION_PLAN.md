# Feature Selection Integration Plan
**Solar PV Forecasting Capstone — Ktora Plant**
**Audit date:** 2026-05-11 | **Auditor:** Claude Sonnet 4.6

This document audits the depth of feature selection coverage in the main report and provides draft text for a stronger Feature Selection chapter.

---

## 1. What the Current Main Report Says

Based on the report extraction, the Feature Selection section covers:

- Four-stage pipeline: SHAP backward elimination → stability selection → mutual information → 2-of-3 consensus
- Tier definitions mentioned (CORE, SUPPORTED, CONDITIONAL, EXCLUDED)
- Operational availability classification (FORECAST_SAFE, HORIZON_DEPENDENT, PIPELINE_DEPENDENT)
- SHAP elbow threshold, stability floor, MI method
- Figures 19–25 showing the pipeline stages

**What is explicitly stated (from report extraction):**
- "Starting pool contained approximately 86 engineered variables"
- "Feature matrix of approximately 80 columns constructed"
- Solar geometry repeatedly selected; clear-sky irradiance stable
- Rolling statistics capture plant condition
- For direct target: DC power and recent AC lags especially stable

---

## 2. What Is Missing or Insufficiently Described

### 2.1 — Feature Counts Per Method and Horizon Are Missing
The main body does not state how many features each method selected. From the 05e JSON for indirect D+1:

| Method | Features selected |
|--------|-----------------|
| SHAP elbow (1.0% threshold) | 40 |
| Stability selection (threshold=0.60, 30 subsamples) | 19 |
| Mutual information (top-40) | 40 |
| 2-of-3 consensus | 40 |
| Unanimous (3-of-3) | 19 |

These counts should appear in the report to demonstrate the level of agreement across methods.

### 2.2 — 05e Feature JSON Contains Leakage Features (Not Disclosed in Main Body)
This is the most critical integration issue. The `final_features_indirect_D1.json` includes:
- `fcst_ghi`, `fcst_dni`, `fcst_dhi` (same-timestamp NWP reanalysis — leakage)
- `fcst_temp_2m`, `fcst_wind_speed` (same-timestamp NWP — leakage)
- `poa_global_forecast`, `poa_direct_forecast`, `poa_diffuse_forecast` (derived from leakage features)

These appear in the 05e recommended set with `"_leakage_fix_removed": []` — meaning the JSON was NOT updated after the NWP leakage was identified (April 2026). The XGB and LSTM training scripts exclude these via the EXCL list.

**The report does not explain this discrepancy.** Readers who look at the 05e JSON artifact will find leakage features in the "official modelling set." This must be disclosed.

### 2.3 — 05f Tier Counts Per Horizon Not Cited
From `refined_features_indirect_D1_05f.json`:
- CORE: 15 features (stability ≥ 0.75 AND MI ≥ 75th percentile)
- SUPPORTED: 8 features (stable but moderate MI, or vice versa)
- CONDITIONAL: 1 feature (`site_total_rolling_mean_3d_same_tod`)
- EXCLUDED: 18 features (stability < 0.30 floor)
- 05f recommended total: 24 features

The main report may reference these tiers but does not cite specific counts.

### 2.4 — 05f is Post-Hoc Only — Must Be Stated More Clearly
From the 05f JSON: *"05f is an audit/analysis artifact only. 06_modeling_v2 uses 05e feature sets exclusively. recommended_05f is available for interpretive discussion and sensitivity analysis in reporting, but does NOT change modeling inputs."*

This disclaimer is explicit in the artifact but must also appear verbatim (or equivalently) in the main report. The Feature Selection Report companion document should also be checked for this statement.

### 2.5 — Stability Selection Probabilities Not Cited
From 05e JSON, the strongest indirect D+1 features by stability probability:
- `poa_global_clearsky`: 1.00 (selected in all 30 bootstrap subsamples)
- `aoi`: 1.00
- `sun_azimuth`: 1.00
- `solar_azimuth`: 1.00
- `fcst_dni`: 1.00
- `hour_sin`: 1.00
- `doy_cos`: 1.00

These probabilities support the stability selection claim and could strengthen Figure 21 (stability selection probabilities).

### 2.6 — 05c Baseline nRMSE Not Cited in Main Body
From 05e JSON: `"baseline_nrmse": 16.0327` for indirect D+1. This is the full-feature validation nRMSE before any elimination (on validation fold). This value provides context for the SHAP elbow curve in Figure 20.

### 2.7 — Starting Pool Discrepancy (86 vs 42)
The report says "starting pool ~86 features." The 05e JSON shows `n_starting_features=42`. These are not necessarily contradictory:
- ~86 may be the full pre-Stage-1 feature pool
- 42 may be what entered Stage 1 filtering before the 05e consensus step
- The report must clarify WHEN in the pipeline the pool is 86 vs 42

### 2.8 — Permutation Importance Role Not Described
The 05e JSON includes a `permutation_importance` section with top-15 features. The 3-method consensus (SHAP + Stability + MI) does not include permutation importance in the voting, but it appears as a supplementary diagnostic. The report should clarify whether permutation importance influenced feature selection.

---

## 3. What Should Be Added

### 3.1 — Mandatory: 05e vs 05f Distinction (Report Body)
**Add to Feature Selection section body (after current pipeline description):**

> Stage 05e (consensus method) produced the official modelling feature sets. These sets, stored in `artifacts/05_feature_selection/final_features_{target}_D{h}.json`, were used as the input pool for all XGB-Opt and BiLSTM-Opt models. Feature sets for TFT-96, TFT-168, and PatchTST were specified separately based on architecture requirements.
>
> Separately, Stage 05f (`05f_feature_analysis.ipynb`) performed a post-hoc robustness audit of the 05e feature sets. The 05f analysis applied stricter criteria — a stability floor of 0.30 (features selected in fewer than 30% of bootstrap subsamples were excluded) and a mutual information threshold at the 75th percentile of the distribution — to tier features into CORE, SUPPORTED, CONDITIONAL, and EXCLUDED categories. The 05f analysis is an interpretability and sensitivity tool; it did not alter the modelling inputs. All final models were trained on the 05e feature sets.

### 3.2 — Mandatory: Leakage–Feature-Selection Interaction Disclosure
**Add after pipeline description:**

> The 05e feature selection was completed prior to the identification of NWP archive reanalysis leakage in April 2026. Consequently, the 05e JSON artifacts (`final_features_*.json`) include features subsequently classified as leakage: `fcst_ghi`, `fcst_dni`, `fcst_dhi`, `fcst_temp_2m`, `fcst_wind_speed`, and their plane-of-array derived counterparts. These features appear in the 05e recommended sets because at feature selection time they were treated as forecast inputs. They were excluded at model training time via the EXCL list in `production/02_models/xgb_sota.py` (lines 51–73). The effective modelling feature set is therefore the 05e consensus set minus the 19 leakage-identified features.

### 3.3 — Add: Feature Count Table by Selection Stage

**Add as Table X in Feature Selection section:**

| Stage | Input features | Output features | Criterion |
|-------|---------------|-----------------|-----------|
| Engineering pool | ~86 (full) | ~86 | All engineered features |
| Stage 1 domain filtering | ~86 | 42 | Remove constant, near-zero-variance, and manually flagged features |
| 05c SHAP backward elimination | 42 | 40 | Remove features with SHAP contribution < 1% (elbow threshold) |
| 05e SHAP elbow | 42 | 40 | 1.0% SHAP threshold |
| 05e Stability selection | 42 | 19 | Selection probability ≥ 0.60 across 30 bootstrap subsamples |
| 05e Mutual information | 42 | 40 | Top-40 by MI score |
| 05e Consensus (2-of-3) | 42 | 40 | Endorsed by ≥ 2 of 3 methods (indirect D+1) |
| 05f CORE | 40 | 15 | Stability ≥ 0.75 AND MI ≥ 75th percentile |
| 05f SUPPORTED | 40 | 8 | Moderate stability and/or MI |
| 05f CONDITIONAL | 40 | 1 | Borderline |
| 05f EXCLUDED | 40 | 18 | Stability < 0.30 floor |
| Effective model input | 40 | 40 minus leakage = ~31–33 | 05e minus EXCL list |

Note: Counts are for indirect D+1. Feature counts vary slightly by target and horizon.

### 3.4 — Add: Top Stability Selection Probabilities

**Add paragraph or sidebar:**

> Seven features achieved perfect stability (probability 1.00 across all 30 bootstrap subsamples) for the indirect D+1 target: plane-of-array clear-sky irradiance (`poa_global_clearsky`), angle of incidence (`aoi`), solar geometry angles (`sun_azimuth`, `solar_azimuth`), forecast direct normal irradiance (`fcst_dni`), and time-of-day encodings (`hour_sin`, `doy_cos`). These features represent the core of the deterministic solar-atmospheric signal available to the model.

### 3.5 — Add or Verify: Operational Availability Table

From the 05e JSON, features cluster into:
- **FORECAST_SAFE:** Clear-sky irradiance (pvlib-computed from geometry), sun geometry, time encodings, rolling production statistics (ending at T-24h)
- **HORIZON_DEPENDENT:** Lag features (lag_1d excluded for D+2, lag_1d+lag_2d for D+3)
- **PIPELINE_DEPENDENT:** NWP forecasts from actual forecast API (not archive reanalysis) — note these are DIFFERENT from the leaky fcst_* archive features

**Action:** Verify `feature_availability_classification.csv` exists and add count per category to the report.

---

## 4. Figures to Integrate or Strengthen

| Figure | Current status | Recommended action |
|--------|---------------|-------------------|
| Fig 19: Four-step pipeline diagram | Present | Verify stages labeled 05c, 05e, 05f correctly with outputs shown |
| Fig 20: SHAP backward elimination curves | Present | Add baseline nRMSE annotation (16.03% for indirect D+1) |
| Fig 21: Stability selection probabilities | Present | Add probability values for top-7 (1.00 each) in caption |
| Fig 22: Feature count evolution | Present | Align with corrected Stage 1 → 05c → 05e count table |
| Fig 23: MI vs Stability scatter | Present | Add 75th-percentile horizontal line with value annotation |
| Fig 24: Operational availability distribution | Present | Verify sourced from feature_availability_classification.csv |
| Fig 25: Sensitivity analysis of stability floor | Present | Confirm thresholds 0.25–0.40 match 05f notebook output |

---

## 5. Draft Text for Strengthened Feature Selection Chapter

### 5.1 Opening paragraph (add/replace)

> Feature selection was conducted through a four-stage transparent pipeline designed to produce operationally valid, stable feature sets independently of model training. The pipeline is shown in Figure 19.
>
> In Stage 05c, SHAP-guided backward elimination was applied to the full candidate pool of 42 features (after domain-knowledge pre-filtering from an initial pool of approximately 86 engineered variables). Features were removed in order of ascending SHAP importance until the elbow of the nRMSE curve was reached; for the indirect D+1 target, the 1.0% SHAP threshold retained 40 features from a baseline validation nRMSE of 16.03%.
>
> In Stage 05e, three independent selection methods were applied in parallel to the 40-feature pool:
> - *SHAP-guided elbow*: retained 40 features (1.0% threshold)
> - *Stability selection*: retained 19 features (selection probability ≥ 0.60 across 30 bootstrap subsamples of 70% of training data)
> - *Mutual information*: retained 40 features (top-40 by MI score)
>
> Features endorsed by at least 2 of 3 methods (the 2-of-3 consensus) formed the official modelling feature set: 40 features for the indirect D+1 target; feature counts vary by target and horizon. These sets are stored in `artifacts/05_feature_selection/final_features_{target}_D{h}.json` and represent the sole modelling input for XGB-Opt and BiLSTM-Opt.

### 5.2 05f section (add)

> **Post-Hoc Robustness Audit (Stage 05f)**
>
> After model training was complete, a post-hoc audit of the 05e feature sets was performed in Stage 05f to quantify the stability and reliability of each selected feature. Two additional criteria were applied: (i) a stability floor of 0.30 — features selected in fewer than 30% of bootstrap subsamples were marked as unstable — and (ii) a mutual information threshold at the 75th percentile of the MI distribution (0.6279 nats for indirect D+1).
>
> Features were tiered as follows for the indirect D+1 target:
> - *CORE* (15 features): stability ≥ 0.75 and MI ≥ 75th percentile; e.g., `poa_global_clearsky`, `aoi`, `sun_azimuth`, `fcst_dni`, `hour_cos`, `s01_inv1_rolling_mean_7d_same_tod`
> - *SUPPORTED* (8 features): moderate stability and/or MI; e.g., `ambient_temp_lag_1d`, `s01_inv2_rolling_mean_3d_same_tod`
> - *CONDITIONAL* (1 feature): `site_total_rolling_mean_3d_same_tod`
> - *EXCLUDED* (18 features): stability below the 0.30 floor or insufficient votes
>
> The 05f analysis identified 18 of 40 features as borderline or unstable. It is important to note that **Stage 05f did not change the modelling inputs**: all XGB-Opt and BiLSTM-Opt models were trained on the 05e consensus feature sets. The 05f tiers serve as an interpretability and sensitivity resource; researchers interested in a more conservative feature set can restrict attention to CORE + SUPPORTED features (23 features for indirect D+1), with the understanding that this represents a post-hoc audit rather than an optimized modelling decision.

### 5.3 Leakage–selection interaction (add)

> **Interaction Between Feature Selection and Leakage Audit**
>
> The 05c and 05e feature selection stages were completed prior to the identification of NWP archive reanalysis leakage in April 2026. Consequently, same-timestamp atmospheric reanalysis features (`fcst_ghi`, `fcst_dni`, `fcst_dhi`, `fcst_temp_2m`, `fcst_wind_speed`) and their plane-of-array derived counterparts appear in the 05e recommended sets. These features were selected by all three methods in 05e because at the time of feature selection they exhibited high predictive value — which is precisely what makes same-timestamp features so dangerous as leakage.
>
> These features were excluded at model training time via the EXCL list in `production/02_models/xgb_sota.py` (lines 51–73). The 05f robustness audit (`05f_feature_analysis.ipynb`) was conducted on the full 05e set including these features; they appear in the EXCLUDED tier of 05f for some combinations (where stability was artificially high only when the leaky feature was present) and CORE for others. Researchers should treat any 05f tier assignment for the nine leakage-identified features as illustrative only.
>
> The effective modelling feature set for XGB-Opt and BiLSTM-Opt is the 05e consensus set minus the 19 leakage-identified features, yielding approximately 31–33 operational features depending on target and horizon.
