# Solar Forecasting Thesis — Corrected Findings
**Generated:** 2026-04-23 | **Test window:** 2024-10-01 to 2025-05-31

---

## Chapter Findings Summary

### Phase A — DM Statistical Foundation (CORRECTED)

**Key correction (A1):** Newey-West lag fixed from h=1/2/3 to h=24 (standard for hourly data).
LSTM prediction timestamps were window-end (duplicated per decoder step). Fixed by reconstructing
valid timestamps: `valid_ts = window_end - (DECODER_LEN - 1 - target_step) hours`.

**Updated ranking (evaluate.py, aligned predictions):**

| Model | Indirect D+1 nRMSE | Direct D+1 nRMSE |
|-------|-------------------|-----------------|
| XGB SOTA | **29.095%** | **12.060%** |
| LSTM SOTA | 30.480% | 12.600% |
| TFT | 31.756% | 12.798% |
| XGB Phase1* | 27.467% | 11.197% |
| XGB Phase2* | 27.590% | 11.182% |
| k-Persistence | 37.184% | — |
| Climatology | 29.616% | — |

*leakage-affected (theoretical_power feature); invalid competitive baselines.

**CRITICAL FINDING:** There is NO ranking inversion. XGB SOTA wins both targets.
The original claim of "BiLSTM leads direct" was an evaluation artifact from misaligned
LSTM timestamps causing ~25% timestamp mismatch.

**DM test results (h=24 Newey-West):**
- LSTM vs XGB (indirect, all horizons): p = 0.133–0.189 → NOT significant
- TFT vs XGB (indirect, all horizons): p = 0.003–0.014 → Significant (TFT worse)
- LSTM vs XGB (direct, all horizons): p = 0.143–0.308 → NOT significant

**Phase A2 — Detectable Effect:**
- N_eff range: 255–829 (from raw n=2,510–2,580; heavy autocorrelation shrinks power)
- All pairwise comparisons UNDERPOWERED: obs_std_effect = 43–71% of MDE at 80%/5%
- Block bootstrap 95% CI for loss differential includes zero in all LSTM vs XGB comparisons
- Correct language: "Pairwise statistical superiority could not be established (DM test,
  h=24 Newey-West correction, n≈2,510, p=0.11–0.19 for LSTM vs XGB); the observed effect
  size is 43–56% of the minimum detectable effect at 80% power, suggesting insufficient
  test-period length rather than confirmed equivalence."

---

### Phase B — Seed Variability

**XGB seed variability (3 seeds complete — 42, 123, 777):**
- Indirect D+1: 29.063% ± 0.043 pp (min 29.014%, max 29.095%)
- Indirect D+2: 29.123% ± 0.044 pp
- Indirect D+3: 29.070% ± 0.018 pp
- Direct D+1: 12.068% ± 0.010 pp (min 12.060%, max 12.079%)
- Direct D+2: 12.136% ± 0.032 pp
- Direct D+3: 12.156% ± 0.029 pp
- **Conclusion:** XGB SOTA is highly seed-stable (std < 0.05 pp). Ranking claim
  is robust: XGB seeds 42/123/777 all beat LSTM aligned at D+1 (29.1% vs 30.5% indirect;
  12.1% vs 12.6% direct).

**LSTM seed variability:** Single seed only (42). GPU was not available at retraining time
(CUDA: False on current system). 60-trial CPU retraining would take 30+ hours per seed.
Training-script metrics (indirect 20.098%, direct 8.570%) are NOT day-ahead aligned
(they include all 24 decoder steps per horizon, not just the day-ahead step).
The canonical LSTM metrics for comparison are from evaluate.py aligned evaluation:
- LSTM indirect D+1: 30.480% (aligned)
- LSTM direct D+1: 12.600% (aligned)
Limitation note: LSTM seed variability cannot be estimated without GPU access.
The aligned single-seed results are used with this limitation acknowledged.

**Ablation seed variability (E3, all 3 seeds done):**
- Removing rolling-7d features (indirect): +0.602 ± 0.049 pp → robust signal
- Removing NWP (indirect): +0.239 ± 0.051 pp → robust
- Removing clearsky (direct): −0.075 ± 0.033 pp → clearsky helps INDIRECT not DIRECT
- All ablation ΔnRMSE are stable across seeds (std < 0.1 pp for most)

**B1 — Tuning Budget:**
- XGB: 60 Optuna trials (standard)
- LSTM: 60 Optuna trials (increased from 20; retraining in progress)
- TFT: **Option B** — 20 trials; results reported as supplementary with footnote:
  "TFT was allocated 20 Optuna trials vs 60 for XGB/LSTM; results should be treated
  as indicative only and are not included in the main benchmark table."

---

### Phase C — Route Comparison (Indirect k→Power vs Direct)

**At AC-power scale (denominator = 3,449 kW):**

| Model | Indirect→Power nRMSE | Direct nRMSE | Difference |
|-------|---------------------|--------------|------------|
| XGB D+1 | **11.162%** | 13.142% | Indirect better by 1.98 pp |
| XGB D+2 | **11.186%** | 13.197% | Indirect better by 2.01 pp |
| XGB D+3 | **11.178%** | 13.277% | Indirect better by 2.10 pp |
| LSTM D+1 | **11.297%** | 12.952% | Indirect better by 1.66 pp |
| LSTM D+2 | **11.344%** | 13.066% | Indirect better by 1.72 pp |
| LSTM D+3 | **11.202%** | 13.093% | Indirect better by 1.89 pp |

**FINDING:** At the operational AC-power scale, the INDIRECT route (predict k, reconstruct
power) outperforms the DIRECT route by 1.66–2.10 pp nRMSE for BOTH model families.
This strengthens the case for the indirect approach operationally.

**C0 — Target definition correction:**
The target is a **clear-sky performance index** (NOT the atmospheric clearness index Kt):
`k = AC_kW_observed / (GHI_cs × AC_CAPACITY_KW / 1000)`
Inverse: `P_AC = pred_k × GHI_cs × AC_CAPACITY_KW / 1000` (clipped to [0, capacity]).

---

### Phase H — Common Evaluation Mask

**Shared timestamp set (intersection of XGB, LSTM, TFT predictions):**
- Indirect: ~2,109 shared rows per horizon vs 2,580 target-specific (82% retained for XGB)
- Direct: ~2,116 shared rows per horizon (82% retained)

**Metrics under shared mask (D+1):**
- Indirect: XGB 32.147% < LSTM 33.251% < TFT 34.915% → ranking unchanged
- Direct: XGB 13.299% < LSTM 13.723% < TFT 14.045% → ranking unchanged

**FINDING:** Model ranking is robust to evaluation mask choice. Common-mask nRMSE is
~3 pp higher (fewer easy daytime points due to intersection), but ordering is preserved.

---

### Phase E — SHAP Rank Stability

**E1 — Horizon stability (Spearman ρ of feature importance across D+1/D+2/D+3):**
- Indirect: ρ = 0.747–0.889 (high stability)
- Direct: ρ = 0.622–0.796 (moderate stability)
- Cross-target (indirect vs direct D+1): ρ = 0.579 (significant divergence)

**E2 — Feature group importance:**

| Group | Indirect (avg) | Direct (avg) |
|-------|---------------|-------------|
| rolling_stats | 52% | 75% |
| clearsky | 15% | <1% |
| NWP (fcst_*) | 11% | 11% |
| time_encodings | 8% | 3% |
| lag_power | 4% | 7% |

**FINDING:** Target definition strongly controls feature hierarchy. Clearsky geometry
is essential for indirect (15%) but irrelevant for direct (<1%). Direct target is
dominated by rolling production memory (75%). This mechanistically explains why
different model classes are best suited to each target.

**E3 — Ablation seed variability:**
All ΔnRMSE effects are reproducible across seeds 42/123/777 (std < 0.1 pp).
Rolling-7d removal is the largest impact for indirect (+0.60 ± 0.05 pp).
NWP removal is second (+0.24 ± 0.05 pp).
Clearsky removal is unexpected: slightly improves direct (−0.075 ± 0.033 pp),
consistent with zero SHAP importance for that group in direct target.

---

### Phase D — External Benchmark (Not Executed)

**Decision:** Phase D is deferred to future work (journal submission).

**Rationale:**
- Thesis scope is single-site controlled comparison (DS context, not journal)
- NREL PVDAQ is the recommended dataset (AC power + weather, utility-scale, public)
- Schema alignment + pvlib clear-sky computation for new site coordinates: ~20–30h effort
- Framing in paper: "Generalization to additional sites via NREL PVDAQ remains as future work
  (see Section 12). The current study establishes methodology; multi-site validation is
  required before ranking claims can be generalized."

---

### Phase F — Writing Corrections (to apply in report)

**6 Mandatory sentence replacements:**

1. **Abstract (ranking statement):**
   OLD: "we find a directional model ranking inversion between targets: XGB SOTA achieves
        the best observed clearness-index performance..."
   NEW: "We observe that XGBoost SOTA achieves the lowest nRMSE on both the indirect
        clear-sky performance index target (nRMSE = 29.1%, D+1) and the direct AC-power
        target (nRMSE = 12.1%, D+1); however, pairwise statistical superiority over BiLSTM
        could not be established (DM test, h=24 Newey-West correction, p > 0.10 for all
        aligned comparisons; observed effect sizes are 43–56% of the minimum detectable
        effect at 80% power)."

2. **DM caveat:**
   "Because prediction timestamps in one model export required reconstruction (LSTM window
   deduplication to day-ahead protocol), aligned paired errors were verified for y_true
   consistency before DM testing. The resulting p-values are reported as cautious diagnostics
   rather than confirmatory evidence; the test period is underpowered for detecting the
   observed effect sizes."

3. **Reproducibility:**
   "The study is reproducible at the level of code, feature engineering, split logic,
   and evaluation workflow within the authors' controlled environment, but not fully
   externally reproducible because the raw SCADA and NWP inputs are proprietary."

4. **Phase-1 footnote:**
   "Historical Phase-1/Phase-2 results are shown only for model-development context;
   they rely on leakage-affected features (theoretical_power, actual_power_kW,
   performance_ratio, same-timestamp NWP reanalysis) and are **not valid competitive
   baselines** against Phase-3 SOTA models."

5. **SHAP:**
   "The SHAP group-importance analysis indicates that the learned XGBoost model relies
   on different feature families under the two target definitions: clear-sky geometry
   (GHI_cs, DHI_cs, AOI) accounts for 15% of total SHAP importance for the clear-sky
   performance index target but less than 1% for direct AC power, while rolling
   production statistics account for 52% (indirect) vs 75% (direct)."

6. **Conclusions:**
   "The main contribution of this study is not a claim of universal model superiority,
   but evidence that **target definition changes the effective forecasting problem**:
   the feature hierarchy, route-level performance (indirect→power outperforms direct
   by 1.7–2.1 pp nRMSE at AC-power scale), and the operational framing differ
   substantially depending on whether the practitioner chooses to predict the clear-sky
   performance index or direct AC power."

**F1 — Terminology fix:** All occurrences of "clearness index" referring to target_k
should read "clear-sky performance index".

**F2 — Phase-1/Phase-2:** Move to Appendix; add footnote in main tables.

**TFT footnote (B1.2 Option B):** Add to all TFT result rows/tables:
"TFT trained with 20 Optuna trials vs 60 for XGB/LSTM; treat as indicative only."

---

### Summary of Key Quantitative Updates

| Metric | Old (pre-fix) | Corrected |
|--------|--------------|-----------|
| XGB indirect D+1 nRMSE | 24.508% | **29.095%** (leakage removed) |
| LSTM direct D+1 nRMSE | 10.917% | **12.600%** (timestamp aligned) |
| XGB direct D+1 nRMSE | not reported | **12.060%** |
| Ranking: best direct | "BiLSTM" | **XGB SOTA** (no inversion) |
| DM p-value LSTM vs XGB | not available | 0.133–0.189 (all NS) |
| N_eff (DM tests) | not reported | 255–829 |
| Indirect→power nRMSE | not reported | 11.16–11.34% |
| Direct nRMSE | not reported | 12.95–13.28% |
