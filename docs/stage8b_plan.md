# Stage 8B plan: the evaluation layer on the corrected tables

Written on 2026-10-25, before the code. Stage 8A (2026-10-14) built the evaluation code and
proved it on the surviving old predictions (T-15). Part B connects it to the stage-6 tables:
- the truth and the theoretical power come from the feature tables;
- the indirect denominator becomes the new `MEAN_DAYTIME_K`;
- the reference baselines are added;
- the secondary views are added (D-21, D-22).

No model is trained in this stage. The run evaluates the reference baselines on **validation
only**: they exercise the whole chain, and the test split is not opened.

Scope rows covered: S-15 and the evaluation side of S-23–S-31, S-37 (docs/scope_register.md).

## 1. Self-review before the code

### 1a. Mistakes of the original evaluation that must not come back

Read in `submission_evidence/12_scripts/` and the v6 report:

| ID | Original | Where | Guard in the rebuild |
|---|---|---|---|
| E1 | The route comparison rebuilt the "true" kW of the indirect route from the clipped k | `compare_routes.py` (C-16) | Actual AC is the truth for both routes (8A; T-12) |
| E2 | The two routes were scored on different hours: the indirect route kept rows with k > 0.01, the direct route rows with AC > 10 kW. The report (§10) calls this a paired subset, but the n differ: 2,165 vs 2,172 for XGB and 2,243 vs 2,520 for BiLSTM | `compare_routes.py`, `route_comparison.csv` | Both routes on the identical row set; a test with extra rows on one side (new, C-34) |
| E3 | Daytime was chosen from the truth (y_true > 0.01 or > 10 kW) in the route and common-mask scripts. This removes the hours of zero output, i.e. downtime: selection on the outcome | `compare_routes.py`, `common_mask_analysis.py` (C-05) | One daytime definition, F-02, from solar geometry only. A test proves that a daytime hour with y_true = 0 is kept |
| E4 | The common mask used only XGB, BiLSTM and the TFT baseline | `common_mask_analysis.py` | All models in the run |
| E5 | Seed variability came from the training scripts' own metrics; for BiLSTM these were all-hours training metrics | `compile_seed_variability.py` (C-10) | Every seed is scored by evaluate.py; the seed table is built from the benchmark |
| E6 | DM without the HLN correction | `detectable_effect.py` (C-15) | Done in 8A (T-03) |
| E7 | Skill against "same time yesterday" at every horizon | report §4.2 (C-08) | Persistence of h days (8A); now also DST-safe (§1b) |
| E8 | y_true was read from each prediction file | Q-06 | Truth only from the feature table; the file's own y_true is used only as an alignment check (§3.3) |
| E9 | Each model was scored on a different n (2,510–2,580), because of input windows and the BiLSTM timestamp bug | report §7, §13.3 | Coverage rule: every model must predict every daytime hour of the split. Strict mode fails otherwise (§3.4) |
| E10 | Nothing stopped a test-set evaluation during development | — | Validation by default; test only with an explicit flag, logged (§3.1) |
| E11 | 0.6279 was the indirect denominator everywhere | `constants.py` | The derived constant; the legacy value is allowed only in fixture mode (test) |

### 1b. Review of the stage-8A code

| ID | Finding | Fix |
|---|---|---|
| A1 | Persistence was built from the y_true of the prediction files. Those cover only the test hours, so the first h days of test had no reference, and `n_skill` < n | Build the references from the full truth series of the feature table |
| A2 | Persistence used a plain 24·h-hour shift. On the 25-hour fall-back day this reads an hour that is not known at issuance, the same leak the features guard against (OP-37) | Use the DST guard of stage 6 (`past_value`); fill gaps with the same-hour 30-day median of earlier days (OP-38 rule) |
| A3 | `load_predictions` did not validate the target, horizon or split values, or finite predictions | Validate against §6.10 |
| A4 | The daytime mask was recomputed but never compared with the feature table's `daytime` | Assert that they are equal (one definition) |
| A5 | `route_comparison` reports `advantage_pp` without stating its sign | Document it: direct − indirect, so positive means the indirect route is better |
| A6 | `NotImplementedError` in the non-fixture path | Implemented here |

### 1c. Expected differences from v6, recorded now

- Test daytime hours with truth: **2,575**, not 2,580. Five daytime hours have no valid AC after
  cleaning (communication freezes, A-09). The old truth was taken from the prediction files.
- The indirect nRMSE uses 0.788157, so it is not on the v6 scale (methodology_changes §2.3).
- The route comparison is paired, so its n is the same for both routes.

## 2. Reference baselines (§6.9, §6.11)

All three are computed by the evaluation from the truth table, with a single code path, for the
split being evaluated:

- **Persistence (h):** the target at the same UTC hour h days earlier, if that hour ended by T0
  (DST guard, D-16). Otherwise the median of the same hour over the previous 30 days (OP-38). It
  is naive: operational limits are not masked. It is the reference of the skill score.
- **Climatology:** the mean of the target in train for each (UTC month, UTC hour), over hours
  with a known target and normal operation. These are the hours a model learns from (D-22).
  Horizon-independent.
- **Clear sky × MEAN_K:**
  - direct: `clip(theo_kw × MEAN_DAYTIME_K, 0, 3,449)`;
  - indirect: `MEAN_DAYTIME_K` where theo > 50 kW, and 0 otherwise.

  Horizon-independent. Its indirect → kW conversion equals its direct forecast exactly, which
  gives a built-in check of the route code (advantage = 0).

## 3. Evaluation changes

### 3.1 Split guard

- `--split val` is the default.
- (Superseded by D-32 and R-G1-05: the CLI scores `--split val` only; the test split is scored by the step `test` of a final run.) `--split test` required `--allow-test`. Each such run appends a line to
  `outputs/07_evaluation/test_access_log.csv` (time, commit, command, input files) and writes a
  warning to its run log.
- This is the record G-1 asks for ("the test touched no decision; a log of the files read").

### 3.2 Truth

`truth.py` reads `features_1H_D1.parquet`; the targets are identical in the three horizon tables
(T-33). It provides:
- `indirect` = `target_k`;
- `direct` = `site_total_AC_kW`;
- `theo_kw`;
- `daytime`;
- `split`;
- `op_limited_flag`;
- `hour_source`.

Daytime hours whose AC is missing drop out of every model alike.

### 3.3 Alignment check

- The y_true written in a prediction file must equal the truth at its timestamps. Tolerance:
  1e-6 for k, 1e-3 kW for power.
- A file whose timestamps are shifted (the v6 BiLSTM bug) or whose targets are swapped fails in
  strict mode.
- The largest difference per model, target and horizon goes to the manifest.

### 3.4 Coverage

- Every model, seed, target and horizon must cover every daytime hour of the split that has
  truth.
- In strict mode (the default outside the fixture), a gap is an error that names the counts.
- The fixture mode keeps the old n.

### 3.5 Tables

- **`benchmark_summary`:** as in 8A, plus:
  - `nRMSE_kW_pct` for both targets. The indirect forecast is converted with
    `clip(k̂ · theo, 0, 3,449)` (§6.9).
  - `nRMSE_mean_pct`: RMSE in kW / the mean actual AC of the evaluated rows (L-16).
  - `skill_cliper` (L-13): 1 − RMSE / RMSE_cp, with RMSE_cp = σ · √(1 − γ(h)²). σ and γ(h) are
    the standard deviation and the lag-24·h autocorrelation of the daytime target series in
    train.
  - Persistence stays the main skill reference, as in v6.
- **`benchmark_views`** (D-21, D-22), per model, seed, target, horizon and view:
  - `all`;
  - `normal_operation`: no op-limited hours;
  - `source_minute`, `source_polled`, `source_mixed`;
  - `no_polled`: minute hours only.

  Each row gives n, `nRMSE_pct` and `nRMSE_kW_pct`. The main number stays `all`.
- **`dm_tests`:** as in 8A, plus a block-bootstrap CI of the mean loss differential, as in the
  original `dm_tests_extended`: 24-row circular blocks, 2,000 resamples, seed 42.
- **`route_comparison`:** paired rows only (E2). The theoretical power comes from the truth table.
- **`common_mask_summary`** and **`common_mask_sensitivity`:** all models, plus n before and after
  the intersection, as in the original sensitivity file.
- **`seed_variability_summary`:** from the benchmark, per model, target and horizon: number of
  seeds, mean, std (ddof 1), minimum, maximum and range (E5).
- **`seasonal_metrics`:** unchanged.

Model names in the files follow the v6 identifiers (`xgb_sota`, `lstm_sota`, `tft_96`,
`tft_168`, `tft`, `patchtst`) and map to the report names (XGB-Opt … PatchTST). The references
are `persistence`, `climatology` and `clearsky_k`.

### 3.6 Run documentation

- `RunLog` (stage `07_evaluation`).
- A manifest with the input hashes, the split, the mean k, the coverage and alignment results,
  and the row counts of every table.
- An entry in the run journal; OP entries in the data-processing record; numbers N12.

## 4. Literature ideas of this stage (§16b)

- **L-13 and L-16** are added as extra columns. They strengthen the existing skill and metric
  tables without changing them (scope rule).
- **L-14** (potential skill for the quantile-trained BiLSTM) and **L-15** (distribution
  diagnostics) belong to the error analysis of stage 11 (§12.3), where they are produced. The
  functions are not written now.

## 5. Tests (IDs continue PROJECT_MASTER.md §8.2)

| ID | Level | Test |
|---|---|---|
| T-38 | unit | References: persistence with the DST guard (the 25th hour reads no hour at or after T0) and gap filling from earlier days only; climatology from normal-operation train rows only (changing val or test values changes nothing); clear sky × MEAN_K formulas |
| T-39 | unit | Evaluation rules: the split guard; the alignment check catches a one-hour shift and a target swap; the coverage check catches a missing daytime hour; a daytime hour with y_true = 0 is kept (E3); the route comparison stays paired when one route has extra rows (E2); for `clearsky_k` the route advantage is exactly 0 |
| T-40 | unit | New metrics: CLIPER on a synthetic AR(1) series; nRMSE by the mean; kW conversion of the indirect forecast; seed table; DM CI |
| T-41 | contract | The real run on validation: the truth equals the feature table; daytime = the table's `daytime`; n = validation daytime hours with truth (2,901) for every reference, target and horizon; view counts equal the table's flags; no test row was read |
| T-15 | regression | Unchanged: the fixture still reproduces the old metrics |

## 6. Acceptance

- `python -m solarpv.evaluation.evaluate` (validation, references only) runs from the clean commit
  and writes every table, its manifest and its run log.
- The new tests pass, together with T-15 and the full suite.
- The references behave as expected on validation: persistence and climatology are plausible,
  and the clear-sky route advantage is 0.
- The record (OP entries, N12), the run journal, the errata (C-34) and PROJECT_MASTER are updated.

## 7. Changes during the stage

- **CLIPER reference (L-13): from the evaluated sample, not from train.**
  - The plan (§3.5) took σ and γ(h) from train.
  - On validation, the first run (R8-01) gave plain persistence a positive skill against that
    reference: direct D+1 +0.15.
  - The blend σ·√(1 − γ²) is optimal, and never worse than persistence, only on the sample it
    is computed from: 2σ²(1 − γ) ≥ σ²(1 − γ²). Train statistics describe a different, harder
    sample.
  - σ and γ now come from the observed daytime series of the evaluated split. Only observations
    enter, no forecast, so it is a property of the sample, like the mean in L-16.
  - A test asserts that persistence does not beat the reference (unit and contract).
