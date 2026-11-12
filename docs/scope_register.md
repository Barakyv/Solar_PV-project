# Scope register: the approved scope of the submitted report (v6)

The approved scope of the project is the work reported in
`submission_evidence/13_report/Solar_PV_Forecasting_Report_CORRECTED_v6.docx`. On 2026-10-25 the
user decided to rebuild **all of it**: the six models, every analysis of chapters 3 and 7–12, and
the 25 figures. This register lists each deliverable (S-xx), the stage that produces it and its
status, so that nothing leaves the plan silently.

`tests/contracts/test_scope_register.py` (T-37) checks three things:
- every figure 1–25 and every numbered section of v6 has exactly one row;
- the IDs are complete;
- the six models are named in D-08.

Changes of method are recorded in `docs/methodology_changes.md`. This register is about scope.

**Status:**
- ✅ done (stage);
- ⬜ planned;
- 📝 text only (written in stage 11 from results of earlier stages).

**Producers (gate G-1, 2026-11-12, X60):** every row that is not text only names the code that
produces it in development (a module, a runner step or a script, in backticks), and the final
run reproduces it with the v6 budgets. A row without such code says why: X59 of
`docs/errata.md`, a stage-11 analysis or figure that no code produces yet (on the G-1 list for the
user). `tests/contracts/test_scope_register.py` (T-97) checks that every named path exists.

**Models (v6 names):**
- **XGB-Opt;**
- **BiLSTM-Opt;**
- **TFT-96;**
- **TFT-168;**
- **TFT baseline;**
- **PatchTST.**

## Register

| ID | v6 section | Deliverable | Figures | Models | Stage | Status |
|---|---|---|---|---|---|---|
| S-01 | 3.1 | Plant description and parameter table (wording corrected, C-30) | — | — | 11 | 📝 (data: stages 2, 5) |
| S-02 | 3.2 | Data processing: hourly means, clear sky, daytime definition | — | — | 2–6 | ✅ (stages 2–6: `src/solarpv/data/ingest.py`, `src/solarpv/data/preprocess.py`, `src/solarpv/data/clean.py`, `src/solarpv/features/build.py`) |
| S-03 | 3.3 | The two targets and MEAN_DAYTIME_K | — | — | 6 | ✅ (stage 6: `src/solarpv/features/build.py`, `src/solarpv/features/decide_k.py`) |
| S-04 | 3.4 | Feature table, inputs of the sequence models (TFT known reals, PatchTST target only) | — | all | 6, 9 | ✅ tabular (stage 6, `src/solarpv/features/build.py`) / ✅ sequence inputs (`src/solarpv/models/sequence_data.py`) (stage 9, W3: D-25–D-28, OP-55; the TFT gets the known part of the stage-7 selection, 10 / 27 features, not 18) |
| S-05 | 3.5 | End-to-end milestones table | — | — | 11 | 📝 |
| S-06 | 3.6.1–3.6.3 | Candidate pool, stage-1 screening, SHAP backward elimination | 19, 20 | XGB-Opt | 7 | ✅ (stage 7, `src/solarpv/selection/run.py`) |
| S-07 | 3.6.4 | Three-method consensus (05e), counts per method and loop | 21, 22 | XGB-Opt | 7 | ✅ (stage 7, `src/solarpv/selection/run.py`) |
| S-08 | 3.6.5 | 05f tiered audit | 23 | — | 7 | ✅ (stage 7, `src/solarpv/selection/run.py`) |
| S-09 | 3.6.6 | Operational availability classes | 24 | — | 7 | ✅ (stage 7, `src/solarpv/selection/run.py`) |
| S-10 | 3.6.7–3.6.8 | Stability-floor sensitivity; findings carried into modelling (the input of stage 9 per target × horizon, `model_inputs.json`) | 25 | — | 7 | ✅ (stage 7, `src/solarpv/selection/run.py`) |
| S-11 | 3.7 | Anti-leakage protocol and leakage list | — | all | 1, G-1 | ✅ list and tests (stage 1, `src/solarpv/features/leakage.py`) / ✅ manual audit (G-1, `docs/leakage_audit.md` §3, `scripts/g1_audit.py`) |
| S-12 | 3.8 | Split table and timeline | 11 | — | 3, 11 | ✅ table (stage 3, `src/solarpv/eda/figures.py`) / ✅ figure in the figure book (S01-02, `src/solarpv/figures/s00_s02.py`) |
| S-13 | 3.9 | Lag exclusion per horizon | — | — | 6 | ✅ (stage 6, `src/solarpv/features/build.py`) |
| S-14 | 4.1 | Optuna protocol: XGB and BiLSTM 60 trials, TFT 20 trials, objective daytime nRMSE on val | — | XGB-Opt, BiLSTM-Opt, TFT-96, TFT-168 | 9, 10 | ✅ dev (W2, W4: `src/solarpv/models/xgb_sota.py`, `src/solarpv/models/neural_worker.py`; budgets in `src/solarpv/pipeline/config.py`) / ⬜ final (stage 10) |
| S-15 | 4.2, 6.1–6.4 | Metrics, bootstrap CI, DM with HLN, skill score | — | all | 8 | ✅ (stage 8: part A 2026-10-14, part B 2026-10-25; `src/solarpv/evaluation/evaluate.py`, `src/solarpv/evaluation/stats_tests.py`) |
| S-16 | 5.1 | XGB-Opt: monotone constraints, 60 trials, seeds 42/123/777, search space | — | XGB-Opt | 9, 10 | ✅ dev (W2, step `xgb`: `src/solarpv/models/xgb_sota.py`) / ⬜ final |
| S-17 | 5.2 | BiLSTM-Opt: encoder-decoder, 72 steps, quantile loss, 60 trials, 3 seeds | — | BiLSTM-Opt | 9, 10 | ✅ dev (W4, step `bilstm`: `src/solarpv/models/lstm_sota.py`) / ⬜ final |
| S-18 | 5.3 | TFT-96: encoder 96 h, decoder 72 h, 20 trials, 18 known reals | — | TFT-96 | 9, 10 | ✅ dev (W4, step `tft96`: `src/solarpv/models/tft_model.py`; the known reals of D-26) / ⬜ final |
| S-19 | 5.4 | TFT-168: encoder 168 h, otherwise as TFT-96 | — | TFT-168 | 9, 10 | ✅ dev (W4, step `tft168`: `src/solarpv/models/tft_model.py`) / ⬜ final |
| S-20 | 5.5 | TFT baseline: fixed hyperparameters (encoder 168, hidden 64, dropout 0.1, up to 150 epochs) | — | TFT baseline | 9, 10 | ✅ dev (W4, step `tftbase`: `src/solarpv/models/tft_model.py`) / ⬜ final |
| S-21 | 5.6 | PatchTST: univariate, patch 16, stride 8, 243 rolling origins | — | PatchTST | 9, 10 | ✅ dev (W4, step `patchtst`: `src/solarpv/models/patchtst_model.py`; 245 origins, D-25) / ⬜ final |
| S-22 | 5 | Architecture overview of the six models | 12 | all | 11 | ⬜ no code (X59: a stage-11 figure) |
| S-23 | 7.1 | Benchmark of all models × 2 targets × 3 horizons with CIs | 1 | all | 11 | ✅ table in dev (`src/solarpv/evaluation/evaluate.py`, `benchmark_summary.csv`) / figure in the book (S09-25, `src/solarpv/figures/s09_models.py`) / ⬜ final |
| S-24 | 7.2–7.3 | Ranking tables per target; predicted vs observed scatter | 13 | all | 11 | ✅ ranking from `benchmark_summary.csv` (`src/solarpv/evaluation/evaluate.py`) / ⬜ scatter: no code (X59) |
| S-25 | 7.4 | Common-mask evaluation | — | all | 11 | ✅ dev (`src/solarpv/evaluation/evaluate.py`, `common_mask_summary.csv`) / ⬜ final |
| S-26 | 7.5 | Horizon robustness table | 2 | all | 11 | ✅ data in dev (`benchmark_summary.csv` by horizon, `src/solarpv/evaluation/evaluate.py`) / ⬜ table and figure: no code (X59) |
| S-27 | 8.1 | Diebold-Mariano against XGB-Opt | 5 | all | 11 | ✅ dev (`src/solarpv/evaluation/evaluate.py`, `dm_tests_kw.csv`, Q-19) / ⬜ figure: no code (X59) |
| S-28 | 8.2 | Statistical power (observed effect / MDE) | — | all | 11 | ✅ dev (`src/solarpv/evaluation/stats_tests.py`, columns `min_detectable_effect` and `pct_of_mde` of `dm_tests_kw.csv`) |
| S-29 | 9.1 | Seasonal performance: table, heatmap, radar | 3, 14 | all | 11 | ✅ table in dev (`src/solarpv/evaluation/evaluate.py`, `seasonal_metrics.csv`) / ⬜ heatmap and radar: no code (X59) |
| S-30 | 9.2 | Error by local hour × season (top four models) | 4 | all | 11 | ⬜ no code (X59; the prediction files hold everything it needs) |
| S-31 | 10 | Route comparison, indirect → power vs direct (truth = actual AC, C-16) | 10 | XGB-Opt, BiLSTM-Opt | 11 | ✅ dev (`src/solarpv/evaluation/evaluate.py`, `route_comparison.csv`) / figure in the book (S09-27, `src/solarpv/figures/s09_models.py`) |
| S-32 | 11.1 | SHAP group importance; SHAP rank stability across horizons | 6 | XGB-Opt | 11 | ✅ dev (W5, step `explain`: `src/solarpv/pipeline/explain.py`) / ⬜ final, and on test in phase 2 (D-31) |
| S-33 | 11.2 | TFT temporal attention | 7 | TFT-96, TFT-168 | 10, 11 | ✅ dev (W4 storage, W5 step `explain`: `src/solarpv/pipeline/explain.py`) / ⬜ final |
| S-34 | 11.3 | TFT encoder-length ablation (96 vs 168) | — | TFT-96, TFT-168 | 11 | ✅ both models scored on the same hours in dev (`benchmark_summary.csv`, `dm_tests_kw.csv`) / ⬜ the comparison table: no code (X59) |
| S-35 | 11.4 | PatchTST autoregressive structure | — | PatchTST | 11 | ⬜ no code (X59) |
| S-36 | 11.5 | Optuna convergence and hyperparameter importance (TFT-96) | 8 | TFT-96 | 10, 11 | ✅ dev (W5, step `explain`: fANOVA over ten seeds, `src/solarpv/pipeline/explain.py`) / ⬜ final |
| S-37 | 12.1 | Seed variability (each seed scored by evaluate.py, C-10) | 17 | XGB-Opt, BiLSTM-Opt | 10, 11 | ✅ code (`src/solarpv/evaluation/evaluate.py`, `seed_variability_summary.csv`; the report's spread in kW, X42); one seed in dev, three in the final run / ⬜ figure: no code (X59) |
| S-38 | 12.2 | 5-fold expanding-window CV vs test (script rewritten) | 18 | XGB-Opt | 11 | ⬜ no code (X59; it trains models, so it must exist before the tag or reopen G-1) |
| S-39 | 12.3 | Bias² + variance decomposition | 9 | XGB-Opt, BiLSTM-Opt | 11 | ⬜ no code (X59; from the seed forecasts of the final run) |
| S-40 | 12.4 | Production quintile analysis | 16 | XGB-Opt | 11 | ⬜ no code (X59; from the prediction files) |
| S-41 | 12.5 | Feature-group ablation, 3 seeds (leak-free base, C-07) | 15 | XGB-Opt | 11 | ⬜ no code (X59; it trains models, so it must exist before the tag or reopen G-1) |
| S-42 | 13.1–13.6 | Discussion | — | all | 11 | 📝 |
| S-43 | 14 | Conclusions C1–C8, recommendations, future work | — | all | 11 | 📝 |
| S-44 | 15 | Academic disclosures D1–D5 (D1 rewritten, F-01c) | — | all | 11 | 📝 |
| S-45 | 16 | Data and code availability; reproducibility package | — | — | 11 | ✅ the reproduction from the raw data (W6, `scripts/reproduce_paper_results.py`) / ⬜ the stage-11 options `--xgb-only` and `--skip-train` |

## Required by the corrections (not new scope)

These follow from the bugs that were fixed, or from the decisions approved by the user (D-21,
D-22, D-12). They are reported beside the v6 deliverables, without changing the structure of
the report.

- A "Reproduction and correction note" and an errata appendix with old vs new values (D-02).
- A data appendix from `docs/data_processing_record.md`.
- Reference baselines: persistence matched to the horizon, climatology, clear sky × MEAN_K.
  They are the reference of the skill score (§6.9).
- Secondary evaluation views:
  - without op-limited hours (D-22);
  - by data source, and without polled hours (D-21).
- GFS forecasts (D-12, stage 6b). They restore the NWP component of the original pipeline with
  a source that is available at issuance (the first Future Work item of v6). If admitted, GFS
  columns are offered to every model that takes exogenous inputs (XGB-Opt, BiLSTM-Opt, the
  TFTs). PatchTST stays univariate, as in v6.

## Outside the approved scope

Literature ideas that would add model families or change the benchmark (L-17 other learners,
L-18 blending, an exogenous PatchTST) are not part of the scope. They may appear only in a marked
appendix, and only after the whole register is done (PROJECT_MASTER.md §16b).
