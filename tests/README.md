# Tests

IDs follow PROJECT_MASTER.md §8.2. Run with `pytest`; `pytest -m "not slow"` for a quick pass.

| ID | Test | File | Status |
|---|---|---|---|
| T-01 | No leaky features in X (rule level; matrix level on the stage-6 tables) | `leakage/test_leakage_rules.py`, `contracts/test_features_output.py` | ✅ stage 1 / ✅ stage 6 (model matrices in stage 9) |
| T-02 | Sequence alignment for every sequence model (BiLSTM, three TFTs, PatchTST): unique valid timestamps, encoder ends at T0, decoder step assigned to a horizon by the local day of its valid time, including DST days | — | stage 9 (smoke) |
| T-03 | DM with HLN matches a manual example | `unit/test_stats_tests.py` | ✅ stage 8 |
| T-04 | y_true identical across models | `unit/test_evaluate_helpers.py` | ✅ stage 8 |
| T-05 | DST transitions to UTC | `unit/test_time_conversion.py`, `contracts/test_ingest_output.py` | ✅ stage 2 |
| T-06 | Site coordinates consistent | `unit/test_daytime_mask.py` (indirectly) | stage 2 |
| T-07 | Daytime mask counts (2,580; 2,510/2,512/2,514) | `unit/test_daytime_mask.py` | ✅ stage 1 |
| T-08 | Production centroid at solar noon: ±30 min on raw stamps, ±2 min after the clock correction | `contracts/test_ingest_output.py`, `contracts/test_preprocess_output.py` | ✅ stage 2 / ✅ stage 4 (D-19) |
| T-09 | Lags and rolling features horizon-safe: changing any value from T0 on changes no feature (synthetic, both DST windows, every horizon; real data around 2022-10-30) | `unit/test_features_build.py`, `contracts/test_features_output.py` | ✅ stage 6 |
| T-10 | Feature contract (74 after Q-05; 74 / 67 / 60 per horizon) | `unit/test_feature_contract.py` | ✅ stage 1 / ✅ stage 6 |
| T-11 | k ↔ kW inversion exact (synthetic and real, max error 2.3e-13 kW) | `unit/test_target.py`, `contracts/test_features_output.py` | ✅ stage 6 |
| T-12 | Route truth is actual AC | `unit/test_evaluate_helpers.py` | ✅ stage 8 |
| T-13 | No target clipping: nothing at exactly 1.2, values above 1.2 kept, k_suspect = k > 1.5 | `unit/test_target.py`, `contracts/test_features_output.py` | ✅ stage 6 |
| T-15 | evaluate.py reproduces the LSTM fixture | `regression/test_evaluate_fixture.py` | ✅ stage 8 |
| T-16 | Cleaning never zeroes daytime production | `contracts/test_clean_output.py` | ✅ stage 5 |
| T-17 | GFS run available before T0 | — | stage 6b |
| T-18 | Issuance boundary across DST: the 25th hour of the fall-back day is guarded | `unit/test_features_build.py`, `contracts/test_features_output.py` | ✅ stage 6 |
| T-19 | Polled data rebuilt without the hold lag; source and era detection (F-18) | `unit/test_resolution.py` | ✅ stage 4 |
| T-20 | Clock correction: known shifts per group and source, exact step days, no smoothing across steps (D-19) | `unit/test_clock.py` | ✅ stage 4 |
| T-21 | Hourly rule: left label, 45/60 coverage; stage-4 output contract | `unit/test_preprocess.py`, `contracts/test_preprocess_output.py` | ✅ stage 4 |
| T-22 | Cleaning rules R1–R7, flags F1–F2, precedence, idempotence, hourly flags (synthetic) | `unit/test_clean.py` | ✅ stage 5 |
| T-23 | Reconciliation: every change explained by one rule, every rule hit justified, nothing missed | `contracts/test_clean_output.py` | ✅ stage 5 |
| T-24 | Physical ranges after cleaning; no power far below the horizon; no dead DC | `contracts/test_clean_output.py` | ✅ stage 5 |
| T-25 | Flags never change values; flags match their definition | `contracts/test_clean_output.py` | ✅ stage 5 |
| T-26 | Hourly clean output: axis, means, site total, flag shares, op-limited, hour_source | `contracts/test_clean_output.py` | ✅ stage 5 |
| T-27 | Daytime energy unchanged on fault-free days (±0.1 %) | `contracts/test_clean_output.py` | ✅ stage 5 |
| T-28 | Log and manifest equal the output; policy and file hashes | `contracts/test_clean_output.py` | ✅ stage 5 |
| T-29 | Data-processing record complete and current (numbers from manifests, every stage, rule and OP) | `contracts/test_data_record.py` | ✅ |
| T-30 | Stage-3 EDA outputs: planned items exist, tables agree with stage 5 and physics, made from current inputs | `contracts/test_eda_output.py` | ✅ stage 3 |
| T-31 | Stage-3b quality outputs and notebook current; the claims of the notes hold | `contracts/test_quality_eda_output.py` | ✅ stage 3b |
| T-32 | Literature ideas L-xx: register complete and consistent | `contracts/test_literature_ideas.py` | ✅ |
| T-33 | Stage-6 tables: 32,856 rows per horizon, feature sets, UTC axis, metadata equal to stage 5, issuance times, decided target columns, manifest, dictionary (a definition and a unit per feature), target checks | `contracts/test_features_output.py` | ✅ stage 6 |
| T-34 | Imputation from the same UTC hour on earlier days only; later values change nothing | `unit/test_features_build.py` | ✅ stage 6 |
| T-35 | D-22 masks: each inverter by its own flags, site total by op-limited, weather not masked; no masked value reaches a lag (real data) | `unit/test_features_build.py`, `contracts/test_features_output.py` | ✅ stage 6 |
| T-36 | Calendar and geometry: Israeli weekend, solar noon from the transit, hourly means of minute values | `unit/test_features_build.py` | ✅ stage 6 |
| — | Stage-6 notebook current, a note after every code cell; target definition matches the configs | `contracts/test_features_notebook.py`, `contracts/test_target_definition.py` | ✅ stage 6 |
| T-37 | Scope register: every v6 figure (1–25) and numbered section has exactly one row; the six v6 models stay in D-08 | `contracts/test_scope_register.py` | ✅ (2026-10-25) |

Stage-5 test plan: `docs/stage5_test_plan.md`. Mutation check (do the tests catch a wrong cleaning?): `00_data_audit/cleaning_2026-10-15/mutation_check.py`.

Stage-6 tests: `docs/stage6_plan.md`. Mutation check (do the tests catch a leaky feature construction?): `00_data_audit/features_2026-10-20/mutation_check.py`, five planted leaks, all caught.
