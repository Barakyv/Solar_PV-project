# Tests

IDs follow PROJECT_MASTER.md §8.2. Run with `pytest`; `pytest -m "not slow"` for a quick pass.

| ID | Test | File | Status |
|---|---|---|---|
| T-01 | No leaky features in X (rule level) | `leakage/test_leakage_rules.py` | ✅ stage 1 (matrix-level check added in stage 9) |
| T-02 | LSTM alignment: unique valid timestamps | — | stage 9 (smoke) |
| T-03 | DM with HLN matches a manual example | `unit/test_stats_tests.py` | ✅ stage 8 |
| T-04 | y_true identical across models | `unit/test_evaluate_helpers.py` | ✅ stage 8 |
| T-05 | DST transitions to UTC | `unit/test_time_conversion.py`, `contracts/test_ingest_output.py` | ✅ stage 2 |
| T-06 | Site coordinates consistent | `unit/test_daytime_mask.py` (indirectly) | stage 2 |
| T-07 | Daytime mask counts (2,580; 2,510/2,512/2,514) | `unit/test_daytime_mask.py` | ✅ stage 1 |
| T-08 | Production centroid at solar noon: ±30 min on raw stamps, ±2 min after the clock correction | `contracts/test_ingest_output.py`, `contracts/test_preprocess_output.py` | ✅ stage 2 / ✅ stage 4 (D-19) |
| T-09 | Rolling features horizon-safe (perturbation) | — | stage 6 |
| T-10 | Feature contract (75) | `unit/test_feature_contract.py` | ✅ stage 1 |
| T-11 | k ↔ kW inversion exact | — | stage 6 |
| T-12 | Route truth is actual AC | `unit/test_evaluate_helpers.py` | ✅ stage 8 |
| T-13 | No target clipping | — | stage 6 |
| T-15 | evaluate.py reproduces the LSTM fixture | `regression/test_evaluate_fixture.py` | ✅ stage 8 |
| T-16 | Cleaning never zeroes daytime production | `contracts/test_clean_output.py` | ✅ stage 5 |
| T-17 | GFS run available before T0 | — | stage 6b |
| T-18 | Issuance boundary (perturbation after T0) | — | stage 6 |
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

Stage-5 test plan: `docs/stage5_test_plan.md`. Mutation check (do the tests catch a wrong cleaning?): `00_data_audit/cleaning_2026-10-15/mutation_check.py`.
