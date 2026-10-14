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
| T-16 | Cleaning never zeroes daytime production | — | stage 5 |
| T-17 | GFS run available before T0 | — | stage 6b |
| T-18 | Issuance boundary (perturbation after T0) | — | stage 6 |
| T-19 | Polled data rebuilt without the hold lag; source and era detection (F-18) | `unit/test_resolution.py` | ✅ stage 4 |
| T-20 | Clock correction: known shifts per group and source, exact step days, no smoothing across steps (D-19) | `unit/test_clock.py` | ✅ stage 4 |
| T-21 | Hourly rule: left label, 45/60 coverage; stage-4 output contract | `unit/test_preprocess.py`, `contracts/test_preprocess_output.py` | ✅ stage 4 |
