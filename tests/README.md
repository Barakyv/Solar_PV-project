# Tests

IDs follow PROJECT_MASTER.md §8.2. Run with `pytest`; `pytest -m "not slow"` for a quick pass.

| ID | Test | File | Status |
|---|---|---|---|
| T-01 | No leaky features in X (rule level) | `leakage/test_leakage_rules.py` | ✅ stage 1 (matrix-level check added in stage 9) |
| T-02 | LSTM alignment: unique valid timestamps | — | stage 9 (smoke) |
| T-03 | DM with HLN matches a manual example | — | stage 8 |
| T-04 | y_true identical across models | — | stage 8 |
| T-05 | DST transitions to UTC | — | stage 2 |
| T-06 | Site coordinates consistent | `unit/test_daytime_mask.py` (indirectly) | stage 2 |
| T-07 | Daytime mask counts (2,580; 2,510/2,512/2,514) | `unit/test_daytime_mask.py` | ✅ stage 1 |
| T-08 | Production centroid at solar noon | — | stage 2 |
| T-09 | Rolling features horizon-safe (perturbation) | — | stage 6 |
| T-10 | Feature contract (75) | `unit/test_feature_contract.py` | ✅ stage 1 |
| T-11 | k ↔ kW inversion exact | — | stage 6 |
| T-12 | Route truth is actual AC | — | stage 8 |
| T-13 | No target clipping | — | stage 6 |
| T-15 | evaluate.py reproduces the LSTM fixture | — | stage 8 |
| T-16 | Cleaning never zeroes daytime production | — | stage 5 |
| T-17 | GFS run available before T0 | — | stage 6b |
| T-18 | Issuance boundary (perturbation after T0) | — | stage 6 |
