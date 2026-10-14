"""Point metrics on daytime rows (stage 8, PROJECT_MASTER.md §6.9).

To implement:
  nrmse_pct(y_true, y_pred, denominator)  RMSE / denominator * 100
  mae, rmse, r2                           on the same rows
  skill(rmse_model, rmse_persistence)     1 - ratio
  block_bootstrap_ci(...)                 circular blocks of 24 h, 1,000 reps, seed 42
Acceptance: test T-15 (old LSTM predictions reproduce the old benchmark rows).
"""
