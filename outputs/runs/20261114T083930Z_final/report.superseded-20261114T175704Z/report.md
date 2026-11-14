# Run report: 20261114T083930Z_final

Mode `final`. Every number below is read from a file of this run; the files and their SHA-256 are listed at the end (M31). Validation numbers are tuning-split numbers (D-10): validation chose the hyper-parameters and the stopping round.

## Steps

| step | status | commit | duration_s | n_warnings | n_resumed |
|---|---|---|---|---|---|
| select | ok | 10611a6 | 4.1 | 0 | 0 |
| baselines | ok | 10611a6 | 4.5 | 1 | 0 |
| xgb | ok | 10611a6 | 3214.3 | 22 | 0 |
| cv | ok | 10611a6 | 121.1 | 0 | 0 |
| ablation | ok | 10611a6 | 389.0 | 3 | 0 |
| bilstm | ok | 10611a6 | 4705.8 | 8 | 0 |
| tft96 | ok | 10611a6 | 8700.5 | 2 | 0 |
| tft168 | ok | 10611a6 | 12209.9 | 3 | 0 |
| tftbase | ok | 10611a6 | 721.5 | 1 | 0 |
| patchtst | ok | 10611a6 | 899.1 | 1 | 0 |
| evaluate | ok | 10611a6 | 28.3 | 0 | 0 |
| explain | ok | 10611a6 | 332.7 | 0 | 0 |

## Benchmark on val (nRMSE in kW, % of 3,449 kW)

| target | horizon | model | seed | n | nRMSE_kW_pct | skill_score_kw |
|---|---|---|---|---|---|---|
| direct | 1 | clearsky_k | -1 | 2901 | 8.959 | 0.074 |
| direct | 1 | climatology | -1 | 2901 | 8.041 | 0.169 |
| direct | 1 | lstm_sota | 42 | 2901 | 6.671 | 0.311 |
| direct | 1 | lstm_sota | 123 | 2901 | 6.613 | 0.317 |
| direct | 1 | lstm_sota | 777 | 2901 | 6.612 | 0.317 |
| direct | 1 | patchtst | 42 | 2901 | 8.090 | 0.164 |
| direct | 1 | persistence | -1 | 2901 | 9.676 |  |
| direct | 1 | tft | 42 | 2901 | 6.873 | 0.290 |
| direct | 1 | tft_168 | 42 | 2901 | 6.756 | 0.302 |
| direct | 1 | tft_96 | 42 | 2901 | 6.656 | 0.312 |
| direct | 1 | xgb_sota | 42 | 2901 | 6.628 | 0.315 |
| direct | 1 | xgb_sota | 123 | 2901 | 6.598 | 0.318 |
| direct | 1 | xgb_sota | 777 | 2901 | 6.644 | 0.313 |
| direct | 2 | clearsky_k | -1 | 2901 | 8.959 | 0.172 |
| direct | 2 | climatology | -1 | 2901 | 8.041 | 0.257 |
| direct | 2 | lstm_sota | 42 | 2901 | 6.833 | 0.369 |
| direct | 2 | lstm_sota | 123 | 2901 | 6.738 | 0.377 |
| direct | 2 | lstm_sota | 777 | 2901 | 6.828 | 0.369 |
| direct | 2 | patchtst | 42 | 2901 | 8.266 | 0.236 |
| direct | 2 | persistence | -1 | 2901 | 10.822 |  |
| direct | 2 | tft | 42 | 2901 | 7.042 | 0.349 |
| direct | 2 | tft_168 | 42 | 2901 | 6.916 | 0.361 |
| direct | 2 | tft_96 | 42 | 2901 | 6.830 | 0.369 |
| direct | 2 | xgb_sota | 42 | 2901 | 6.795 | 0.372 |
| direct | 2 | xgb_sota | 123 | 2901 | 6.833 | 0.369 |
| direct | 2 | xgb_sota | 777 | 2901 | 6.803 | 0.371 |
| direct | 3 | clearsky_k | -1 | 2901 | 8.959 | 0.164 |
| direct | 3 | climatology | -1 | 2901 | 8.041 | 0.250 |
| direct | 3 | lstm_sota | 42 | 2901 | 6.471 | 0.396 |
| direct | 3 | lstm_sota | 123 | 2901 | 6.466 | 0.397 |
| direct | 3 | lstm_sota | 777 | 2901 | 6.471 | 0.396 |
| direct | 3 | patchtst | 42 | 2901 | 8.292 | 0.227 |
| direct | 3 | persistence | -1 | 2901 | 10.720 |  |
| direct | 3 | tft | 42 | 2901 | 6.714 | 0.374 |
| direct | 3 | tft_168 | 42 | 2901 | 6.559 | 0.388 |
| direct | 3 | tft_96 | 42 | 2901 | 6.581 | 0.386 |
| direct | 3 | xgb_sota | 42 | 2901 | 6.426 | 0.401 |
| direct | 3 | xgb_sota | 123 | 2901 | 6.453 | 0.398 |
| direct | 3 | xgb_sota | 777 | 2901 | 6.429 | 0.400 |
| indirect | 1 | clearsky_k | -1 | 2901 | 8.959 | 0.075 |
| indirect | 1 | climatology | -1 | 2901 | 8.062 | 0.167 |
| indirect | 1 | lstm_sota | 42 | 2901 | 7.017 | 0.275 |
| indirect | 1 | lstm_sota | 123 | 2901 | 6.994 | 0.278 |
| indirect | 1 | lstm_sota | 777 | 2901 | 6.972 | 0.280 |
| indirect | 1 | patchtst | 42 | 2901 | 8.766 | 0.095 |
| indirect | 1 | persistence | -1 | 2901 | 9.683 |  |
| indirect | 1 | tft | 42 | 2901 | 6.834 | 0.294 |
| indirect | 1 | tft_168 | 42 | 2901 | 6.822 | 0.295 |
| indirect | 1 | tft_96 | 42 | 2901 | 6.744 | 0.303 |
| indirect | 1 | xgb_sota | 42 | 2901 | 6.644 | 0.314 |
| indirect | 1 | xgb_sota | 123 | 2901 | 6.589 | 0.320 |
| indirect | 1 | xgb_sota | 777 | 2901 | 6.643 | 0.314 |
| indirect | 2 | clearsky_k | -1 | 2901 | 8.959 | 0.174 |
| indirect | 2 | climatology | -1 | 2901 | 8.062 | 0.256 |
| indirect | 2 | lstm_sota | 42 | 2901 | 7.247 | 0.331 |
| indirect | 2 | lstm_sota | 123 | 2901 | 7.167 | 0.339 |
| indirect | 2 | lstm_sota | 777 | 2901 | 7.100 | 0.345 |
| indirect | 2 | patchtst | 42 | 2901 | 8.904 | 0.179 |
| indirect | 2 | persistence | -1 | 2901 | 10.841 |  |
| indirect | 2 | tft | 42 | 2901 | 7.097 | 0.345 |
| indirect | 2 | tft_168 | 42 | 2901 | 7.038 | 0.351 |
| indirect | 2 | tft_96 | 42 | 2901 | 6.943 | 0.360 |
| indirect | 2 | xgb_sota | 42 | 2901 | 6.888 | 0.365 |
| indirect | 2 | xgb_sota | 123 | 2901 | 6.949 | 0.359 |
| indirect | 2 | xgb_sota | 777 | 2901 | 6.793 | 0.373 |
| indirect | 3 | clearsky_k | -1 | 2901 | 8.959 | 0.167 |
| indirect | 3 | climatology | -1 | 2901 | 8.062 | 0.251 |
| indirect | 3 | lstm_sota | 42 | 2901 | 6.923 | 0.357 |
| indirect | 3 | lstm_sota | 123 | 2901 | 6.791 | 0.369 |
| indirect | 3 | lstm_sota | 777 | 2901 | 6.813 | 0.367 |
| indirect | 3 | patchtst | 42 | 2901 | 8.860 | 0.177 |
| indirect | 3 | persistence | -1 | 2901 | 10.759 |  |
| indirect | 3 | tft | 42 | 2901 | 6.809 | 0.367 |
| indirect | 3 | tft_168 | 42 | 2901 | 6.597 | 0.387 |
| indirect | 3 | tft_96 | 42 | 2901 | 6.575 | 0.389 |
| indirect | 3 | xgb_sota | 42 | 2901 | 6.546 | 0.392 |
| indirect | 3 | xgb_sota | 123 | 2901 | 6.537 | 0.392 |
| indirect | 3 | xgb_sota | 777 | 2901 | 6.586 | 0.388 |

## Seed spread (stage-10 rule: below 0.15 pp in kW)

| model | target | horizon | n_seeds | std_kw_pp | std_target_unit_pp |
|---|---|---|---|---|---|
| lstm_sota | direct | 1 | 3 | 0.034 | 0.034 |
| lstm_sota | direct | 2 | 3 | 0.053 | 0.053 |
| lstm_sota | direct | 3 | 3 | 0.003 | 0.003 |
| lstm_sota | indirect | 1 | 3 | 0.023 | 0.227 |
| lstm_sota | indirect | 2 | 3 | 0.074 | 0.023 |
| lstm_sota | indirect | 3 | 3 | 0.071 | 0.108 |
| patchtst | direct | 1 | 1 |  |  |
| patchtst | direct | 2 | 1 |  |  |
| patchtst | direct | 3 | 1 |  |  |
| patchtst | indirect | 1 | 1 |  |  |
| patchtst | indirect | 2 | 1 |  |  |
| patchtst | indirect | 3 | 1 |  |  |
| tft | direct | 1 | 1 |  |  |
| tft | direct | 2 | 1 |  |  |
| tft | direct | 3 | 1 |  |  |
| tft | indirect | 1 | 1 |  |  |
| tft | indirect | 2 | 1 |  |  |
| tft | indirect | 3 | 1 |  |  |
| tft_168 | direct | 1 | 1 |  |  |
| tft_168 | direct | 2 | 1 |  |  |
| tft_168 | direct | 3 | 1 |  |  |
| tft_168 | indirect | 1 | 1 |  |  |
| tft_168 | indirect | 2 | 1 |  |  |
| tft_168 | indirect | 3 | 1 |  |  |
| tft_96 | direct | 1 | 1 |  |  |
| tft_96 | direct | 2 | 1 |  |  |
| tft_96 | direct | 3 | 1 |  |  |
| tft_96 | indirect | 1 | 1 |  |  |
| tft_96 | indirect | 2 | 1 |  |  |
| tft_96 | indirect | 3 | 1 |  |  |
| xgb_sota | direct | 1 | 3 | 0.023 | 0.023 |
| xgb_sota | direct | 2 | 3 | 0.020 | 0.020 |
| xgb_sota | direct | 3 | 3 | 0.015 | 0.015 |
| xgb_sota | indirect | 1 | 3 | 0.032 | 0.277 |
| xgb_sota | indirect | 2 | 3 | 0.079 | 0.848 |
| xgb_sota | indirect | 3 | 3 | 0.026 | 0.423 |

## Training diagnostics

| step | subject | unit_of_best | best | cap | run | gap_ratio | value_kw |
|---|---|---|---|---|---|---|---|
| xgb | indirect_D1_seed42 | round | 220 | 1064 | 271 | 2.565 | 6.644 |
| xgb | indirect_D1_seed123 | round | 192 | 826 | 243 | 2.677 | 6.589 |
| xgb | indirect_D1_seed777 | round | 234 | 1140 | 285 | 2.689 | 6.643 |
| xgb | direct_D1_seed42 | round | 280 | 1355 | 331 | 1.406 | 6.628 |
| xgb | direct_D1_seed123 | round | 204 | 205 | 205 | 1.072 | 6.598 |
| xgb | direct_D1_seed777 | round | 50 | 738 | 101 | 1.128 | 6.644 |
| xgb | indirect_D2_seed42 | round | 44 | 1630 | 95 | 1.483 | 6.888 |
| xgb | indirect_D2_seed123 | round | 317 | 1972 | 368 | 1.044 | 6.949 |
| xgb | indirect_D2_seed777 | round | 27 | 1738 | 78 | 1.333 | 6.793 |
| xgb | direct_D2_seed42 | round | 233 | 234 | 234 | 1.165 | 6.795 |
| xgb | direct_D2_seed123 | round | 365 | 1808 | 416 | 1.221 | 6.833 |
| xgb | direct_D2_seed777 | round | 57 | 1121 | 108 | 1.373 | 6.803 |
| xgb | indirect_D3_seed42 | round | 223 | 977 | 274 | 1.879 | 6.546 |
| xgb | indirect_D3_seed123 | round | 483 | 1534 | 534 | 0.954 | 6.537 |
| xgb | indirect_D3_seed777 | round | 135 | 689 | 186 | 2.138 | 6.586 |
| xgb | direct_D3_seed42 | round | 31 | 1011 | 82 | 1.200 | 6.426 |
| xgb | direct_D3_seed123 | round | 245 | 780 | 296 | 1.410 | 6.453 |
| xgb | direct_D3_seed777 | round | 385 | 1108 | 436 | 1.171 | 6.429 |
| bilstm | indirect seed 42 | epoch | 55 | 60 | 60 | 0.769 | 7.064 |
| bilstm | direct seed 42 | epoch | 21 | 60 | 31 | 0.664 | 6.660 |
| bilstm | indirect seed 123 | epoch | 60 | 60 | 60 | 0.642 | 6.986 |
| bilstm | direct seed 123 | epoch | 9 | 60 | 19 | 0.674 | 6.607 |
| bilstm | indirect seed 777 | epoch | 57 | 60 | 60 | 0.833 | 6.962 |
| bilstm | direct seed 777 | epoch | 16 | 60 | 26 | 0.746 | 6.638 |
| tft96 | indirect seed 42 | epoch | 15 | 100 | 25 | 0.763 | 6.756 |
| tft96 | direct seed 42 | epoch | 12 | 100 | 22 | 0.724 | 6.690 |
| tft168 | indirect seed 42 | epoch | 46 | 100 | 56 | 0.778 | 6.821 |
| tft168 | direct seed 42 | epoch | 46 | 100 | 56 | 0.720 | 6.745 |
| tftbase | indirect seed 42 | epoch | 18 | 150 | 28 | 0.741 | 6.915 |
| tftbase | direct seed 42 | epoch | 58 | 150 | 68 | 0.771 | 6.878 |
| patchtst | indirect seed 42 | step | 4900 | 5000 | 4999 |  | 8.844 |
| patchtst | direct seed 42 | step | 4100 | 5000 | 4999 |  | 8.216 |

## Studies

| step | subject | n_trials | best_trial | best_value_kw | edges |
|---|---|---|---|---|---|
| xgb | indirect_D1_seed42 | 60 | 23 | 6.773 | max_depth |
| xgb | indirect_D1_seed123 | 60 | 50 | 6.772 | max_depth, min_child_weight |
| xgb | indirect_D1_seed777 | 60 | 29 | 6.756 | max_depth, colsample_bytree |
| xgb | direct_D1_seed42 | 60 | 55 | 6.628 | reg_lambda |
| xgb | direct_D1_seed123 | 60 | 16 | 6.598 | n_estimators |
| xgb | direct_D1_seed777 | 60 | 14 | 6.644 |  |
| xgb | indirect_D2_seed42 | 60 | 22 | 7.088 | max_depth |
| xgb | indirect_D2_seed123 | 60 | 40 | 7.160 | n_estimators, subsample |
| xgb | indirect_D2_seed777 | 60 | 42 | 7.019 | max_depth |
| xgb | direct_D2_seed42 | 60 | 55 | 6.795 | n_estimators, max_depth, colsample_bytree |
| xgb | direct_D2_seed123 | 60 | 23 | 6.833 | learning_rate, min_child_weight, reg_lambda |
| xgb | direct_D2_seed777 | 60 | 25 | 6.803 |  |
| xgb | indirect_D3_seed42 | 60 | 38 | 6.638 |  |
| xgb | indirect_D3_seed123 | 60 | 45 | 6.703 |  |
| xgb | indirect_D3_seed777 | 60 | 11 | 6.661 | max_depth |
| xgb | direct_D3_seed42 | 60 | 10 | 6.426 |  |
| xgb | direct_D3_seed123 | 60 | 27 | 6.453 | max_depth |
| xgb | direct_D3_seed777 | 60 | 50 | 6.429 | learning_rate |
| bilstm | indirect seed 42 | 60 | 53 | 6.817 | hidden, n_layers |
| bilstm | direct seed 42 | 60 | 34 | 6.660 | batch_size |
| bilstm | indirect seed 123 | 60 | 58 | 6.748 | hidden, n_layers, batch_size |
| bilstm | direct seed 123 | 60 | 36 | 6.607 | batch_size |
| bilstm | indirect seed 777 | 60 | 52 | 6.799 | hidden, n_layers |
| bilstm | direct seed 777 | 60 | 11 | 6.638 | hidden, n_layers, batch_size |
| tft96 | indirect seed 42 | 20 | 7 | 6.756 |  |
| tft96 | direct seed 42 | 20 | 18 | 6.690 | hidden_size, lstm_layers |
| tft168 | indirect seed 42 | 20 | 19 | 6.821 | hidden_continuous_size |
| tft168 | direct seed 42 | 20 | 13 | 6.745 | attention_head_size |

## Automatic warnings (41)

| step | kind | status | text |
|---|---|---|---|
| baselines | test_access | step log only | TEST ROWS READ (forecast the references of the test window (observed AC and k before each forecast hour), 5829 rows); recorded in test_access_log.csv |
| xgb | test_access | step log only | TEST ROWS READ (forecast the test window of D+1 (features), 5829 rows); recorded in test_access_log.csv |
| xgb | gap | matched | indirect_D1_seed42: validation RMSE 0.1001 is 2.6 x the train RMSE 0.03901 at the best iteration |
| xgb | edge_space | matched | indirect_D1_seed42: best trial on the edge of the search space for max_depth = 10 |
| xgb | gap | matched | indirect_D1_seed123: validation RMSE 0.1001 is 2.7 x the train RMSE 0.0374 at the best iteration |
| xgb | edge_space | matched | indirect_D1_seed123: best trial on the edge of the search space for max_depth = 10, min_child_weight = 1 |
| xgb | gap | matched | indirect_D1_seed777: validation RMSE 0.09944 is 2.7 x the train RMSE 0.03698 at the best iteration |
| xgb | edge_space | matched | indirect_D1_seed777: best trial on the edge of the search space for max_depth = 10, colsample_bytree = 0.503 |
| xgb | edge_space | matched | direct_D1_seed42: best trial on the edge of the search space for reg_lambda = 1.24e-08 |
| xgb | cap | matched | direct_D1_seed123: best iteration 204 at the cap of 205 rounds: it may not have converged (M12d) |
| xgb | edge_space | matched | direct_D1_seed123: best trial on the edge of the search space for n_estimators = 205 |
| xgb | test_access | step log only | TEST ROWS READ (forecast the test window of D+2 (features), 5829 rows); recorded in test_access_log.csv |
| xgb | edge_space | matched | indirect_D2_seed42: best trial on the edge of the search space for max_depth = 10 |
| xgb | edge_space | matched | indirect_D2_seed123: best trial on the edge of the search space for n_estimators = 1972, subsample = 0.603 |
| xgb | edge_space | matched | indirect_D2_seed777: best trial on the edge of the search space for max_depth = 10 |
| xgb | cap | matched | direct_D2_seed42: best iteration 233 at the cap of 234 rounds: it may not have converged (M12d) |
| xgb | edge_space | matched | direct_D2_seed42: best trial on the edge of the search space for n_estimators = 234, max_depth = 10, colsample_bytree = 0.5046 |
| xgb | edge_space | matched | direct_D2_seed123: best trial on the edge of the search space for learning_rate = 0.01019, min_child_weight = 20, reg_lambda = 1.085e-08 |
| xgb | test_access | step log only | TEST ROWS READ (forecast the test window of D+3 (features), 5829 rows); recorded in test_access_log.csv |
| xgb | gap | matched | indirect_D3_seed777: validation RMSE 0.1034 is 2.1 x the train RMSE 0.04838 at the best iteration |
| xgb | edge_space | matched | indirect_D3_seed777: best trial on the edge of the search space for max_depth = 10 |
| xgb | edge_space | matched | direct_D3_seed123: best trial on the edge of the search space for max_depth = 10 |
| xgb | edge_space | matched | direct_D3_seed777: best trial on the edge of the search space for learning_rate = 0.01003 |
| ablation | test_access | step log only | TEST ROWS READ (forecast the test window of D+1 for the ablation (features), 5829 rows); recorded in test_access_log.csv |
| ablation | test_access | step log only | TEST ROWS READ (forecast the test window of D+2 for the ablation (features), 5829 rows); recorded in test_access_log.csv |
| ablation | test_access | step log only | TEST ROWS READ (forecast the test window of D+3 for the ablation (features), 5829 rows); recorded in test_access_log.csv |
| bilstm | test_access | step log only | TEST ROWS READ (forecast the test window (the encoder series and the known inputs), 5829 rows); recorded in test_access_log.csv |
| bilstm | edge_space | matched | bilstm: indirect seed 42: best trial on the edge of the search space for hidden = 256, n_layers = 1 (M23) |
| bilstm | last_epoch | matched | bilstm: indirect seed 123: best epoch 60 at the last epoch: it may not have converged |
| bilstm | edge_space | matched | bilstm: indirect seed 123: best trial on the edge of the search space for hidden = 64, n_layers = 1, batch_size = 128 (M23) |
| bilstm | edge_space | matched | bilstm: indirect seed 777: best trial on the edge of the search space for hidden = 64, n_layers = 1 (M23) |
| bilstm | edge_space | matched | bilstm: direct seed 42: best trial on the edge of the search space for batch_size = 128 (M23) |
| bilstm | edge_space | matched | bilstm: direct seed 123: best trial on the edge of the search space for batch_size = 32 (M23) |
| bilstm | edge_space | matched | bilstm: direct seed 777: best trial on the edge of the search space for hidden = 256, n_layers = 1, batch_size = 32 (M23) |
| tft96 | test_access | step log only | TEST ROWS READ (forecast the test window (the encoder series and the known inputs), 5829 rows); recorded in test_access_log.csv |
| tft96 | edge_space | matched | tft96: direct seed 42: best trial on the edge of the search space for hidden_size = 64, lstm_layers = 1 (M23) |
| tft168 | test_access | step log only | TEST ROWS READ (forecast the test window (the encoder series and the known inputs), 5829 rows); recorded in test_access_log.csv |
| tft168 | edge_space | matched | tft168: indirect seed 42: best trial on the edge of the search space for hidden_continuous_size = 32 (M23) |
| tft168 | edge_space | matched | tft168: direct seed 42: best trial on the edge of the search space for attention_head_size = 4 (M23) |
| tftbase | test_access | step log only | TEST ROWS READ (forecast the test window (the encoder series and the known inputs), 5829 rows); recorded in test_access_log.csv |
| patchtst | test_access | step log only | TEST ROWS READ (forecast the test window (the encoder series and the known inputs), 5829 rows); recorded in test_access_log.csv |

## Explanation: the three largest mean |SHAP| (kW) per booster

| target | horizon | seed | rank | feature | group | mean_abs_kw |
|---|---|---|---|---|---|---|
| indirect | 1 | 42 | 1 | fcst_gfs_k_phys | nwp_gfs | 45.27 |
| indirect | 1 | 42 | 2 | fcst_gfs_kt_poa_daymean | nwp_gfs | 34.57 |
| indirect | 1 | 42 | 3 | fcst_gfs_lcc | nwp_gfs | 20.61 |
| indirect | 1 | 123 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 49.89 |
| indirect | 1 | 123 | 2 | fcst_gfs_k_phys | nwp_gfs | 45.25 |
| indirect | 1 | 123 | 3 | fcst_gfs_pwat | nwp_gfs | 20.08 |
| indirect | 1 | 777 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 37.13 |
| indirect | 1 | 777 | 2 | fcst_gfs_k_phys | nwp_gfs | 34.86 |
| indirect | 1 | 777 | 3 | fcst_gfs_pwat | nwp_gfs | 18.91 |
| direct | 1 | 42 | 1 | fcst_gfs_pv_kw | nwp_gfs | 383.01 |
| direct | 1 | 42 | 2 | fcst_gfs_poa | nwp_gfs | 359.87 |
| direct | 1 | 42 | 3 | poa_global_clearsky | clearsky | 84.96 |
| direct | 1 | 123 | 1 | fcst_gfs_poa | nwp_gfs | 389.18 |
| direct | 1 | 123 | 2 | fcst_gfs_pv_kw | nwp_gfs | 319.57 |
| direct | 1 | 123 | 3 | poa_global_clearsky | clearsky | 64.26 |
| direct | 1 | 777 | 1 | fcst_gfs_pv_kw | nwp_gfs | 438.16 |
| direct | 1 | 777 | 2 | fcst_gfs_poa | nwp_gfs | 362.61 |
| direct | 1 | 777 | 3 | poa_global_clearsky | clearsky | 68.64 |
| indirect | 2 | 42 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 49.48 |
| indirect | 2 | 42 | 2 | fcst_gfs_k_phys | nwp_gfs | 29.81 |
| indirect | 2 | 42 | 3 | fcst_gfs_tcell | nwp_gfs | 22.94 |
| indirect | 2 | 123 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 41.00 |
| indirect | 2 | 123 | 2 | fcst_gfs_k_phys | nwp_gfs | 36.46 |
| indirect | 2 | 123 | 3 | aoi | other | 20.72 |
| indirect | 2 | 777 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 43.30 |
| indirect | 2 | 777 | 2 | poa_direct_clearsky | clearsky | 28.26 |
| indirect | 2 | 777 | 3 | fcst_gfs_k_phys | nwp_gfs | 26.22 |
| direct | 2 | 42 | 1 | fcst_gfs_poa | nwp_gfs | 424.44 |
| direct | 2 | 42 | 2 | fcst_gfs_pv_kw | nwp_gfs | 195.10 |
| direct | 2 | 42 | 3 | poa_global_clearsky | clearsky | 118.99 |
| direct | 2 | 123 | 1 | fcst_gfs_poa | nwp_gfs | 396.20 |
| direct | 2 | 123 | 2 | fcst_gfs_pv_kw | nwp_gfs | 298.94 |
| direct | 2 | 123 | 3 | poa_global_clearsky | clearsky | 121.54 |
| direct | 2 | 777 | 1 | fcst_gfs_poa | nwp_gfs | 336.05 |
| direct | 2 | 777 | 2 | fcst_gfs_pv_kw | nwp_gfs | 293.13 |
| direct | 2 | 777 | 3 | poa_global_clearsky | clearsky | 203.66 |
| indirect | 3 | 42 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 47.68 |
| indirect | 3 | 42 | 2 | fcst_gfs_k_phys | nwp_gfs | 41.70 |
| indirect | 3 | 42 | 3 | fcst_gfs_mcc | nwp_gfs | 26.34 |
| indirect | 3 | 123 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 45.79 |
| indirect | 3 | 123 | 2 | fcst_gfs_k_phys | nwp_gfs | 35.45 |
| indirect | 3 | 123 | 3 | fcst_gfs_mcc | nwp_gfs | 33.49 |
| indirect | 3 | 777 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 40.93 |
| indirect | 3 | 777 | 2 | fcst_gfs_k_phys | nwp_gfs | 38.69 |
| indirect | 3 | 777 | 3 | fcst_gfs_mcc | nwp_gfs | 26.78 |
| direct | 3 | 42 | 1 | fcst_gfs_pv_kw | nwp_gfs | 417.55 |
| direct | 3 | 42 | 2 | fcst_gfs_poa | nwp_gfs | 251.62 |
| direct | 3 | 42 | 3 | poa_global_clearsky | clearsky | 114.99 |
| direct | 3 | 123 | 1 | fcst_gfs_pv_kw | nwp_gfs | 428.37 |
| direct | 3 | 123 | 2 | fcst_gfs_poa | nwp_gfs | 226.63 |
| direct | 3 | 123 | 3 | poa_global_clearsky | clearsky | 157.38 |
| direct | 3 | 777 | 1 | fcst_gfs_pv_kw | nwp_gfs | 466.67 |
| direct | 3 | 777 | 2 | fcst_gfs_poa | nwp_gfs | 244.09 |
| direct | 3 | 777 | 3 | poa_global_clearsky | clearsky | 189.81 |

## Explanation: TFT attention on the same hour of earlier days

| step | target | block | same_hour_ratio | argmax_lag_h | first_position_ratio |
|---|---|---|---|---|---|
| tft96 | indirect | all | 1.00 | -16 | 0.96 |
| tft96 | indirect | 1 | 1.00 | -16 | 0.95 |
| tft96 | indirect | 2 | 1.00 | -16 | 0.97 |
| tft96 | indirect | 3 | 1.00 | -16 | 0.97 |
| tft96 | direct | all | 1.00 | -5 | 0.89 |
| tft96 | direct | 1 | 0.99 | -5 | 0.88 |
| tft96 | direct | 2 | 1.00 | -5 | 0.89 |
| tft96 | direct | 3 | 1.00 | -5 | 0.89 |
| tft168 | indirect | all | 1.02 | -66 | 1.37 |
| tft168 | indirect | 1 | 1.02 | -114 | 1.37 |
| tft168 | indirect | 2 | 1.02 | -90 | 1.37 |
| tft168 | indirect | 3 | 1.02 | -66 | 1.37 |
| tft168 | direct | all | 1.06 | -4 | 0.77 |
| tft168 | direct | 1 | 1.06 | -4 | 0.78 |
| tft168 | direct | 2 | 1.06 | -4 | 0.77 |
| tft168 | direct | 3 | 1.06 | -4 | 0.76 |
| tftbase | indirect | all | 0.82 | -138 | 0.49 |
| tftbase | indirect | 1 | 0.83 | -138 | 0.54 |
| tftbase | indirect | 2 | 0.82 | -138 | 0.48 |
| tftbase | indirect | 3 | 0.80 | -138 | 0.45 |
| tftbase | direct | all | 1.07 | -161 | 0.94 |
| tftbase | direct | 1 | 1.08 | -161 | 0.95 |
| tftbase | direct | 2 | 1.07 | -161 | 0.94 |
| tftbase | direct | 3 | 1.07 | -161 | 0.93 |

## Explanation: the most important hyper-parameter per study (fANOVA, ten seeds)

| step | target | horizon | param | importance_mean | importance_std | n_complete | all_in_startup |
|---|---|---|---|---|---|---|---|
| tft96 | direct | all | learning_rate | 0.80 | 0.03 | 20 | False |
| xgb | direct | 3 | learning_rate | 0.70 | 0.04 | 60 | False |
| xgb | direct | 1 | min_child_weight | 0.67 | 0.06 | 60 | False |
| bilstm | indirect | all | lr | 0.66 | 0.04 | 60 | False |
| xgb | indirect | 2 | learning_rate | 0.64 | 0.05 | 60 | False |
| xgb | indirect | 3 | learning_rate | 0.57 | 0.06 | 60 | False |
| xgb | indirect | 2 | colsample_bytree | 0.56 | 0.05 | 60 | False |
| bilstm | indirect | all | n_layers | 0.54 | 0.04 | 60 | False |
| xgb | direct | 2 | learning_rate | 0.52 | 0.05 | 60 | False |
| bilstm | direct | all | lr | 0.51 | 0.03 | 60 | False |
| xgb | direct | 2 | max_depth | 0.49 | 0.04 | 60 | False |
| xgb | direct | 1 | learning_rate | 0.49 | 0.05 | 60 | False |
| xgb | indirect | 1 | max_depth | 0.48 | 0.06 | 60 | False |
| tft96 | indirect | all | attention_head_size | 0.48 | 0.04 | 20 | False |
| xgb | direct | 3 | max_depth | 0.48 | 0.07 | 60 | False |
| bilstm | direct | all | dropout | 0.47 | 0.03 | 60 | False |
| bilstm | direct | all | batch_size | 0.42 | 0.03 | 60 | False |
| xgb | indirect | 1 | learning_rate | 0.42 | 0.04 | 60 | False |
| xgb | indirect | 3 | max_depth | 0.40 | 0.06 | 60 | False |
| xgb | indirect | 3 | colsample_bytree | 0.39 | 0.08 | 60 | False |
| xgb | direct | 1 | colsample_bytree | 0.37 | 0.04 | 60 | False |
| bilstm | indirect | all | lr | 0.35 | 0.04 | 60 | False |
| tft168 | indirect | all | hidden_continuous_size | 0.34 | 0.05 | 20 | False |
| tft168 | direct | all | hidden_continuous_size | 0.33 | 0.05 | 20 | False |
| xgb | direct | 3 | learning_rate | 0.28 | 0.02 | 60 | False |
| xgb | indirect | 1 | max_depth | 0.27 | 0.05 | 60 | False |
| xgb | indirect | 2 | min_child_weight | 0.27 | 0.05 | 60 | False |
| xgb | direct | 2 | min_child_weight | 0.23 | 0.03 | 60 | False |

## Cross-validation of the xgb models on the train rows (S-38, nRMSE in kW, mean and std over five folds)

| target | horizon | seed | cv_mean | cv_std | validation_uncorrected | validation_minus_cv_pp |
|---|---|---|---|---|---|---|
| indirect | 1 | 42 | 7.40 | 2.32 | 6.77 | -0.62 |
| indirect | 1 | 123 | 7.62 | 2.40 | 6.77 | -0.84 |
| indirect | 1 | 777 | 7.40 | 2.35 | 6.76 | -0.65 |
| direct | 1 | 42 | 7.47 | 2.36 | 6.63 | -0.84 |
| direct | 1 | 123 | 7.48 | 2.21 | 6.60 | -0.88 |
| direct | 1 | 777 | 7.50 | 2.38 | 6.64 | -0.86 |
| indirect | 2 | 42 | 7.82 | 2.36 | 7.09 | -0.73 |
| indirect | 2 | 123 | 7.71 | 2.28 | 7.16 | -0.55 |
| indirect | 2 | 777 | 7.82 | 2.30 | 7.02 | -0.80 |
| direct | 2 | 42 | 7.73 | 2.04 | 6.80 | -0.94 |
| direct | 2 | 123 | 7.65 | 2.23 | 6.83 | -0.81 |
| direct | 2 | 777 | 7.74 | 2.25 | 6.80 | -0.93 |
| indirect | 3 | 42 | 8.02 | 2.40 | 6.64 | -1.38 |
| indirect | 3 | 123 | 7.90 | 2.42 | 6.70 | -1.20 |
| indirect | 3 | 777 | 8.04 | 2.57 | 6.66 | -1.38 |
| direct | 3 | 42 | 8.11 | 2.50 | 6.43 | -1.68 |
| direct | 3 | 123 | 8.01 | 2.51 | 6.45 | -1.56 |
| direct | 3 | 777 | 8.01 | 2.44 | 6.43 | -1.58 |

## Feature-group ablation of the xgb models (S-41, delta of nRMSE in kW against the full fit, pp)

With one seed (`n_seeds` 1) the margin is only the 0.05 pp floor and a verdict is provisional: the seed spread of XGB-Opt is of that size (M41).

| target | horizon | mode | n_seeds | mean_delta_pp | std_delta_pp | margin_pp | verdict |
|---|---|---|---|---|---|---|---|
| direct | 1 | minimal | 3 | 1.286 | 0.023 | 0.082 | helpful |
| direct | 1 | no_clearsky | 3 | 0.014 | 0.013 | 0.082 | neutral |
| direct | 1 | no_history | 3 | 0.057 | 0.051 | 0.082 | neutral |
| direct | 1 | no_history_other | 3 | 0.012 | 0.023 | 0.082 | neutral |
| direct | 1 | no_lag_power | 3 | 0.009 | 0.001 | 0.082 | neutral |
| direct | 1 | no_nwp_gfs | 3 | 0.977 | 0.026 | 0.082 | helpful |
| direct | 1 | no_other | 3 | 0.035 | 0.028 | 0.082 | neutral |
| direct | 1 | no_rolling_stats | 3 | 0.053 | 0.039 | 0.082 | neutral |
| direct | 1 | no_time_encodings | 3 | 0.052 | 0.092 | 0.082 | neutral |
| direct | 2 | minimal | 3 | 1.182 | 0.111 | 0.117 | helpful |
| direct | 2 | no_clearsky | 3 | 0.062 | 0.064 | 0.117 | neutral |
| direct | 2 | no_history | 3 | 0.091 | 0.075 | 0.117 | neutral |
| direct | 2 | no_history_other | 3 | 0.021 | 0.018 | 0.117 | neutral |
| direct | 2 | no_lag_power | 3 | 0.070 | 0.033 | 0.117 | neutral |
| direct | 2 | no_nwp_gfs | 3 | 0.963 | 0.063 | 0.117 | helpful |
| direct | 2 | no_other | 3 | 0.052 | 0.032 | 0.117 | neutral |
| direct | 2 | no_rolling_stats | 3 | 0.025 | 0.008 | 0.117 | neutral |
| direct | 2 | no_time_encodings | 3 | 0.043 | 0.049 | 0.117 | neutral |
| direct | 3 | minimal | 3 | 1.499 | 0.022 | 0.084 | helpful |
| direct | 3 | no_clearsky | 3 | 0.046 | 0.036 | 0.084 | neutral |
| direct | 3 | no_history | 3 | 0.124 | 0.053 | 0.084 | helpful |
| direct | 3 | no_history_other | 3 | 0.059 | 0.024 | 0.084 | neutral |
| direct | 3 | no_lag_power | 3 | 0.076 | 0.072 | 0.084 | neutral |
| direct | 3 | no_nwp_gfs | 3 | 1.441 | 0.033 | 0.084 | helpful |
| direct | 3 | no_other | 3 | 0.041 | 0.033 | 0.084 | neutral |
| direct | 3 | no_rolling_stats | 3 | 0.085 | 0.028 | 0.084 | helpful |
| direct | 3 | no_time_encodings | 3 | 0.083 | 0.050 | 0.084 | neutral |
| indirect | 1 | minimal | 3 | 1.295 | 0.010 | 0.050 | helpful |
| indirect | 1 | no_clearsky | 3 | 0.015 | 0.028 | 0.050 | neutral |
| indirect | 1 | no_history | 3 | 0.048 | 0.016 | 0.050 | neutral |
| indirect | 1 | no_history_other | 3 | 0.016 | 0.034 | 0.050 | neutral |
| indirect | 1 | no_lag_power | 3 | 0.072 | 0.018 | 0.050 | helpful |
| indirect | 1 | no_nwp_gfs | 3 | 1.272 | 0.027 | 0.050 | helpful |
| indirect | 1 | no_other | 3 | 0.016 | 0.012 | 0.050 | neutral |
| indirect | 1 | no_time_encodings | 3 | 0.008 | 0.003 | 0.050 | neutral |
| indirect | 2 | no_clearsky | 3 | 0.114 | 0.132 | 0.250 | neutral |
| indirect | 2 | no_history | 3 | 0.122 | 0.117 | 0.250 | neutral |
| indirect | 2 | no_history_other | 3 | 0.106 | 0.107 | 0.250 | neutral |
| indirect | 2 | no_lag_power | 3 | 0.141 | 0.125 | 0.250 | neutral |
| indirect | 2 | no_nwp_gfs | 3 | 1.003 | 0.139 | 0.250 | helpful |
| indirect | 2 | no_other | 3 | 0.115 | 0.124 | 0.250 | neutral |
| indirect | 2 | no_rolling_stats | 3 | 0.073 | 0.127 | 0.250 | neutral |
| indirect | 3 | no_clearsky | 3 | 0.057 | 0.053 | 0.084 | neutral |
| indirect | 3 | no_history | 3 | 0.128 | 0.033 | 0.084 | helpful |
| indirect | 3 | no_history_other | 3 | 0.080 | 0.047 | 0.084 | neutral |
| indirect | 3 | no_lag_power | 3 | 0.060 | 0.020 | 0.084 | neutral |
| indirect | 3 | no_nwp_gfs | 3 | 1.402 | 0.064 | 0.084 | helpful |
| indirect | 3 | no_other | 3 | 0.082 | 0.021 | 0.084 | neutral |
| indirect | 3 | no_rolling_stats | 3 | 0.059 | 0.034 | 0.084 | neutral |

## Points for the G-1 list (facts, no verdict)

| point | subject | fact |
|---|---|---|
| final fit against its best trial | bilstm indirect seed 42 | final 7.064 %, best trial 6.817 % (+0.247 pp) |
| final fit against its best trial | bilstm indirect seed 123 | final 6.986 %, best trial 6.748 % (+0.238 pp) |
| final fit against its best trial | bilstm indirect seed 777 | final 6.962 %, best trial 6.799 % (+0.163 pp) |
| best check against the step budget | patchtst indirect seed 42 | best step 4900 of a budget of 5000 |
| best check against the step budget | patchtst direct seed 42 | best step 4100 of a budget of 5000 |
| best value on the edge of the search space | xgb indirect_D1_seed42 | max_depth |
| best value on the edge of the search space | xgb indirect_D1_seed123 | max_depth, min_child_weight |
| best value on the edge of the search space | xgb indirect_D1_seed777 | max_depth, colsample_bytree |
| best value on the edge of the search space | xgb direct_D1_seed42 | reg_lambda |
| best value on the edge of the search space | xgb direct_D1_seed123 | n_estimators |
| best value on the edge of the search space | xgb indirect_D2_seed42 | max_depth |
| best value on the edge of the search space | xgb indirect_D2_seed123 | n_estimators, subsample |
| best value on the edge of the search space | xgb indirect_D2_seed777 | max_depth |
| best value on the edge of the search space | xgb direct_D2_seed42 | n_estimators, max_depth, colsample_bytree |
| best value on the edge of the search space | xgb direct_D2_seed123 | learning_rate, min_child_weight, reg_lambda |
| best value on the edge of the search space | xgb indirect_D3_seed777 | max_depth |
| best value on the edge of the search space | xgb direct_D3_seed123 | max_depth |
| best value on the edge of the search space | xgb direct_D3_seed777 | learning_rate |
| best value on the edge of the search space | bilstm indirect seed 42 | hidden, n_layers |
| best value on the edge of the search space | bilstm direct seed 42 | batch_size |
| best value on the edge of the search space | bilstm indirect seed 123 | hidden, n_layers, batch_size |
| best value on the edge of the search space | bilstm direct seed 123 | batch_size |
| best value on the edge of the search space | bilstm indirect seed 777 | hidden, n_layers |
| best value on the edge of the search space | bilstm direct seed 777 | hidden, n_layers, batch_size |
| best value on the edge of the search space | tft96 direct seed 42 | hidden_size, lstm_layers |
| best value on the edge of the search space | tft168 indirect seed 42 | hidden_continuous_size |
| best value on the edge of the search space | tft168 direct seed 42 | attention_head_size |
| trials stopped at their cap | xgb | 13 of 1080 trials |
| mean bias on validation | bilstm indirect seed 42 | +1.16 % of capacity (+40.0 kW) |
| mean bias on validation | bilstm direct seed 42 | +0.52 % of capacity (+17.9 kW) |
| mean bias on validation | bilstm indirect seed 123 | +1.38 % of capacity (+47.5 kW) |
| mean bias on validation | bilstm direct seed 123 | -0.15 % of capacity (-5.2 kW) |
| mean bias on validation | bilstm indirect seed 777 | +0.96 % of capacity (+33.0 kW) |
| mean bias on validation | bilstm direct seed 777 | -0.00 % of capacity (-0.1 kW) |
| mean bias on validation | tft96 indirect seed 42 | +0.83 % of capacity (+28.5 kW) |
| mean bias on validation | tft96 direct seed 42 | +0.58 % of capacity (+20.0 kW) |
| mean bias on validation | tft168 indirect seed 42 | +0.75 % of capacity (+25.8 kW) |
| mean bias on validation | tft168 direct seed 42 | +0.70 % of capacity (+24.2 kW) |
| mean bias on validation | tftbase indirect seed 42 | +0.92 % of capacity (+31.6 kW) |
| mean bias on validation | tftbase direct seed 42 | +0.68 % of capacity (+23.5 kW) |
| mean bias on validation | patchtst indirect seed 42 | +2.73 % of capacity (+94.2 kW) |
| mean bias on validation | patchtst direct seed 42 | +0.72 % of capacity (+25.0 kW) |
| margin against climatology (kW DM) | xgb_sota direct D+1 | 105 % of the minimum detectable effect, HLN p 0.0007 |
| margin against climatology (kW DM) | xgb_sota direct D+2 | 93 % of the minimum detectable effect, HLN p 0.0014 |
| margin against climatology (kW DM) | xgb_sota direct D+3 | 110 % of the minimum detectable effect, HLN p 0.0002 |
| margin against climatology (kW DM) | xgb_sota indirect D+1 | 103 % of the minimum detectable effect, HLN p 0.0010 |
| margin against climatology (kW DM) | xgb_sota indirect D+2 | 91 % of the minimum detectable effect, HLN p 0.0029 |
| margin against climatology (kW DM) | xgb_sota indirect D+3 | 105 % of the minimum detectable effect, HLN p 0.0006 |

## Sources

- `ablation/ablation_summary.csv` sha256 `94e3ae888e2c3c68`
- `bilstm/best_params.json` sha256 `cf9a2c8518f69033`
- `bilstm/mbe.csv` sha256 `3bcecfa67fc93ae4`
- `cv/cv_summary.csv` sha256 `4fb681f03f566abd`
- `evaluate/val/benchmark_summary.csv` sha256 `857d40bd24c6beb9`
- `evaluate/val/dm_tests_kw.csv` sha256 `6868f74376b1dccf`
- `explain/hparam_importance.csv` sha256 `a797199b259050df`
- `explain/tft/tft_attention_summary.csv` sha256 `b3a986d7955afbab`
- `explain/xgb/xgb_shap_features.csv` sha256 `81999b423610c540`
- `explain/xgb/xgb_shap_groups.csv` sha256 `ae2cd933b2439105`
- `patchtst/best_params.json` sha256 `5a269793a326fdca`
- `patchtst/mbe.csv` sha256 `101757ce92e70404`
- `tft168/best_params.json` sha256 `992302261d563491`
- `tft168/mbe.csv` sha256 `8f9e93978645e820`
- `tft96/best_params.json` sha256 `69dc8484d879f54d`
- `tft96/mbe.csv` sha256 `93df329e43eec2e4`
- `tftbase/best_params.json` sha256 `b386d1f1e79cb834`
- `tftbase/mbe.csv` sha256 `27bd348675aeddc5`
- `xgb/best_params.json` sha256 `a72ace571d755d81`
- `xgb/learning_curves.csv` sha256 `f109a881370587c3`
- `xgb/trials.csv` sha256 `e6ae2a4456382527`
