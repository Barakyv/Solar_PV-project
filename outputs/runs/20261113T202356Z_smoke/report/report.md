# Run report: 20261113T202356Z_smoke

Mode `smoke`. Every number below is read from a file of this run; the files and their SHA-256 are listed at the end (M31). Validation numbers are tuning-split numbers (D-10): validation chose the hyper-parameters and the stopping round.

## Steps

| step | status | commit | duration_s | n_warnings | n_resumed |
|---|---|---|---|---|---|
| select | ok | ac63bb0 | 2.8 | 0 | 0 |
| baselines | ok | ac63bb0 | 0.8 | 0 | 0 |
| xgb | ok | ac63bb0 | 22.3 | 12 | 0 |
| cv | ok | ac63bb0 | 13.9 | 30 | 0 |
| ablation | ok | ac63bb0 | 52.2 | 0 | 0 |
| bilstm | ok | ac63bb0 | 39.4 | 2 | 0 |
| tft96 | ok | ac63bb0 | 48.5 | 4 | 0 |
| tft168 | ok | ac63bb0 | 49.0 | 4 | 0 |
| tftbase | ok | ac63bb0 | 42.4 | 2 | 0 |
| patchtst | ok | ac63bb0 | 39.7 | 2 | 0 |
| evaluate | ok | ac63bb0 | 4.9 | 0 | 0 |
| explain | ok | ac63bb0 | 4.5 | 0 | 0 |

## Benchmark on val (nRMSE in kW, % of 3,449 kW)

| target | horizon | model | seed | n | nRMSE_kW_pct | skill_score_kw |
|---|---|---|---|---|---|---|
| direct | 1 | clearsky_k | -1 | 321 | 14.042 | 0.134 |
| direct | 1 | climatology | -1 | 321 | 12.781 | 0.211 |
| direct | 1 | lstm_sota | 42 | 321 | 41.592 | -1.566 |
| direct | 1 | patchtst | 42 | 321 | 15.426 | 0.048 |
| direct | 1 | persistence | -1 | 321 | 16.206 |  |
| direct | 1 | tft | 42 | 321 | 26.539 | -0.638 |
| direct | 1 | tft_168 | 42 | 321 | 48.483 | -1.992 |
| direct | 1 | tft_96 | 42 | 321 | 48.577 | -1.997 |
| direct | 1 | xgb_sota | 42 | 321 | 12.253 | 0.244 |
| direct | 2 | clearsky_k | -1 | 321 | 14.042 | 0.198 |
| direct | 2 | climatology | -1 | 321 | 12.781 | 0.270 |
| direct | 2 | lstm_sota | 42 | 321 | 41.698 | -1.382 |
| direct | 2 | patchtst | 42 | 321 | 15.999 | 0.086 |
| direct | 2 | persistence | -1 | 321 | 17.507 |  |
| direct | 2 | tft | 42 | 321 | 26.481 | -0.513 |
| direct | 2 | tft_168 | 42 | 321 | 47.529 | -1.715 |
| direct | 2 | tft_96 | 42 | 321 | 47.708 | -1.725 |
| direct | 2 | xgb_sota | 42 | 321 | 11.598 | 0.338 |
| direct | 3 | clearsky_k | -1 | 321 | 14.042 | 0.183 |
| direct | 3 | climatology | -1 | 321 | 12.781 | 0.256 |
| direct | 3 | lstm_sota | 42 | 321 | 41.801 | -1.432 |
| direct | 3 | patchtst | 42 | 321 | 16.324 | 0.050 |
| direct | 3 | persistence | -1 | 321 | 17.189 |  |
| direct | 3 | tft | 42 | 321 | 25.616 | -0.490 |
| direct | 3 | tft_168 | 42 | 321 | 48.597 | -1.827 |
| direct | 3 | tft_96 | 42 | 321 | 48.843 | -1.842 |
| direct | 3 | xgb_sota | 42 | 321 | 12.604 | 0.267 |
| indirect | 1 | clearsky_k | -1 | 321 | 14.042 | 0.134 |
| indirect | 1 | climatology | -1 | 321 | 12.884 | 0.205 |
| indirect | 1 | lstm_sota | 42 | 321 | 49.011 | -2.024 |
| indirect | 1 | patchtst | 42 | 321 | 20.278 | -0.251 |
| indirect | 1 | persistence | -1 | 321 | 16.210 |  |
| indirect | 1 | tft | 42 | 321 | 13.878 | 0.144 |
| indirect | 1 | tft_168 | 42 | 321 | 19.257 | -0.188 |
| indirect | 1 | tft_96 | 42 | 321 | 19.073 | -0.177 |
| indirect | 1 | xgb_sota | 42 | 321 | 10.086 | 0.378 |
| indirect | 2 | clearsky_k | -1 | 321 | 14.042 | 0.199 |
| indirect | 2 | climatology | -1 | 321 | 12.884 | 0.265 |
| indirect | 2 | lstm_sota | 42 | 321 | 49.011 | -1.797 |
| indirect | 2 | patchtst | 42 | 321 | 20.562 | -0.174 |
| indirect | 2 | persistence | -1 | 321 | 17.520 |  |
| indirect | 2 | tft | 42 | 321 | 13.955 | 0.203 |
| indirect | 2 | tft_168 | 42 | 321 | 19.121 | -0.091 |
| indirect | 2 | tft_96 | 42 | 321 | 18.886 | -0.078 |
| indirect | 2 | xgb_sota | 42 | 321 | 11.253 | 0.358 |
| indirect | 3 | clearsky_k | -1 | 321 | 14.042 | 0.185 |
| indirect | 3 | climatology | -1 | 321 | 12.884 | 0.252 |
| indirect | 3 | lstm_sota | 42 | 321 | 49.011 | -1.846 |
| indirect | 3 | patchtst | 42 | 321 | 18.682 | -0.085 |
| indirect | 3 | persistence | -1 | 321 | 17.221 |  |
| indirect | 3 | tft | 42 | 321 | 13.871 | 0.195 |
| indirect | 3 | tft_168 | 42 | 321 | 18.171 | -0.055 |
| indirect | 3 | tft_96 | 42 | 321 | 18.036 | -0.047 |
| indirect | 3 | xgb_sota | 42 | 321 | 11.201 | 0.350 |

## Seed spread (stage-10 rule: below 0.15 pp in kW)

| model | target | horizon | n_seeds | std_kw_pp | std_target_unit_pp |
|---|---|---|---|---|---|
| lstm_sota | direct | 1 | 1 |  |  |
| lstm_sota | direct | 2 | 1 |  |  |
| lstm_sota | direct | 3 | 1 |  |  |
| lstm_sota | indirect | 1 | 1 |  |  |
| lstm_sota | indirect | 2 | 1 |  |  |
| lstm_sota | indirect | 3 | 1 |  |  |
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
| xgb_sota | direct | 1 | 1 |  |  |
| xgb_sota | direct | 2 | 1 |  |  |
| xgb_sota | direct | 3 | 1 |  |  |
| xgb_sota | indirect | 1 | 1 |  |  |
| xgb_sota | indirect | 2 | 1 |  |  |
| xgb_sota | indirect | 3 | 1 |  |  |

## Training diagnostics

| step | subject | unit_of_best | best | cap | run | gap_ratio | value_kw |
|---|---|---|---|---|---|---|---|
| xgb | indirect_D1_seed42 | round | 22 | 874 | 73 | 4.649 | 10.086 |
| xgb | direct_D1_seed42 | round | 158 | 874 | 209 | 263.594 | 12.253 |
| xgb | indirect_D2_seed42 | round | 11 | 874 | 62 | 2.690 | 11.253 |
| xgb | direct_D2_seed42 | round | 37 | 874 | 88 | 8.642 | 11.598 |
| xgb | indirect_D3_seed42 | round | 11 | 874 | 62 | 2.674 | 11.201 |
| xgb | direct_D3_seed42 | round | 329 | 874 | 380 | 2835.508 | 12.604 |
| bilstm | indirect seed 42 | epoch | 2 | 2 | 2 | 1.042 | 49.011 |
| bilstm | direct seed 42 | epoch | 2 | 2 | 2 | 1.180 | 41.697 |
| tft96 | indirect seed 42 | epoch | 2 | 2 | 2 | 0.949 | 18.670 |
| tft96 | direct seed 42 | epoch | 2 | 2 | 2 | 0.880 | 48.378 |
| tft168 | indirect seed 42 | epoch | 2 | 2 | 2 | 0.948 | 18.856 |
| tft168 | direct seed 42 | epoch | 2 | 2 | 2 | 0.880 | 48.205 |
| tftbase | indirect seed 42 | epoch | 2 | 2 | 2 | 0.427 | 13.901 |
| tftbase | direct seed 42 | epoch | 2 | 2 | 2 | 0.782 | 26.215 |
| patchtst | indirect seed 42 | step | 10 | 20 | 19 |  | 19.858 |
| patchtst | direct seed 42 | step | 20 | 20 | 19 |  | 15.921 |

## Studies

| step | subject | n_trials | best_trial | best_value_kw | edges |
|---|---|---|---|---|---|
| xgb | indirect_D1_seed42 | 1 | 0 | 10.228 | max_depth |
| xgb | direct_D1_seed42 | 1 | 0 | 12.253 | max_depth |
| xgb | indirect_D2_seed42 | 1 | 0 | 11.426 | max_depth |
| xgb | direct_D2_seed42 | 1 | 0 | 11.598 | max_depth |
| xgb | indirect_D3_seed42 | 1 | 0 | 11.269 | max_depth |
| xgb | direct_D3_seed42 | 1 | 0 | 12.604 | max_depth |
| bilstm | indirect seed 42 | 1 | 0 | 49.011 |  |
| bilstm | direct seed 42 | 1 | 0 | 41.697 |  |
| tft96 | indirect seed 42 | 1 | 0 | 18.670 | lstm_layers |
| tft96 | direct seed 42 | 1 | 0 | 48.378 | lstm_layers |
| tft168 | indirect seed 42 | 1 | 0 | 18.856 | lstm_layers |
| tft168 | direct seed 42 | 1 | 0 | 48.205 | lstm_layers |

## Automatic warnings (56)

| step | kind | status | text |
|---|---|---|---|
| xgb | gap | matched | indirect_D1_seed42: validation RMSE 0.194 is 4.6 x the train RMSE 0.04174 at the best iteration |
| xgb | edge_space | matched | indirect_D1_seed42: best trial on the edge of the search space for max_depth = 10 |
| xgb | gap | matched | direct_D1_seed42: validation RMSE 273.8 is 263.6 x the train RMSE 1.039 at the best iteration |
| xgb | edge_space | matched | direct_D1_seed42: best trial on the edge of the search space for max_depth = 10 |
| xgb | gap | matched | indirect_D2_seed42: validation RMSE 0.2146 is 2.7 x the train RMSE 0.07976 at the best iteration |
| xgb | edge_space | matched | indirect_D2_seed42: best trial on the edge of the search space for max_depth = 10 |
| xgb | gap | matched | direct_D2_seed42: validation RMSE 254.2 is 8.6 x the train RMSE 29.42 at the best iteration |
| xgb | edge_space | matched | direct_D2_seed42: best trial on the edge of the search space for max_depth = 10 |
| xgb | gap | matched | indirect_D3_seed42: validation RMSE 0.2147 is 2.7 x the train RMSE 0.08027 at the best iteration |
| xgb | edge_space | matched | indirect_D3_seed42: best trial on the edge of the search space for max_depth = 10 |
| xgb | gap | matched | direct_D3_seed42: validation RMSE 276.3 is 2835.5 x the train RMSE 0.09745 at the best iteration |
| xgb | edge_space | matched | direct_D3_seed42: best trial on the edge of the search space for max_depth = 10 |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D1_seed42 fold 0: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D1_seed42 fold 1: 81 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D1_seed42 fold 2: 82 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D1_seed42 fold 3: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D1_seed42 fold 4: 78 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D1_seed42 fold 0: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D1_seed42 fold 1: 81 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D1_seed42 fold 2: 82 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D1_seed42 fold 3: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D1_seed42 fold 4: 78 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D2_seed42 fold 0: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D2_seed42 fold 1: 81 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D2_seed42 fold 2: 82 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D2_seed42 fold 3: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D2_seed42 fold 4: 78 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D2_seed42 fold 0: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D2_seed42 fold 1: 81 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D2_seed42 fold 2: 82 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D2_seed42 fold 3: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D2_seed42 fold 4: 78 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D3_seed42 fold 0: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D3_seed42 fold 1: 81 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D3_seed42 fold 2: 82 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D3_seed42 fold 3: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: indirect_D3_seed42 fold 4: 78 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D3_seed42 fold 0: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D3_seed42 fold 1: 81 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D3_seed42 fold 2: 82 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D3_seed42 fold 3: 90 held hours |
| cv | other | step log only | cv: a fold with fewer than 500 held hours: direct_D3_seed42 fold 4: 78 held hours |
| bilstm | last_epoch | matched | bilstm: indirect seed 42: best epoch 2 at the last epoch: it may not have converged |
| bilstm | last_epoch | matched | bilstm: direct seed 42: best epoch 2 at the last epoch: it may not have converged |
| tft96 | last_epoch | matched | tft96: indirect seed 42: best epoch 2 at the last epoch: it may not have converged |
| tft96 | edge_space | matched | tft96: indirect seed 42: best trial on the edge of the search space for lstm_layers = 3 (M23) |
| tft96 | last_epoch | matched | tft96: direct seed 42: best epoch 2 at the last epoch: it may not have converged |
| tft96 | edge_space | matched | tft96: direct seed 42: best trial on the edge of the search space for lstm_layers = 3 (M23) |
| tft168 | last_epoch | matched | tft168: indirect seed 42: best epoch 2 at the last epoch: it may not have converged |
| tft168 | edge_space | matched | tft168: indirect seed 42: best trial on the edge of the search space for lstm_layers = 3 (M23) |
| tft168 | last_epoch | matched | tft168: direct seed 42: best epoch 2 at the last epoch: it may not have converged |
| tft168 | edge_space | matched | tft168: direct seed 42: best trial on the edge of the search space for lstm_layers = 3 (M23) |
| tftbase | last_epoch | matched | tftbase: indirect seed 42: best epoch 2 at the last epoch: it may not have converged |
| tftbase | last_epoch | matched | tftbase: direct seed 42: best epoch 2 at the last epoch: it may not have converged |
| patchtst | first_check | matched | patchtst: indirect seed 42: best validation check at the first check |
| patchtst | last_step | matched | patchtst: direct seed 42: best validation check at the last step: it may not have converged |

## Explanation: the three largest mean |SHAP| (kW) per booster

| target | horizon | seed | rank | feature | group | mean_abs_kw |
|---|---|---|---|---|---|---|
| indirect | 1 | 42 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 56.49 |
| indirect | 1 | 42 | 2 | fcst_gfs_tcc | nwp_gfs | 40.18 |
| indirect | 1 | 42 | 3 | fcst_gfs_lcc | nwp_gfs | 39.57 |
| direct | 1 | 42 | 1 | fcst_gfs_ghi | nwp_gfs | 315.96 |
| direct | 1 | 42 | 2 | fcst_gfs_poa | nwp_gfs | 231.47 |
| direct | 1 | 42 | 3 | ghi_clearsky | clearsky | 149.27 |
| indirect | 2 | 42 | 1 | fcst_gfs_pwat | nwp_gfs | 38.88 |
| indirect | 2 | 42 | 2 | fcst_gfs_kt_poa_daymean | nwp_gfs | 38.31 |
| indirect | 2 | 42 | 3 | fcst_gfs_lcc | nwp_gfs | 31.61 |
| direct | 2 | 42 | 1 | ghi_clearsky | clearsky | 344.53 |
| direct | 2 | 42 | 2 | fcst_gfs_ghi | nwp_gfs | 160.53 |
| direct | 2 | 42 | 3 | fcst_gfs_poa | nwp_gfs | 122.25 |
| indirect | 3 | 42 | 1 | fcst_gfs_pwat | nwp_gfs | 50.72 |
| indirect | 3 | 42 | 2 | fcst_gfs_kt_poa_daymean | nwp_gfs | 29.96 |
| indirect | 3 | 42 | 3 | fcst_gfs_lcc | nwp_gfs | 29.04 |
| direct | 3 | 42 | 1 | s01_inv1_rolling_mean_7d_same_tod | rolling_stats | 352.36 |
| direct | 3 | 42 | 2 | ghi_clearsky | clearsky | 223.55 |
| direct | 3 | 42 | 3 | fcst_gfs_ghi | nwp_gfs | 94.75 |

## Explanation: TFT attention on the same hour of earlier days

| step | target | block | same_hour_ratio | argmax_lag_h | first_position_ratio |
|---|---|---|---|---|---|
| tft96 | indirect | all | 0.93 | -20 | 0.81 |
| tft96 | indirect | 1 | 0.92 | -20 | 0.83 |
| tft96 | indirect | 2 | 0.93 | -41 | 0.81 |
| tft96 | indirect | 3 | 0.94 | -17 | 0.77 |
| tft96 | direct | all | 0.92 | -7 | 0.91 |
| tft96 | direct | 1 | 0.91 | -31 | 0.90 |
| tft96 | direct | 2 | 0.92 | -7 | 0.90 |
| tft96 | direct | 3 | 0.92 | -79 | 0.94 |
| tft168 | indirect | all | 0.93 | -164 | 0.81 |
| tft168 | indirect | 1 | 0.92 | -65 | 0.81 |
| tft168 | indirect | 2 | 0.93 | -164 | 0.80 |
| tft168 | indirect | 3 | 0.95 | -140 | 0.81 |
| tft168 | direct | all | 0.91 | -7 | 1.00 |
| tft168 | direct | 1 | 0.91 | -31 | 0.94 |
| tft168 | direct | 2 | 0.91 | -103 | 1.01 |
| tft168 | direct | 3 | 0.91 | -152 | 1.05 |
| tftbase | indirect | all | 1.09 | -58 | 0.93 |
| tftbase | indirect | 1 | 1.09 | -58 | 0.93 |
| tftbase | indirect | 2 | 1.09 | -154 | 0.92 |
| tftbase | indirect | 3 | 1.09 | -58 | 0.95 |
| tftbase | direct | all | 1.01 | -83 | 1.49 |
| tftbase | direct | 1 | 1.01 | -107 | 1.47 |
| tftbase | direct | 2 | 1.01 | -83 | 1.50 |
| tftbase | direct | 3 | 1.00 | -59 | 1.50 |

## Cross-validation of the xgb models on the train rows (S-38, nRMSE in kW, mean and std over five folds)

| target | horizon | seed | cv_mean | cv_std | validation_uncorrected | validation_minus_cv_pp |
|---|---|---|---|---|---|---|
| indirect | 1 | 42 | 9.91 | 2.42 | 10.23 | 0.32 |
| direct | 1 | 42 | 10.25 | 2.10 | 12.25 | 2.00 |
| indirect | 2 | 42 | 10.17 | 1.90 | 11.43 | 1.26 |
| direct | 2 | 42 | 10.32 | 2.14 | 11.60 | 1.28 |
| indirect | 3 | 42 | 10.44 | 2.04 | 11.27 | 0.83 |
| direct | 3 | 42 | 11.90 | 2.88 | 12.60 | 0.71 |

## Feature-group ablation of the xgb models (S-41, delta of nRMSE in kW against the full fit, pp)

With one seed (`n_seeds` 1) the margin is only the 0.05 pp floor and a verdict is provisional: the seed spread of XGB-Opt is of that size (M41).

| target | horizon | mode | n_seeds | mean_delta_pp | std_delta_pp | margin_pp | verdict |
|---|---|---|---|---|---|---|---|
| direct | 1 | minimal | 1 | 3.857 |  | 0.050 | helpful |
| direct | 1 | no_clearsky | 1 | 0.169 |  | 0.050 | helpful |
| direct | 1 | no_history | 1 | -1.708 |  | 0.050 | harmful |
| direct | 1 | no_history_other | 1 | 0.266 |  | 0.050 | helpful |
| direct | 1 | no_lag_power | 1 | -0.830 |  | 0.050 | harmful |
| direct | 1 | no_nwp_gfs | 1 | 3.268 |  | 0.050 | helpful |
| direct | 1 | no_other | 1 | 0.341 |  | 0.050 | helpful |
| direct | 1 | no_rolling_stats | 1 | 0.471 |  | 0.050 | helpful |
| direct | 1 | no_time_encodings | 1 | -0.159 |  | 0.050 | harmful |
| direct | 2 | minimal | 1 | 4.512 |  | 0.050 | helpful |
| direct | 2 | no_clearsky | 1 | 0.080 |  | 0.050 | helpful |
| direct | 2 | no_history | 1 | 0.239 |  | 0.050 | helpful |
| direct | 2 | no_history_other | 1 | -0.150 |  | 0.050 | harmful |
| direct | 2 | no_lag_power | 1 | -0.015 |  | 0.050 | neutral |
| direct | 2 | no_nwp_gfs | 1 | 4.491 |  | 0.050 | helpful |
| direct | 2 | no_other | 1 | -0.005 |  | 0.050 | neutral |
| direct | 2 | no_rolling_stats | 1 | 0.698 |  | 0.050 | helpful |
| direct | 2 | no_time_encodings | 1 | -0.046 |  | 0.050 | neutral |
| direct | 3 | minimal | 1 | 3.506 |  | 0.050 | helpful |
| direct | 3 | no_clearsky | 1 | -0.056 |  | 0.050 | harmful |
| direct | 3 | no_history | 1 | -0.052 |  | 0.050 | harmful |
| direct | 3 | no_history_other | 1 | -0.134 |  | 0.050 | harmful |
| direct | 3 | no_lag_power | 1 | -0.086 |  | 0.050 | harmful |
| direct | 3 | no_nwp_gfs | 1 | 2.890 |  | 0.050 | helpful |
| direct | 3 | no_other | 1 | -1.004 |  | 0.050 | harmful |
| direct | 3 | no_rolling_stats | 1 | -0.467 |  | 0.050 | harmful |
| direct | 3 | no_time_encodings | 1 | -0.873 |  | 0.050 | harmful |
| indirect | 1 | minimal | 1 | 2.369 |  | 0.050 | helpful |
| indirect | 1 | no_clearsky | 1 | 0.737 |  | 0.050 | helpful |
| indirect | 1 | no_history | 1 | 0.524 |  | 0.050 | helpful |
| indirect | 1 | no_history_other | 1 | 0.400 |  | 0.050 | helpful |
| indirect | 1 | no_lag_power | 1 | 0.434 |  | 0.050 | helpful |
| indirect | 1 | no_nwp_gfs | 1 | 2.606 |  | 0.050 | helpful |
| indirect | 1 | no_other | 1 | 0.420 |  | 0.050 | helpful |
| indirect | 1 | no_time_encodings | 1 | 0.293 |  | 0.050 | helpful |
| indirect | 2 | no_clearsky | 1 | 1.167 |  | 0.050 | helpful |
| indirect | 2 | no_history | 1 | -0.074 |  | 0.050 | harmful |
| indirect | 2 | no_history_other | 1 | -0.007 |  | 0.050 | neutral |
| indirect | 2 | no_lag_power | 1 | 0.308 |  | 0.050 | helpful |
| indirect | 2 | no_nwp_gfs | 1 | 1.377 |  | 0.050 | helpful |
| indirect | 2 | no_other | 1 | 0.509 |  | 0.050 | helpful |
| indirect | 2 | no_rolling_stats | 1 | -0.081 |  | 0.050 | harmful |
| indirect | 3 | no_clearsky | 1 | -0.118 |  | 0.050 | harmful |
| indirect | 3 | no_history | 1 | -0.106 |  | 0.050 | harmful |
| indirect | 3 | no_history_other | 1 | -0.341 |  | 0.050 | harmful |
| indirect | 3 | no_lag_power | 1 | 0.050 |  | 0.050 | neutral |
| indirect | 3 | no_nwp_gfs | 1 | 1.692 |  | 0.050 | helpful |
| indirect | 3 | no_other | 1 | -0.127 |  | 0.050 | harmful |
| indirect | 3 | no_rolling_stats | 1 | 0.950 |  | 0.050 | helpful |

## Points for the G-1 list (facts, no verdict)

| point | subject | fact |
|---|---|---|
| best check against the step budget | patchtst indirect seed 42 | best step 10 of a budget of 20 |
| best check against the step budget | patchtst direct seed 42 | best step 20 of a budget of 20 |
| best value on the edge of the search space | xgb indirect_D1_seed42 | max_depth |
| best value on the edge of the search space | xgb direct_D1_seed42 | max_depth |
| best value on the edge of the search space | xgb indirect_D2_seed42 | max_depth |
| best value on the edge of the search space | xgb direct_D2_seed42 | max_depth |
| best value on the edge of the search space | xgb indirect_D3_seed42 | max_depth |
| best value on the edge of the search space | xgb direct_D3_seed42 | max_depth |
| best value on the edge of the search space | tft96 indirect seed 42 | lstm_layers |
| best value on the edge of the search space | tft96 direct seed 42 | lstm_layers |
| best value on the edge of the search space | tft168 indirect seed 42 | lstm_layers |
| best value on the edge of the search space | tft168 direct seed 42 | lstm_layers |
| study inside TPE's random start-up | xgb indirect_D1_seed42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | xgb direct_D1_seed42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | xgb indirect_D2_seed42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | xgb direct_D2_seed42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | xgb indirect_D3_seed42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | xgb direct_D3_seed42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | bilstm indirect seed 42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | bilstm direct seed 42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | tft96 indirect seed 42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | tft96 direct seed 42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | tft168 indirect seed 42 | 1 trials (start-up 10) |
| study inside TPE's random start-up | tft168 direct seed 42 | 1 trials (start-up 10) |
| mean bias on validation | bilstm indirect seed 42 | -43.29 % of capacity (-1493.2 kW) |
| mean bias on validation | bilstm direct seed 42 | -35.25 % of capacity (-1215.7 kW) |
| mean bias on validation | tft96 indirect seed 42 | -4.73 % of capacity (-163.2 kW) |
| mean bias on validation | tft96 direct seed 42 | +16.88 % of capacity (+582.1 kW) |
| mean bias on validation | tft168 indirect seed 42 | -4.75 % of capacity (-164.0 kW) |
| mean bias on validation | tft168 direct seed 42 | +16.35 % of capacity (+563.8 kW) |
| mean bias on validation | tftbase indirect seed 42 | +5.94 % of capacity (+205.0 kW) |
| mean bias on validation | tftbase direct seed 42 | -2.39 % of capacity (-82.3 kW) |
| mean bias on validation | patchtst indirect seed 42 | +3.26 % of capacity (+112.5 kW) |
| mean bias on validation | patchtst direct seed 42 | +0.32 % of capacity (+11.1 kW) |
| margin against climatology (kW DM) | xgb_sota direct D+1 | 17 % of the minimum detectable effect, HLN p 0.7723 |
| margin against climatology (kW DM) | xgb_sota direct D+2 | 32 % of the minimum detectable effect, HLN p 0.4594 |
| margin against climatology (kW DM) | xgb_sota direct D+3 | 6 % of the minimum detectable effect, HLN p 0.9068 |
| margin against climatology (kW DM) | xgb_sota indirect D+1 | 96 % of the minimum detectable effect, HLN p 0.0679 |
| margin against climatology (kW DM) | xgb_sota indirect D+2 | 78 % of the minimum detectable effect, HLN p 0.1076 |
| margin against climatology (kW DM) | xgb_sota indirect D+3 | 64 % of the minimum detectable effect, HLN p 0.1048 |

## Sources

- `ablation/ablation_summary.csv` sha256 `f9dcdd39227dfb5a`
- `bilstm/best_params.json` sha256 `906163589d087d33`
- `bilstm/mbe.csv` sha256 `18d5ba4e01083f1b`
- `cv/cv_summary.csv` sha256 `19f38ed5b56212c3`
- `evaluate/val/benchmark_summary.csv` sha256 `9fcb7e56d81965a5`
- `evaluate/val/dm_tests_kw.csv` sha256 `83eb713ceb1da8b3`
- `explain/hparam_importance.csv` sha256 `c8218883994ad74a`
- `explain/tft/tft_attention_summary.csv` sha256 `d62988eab15662c4`
- `explain/xgb/xgb_shap_features.csv` sha256 `6ec516ec48624d01`
- `explain/xgb/xgb_shap_groups.csv` sha256 `da27c2a9d4acccd0`
- `patchtst/best_params.json` sha256 `f9229bc750803540`
- `patchtst/mbe.csv` sha256 `45b80b024db1cc79`
- `tft168/best_params.json` sha256 `32554f3b87b72d40`
- `tft168/mbe.csv` sha256 `06c9309b1cf2892f`
- `tft96/best_params.json` sha256 `54256ad321d209b6`
- `tft96/mbe.csv` sha256 `e276c5959d42c94a`
- `tftbase/best_params.json` sha256 `f7a7f2e430b7ec79`
- `tftbase/mbe.csv` sha256 `a7567466007fa7e6`
- `xgb/best_params.json` sha256 `98bbb2e802fd90b5`
- `xgb/learning_curves.csv` sha256 `1971fa2fff9929a9`
- `xgb/trials.csv` sha256 `6634d9b57b1eaf9e`
