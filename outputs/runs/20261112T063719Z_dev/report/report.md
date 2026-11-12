# Run report: 20261112T063719Z_dev

Mode `dev`. Every number below is read from a file of this run; the files and their SHA-256 are listed at the end (M31). Validation numbers are tuning-split numbers (D-10): validation chose the hyper-parameters and the stopping round.

## Steps

| step | status | commit | duration_s | n_warnings |
|---|---|---|---|---|
| select | ok | 46881ae | 2.8 | 0 |
| baselines | ok | 46881ae | 2.0 | 0 |
| xgb | ok | 46881ae | 321.9 | 5 |
| evaluate | ok | 46881ae | 9.5 | 0 |
| explain | ok | 46881ae | 56.9 | 0 |

## Benchmark on val (nRMSE in kW, % of 3,449 kW)

| target | horizon | model | seed | n | nRMSE_kW_pct | skill_score_kw |
|---|---|---|---|---|---|---|
| direct | 1 | clearsky_k | -1 | 2901 | 8.959 | 0.074 |
| direct | 1 | climatology | -1 | 2901 | 8.041 | 0.169 |
| direct | 1 | persistence | -1 | 2901 | 9.676 |  |
| direct | 1 | xgb_sota | 42 | 2901 | 6.656 | 0.312 |
| direct | 2 | clearsky_k | -1 | 2901 | 8.959 | 0.172 |
| direct | 2 | climatology | -1 | 2901 | 8.041 | 0.257 |
| direct | 2 | persistence | -1 | 2901 | 10.822 |  |
| direct | 2 | xgb_sota | 42 | 2901 | 6.811 | 0.371 |
| direct | 3 | clearsky_k | -1 | 2901 | 8.959 | 0.164 |
| direct | 3 | climatology | -1 | 2901 | 8.041 | 0.250 |
| direct | 3 | persistence | -1 | 2901 | 10.720 |  |
| direct | 3 | xgb_sota | 42 | 2901 | 6.426 | 0.401 |
| indirect | 1 | clearsky_k | -1 | 2901 | 8.959 | 0.075 |
| indirect | 1 | climatology | -1 | 2901 | 8.062 | 0.167 |
| indirect | 1 | persistence | -1 | 2901 | 9.683 |  |
| indirect | 1 | xgb_sota | 42 | 2901 | 6.686 | 0.310 |
| indirect | 2 | clearsky_k | -1 | 2901 | 8.959 | 0.174 |
| indirect | 2 | climatology | -1 | 2901 | 8.062 | 0.256 |
| indirect | 2 | persistence | -1 | 2901 | 10.841 |  |
| indirect | 2 | xgb_sota | 42 | 2901 | 6.911 | 0.363 |
| indirect | 3 | clearsky_k | -1 | 2901 | 8.959 | 0.167 |
| indirect | 3 | climatology | -1 | 2901 | 8.062 | 0.251 |
| indirect | 3 | persistence | -1 | 2901 | 10.759 |  |
| indirect | 3 | xgb_sota | 42 | 2901 | 6.555 | 0.391 |

## Seed spread (stage-10 rule: below 0.15 pp in kW)

| model | target | horizon | n_seeds | std_kw_pp | std_target_unit_pp |
|---|---|---|---|---|---|
| xgb_sota | direct | 1 | 1 |  |  |
| xgb_sota | direct | 2 | 1 |  |  |
| xgb_sota | direct | 3 | 1 |  |  |
| xgb_sota | indirect | 1 | 1 |  |  |
| xgb_sota | indirect | 2 | 1 |  |  |
| xgb_sota | indirect | 3 | 1 |  |  |

## Training diagnostics

| step | subject | unit_of_best | best | cap | run | gap_ratio | value_kw |
|---|---|---|---|---|---|---|---|
| xgb | indirect_D1_seed42 | round | 340 | 699 | 391 | 2.662 | 6.686 |
| xgb | direct_D1_seed42 | round | 45 | 1110 | 96 | 1.428 | 6.656 |
| xgb | indirect_D2_seed42 | round | 29 | 871 | 80 | 1.225 | 6.911 |
| xgb | direct_D2_seed42 | round | 212 | 213 | 213 | 1.236 | 6.811 |
| xgb | indirect_D3_seed42 | round | 92 | 928 | 143 | 1.837 | 6.555 |
| xgb | direct_D3_seed42 | round | 31 | 1011 | 82 | 1.200 | 6.426 |

## Studies

| step | subject | n_trials | best_trial | best_value_kw | edges |
|---|---|---|---|---|---|
| xgb | indirect_D1_seed42 | 20 | 18 | 6.792 | max_depth, learning_rate |
| xgb | direct_D1_seed42 | 20 | 19 | 6.656 | min_child_weight |
| xgb | indirect_D2_seed42 | 20 | 17 | 7.122 |  |
| xgb | direct_D2_seed42 | 20 | 10 | 6.811 | n_estimators |
| xgb | indirect_D3_seed42 | 20 | 17 | 6.656 |  |
| xgb | direct_D3_seed42 | 20 | 10 | 6.426 |  |

## Automatic warnings (5)

| step | kind | status | text |
|---|---|---|---|
| xgb | gap | matched | indirect_D1_seed42: validation RMSE 0.1004 is 2.7 x the train RMSE 0.03769 at the best iteration |
| xgb | edge_space | matched | indirect_D1_seed42: best trial on the edge of the search space for max_depth = 10, learning_rate = 0.01031 |
| xgb | edge_space | matched | direct_D1_seed42: best trial on the edge of the search space for min_child_weight = 20 |
| xgb | cap | matched | direct_D2_seed42: best iteration 212 at the cap of 213 rounds: it may not have converged (M12d) |
| xgb | edge_space | matched | direct_D2_seed42: best trial on the edge of the search space for n_estimators = 213 |

## Explanation: the three largest mean |SHAP| (kW) per booster

| target | horizon | seed | rank | feature | group | mean_abs_kw |
|---|---|---|---|---|---|---|
| indirect | 1 | 42 | 1 | fcst_gfs_k_phys | nwp_gfs | 45.36 |
| indirect | 1 | 42 | 2 | fcst_gfs_kt_poa_daymean | nwp_gfs | 41.77 |
| indirect | 1 | 42 | 3 | fcst_gfs_lcc | nwp_gfs | 20.87 |
| direct | 1 | 42 | 1 | fcst_gfs_poa | nwp_gfs | 455.92 |
| direct | 1 | 42 | 2 | fcst_gfs_pv_kw | nwp_gfs | 385.84 |
| direct | 1 | 42 | 3 | poa_global_clearsky | clearsky | 67.88 |
| indirect | 2 | 42 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 48.11 |
| indirect | 2 | 42 | 2 | fcst_gfs_k_phys | nwp_gfs | 36.05 |
| indirect | 2 | 42 | 3 | fcst_gfs_lcc | nwp_gfs | 19.72 |
| direct | 2 | 42 | 1 | fcst_gfs_poa | nwp_gfs | 443.55 |
| direct | 2 | 42 | 2 | fcst_gfs_pv_kw | nwp_gfs | 204.31 |
| direct | 2 | 42 | 3 | poa_global_clearsky | clearsky | 125.65 |
| indirect | 3 | 42 | 1 | fcst_gfs_kt_poa_daymean | nwp_gfs | 46.82 |
| indirect | 3 | 42 | 2 | fcst_gfs_k_phys | nwp_gfs | 40.47 |
| indirect | 3 | 42 | 3 | fcst_gfs_mcc | nwp_gfs | 23.89 |
| direct | 3 | 42 | 1 | fcst_gfs_pv_kw | nwp_gfs | 417.55 |
| direct | 3 | 42 | 2 | fcst_gfs_poa | nwp_gfs | 251.62 |
| direct | 3 | 42 | 3 | poa_global_clearsky | clearsky | 114.99 |

## Explanation: the most important hyper-parameter per study (fANOVA, ten seeds)

| step | target | horizon | param | importance_mean | importance_std | n_complete | all_in_startup |
|---|---|---|---|---|---|---|---|
| xgb | direct | 1 | min_child_weight | 0.71 | 0.05 | 20 | False |
| xgb | direct | 3 | learning_rate | 0.68 | 0.06 | 20 | False |
| xgb | indirect | 3 | max_depth | 0.38 | 0.05 | 20 | False |
| xgb | indirect | 1 | learning_rate | 0.37 | 0.05 | 20 | False |
| xgb | indirect | 2 | learning_rate | 0.26 | 0.05 | 20 | False |
| xgb | direct | 2 | colsample_bytree | 0.25 | 0.03 | 20 | False |

## Points for the G-1 list (facts, no verdict)

| point | subject | fact |
|---|---|---|
| best value on the edge of the search space | xgb indirect_D1_seed42 | max_depth, learning_rate |
| best value on the edge of the search space | xgb direct_D1_seed42 | min_child_weight |
| best value on the edge of the search space | xgb direct_D2_seed42 | n_estimators |
| trials stopped at their cap | xgb | 3 of 120 trials |
| margin against climatology (kW DM) | xgb_sota direct D+1 | 103 % of the minimum detectable effect, HLN p 0.0008 |
| margin against climatology (kW DM) | xgb_sota direct D+2 | 95 % of the minimum detectable effect, HLN p 0.0013 |
| margin against climatology (kW DM) | xgb_sota direct D+3 | 110 % of the minimum detectable effect, HLN p 0.0002 |
| margin against climatology (kW DM) | xgb_sota indirect D+1 | 101 % of the minimum detectable effect, HLN p 0.0013 |
| margin against climatology (kW DM) | xgb_sota indirect D+2 | 93 % of the minimum detectable effect, HLN p 0.0024 |
| margin against climatology (kW DM) | xgb_sota indirect D+3 | 104 % of the minimum detectable effect, HLN p 0.0007 |

## Sources

- `evaluate/val/benchmark_summary.csv` sha256 `eedd2ac0732c03c7`
- `evaluate/val/dm_tests_kw.csv` sha256 `a4297f30ab1bfdf6`
- `explain/hparam_importance.csv` sha256 `1748710b387ded99`
- `explain/xgb/xgb_shap_features.csv` sha256 `117836f49c510304`
- `explain/xgb/xgb_shap_groups.csv` sha256 `40a79a23c90f8146`
- `xgb/best_params.json` sha256 `1aa1e9221d6c64c6`
- `xgb/learning_curves.csv` sha256 `57aaccf917f57cdd`
- `xgb/trials.csv` sha256 `5cabaf9bb5d0d225`
