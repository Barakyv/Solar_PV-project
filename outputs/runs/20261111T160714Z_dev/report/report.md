# Run report: 20261111T160714Z_dev

Mode `dev`. Every number below is read from a file of this run; the files and their SHA-256 are listed at the end (M31). Validation numbers are tuning-split numbers (D-10): validation chose the hyper-parameters and the stopping round.

## Steps

| step | status | commit | duration_s | n_warnings |
|---|---|---|---|---|
| baselines | ok | 6a1ffdb | 2.3 | 0 |
| bilstm | ok | 6a1ffdb | 572.0 | 1 |
| tft96 | ok | 04764aa | 4327.2 | 2 |
| tft168 | ok | c9a75bc | 5180.8 | 1 |
| tftbase | ok | c9a75bc | 673.9 | 0 |
| patchtst | ok | c9a75bc | 786.0 | 0 |
| evaluate | ok | c9a75bc | 19.4 | 0 |
| select | ok | 46881ae | 2.8 | 0 |
| explain | ok | 46881ae | 18.8 | 0 |

## Benchmark on val (nRMSE in kW, % of 3,449 kW)

| target | horizon | model | seed | n | nRMSE_kW_pct | skill_score_kw |
|---|---|---|---|---|---|---|
| direct | 1 | clearsky_k | -1 | 2901 | 8.959 | 0.074 |
| direct | 1 | climatology | -1 | 2901 | 8.041 | 0.169 |
| direct | 1 | lstm_sota | 42 | 2901 | 6.751 | 0.302 |
| direct | 1 | patchtst | 42 | 2901 | 8.090 | 0.164 |
| direct | 1 | persistence | -1 | 2901 | 9.676 |  |
| direct | 1 | tft | 42 | 2901 | 6.873 | 0.290 |
| direct | 1 | tft_168 | 42 | 2901 | 6.789 | 0.298 |
| direct | 1 | tft_96 | 42 | 2901 | 6.748 | 0.303 |
| direct | 2 | clearsky_k | -1 | 2901 | 8.959 | 0.172 |
| direct | 2 | climatology | -1 | 2901 | 8.041 | 0.257 |
| direct | 2 | lstm_sota | 42 | 2901 | 6.953 | 0.357 |
| direct | 2 | patchtst | 42 | 2901 | 8.266 | 0.236 |
| direct | 2 | persistence | -1 | 2901 | 10.822 |  |
| direct | 2 | tft | 42 | 2901 | 7.042 | 0.349 |
| direct | 2 | tft_168 | 42 | 2901 | 7.013 | 0.352 |
| direct | 2 | tft_96 | 42 | 2901 | 6.941 | 0.359 |
| direct | 3 | clearsky_k | -1 | 2901 | 8.959 | 0.164 |
| direct | 3 | climatology | -1 | 2901 | 8.041 | 0.250 |
| direct | 3 | lstm_sota | 42 | 2901 | 6.508 | 0.393 |
| direct | 3 | patchtst | 42 | 2901 | 8.292 | 0.227 |
| direct | 3 | persistence | -1 | 2901 | 10.720 |  |
| direct | 3 | tft | 42 | 2901 | 6.714 | 0.374 |
| direct | 3 | tft_168 | 42 | 2901 | 6.650 | 0.380 |
| direct | 3 | tft_96 | 42 | 2901 | 6.574 | 0.387 |
| indirect | 1 | clearsky_k | -1 | 2901 | 8.959 | 0.075 |
| indirect | 1 | climatology | -1 | 2901 | 8.062 | 0.167 |
| indirect | 1 | lstm_sota | 42 | 2901 | 6.971 | 0.280 |
| indirect | 1 | patchtst | 42 | 2901 | 8.766 | 0.095 |
| indirect | 1 | persistence | -1 | 2901 | 9.683 |  |
| indirect | 1 | tft | 42 | 2901 | 6.834 | 0.294 |
| indirect | 1 | tft_168 | 42 | 2901 | 6.769 | 0.301 |
| indirect | 1 | tft_96 | 42 | 2901 | 6.777 | 0.300 |
| indirect | 2 | clearsky_k | -1 | 2901 | 8.959 | 0.174 |
| indirect | 2 | climatology | -1 | 2901 | 8.062 | 0.256 |
| indirect | 2 | lstm_sota | 42 | 2901 | 7.168 | 0.339 |
| indirect | 2 | patchtst | 42 | 2901 | 8.904 | 0.179 |
| indirect | 2 | persistence | -1 | 2901 | 10.841 |  |
| indirect | 2 | tft | 42 | 2901 | 7.097 | 0.345 |
| indirect | 2 | tft_168 | 42 | 2901 | 7.048 | 0.350 |
| indirect | 2 | tft_96 | 42 | 2901 | 6.960 | 0.358 |
| indirect | 3 | clearsky_k | -1 | 2901 | 8.959 | 0.167 |
| indirect | 3 | climatology | -1 | 2901 | 8.062 | 0.251 |
| indirect | 3 | lstm_sota | 42 | 2901 | 6.873 | 0.361 |
| indirect | 3 | patchtst | 42 | 2901 | 8.860 | 0.177 |
| indirect | 3 | persistence | -1 | 2901 | 10.759 |  |
| indirect | 3 | tft | 42 | 2901 | 6.809 | 0.367 |
| indirect | 3 | tft_168 | 42 | 2901 | 6.660 | 0.381 |
| indirect | 3 | tft_96 | 42 | 2901 | 6.684 | 0.379 |

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

## Training diagnostics

| step | subject | unit_of_best | best | cap | run | gap_ratio | value_kw |
|---|---|---|---|---|---|---|---|
| bilstm | indirect seed 42 | epoch | 51 | 60 | 60 | 0.723 | 7.005 |
| bilstm | direct seed 42 | epoch | 17 | 60 | 27 | 0.719 | 6.740 |
| tft96 | indirect seed 42 | epoch | 16 | 100 | 26 | 0.716 | 6.808 |
| tft96 | direct seed 42 | epoch | 32 | 100 | 42 | 0.677 | 6.756 |
| tft168 | indirect seed 42 | epoch | 26 | 100 | 36 | 0.726 | 6.827 |
| tft168 | direct seed 42 | epoch | 41 | 100 | 51 | 0.743 | 6.819 |
| tftbase | indirect seed 42 | epoch | 18 | 150 | 28 | 0.741 | 6.915 |
| tftbase | direct seed 42 | epoch | 58 | 150 | 68 | 0.771 | 6.878 |
| patchtst | indirect seed 42 | step | 4900 | 5000 | 4999 |  | 8.844 |
| patchtst | direct seed 42 | step | 4100 | 5000 | 4999 |  | 8.216 |

## Studies

| step | subject | n_trials | best_trial | best_value_kw | edges |
|---|---|---|---|---|---|
| bilstm | indirect seed 42 | 20 | 19 | 6.844 | hidden, n_layers, batch_size |
| bilstm | direct seed 42 | 20 | 13 | 6.740 |  |
| tft96 | indirect seed 42 | 7 | 4 | 6.808 | hidden_size, attention_head_size |
| tft96 | direct seed 42 | 7 | 4 | 6.756 | hidden_size, attention_head_size |
| tft168 | indirect seed 42 | 7 | 2 | 6.827 | hidden_continuous_size |
| tft168 | direct seed 42 | 7 | 5 | 6.819 | attention_head_size |

## Automatic warnings (5)

| step | kind | status | text |
|---|---|---|---|
| bilstm | edge_space | matched | bilstm: indirect seed 42: best trial on the edge of the search space for hidden = 256, n_layers = 1, batch_size = 128 (M23) |
| tft96 | edge_space | matched | tft96: indirect seed 42: best trial on the edge of the search space for hidden_size = 64, attention_head_size = 4 (M23) |
| tft96 | edge_space | matched | tft96: direct seed 42: best trial on the edge of the search space for hidden_size = 64, attention_head_size = 4 (M23) |
| tft168 | edge_space | matched | tft168: direct seed 42: best trial on the edge of the search space for attention_head_size = 4 (M23) |
| tft168 | edge_space | recomputed only | tft168: indirect seed 42: best trial on the edge of the search space for hidden_continuous_size = 32 (M23) |

## Explanation: TFT attention on the same hour of earlier days

| step | target | block | same_hour_ratio | argmax_lag_h | first_position_ratio |
|---|---|---|---|---|---|
| tft96 | indirect | all | 0.95 | -84 | 1.15 |
| tft96 | indirect | 1 | 0.96 | -84 | 1.17 |
| tft96 | indirect | 2 | 0.94 | -84 | 1.13 |
| tft96 | indirect | 3 | 0.94 | -84 | 1.13 |
| tft96 | direct | all | 0.95 | -12 | 1.08 |
| tft96 | direct | 1 | 0.95 | -12 | 1.08 |
| tft96 | direct | 2 | 0.95 | -12 | 1.08 |
| tft96 | direct | 3 | 0.95 | -12 | 1.08 |
| tft168 | indirect | all | 0.98 | -168 | 1.28 |
| tft168 | indirect | 1 | 0.97 | -168 | 1.27 |
| tft168 | indirect | 2 | 0.98 | -168 | 1.28 |
| tft168 | indirect | 3 | 0.98 | -168 | 1.28 |
| tft168 | direct | all | 0.98 | -30 | 1.11 |
| tft168 | direct | 1 | 0.98 | -54 | 1.10 |
| tft168 | direct | 2 | 0.98 | -30 | 1.11 |
| tft168 | direct | 3 | 0.98 | -6 | 1.12 |
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
| tft168 | indirect | all | attention_head_size | 0.57 | 0.07 | 7 | True |
| tft96 | indirect | all | hidden_continuous_size | 0.47 | 0.03 | 7 | True |
| tft96 | direct | all | learning_rate | 0.45 | 0.07 | 7 | True |
| bilstm | indirect | all | n_layers | 0.43 | 0.06 | 20 | False |
| tft168 | direct | all | learning_rate | 0.35 | 0.07 | 7 | True |
| bilstm | direct | all | batch_size | 0.34 | 0.05 | 20 | False |

## Points for the G-1 list (facts, no verdict)

| point | subject | fact |
|---|---|---|
| final fit against its best trial | bilstm indirect seed 42 | final 7.005 %, best trial 6.844 % (+0.161 pp) |
| best check against the step budget | patchtst indirect seed 42 | best step 4900 of a budget of 5000 |
| best check against the step budget | patchtst direct seed 42 | best step 4100 of a budget of 5000 |
| best value on the edge of the search space | bilstm indirect seed 42 | hidden, n_layers, batch_size |
| best value on the edge of the search space | tft96 indirect seed 42 | hidden_size, attention_head_size |
| best value on the edge of the search space | tft96 direct seed 42 | hidden_size, attention_head_size |
| best value on the edge of the search space | tft168 indirect seed 42 | hidden_continuous_size |
| best value on the edge of the search space | tft168 direct seed 42 | attention_head_size |
| study inside TPE's random start-up | tft96 indirect seed 42 | 7 trials (start-up 10) |
| study inside TPE's random start-up | tft96 direct seed 42 | 7 trials (start-up 10) |
| study inside TPE's random start-up | tft168 indirect seed 42 | 7 trials (start-up 10) |
| study inside TPE's random start-up | tft168 direct seed 42 | 7 trials (start-up 10) |
| mean bias on validation | bilstm indirect seed 42 | +1.30 % of capacity (+44.9 kW) |
| mean bias on validation | bilstm direct seed 42 | +0.37 % of capacity (+12.9 kW) |
| mean bias on validation | tft96 indirect seed 42 | +0.70 % of capacity (+24.3 kW) |
| mean bias on validation | tft96 direct seed 42 | +0.46 % of capacity (+15.8 kW) |
| mean bias on validation | tft168 indirect seed 42 | +0.99 % of capacity (+34.1 kW) |
| mean bias on validation | tft168 direct seed 42 | +1.12 % of capacity (+38.6 kW) |
| mean bias on validation | tftbase indirect seed 42 | +0.92 % of capacity (+31.6 kW) |
| mean bias on validation | tftbase direct seed 42 | +0.68 % of capacity (+23.5 kW) |
| mean bias on validation | patchtst indirect seed 42 | +2.73 % of capacity (+94.2 kW) |
| mean bias on validation | patchtst direct seed 42 | +0.72 % of capacity (+25.0 kW) |
| margin against climatology (kW DM) | persistence direct D+1 | 109 % of the minimum detectable effect, HLN p 0.0012 |
| margin against climatology (kW DM) | persistence direct D+2 | 136 % of the minimum detectable effect, HLN p 0.0000 |
| margin against climatology (kW DM) | persistence direct D+3 | 117 % of the minimum detectable effect, HLN p 0.0001 |
| margin against climatology (kW DM) | persistence indirect D+1 | 109 % of the minimum detectable effect, HLN p 0.0014 |
| margin against climatology (kW DM) | persistence indirect D+2 | 135 % of the minimum detectable effect, HLN p 0.0000 |
| margin against climatology (kW DM) | persistence indirect D+3 | 117 % of the minimum detectable effect, HLN p 0.0000 |

## Sources

- `bilstm/best_params.json` sha256 `929123522c5e4a09`
- `bilstm/mbe.csv` sha256 `322e3559fb5dcd02`
- `evaluate/val/benchmark_summary.csv` sha256 `d4fa8a405e639fe6`
- `evaluate/val/dm_tests_kw.csv` sha256 `5975bf8b10858b4b`
- `explain/hparam_importance.csv` sha256 `28457b4a75bb9f67`
- `explain/tft/tft_attention_summary.csv` sha256 `b4ed913a1c1e3392`
- `patchtst/best_params.json` sha256 `a9f41f10356860e0`
- `patchtst/mbe.csv` sha256 `101757ce92e70404`
- `tft168/best_params.json` sha256 `f46775da896098f5`
- `tft168/mbe.csv` sha256 `a2194bcfe7a3d339`
- `tft96/best_params.json` sha256 `2f28aa07c02c99c9`
- `tft96/mbe.csv` sha256 `5f0513aac9d4af89`
- `tftbase/best_params.json` sha256 `2487cd25faab68a2`
- `tftbase/mbe.csv` sha256 `27bd348675aeddc5`
