# Target definition (stage 6)

The two targets of the project, as built by `python -m solarpv.features.build`. The decisions
are in `configs/target.yaml` (D-11, L-04) and `configs/derived_constants.yaml`
(`MEAN_DAYTIME_K`). The evidence is in `outputs/06_features/` and
`notebooks/04b_features_target.ipynb`; the operations are OP-33 to OP-35 in
`docs/data_processing_record.md`, and all numbers are in N11 of `docs/data_record_numbers.md`.
`tests/contracts/test_target_definition.py` checks that the values quoted here match the
configuration files.

## Direct target

`site_total_AC_kW`: the hourly mean of the cleaned minute AC power of the two inverters (stage 5),
in kW. The row labelled t covers [t, t+1h) in true UTC.

## Indirect target: the clear-sky index k

```
CS_t   = mean over the 60 minutes of hour t of the clear-sky irradiance on the panel plane
         (POA_cs: pvlib Ineichen, Hay-Davies transposition, tilt 21°, azimuth 185°, albedo 0.2)
theo_t = CS_t × 3,449 / 1,000                                           [kW]
k_t    = AC_t / theo_t        if theo_t > 50 kW and AC_t is known       (k_valid = True)
k_t    = 0                    if theo_t ≤ 50 kW (night, deep twilight)  (k_valid = False)
k_t    = NaN                  if AC_t is missing
```

- **k is not clipped.** `k_suspect` = k > 1.5 is a flag only. The 23 such hours all have the sun
  below 10° (N11).
- **Exact inversion:** `AC_t = k_t × theo_t` on every valid hour (T-11). A forecast of k is
  converted to kW as `clip(k̂ × theo_t, 0, 3,449)` (§6.5).
- Every daytime hour of the evaluation mask (sun above 5° at mid-hour, F-02) has theo above
  50 kW, so every daytime hour with AC is valid.
- The model loss for k uses the valid hours outside the warm-up and outside op-limited hours
  (D-22).

## Decisions

| Item | Value | Basis |
|---|---|---|
| Denominator (D-11) | **POA_cs** on the panel plane | Wins 3 of 3 criteria: share of k > 1 on train 0.054 vs 0.324 (GHI_cs); spread of the monthly clear-day median k 0.108 vs 0.413; validation nRMSE in kW 7.94 ± 0.09 % vs 8.09 ± 0.06 % (fixed XGBoost, 3 seeds) |
| Panel plane (A-04) | tilt 21°, azimuth 185° | Measured on the clock-corrected data (F-23, OP-32) |
| Training threshold on theo (L-04) | 50 kW | 100 and 150 kW gave no gain beyond the seed noise (best 0.053 pp against 0.119 pp needed) |
| Evaluation threshold on theo | 50 kW | §6.5; no daytime hour is below it |
| `MEAN_DAYTIME_K` | 0.788157 | Mean k over valid daytime train hours; the denominator of the indirect nRMSE (§6.9). The original 0.6279 is used only for the fixture (T-15) |

## Why not the original definition

The original clipped k at 1.2, averaged 15-minute ratios (so k could not be inverted to kW), used
GHI_cs, and ran on local time labelled UTC (F-01, F-06). With GHI_cs, 32 % of daytime train
hours have k above 1: in winter the tilted panels receive more than the horizontal reference.
With POA_cs, 5 % do.
