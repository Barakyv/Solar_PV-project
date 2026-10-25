# Stage 6b plan: GFS forecasts available at issuance

Written on 2026-10-25, before the code. Decision D-12 (user, 2026-10-14): the NWP of the original
(Open-Meteo archive) is not reproduced. The replacement is the NOAA GFS archive on AWS, using
only a run that was published before the issuance time. GFS enters the models only if:
1. it is leak-free (T-17);
2. it improves validation nRMSE;
3. it passes the feature selection (stage 7).

Spec: PROJECT_MASTER.md §6.12, finding F-11. This restores the NWP component of the original
pipeline and carries out the first Future Work item of the report.

## 1. Facts checked at the start (2026-10-25)

Read from the public bucket listing and from the `.idx` files of the 12Z run of 2022-06-15. Only
indexes and listings were read; no GRIB data was downloaded.

| Item | Finding |
|---|---|
| Resolutions in the archive | 0.25° (`pgrb2.0p25`): hourly files to f120, then 3-hourly, about 553 MB per file. 0.5° and 1° (`pgrb2.1p00`): 3-hourly, about 163 and 46 MB per file |
| Size of one global message | 0.25°: about 0.85 MB per variable. 1°: about 0.07 MB |
| Radiation fields | `DSWRF:surface` is an average from the start of each 6-hour window: `6-9 hour ave`, `6-12 hour ave`, then `12-13 hour ave` in the hourly files. Cloud cover (`TCDC`, `LCDC`, `MCDC`, `HCDC`) has the same averaged form, and also an instantaneous form |
| Other fields | `TMP`, `RH` at 2 m; `UGRD`, `VGRD` at 10 m; `PWAT`; `SUNSD`. All instantaneous at the lead |
| Publication (A-13) | The 12Z files appear in the bucket 3.5–4.0 h after init: f000 at +3.5 h and f096 at +4.0 h, on 2021-09-01, 2023-06-15, 2025-05-01 and 2025-10-20. They are uploaded in real time, not backfilled |
| Issuance | T0 = 00:00 local = 21:00Z (summer) or 22:00Z (winter) of the UTC day before the first forecast day. The 12Z run of that UTC day is available at least 5 h before T0. The 18Z run is not: it appears around 22Z |
| Leads needed | Hour ends at leads 10–82 for D+1 to D+3, so files f009–f084 every 3 h: 26 files per run |
| Decoding | ecCodes 2.49 and cfgrib are installed. The DLL is found only when `Library/bin` is on PATH, so the module adds it itself |

## 2. Self-review before the code

### 2a. Mistakes of the original NWP that must not come back

Read in `old files/04_feature_engineering_v4.ipynb` (cell 2.5) and the evidence package:

| ID | Original | Guard |
|---|---|---|
| N1 | The Open-Meteo **archive** API at the target timestamp, labelled in the code as "FORECAST data (not reanalysis)". These values were not known at the midnight issuance (C-06, F-11) | Only a GFS run with init + 6 h ≤ T0, **and** the actual upload time of every file used ≤ T0, checked row by row (T-17) |
| N2 | One NWP series for every horizon: D+3 used the same value as D+1 | Each horizon has its own columns from the run of its own issuance and its own leads. Check: the correlation with the observation must not rise with the horizon |
| N3 | NWP in true UTC merged at the same timestamp onto SCADA labelled UTC but in local time (F-01), so a 2–3 h shift | Everything in true UTC. A label test: the left-labelled row t receives the GFS mean over (t, t+1]. A sanity check: on clear days the cross-correlation of GFS and measured irradiance peaks at lag 0 |
| N4 | Gaps filled with `ffill().bfill()`: backward fill uses the future | A missing run is replaced by the latest earlier run that is allowed (06Z, then 00Z), and flagged. A remaining gap stays NaN |
| N5 | POA forecasts derived from the leaky source entered the 05e sets; they were removed only at training time, through EXCL | Every GFS feature has the `fcst_gfs_` prefix; the leakage rules allow no other `fcst_*` (T-01). The 05e sets are regenerated in stage 7 (C-09) |
| N6 | Monotone +1 constraints by substring on `fcst_ghi`, `fcst_dni` and `fcst_dhi`, for both targets (C-33) | No constraint on GFS features unless decided on validation in stage 9 (D-15) |
| N7 | Radiation from an API that labels the mean of the preceding hour at the hour end, merged onto left-labelled data: a one-hour shift is easy | The same label test as N3, on synthetic data with a known shape |
| N8 | Coordinates of a point inside the kibbutz | Bilinear interpolation of the 1° grid to the park centre (D-05); nearest grid point as a sensitivity check |

### 2b. Review of my earlier work for this stage

| ID | Finding | Action |
|---|---|---|
| B1 | §6.12 gives the D+1 leads as "about 9–34 h". The exact hour ends are 10–33 (summer) and 11–34 (winter) at D+1, up to 82 at D+3 | Use the exact leads. Files f009–f084 (the first block needs f009) |
| B2 | §6.12's fallback (06Z, then 00Z) needs longer leads, up to 94 h | The downloader fetches a fallback run only when the 12Z run is missing, with the shifted leads |
| B3 | A-13 was an assumption | Verified from the upload times (§1). The downloader records the `Last-Modified` time of every file it reads, for T-17 |
| B4 | The leakage rules already allow only `fcst_gfs_` (test exists) | Kept; GFS columns are added to the contract groups as `nwp_gfs` |
| B5 | The 1° grid orientation is assumed: lat 90 → −90, lon 0 → 359 | Read the grid keys from the first decoded message and assert them |

## 3. Decision proposed: resolution (L-05)

| Option | Time step | Nearest grid point | Transfer for the whole period | Stored |
|---|---|---|---|---|
| 0.25° (L-05) | 1 h | about 2 km | about 260 GB for only DSWRF, TCDC and TMP; about 860 GB for the full set | point values only |
| **1° (§6.12)** | 3 h | about 7 km | **about 25–30 GB** for 11 variables | point values only, about 0.1 GB |

**Proposal:** 1°, as in the spec. L-05 is checked and not adopted: the GRIB messages are global,
so the 0.25° files cost about ten times more, and the archive offers no point subsetting.
Hourly values come from the 3-hour means through the clear-sky index, as L-05 itself recommends
for the 3-hourly case.

**Optional check:** a 30-day sample of 0.25° DSWRF on validation days, about 2 GB. It would
measure how much the hourly 0.25° irradiance gains over the reconstructed 1° irradiance, against
the measured in-plane irradiance. The user decides.

**Download needs the user's explicit approval** (filenames, source, size). Before the full run, a
one-run sample (about 2 MB) verifies decoding, the grid, the averaging windows and the label
convention on real data.

## 4. Steps

### S1. Download (`solarpv.nwp.download`)

- For each UTC day from 2021-08-29 (the first D+3 issuance) to 2025-05-30, download the 12Z run.
- For each file f009–f084 every 3 h:
  - read the `.idx`;
  - select the 11 messages:
    - `DSWRF:surface` (average);
    - the cloud averages: `TCDC:entire atmosphere`, `LCDC`, `MCDC`, `HCDC`;
    - instantaneous: `TMP:2 m`, `RH:2 m`, `PWAT`, `UGRD:10 m`, `VGRD:10 m`, `SUNSD`;
  - fetch them with HTTP Range requests. Neighbouring messages are merged into one request when
    the gap is small. S3 serves one range per request.
- Decode in memory with ecCodes and keep only the 3×3 grid points around 30N 35E. Nothing else
  is stored.
- Store per run: `data/external/gfs_1p00/YYYYMMDD_12z.parquet` (lead, variable, lat, lon, value)
  and a JSON record: URL, byte ranges, `Last-Modified`, message descriptions.
- Resume: a run with a complete record is skipped. Retries with back-off. A missing 12Z run
  triggers the 06Z and then the 00Z run of the same day, flagged.
- `RunLog` for every run of the command; counts of runs, files, bytes and fallbacks.

### S2. Hourly reconstruction (`solarpv.nwp.gfs`)

- **De-averaging (L-06):** the 3-hour mean over (n−3, n] = ((n − s) · A(s, n) − (n − 3 − s) ·
  A(s, n − 3)) / 3, where s is the start of the 6-hour window. For the first block of a window
  it is A(s, s + 3) itself. A synthetic unit test recovers known 3-hour means exactly.
- **Irradiance to hours:**
  - The 3-hour mean is divided by the mean clear-sky GHI at the site over the same three hours;
    this gives the clear-sky index of the block.
  - The hourly GHI is that index times the hourly clear-sky GHI. This keeps every 3-hour mean
    exactly (energy-conserving).
  - Where the block's clear-sky mean is below 20 W/m² (twilight), the 3-hour mean is spread
    evenly.
- **Averaged cloud fields:** piecewise constant within each block.
- **Instantaneous fields:** linear interpolation in time to the middle of each hour.
- **Space:** bilinear interpolation to the site from the 2×2 cell that contains it. The nearest
  point and the 3×3 mean and std are kept for the sensitivity check and L-08.
- **Label convention:** row t (hour [t, t+1)) receives the GFS value for (t, t+1], i.e. the block
  that contains lead (t + 1 − init). Tested on synthetic data.

### S3. Features per horizon

For each target hour t and horizon h:
- the issuance T0 comes from stage 6 (`issuance_utc`);
- the run is the 12Z run of the UTC date of T0, or the flagged fallback;
- T-17: init + 6 h ≤ T0 and every file's upload time ≤ T0.

Candidate columns (prefix `fcst_gfs_`):

| Group | Columns |
|---|---|
| Irradiance | `ghi`; `kt` = ghi / GHI_cs (L-03); `poa`, from Erbs separation and Hay–Davies on the panel plane 21°/185° (L-01); `kt_poa` = poa / POA_cs, aligned with the k target of D-11 (L-03) |
| Clouds | `tcc`, `lcc`, `mcc`, `hcc`; `tcc_mean3x3`, `tcc_std3x3` (L-08) |
| Weather | `t2m`, `rh2m`, `pwat`, `wind10`, `sunsd` (L-07) |
| Physical chain (L-01) | `tcell` (Faiman, from poa, t2m and wind); `pv_kw` = poa × 3,449/1,000 × (1 + γ(tcell − 25)), with γ = −0.42 %/K measured in stage 3; `k_phys` = pv_kw / theo_kw |
| Temporal (L-02) | the daily mean of `kt_poa` and `tcc` over the target's local day; the centred 3-hour mean of `kt_poa`. All from the same run, so known at T0 |

Also a candidate, but not from GFS (L-11a):
- `irradiance_clear_index_prevday`: the 90th percentile of the measured in-plane irradiance /
  POA_cs over daytime hours (sun above 20°) of the last full local day before T0.
- It reflects aerosol and dust attenuation, and is known at T0 by construction.

Output: `data/processed/06b_gfs/gfs_features_D{h}.parquet` (UTC index = target hour). Metadata
columns:
- `gfs_init`;
- `gfs_run` (12, 6 or 0);
- `gfs_fallback`;
- `gfs_uploaded_max`;
- `gfs_lead_end_h`;
- `t0_utc`.

Plus a manifest and a feature dictionary.

### S4. Sanity checks (train and validation, never test)

| Check | Pass |
|---|---|
| Coverage | ≥ 99 % of runs from 12Z; every fallback flagged |
| Correlation of `kt_poa` with the measured k, by horizon | does not rise from D+1 to D+3; above 0.95 at D+3 is a red flag |
| Time alignment (N3) | on clear train days, the cross-correlation of GFS GHI and measured irradiance peaks at lag 0 h |
| Bias | mean GFS GHI vs clear sky and vs the measurement, reported by season (GFS is known to be biased; this is reported, not corrected here) |

### S5. Value check (D-12, criterion 2)

- Protocol: the fixed XGBoost of D-11 (400 trees, depth 6, learning rate 0.05; seeds
  42/123/777), trained on train and scored on validation, nRMSE in kW, for every horizon.
- Compared: the contract alone vs the contract plus the GFS candidates.
- Rule, fixed now (as for L-04): GFS is admitted if the mean gain exceeds max(0.05 pp, 2 × the
  pooled seed std) at D+1, and it is not worse at D+2 or D+3.
- The decision is written to D-12. Stage 7 then selects among the admitted candidates.

### S6. Ideas of this stage

| ID | Plan |
|---|---|
| L-05 | Checked; proposal §3 |
| L-06 | Implemented and tested |
| L-07 | The variable set of S1 |
| L-08 | 3×3 mean and std stored; two cloud candidates |
| L-09 | MOS bias correction is model-level; stage 9 |
| L-10 | GEFS: about 31 times the 1° volume; not in scope |
| L-11 | (a) implemented as a candidate. (b) CAMS needs an account the user must create (Q-14); not started. (c) GEFS-Aerosols: not in scope |

## 5. Tests (IDs continue PROJECT_MASTER.md §8.2)

| ID | Level | Test |
|---|---|---|
| T-17 | unit + contract | For every row and horizon: init + 6 h ≤ T0, every file's upload time ≤ T0, the run is the latest allowed |
| T-42 | unit | De-averaging recovers synthetic 3-hour means; the hourly reconstruction preserves them; the label convention (row t gets (t, t+1]) |
| T-43 | unit | Run choice: a run published after T0 is never used, including the 18Z run and DST days; a missing run falls back to earlier runs only; perturbing a later run changes nothing |
| T-44 | contract | The real tables: coverage, horizon columns differ, the correlation does not rise with the horizon, the clear-day lag is 0, no test row was used in any decision |

## 6. Acceptance

- Download complete (≥ 99 % of 12Z runs), with records and run logs.
- T-17 and T-42–T-44 pass, together with the full suite.
- The sanity checks pass, and the value check is recorded in D-12.
- OP entries, numbers N13, the run journal and PROJECT_MASTER are updated.
