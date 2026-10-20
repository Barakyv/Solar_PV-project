# Stage 6 plan: target and features

Written on 2026-10-20, before the code. Builds the modelling table from the cleaned hourly data
(stage 5): the two targets (§6.5) and the feature contract (§6.6), for each horizon D+1…D+3.
It also takes the open decisions of this stage:
- D-11, the denominator of k;
- A-04, the orientation of the panels;
- Q-05, `sun_azimuth` against `solar_azimuth`;
- L-04, the handling of low sun.

GFS features belong to stage 6b.

## Rules

- **True UTC everywhere** (F-01). A row is the hour [t, t+1h), labelled t.
- **Issuance (D-16).** The forecast for local day D+h is issued at T0 = 00:00 local time of day
  D+1. A feature value may use only data from hours that end at or before T0.
- **Decisions on train only**, except D-11's third criterion, which uses validation as
  specified in PROJECT_MASTER.md. The test split is used for nothing.
- **D-22 masks.** An hour flagged op-limited is out of the training loss. It is NaN in the
  past-day features of the power columns before imputation. Weather columns are not masked:
  a plant limitation does not change the weather.
- **No clipping of k** (F-06). k above 1.5 is flagged as suspect, never cut.
- **Documentation of every run** (user request, 2026-10-20; see "Run documentation" below).

## Steps

### S1. Panel orientation from the corrected data (A-04)

F-13 measured tilt ~20° and azimuth ~180°, but on clocks not yet corrected. After the clock
correction, the DC production centre lies 1–2.7 min after solar noon, so the array may face
2–3° west.

Method, as in F-13 but on the stage-5 minutes:
- clear days chosen from AC/DC smoothness only;
- temperature-corrected DC, using the measured coefficient of −0.42 %/K and the module
  temperature;
- grid fit of tilt and azimuth against clear-sky POA (Ineichen, Hay-Davies), on train months.

Decision rule: if the methods agree on a panel plane that differs from 20°/180° by ≥ 2° in
azimuth or ≥ 2° in tilt, the panel plane becomes the orientation of the theoretical power and of
the POA features. The clock frame of stage 4 keeps the sensor plane 20°/180°, on which it was
estimated.

Evidence goes to `00_data_audit/orientation_2026-10-20/`.

### S2. Hourly sun geometry and clear sky

Minute-level pvlib values, averaged over each hour:
- Solar position.
- Ineichen clear sky, with the default Linke turbidity.
- POA components: Hay-Davies, albedo 0.2.
  - Hay-Davies is the model used for clear-sky POA since stage 2; §6.3 still names isotropic,
    the v4 default.
- AOI and air mass.

**Deviation from §6.4:** the mean is taken over all 60 minutes, not only over the minutes where
AC is valid. The theoretical power of a future hour must be computable at T0, and the same
`theo_kw` must invert k back to kW exactly. Hours with partial AC coverage are 13 of 9,698 train
daytime hours (stage 3b, Q17). This deviation goes into `docs/deviations.md`.

### S3. Targets (§6.5, D-11)

- `site_total_AC_kW`: the hourly mean from stage 5 (the direct target).
- For each candidate denominator X ∈ {GHI_cs, POA_cs}:
  - `theo_kw` = mean clear-sky X over the hour × 3,449 / 1,000;
  - `k` = AC / theo_kw if theo_kw > 50 kW, otherwise `k_valid = False` and k = 0 (§6.5).
- No clipping. `k_suspect` = k > 1.5.

**D-11, three criteria (PROJECT_MASTER.md):**
1. Share of valid daytime train hours with k > 1.
2. Flatness of clear-day k across the months of train: the spread of the monthly median of k on
   clear days.
3. nRMSE in kW on validation. One fixed, untuned XGBoost per denominator (same features, same
   seed, D+1); its k forecast is converted to kW with `theo_kw`. This is the only use of
   validation in this stage.

`MEAN_DAYTIME_K` = mean of k over the valid daytime train hours (evaluation mask F-02), written
to `configs/derived_constants.yaml`.

### S4. Low sun (L-04)

Distribution of k by sun-elevation band on train, and the share of k > 1.2 and k > 1.5 by band.
If the low-sun hours are unstable (stage 3b, Q22), test on validation a higher validity
threshold for the k loss (theo > 100 or 150 kW) or a weight proportional to `theo_kw`. The
evaluation mask itself does not change (F-02).

### S5. Features (§6.6, D-04)

- **The contract becomes 74 features:** `sun_azimuth` equals `solar_azimuth` by construction
  (both pvlib azimuth), so Q-05 closes and `sun_azimuth` leaves the contract. In the original it
  came from the lost preprocessing step, computed on the wrong time base (F-01).
- **Geometry and clear sky** (9 + 6): the hourly means of S2. `time_to_solar_noon` uses the pvlib
  transit (not local hour − 12), `day_length` comes from sunrise and sunset, and
  `daylight_fraction` is the day length divided by the longest day of the year (computed, not the
  v4 approximation).
- **Calendar** (8 + 3):
  - `hour_sin` and `hour_cos` use the UTC hour.
  - `hour_local`, `month`, `season`, `doy_*`, `day_of_week` and `is_weekend` use the local
    calendar day of the target hour.
  - **`is_weekend` = Friday or Saturday,** the Israeli weekend. The original used Saturday and
    Sunday. Stage 3b found no weekday effect (Q28), but the definition should be correct; this
    is recorded as a deviation.
- **Lags** (7 sources × 5 lags): source value at t − 24·d hours, for d ∉ LAG_DROP(h).
  **DST guard:** a value is valid only if its hour ends at or before T0(t). On the day the clock
  falls back (a 25-hour day), the last hour of D+h would otherwise see a value from inside the
  forecast day. Such values become NaN.
- **Rolling, per horizon (D-04):** the mean and standard deviation of the same hour over
  shifts h…h+N−1 days (N = 3 or 7), with the same DST guard. At least 2 valid values are needed
  for the standard deviation, 1 for the mean.
- **D-22 masks before lags and rolling.** Each inverter's AC is NaN in hours where its own flags
  (F1, F3) or curtailment (F2) cover at least 5 % of the minutes. The site total is NaN in
  op-limited hours.
- **Imputation (fix of the original).** v4 filled with the expanding median of the whole column,
  nights included, which pulls daytime gaps towards zero. Here: the expanding median **of the
  same UTC hour over earlier days** (shifted by one day, so only older values), then forward fill
  within the same hour. Never backward fill. The share of imputed values per feature is reported.
- **Warm-up:** the first 14 days carry `warmup = True` and are excluded from training.
- **Per-row metadata:** `t0_utc`, `split`, `daytime` (F-02), `op_limited_flag`, `k_valid`,
  `k_suspect`, `hour_source`.

### S6. Outputs

- `data/processed/06_features/features_1H_D{1,2,3}.parquet`: one table per horizon, with
  targets, metadata and the features of that horizon. Lags in LAG_DROP are absent, and rolling
  columns are the horizon's own version under the common name.
- `features_manifest.json`: input hashes, git commit, counts, imputation shares and the D-11
  numbers.
- `outputs/06_features/`:
  - `feature_dictionary.csv`: name, group, formula, source, unit, available at T0 (yes / no and
    why), imputed share;
  - D-11 and L-04 tables and figures;
  - run logs.
- `configs/derived_constants.yaml` (`mean_daytime_k`, the denominator, the orientation).
- `docs/target_definition.md`: the D-11 decision and its numbers.

### S7. Notebook

`notebooks/04b_features_target.ipynb`, built by `scripts/make_features_notebook.py`. A markdown
note follows every code cell; the numbers come from the manifest.

## Run documentation (user request, 2026-10-20)

- **Code.** Every module, function and non-obvious line explains *why*, not only what.
  Docstrings name the spec section, decision or test.
- **Automatic run log.** Every run of a stage command writes
  `outputs/<stage>/runs/<UTC timestamp>.log`:
  - command, git commit and versions;
  - input hashes;
  - one timestamped line per step with its intermediate result (counts, checks, decisions
    taken), and warnings;
  - outputs with hashes;
  - duration and final status.
  The same lines go to the console. The module is `solarpv.config.runlog`, used by stage 6 and
  every later stage.
- **Run journal.** `docs/run_journal.md` is a human-written entry for each run of a stage:
  purpose, command, commit, log file, results, observations and intermediate notes, problems
  found and what was done about them. Earlier stages are summarised from their manifests; full
  entries start with stage 6.

## Tests (IDs continue PROJECT_MASTER.md §8.2)

| ID | Level | Test | Pass criterion |
|---|---|---|---|
| T-09 | unit + contract | rolling and lags horizon-safe | change every source value at or after T0 of a chosen day: no feature of D+h on that day changes (h = 1, 2, 3) |
| T-18 | unit + contract | issuance boundary on DST days | same perturbation around the spring-forward and fall-back days: no feature changes; the 25th hour gets NaN where its lag would cross T0 |
| T-10 | unit | contract | 74 unique features; group sizes; equals the original 76 minus `theoretical_power` and `sun_azimuth`; horizon drops 7 / 14 |
| T-11 | contract | k ↔ kW | `max |k·theo_kw − AC| < 1e-6` on valid hours; conversion of a forecast k is exact |
| T-13 | contract | no clipping | no mass at 1.2; k > 1.2 exists and is kept; k ≥ 0 |
| T-33 | contract | stage-6 output | 32,856 rows per horizon; the exact feature set per horizon; no leaky column (T-01 at matrix level); metadata consistent with stage 5 (`op_limited_flag`, `hour_source`, split counts) |
| T-34 | unit | imputation | fills only from the same UTC hour on earlier days; never backward fill; warm-up flagged; imputed shares reported |
| T-35 | unit + contract | D-22 masks | per-inverter masks applied before lags; flagged values never appear in any power lag or rolling feature |
| T-36 | unit | geometry and calendar | `time_to_solar_noon` = 0 at the pvlib transit; `is_weekend` on Friday and Saturday local; hour means of minute values |

Mutation check: plant (a) a rolling window that starts at shift 24 for D+2, (b) an imputation
with backward fill, (c) a lag that ignores the DST guard. Each must be caught.

## Acceptance (PROJECT_MASTER.md §7, stage 6)

- 32,856 rows per horizon and 74 features by contract.
- k ↔ AC inversion exact (T-11); the share of k > 1.2 reported (expected far below 16.8 %).
- Perturbation tests pass for every horizon (T-09, T-18).
- The new MEAN_DAYTIME_K documented, D-11 decided, A-04 updated.
- OP entries in the data-processing record, the numbers file regenerated.
- The notebook built, the run journal written.

## Changes during the stage (2026-10-20)

The plan above is kept as written. What changed while it was carried out, and why (details in
`docs/run_journal.md`, `docs/deviations.md` and the data-processing record, OP-32 to OP-39):

- **S1:** the rule applied. The panel plane is 21°/185° (F-23); the sensor plane 20°/180° stays
  the clock frame.
- **S2:** the partial-coverage count of the plan (13 of 9,698 train hours, from stage 3b) is out of
  date after the stage-3b changes to cleaning. The build now counts the daytime hours whose AC
  mean rests on fewer than 60 minutes: 26 (7 train, 19 test, N11).
- **S3, criterion 3:** three seeds instead of one, reported as mean ± std. One fit was within the
  seed noise for L-04, so the same care was applied to D-11. The decision did not change.
- **S4, L-04:** the plan did not fix a rule for adopting a higher threshold. The first rule (gain
  ≥ 0.05 pp on one fit) chose 150 kW from noise. It was replaced before any use: mean of 3 seeds,
  gain > max(0.05 pp, 2 × pooled seed std). Result: 50 kW stays.
- **S5, imputation:** the median of the same UTC hour over the previous **30 days** instead of an
  expanding median with forward fill. An expanding median mixes seasons after the first year, and
  a forward fill is not needed: rows without an earlier value at that hour are in the warm-up.
- **S6:** `feature_dictionary.csv` first gave one definition and unit per group. Every feature now
  has its own (R6-08). The manifest also carries `target_checks`, the facts the record quotes.
  `docs/target_definition.md` was written at the end of the stage.
- **Tests:** the mutation check found two weak tests, and both were fixed:
  - the synthetic data had no gaps;
  - the perturbation window skipped the DST day.

  It now plants five leaks instead of three.

**Findings of the stage:**
- F-23: the panel orientation.
- F-24: the original imputation.
- F-25: the original calendar features.
- Not a bug, an observation for stage 7: the k-lag proposal. Raw power lags correlate −0.28
  with k, because power carries the cycle that k removes.
