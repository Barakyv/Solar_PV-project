# Run journal

A human-written record of every run of a stage command: why it was run, what came out, what was
noticed along the way, and what was done about it. It started on 2026-10-20 at the user's
request. Every run since stage 6 also writes an automatic log (`solarpv.config.runlog`, file
`outputs/<stage>/runs/<UTC timestamp>.log`). The log holds the command, the commit, versions,
input and output hashes, steps with durations, intermediate notes and values. This journal
points to the log and adds the judgement. Operations on the data are documented in
`docs/data_processing_record.md`; the plan of each stage in `docs/stage*_plan.md`.

**Entry template.** `### <date> · R<stage>-<n> · <title>`, then:
- **Purpose:** why the run was made.
- **Command and commit:** what was run, at which commit.
- **Log:** the path of the automatic log.
- **Results:** the main numbers.
- **Intermediate notes:** observations during the run.
- **Problems and fixes:** what went wrong and what was done.
- **Decision / next:** what follows from the run.

## Runs before stage 6 (summary from the stage manifests)

Earlier runs did not have the automatic log. Their manifests (git commit, input and output
hashes, numbers) and the commit messages are the record:

| Stage | Command | Code commit of the current outputs | Date | Main result | Record |
|---|---|---|---|---|---|
| 2 ingest | `python -m solarpv.data.ingest` | 8bbd719 (code entered git at 0b856c9, see the record) | 2026-10-14 | 1,971,360 true-UTC minutes, 45 files verified | `data/processed/01_ingest/ingest_manifest.json`, OP-01–OP-05 |
| clock audit | `python scripts/estimate_clock_drift.py` | 0b856c9 | 2026-10-14 | offsets per source and clock group, residual ≤ 1.74 min | `00_data_audit/clock_drift_2026-10-14/`, OP-08–OP-13 |
| 4 preprocess | `python -m solarpv.data.preprocess` | ee4799a | 2026-10-15 | 32,856 hours, no hour below 45/60 | `preprocess_manifest.json`, OP-14–OP-17 |
| 5 clean | `python -m solarpv.data.clean` | 19dbcfd | 2026-10-18 | R1–R7, F1–F3; op-limited daytime hours 379/146/186 | `outputs/05_cleaning/cleaning_manifest.json`, OP-18–OP-28 |
| 3 EDA | `python -m solarpv.eda`, `scripts/make_eda_notebook.py` | 19dbcfd | 2026-10-18 | 10 figures, 5 tables | `outputs/03_eda/eda_manifest.json`, OP-29 |
| 3b quality | `python -m solarpv.eda.quality`, `scripts/make_quality_notebook.py` | 19dbcfd | 2026-10-18 | 31 analyses; F-20, F-21, F-22 fed back | `outputs/03b_quality/quality_manifest.json`, OP-31 |

## Stage 6: target and features

### 2026-10-20 · R6-01 · A-04: panel orientation on the corrected data

- **Purpose:** F-13 measured 20°/180° on clocks not yet corrected. After the clock correction,
  the DC production centre was 1–2.7 min late. Stage 6 needs the right panel plane for the
  theoretical power (k) and the POA features (plan S1).
- **Command and commit:** `python 00_data_audit/orientation_2026-10-20/orientation_check.py`
  at `1da1292`.
- **Log:** `outputs/06_features/runs/20261020T094900Z.log`. A first run at 8679086+dirty, with
  the same numbers, was discarded so that the kept log names a clean commit.
- **Results:**
  - **Data:** 174 clear train days (6 per month, 29 months); 9,852 ten-minute samples with the
    sun above 15°; DC temperature-corrected with −0.421 %/K.

  | Method | Tilt | Azimuth |
  |---|---|---|
  | M1, DC one scale | 21.0° | 185.0° (within 1 %: 183.5–186.5) |
  | M2, DC scale per day | 15.5° (weak) | 186.5° |
  | M3, production centre | — | 184.7° (+3.0 min late at 180°) |
  | M0, irradiance sensor | 16.5° | 180.5° |

- **Intermediate notes:**
  - The three DC methods agree on about 185°, while the irradiance sensor sits at 180°. The
    panels, not the clocks, are what face west of south. The clock correction is framed on the
    sensor plane, so it is unaffected.
  - Morning dew or soiling on the modules could also delay the DC centre. The sensor is at the
    same site and is not delayed, and M1 and M2 fit the whole daily shape, not only the centre.
  - The tilt from the shape alone (M2) is weakly identified; M1 uses the seasonal level and
    gives 21° (20–21.5 within 1 %).
  - A RuntimeWarning in the clear-day selection (a day with zero DC) was removed: such days are
    now skipped explicitly.
- **Decision / next:**
  - The rule of plan S1 applies: panel plane **21°/185°** goes to `configs/panel_orientation.yaml`
    (finding F-23, A-04 updated).
  - The sensor plane 20°/180° stays the frame of the clock correction.
  - External confirmation of the row orientation is still open (Q-11).
  - Next: the targets and D-11 with the panel plane.

### 2026-10-20 · R6-02 · First build of the tables (D-11 still open)

- **Purpose:** first end-to-end run of plan steps S2–S6 with the panel plane of R6-01. Both
  candidate k columns are written, so that D-11 can be decided from the table itself.
- **Command and commit:** `python -m solarpv.features.build` at `1da1292+dirty` (the stage-6
  code was not yet committed).
- **Log:** `outputs/06_features/runs/20261020T095453Z.log`.
- **Results:**
  - 32,856 rows per horizon; 74 / 67 / 60 features at D+1 / D+2 / D+3.
  - Valid daytime train hours: 9,656 for both denominators.

    | Denominator | k > 1 | k > 1.2 | k > 1.5 |
    |---|---|---|---|
    | GHI_cs | 32.4 % | 12.5 % | 0.99 % |
    | POA_cs | 5.4 % | 0.61 % | 0.04 % |

  - D-22 masks before the past-day features: 474 hours (inverter 1), 548 (inverter 2), 733
    (site total).
  - DST guard: 28 / 56 / 84 values removed at D+1 / D+2 / D+3.
  - The largest imputed share of a past-day feature is 3.56 %.
- **Intermediate notes:**
  - With GHI_cs, a third of daytime hours have k above 1. That is the reference, not the
    weather: the tilted panels receive more than the horizontal in winter.
  - k is never clipped. The original clipped 16.8 % of daytime test hours at exactly 1.2.
- **Decision / next:** run the D-11 criteria (`decide_k`).

### 2026-10-20 · R6-03 · D-11 and L-04, first run (one seed)

- **Purpose:** compute the three D-11 criteria, all fixed in the plan before the run, and test
  L-04 (training threshold on theo).
- **Command and commit:** `python -m solarpv.features.decide_k` at `1da1292+dirty`.
- **Log:** `outputs/06_features/runs/20261020T095633Z.log`.
- **Results:**
  - **D-11:** POA wins all three criteria: share of k > 1 0.054 vs 0.324; clear-day spread
    0.108 vs 0.413; validation nRMSE 7.885 % vs 8.152 % (one seed).
  - **L-04 (one seed):** 50 kW 7.885 %, 100 kW 8.066 %, 150 kW 7.831 %. The rule as first
    written (gain ≥ 0.05 pp) selected **150 kW**, and `configs/target.yaml` was written with 150.
- **Intermediate notes:**
  - 100 kW is 0.18 pp *worse* than 50 kW, and 150 kW is 0.054 pp better. Threshold effects do
    not jump back and forth like that. The differences are most likely the noise of one fit,
    and the 0.05 pp margin is below that noise.
- **Problems and fixes:** the L-04 rule was too weak. It was revised **before** any threshold
  was used by another stage: three seeds for every threshold, and the gain must exceed
  max(0.05 pp, 2 × the pooled seed std). D-11's third criterion also moved to three seeds. The
  rule is in the docstring of `decide_k.py` and in `target.yaml` (`l04_rule`).
- **Decision / next:** rerun with the seed-robust rule.

### 2026-10-20 · R6-03b · Repeated run (the edit had not been applied)

- **Command and commit:** the same as R6-03, `1da1292+dirty`.
- **Log:** `outputs/06_features/runs/20261020T095739Z.log`.
- **What happened:** the patch that introduced the seeds did not apply: a Python patch inside a
  shell heredoc whose anchor did not match. The run therefore repeated R6-03 exactly (same
  numbers, same output hashes), and again wrote 150 kW. The edit was redone with the editor tool,
  and its effect checked in the next run (the log shows "3 seeds"). The log is kept because the
  run happened. It changes nothing.

### 2026-10-20 · R6-04 · D-11 and L-04 with three seeds

- **Command and commit:** `python -m solarpv.features.decide_k` at `1da1292+dirty`.
- **Log:** `outputs/06_features/runs/20261020T095843Z.log`.
- **Results:**

  | Criterion | GHI_cs | POA_cs | Winner |
  |---|---|---|---|
  | Share k > 1 (train) | 0.324 | 0.054 | POA |
  | Spread of monthly clear-day median k (train) | 0.413 | 0.108 | POA |
  | Validation nRMSE in kW, seeds 42/123/777 | 8.093 ± 0.056 % | 7.941 ± 0.088 % | POA |

  | L-04 threshold | Validation nRMSE (3 seeds) |
  |---|---|
  | 50 kW | 7.941 ± 0.088 % |
  | 100 kW | 8.057 ± 0.021 % |
  | 150 kW | 7.887 ± 0.049 % |

  - The best gain is 0.053 pp; the rule needs more than 0.119 pp. **L-04: the threshold stays
    50 kW**, and `target.yaml` was overwritten with 50.
- **Intermediate notes:**
  - D-11 holds with three seeds: POA's margin in criterion 3 (0.15 pp) is about twice the
    pooled seed std, and criteria 1 and 2 do not involve a fit.
  - L-04 by elevation: k at 5–10° has p99 1.42 and 8 % of hours above 1.2, against 0 % above
    15°. The instability at low sun is real; excluding those hours from training does not
    reliably help.
- **Decision / next:** D-11 = POA_cs (3 of 3); training and evaluation threshold 50 kW. Rebuild
  the tables with the decision.

### 2026-10-20 · R6-05 · Build with the D-11 decision

- **Command and commit:** `python -m solarpv.features.build` at `1da1292+dirty`.
- **Log:** `outputs/06_features/runs/20261020T100150Z.log`.
- **Results:**
  - The tables now carry `theo_kw`, `target_k`, `k_valid` and `k_suspect` for POA.
  - `MEAN_DAYTIME_K` = 0.788157 is written to `configs/derived_constants.yaml` (the original:
    0.6279).
  - Everything else is as in R6-02.
- **Intermediate notes:** k × theo reproduces AC to 2.3e-13 kW on 15,809 valid hours (T-11).
  k > 1.5 (k_suspect) marks 0.04 % of valid train hours; they stay in the data, flagged.

### 2026-10-20 · R6-06 · Tests and the mutation check

- **Commands:** `pytest tests/unit/test_features_build.py tests/unit/test_target.py
  tests/unit/test_feature_contract.py tests/contracts/test_features_output.py`, then
  `python 00_data_audit/features_2026-10-20/mutation_check.py`. These are not stage commands,
  so they have no run log.
- **Output:** `00_data_audit/features_2026-10-20/mutation_check_output.txt`.
- **What the check does:** it plants five known leaks in the feature code, one at a time:
  - (a) the original's rolling window (F-07);
  - (b) backward fill;
  - (b2) a median from later days;
  - (c) lags without the DST guard, two forms.

  It then runs the matching tests.
- **Intermediate notes and fixes:** the first round of the check found that two of the tests
  were weak.
  - The synthetic sources had no gaps, so the imputation never ran and backward fill passed.
    Fix: 5 % holes in the synthetic data. `test_rolling` then needed `holes=0.0`, because it
    compares against an exact mean.
  - The perturbation window started too late and skipped the fall-back day, so the missing DST
    guard passed. Fix: wider windows, plus an assertion that the DST day is among the tested
    days.

  After the fixes, all five mutations are caught and the unmodified code passes.
- **Decision / next:** commit the code (`034f72d`) and rerun everything from the clean commit.

### 2026-10-20 · R6-07 · Final runs from a clean commit

- **Commands and commit:** `build` → `decide_k` → `build` at `034f72d`, no local changes.
- **Logs:** `20261020T100835Z.log`, `20261020T100904Z.log`, `20261020T100959Z.log`.
- **Results:** identical to R6-04 and R6-05: D-11 POA, L-04 50 kW, `MEAN_DAYTIME_K` 0.788157.
  The parquet hashes are the same as in the dirty runs (D1 `b740b2c4…`).
- **Intermediate notes:** the runs R6-02 to R6-05 were made with code that was not yet
  committed (`+dirty`). They are kept as the history of the decisions. The outputs in use come
  from R6-07 and R6-08.

### 2026-10-20 · R6-08 · Per-feature dictionary, and the notebook

- **Purpose:** while the notebook was being checked, the feature dictionary turned out to give
  one definition and a mixed unit ("W/m2 or deg") per *group*. That is not full documentation.
  `describe_feature` now gives every feature its own definition and unit (commit `bc4f8b1`).
  The tests check that the definitions are unique and the units valid.
- **Commands and commits:**
  - `python -m solarpv.features.build` at `ef8a4a0`, log `20261020T101937Z.log`.
  - `python scripts/make_features_notebook.py`, which writes
    `notebooks/04b_features_target.ipynb`.
- **Results:**
  - The parquet files and `derived_constants.yaml` are byte-identical to R6-07; only
    `feature_dictionary.csv` changed.
  - The notebook has 12 code cells, each followed by a note, with numbers taken from the
    outputs.
- **Problems and fixes:**
  - A first rebuild ran at `bc4f8b1+dirty`: `pyproject.toml`, with the lint exception for the
    notebook builder, was not yet committed. Its outputs were identical. Its log
    (`20261020T101854Z`) was deleted before this entry was written, and the run was repeated
    after the commit. It is recorded here instead. From now on, superseded logs are kept.
  - In the notebook's first version, the DST cell showed only night-time zeros, so the guard
    could not be seen. It now shows the 25th hour, the source hour (= T0) and the NaN from the
    guard before imputation.

### 2026-10-20 · R6-09 · Target checks in the manifest

- **Purpose:** while writing the data-processing record (OP-33, OP-34), some facts about the
  target were typed into the record:
  - the smallest theo of a daytime hour;
  - the k_suspect hours and where they lie;
  - the daytime hours with partial AC coverage.

  The record's rule is that its numbers come from manifests. The build now computes these facts
  (`target_checks`, commit `4c5cb3c`), logs them, writes them to the manifest, and the contract
  test asserts them.
- **Commands and commits:** `python -m solarpv.features.build`, twice:
  - log `20261020T102954Z.log` at `4c5cb3c+dirty`;
  - log `20261020T103033Z.log` at `4c5cb3c`.

  Then `python scripts/build_data_record.py` and `python scripts/make_features_notebook.py`.
- **Results:**
  - The smallest theo of a daytime hour is 62.2 kW, above the 50 kW threshold, as §6.5 assumes.
  - 23 k_suspect hours: 9 train, 7 val, 7 test; 8 of them inside the daytime mask. None is
    op-limited; all have the sun at 1.4–9.7° and theo at most 198.5 kW.
  - 26 daytime hours have an AC mean from 46–59 minutes (7 train, 19 test).
  - Parquet files identical to R6-07.
- **Intermediate notes:**
  - A draft of the record said "74 daytime hours with partial coverage". That count included 48
    daytime hours whose AC is missing altogether. The right number is 26, and it is now computed
    by the build.
  - The plan's figure (13 of 9,698 train hours, from stage 3b) predates the stage-3b cleaning
    changes.
- **Problems and fixes:**
  - The first run was `+dirty` because `scripts/build_data_record.py`, which the build does not
    use, was modified. The script was stashed for the second run and restored afterwards.
  - Both logs are kept.
- **Current outputs of stage 6:**
  - Data: identical in R6-07, R6-08 and R6-09.
  - Decisions: R6-07 (`decide_k`, `target.yaml` `git_commit: 034f72d`).
  - Dictionary, manifest and target checks: R6-09 (`4c5cb3c`).

## Stage 8B: evaluation on the corrected tables

### 2026-10-25 · R8-01 · First validation run (references only)

- **Purpose:** plan step §6. The run scores the three references on validation. This checks
  every table of the chain without a trained model and without opening the test split.
- **Command and commit:** `python -m solarpv.evaluation.evaluate` at `a432689`.
- **Log:** `outputs/07_evaluation/runs/20261025T071437Z.log`.
- **Results:**
  - 2,901 validation daytime hours with truth for every reference, target and horizon.
  - Direct, D+1: persistence 9.68 %, climatology 8.04 %, clear sky × MEAN_K 8.96 % of
    3,449 kW.
  - The clear-sky route advantage is exactly 0, the built-in check.
- **Intermediate notes:**
  - Persistence got a positive skill against the CLIPER reference: +0.15 at direct D+1. This
    should not happen: the reference is the optimal blend of climatology and persistence.
  - The cause was the source of its σ and γ. They came from train, a harder sample than
    validation (Feb–Sep, with the summer).
- **Decision / next:** take σ and γ from the evaluated sample (plan §7, deviations), with a test
  that persistence does not beat the reference.

### 2026-10-25 · R8-02 · Validation run with the corrected CLIPER

- **Command and commit:** `python -m solarpv.evaluation.evaluate` at `3df79b9`.
- **Log:** `outputs/07_evaluation/runs/20261025T071608Z.log`.
- **Results:**
  - The persistence skill against CLIPER is now between −0.03 and −0.24, never positive, as
    the theory requires.
  - The other numbers are as in R8-01 (N12).
- **Intermediate notes:**
  - Climatology beats persistence on validation (8.04 % vs 9.68 % direct D+1). The validation
    period includes the clear summer, where the mean diurnal curve is a strong forecast.
  - Persistence degrades from D+1 to D+2 (9.68 → 10.82 %), which is the horizon effect the
    stage-6 fix (F-07) should also produce in the models.

### 2026-10-25 · R8-03 · Fixture rerun (T-15)

- **Command and commit:** `python -m solarpv.evaluation.evaluate --fixture` at `3df79b9`. The
  fixture mode has no run log, so its manifest is the record.
- **Why:** persistence changed (DST guard, gap filling, A1–A2), and new columns were added.
- **Results:**
  - The BiLSTM rows of the old predictions are identical to the stage-8A output: nRMSE, n and
    skill differ by 0.
  - The files differ only in the new columns, in the common-mask rows (persistence is now
    included) and in the DM CI.
  - T-15 passes.
- **Self-review of the stage:**
  - Reading the original scripts against the report found that the route comparison was not
    paired (C-34), and that the common mask chose daytime by the truth (C-35). Both are in the
    errata.
  - In my own stage-8A code: the persistence reference lacked the first test days and the DST
    guard (A1, A2). Fixed.

## Stage 6b: GFS forecasts

### 2026-10-25 · R6b-01 · First real run (2021-10-12, 12Z)

- **Purpose:** verify decoding, the grid and the averaging windows on real data before any
  larger download. The user approved a sample; this was its first run.
- **Command and commit:** `python -m solarpv.nwp.download --start 2021-10-12 --end 2021-10-12`.
  - First attempt at `3ae09b2`, log `outputs/06b_gfs/runs/20261025T080102Z.log`. The process
    was killed by ecCodes ("fatal flex scanner internal error"), so the log has no status line.
  - Second attempt at `3ae09b2+dirty`, after the fix below and before its commit (`0b792bd`):
    log `20261025T080124Z.log`, 19.5 MB in 13.8 s.
- **Problems and fixes:**
  - **ecCodes is not thread-safe.** Two download threads decoding at once aborted the process.
    Decoding is now serialised with a lock; downloads stay parallel.
  - **SUNSD is accumulated, not instantaneous.** The index and the GRIB step type say
    "instantaneous", but the values are 10,800 s at the third hour of a sunny 6-hour window and
    21,600 s at the sixth. It is now de-accumulated into 3-hour sunshine fractions. A value
    longer than the elapsed window raises (test).
- **Results:**
  - 26 files and 11 messages per file decoded at the 3×3 points.
  - Averaging windows 6-9, 6-12, 12-15, …, as expected.
  - Files uploaded 15:39–15:59 UTC, i.e. 3.7–4.0 h after init and 5 h before T0.

### 2026-10-25 · R6b-02 · Multi-period sample: download, then a failure in the physical chain

- **Purpose:** the user asked for a sample from different periods before deciding on the full
  download: "so that there is no problem as with the previous forecasts". 10 blocks of 3
  consecutive runs:
  - train: every season and both DST days;
  - validation;
  - test: technical checks only.
- **Command and commit:** `python 00_data_audit/gfs_sample_2026-10-25/check_gfs_sample.py` at
  `0b792bd`, log `00_data_audit/gfs_sample_2026-10-25/runs/20261025T080307Z.log`.
- **Results:** the 29 new runs were downloaded in 295 s (about 0.57 GB), with no fallback run.
- **Problems and fixes:**
  - The run then stopped in `physical_features`. The extraterrestrial irradiance at mid-hour
    came back on the shifted index, and pandas aligned it against the row index (74 vs 148).
  - That function was the only one without a unit test. Fixed (`f96b894`), and a clear-day test
    on real geometry was added.

### 2026-10-25 · R6b-03 · Sample check: three findings

- **Command and commit:** the same script at `f96b894`, log `20261025T080910Z.log`.
- **Results:** 30 of 30 runs complete; all available at least 5.0 h before T0; phase against
  the clear sky and against the measurement at lag 0 in every block; the three horizons of the
  same target day differ.
- **Intermediate notes (self-review of the output):**
  - **Dark hours with irradiance.** Up to 27 W/m² of GFS irradiance at night, in spring. The
    twilight rule spread a sunrise block evenly over its three hours, including a dark hour.
  - **Packing noise.** Tiny negatives (−0.02 W/m²) and −0.x % cloud.
  - **The wrong accuracy measure.** The script compared GFS kt_poa with the plant's k. The
    plant's k includes the system losses (mean 0.79), so the "error" of 0.33 was mostly the
    difference of scale.
- **Fixes (`d0b743b`):**
  - Every block with any clear-sky irradiance is now distributed in proportion to it: the
    block mean is kept, and dark hours get 0. Only a fully dark block is spread evenly.
  - The packing noise is clipped.
  - The comparison now uses the same quantity on both sides: GFS kt_poa against the measured
    irradiance / POA_cs, hourly and daily.

### 2026-10-25 · R6b-04 · Sample check: all checks pass

- **Command and commit:** the same script at `d0b743b`, log `20261025T081130Z.log`.
- **Results** (`sample_summary.json`):
  - 30 of 30 runs complete; no fallback; averaging windows as expected; all files uploaded at
    least 5.0 h before T0.
  - Irradiance never negative; at most 1.1 W/m² in hours whose clear sky is below 1 W/m².
  - Phase: lag 0 against the clear sky in all 30 runs, and against the measurement in all 8
    train and validation blocks.
  - The forecasts of the same target day from D+1, D+2 and D+3 differ in every block (N2).
  - GFS kt_poa against the measured clear-sky index, train and validation only:

    | Horizon | Hourly r | Daily r | Bias | MAE |
    |---|---|---|---|---|
    | D+1 | 0.64 | 0.83 | +0.02 | 0.11 |
    | D+2 | 0.71 | 0.91 | +0.01 | 0.10 |
    | D+3 | 0.37 | 0.44 | −0.01 | 0.11 |

    About 245 hours and 24 days per horizon.
  - Neither red flag of the original's problem is raised: skill is far from perfect at D+3, and
    it does not rise with the horizon.
- **Intermediate notes:**
  - D+2 above D+1 is a sample effect. Each horizon covers a slightly different set of 24
    days, and the cloudy days differ.
  - The full period will measure the horizon effect on all train days.
  - The figure shows the forecast following the measurement, including the overcast day of
    2022-01-19, forecast correctly as low irradiance.
- **Decision / next:** the full download (1°, about 27 GB of transfer) and the optional 0.25°
  comparison are left to the user, as agreed.

### 2026-10-25 · R6b-05 · Full download

- **Purpose:** after reviewing the sample, the user chose the full set of 11 messages ("the
  first option, download everything").
- **Command and commit:** `python -m solarpv.nwp.download --workers 16` at `b8a712e`, log
  `outputs/06b_gfs/runs/20261025T082013Z.log`.
- **Results:**
  - 1,341 new runs in 2.2 h, besides the 30 of the sample: 26.5 GB transferred, 48 MB stored.
  - No 12Z run missing; no fallback.
- **Intermediate notes:** 16 parallel requests per run brought it from 10 to 6 s per run.

### 2026-10-25 · R6b-06 · The guard catches late re-uploads

- **Command and commit:** `python -m solarpv.nwp.gfs` at `069c421`, log
  `20261025T103052Z.log`: failed, as designed.
- **What happened:**
  - The T-17 guard of the build refused run 2022-01-24 12Z. One of its 26 files (f027) carries
    `Last-Modified` 22:06Z, six minutes after T0 (22:00Z).
  - A scan of all 35,646 files found 7 such files, in 2 runs (2022-01-24 and 2022-07-23). They
    were re-uploaded about 10.4 h after init; every other file appeared within 5.2 h.
  - The archive keeps only the latest version, so the version that was public at T0 cannot be
    shown.
- **Fix (`9cd4221`):**
  - A run with any file uploaded after T0 is rejected (record kept, status
    `rejected_late_upload`).
  - The 06Z run of the same day is used, flagged.
  - `python -m solarpv.nwp.download --recheck` (log `20261025T103312Z.log`) applied it: 2
    runs rejected, 2 runs of 06Z fetched (40 MB).

### 2026-10-25 · R6b-07 · Tables, sanity, value check, and a red flag examined

- **Commands and commit:** at `9cd4221`:
  - `python -m solarpv.nwp.gfs` (log `20261025T103344Z.log`, 284 s);
  - `python -m solarpv.nwp.check` (log `20261025T103845Z.log`).
- **Results:**
  - Every hour of D+1, D+2 and D+3 has GFS values (32,856 rows each).
  - **Sanity (train and validation):**
    - daily r of GFS kt_poa with the measured clear-sky index: 0.82 / 0.79 / 0.74, falling
      with the horizon;
    - best lag 0 h;
    - seasonal bias from −0.025 to +0.012;
    - no red flag.
  - **Value check** (fixed XGBoost of D-11, 3 seeds, validation nRMSE in kW):

    | Target, horizon | Without GFS | With GFS |
    |---|---|---|
    | indirect D+1 | 7.94 % | 6.85 % |
    | indirect D+2 | 8.31 % | 7.31 % |
    | indirect D+3 | 8.39 % | 6.72 % |
    | direct D+1 | 7.76 % | 6.88 % |
    | direct D+2 | 8.13 % | 7.06 % |
    | direct D+3 | 8.20 % | 6.63 % |

  - **D-12: admitted.** Every gain is 0.9–1.7 pp, against margins of at most 0.13 pp.
- **Intermediate notes (self-review):**
  - With GFS, D+3 is better than D+1 on validation. GFS itself loses skill with the horizon,
    so this is a red flag by the plan.
  - Hypothesis 1, the past-day features: refuted. The same order appears with GFS features
    alone (`diagnose_horizons.py`, log `00_data_audit/gfs_value_2026-10-25/runs/20261025T104116Z.log`).
  - Hypothesis 2, a wrong mapping of horizons to runs: refuted. The leads are 10–34 / 34–58 /
    58–82 h, and the init lies 9 or 10 h before T0, 16 h for the two fallbacks.
  - The cause: on validation alone, GFS at D+3 correlates as well as at D+1 (0.62 / 0.57 /
    0.62). On train it is 0.64 / 0.63 / 0.56.
  - Confirmation (`diagnose_sample.py`, log `20261025T104450Z.log`):
    - on an inner period of train, the order is the expected one: indirect 7.17 / 7.51 / 7.71;
    - on validation, the paired bootstrap CI of RMSE(D+3) − RMSE(D+1) includes 0: indirect
      −0.20 pp [−0.49, +0.03], direct −0.27 pp [−0.59, +0.03].

    A sample effect, not a leak.

### 2026-10-25 · R6b-08 · Tests and documentation

- The full suite passes: 317 tests, including T-17 row by row on the real tables and T-44.
- OP-42–OP-44 and N13 are recorded; PROJECT_MASTER v3.0.

## Audit: the original backup

### 2026-11-01 · RA-01 · Inventory of the backup

- **Purpose:** the user placed an older backup of the original project (about January 2026) in
  `old files/solar_ds_capstone/` and asked for a thorough review: a step on the way, not the
  final project.
- **Command:** `python 00_data_audit/original_backup_2026-11-01/inventory.py`, log in
  `00_data_audit/original_backup_2026-11-01/runs/`.
- **Results:** 508 files in the working copy; a partial `.git` (no HEAD, config or root tree)
  with 483 loose blobs, 400 of which match no working-copy file and were recovered with zlib
  only. None of the final model code (xgb_sota, lstm_sota, TFT, PatchTST) or of 05c/05e/05f.
- **Intermediate notes:** the first classification put the notebooks by words found anywhere in
  their code (a feature notebook that loads a model file looked like modelling); it now uses the
  title first and the code only as a fallback.

### 2026-11-01 · RA-02 · Feature-selection evidence

- **Command:** `python 00_data_audit/original_backup_2026-11-01/selection_evidence.py`.
- **Results:** the January base notebooks end with the 42 starting features of the submitted D+1
  05e files (both targets); D+2 and D+3 start from the same list without the unavailable lags.
  No split or date filter (F-26). The direct one did not forbid per-inverter DC: 95 % of the SHAP
  importance (F-27). Parameters: MI 0.02, a core of 12, RFE one feature per step to 42.

### 2026-11-01 · RA-03 · The original weather source

- **Command:** `python 00_data_audit/original_backup_2026-11-01/weather_source.py` (train and
  validation only).
- **Results:** the archive endpoint with `gfs_seamless`, one request for the whole period (F-28).
  Best alignment at a one-hour shift (hourly r 0.631); 0.301 / 0.268 at the original 2–3 h join.
  Daily r 0.822, against 0.806 for the real D+1 GFS run.
- **Self-review:** the direction of the original offset was derived from F-01 (local labels
  treated as UTC) and the hour-end label of the API; whichever direction, every shift other than
  the best one keeps less than half of the hourly correlation.

## Stage 7: feature selection

### 2026-11-01 · R7-01 · First run: the duplicate step removed real information

- **Command and commit:** `python -m solarpv.selection.run` at `98c208f`, log
  `outputs/07_feature_selection/runs/20261101T123534Z.log` (815 s).
- **S0 (k history, inside train):** gain at D+1 +0.026 pp against a margin of 0.054 pp: not
  admitted (D-24).
- **Problem found:** the duplicate step removed 33–43 features per set, among them every rolling
  mean of production ("duplicate of aoi") and every power lag ("duplicate of irradiance lag").
  Groups were connected components of |r| > 0.98, so they chained, and on daytime rows every
  same-time-of-day variable follows the diurnal shape. The direct 05e sets were then 0.4–0.5 pp
  worse than the full pool on validation.
- **Fix (`46ef699`):** the January base stage had no such step; pairs are now reported, not
  removed, and the step-1 RFE removes redundant copies. Plan §7. The validation comparison of
  this run is void.

### 2026-11-01 · R7-02 · Second run, and the sign of the red flag

- **Command and commit:** the same at `46ef699`, log `20261101T125246Z.log` (1,780 s); then
  `python -m solarpv.selection.run --validation-only` at `2aa990c`, log `20261101T132514Z.log`.
- **Results:**
  - 05e sets: indirect 25 / 26 / 26, direct 19 / 16 / 12 (submission: 40 / 23 / 19 and
    38 / 32 / 25). GFS features lead every SHAP ranking; no dominant feature.
  - The MI vote endorsed all 42 starting features in every set: the base-stage MI filter
    (0.02) had already removed the features near the shuffled null, so the 2-of-3 consensus
    equals the union of the SHAP elbow and the stability votes. Kept as registered (D-23).
  - Validation, once (`validation_check.csv`): the 05e set minus the full pool is −0.015 /
    +0.005 / +0.088 pp (indirect D+1 / D+2 / D+3) and −0.005 / +0.333 / +0.425 pp (direct).
    Red flags, above the margins: indirect D+3, direct D+2 and D+3.
- **Second problem found:** S8 had negated the gap, so the manifest showed no red flag. Fixed in
  `2aa990c` with a test that checks gap, flag and manifest against each other (it failed on the
  old outputs). The recomputed numbers are identical; three red flags.

### 2026-11-01 · R7-03 · Diagnosis of the red flags

- **Command:** `python 00_data_audit/selection_2026-11-01/diagnose_gap.py` at `2aa990c`
  (`gap_inner_train.csv`, `gap_validation_views.csv`).
- **Direct, a method problem.** The fixed XGBoost of D-11 is also worse with the 05e set inside
  train: +0.27 / +0.09 / +0.42 pp on the date folds and +0.11 / +0.22 / +0.27 pp on an inner
  hold-out (fit to April 2023, scored May 2023 to January 2024). On validation the gap is in
  normal-operation hours (+0.42 / +0.50 pp at D+2 / D+3) and in most months, so it is not the
  curtailment of spring 2024. The selection model (100 trees, depth 5, as in the base notebooks)
  cannot use much beyond the GFS power features: at D+3 one feature came within 1 % of all 42.
  The sets are therefore too small for the model that uses them.
- **Indirect, a sample effect.** No gap inside train (−0.12 to +0.06 pp); the +0.09 pp at D+3 on
  validation is spread over the months, like the D+3 effect of R6b-07.
- **Open:** how to fix the direct selection (user decision).

### 2026-11-03 · R7-04 · Rerun with the D-11 model, and the inside-train check

- **Command and commit:** `python -m solarpv.selection.run --skip-validation` at `190e4df` on the
  college machine, log `outputs/07_feature_selection/runs/20261103T113918Z.log` (5,585 s, no
  warnings). Both routes selected with the fixed XGBoost of D-11 on 8 threads (plan §7, R7-04).
- **S0 (k history, inside train):** gain +0.048 / +0.059 / +0.020 pp at D+1 / D+2 / D+3 against
  margins of 0.050 / 0.050 / 0.076 pp. D+1 misses its margin by 0.002 pp, so the k history is not
  admitted (D-24), as on the laptop (R7-01). The rule is applied as registered.
- **05e sets:** indirect 31 / 28 / 32, direct 18 / 19 / 15 (laptop R7-02: 25 / 26 / 26 and
  19 / 16 / 12; in common with it: 24 / 23 / 23 and 17 / 14 / 12). The MI vote again endorsed all
  42 starting features in every set (R7-02). No red flag of S7; GFS features lead every ranking.
- **Inside-train check** (`inner_train_check.csv`, sha256 `c457837a60b2831e`): the 05e set minus
  the full pool, date folds / inner hold-out, in pp, with the margins:
  - indirect D+1: −0.058 / −0.012 (0.050 / 0.050), passes;
  - indirect D+2: −0.074 / +0.054 (0.081 / 0.093), passes;
  - indirect D+3: −0.014 / −0.029 (0.053 / 0.067), passes;
  - direct D+1: +0.287 / +0.206 (0.050 / 0.050), fails;
  - direct D+2: +0.102 / +0.131 (0.064 / 0.070), fails;
  - direct D+3: +0.714 / +0.202 (0.170 / 0.082), fails.
  Sanity: the full-pool error rises with the horizon on both samples and in both routes.
- **Finding F-29, the direct sets are too small for a reason the model does not change.** From
  `selection_votes.csv` (mean |SHAP| of the 42 starting features): in the direct route the top
  three features hold 64.5 / 54.4 / 37.0 % of the importance, against 31.2 / 31.7 / 27.6 % in the
  indirect route, and only 15 / 16 / 19 of the 42 hold at least 1 %, against 34 / 38 / 35. The
  stability vote needs a 1 % share, so it keeps 12 / 11 / 15 features; the 24 / 23 / 27 starting
  features left out of 05e still carry 14.1 / 14.7 / 29.1 % of the importance together. The 05c
  curve is one seed and noisy: at direct D+3 one feature scores 8.516 against 8.531 for all 42,
  so the 1 % elbow is a single feature; that 1 % (0.085 pp) equals one pooled seed std of the
  score (the margin of 0.170 pp is two). A limitation of the registered rules (a relative
  share and a one-seed curve), not a code bug. The original pipeline never compared its 05e sets
  with the full pool, so it could not see it.
- **Decision, before the validation look (R7-04 steps 3 and 4):** the direct route takes the full
  pool at every horizon (96 / 89 / 82 features); validation is not used for that decision. The
  indirect route passed, so the third validation look decides between its 05e set and its full
  pool; the direct rows of that look are reported only. New `model_inputs.json`, written by
  `run.decide_input` (unit and contract tests, T-48 and T-50), records the input of stage 9.
  Committed as `ed78b59` before the look.

### 2026-11-03 · R7-05 · The third validation look, and the input of stage 9

- **Command and commit:** `python -m solarpv.selection.run --validation-only` at `ed78b59`
  (clean), log `outputs/07_feature_selection/runs/20261103T131740Z.log` (75 s). The third look of
  stage 7: R7-01 (void) and R7-02 were the first two, on the laptop with the same data.
- **Results** (`validation_check.csv`; the 05e set minus the full pool, pp, with the margin):
  - indirect: +0.021 (0.050) / −0.016 (0.078) / +0.033 (0.122) at D+1 / D+2 / D+3, no red flag;
  - direct, reported only: −0.068 (0.058) / +0.192 (0.101) / +0.378 (0.089), red flags at D+2
    and D+3.
- **Input of stage 9** (`model_inputs.json`): indirect, the 05e sets of 31 / 28 / 32 features;
  direct, the full pool of 96 / 89 / 82 (decided in R7-04).
- **Review (forecasting-validity-review):**
  - Train rows only for the selection; validation read once, by the registered step; test not
    read (manifest: `splits_read_for_selection` = train, `validation_read` = S8 only).
  - The indirect D+3 red flag of R7-02 is gone with the larger selection model.
  - For the direct route at D+1 inside train and validation disagree (+0.29 / +0.21 pp inside
    train, −0.068 pp on validation). The registered rule decides, so the full pool stays; a
    choice made from this number would be a fourth look. Reported as a limitation.
  - The error on validation is lower at D+3 than at D+1 in both routes. This is the
    validation-sample effect examined in R6b-07 (inner train in the expected order, bootstrap
    interval including 0), not a new finding; inside train the order is D+1 < D+2 < D+3 here.
- **Decision / next:** stage 7 is closed. Next: notebook 05, then stage 9 with these inputs.

## Hand-off to the college machine

### 2026-11-03 · RH-01 · The package tested on an unpacked copy

- **Purpose:** prove that the hand-off package (`scripts/handoff_pack.py`, `docs/HANDOFF.md`)
  works before it leaves the laptop.
- **Command and commit:** `python scripts/handoff_pack.py` at `b18ac63`, unpacked into a scratch
  folder; there `python scripts/handoff_pack.py --verify`, `git status`, `git fsck` and the full
  test suite with the copy's own `src/` first on the path.
- **Results:** 3,477 files, 1.71 GB before compression and 1.07 GB packed. All 3,392 files outside
  `.git` matched the manifest, `git status` showed only the untracked stage-7 outputs, and
  `git fsck` found no errors (only dangling objects).
- **Problems and fixes:** without `configs/site.private.yaml` (left out of the package, Q-16),
  7 tests failed instead of being skipped. Eight contract tests compare the real outputs with the
  sun of the true site: the five that use the `sun` fixture in `test_clean_output.py` and the
  three clock tests in `test_preprocess_output.py`. With the placeholder site (longitude 34.9°)
  solar noon moves and, for example, the meteo centroid offset reached 2.26 min, above the
  2-minute tolerance. They now carry `needs_private_site`, as the tests of the daytime mask
  already did. After the fix, on the copy without the file: 319 passed, 38 skipped, none failed.
  On the laptop with the file: 357 passed.
- **Decision / next:** the package is rebuilt at the commit of this fix.

### 2026-11-03 · RH-02 · First session on the college machine

- **Purpose:** the first-session steps of `docs/HANDOFF.md` §3 on the college machine.
- **Command and commit:** the package of `ee976e4` (`HANDOFF_INFO.json`), unpacked into
  `C:\Users\204323141\Desktop\solar`.
- **Results, before any tool was installed:**
  - The machine had no Python, git or conda.
  - The manifest check was repeated with PowerShell (`Get-FileHash -Algorithm SHA256` over every
    line of `HANDOFF_MANIFEST.sha256`, the same check as `verify()`): 3,392 files, 0 missing,
    0 changed.
  - `configs/site.private.yaml` is present (copied by the user; its content was not read).
  - The machine was inspected (`docs/stage9_plan.md` §1.1).
  - The four memory notes of `docs/claude_memory/` were copied to Claude's memory directory,
    with one new note on the machine.
- **Tools installed (the user approved, same day):** per user, no administrator rights, not on
  `PATH`; both installers matched the SHA-256 that GitHub publishes for the release asset and
  carry a valid Authenticode signature.
  - Miniforge 26.9.4-1 (conda 26.9.4) in `C:\Users\204323141\miniforge3`.
  - PortableGit 2.56.0.4 in `C:\Users\204323141\PortableGit` (laptop: Git 2.55).
- **Results, after the install:**
  - `git worktree prune` removed the stale laptop worktree `college-vm-connection-7492d4`.
  - `git status`: `main` at `ee976e4`; untracked only the stage-7 outputs listed in
    `HANDOFF_INFO.json`; modified only the three documents edited in this session.
  - `git fsck`: no errors, only dangling objects (as in RH-01).
  - `python scripts/handoff_pack.py --verify`: 3,392 files, 0 missing, 3 changed, and the three
    are the documents edited in this session after the PowerShell check above.
  - `conda env create -f environment.yml`: done. Against `environment.lock.yml`, 92 of 299
    versions differ. Unchanged: numpy 2.4.6, scikit-learn 1.9.1, xgboost 3.4.2, shap 0.51.0,
    pvlib 0.16.1, scipy 1.17.1, statsmodels 0.15.0, optuna 5.0.0, xarray 2026.9.0, cfgrib
    0.9.15.1, matplotlib 3.11.2, OpenBLAS 0.3.34. Changed: python 3.11.17 → 3.11.18, pandas
    3.0.6 → 3.1.0, pyarrow 25 → 26, python-dateutil 2.9.0 → 4.0.0, eccodes 2.49 → 2.50, hdf5
    1.14.6 → 2.2.0, tzdata 2026c → 2026e, numba 0.68.0 → 0.68.1, pytest 9.1.1 → 9.2.0, ruff
    0.17.0 → 0.17.3. conda saw the GPU and chose the CUDA build of XGBoost (`cuda129`, with
    `cuda-cudart` 12.9); the CPU `hist` path is the same code.
  - `pytest -q`: **357 passed, none skipped, none failed**, in 3.1 minutes, which equals the
    laptop baseline with `site.private.yaml` (`docs/HANDOFF.md` §8).
- **Problems and fixes:**
  - 21 warnings that pandas 3.1 adds: `DatetimeIndex.dayofyear` and `.dayofweek` are deprecated
    (in pvlib and in `solarpv/features/build.py` and `solarpv/nwp/gfs.py`). No behaviour change
    now; they break under pandas 4, which `environment.yml` (`pandas>=2.2`) does not exclude.
    Not fixed in this session.
- **Decision / next:** offered the choice of aligning the environment with the laptop lock
  (pandas 3.0.6, pyarrow 25, CPU XGBoost) before the stage-7 rerun, the user kept the
  environment as created (2026-11-03). `environment.lock.yml` still describes the laptop; the
  versions above are those of every run on the college machine from now on. Next: the open
  decisions of `docs/HANDOFF.md` §5.

### 2026-11-03 · RH-03 · Provenance and thread count on the college machine

- **Purpose:** the first stage-7 run here (`92a2309`, log `20261103T112436Z.log`, stopped after
  S0 and indirect D+1 when the user changed the selection model, R7-04) showed two machine
  effects.
- **Problem 1, provenance:** `git_commit()` returned `unknown`, because git is the per-user
  PortableGit, not on `PATH`, and the fallback was the laptop's `C:\Program Files\Git`. Every
  manifest written here would have lost its commit. Fix: `find_git()` also tries
  `~\PortableGit\cmd\git.exe` (`tests/unit/test_provenance.py`).
- **Problem 2, thread count:** S0 at D+1 gave a gain of +0.048 pp against a margin of 0.050 pp,
  where the laptop gave +0.026 pp against 0.054 pp (R7-01). Same XGBoost 3.4.2. Check: the same
  `cv_score` twice with all cores gave the same value (7.620977), with 4 threads 0.013 pp less.
  `hist` sums its histograms per thread, so the result depends on the thread count, and
  `n_jobs=-1` made it depend on the machine. Fix: `XGB_THREADS = 8` in every XGBoost parameter
  set; on this machine (8 vCPUs) 8 threads reproduce the all-core value exactly (difference 0.0).
- **Decision / next:** the laptop work is a bootstrap, not a target (user, 2026-11-03), so
  laptop numbers are not reproduced; the college runs are canonical from here.
  `environment.lock.yml` regenerated from the college environment (the laptop's lock stays in git
  history).

### 2026-11-03 · RH-04 · No Jupyter kernel on the college machine; notebook 05 built in process

- **Purpose:** build `notebooks/05_feature_selection.ipynb`, the last open item of stage 7.
- **Problem:** `python scripts/make_feature_selection_notebook.py` aborted as soon as nbclient
  started the kernel: `Bad file descriptor (...\zeromq...\src\epoll.cpp:73)`, exit code
  0x40000015. A bare `zmq.Context().socket(zmq.PAIR)` aborts the same way, inside and outside the
  command sandbox, so no Jupyter kernel or notebook server can start on this machine. pyzmq
  27.2.0 with the conda-forge libzmq 4.3.5, whose Windows poller (wepoll) cannot be created
  here. `netsh winsock show catalog` lists no layered provider on TCP (only Microsoft providers
  and VMware vSockets), so the cause is outside the repository, most likely the VDI's endpoint
  security; not confirmed.
- **Fix, without changing the environment:** `scripts/_notebook.py` checks in a child process
  whether a zmq socket can be created (libzmq aborts the whole process, so the check cannot run
  in the builder). If not, the cells run in an IPython shell in the builder's process; stdout,
  display calls and the cell's value are stored as nbclient stores them, a failing cell raises,
  and the notebook metadata records the executor. Where a kernel starts (the laptop) nothing
  changes. Test: `tests/unit/test_notebook_helper.py`.
- **Results:** notebook 05 built and executed in 11 s, 12 code cells, each followed by a note
  formatted from the outputs; every interpretive sentence is guarded by an assert in the
  builder. `ruff check src scripts tests notebooks` clean; `pytest -q`: 372 passed, none skipped,
  none failed (369 + notebook 05 + 2 executor tests). The 227 ruff findings of `ruff check .`
  are all in the third-party `.claude/skills/` and predate this session.
- **Review (forecasting-validity-review) of the numbers shown:** train only for the selection,
  validation once (S8), test not read (manifest); both routes scored on the same 9,508 hours in
  kW; margins fixed in advance, three seeds; `assert_no_leakage` on every stage-9 input in the
  notebook; the decision rule recomputed from the two checks gives the recorded source for all
  six sets; D+3 below D+1 on validation is the R6b-07 sample effect, inside train the order is
  the expected one. No new decision.
- **Decision / next:** a live demonstration in Jupyter is not possible on this machine until the
  libzmq problem is solved (for example a libzmq built with the `select` poller, which needs a
  download, or a fix on the machine: the user's decision). Next: stage 9 per
  `docs/stage9_plan.md` W1–W6.

### 2026-11-05 · RH-05 · Jupyter kernels start on the college machine: libzmq on TCP via a long TMP

- **Purpose:** a real Jupyter kernel on the college machine (the open item of RH-04), so the stage
  notebooks can be run live for the committee.
- **Diagnosis, with nothing changed:**
  - The abort is not in wepoll. Repeated with ctypes, every call wepoll makes succeeds:
    `NtCreateKeyedEvent`, a new I/O completion port, `NtCreateFile` on `\Device\Afd\Wepoll` with
    its port association and completion modes, and `SIO_BASE_HANDLE` on a TCP and an AF_UNIX
    socket (both Microsoft IFS providers, protocol chain of length 1).
  - `epoll.cpp:73` is the assert after `epoll_ctl(ADD)` in `add_fd`, not the creation of the
    poller. The socket it adds belongs to the pair that libzmq builds behind every zmq socket
    (the mailbox signaler).
  - This libzmq is built with ipc (`zmq.has('ipc')` is true), so on Windows it builds that pair
    over AF_UNIX, with a socket file in `GetTempPath()`, which reads TMP first. Here `bind`
    succeeds and `connect` fails with `WSAEINVAL` (10022), in the temp folder under its short and
    its long name, inside and outside the command sandbox.
  - libzmq falls back to its TCP loopback pair only when the AF_UNIX socket or address cannot be
    made. After a successful `bind` it returns an error instead, the signaler keeps an invalid
    socket, and `epoll_ctl` rejects it (EBADF). Each failed probe left two `xNNN.N\socket`
    folders in the temp folder (26 by now).
  - Those socket files cannot be opened, not even with `FILE_FLAG_OPEN_REPARSE_POINT`, nor
    deleted (`ERROR_CANT_ACCESS_FILE`, 1920). A file-system filter of the VDI most likely rejects
    the AF_UNIX reparse tag. Not identified: `fltmc` needs administrator rights, and the
    registered filters are Microsoft's plus `vmwcdrfilter` (Horizon client drive redirection).
- **Correction to RH-04 (self-review):** the poller was not the cause, so a libzmq with the
  `select` poller would have aborted the same way. RH-04's "cannot create its poller" is wrong;
  the cause is the AF_UNIX signaler.
- **Fix, with no download and no environment change:** an AF_UNIX address holds 108 bytes
  (`sun_path`). With a longer TMP libzmq cannot form it and uses its TCP loopback pair, as it does
  where AF_UNIX is missing.
  - `scripts/_notebook.py`: `enable_kernel()` sets TMP to
    `%LOCALAPPDATA%\solarpv\zmq-tmp\path-longer-than-af-unix-sun-path-rh05_…` (109 characters
    here) and probes zmq in a child process; the kernel inherits TMP. This is tried first on
    every machine, because a failed AF_UNIX probe leaves another undeletable file in the temp
    folder; only if it fails is the plain TMP tried, and then the in-process executor of RH-04.
  - Python's `tempfile` reads TEMP before TMP, so its files do not move.
  - On the TCP fallback libzmq leaves one empty folder per socket pair; `clean_zmq_tmp()`
    removes the empty folders after the run.
  - The notebook metadata keeps `executor` and adds `zmq` when the long TMP was used.
  - `scripts/jupyter_lab.py` starts JupyterLab on the repository with the same TMP, for a live
    run.
- **Tests:** four new tests in `tests/unit/test_notebook_helper.py`: the TMP is longer than 108
  characters for any base folder; the long TMP is tried first and TMP is restored when it fails;
  only empty folders are removed; a zmq socket can be created on this machine.
- **Results:**
  - The plain probe still aborts (the machine is unchanged); with the long TMP it passes.
  - `python scripts/make_feature_selection_notebook.py`: built through nbclient in 26 s,
    `executor` = "nbclient, python3 kernel". Same source hashes, the same 12 code cells and output
    types, no error, and the HTML of every table and every text output are equal to the
    in-process build. The only difference is the plain-text form of wide tables: the in-process
    shell had cut them to the terminal width ("..."), the kernel wraps them. Jupyter shows the
    HTML, so nothing visible changed; this is a known limitation of the RH-04 fallback.
  - `scripts/jupyter_lab.py`: JupyterLab 4.6.5 (Jupyter Server 2.21.1) started on localhost; a
    kernel opened through the REST API ran a cell, with the 109-character TMP inherited; the
    server stopped cleanly, and no folder was left.
  - No new folder in the temp folder after the final build (26 before and after).
  - `ruff check src scripts tests notebooks` clean; `pytest -q`: 376 passed, none skipped,
    none failed (372 + 4).
- **Left on the machine:** the 26 libzmq folders and 9 socket files of this diagnosis in
  `%TEMP%`, 0 bytes each, which cannot be deleted (1920). They are harmless; a refresh of the VDI
  or the college IT can remove them.
- **Decision / next:** a live run: `python scripts/jupyter_lab.py`. A report to the college IT
  about the filter is optional (the user's decision). The older builders
  `make_eda_notebook.py`, `make_features_notebook.py` and `make_quality_notebook.py` call
  NotebookClient directly and would still abort here; they are not in the stage-notebook list.

### 2026-11-11 · RH-06 · The GPU environment `solar-gpu` (D-29, approved by the user)

- **Decision (user, 2026-11-11):** proposal A of `docs/stage9_plan.md` §18.5. `solar` was not
  touched.
- **Conda part** (`environment-gpu.yml`, conda-forge): a dry run first, then
  `conda env create`; Python 3.11.18, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, scikit-learn
  1.9.1, pyarrow 26.0.0, optuna 5.0.0, pytest 9.2.0. The full export is
  `environment-gpu.lock.yml` (163 lines).
- **Pip part** (`requirements-gpu.txt`, direct pins):
  - `pip download` of `torch-2.5.1+cu121-cp311-cp311-win_amd64.whl` from
    `download.pytorch.org/whl/cu121` into a new folder outside the repository,
    `C:\Users\204323141\gpu_wheels\cu121\`: 2,449,385,544 bytes, SHA-256
    `4bcee18f00c43c815efad8efaa3bca584ffdc8d2cd35ef4c44c814f2739d9191`, equal to the index.
  - `pip install --dry-run --report` resolved 43 packages: torch from the local folder, 42 from
    PyPI, none of the conda packages replaced. The 42 files (48,758,096 bytes) went to
    `C:\Users\204323141\gpu_wheels\pypi\`; every file was checked against the report's
    SHA-256, and `requirements-gpu.lock.txt` lists all 43 with their hashes.
  - Installed with `--no-index --find-links <the two folders> --require-hashes --no-deps`, then
    `pip install -e . --no-deps --no-build-isolation`. `pip check`: no broken requirement.
- **Verification** (`python -m solarpv.models.neural_env`, exit 0): torch 2.5.1+cu121, CUDA
  runtime 12.1, cuDNN 9.1.0, arch list sm_50–sm_90 with `sm_61`, GPU "GRID P40-24Q" capability
  6.1 with 24 GB; an FP32 product of two 512 × 512 matrices on the GPU differs from the CPU by
  8.9e-7 of the largest entry; lightning, pytorch_forecasting and neuralforecast import.
- **Left on the machine:** the two wheel folders (about 2.5 GB), the hash-checked source of any
  rebuild of the environment.

## Stage 9: models and the runner

### 2026-11-05 · R9-01 · Self-review before the runner code (W1)

- **Purpose:** the review the project rules ask for before the code of a stage
  (`docs/stage9_plan.md` §10): the original orchestration, modelling and evaluation scripts
  (`submission_evidence/12_scripts/`), and my earlier code that the runner builds on.
- **Blocking decisions (§3):** none for W1. §1 lists no blocker. D-17 (development budgets) is
  "proposed" and is used only from W2; D-15 is decided on validation inside stage 9 (W2).
- **Findings in the original (M1–M6, errata C-36–C-41):** the reproduction went on after a
  failed step and scored whatever files were in `outputs/`; it printed hard-coded "expected"
  results; it overwrote its logs and outputs with no provenance; it fell back silently to default
  hyper-parameters and skipped missing seed files; it evaluated the test split in every run, and
  the ablation's `--quick` smoke test was scored on test; duplicated LSTM forecasts were dropped
  with `keep="last"` after an unstable sort.
- **Findings in my code (X1–X8):** reproduced before the fix where possible.
  - X2: two copies of one prediction file doubled the benchmark n (54 → 108, synthetic data);
    the run then failed in the DM table with an unrelated shape error.
  - X5: with a fixed clock, two `RunLog`s opened in the same second shared one file, and the
    first run's note was lost.
  - X1, X3, X4, X6, X7 and X8 were read from the code: a glob in `evaluate.main`, test rows read
    in every mode, the stale manifest, untracked code under a clean commit, split labels
    filtered silently, and the plan's copy of every prediction file to the canonical paths.
- **Fixes and tests:** T-51–T-55 (`tests/unit/test_pipeline_runner.py`,
  `tests/contracts/test_pipeline_output.py`, and new cases in `test_evaluation_8b.py`,
  `test_runlog.py`, `test_provenance.py`). Full suite: 407 passed, none skipped (376 + 31).
- **Laptop-era items of the plan** (`bundle`, `collect`, transfer, Q-16, Q-18): proposals in
  §11 of the plan, waiting for the user; none was implemented or dropped.
- **Literature (§12 of the plan):** L-09 registered for W2 with the D-12 margin and three seeds;
  nothing adopted in W1, so nothing was tested on validation.
- **Commit:** `c70af25` (code, tests, OP-49, errata).

### 2026-11-05 · R9-02 · `check` and `run --only baselines,evaluate` in smoke and dev mode

- **Purpose:** the acceptance of W1: the runner checks the environment and the inputs, runs the
  two steps that train nothing, and records everything; the dev run must equal the stage-8B
  validation tables, which were made by the evaluation layer alone.
- **Commands and commit,** all at `c70af25` with a clean code tree:
  - `python -m solarpv.pipeline check --mode smoke`: passed. Python 3.11.18, pandas 3.1.0,
    xgboost 3.4.2; torch, lightning, pytorch-forecasting and neuralforecast absent (needed from
    W4); 8 CPUs, 64 GB, GRID P40-24Q; private site; 11 of 11 inputs as recorded by their stage
    manifests or git.
  - `python -m solarpv.pipeline run --mode smoke --only baselines,evaluate`: run
    `outputs/runs/20261105T195802Z_smoke/`, ok; baselines 0.8 s, evaluate 1.5 s.
  - `python -m solarpv.pipeline run --mode dev --only baselines,evaluate`: run
    `outputs/runs/20261105T195821Z_dev/`, ok; baselines 2.1 s, evaluate 5.0 s.
  - The same dev command with `--resume 20261105T195821Z_dev`: both steps "skipped (up to
    date)"; `status` shows both outputs as recorded.
  - `check --mode final`: refused, as it must be (no `g1-frozen` tag).
- **Results:**
  - Smoke window: local days 2024-03-15 to 2024-04-11, 671 hours with the spring DST day, 321
    daytime hours with truth for every reference, target and horizon. The numbers (for example
    persistence D+1 16.21 %, climatology 12.78 % of capacity for the direct target) are a code
    check on four spring weeks, not a result of the study.
  - Dev run: all eight tables equal the committed `outputs/07_evaluation/val/` tables with a
    largest difference of 0 at the written precision (N12: 2,901 daytime hours; direct D+1
    persistence 9.68 %, climatology 8.04 %, clear sky 8.96 %), although the runner never loaded
    a test row and took the references from its own prediction files.
  - Nothing outside `outputs/runs/` changed (git status), and the test-access log was not
    touched.
- **Review (forecasting-validity-review):** issuance: the references only use the truth before T0
  (persistence DST-safe, T-38); split isolation: no test row loaded (spy in T-55); comparable
  evaluation: every reference on the same hours of each window, in kW; no decision taken, so no
  margin is needed; provenance: logs, step records with commit and hashes, OP-49. Persistence
  D+3 is slightly better than D+2 in both windows (dev 10.72 % against 10.82 %): persistence of h
  days is not a forecast that must degrade monotonically, and this is the validation-sample
  pattern of R6b-07; no action.
- **Problems and fixes:**
  - **X9 (found by this run):** `check` reported `clean tree False` for the dev run, because the
    smoke run folder was untracked. The final gate would then refuse after any smoke run,
    although the code was exactly the commit. Fix: a clean tree means no modified tracked file
    and no untracked file in the code paths; untracked outputs are counted but do not block. Test
    added (T-52). The smoke and dev runs above are unaffected (the gate applies to final only).
  - The `OUTPUT` lines of a step log name the temporary folder (`.evaluate.tmp/`), because the
    log is written before the rename; `step.json` holds the final paths. Left as is.
- **Decision / next:** W1 is done. Next: W2 (`xgb` with its training logs), after the user's
  answer on the proposals of §11 and a confirmation of D-17.

### 2026-11-07 · R9-03 · Self-review of W2 and the code of the `xgb` step

- **Purpose:** the review the project rules ask for before the code of a work item, and its
  implementation. The review itself (M7–M12 against the original, X10–X13 against my code) and
  the decision rules of §14 were written and committed on 2026-11-05 (`0cc8d17`), before any
  code or XGB result; this entry records how each finding was implemented and what the code
  added.
- **Blocking decisions (§3):** none. D-17 approved (20 trials, seed 42); D-15 is decided by
  the rule of §14.3 in this work item.
- **Findings and their fixes** (errata C-20, C-33, C-42–C-46, X10–X13; tests T-56–T-63):
  - M7: constraints by exact name, a name → sign map recorded per target × horizon and read back
    from the booster (`(1,0,0)` in the test);
  - M8: no imputation; NaN counts per feature in `xgb_config.json`. **§6.6 of
    `PROJECT_MASTER.md` still said "remaining NaN gets 0 in XGB"**: corrected with this work item
    (C-42);
  - M9 and X12: the objective calls `evaluate.to_kw`; in the synthetic smoke run the recorded
    objective equals the evaluation's `nRMSE_kW_pct` of every target × horizon (to the 6 written
    decimals);
  - M10: `best_params.json` holds the full parameter set, the best iteration and the rounds
    used; the boosters are saved with all their rounds (best + 50) and every prediction passes
    `iteration_range=(0, best + 1)`; a reloaded booster reproduces the written predictions
    exactly;
  - M11: fit masks per target; M12: allow-list, seed in both sampler and model, cap warning;
  - X10: `pipeline.data.model_table`; X11: the stage-7 provenance rows; X13: a new dict per
    fit, and the D-15 check refuses to finish if `XGB_PARAMS` changed.
- **Choices made in the code (not in the plan's text):**
  - `min_child_weight` is searched as an integer in [1, 20], like the other bracketed integer
    ranges of §6.11 (v6 does not say).
  - The automatic warnings of §3.5 have these thresholds: a best parameter within 2 % of a bound
    (in its sampling scale; at the bound for small integer ranges); a validation RMSE more than
    twice the train RMSE at the best iteration; the best iteration at round 0 or at the cap; a
    NaN loss; fewer completed trials than the budget; a refit that differs from its trial; a
    seed spread above 0.15 pp (final runs).
  - The D-15 decision file is in `checks.TRACKED_INPUTS`, but while it does not exist `check`
    reports it as **pending** (a warning, not an error), so the `baselines` and `evaluate` steps
    still run before D-15 is decided; the `xgb` step refuses without it. Once it exists it must
    be tracked and unmodified like any input.
  - X11 checks the stage-7 selection run log (`20261103T113918Z.log`, six tables). The log of
    the validation look (`20261103T131740Z.log`) names no input, so it adds no row.
  - `optuna` and `SQLAlchemy` moved to the required packages of `check`.
- **Check before any run:** `python -m solarpv.pipeline check --mode dev` passes on the real
  inputs: 18 of 19 inputs as recorded, the six stage-7 provenance rows match, and the D-15
  decision is pending.
- **Tests:** 24 new (T-56–T-63) and one changed (`bilstm (W4)` is now the first unimplemented
  step); the synthetic smoke run of `baselines,xgb,evaluate` passes. Full suite: 432 passed,
  none skipped (408 + 24); ruff clean.
- **Commit:** `a3b8467` (code, tests, errata, OP-50–OP-52, this entry), before any run on real
  data.

### 2026-11-07 · R9-04 · D-15: monotone constraints for k, decided on validation

- **Purpose:** the rule of §14.3, registered on 2026-11-05 before any XGB result: keep the v6
  constraint (+1 on the D-15 features present in the indirect input) unless dropping it wins.
- **Command and commit:** `python -m solarpv.models.xgb_checks d15` at `a3b8467` (clean code
  tree), 1.5 min; log `outputs/09_xgb_checks/runs/20261107T104005Z.log`. Train and validation
  rows only.
- **Model and rows:** the fixed XGBoost of D-11, seeds 42, 123, 777; 9,575 indirect fit rows of
  train; scored on the 2,901 daytime validation hours with AC truth (op-limited included). In
  the indirect inputs only `ghi_clearsky` is a D-15 feature, at every horizon.
- **Results** (nRMSE in kW, % of 3,449 kW, mean ± seed std):

  | Horizon | A, `ghi_clearsky` +1 | B, none | gain A − B | margin | passes |
  |---|---|---|---|---|---|
  | D+1 | 6.806 ± 0.001 | 6.836 ± 0.012 | −0.029 | 0.050 | no |
  | D+2 | 7.252 ± 0.031 | 7.257 ± 0.020 | −0.005 | 0.051 | yes |
  | D+3 | 6.701 ± 0.040 | 6.708 ± 0.085 | −0.007 | 0.133 | yes |

- **Decision (by the rule):** B needs gain_1 > margin_1; the gain is negative at every horizon
  (the constraint helps slightly), so **A, the v6 setting, stays for k**. The direct target keeps
  the D-15 list with +1 (not part of the question). `configs/monotone_decision.yaml`.
- **Cross-check:** setting B is the stage-7 model on the 05e sets, and its means equal the
  stage-7 validation look exactly (6.8356, 7.2567, 6.7078 in `validation_check.csv`, R7-05):
  `model_table` gives the same tables as the stage-7 loader.
- **C-33, the marginal sign** on the indirect fit rows of train (Spearman): k against
  `ghi_clearsky` −0.38, `poa_global_clearsky` −0.41, `sun_elevation` −0.37; AC +0.90, +0.93,
  +0.89 (the 21.10 values were −0.37/−0.40/−0.36 and +0.88 to +0.91 on slightly different
  rows). The marginal sign of k is negative, yet the conditional constraint does not hurt: with
  the cloud and calendar features fixed, k does not fall with clear-sky irradiance in the model.
  Reported, as §14.3 asks; the rule decides.
- **Review (forecasting-validity-review):** issuance: the inputs are those of stage 7 (T-17,
  T-18 hold); split isolation: no test row read (X10); identical rows for A and B; the margin is
  the registered D-12 form; validation is used as a decision split, as every earlier decision.
- **Commit:** `45c7a84` (the decision file, `outputs/09_xgb_checks/`, this entry).

### 2026-11-07 · R9-05 · Smoke run of `baselines,xgb,evaluate`

- **Purpose:** the acceptance of W2's code on the real tables: the step runs end to end, its
  predictions pass the evaluation's checks, and the records are complete. Not a result.
- **Command and commit:** `python -m solarpv.pipeline run --mode smoke --only
  baselines,xgb,evaluate` at `45c7a84`; run `outputs/runs/20261107T104110Z_smoke/`, ok in 25 s
  (xgb 19 s). Preflight: 19 of 19 inputs as recorded (the D-15 decision now present).
- **Rows** (D+1 to D+3 alike): fit 496 indirect and 1,239 direct (the last 56 local days of
  train); early stopping 296 and 631 rows of the 28-day window; objective 321 daytime hours, 40
  of them op-limited. NaN passed to XGBoost: `irradiance_clear_index_prevday` on 6 and 10
  indirect fit rows (D+1, D+3), the GFS ratios at night on about 690 direct fit and 302
  early-stopping rows; none on a scoring row.
- **Evaluation:** every reference and XGB on the same 321 hours; the values are a code check on
  four spring weeks with one random trial, not a result (for example XGB direct D+1 12.25 %
  against climatology 12.78 %).
- **Warnings (12, all expected with one trial on 56 days):** at every target × horizon, the one
  trial has `max_depth` = 10 (the bound) and the validation RMSE is 2.7–2,836 times the train
  RMSE (depth 10 on about 500–1,200 rows fits train almost exactly). The first trial of every
  study is the same draw, because every study is seeded with the same seed (M12c).
- **Records:** predictions, trials, learning curves, best parameters, configuration, two
  figures, six boosters and the Optuna study, all hashed in `step.json`.

### 2026-11-07 · R9-06 · Dev run of `baselines,xgb,evaluate` (D-17: 20 trials, seed 42), and its review

- **Purpose:** XGB-dev of stage 9 and the acceptance of §14.7. Every number below is a
  **tuning-split number**: validation chose the hyper-parameters and the stopping round (D-10,
  M12b), so it is optimistic by construction; the test split stays the only unbiased estimate.
- **Command and commit:** `python -m solarpv.pipeline run --mode dev --only
  baselines,xgb,evaluate` at `45c7a84`, clean tree, 19 of 19 inputs as recorded; run
  `outputs/runs/20261107T104151Z_dev/`, ok in 6.2 min (xgb 363 s, about 1 s per indirect and
  4 s per direct trial on 8 threads).
- **Rows** (equal to the counts checked before the session): fit 9,575 indirect and 20,393
  direct (nights included); early stopping 2,825 and 5,685 validation rows; objective 2,901
  daytime hours, 146 op-limited. NaN passed to XGBoost: `irradiance_clear_index_prevday` on 6
  and 10 indirect fit rows (D+1, D+3); the GFS ratios at night on about 9,650 direct fit rows,
  2,469 early-stopping rows and 2,469 predicted validation rows; none on a scoring hour.
- **Results** (nRMSE in kW, % of 3,449 kW, the 2,901 validation daytime hours; from
  `evaluate/val/benchmark_summary.csv`):

  | Target | Horizon | XGB-dev | Persistence | Climatology | Clear sky | Skill vs persistence | Skill vs CLIPER |
  |---|---|---|---|---|---|---|---|
  | indirect | D+1 | 6.79 | 9.68 | 8.06 | 8.96 | 0.33 | 0.21 |
  | indirect | D+2 | 7.12 | 10.84 | 8.06 | 8.96 | 0.37 | 0.22 |
  | indirect | D+3 | 6.66 | 10.76 | 8.06 | 8.96 | 0.40 | 0.26 |
  | direct | D+1 | 6.66 | 9.68 | 8.04 | 8.96 | 0.31 | 0.29 |
  | direct | D+2 | 6.81 | 10.82 | 8.04 | 8.96 | 0.37 | 0.34 |
  | direct | D+3 | 6.43 | 10.72 | 8.04 | 8.96 | 0.40 | 0.38 |

  The indirect route in k: 13.44 / 14.07 / 13.51 % of MEAN_DAYTIME_K. Normal-operation view
  (D-22, 2,755 hours): 4.85 / 5.36 / 4.75 indirect, 4.85 / 5.38 / 4.67 direct; the 146
  op-limited hours carry a large share of the error. Route comparison (paired, actual AC):
  the direct route is better by 0.14, 0.31 and 0.23 pp (v6 reported the opposite, by 1.7–2.1
  pp; one seed on the tuning split, so this is no conclusion; C2 is decided on test in stage 11).
  Against the fixed model of stage 7 on the same inputs (R7-05: 6.84 / 7.26 / 6.71 indirect,
  6.90 / 7.08 / 6.63 direct), tuning gains 0.04–0.27 pp.
- **Acceptance of §14.7:**
  1. XGB-dev is below persistence and climatology at every target × horizon: **yes** (DM with
     HLN and Bonferroni significant against all three references at every target × horizon;
     the smallest gap is 0.94 pp, indirect D+2 against climatology).
  2. Error rising with the horizon: **no, D+3 is below D+1 in both routes**, so the order was
     checked as the rule says. Inside train the order is normal for the same inputs (stage-7
     date folds: indirect 7.57 / 7.87 / 8.25, direct full pool 7.66 / 7.94 / 8.35; inner
     hold-out alike). On validation the paired circular block bootstrap of the squared-error
     difference D+3 − D+1 includes 0 in both routes (indirect −2,171 kW², 95 % CI [−5,852,
     1,000]; direct −3,582 kW², [−8,057, 102]); D+2 is the high one (D+3 − D+2 below 0 in both
     routes). T-09 passes. This is the validation-sample pattern of R6b-07 and R7-05; no action.
  3. The warnings were read (five): indirect D+1 validation RMSE 2.7 times the train RMSE and
     `max_depth` 10 and `learning_rate` 0.0103 at the edges; direct D+1 `min_child_weight` 20
     at the edge; **direct D+2 stopped at its cap** (best iteration 212 of 213 rounds,
     `n_estimators` near its lower bound, learning rate 0.014): the best of 20 trials had not
     converged. In all, 3 of the 120 trials stopped at their cap (two of direct D+2, one of
     indirect D+1). None is a defect of the code;
     they say that 20 trials leave the search unfinished, and that the v6 space is tight at
     some bounds. The space of §6.11 is not changed here (a change would be a new decision with
     its own rule); listed for G-1.
- **Review (forecasting-validity-review):**
  - Issuance: inputs from `model_inputs.json` through the allow-list and the leakage rules; GFS
    rows guarded by T-17; lags and rolling windows by T-09, T-18.
  - Split isolation: fit on train rows only; validation as the early-stopping set and the
    objective (the registered design); no test row read (X10); `test_access_log.csv` and
    `outputs/07_evaluation/` untouched (git status).
  - Comparable evaluation: every model on the same 2,901 hours, truth from the stage-6 table,
    kW for both routes, paired route comparison, CLIPER skill positive everywhere.
  - Uncertainty: one seed (D-17); bootstrap intervals of nRMSE (for example direct D+1
    [5.59, 7.79] %); no decision is taken from these numbers.
  - Red flags: no dominant feature by the stage-7 measure (mean |SHAP| on the validation
    daytime hours; top over third at most 6.7 for direct D+1, threshold 10): the leaders are
    GFS forecasts (`fcst_gfs_poa`, `fcst_gfs_pv_kw` for direct; `fcst_gfs_kt_poa_daymean`,
    `fcst_gfs_k_phys` for k), all public before T0. By total gain the two direct GFS features
    are 94 % at D+1, which is the gain measure's concentration on the first splits, not a
    leak. The horizon order is explained above. The margin over the references is plausible
    (skill vs persistence 0.31–0.40).
  - Provenance: the run folder with logs, step records and hashes; commit `45c7a84`; OP-50.
- **Decision / next:** XGB-dev is accepted as the stage-9 light model. Next: L-09 and R-14 on
  this run (§14.4, §14.5).

### 2026-11-07 · R9-07 · L-09 (bias correction), R-14 (mean bias) and A-14 on the dev run

- **Commands and commit:** at `45c7a84`, clean code tree, on the dev run of R9-06 (read only,
  every file checked against its step record):
  - `python -m solarpv.models.xgb_checks l09 --run 20261107T104151Z_dev`: 2.5 min, log
    `outputs/09_xgb_checks/runs/20261107T105022Z.log`;
  - `python -m solarpv.models.xgb_checks r14 --run 20261107T104151Z_dev`: 1 s, log
    `outputs/09_xgb_checks/runs/20261107T105328Z.log`.
- **L-09 (§14.4, registered before any model result):** base model the dev best parameters
  with the dev rounds fixed, seeds 42, 123, 777; OLS on 7,834 out-of-fold daytime train hours
  (five expanding date folds, normal operation); applied to the 2,901 validation daytime hours.
  Seed 42 on all of train with the dev rounds reproduces the dev forecast exactly (largest
  difference 0.0), so the base model is the dev final fit.

  | Target | Horizon | Uncorrected | Corrected | Gain | Margin | Passes | a (kW) | b |
  |---|---|---|---|---|---|---|---|---|
  | indirect | D+1 | 6.809 | 6.699 | +0.110 | 0.050 | yes | +15.0 | 0.985 |
  | indirect | D+2 | 7.216 | 7.001 | +0.216 | 0.175 | yes | +20.1 | 0.971 |
  | indirect | D+3 | 6.692 | 6.599 | +0.093 | 0.089 | yes | +29.8 | 0.979 |
  | direct | D+1 | 6.704 | 6.762 | −0.058 | 0.076 | no | −34.9 | 1.020 |
  | direct | D+2 | 6.805 | 6.936 | −0.131 | 0.050 | no | −65.0 | 1.058 |
  | direct | D+3 | 6.451 | 6.562 | −0.111 | 0.076 | no | −40.1 | 1.029 |

  (means over the three seeds, nRMSE in kW, %.) **By the rule: adopted for the indirect route,
  not adopted for the direct route** (`configs/mos_decision.yaml`). Reading: for k the
  correction lowers the kW forecast slightly (b < 1, small positive a), which removes part of the
  positive level bias found by R-14 below. For the direct target the out-of-fold fit asks to
  stretch the forecast (b > 1, negative a): the fold models with the few dev rounds (31–212)
  trained on less data are flatter than the full model, and on validation the stretch hurts.
  Side note: the tuned seed is the best of the three at every target × horizon (the
  hyper-parameters were chosen with it), so its value is optimistic by 0.02–0.09 pp against the
  three-seed mean.
- **Not yet done for L-09:** §14.4 says that an adopted correction "becomes part of the `xgb`
  step in a later commit, before G-1". That is a design change of the step (the correction
  must be refitted in every run from that run's best parameters and seeds, and the indirect
  prediction file holds k, so the corrected kW must be mapped back to k on daytime hours, or
  written as a separate model). It is left for the next session, with its own short design note
  before the code.
- **R-14 (§14.5, reported, not decided):** mean bias error of the dev forecast (forecast −
  actual) on the 2,901 validation daytime hours, 95 % circular block bootstrap interval (block
  24, 1,000 resamples, seed 42):

  | Target | Horizon | MBE (kW) | MBE (% of capacity) | 95 % CI (%) | Sign |
  |---|---|---|---|---|---|
  | direct | D+1 | +25.5 | +0.74 | [+0.31, +1.21] | positive |
  | direct | D+2 | −6.6 | −0.19 | [−0.70, +0.34] | includes 0 |
  | direct | D+3 | +24.2 | +0.70 | [+0.23, +1.19] | positive |
  | indirect | D+1 | +45.0 | +1.31 | [+0.88, +1.77] | positive |
  | indirect | D+2 | +44.8 | +1.30 | [+0.78, +1.80] | positive |
  | indirect | D+3 | +42.5 | +1.23 | [+0.76, +1.72] | positive |

  **R-14 is confirmed for the direct target at D+1 and D+3, not at D+2**, and it stays a
  declared limitation (no trend feature, as registered). By month the direct bias is largest in
  April and May (+2.5 % and +1.4 % at D+1) and about 0 in August and September. The indirect
  route over-forecasts more and at every horizon. Part of both biases comes from the 146
  op-limited hours, which are in the objective but not in the fit rows (curtailment cannot be
  forecast); the per-view bias is not computed here.
- **A-14 (§14.6):** the `xgb` step fits no scaler and no imputer; the boosters see train rows
  only, and validation enters only as the early-stopping set and the objective. T-61 shows that
  permuted validation labels change only the stopping round, never the trees, and do not change
  the L-09 coefficients. **A-14 is ✅ for XGB**; the scalers of the sequence models are checked in
  W4.
- **Review (forecasting-validity-review):** the rules and margins are the registered ones
  (§14.4, §14.5); three seeds for the decision; train folds chronological with a 3-day gap; the
  correction is fitted on train only and validated once; the test split is not read.

### 2026-11-07 · R9-08 · The adopted L-09 correction in the `xgb` step: review and code

- **Design note before the code:** `docs/stage9_plan.md` §15. No rule of §14 changes: the
  correction is the registered one, and the base model is tuned and stopped as before.
- **Self-review (X14–X16, errata rows):**
  - X14: the correction existed only in the L-09 check; a second copy in the step could drift.
    One implementation now lives in `models/xgb_sota.py` and serves both.
  - X15: the indirect file holds k and the correction is in kW. Daytime hours are written as
    k' = corrected kW / theo, nights keep the raw k. Checked on the data before the design: every
    daytime hour with AC truth has theo of at least about 62 kW in train and in validation, so
    the division is defined and `evaluate.to_kw` returns the corrected kW.
  - X16: two readers took the prediction file for the raw booster output (the consistency check
    of `run_l09` and T-59). The step now also writes `uncorrected_xgb_sota_seed<S>.csv`, which
    `evaluate` does not score, and both readers use it.
- **A choice the plan did not settle, made here:** L-09 is decided on a dev run, so the step
  cannot require the decision in every mode (that would make the first dev run impossible). A
  smoke or dev run without `configs/mos_decision.yaml` writes the raw forecast with a warning and
  `check` reports the decision as pending; a final run refuses.
- **Records:** `value_kw` keeps its meaning (the base model's objective, which the refit must
  equal); `value_kw_forecast` is the value of the written forecast and equals the evaluation. The
  seed spread of stage 10 uses the written forecast.
- **Tests:** T-64 (four new tests), and T-58, T-59 and the L-09 test of T-63 adapted to the
  corrected file. On a synthetic dev run, the step's seed-42 coefficients equal the check's, and
  the evaluation's `nRMSE_kW_pct` of the indirect route equals the check's corrected value.

### 2026-11-07 · R9-09 · Smoke and dev runs with the L-09 correction, and their review

- **Commands and commit:** at `9c5d1b9`, clean tree; `check --mode dev` passes (20 of 20
  inputs as recorded, the L-09 decision among them).
  - `python -m solarpv.pipeline run --mode smoke --only baselines,xgb,evaluate`: run
    `20261107T132101Z_smoke`, 25 s, ok. The base models are those of R9-05 (equal parameters,
    rounds and uncorrected values); the indirect route is corrected at every horizon, the direct
    route is not.
  - `python -m solarpv.pipeline run --mode dev --only baselines,xgb,evaluate`: run
    `20261107T132142Z_dev`, 6.5 min (the `xgb` step 380 s against 363 s in R9-06: the 15
    out-of-fold fits cost about 17 s).
- **Verification against R9-06 and R9-07 (all exact):**
  - the study reproduces R9-06: the same best parameters and rounds, uncorrected values equal
    (difference 0) at every target × horizon;
  - the step's seed-42 coefficients, out-of-fold hours (7,834) and corrected values equal the
    seed-42 rows of `outputs/09_xgb_checks/l09_check.csv` (difference 0);
  - the evaluation's `nRMSE_kW_pct` equals `value_kw_forecast` to the table's six decimals;
  - `uncorrected_xgb_sota_seed42.csv` equals the R9-06 forecast of the indirect route, the direct
    forecast is unchanged, night hours are unchanged, and `evaluate.to_kw` of the written k
    returns the corrected kW to 2.3e-13 kW;
  - the warnings are the five of R9-06 (the booster is the same).
- **Result (validation, the tuning split, one seed; nRMSE in kW, % of 3,449 kW):**

  | Target | Horizon | R9-06 | With L-09 | Climatology | Persistence |
  |---|---|---|---|---|---|
  | indirect | D+1 | 6.792 | 6.686 | 8.06 | 9.68 |
  | indirect | D+2 | 7.122 | 6.911 | 8.06 | 10.84 |
  | indirect | D+3 | 6.656 | 6.555 | 8.06 | 10.76 |
  | direct | D+1 | 6.656 | 6.656 | 8.04 | 9.68 |
  | direct | D+2 | 6.811 | 6.811 | 8.04 | 10.82 |
  | direct | D+3 | 6.426 | 6.426 | 8.04 | 10.72 |

  The acceptance of §14.7 still holds. In the paired route comparison the direct route stays
  ahead, by 0.03, 0.10 and 0.13 pp (0.14, 0.31 and 0.23 in R9-06).
- **Review (forecasting-validity-review):**
  - issuance and split isolation: inputs unchanged; the coefficients come from out-of-fold train
    predictions only; no test row was read;
  - comparable evaluation: the same 2,901 daytime hours for every model and both routes;
  - uncertainty: one seed of a dev run on the tuning split (optimistic by construction, D-10);
    L-09 itself was decided with three seeds (R9-07);
  - red flags: D+3 below D+1 remains for the corrected indirect route too (6.555 against
    6.686), the validation-sample pattern of R9-06; no dominant-feature change (same boosters).
- **Finding of the review: the k-unit statistics of the indirect route get worse.** The
  correction is fitted and decided in kW. Its intercept (+14.6, +18.7, +28.1 kW for seed 42)
  divided by the small theo of the low-sun hours becomes a large shift in k: the largest daytime
  k rises from 1.338 to 1.645 (the largest true k in validation is 1.523). The evaluation layer
  computes `nRMSE_pct`, `skill_score` and the DM tests of the indirect route in k units (§6.9 of
  `PROJECT_MASTER.md`, D-14):

  | Horizon | nRMSE in k, R9-06 → L-09 | skill in k, R9-06 → L-09 | DM vs climatology, HLN p, R9-06 → L-09 |
  |---|---|---|---|
  | D+1 | 13.44 → 13.86 | 0.331 → 0.310 | 0.000009 → 0.000087 |
  | D+2 | 14.07 → 14.52 | 0.369 → 0.349 | 0.000039 → 0.000512 |
  | D+3 | 13.51 → 15.28 | 0.400 → 0.321 | 0.000033 → 0.035 (not significant at the Bonferroni α of 0.0167) |

  In kW the same forecast is better (the table above; the kW RMSE against persistence improves
  at every horizon in `route_comparison.csv`). So the corrected indirect route is better in the
  unit of every decision (§14.1) and worse in the k unit, where every daytime hour weighs the
  same and low-sun hours dominate (L-04). The stage-10 acceptance "XGB clearly better than the
  baselines (DM with HLN)" reads the DM table, which is in k for this route. **Nothing was
  changed:** which unit the DM and skill of the indirect route use is a decision for the user
  (Q-19). The kW-unit DM was not computed, so that the choice is not made on its result.

### 2026-11-07 · R9-10 · Q-19 decided; the kW statistics: review and code

- **Decision (user, 2026-11-07):** Q-19 option B. DM and skill are computed also in kW for both
  routes; the target-unit tables stay as secondary ones; the kW DM table is the one the stage-10
  acceptance reads. Recorded before the code as a dated amendment of D-14, §6.9 and the stage-10
  acceptance in `PROJECT_MASTER.md`, and in `docs/stage9_plan.md` §16. No kW DM had been computed
  when the decision was taken.
- **Self-review before the code (X17–X18, errata rows):**
  - X17: the stage-8B validation command still read the test rows of the truth (X3 was fixed
    only in the runner). It now reads the truth without the test split.
  - X18: one conversion of the scored rows (`evaluate.in_kw`, through `to_kw`) feeds the same DM
    and skill code, so no fifth k → kW copy and no second DM loop.
- **Code:** `dm_tests_kw.csv` (the columns of `dm_tests.csv` and `unit`), `skill_score_kw` and
  `skill_cliper_kw` in `benchmark_summary.csv`; only with a truth table. No existing column or
  table changes. Notebook `07_evaluation.ipynb` gets a cell on the kW DM.
- **Tests:** T-65 (two new tests, and the `main` test checks the truth is read without the test
  split). T-55 now also compares `dm_tests_kw.csv`; it needs the regenerated stage-8B tables, so
  it passes only after the rerun of R9-11.

### 2026-11-07 · R9-11 · The stage-8B tables regenerated, and a dev run with the kW statistics

- **Commands and commit:** at `f529913`, clean tree.
  - `python -m solarpv.evaluation.evaluate` (validation, references only; now without reading a
    test row): log `outputs/07_evaluation/runs/20261107T135905Z.log`. Only `benchmark_summary.csv`
    changed, by the two new columns; every existing column, and every other table, is
    byte-identical to the committed version, so excluding the test rows changes nothing (X17).
    New: `dm_tests_kw.csv`. For the direct target the kW skill and DM equal the existing ones
    exactly.
  - `python scripts/make_evaluation_notebook.py`: notebook 07 with a cell on the kW DM.
  - `python -m solarpv.pipeline run --mode dev --only baselines,xgb,evaluate`: run
    `20261107T135938Z_dev`, ok; every existing table equals that of `20261107T132142Z_dev`
    (R9-09).
- **The kW DM of the dev run (reference XGB, HLN, Bonferroni α 0.0167):** XGB is significantly
  better than all three references in all 18 comparisons, both routes and every horizon. Against
  climatology the HLN p-values are 0.0008, 0.0013 and 0.0002 (direct) and 0.0013, 0.0024 and
  0.0007 (indirect). In the k-unit table the indirect D+3 comparison with climatology was not
  significant (p 0.035, R9-09); in kW it is.
- **Skill in kW (against persistence of h days):** indirect 0.310, 0.363, 0.391; direct 0.312,
  0.371, 0.401 (in k the indirect route had 0.310, 0.349, 0.321).
- **Review (forecasting-validity-review):**
  - the kW statistics use the same rows (2,901 daytime hours) and the same pairs as the k ones;
    the truth is actual AC from the stage-6 table;
  - one seed of a dev run on the tuning split: these are tuning-split numbers, optimistic by
    construction (D-10); the acceptance of stage 10 is read on the final run;
  - against climatology the effect is about the minimum detectable effect (`pct_of_mde` 93–110 %
    in both routes), so "clearly better than climatology" holds, but with a modest margin;
  - the skill rises with the horizon because persistence of h days gets worse faster than XGB;
    the errors themselves were reviewed in R9-06 and R9-09.
- **Not in scope (stage-9 plan §16.2):** the seasonal and common-mask tables stay in the
  target's unit.

### 2026-11-07 · R9-12 · W3: self-review, the user's decisions, and the code of the sequence datasets

- **Baseline at the start:** HEAD `0899b39`, clean tree; 438 tests passed, none skipped; ruff
  clean.
- **Plan before the code (`docs/stage9_plan.md` §17, commit `748959f`):** what the original
  sequence models did, reconstructed from the submission (E1–E7). Three facts were computed
  here, read-only:
  - the surviving BiLSTM file has 5,689 rows per target × horizon, hourly, no duplicate, D+1 /
    D+2 / D+3 starting 95 / 119 / 143 h after the test start: an origin every hour, a 72-hour
    encoder inside the test, one decoder step kept per horizon (23 / 47 / 71);
  - the TFT n (2,554 / 2,556 / 2,558) is exactly 2,580 minus the daytime hours of May 30–31,
    Oct 1 and May 31, Oct 1–2 on the original labels: daily origins, whole decoders inside the
    test;
  - the PatchTST n (2,580 / 2,569 / 2,558) is 2,580 minus nothing, Oct 1, Oct 1–2.
- **Self-review (errata rows):** in the original M13–M18 (C-47–C-52): the naive local clock,
  horizons by step number and the hourly fixed-lead BiLSTM, a different set of scored hours per
  model, silent fallbacks and dropped rows in the intermediate LSTM, the split by proportions
  and the MinMax on k, the valid time attached after training. In my own code X19–X23: the
  docstrings with the original formula and the test-only origin count, the inverse of
  `issuance_utc`, a loader by split label, the D-22 masks that exist only inside
  `masked_sources`, and the loss mask of XGB.
- **Decisions (user, 2026-11-07), before the code:** D-25 (A), the decoder covers the three
  local days (71–73 steps); D-27 (A), the GFS of the stage-7 selection; D-28 (A), the BiLSTM on
  the known-at-T0 part of the stage-7 input, encoder 72 h. On D-26 the user rejected both options:
  the known reals should come from the feature selection. Checked: the calendar, clear-sky and
  geometry names of the six submitted 05e sets add up to 17 (one duplicate), close to v6's 18. The
  new option, per target the known features of the stage-7 selection (10 indirect, 27 direct),
  was chosen.
- **Code:** `solarpv.models.sequence_data` (origins, the step → horizon mapping, a time-range
  loader, the encoder series through `masked_sources`, the known inputs, the scaler, `build`, the
  writer `predictions_frame`, the checks and the command); docstrings of `lstm_sota.py` and
  `patchtst_model.py` (X19). OP-55.
- **Tests:** T-66–T-71 in `tests/unit/test_sequence_data.py` (32 tests, synthetic, both DST
  days); T-02 in `tests/contracts/test_sequence_alignment.py` (16 cases on the real tables).
  On the first unit run one test failed for a reason in the test, not the code: an in-memory
  build was given the synthetic test rows, and the development guard refused them. The tests now
  cut the data before TEST_START, as the loader does, and a test asserts the refusal. The real-data
  test is run only after the commit.

### 2026-11-07 · R9-13 · The sequence datasets on train and validation, checked against theory

- **Commands and commits:** code committed at `69ee067` before any run on real data.
  - `pytest tests/contracts/test_sequence_alignment.py` (T-02): 16 of 16 pass.
  - `python -m solarpv.models.sequence_data --mode dev`: all checks pass, 0 warnings, 25.6 s.
    Finding X24: the run log went to `outputs/09_sequence_data/runs/runs/` (`RunLog` adds `runs`
    itself). Fixed at `854b0f7` with a test; the log was moved to `runs/`.
  - Rerun at `854b0f7`: `--mode dev` (log `runs/20261107T175754Z.log`) and `--mode smoke` (log
    `runs/20261107T175824Z.log`), all checks pass. `checks_dev.csv` is byte-identical to the
    first run.
- **Loaded:** 27,027 hours, the last at 2024-09-30 23:00Z; every file read with
  `timestamp_utc < TEST_START` inside the reader.
- **Against theory (dev, all 20 datasets: 5 models × 2 targets × train and validation):**
  - origins: train 872 (first T0 2021-09-12 21:00Z = 2021-09-13 00:00 local, last 2024-01-31
    22:00Z = 2024-02-01 00:00 local); validation 246 (2024-01-29 22:00Z to 2024-09-30 21:00Z);
  - three-day decoders: train 857 of 72 steps, 6 of 71 (the spring days 2022-03-25 and
    2023-03-24, three origins each) and 9 of 73 (the fall days 2021-10-31, 2022-10-30,
    2023-10-29); validation 243 of 72 and 3 of 71 (2024-03-29);
  - every D+h block that ends before the limit equals its local day: train 6 blocks of 23
    hours, 2,595 of 24, 9 of 25; validation 3 of 23 and 729 of 24 (the six blocks of each scope
    that cross its limit are not complete days);
  - every hour of the scope exactly once per horizon: 20,859 train hours (train after the
    14-day warm-up) and 5,832 validation hours;
  - `issuance_utc(valid, h) == T0` on every step; every encoder ends at T0; no time at or after
    TEST_START; training decoders end before VAL_START;
  - loss rows per horizon equal the XGB-dev rows exactly: fit 9,575 (indirect) and 20,393
    (direct), and on validation 2,825 and 5,685, the early-stopping rows of `xgb_config.json`
    (`20261107T135938Z_dev`);
  - channels: indirect encoder 46 (8 series, 8 fill flags, 10 + 20 known), decoder 30; direct
    63 and 49; PatchTST 1 and 0;
  - the encoder series equal the stage-6 `lag_1d` on every hour (0 mismatches in all seven
    series); masked power hours before TEST_START 343 / 405 / 542 (inverter 1, inverter 2, site
    total), and 131 / 143 / 191 in test, which add up to the stage-6 manifest;
  - filled encoder hours: train 492 for the site total and k (op-limited or missing AC), 154–378
    for the other series; validation 147 and 1–124;
  - scaler: 20,859 train hours; recomputed independently, and unchanged when every value outside
    train is replaced by noise;
  - known inputs missing (set to the train mean): `airmass` at night (pvlib leaves it undefined
    below the horizon; direct route only, 9,664 train hours), and
    `irradiance_clear_index_prevday` on 24 train hours; none in validation for the indirect
    route.
- **Smoke:** 58 training origins (2023-12-06 to 2024-02-01, 1,322 hours) and 30 validation
  origins (2024-03-13 to 2024-04-11, the 671 hours of the W1 window, 3 of them with 71 steps).
- **Review (forecasting-validity-review):**
  - issuance: every encoder hour ends by T0; decoder inputs are deterministic or GFS of a run with
    init + 6 h ≤ T0, read from the table of the step's own horizon whose `t0_utc` is the origin's
    T0 (asserted on every step); no lag, rolling statistic or decoder-hour measurement; the fill
    uses older hours only; the inputs pass the allow-list and `assert_no_leakage`;
  - split isolation: nothing at or after TEST_START is read; the scaler sees train hours only;
    training windows hold train labels only; validation labels appear only in validation
    windows, outside every training loss; nothing in W3 was decided on validation;
  - comparability: every model gets every hour of the split at every horizon, the hours of XGB
    and the references; the loss rows equal those of XGB; the truth comes from the stage-6 table;
  - no model result yet, so no red flag can show; two points carried to W4: the night value of
    `airmass` (the train mean) is an input convention, visible to the network through the
    clear-sky and sun-elevation channels; and the library normalisers of W4 must act per window
    or be fitted on the training windows only (§17.5).

### 2026-11-11 · R9-14 · W4: self-review, registered settings, and the two proposals for the user

- **Baseline at the start:** HEAD `f5e182d`, clean tree. No W4 code, no GPU environment, no
  neural result.
- **Read first:** `PROJECT_MASTER.md` §1 and §16b, stage-9 plan §4.3, §5, §7 and §17, R9-12,
  R9-13, the college-machine note (P40, `sm_61`, driver 538.46, CUDA ≤ 12.2). Read-only: v6 §4.1,
  §5.2–§5.6, §11.2–§11.5 and §12.1 (text extracted to the session scratchpad), the GPU line of
  `reproduce_paper_results.py` (`torch==2.5.1+cu121`), the Supplement in `old files/`, and
  `07_LSTM_Attention_After_Training.ipynb` (a Keras intermediate).
- **Package metadata (2026-11-11, JSON and index pages only, no package file; saved in the
  session scratchpad and parsed with `python -I`):** pytorch-forecasting 1.7.0 needs pandas <
  3.1.0, lightning < 2.7, scikit-base < 0.14; neuralforecast 3.1.7 needs torch ≥ 2.4 and
  ray[train,tune]; the last Windows / Python 3.11 PyTorch builds are 2.5.1 for cu121, 2.7.1 for
  cu118, and 2.9.1–2.14.1 for cu126; the cu121 2.5.1 wheel is 2,449,385,544 bytes (HEAD request).
- **Self-review (stage-9 plan §18.2–§18.3), errata rows written with the plan:** in the original
  M19–M24 (C-10, C-53–C-57): BiLSTM seeds scored by training metrics, two objectives for the
  BiLSTM, the TFT baseline from an earlier code version, a −168 h attention peak in a 96-hour
  encoder, the TFT hidden size at the edge of its space, and unexplained neural biases (NMBE up
  to +6.5 % and −6.0 % on indirect D+1). How the libraries could bring back M13–M18 and C-41,
  with a guard for each. In my own code X25–X33: a second environment with another pandas, the
  pvlib import behind `evaluate.to_kw`, undecided neural budgets in `BUDGETS`, step and model
  names that differ, the library normalisers, the night `airmass` fill, the loss mask in the
  libraries, GPU determinism, and the unset PatchTST input size.
- **Registered before any result (§18.4):** one objective for every model (§14.1, M20), the
  losses, early stopping, the BiLSTM architecture of D-28, the TFT search space (v6 values where
  given, otherwise the library's own tuning ranges and the BiLSTM ranges of v6), the TFT baseline,
  the PatchTST input size (168 h), the seeds and the device. The acceptance of the dev run (§18.8).
- **Proposed to the user, nothing installed (§18.5, §18.6):** D-29, a separate `solar-gpu`
  environment for a training worker only, torch 2.5.1+cu121, pytorch-forecasting 1.7.0,
  neuralforecast 3.1.7, lightning 2.6.6, pandas 3.0.6, about 2.7 GB of downloads into a new folder
  outside the repository, installed from there with hashes; and the neural dev budgets (one third
  of the v6 trials with the full epoch budgets, one seed).
- **Decisions (user, 2026-11-11):** §18.5 (A), §18.6 (A), and the settings of §18.4 as
  registered. The environment was built the same day (RH-06).

### 2026-11-11 · R9-15 · W4: the code of the neural steps and its tests

- **Commits:** `d089a22` (units module X26, environment report, job format, training loop,
  BiLSTM and TFT modules), `c083344` (worker, PatchTST adapter, the five runner steps, budgets
  X27, names X28, PatchTST input size X33, tests), `6a1ffdb` (fix X35, below).
- **Design as built:** the runner side (`pipeline/neural.py`, env `solar`) builds the W3
  datasets, checks each with `check_dataset`, guards the missing known inputs (X30), and writes
  the job (compressed arrays with SHA-256, no timestamp). The worker (`models/neural_worker.py`,
  env `solar-gpu`) checks the hashes, trains, and writes forecasts, fits, curves, trials, models
  and `worker.json` with every output's hash. The TFT is pytorch-forecasting's network fed the
  W3 windows directly (no `TimeSeriesDataSet`); PatchTST is neuralforecast's model with a
  callback that restores the best validation check (neuralforecast keeps the last weights).
- **Finding X34 (first GPU tests):** the TFT stopped under `torch.use_deterministic_algorithms`:
  `upsample_linear1d_backward_out_cuda` has no deterministic kernel (pytorch-forecasting's
  `TimeDistributedInterpolation`). Replaced inside the worker by the same linear map as a matrix
  product; equal values and gradients to 1e-12. Errata X34.
- **Second finding during the tests:** `NeuralForecast.fit` trains a copy of the model, so the
  callback object passed in is not the one that ran; the adapter takes the callback from the
  fitted model (a test asserts two validation checks recorded). Not an errata row: no result
  depended on it.
- **Tests:** `tests/gpu/test_neural_gpu.py` 22 passed in `solar-gpu` (the masked loss, masked
  targets that leave the network unchanged, the same seed giving the same network for the BiLSTM
  and the TFT, the restore of the best epoch, the TFT's inputs and identity output, no read of the
  unknown reals on decoder steps, the forecast of an origin independent of its batch with 71- and
  72-step decoders, the attention over the own encoder, a causal BiLSTM decoder, PatchTST's
  origin mapping and weights independent of the validation values, the worker end to end and its
  refusal of a changed array). `tests/unit/test_neural_runner.py` 16 passed in `solar` with a stub
  worker; `tests/unit/test_neural_gpu_suite.py` runs the GPU tests from the `solar` suite.

### 2026-11-11 · R9-16 · Smoke runs of the neural steps on real data

- **First smoke run** (`20261111T155030Z_smoke`, commit `c083344`): baselines, PatchTST and
  evaluate passed; BiLSTM and the three TFT steps stopped before training on the X30 guard:
  `fcst_gfs_kt_poa_ma3` missing on 1 hour at every horizon. Diagnosis: a centred 3-hour GFS mean,
  undefined at the first hour of the data (2021-08-31 21:00Z, a warm-up night hour; the first
  encoder hour of any window is 2021-09-05 21:00Z) and at the last (2025-05-31 20:00Z, in test).
  The guard had checked every loaded hour. Finding X35, fixed at `6a1ffdb` with a test: the guard
  checks the hours that the windows read, and a centred GFS mean may be missing only at the two
  edge hours of the data at night.
- **Second smoke run** (`20261111T155739Z_smoke`, commit `6a1ffdb`): all seven steps passed. The
  evaluation scored every model on the 321 daytime hours of each horizon in the smoke window, with
  its strict alignment and coverage checks. Per model and target, the step's objective equals the
  evaluation's `nRMSE_kW_pct` to at most 4.8e-7 (the table's precision). The only missing known
  inputs are the 24 train hours of `irradiance_clear_index_prevday` and the night `airmass` (766
  train and 377 validation hours read). PatchTST: `EXOGENOUS_FUTR = False` (L-19, for the report).
  The warnings are those of 2 epochs: best epoch at the last epoch everywhere, and in the one-trial
  TFT studies `lstm_layers = 3` on the edge. The BiLSTM indirect forecast is constant at k ≈
  −0.06 after 2 epochs on 58 origins (0 kW after the conversion), so its three horizons score the
  same: an untrained network, not a mapping error (the mapping is tested by T-77).
- **Determinism (X32):** a third smoke run (`20261111T160241Z_smoke`) of BiLSTM, TFT-96, the TFT
  baseline and PatchTST gave byte-identical prediction and quantile files; the learning curves
  differ only in the `seconds` column.
- **Timing and the dev estimate:** median seconds per epoch in smoke (58 training and 30
  validation origins): BiLSTM 0.04, TFT 0.61; PatchTST about 21 s per target with imports. Scaled
  to 872 and 246 origins: about 4 h expected for the whole dev run, about 8 h if every TFT fit ran
  its full 100 epochs; below the 8-hour line of §18.6, so the dev run was started without asking
  again.

### 2026-11-11 · R9-17 · The dev run of the neural models, and its review

- **Command:** `python -m solarpv.pipeline run --mode dev --only baselines,bilstm,tft96,tft168,
  tftbase,patchtst,evaluate --keep-going`, run `20261111T160714Z_dev`, budgets of D-17 (§18.6),
  seed 42. All seven steps ok, 11,569 s (3.2 h): BiLSTM 572 s, TFT-96 4,327 s, TFT-168 5,181 s,
  TFT baseline 674 s, PatchTST 786 s, evaluate 19 s. The steps recorded commits `6a1ffdb`,
  `04764aa` and `c9a75bc` (documents were committed during the run); `git diff` of the code paths
  between them is empty, and none is `+dirty`.
- **Validation nRMSE in kW (% of 3,449 kW), D+1 / D+2 / D+3, from the run's
  `evaluate/val/benchmark_summary.csv` (2,901 daytime hours per horizon for every model):**
  - indirect: BiLSTM 6.97 / 7.17 / 6.87; TFT-96 6.78 / 6.96 / 6.68; TFT-168 6.77 / 7.05 / 6.66;
    TFT baseline 6.83 / 7.10 / 6.81; PatchTST 8.77 / 8.90 / 8.86;
  - direct: BiLSTM 6.75 / 6.95 / 6.51; TFT-96 6.75 / 6.94 / 6.57; TFT-168 6.79 / 7.01 / 6.65;
    TFT baseline 6.87 / 7.04 / 6.71; PatchTST 8.09 / 8.27 / 8.29;
  - references on the same hours: persistence 9.68 / 10.84 / 10.76 (indirect), climatology 8.06
    (indirect) and 8.04 (direct); XGB-dev (`20261107T135938Z_dev`) 6.69 / 6.91 / 6.55 (indirect)
    and 6.66 / 6.81 / 6.43 (direct).
- **Review (forecasting-validity-review):**
  - issuance: the W3 windows, `check_dataset` passed on every dataset of every step; the missing
    known inputs are only the 24 train hours of `irradiance_clear_index_prevday` and the night
    `airmass` (X30, X35); no timestamp and no test row reached the worker;
  - split isolation: nothing at or after TEST_START was loaded (asserted in the step); validation
    entered only as the early-stopping set and the objective; the scaler is the W3 one;
  - comparability: every model, the references and XGB-dev on the same 2,901 daytime hours per
    horizon; the references equal those of the XGB-dev run exactly (difference 0.0); per model and
    target the step's objective equals the evaluation's `nRMSE_kW_pct` within 4.7e-7;
  - uncertainty: one seed (D-17), so no comparison between families is decided here; in kW every
    neural model is significantly better than persistence at every target and horizon (DM with HLN
    and Bonferroni; the DM reference is persistence because XGB is not in this run);
  - red flags: D+3 below D+1 for every neural model, the same order as XGB-dev on these hours
    (D+2 the hardest): the validation sample, as in R9-06; the mapping is tested (T-77, T-02, X20).
    No implausible margin: the neural models are within 0.0–0.3 points of XGB-dev, as C1 expects;
  - provenance: run logs, `step.json` with hashes, worker records with hashes, one code version.
- **Acceptance (§18.8):** required, all met (every step completed, every file passed the strict
  alignment and coverage checks, no NaN loss, the warnings read, MBE reported). Expected, met:
  every neural model below persistence at every target × horizon (30 of 30). The error falls from
  D+2 to D+3, explained as above. PatchTST is worse than climatology in both routes, the expected
  weakness of a univariate model (L-19, C1), not a defect. Nothing was retuned on validation.
- **Points for the list of G-1 (no change made):**
  - the BiLSTM indirect final fit (best epoch 51 of 60 by the validation quantile loss) scores
    7.00 %, its best 25-epoch trial 6.84 %: the early-stopping loss and the kW objective disagree
    by 0.16 points (the registered rule of §18.4);
  - best values on the edge of the search space (M23) in 4 of 6 studies: BiLSTM indirect (hidden
    256, 1 layer, batch 128), TFT-96 both targets (hidden size 64, 4 heads), TFT-168 direct (4
    heads); v6 reported the bottom edge (hidden 16), the rebuild hits the top;
  - with 7 trials the TFT studies stay inside Optuna's 10 random start-up trials, which depend on
    the seed only: both targets and both encoder lengths tried the same 7 candidates (the final
    run's 20 trials go past it);
  - PatchTST indirect had its best validation check at step 4,900 of 5,000 (direct 4,100): near
    the end of the default step budget;
  - mean bias on validation +0.37 to +2.73 % of capacity, positive for every model (XGB-dev R-14:
    +0.74 % direct, +1.31 % indirect before L-09); the point forecasts are medians (quantile loss,
    MAE), above the mean for a left-skewed k; PatchTST indirect is biased already on its training
    windows (+2.93 %).
- **Determinism on real data:** each retrained best TFT trial reproduced its trial exactly
  (difference 0.0 pp); the BiLSTM direct final fit equals its best trial.
- **Figures:** S09-07 to S09-12 built at `a0ad8ec`; the attention takeaway was rewritten after a
  look at the figure: TFT-168's indirect maximum is at the first encoder position, an edge of the
  window, not a lag (M22), and the daily rhythm is stated instead.

### 2026-11-12 · R9-18 · Review of the W4 figures, and three correction figures

- **Request (user):** finish every figure of the W4 session and keep them as sources for the
  final report.
- **Check:** the six process figures S09-07 to S09-12 were built, but they were missing from the
  table of `docs/figures_plan.md` §5, and no correction figure drew a W4 difference from the
  submission (C-10, C-35, C-54, C-55 were text only in `docs/defense_notes.md`).
- **Findings:**
  - X36: the description of S09-09 said "5,832 validation hours per horizon", a typed count of
    all validation hours, day and night; the scored daytime hours are 2,901. S09-07's description
    also typed window counts (correct, but typed). The count is now a fact of S09-09, asserted to
    be shared by every neural model, XGB-dev and the references; a test forbids typed counts in
    descriptions;
  - X37: C-10 said the seed-123 and seed-777 rows of the submitted BiLSTM seed table were
    training-script metrics, but the seed-42 row (19.554 % indirect D+1) is also the training
    output (`metrics_lstm_sota.csv`), against 30.237 % scored by the evaluation. C-10, M19 and the
    defense notes corrected; C-07 asserts both facts.
- **New correction figures (built from the dev runs and the submitted files; no test row of the
  rebuild read):**
  - C-07, the seed table: at D+1 the three rows average 19.7 % (indirect) and 8.5 % (direct),
    the scored seed-42 forecast 30.2 % and 12.6 %; the reported 2.2 % relative spread rests on
    training-script numbers;
  - C-08, scored hours: submitted models on 2,510 to 2,580 hours at D+1, the rebuild's six models
    and three references on the same 2,901;
  - C-09, attention window: TFT-96 has no position at -168 h; in all four model-route pairs the
    first encoder position is 8 % to 28 % above the same hour of the later days, which agree to
    within 0.40 %: an edge of the window, whatever its length.
- **Not drawn:** C-57 (the Supplement's indirect NMBE is normalised by a mean k of 0.6279, not by
  the capacity of the rebuild's bias figure), C-53 and C-56 (no data beyond the report's text).
- **Records:** errata X36, X37 and C-10; `docs/defense_notes.md` §7 (the neural models, with the
  three figures); `docs/figures_plan.md` §5; T-72 extended in `tests/contracts/test_figures.py`;
  catalog and gallery regenerated (46 figures: 37 process, 9 correction).

### 2026-11-12 · R9-19 · W5: self-review before the code, and four proposals for the user

- **Read first:** `PROJECT_MASTER.md` §1 (status, the G-1 list) and §16b (L-14, L-15, L-20), the
  stage-9 plan (§3.2–§3.6, §7, §9, X8, §15.2, §18.8), R7-04, R7-05, R9-06, R9-11, R9-17, and the
  college-machine note (two environments).
- **Sources of the original's explanation stage** (read-only): v6 §11.1–§11.5 and C6 (the docx
  read as text), `07_shap/*`, `12_scripts/shap_rank_stability.py`, `reproduce_paper_results.py`
  (phases 1–9: no selection step), the claim matrix rows 70–77, 100 and 101.
- **The original (M25–M31, errata C-58–C-64):** selection outside the reproduction; SHAP of the
  76-feature model led by leaky rolling groups; group shares by first-match substrings and sums
  of mean |SHAP|; no record of the rows, unit or explainer, and zero-filled rank correlations;
  attention offered as validation of SHAP (with a −168 h peak in TFT-96 and a 7-hour smoothing);
  one unseeded fANOVA on 20 trials; report numbers without a file.
- **My own code (X38–X49, errata):** the final gate only in the preflight (commits were made
  during R9-17); final slices allowed; model steps' test reads unlogged; test scored next to the
  stage-10 acceptance; the seed spread in two units; no final time estimate; `select` described
  as a rerun; the sampled against the effective `hidden_continuous_size`; **the saved XGB booster
  holds 50 more rounds than the forecast uses** (an explainer on it would explain another model);
  the L-09 map not additive in k; the TFT weights split a series and its fill flag; warnings as
  free text only.
- **Facts checked for the proposals:** the selection code changed since `190e4df` only by
  `decide_input` and the validation-only command (before the look); the neural dev run folder
  is 192 MB (125 MB of arrays, 29 MB of checkpoints; studies and models are ignored by git); the
  XGB dev run `20261107T135938Z_dev` cannot be resumed (created with the pre-W4 budgets, X27);
  SHAP 0.51.0, XGBoost 3.4.2 and Optuna 5.0.0 in `solar`.
- **Proposals for the user** (stage-9 plan §19.4–§19.8): Q-20 (`select`: verify and reuse, or a
  rerun of stage 7), Q-21 (the scope of `explain` and `archive`), Q-22 (the test protocol of a
  final run), Q-23 (the runbook: about 9 h expected, up to about 16 h; the questions to the
  college IT). Nothing is coded before the answers.
- **The user's answers (2026-11-12):** every recommended option. D-30: `select` verifies and
  reuses. D-31: the full `explain` scope with `archive`. D-32: two phases of one final run. Q-23:
  the two questions go to the college IT before W6.

### 2026-11-12 · R9-20 · W5: the code, its tests, and the smoke runs of the whole sequence

- **Code** (`ebc2187`, then `46881ae`): `pipeline/select_step.py`, `explain.py`, `report.py`,
  `archive.py`, `promote.py`, the step `test`, the gate before every final step, the final
  scope, the test-access lines, the effective `hidden_continuous_size`, the final estimate.
  Tests T-82–T-91 in `tests/unit/test_w5_steps.py`.
- **Found while testing and writing, before any real run:**
  - X50, the first synthetic run of `explain`: an absolute additivity tolerance failed the
    direct boosters on float32 precision (0.003 kW on outputs up to 3,449 kW); the tolerances
    are now relative to the output's scale;
  - X51, while writing the runbook: a commit recording the stage-10 acceptance would move
    `HEAD` off the tag and the gate would refuse the test phase; the acceptance goes to an
    untracked file in the run folder, which the step `test` requires.
- **First smoke** (`20261112T060949Z_smoke`, all 11 steps ok, 4.5 min): seven steps recorded
  `ebc2187+dirty` because X51 was edited during the run. Not used as evidence; rerun.
- **Smoke at a clean commit** (`20261112T062618Z_smoke`, `46881ae`, all 11 steps ok, 240 s):
  - `select`: the six inputs equal the registered rule on the committed checks (31 / 28 / 32
    05e features, 96 / 89 / 82 full pool), `model_inputs.json` `b693622407f2`, no row read;
  - `explain`: SHAP of 6 boosters on 321 hours each; additivity at most 8.2e-7 of the output
    scale, XGBoost's `pred_contribs` identical (difference 0.0), the kW view at most 0.003 kW;
    **the unsliced boosters would be off by up to 104 kW** (direct D+2: 38 forecast rounds, 88
    saved), which confirms X46 on real data; the TFT attention and 12 studies summarised (one
    trial each, all inside TPE's start-up, as expected in smoke);
  - `report`: 26 warnings, all reproduced exactly from the records; 19 hashed sources; no
    consistency problem;
  - determinism: the 9 prediction files and the explanation tables are byte-identical to the
    first smoke run;
  - test split: no step read a test row, the access log is unchanged.

### 2026-11-12 · R9-21 · W5 dev runs of the new steps, their review, figures and notebook 06

- **Commands** (commit `46881ae`, clean tree):
  - `python -m solarpv.pipeline run --mode dev --only select,baselines,xgb,evaluate,explain,
    report`: run `20261112T063719Z_dev`, ok (xgb 322 s, explain 57 s); the XGB dev run of R9-11
    could not be resumed, because it was created with the pre-W4 budgets (X27);
  - `python -m solarpv.pipeline run --mode dev --resume 20261111T160714Z_dev --only select,
    explain,report`: ok (its configuration is unchanged since W4, so the resume was accepted).
- **Reproduction:** every prediction file, the uncorrected L-09 file and the benchmark and DM
  tables of the new run equal those of `20261107T135938Z_dev` byte for byte.
- **`select`:** both runs verified the same `model_inputs.json` (`b693622407f2`).
- **`explain`, XGB** (2,901 scored hours per horizon): additivity at most 1.4e-6 of the output
  scale, `pred_contribs` identical; the unsliced boosters would be off by up to 199 kW (X46).
  First feature: `fcst_gfs_k_phys` (indirect D+1), `fcst_gfs_kt_poa_daymean` (indirect D+2, D+3),
  `fcst_gfs_poa` (direct D+1, D+2), `fcst_gfs_pv_kw` (direct D+3), the same leaders as the
  stage-7 measure of R9-06; dominance at most 6.7 (direct D+1, threshold 10). Group attribution:
  GFS 56–79 %; the v6 measure would differ by up to 14.1 points (GFS, indirect D+3). Rank
  stability across horizons 0.84–0.90 (common features).
- **`explain`, TFT** (validation forecasts): the same-hour ratio is 0.95 (TFT-96), 0.98
  (TFT-168) and 0.82 / 1.07 (TFT baseline, indirect / direct), so no model prefers the hour it
  forecasts. The daily rhythm of the attention (S09-11) is a preference for fixed clock hours of
  earlier days: in the direct route 12:00 (TFT-96), 18:00 (TFT-168), 16:00 (TFT baseline). TFT-96
  indirect puts only 22 % of its attention on the encoder, the others 77–91 %. The
  variable-selection weights sum to 1 per part; leaders differ by model and route.
- **`explain`, Optuna:** XGB's most important parameter on average is `learning_rate` (0.31);
  4 of 6 neural studies lie inside TPE's random start-up (7 TFT trials).
- **`report`:** XGB 5 warnings, all matched (the five of R9-06); neural 5 warnings, 4 matched and
  one recomputed only: TFT-168 indirect used `hidden_continuous_size` 32, the top of its
  effective range, which the W4 step had judged by the sampled value (X45 confirmed on real
  data). 10 and 28 facts for the G-1 list.
- **Review (forecasting-validity-review):**
  - issuance: the explained boosters' features equal the allow-list of `model_inputs.json`
    (asserted); the TFT attention comes from the W3 windows (T-02);
  - split isolation: no test row read (the access log unchanged); validation is the
    early-stopping and tuning set (D-10), so the explanations describe the model on hours that
    tuned it, not on held-out data; the final run's step `test` explains the test window;
  - comparability: one set of 2,901 hours per horizon for every booster;
  - uncertainty: one seed; the fANOVA spread over ten evaluator seeds is recorded; nothing is
    decided from these numbers;
  - red flags: none (no dominant feature; no implausible margin; D+3 below D+1 is the known
    validation-sample pattern of R9-06);
  - provenance: run logs, `step.json` with hashes, one commit.
- **A pre-written claim refuted:** the draft of figure S09-16 asserted that every TFT prefers the
  same hour of earlier days; the assert would have failed on the data, and the figure was rewritten
  to state what the data show before it was built.
- **Figures:** S09-13–S09-19 and C-10 (`docs/defense_notes.md` §10); S09-07 rebuilt because its
  input `run.json` grew with the resume (same takeaway). Stage 9's minimum raised to 19 (T-72).
- **Notebook:** `notebooks/06_models.ipynb` (`scripts/make_models_notebook.py`), executed in a
  Jupyter kernel; every note formatted from the two runs and every interpretive claim asserted.
- **Decision / next:** W5 is complete. Next: W6 (G-1, the tag, and the final run by
  `docs/runbook_final.md`), after the college IT answers the two questions of Q-15.

### 2026-11-12 · R9-22 · Figures for every task, shown to the user (X52)

- **Request (user):** two sessions in a row ended a task without showing any figure; every task
  must end with figures that help write the process reports, everything documented, and all the
  W5 figures shown.
- **Finding X52:** the W4 and W5 figures had been built (S09-07–S09-19, C-07–C-10) but never
  shown, and nothing grouped them by task. A review of W5 against its figures found five outputs
  without a figure.
- **Change:** rule 7 of `docs/figures_plan.md` and the `CLAUDE.md` rule: a task's figures are
  listed in `WORK_ITEMS`, built by `python -m solarpv.figures task <key>`, which writes the
  sheet `docs/figure_sheets/<key>.md`, then looked at and shown in the app. The gallery is
  grouped by task; T-72 checks the tasks and the sheets.
- **New W5 figures** (from the two dev runs, validation only, nothing decided):
  - S09-20: the select step read no data row; 10 of 10 stage-7 files tracked and unmodified, 6
    of 6 tables those of the selection, inputs equal in both runs;
  - S09-21: in 5 of 6 boosters the saved model holds 50 rounds more than the forecast uses;
    explained whole, up to 199 kW (direct) and 0.11 in k (indirect) off the forecast; cut, within
    1.4e-6 of the scale (X46 on real data);
  - S09-22: first over third feature at most 6.7 (flag 10); Spearman across horizons 0.84–0.90;
  - S09-23: in the decoder the GFS forecasts hold 22–72 % of the TFT selection weight; the
    three TFTs disagree on their top direct-decoder input (`is_weekend` for TFT-96 with 0.42,
    `fcst_gfs_ghi` for TFT-168, `fcst_gfs_rh2m` for the baseline). A selection weight is not an
    effect; with one seed and the development budgets this is noted for stage 11, not acted on;
  - S09-24: the two run reports list 38 points of 7 kinds for G-1.
- **Checks:** every new PNG looked at (one legend moved off the labels of S09-21); T-72 passes.
- **Next:** unchanged: W6 after the answers of the college IT (Q-15).

### 2026-11-12 · R9-23 · W4 figures redone: look-back and one look for every model (X53, X54)

- **Request (user):** the same for the W4 figures; the figures must be uniform across the
  models; every figure production must look back over the earlier stages for figures that can
  be made in hindsight, from now on; update the Claude folder.
- **Rules:** rule 8 (look-back, with a log in `docs/figures_plan.md` §7) and rule 9 (uniform
  across the models, one style per model, coverage matrix §8); the project skill
  `.claude/skills/task-figures/SKILL.md`; `CLAUDE.md`, the auto-memory and
  `docs/claude_memory/figures-every-task.md`. T-72 checks the look-back rows and the styles.
- **W4 figures added** (validation only, dev budgets, one seed, nothing decided):
  - S09-25: the six models side by side; one objective for all (nRMSE in kW of the point);
  - S09-26: every model against persistence (DM in kW): 5 of 6 significantly better in all six
    cases, PatchTST in 5 (not at indirect D+1); none worse. The dev runs used different DM
    references (XGB's run: XGB; the neural run: persistence), which the evaluate step chooses;
  - S09-27: the two routes per model: direct better at D+1 for 4 of 6; largest gap PatchTST;
  - S09-28: three validation days chosen by rule (highest, median, lowest mean k);
  - S09-29: the quantile forecasts: **X53**, quantiles out of order on 66 % (TFT-96 direct) and
    56 % (TFT baseline direct) of the scored hours; coverage of the 80 % interval 62–88 %;
  - S09-30: every model errs most in winter (February) and least in summer; normal operation
    lowers the error by 1.3–1.9 points.
- **Look-back (X54):** S09-31 and S09-32 (W2: XGB curves and searches in the neural layout),
  S09-33 (W1: the 14 runs of stage 9, 3.9 h of steps, 88 % neural; the 4 failed steps of the
  first neural smoke passed in the next run), S08-04 (stage 8: the secondary views; the
  data-source views equal all hours on validation), S09-06 redrawn with all five encoders from
  the W3 report, XGB black in every figure (S09-02 redrawn).
- **Not built:** explanations of the BiLSTM and PatchTST (new computation, D-31), seed spread
  (final run), intervals of XGB and PatchTST (point forecasts), test-split figures (stage 10).
- **Decision needed (user):** X53, whether to sort the quantiles before any interval or the
  L-14 potential skill uses them; on the G-1 list. Decided in R9-24.

### 2026-11-12 · R9-24 · D-33: the TFT quantiles are sorted after training, the point stays (X53)

- **Question (user):** what is the professional solution for the crossing quantiles, how does it
  affect the forecast, and is it converted afterwards so that the models can be compared.
- **Options weighed:** (A) sort for any use as a distribution, point forecast unchanged; (B) sort
  everything including the point; (C) do not sort; (D) prevent the crossing in training. The user
  asked whether prevention and sorting differ: they do, because prevention is a new model (new
  weights, new tuning, about 3 h of development reruns, a change from the submitted TFT) with no
  guarantee against the current one, while sorting keeps the model and has a proof.
- **Decision (user):** sorting after training, option A (D-33).
- **Checks on validation (scored daytime hours, 8,703 per model and route, development run,
  seed 42):**
  - the written point forecast equals the raw 0.5 head in kW for all 8 model and route pairs;
  - sorting then converting equals converting then sorting (largest difference 0.0), because the
    conversion is a product with a non-negative number followed by a clip;
  - crossing: TFT-96 direct 66.4 %, TFT baseline direct 56.1 %, TFT-96 indirect 12.4 %, others
    below 2 %, BiLSTM 0 %;
  - quantile loss (direct, kW): TFT-96 35.59 to 34.66 (−2.6 %), TFT baseline 35.41 to 34.91
    (−1.4 %), TFT-168 unchanged to two decimals; the 80 % interval covers 62.2 to 63.8 % and 72.0
    to 80.5 % for the two TFTs; on the indirect routes of the three TFTs it moves by less than 0.1
    point;
  - had the point been sorted too, it would have moved on 60.5 % of the TFT-96 direct hours (8.7
    kW mean over all hours, 47.6 kW at most) and the nRMSE would have risen by 0.016 points
    (TFT-96 direct) and 0.010 (TFT baseline direct), below the 0.05 margin.
- **Code:** `solarpv.evaluation.quantiles` (`ordered`, `crossed`, `pinball`, `coverage`); T-92
  (`tests/unit/test_quantiles.py`); S09-29 reads the sorted quantiles and asserts that the BiLSTM
  never crosses and that sorting never raises the loss; its legend moved below the panels.
- **Guard added (same day, user request: complete the gap):** nothing but S09-29 reads the
  quantiles as a distribution yet, so D-33 had no enforcement. `tests/contracts/test_quantile_use.py`
  fails when code that reads `quantiles_*.csv` or a `q<level>` column does not import
  `solarpv.evaluation.quantiles`; the writer (`pipeline/neural.py`) must stay raw and the report
  may only filter names. Mutation check: with the import removed from the figure module the
  guard names that file; a planted reader is caught. On the development run the point equals the
  0.5 head for all four quantile models, sorting keeps it, and the sorted loss is not higher.
  The first version of the report check failed on a filter spread over two lines; the check now
  reads the file with whitespace collapsed.
- **To do before the report:** state the cost (on crossed hours the point is not the median of
  the sorted distribution); verify the reference to Chernozhukov, Fernández-Val and Galichon
  (2010), written from memory; record prevention in training as future work.
- **Next:** unchanged: W6 after the answers of the college IT (Q-15).

### 2026-11-12 · R9-25 · W6: self-review before the code of gate G-1

- **Start:** the college IT has not answered Q-15 yet; the user said to start W6 anyway (the
  session stays on all night). The answers decide only the mode of the final run (runbook §3).
- **Read first:** `PROJECT_MASTER.md` §1, §3, §6.13, §7 (gate G-1), §8.2, §16; the runbook; the
  stage-9 plan §1.1 and §19; R9-17–R9-24; errata X38–X54; the original
  `reproduce_paper_results.py` and leakage audit.
- **Full suite at `b2c3f1f`** (before any W6 change): 564 passed, none skipped or failed, in 538 s,
  the GPU suite included (`outputs/g1_audit/pytest_full_junit.xml`). The brief's "about 66
  minutes" was not this suite's time.
- **The original (M32–M36, errata C-65–C-69):** the reproduction started from the feature
  parquet, compared nothing, ignored the environment, and covered two of six models; the leakage
  audit listed names, not availability times.
- **My own code and documents (X55–X61):** an editable install that makes any other checkout
  run this checkout's code (X55); a runbook rule narrower than the gate it describes (X56); look-
  ahead in the data-preparation steps before the hourly table, never audited (X57); decisions
  still "proposed" (X58); stage-9 analyses without code (X59); a stale scope register (X60); A-01
  and A-14 not updated (X61).
- **Facts found while reading:** the test-access log `outputs/07_evaluation/test_access_log.csv`
  does not exist (it is created at the first access); the fixture of T-15 reads the submitted LSTM
  file's own truth, not a rebuild table; the figures that read the stage-6 table filter to train;
  the untracked log and run folders do not make the tree dirty (`--untracked-files=no`).
- **Plan:** stage-9 plan §20; nothing coded before this entry is committed.

### 2026-11-12 · R9-26 · W6: the stage modules ran on `--help`; stages 4–5 rewritten and restored (X62, OP-60)

- **What happened:** to write the reproduction script, I called `python -m <module> --help` in a
  loop over twelve stage modules, to read their options, in this checkout. The modules had no
  argument parsing, so each ran its stage. `solarpv.data.preprocess` and `solarpv.data.clean`
  completed and rewrote the stage-4 and stage-5 files and the tracked
  `outputs/05_cleaning/cleaning_manifest.json`. `ingest`, `features.build`, `features.decide_k`
  and `nwp.gfs` stopped before their writes (the shell closed their output pipe after 14 lines;
  the GFS run was stopped by hand); the loop never reached `nwp.check`, the evaluation or the
  selection (which would have made a fourth validation look).
- **Damage check:** `git status` and the file times: the four parquet files and two manifests of
  stages 4–5 and the tracked cleaning manifest changed; the stage-6 tables, `configs/target.yaml`
  and the GFS tables did not; three new logs of aborted runs.
- **Content:** the regenerated files equal the originals exactly (shape, columns, types, index,
  every value), and every statistic of the regenerated cleaning manifest equals the committed
  one; only the bytes differ, from the parquet writer (pyarrow 26.0.0 and pandas 3.1.0 here, 25.0.0
  and 3.0.6 on the laptop). An accidental confirmation that stages 4–5 reproduce on this machine.
- **Restore:** the nine files were extracted from the hand-off package into a new folder, each
  checked against `HANDOFF_MANIFEST.sha256`, and copied back; the regenerated files were moved
  aside, not deleted; the three logs went to `outputs/g1_audit/incident_2026-11-12/`. After the
  restore, 22 of 22 stage files equal the hand-off hashes, the stage-6 and 6b manifests match
  their tables, and the tree shows only the code being written.
- **Fix (X62):** `solarpv.config.cli`; every entry point parses its arguments before any work,
  `--help` runs nothing, unknown and abbreviated options are errors (`--valid` is no longer
  `--validation-only`, `--allow` no longer `--allow-test`). T-98 checks every entry point
  statically before it calls any of them.
- **Lesson for the reproduction and the external review:** never call a stage module to read its
  help in a checkout that holds data; read the source.

### 2026-11-12 · R9-27 · W6: guards, the smoke run, and the first reproduction attempts (X63, X64)

- **Smoke at `5214bbc`** (`20261112T200502Z_smoke`, after X55 and X62): all 11 steps ok in 260 s,
  26 of 26 warnings matched; the 40 prediction, quantile and array files are byte-identical to
  `20261112T062618Z_smoke` (`46881ae`), so the guards changed no result. Four later steps recorded
  `5214bbc+dirty`, because the reproduction script was being written during the run; the smoke
  run that counts for G-1 is repeated at the commit to be tagged, with nothing edited meanwhile.
- **Dry run of the final run** (`outputs/g1_audit/dry_run_final_09533a3.log`): writes nothing;
  about 31,100 s (8.6 h) expected, up to about 68,600 s (19.1 h), as the runbook says. It prints
  the plan only and does not check the gate; `check --mode final` does.
- **Test access** (`scripts/g1_audit.py --only test_access`): no access log, never in git; 86 step
  records of 15 runs without a test access; 81 prediction files without a test hour.
- **Reproduction, attempt 1** (`e2b6fae`, stopped after three steps): the ingest and the clock
  estimates differed from the reference although the clock step ran on the reference ingest; the
  comparison could not say by how much, and the restore deleted the regenerated files. Stopped,
  improved (the size of every difference, a copy of every differing file, the noise rule of
  §20.5 registered before any size was known), the private site file removed from the clone by
  hand (the stopped process could not).
- **Reproduction, attempt 2** (`09533a3`): the ingest differs by up to 9.7e-14 relative in 12.5 %
  of the values; Python's correctly rounded `float()` agrees with this machine's pandas 3.1.0 on
  132 of 132 sampled values, so the laptop's pandas 3.0.6 parser was not correctly rounded (X63);
  the clock daily estimates differ by up to 3.2e-12 relative on identical inputs (numerical
  libraries), and the offsets file is identical. Preprocess and clean reproduced exactly. The EDA
  failed: it named the raw files relative to the checkout (X64). Fixed; attempt 3 from a new
  clone.

### 2026-11-13 · R9-30 · W6: the reproduction finished; the comparison bug (X67) and the residual differences

- **Result:** `reproduction_20261112T202955Z` (commit `c463603`, fresh clone `r3`), 19 steps,
  about 4 h 37 min: checksums, ingest, clock, preprocess, clean, eda, quality, orientation,
  features, decide_k, features_final, gfs, nwp_check, evaluate, fixture, selection (the train part,
  4,550 s), sequence_data, dev_xgb (398 s), dev_neural (10,518 s). Every step exited 0. The
  script's verdict is `NOT reproduced`, by its rule (any failing file), not by a failed step.
- **Failing files, and their causes:**
  - `decide_k` (3 files) and `nwp_check` (2 files): the laptop's decision checks, numbers
    differ, decisions equal (X66). Not changed.
  - `dev_neural`, `report_steps.csv` and `report_warnings.csv`: explained by X45 (the step now
    emits the `tft168` indirect warning that the W4 step did not) and by the row order of the
    resumed reference report. The same five warnings, the same forecasts (bytes).
  - the comparison bug (X67): a missing text value and the step fingerprint were counted as
    differences. Fixed, with a regression test (T-93); `dev_xgb` now has no failing file.
- **Forecasts:** every prediction, quantile and benchmark file of both dev runs is identical in
  bytes or in content (`recheck_dev.json`).
- **Not yet:** S09-34 and S09-38 are not built. S09-38's builder asserts that every failing file
  belongs to the decision checks; it has to be changed to the explained classes above, once the
  user agrees to that classification. The final reproduction must be rerun at the commit that is
  tagged, because the comparison code changed after this run.

### 2026-11-13 · R9-31 · W6: the user's decisions, the external review, and the plan of `cv` and `ablation`

- **The user's decisions (2026-11-13: "it is clear that the reproduction will not be one to one;
  the principles and the methodology stay", then "work on your recommendations"):** D-34 (the
  reproduction is a consistency check; explained differences are listed, no rule is widened), D-35
  (the centred clock correction is accepted and declared), D-36 (the two analyses that train
  models are written before the tag, the others after it under a written rule), D-37 (no change
  of budget, seed or validation rule; the W4 and W5 points are limitations), Q-26 to Q-28 and the
  confirmation of D-06, D-07 (path corrected), D-14 and D-16. Recorded in `PROJECT_MASTER.md` §3
  and §16, `docs/deviations.md` and the stage-9 plan §20.7.
- **External review:** a separate Claude session (`g1-review`, its own worktree and branch
  `g1-external-review`) started on commit `1cf6610`, with `docs/g1_review_brief.md`; it writes only
  `docs/g1_review_log.md`. The steps `cv` and `ablation` get a review of the difference later.
- **Plan:** stage-9 plan §21 (`cv` for S-38, `ablation` for S-41): the original's mistakes M37–M44
  (an ablation scored on the test, a baseline that is not the XGB-Opt, groups by substring that
  leave the rolling std features in the model, a `minimal` mode that contradicts its text, deltas
  below the seed noise read as findings, a private metric, an unreproducible CV and an unknown CV
  scope), my own X68–X75, the design, the tests T-100–T-103 and the acceptance, all before code.
  Errata C-70–C-77 and X68–X75.
- **Also:** OP-61 written (the reproduction and the audit); R9-30 said 13 steps, it was 19 (fixed).

### 2026-11-13 · R9-32 · W6: the steps `cv` and `ablation` (D-36), their runs and figures

- **Code** (`acea8ea`): `solarpv.pipeline.analyses` (both steps and their phase-2 scoring), the
  refactor of `fit_mos` into `oof_folds` (X68), the registration (config, estimate, `promote`,
  report, the step `test`, the runbook), the after-tag rule (`checks.after_tag_problems`, T-103).
  Tests T-100–T-103: 18 in `tests/unit/test_analyses.py` plus the contract test and the updated
  promote and quantile-guard tests.
- **Found while testing:** a final-mode replay of a smoke record refused its own `full` check:
  the training rows now come from the `xgb` record (`smoke_rows`). The quantile guard (D-33)
  flagged the audit script and the figure module, which only hash or name quantile files; both
  are now listed as such, with a test.
- **Runs:** smoke at `acea8ea` ok (12 steps, 328 s); dev `20261113T075845Z_dev` ok; the xgb
  outputs are byte-identical to the W5 dev run; `full` reproduced the `xgb` value in 55 of 55
  fits. Values and the validity review: stage-9 plan §21.7. A presentation fix followed the
  review: a one-seed verdict is marked provisional in the report, the step summary and S09-41.
- **Figures:** S09-40 (the CV folds against validation) and S09-41 (the ablation deltas with the
  margin), looked at; S09-38 now asserts that every failing file of the reproduction is one that
  D-34 explains; S09-34 trusts earlier evidence only while no code path changed since (D-36).
- **Candidate freeze** (`22f05d2`): the full suite (the figure contract excluded, because it
  checks figures that read this evidence) passed: 607 passed, 1 skipped (T-103, active from the
  tag); the dry run writes nothing.
- **The external review did not start:** the session `g1-review` was created but never ran
  (its setup did not finish, no message reached it), and it opened in the main folder and not in
  a worktree. It keeps its task; the user starts it from its row, or asks for a new one.

### 2026-11-13 · R9-33 · W6: the final reproduction, the audit refreshed, the review started again

- **Reproduction at the candidate** (`0addc5b`, a fresh clone `repro_g1/r4`, `--dev --with-eda
  --rerun-selection`, 19 steps, about 4.7 h): started 10:33 and running while the rest was done;
  its record is read in the entry that follows.
- **The audit refreshed (X76):** the test-access part had been run before the three runs of
  13 November (15 of 18 runs seen) and its glob missed the ablation forecast files. Rerun over all
  18 runs: 120 step records, 103 prediction files, no test hour; the 3 ablation forecast files
  checked apart (194,920 rows, all validation, last hour 2024-09-30). S09-34 and S09-37 now refuse
  an audit older than the runs on disk.
- **External review:** the first session (`g1-review`) still had no message after a second try (a
  message does not start a session that never started). A new session, `g1-review2`, was started
  from the same brief with the user's approval of Q-28 (a separate session, now); it writes only
  `docs/g1_review_log.md`.
- **Notebook 08** (`scripts/make_g1_gate_notebook.py`): the gate's process notebook, notes formatted
  from the audit files, the latest finished reproduction record and the dev run of `cv` and
  `ablation`; its builder refuses an unfinished record and asserts every claim of its notes.

### 2026-11-13 · R9-34 · W6: the external review, its answers and the new candidate

- **The review** (`g1-review2`, stage-9 plan §22): 12 rows, 1 blocking, 3 should fix, 8 notes. The
  first reviewer session never started (Git was not found on its PATH). Every row was verified in
  the code before it was answered.
- **Stopped:** the reproduction that was running on `0addc5b` (it had passed 11 of 19 steps),
  because the review demands code changes and a reproduction at an earlier commit cannot count for
  the tag (D-36). Its partial record is kept apart (`outputs/g1_audit/reproduction_0addc5b_stopped`).
  The private site file of its clone was removed.
- **Blocking, R-G1-01 (X77):** every neural step of a final run wrote `mbe` rows for the test
  scope, computed against the test truth, in phase 1. Fixed, with a test that fails on the old
  code.
- **Should fix (X78–X80):** phase 1 held the test truth in its files (`y_true`, the job arrays);
  the phase-2 command ran `report` before `test`; the acceptance was bound to nothing. While
  testing the phase-2 command a bug of my own showed: a forced rerun of `report` failed on its own
  moved-aside folder, so the runbook's command would have stopped at its first step (X79). The
  command is unchanged; the order, the report's inputs, `promote`, the acceptance binding and the
  files' truth are fixed (T-104–T-106).
- **Notes:** the evaluation CLI no longer scores the test split (X81, T-107); `test_label` for the
  ablation on the test (X82); the L-09 sentence made exact (X83); `k̂ ≥ 0` documented as a kW
  property (R-G1-06); the conversion copies tested against `units.to_kw` (T-108); the runbook
  tells how to read a failure of `full` (R-G1-10).
- **Fix check by the reviewer** (6 rows closed, 6 new: 3 should fix, 3 notes): a retry of the phase-2 command after a failed `report` scored the test split twice (`--force` now never replaces a completed `test`, X84); the report could be older than `evaluate` (X85); the promoted prediction files had a blank test `y_true` (the step `test` writes `truth_test.csv`, `promote` fills the column, X86); the alignment table says how many rows it compared (X87); the wording of what phase 1 holds, and leftovers (X88). Tests T-109 and T-110.
- **Second fix check** (6 rows closed, 5 new: 1 should fix, 4 notes): `promote` now compares what it promotes with what `test` scored (X89, T-111); the benchmark of `test` is written after the SHAP and the analyses (X90); a step folder that `run.json` did not list is adopted (X91, T-112); the runbook says the two refusals that are not defects (X92).
- **Third fix check:** 5 rows closed, 3 notes (X93), `Review completed: 2026-11-13` by the reviewer: every blocking and should-fix row is closed. Each smoke run at the successive candidates was byte-identical to the previous one in all 40 forecast files, so no fix changed a development output.
- **The audit script (X76)** now reads every kind of forecast file (147 files of 18 runs, no test
  hour), with a test; its stand-alone check file was removed.
- **Next:** the full suite, smoke and dry run at the new candidate, the audit over the new runs, the
  reproduction from the raw data at the same commit, and a check of the fixes by the reviewer
  session; the figures and the tag wait for them.

### 2026-11-13 · R9-35 · W6: the reproduction at the final candidate, and the figures of the task

- **Reproduction** (`reproduction_20261113T120823Z`, commit `e262fe3`, a fresh clone `repro_g1/r5`,
  `--dev --with-eda --rerun-selection`): 19 steps, every one exit 0, 4.98 h of step time (14:08 to
  19:09). The two environments are the existing ones and match the lock files; nothing was installed.
  Of 442 files listed: 227 identical in bytes, 29 equal tables and 38 equal data files (other
  bytes), 16 tables that differ only by floating-point noise (relative at most `1e-9`), 81 not
  compared (logs, databases, run folders), 39 step records, worker files and reports that are
  listed as expected to differ (hashes, times, paths of another run), 7 run logs without a
  reference, and 7 failing.
- **Forecast files:** every forecast file, best-parameter file and result table that the record
  lists (12 in the XGB development run, with its `cv` and `ablation` steps, and 16 in the neural
  run) is identical in bytes to the development runs of this checkout. The step records and worker
  files of a rerun differ in hashes and times (listed, not failing).
- **The 7 failing files, all explained (D-34):** `decide_k` (`configs/target.yaml`,
  `d11_criteria.csv`, `l04_training_threshold.csv`) and `nwp_check` (`configs/nwp_decision.yaml`,
  `value_check.csv`) give other validation numbers on this machine and the same decisions (X66);
  `report_steps.csv` and `report_warnings.csv` of the neural run differ because the step now emits
  the `tft168` warning and the row order of a resumed report differs (X45). The script writes
  `NOT reproduced` for any failing file; by D-34 the check is one of consistency with every difference explained, and no
  rule was widened.
- **Notebook 08 and the figures:** `notebooks/08_g1_gate.ipynb` was rebuilt from this record;
  `python -m solarpv.figures task S09-W6` built S09-34 to S09-42 and `docs/figure_sheets/S09-W6.md`.
  Every PNG was looked at. Fix: the checklist figure still said "the review of the new steps
  follows" for the external review; it now reads the completed review.
- **Next:** the user's approval of the tag; the tag; the final run by `docs/runbook_final.md`.

### 2026-11-13 · R9-36 · W6: checkpoints and resume inside the long steps (user request, D-38)

- **Request.** The college IT answered Q-15 (a disconnected session keeps running, no session time
  limit, `C:` not wiped, the machine always on: mode A). The user still asked that the code can
  continue after an interruption, with a checkpoint after every epoch and every model run. The tag
  had not been given; the candidate was reopened (stage-9 plan §23, written before the code).
- **Design (D-38).** `solarpv.models.checkpoint`; `outputs/runs/<run_id>/checkpoints/<step>/`,
  atomic writes, bound to the fingerprint of the step and to the content digest of the arrays,
  deleted when the step succeeds. The neural worker saves after every epoch, every finished trial
  and every finished target x seed; `xgb` after every finished trial and every finished target x
  horizon x seed; PatchTST after every target x seed. A study is replayed from its finished trials
  and its pickled sampler. The runner passes the folder and the fingerprint, shows it in `status`,
  keeps it after a failure, and a step that resumed says so (`notes.resumed`, a warning).
- **Two findings.** X95: the first design (every random state of torch restored) gave other weights
  than an uninterrupted fit for a model with a multi-layer LSTM with dropout between the layers;
  cuDNN keeps a dropout state that torch's random state does not hold, and setting the CUDA
  generator reseeds it. Measured with three small probes, then fixed by running the same training
  passes again on the freshly built model and not setting the CUDA generator. X96: `fit` updated
  Adam's step counters of the state it was given in place (found when the check script resumed twice
  from one state; a checkpoint file is read afresh, so a run never met it).
- **Checks.** T-117 (a fit continued after an epoch equals the uninterrupted fit, bit for bit: BiLSTM
  1 and 2 layers, with and without dropout, TFT 1, 2 and 3 LSTM layers) and T-118 (a worker killed
  at an epoch of a trial, at a trial boundary, in the final fit of the BiLSTM and TFT-96, in the TFT
  baseline and at the end of a target x seed, and started again, writes forecasts, trials, curves
  and records equal to an uninterrupted worker's); T-116 (the xgb step interrupted inside a study,
  after a unit and inside the second study); T-113 to T-115 and T-119. The check script
  (`scripts/g1_resume_check.py`, `outputs/g1_audit/resume_check.json`): the final design is equal in
  7 of 7 configurations, the first design failed in the 4 with dropout between layers (weights up
  to 2e-2 apart); the replay costs 15 to 34 % of the seconds of the epochs it replaces.
- **Nothing else changed.** A smoke run on the working tree before the commit (13 steps ok, no
  checkpoint folder left; the folder was deleted after the comparison, the smoke run at the candidate
  replaces it) had 27 CSV files byte-identical and 10 worker forecast arrays equal to the previous
  smoke run's (`20261113T114945Z_smoke`). Figure S09-43.
- **For the gate.** The code of the worker, the xgb step and the runner changed, so by D-36 the
  evidence of the earlier candidate no longer counts: the full suite, the smoke run, the dry run,
  the audit, the reproduction from the raw data and a review of the diff are repeated. The tag and
  the final run were not started.

### 2026-11-13 · R9-37 · W6: the review of the checkpoints (R-G1-27 to R-G1-35)

- **The review** of `9e816ef` (`g1-review2`): 1 should fix, 8 notes, none blocking. The reproduction
  that had run for 25 minutes on `9e816ef` was stopped (partial record kept apart, the private site
  file of its clone removed), because the answers change the code.
- **R-G1-27, right (X97).** The replay in order fails for the TFT when the windows have short rows and
  several batches per epoch (measured: TFT with 2 and 3 layers; the BiLSTM is equal either way); my
  tests had one batch of rows of one length for the TFT. Fixed with the real batches; the tests, the
  check script and a test of the failing version use short rows and batches of 16.
- **R-G1-33, measured.** The state of the largest BiLSTM is 83 MB and takes 0.44 s to write and
  sync, against epochs of 0.4 s (development run); that of a TFT with 3 LSTM layers is 43 MB and
  1.5 s, against epochs of 5.6 s. A write after every epoch would add up to about 4.7 h to the final
  run (a bound). So a fit's state is written about once a minute (`SOLARPV_CHECKPOINT_INTERVAL_S`; 0
  writes after every epoch, as the user first asked), and the trial and unit checkpoints, which
  are small, are written every time. **The user's request, "after every epoch", is kept for the
  epochs that are long enough to be worth it and changed for the short ones; the user accepted this in chat on 2026-11-13 (D-38).**
- **The notes (X98).** Retried writes, a warning for a fit checkpoint that cannot be written, cleanup
  and a `.gitignore` rule, an unreadable manifest and a unit without its study, the environment in
  the manifest, warnings and the column `n_resumed`, a tab in an errata row.
- **Checks.** 31 CPU tests and 42 GPU tests of the checkpoints pass; the resume check (7 of 7
  configurations equal, the first design failing in 4) was run again on the new windows.
- **Next.** Commit the candidate, repeat the smoke run, the audit and the reproduction at it, then
  the figures, the notebook and the full suite.

### 2026-11-14 · R9-38 · W6: the reproduction at the final candidate `ac63bb0` (D-34), and X99

- **Reproduction** (`reproduction_20261113T202349Z`, commit `ac63bb0`, a fresh clone `repro_g1/r8`,
  `--dev --with-eda --rerun-selection`, started after the review of the checkpoints was answered):
  19 steps, every one exit 0, 4.84 h of step time. The two environments are the existing ones and
  match the lock files; nothing was installed. The script writes `NOT reproduced` because 17 files
  fail, and by D-34 each is explained or the check fails. Of 443 files listed: 227 identical in
  bytes, 52 equal tables or data files in other bytes, 16 tables that differ only by floating-point
  noise (relative at most `1e-9`), 81 not compared, 42 listed as expected to differ (step records,
  worker files and reports of another run), 8 run logs without a reference, and 17 failing.
- **Forecast files:** every forecast file, best-parameter file and result table that the record
  lists, in the XGB development run (with its `cv` and `ablation` steps) and in the neural run, is
  identical in bytes to the development runs of this checkout. No step resumed (`n_resumed` is 0
  for all 16 steps of the two runs) and no checkpoint folder was left behind.
- **The 17 failing files, all explained (D-34), by three causes:**
  - 5 of `decide_k` and `nwp_check`: other validation numbers on this machine, the same decisions
    (X66); the same values as in the reproduction at `e262fe3`;
  - 3 report tables (`report_steps.csv` of the XGB and the neural run, `report_warnings.csv` of the
    neural run): the column `n_resumed` of the checkpoints (D-38), the warning that the neural step
    emits and the row order (X45); the `job.json` files carry the new key `resume`, listed;
  - 9 learning-curve and trial tables of TFT-96 and TFT-168 in which only `gpu_peak_mb` differs
    (X99, new at this commit). Measured with a probe that trains the same TFT eight times, after a
    reference run, with only the number of live Python objects changed: weights equal bit for bit, peak from -0.40 to +1.93
    MB. A peak of the GPU memory is not a result of the computation; it was equal at `e262fe3` and
    moved when the code of the fit changed. The explanation in the figures is tied to this cause
    (T-120), so another column in the same files would still fail. No rule of the comparison was
    widened (`scripts/` is unchanged).
- **Figures and notebook:** S09-38 now counts the explained files by cause, and notebook 08 shows
  the probe.
- **Next:** the full suite at the clean tree, the W6 figures, the documents, and the request for the
  approval of the tag.

### 2026-11-12 · R9-28 · W6: the manual audit, its measures, and the W6 figures (X57, X65)

- **Look-ahead in the data preparation (`scripts/g1_audit.py`, `docs/leakage_audit.md` §3.2):**
  - cleaning: every issuance day (1,364) cleaned up to T0 and with the next day; 66 of 458,304
    source cells change, on 4 train issuance days, all communication freezes from the evening
    into the next morning that R3 removes only once they reach daylight; none is a scored hour;
    the full-window cleaning equals the stage-5 table on every compared cell (the method check);
  - clock: a trailing median instead of the centred one would change the whole-minute shift on
    552 / 541 of 1,369 days (meteo / inverter clock), by at most 3 minutes;
  - polled reconstruction: test period only, a night hour, bounded by 0.68 °C and 0.23 W/m².
- **Figures S09-34–S09-39** (module `solarpv.figures.s09_w6`): S09-35, S09-36, S09-37 and S09-39
  built and looked at (four layout fixes: a cut axis label, two legends over labels, an empty bar
  explained); S09-34 and S09-38 wait for the reproduction. Look-back: the W6 smoke run added to
  S09-33.
- **X65 found by S09-35:** A-13 said the GFS files arrive "at least 5 hours before T0" (four
  sampled dates); the recorded upload times give a median of 5.1 h and a minimum of 1.1 h on the
  1,127 issuances of train and validation (53 under 5 h). The row-by-row rule of T-17 always held;
  the text is corrected.
- **GPU suite** (after X55): passes, and the worker records the file it imported.

### 2026-11-14 · R9-39 · Stage 10: the final run `20261114T083930Z_final`, both phases

- **Before the run:** `g1-frozen` tagged locally on `10611a6` (a descendant of `ac63bb0`, no code
  change, 202 changed files all inside the D-36 list; no push). `check` passed, `neural_env` reported no
  error (CUDA 12.1, `sm_61`), the dry run was ok, 199.5 GB free.
- **Phase 1** (one sequence, `--to report --keep-going`): 13 steps ok, one commit, `n_resumed` 0.
  Seconds: select 4.1, baselines 4.5, xgb 3,214.3, cv 121.1, ablation 389.0, bilstm 4,705.8, tft96
  8,700.5, tft168 12,209.9, tftbase 721.5, patchtst 899.1, evaluate 28.3, explain 332.7, report 1.1.
  Slower than the dry-run estimate: BiLSTM 4,706 s against 1,716 s (x2.7); the others at or below it.
- **Acceptance** (`stage10_acceptance.md`, accepted): XGB seed spread in kW at most 0.079 pp (limit 0.15);
  kW DM on validation: XGB significantly better than all three references in 6 of 6 cells; 41 warnings
  read, none a failure (the G-1 list points of D-37). Archive of phase 1 written and verified (sha256
  `86bdd252…4ac`).
- **Phase 2**, on the user's explicit approval and after the user waived an off-VDI copy of the archive
  (the copy is not made; the project is not backed up outside the VDI): the step `test` 130.7 s, `report`
  rerun. Failure: none (the stderr file was empty). Test scores were not read before the step ended.
  Promote ok; archive `20261114T083930Z_final_20261114T175738Z.zip` verified, sha256
  `7d08139b7534fdf1bd2f012976ccf9fd644eb2aa90ea1e3cb292b526bf54c43f`.
- **Validity review** (`forecasting-validity-review`, written before any claim):
  - *isolation:* test truth appears only in `truth_test.csv` written by `test`; 17 rows in the access log,
    12 forecasts of phase 1 and 5 reads by `test`; the 5 `test` warnings are the SHAP, the CV comparison, the
    ablation and the evaluation. Same rows for every model (n = 2,575 hours with truth).
  - *red flag, a sharp change between inner folds, validation and test:* the XGB error is 11.1–11.6 on
    test against 6.4–7.2 on validation and 7.4–8.1 in the train CV (test minus CV 3.5–4.0 pp in all 18
    fits). The references degrade as well (climatology 8.04 to 12.37–12.52, clear sky × k 8.96 to 12.85,
    persistence 9.7–10.8 to 15.7–17.6), so the shift belongs to the test period and not to one model.
    The views of `benchmark_views.csv`, compared like for like (S10-03, mean over seeds and cells): XGB
    rises from 6.65 to 11.39 on all hours and from 4.93 to 6.55 in normal operation (2,389 of 2,575
    hours); climatology from 8.05 to 12.45 and from 6.76 to 8.95. The 186 operation-limited hours
    (D-22: curtailment, stops, derate) account for 66 % of the rise of XGB and 50 % of climatology's
    (18–68 % over the nine models and references); the rest, +1.6 pp for XGB, is a period shift that the
    references share. (A first reading in this session compared the test normal-operation number with
    the validation all-hours number; the figure S10-03 put the views side by side and corrected it.)
    By source on test: minute 12.62–12.95, polled 7.52–8.03 (XGB, seed 42).
  - *consequence for the acceptance rule:* the D-14 requirement that the XGB is significantly better than
    the references holds on test against clear sky × k, PatchTST and persistence, and fails against
    climatology in five of six cells (HLN p 0.0097–0.0224 against alpha 0.00625), with the margin at
    75–85 % of the minimum detectable effect (109 % in the one passing cell). The limitation was stated
    before the run (D-37).
  - *not concluded:* the BiLSTM, TFT-96 and TFT-168 are not significantly different from the XGB in most
    cells, and TFT-96 is nominally best in the direct route; no model is declared the winner.
- **No change:** no budget, seed or rule was changed after the test numbers. The off-VDI copy and the
  backup of the whole project are open (the user's decision, 14.11).
