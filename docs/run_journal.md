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
