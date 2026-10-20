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
