# Gate G-1: brief for the external review

For a separate reviewer: a new Claude Code session opened in this repository, or the supervisor.
The gate (`PROJECT_MASTER.md` §7) asks that someone other than the author's session reads the code
against the canonical specification (§6) and that every comment is closed before the code is
tagged `g1-frozen`. Written 2026-11-12 (W6, `docs/stage9_plan.md` §20.6).

## What the project is

A day-ahead forecast of the AC power of two inverters of a 40 MW PV park (Ktora), issued at 00:00
local time for the next three local days, by six models (XGBoost, a BiLSTM, three TFT variants,
PatchTST) and three references, on two targets (the clear-sky index k, converted to kW, and kW
directly). It rebuilds a submitted capstone report (v6) whose code was partly lost, and corrects
every defect found (`docs/errata.md`). The test split (2024-10 to 2025-05) has never been read;
it is scored once, after the tag, in phase 2 of the final run (D-32).

## What to read first (in this order)

1. `PROJECT_MASTER.md` §6 (the canonical specification: data, time, site, targets, features,
   exclusions, splits, evaluation, file contracts, models, GFS) and §6.13 (assumptions). The
   document is in Hebrew; the code, the tests and every file under `docs/` are in English.
2. `docs/leakage_audit.md`, especially §3 (the manual audit of G-1).
3. `docs/stage9_plan.md` §3 (the runner), §17 (the input contract of the sequence models), §19
   (the explanation steps and the final gate) and §20 (this gate).
4. `docs/runbook_final.md`.

## The code map

| Area | Code | Tests |
|---|---|---|
| Ingest, time zone, clock, cleaning (stages 2–5) | `src/solarpv/data/` | `tests/unit/test_time_conversion.py`, `test_clock.py`, `test_resolution.py`, `test_preprocess.py`, `test_clean.py`; `tests/contracts/test_ingest_output.py`, `test_preprocess_output.py`, `test_clean_output.py` |
| Features and targets (stage 6) | `src/solarpv/features/` | `tests/unit/test_features_build.py`, `test_feature_contract.py`, `test_target.py`; `tests/leakage/test_leakage_rules.py`; `tests/contracts/test_features_output.py` |
| GFS (stage 6b) | `src/solarpv/nwp/` | `tests/unit/test_gfs.py`, `tests/contracts/test_gfs_output.py` |
| Feature selection (stage 7) | `src/solarpv/selection/` | `tests/unit/test_selection.py`, `tests/contracts/test_selection_output.py` |
| Evaluation (stage 8) | `src/solarpv/evaluation/` | `tests/unit/test_evaluation_8b.py`, `test_metrics.py`, `test_stats_tests.py`, `test_daytime_mask.py`; `tests/regression/test_evaluate_fixture.py` |
| Models and the runner (stage 9) | `src/solarpv/models/`, `src/solarpv/pipeline/` | `tests/unit/test_xgb.py`, `test_sequence_data.py`, `test_neural_runner.py`, `test_pipeline_runner.py`, `test_w5_steps.py`, `test_quantiles.py`; `tests/contracts/test_sequence_alignment.py`, `test_pipeline_output.py`; `tests/gpu/test_neural_gpu.py` |
| Reproduction and the G-1 audit | `scripts/reproduce_paper_results.py`, `scripts/g1_audit.py` | `tests/unit/test_reproduce.py`, `test_g1.py`, `test_g1_audit.py` |

## What to check

For each item, compare what the specification says with what the code does, and say where they
differ. Mark anything you could not verify as not verified.

1. **Time (§6.2, D-16):** every timestamp is true UTC; T0 is 00:00 Asia/Jerusalem; the horizon of
   a row is its local day counted from T0, also around the DST changes.
2. **Issuance (§6.6, `docs/leakage_audit.md` §3):** no input of a D+h row uses information after
   its T0 (lags, rolling windows, imputation, masks, GFS run choice, the encoder and decoder
   windows of the sequence models).
3. **Targets (§6.5, F-06):** k is never clipped at 1.2; k × theo inverts to kW exactly; the
   conversion of a k forecast to kW is the same function everywhere.
4. **Splits (§6.8):** nothing fitted or chosen reads a test row; validation is used only where a
   decision rule says so; the smoke and dev modes filter the test rows inside the parquet reader.
5. **Evaluation (§6.9):** the truth comes from the feature table, never from a prediction file;
   every model and reference is scored on the same hours; the DM test with HLN and Bonferroni;
   the kW tables of Q-19.
6. **Models (§6.11):** budgets, search spaces and settings against `pipeline/config.py`; the final
   budgets equal v6 (D-08); early stopping on validation (D-10); the quantile handling of D-33.
7. **The final gate (§19 of the stage-9 plan):** a final step refuses a moved `HEAD`, a dirty tree
   or a slice; every read of a test row is logged; the step `test` needs the acceptance file.
8. **Anything else** that would make a reported number wrong, unreproducible or not traceable to
   a file.

## Rules for the reviewer

- **Read the source; do not run stage modules to see their options.** A stage module run in this
  checkout rewrites data (X62 happened on 2026-11-12; the modules now refuse unknown options, but
  the rule stays). Running the test suite is fine: `python -m pytest -q` (about 10 minutes, the
  GPU suite included).
- Never run `--mode final`, never pass `--allow-test`, never open a test-split row.
- Do not edit any tracked file. Write comments only into `docs/g1_review_log.md`, one row each.
- Do not read or copy `configs/site.private.yaml` (the true coordinates, T-45).
- No external services, no downloads, no installs.

## The form of a comment

One row in `docs/g1_review_log.md`: an ID (`R-G1-01`, …), the file and line, the rule of §6 (or
the decision, D-xx) it concerns, what the code does, what it should do, and a severity:
**blocking** (a wrong or unreproducible number, a leak, a test row read), **should fix** (a
weakness that does not change a number now), or **note**. The author answers each row: fixed (with
the commit, the test and the errata row) or explained. The tag waits until every blocking and
should-fix row is answered.
