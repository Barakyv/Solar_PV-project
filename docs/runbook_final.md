# Runbook: the final run (W6, stage 10, and the test phase of stage 11)

Written on 2026-11-12 (stage-9 plan §19.8, decisions D-30–D-32 and the user's answer to Q-23).
Run it only after gate G-1 has passed and the code carries the tag `g1-frozen` (D-08). Nothing
in this runbook changes a budget, a seed or a validation decision: the final budgets are those
of v6 (`pipeline/config.BUDGETS["final"]`, asserted by T-80 and T-84).

## 0. Before W6: questions to the college IT (Q-15, open)

Ask before the first long run, and record the answers in `docs/stage9_plan.md` §1.1:

1. Does a **disconnected** Horizon session keep running, and for how long (the disconnect timer
   and the logoff timer for disconnected sessions)? Is an idle session locked, disconnected or
   logged off, and after how long?
2. Is `C:` of this VM kept after a **logoff**, and does the next login return to the same VM
   (the Horizon instant-clone agent is installed, §1.1)?

The answers decide between one sequence (§3, A) and segments (§3, B).

## 1. Facts

| Item | Value |
|---|---|
| Environments | `solar` runs the runner (`C:\Users\204323141\miniforge3\envs\solar\python.exe`); `solar-gpu` runs only the training worker (found by the runner, D-29) |
| Git | `C:\Users\204323141\PortableGit\cmd\git.exe` (not on `PATH`) |
| Budgets | XGB 60 trials × seeds 42, 123, 777; BiLSTM 60 trials × 25 epochs, final 60 epochs, patience 10, three seeds; TFT-96 and TFT-168 20 trials per target, up to 100 epochs, seed 42; TFT baseline up to 150 epochs; PatchTST 5,000 steps (D-08, v6) |
| Time | From `run --mode final --to report --dry-run` on 2026-11-12 (the latest dev duration of each step scaled by the fits, X43; figure S09-18): XGB about 0.8 h (up to 4.1 h if every trial took as long as the longest dev trial); BiLSTM about 0.5 h; TFT-96 about 3.2 h (up to 6.4 h if every fit ran its 100 epochs); TFT-168 about 3.8 h (up to 7.2 h); TFT baseline about 0.2 h (up to 0.6 h); PatchTST about 0.2 h; the rest about 1 minute. **About 8.6 h expected, up to about 19 h**. Rerun in W6 after the steps `cv` (about 102 s) and `ablation` (about 304 s) were added (2026-11-13, `outputs/g1_audit/dry_run_final_22f05d2.log`): 31,600 s expected and 67,316 s at most (8.8 h and 18.7 h). It reads the latest dev runs |
| Longest step | TFT-168, about 4–7 h; a step is the unit of `--resume` and cannot be split |
| Disk | The neural dev run folder took 192 MB; a final run is expected below 1 GB; keep 20 GB free |
| Two phases (D-32) | Phase 1 trains, forecasts validation and test, and scores validation only: the stage-10 acceptance. Phase 2, after the acceptance is written, scores the test split (step `test`) |

## 2. Checks before phase 1

Run from the repository root in a PowerShell window of the Horizon desktop that is **not** a
Claude Code session (the run must not depend on an agent session). Set
`$env:PYTHONIOENCODING="utf-8"` in that window.

1. The tree is clean and `HEAD` carries the tag:
   - `git status` shows nothing to commit;
   - `git tag --points-at HEAD` lists `g1-frozen`.
2. `python -m solarpv.pipeline check --mode final` ends with `check passed` (every input as
   recorded by its stage manifest, the stage-7 provenance, the tag, a clean tree, the private
   site file, the GPU environment).
3. In `solar-gpu`: `python -m solarpv.models.neural_env` reports no error (CUDA 12.1, `sm_61`).
4. `nvidia-smi` shows no other process using the GPU.
5. At least 20 GB free on `C:`.
6. `python -m solarpv.pipeline run --mode final --to report --dry-run` prints the plan and the
   estimates; nothing is written. It does not check the gate: item 2 does.
7. **Nothing is committed and no tracked file may change until phase 2 is complete, documents
   included** (X56): the gate is checked before every step and refuses a moved `HEAD` or any
   modified tracked file anywhere in the tree (X38), so an edit of `docs/run_journal.md`,
   `PROJECT_MASTER.md` or any other tracked document, by hand or by an agent session, stops the
   next step. Notes taken during the run go to a file outside the repository and are copied in
   after phase 2. New untracked files (the run folder, the test-access log) do not stop it.
8. Every command is run from the repository root: the runner refuses a working directory inside
   another checkout of the project (X55).

## 3. Phase 1: training and validation (stage 10)

**A, one sequence** (if a disconnected session survives at least about 20 h):

```
python -m solarpv.pipeline run --mode final --to report --keep-going
```

**B, segments** (each segment shorter than the session limit). The first command creates the
run and prints its id; every later one resumes it:

```
python -m solarpv.pipeline run --mode final --only select,baselines,xgb,cv,ablation
python -m solarpv.pipeline run --mode final --resume <run_id> --only bilstm
python -m solarpv.pipeline run --mode final --resume <run_id> --only tft96
python -m solarpv.pipeline run --mode final --resume <run_id> --only tft168
python -m solarpv.pipeline run --mode final --resume <run_id> --only tftbase,patchtst,evaluate,explain,report
```

After every segment:
- `python -m solarpv.pipeline status <run_id>`: every finished step `ok`, outputs `as recorded`;
- `python -m solarpv.pipeline archive <run_id>`, then copy the zip and its `.sha256` off the VDI.

## 4. The stage-10 acceptance (between the phases)

Read `outputs/runs/<run_id>/report/report.md` before any test number exists:
- every step completed, at one commit, the tagged one: compare the steps of the report with
  `python -m solarpv.pipeline list` (X74), so that a segment run did not skip `cv` or `ablation`
  (`promote` would refuse the run only at the end);
- the warnings of §3.5 read and explained (a warning is not a failure; it goes to the run
  journal);
- the XGB seed spread in kW below 0.15 pp at every target × horizon (the stage-10 rule, read
  in kW as Q-19 implies, X42);
- the kW DM table: XGB significantly better than the references (D-14, Q-19);
- the budgets and the GPU time recorded per model;
- the review with the `forecasting-validity-review` skill.

Write the verdict, with the date and what was read, to
`outputs/runs/<run_id>/stage10_acceptance.md`. The file stays untracked until phase 2 is done:
committing it now would move `HEAD` off the tag, and the gate would refuse the step `test`
(X51). The step `test` refuses to run without this file.

If the acceptance fails because of a bug: fix nothing in this run. Record the bug (F-xx, errata,
a test), reopen G-1, tag the fixed code again and start a new final run (stage 10: "if a bug is
found, fix, reopen G-1, and rerun everything"). Never change a hyper-parameter or a rule after
looking at a final result.

## 5. Phase 2: the test split

```
python -m solarpv.pipeline run --mode final --resume <run_id> --only test,report --force --allow-test
```

`test` scores the test split with the evaluation layer and computes the XGB SHAP files of §6.10
on the test window; every read of a test row is appended to
`outputs/07_evaluation/test_access_log.csv` (X40). The runner runs `test` first and `report`
second (R-G1-02), and `--force` reruns `report` with the test tables (the phase-1 report is moved
aside and kept). `--allow-test` runs only these two steps. Before it scores anything, `test`
checks that the prediction files are those that `evaluate` scored on validation (same SHA-256)
and that no other step finished after `stage10_acceptance.md` was written: **write the acceptance
last, after the final phase-1 step and after reading the report**, and if any step is rerun
afterwards, read the report again and rewrite it (R-G1-03). Phase 1 holds no test truth as a label
or a score (R-G1-04, R-G1-17): the `y_true` of the test rows is blank until the step `test` takes the
truth from the feature table and writes it to `truth_test.csv`; the encoder series and the persistence
forecast hold observed test-period values as past inputs, as any forecast does. The report must
be newer than every step it reports on (R-G1-15). `--force` never replaces a completed step `test`:
the test split is scored once. Then:

1. `python -m solarpv.pipeline promote <run_id>`: copies the run to the §6.10 paths
   (`outputs/06_<model>/`) and writes `outputs/06_promoted.json`; it refuses an incomplete,
   sliced or untagged run and never overwrites a file (X8);
2. `python -m solarpv.pipeline archive <run_id>`, and copy the zip off the VDI;
3. only now commit: the run folder's records, the promoted outputs, the acceptance file and the
   run-journal entries.

## 6. After a failure

- Read the step log `outputs/runs/<run_id>/logs/<step>_<stamp>.log` and the `error` of
  `<step>.failed-<stamp>/step.json`. A failed step leaves no file that looks complete (X4).
- **The environment failed** (out of GPU memory, disk full, a killed process, a logoff): fix the
  cause outside the code and rerun the step with `--resume <run_id> --only <step>`. The seeds
  are fixed and the training is deterministic (X32), so the rerun gives the same result; only
  the time is lost.
- **`report` failed in phase 2** (the step `test` is ok): run `python -m solarpv.pipeline run --mode final --resume <run_id> --only report --force`, without `--allow-test` (R-G1-14). The same
  phase-2 command also works: it skips the completed `test`, scores nothing again and reruns `report`.
- **The code failed** (an exception from a rule or a bug): stop. No code edit during a final run.
  Record it and reopen G-1 (§4). An exception of the step `ablation` that says the mode `full`
  does not reproduce the `xgb` step (R-G1-10) is rerun once with `--resume <run_id> --only
  ablation`; if it fails again it is a code failure; if it passes, write the first failure to
  the run journal as non-determinism.
- **The gate refused a step** (`refused (final gate: ...)`): something was committed or a tracked
  file (code or document) was edited during the run. `git status` names the file. Do not rewrite
  history; an uncommitted edit of a document can be saved outside the repository and undone with
  `git restore <file>`, then the step is resumed; for a commit, ask the user how to restore the
  tagged state.

## 7. After a logoff or a disconnect

- **Disconnect** (the Horizon client closed): if IT confirms that a disconnected session keeps
  running, reconnect and check `status <run_id>`.
- **Logoff or a reset of the VM**: the running step is lost. At the next login: `git status`
  (clean, at the tag), `python -m solarpv.pipeline check --mode final`, `status <run_id>`, then
  resume the remaining steps. The interrupted step's `.<step>.tmp` folder is moved aside as
  `interrupted` and never read.
- **`C:` was not kept**: restore the run from the latest archive. Verify it first
  (`archive --verify <zip>`), unpack it into a new, empty `outputs/runs/<run_id>/` (never over
  an existing copy), check `status <run_id>`, and resume.

## 8. Where the decisions come from

- D-08 (budgets and the frozen code), D-10 (no refit), D-14 with Q-19 (the kW DM),
  D-30 (`select` verifies and reuses), D-31 (`explain` scope and `archive`), D-32 (two phases).
- `docs/stage9_plan.md` §3.4–§3.6 (modes and safety rules) and §19 (W5).
