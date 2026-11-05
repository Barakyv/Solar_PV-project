# Stage 9 plan: the model code and one automated runner for the college machine

Written on 2026-11-01, before the code. **User request (2026-11-01):** no model is trained on the
laptop. One script, run on the college computer, trains every model of the study in sequence,
evaluates them, and keeps the training logs and every parameter, so the quality of each training
can be checked afterwards. It must be possible to run a single model or a section of the
sequence.

This plan covers the code of stage 9 (light models, the input contract of the sequence models,
smoke tests) and the runner that later executes stage 10 (final training on the frozen code) and
the test evaluation of stage 11. The project rules do not change: the test split decides nothing,
and final training happens only on the code tagged `g1-frozen` (D-08).

## 1. Facts at the start

| Item | State |
|---|---|
| Data | Stages 2–6b done: `data/processed/06_features/features_1H_D{h}.parquet`, `06b_gfs/gfs_features_D{h}.parquet`, with manifests and hashes. The models need nothing earlier |
| Feature selection | Stage 7 outputs exist; one decision is open: the direct-route sets are too small because of the weak selection model (R7-03) |
| Evaluation | Stage 8 done: `solarpv.evaluation.evaluate.run` scores prediction files in the contract of §6.10, val by default, test only with `allow_test` and logged |
| Baselines | Done (`solarpv.models.baselines`) |
| Model code | `xgb_sota.py`, `lstm_sota.py`, `patchtst_model.py` are docstrings only; nothing for the TFT models |
| GPU requirements | `requirements-gpu.txt` lists neuralforecast 3.1.7 and lightning, but not pytorch-forecasting 1.7.0, which §6.11 names for the TFT models: to fix |
| College machine | Inspected on 2026-11-03 (§1.1): a VMware Horizon virtual desktop with an NVIDIA GRID P40-24Q vGPU (24 GB, Pascal, CUDA ≤ 12.2), 8 vCPUs, 64 GB RAM, 228 GB free. No Python, git or conda on arrival; Miniforge and Git were installed per user the same day, and all 357 tests pass. Still unknown: the longest allowed job and whether the desktop survives logoff (Q-15, part) |

### 1.1 The college machine (inspected 2026-11-03, first session; answers Q-15)

Read-only inspection with PowerShell, `nvidia-smi` and HTTPS HEAD requests. Nothing was
installed or changed.

| Item | Finding |
|---|---|
| Access | VMware Horizon virtual desktop of the college VDI (brokers `vdi-broker-1/2.management.afeka.local`), host `GPU-STUD-164V6`, domain `ACADEMIC`. The VM is `VMware7,1` with a hypervisor present |
| OS | Windows 11 Pro 21H2, 10.0.22000, 64-bit |
| GPU | NVIDIA GRID P40-24Q: a 24 GB vGPU profile of a Tesla P40 (Pascal, compute capability 6.1), WDDM mode, also drives the desktop (about 0.75 GB in use at idle) |
| Driver and CUDA | Driver 538.46; highest CUDA runtime it supports: 12.2. No CUDA toolkit (`nvcc` absent, `CUDA_PATH` unset), which PyTorch wheels do not need |
| CPU | 2 sockets × 4 cores, Intel Xeon Gold 6136 at 3.0 GHz: 8 logical CPUs |
| RAM | 64 GB (48.5 GB free at inspection) |
| Disk | `C:` 300 GB NTFS, 228 GB free. `X:` is a network share (`\\192.168.20.27\TecShare`, 757 GB free); it is shared, so neither the data nor `site.private.yaml` go there. `D:` is empty |
| Internet | Direct, no proxy. HTTPS reachable: `pypi.org`, `files.pythonhosted.org`, `conda.anaconda.org/conda-forge`, `repo.anaconda.com`, `github.com`, `download.pytorch.org` |
| Scheduler | None: no SLURM, PBS or HPC Pack. Jobs run inside the interactive desktop session |
| Longest job | Unknown. It depends on the Horizon session policies (disconnect and logoff timers): to ask the college IT |
| Persistence | Unknown. The Horizon Instant Clone Agent is installed, the user profile is local and not roaming and was created at the first login (2026-11-03 10:45), and the VM was booted on 2026-10-14. Whether `C:` keeps the project after logoff, and whether the next login returns to the same VM, is to ask IT before any long run |
| Software on arrival | No Python (only the Microsoft Store aliases), no git, no conda or mamba, no Docker. `winget` and `wsl.exe` exist (WSL not checked). The user account is in the local Administrators group, but the session runs unelevated |
| Installed 2026-11-03 | With the user's approval, per user and not on `PATH`: Miniforge 26.9.4-1 in `C:\Users\204323141\miniforge3` and PortableGit 2.56.0.4 in `C:\Users\204323141\PortableGit`. The CPU environment `solar` was created from `environment.yml`; its differences from `environment.lock.yml` are in RH-02 of `docs/run_journal.md`. The user kept it as created rather than aligning it with the laptop lock |

What this means for the plan (decided in §5 when the GPU environment is built):

- **PyTorch build.** Pascal (sm_61) with a driver capped at CUDA 12.2: use wheels built for CUDA
  12.1 or 11.8, and check that `torch.cuda.get_arch_list()` lists `sm_61` before any run. The
  versions pinned in `requirements-gpu.txt` must be checked against this.
- **Precision.** The P40 has very low FP16 throughput, so the neural models train in FP32 (no
  16-bit mixed precision).
- **CPU.** Eight logical CPUs bound XGBoost (`hist`, CPU, §5), the stage-7 rerun and the data
  loaders; `n_jobs` ≤ 8.
- **No scheduler.** A run lives in the desktop session, so `--resume` (§3) and atomic outputs
  matter more; the session timers decide how a long run is split.

## 2. What runs where

| Where | What |
|---|---|
| Laptop | Writing the code, the unit tests, documentation, notebooks, and **smoke runs only**: a few days of data, one trial, one or two epochs, minutes in total, to prove the code runs before it is shipped (needs the user's approval, Q-18). No real training |
| College machine | Everything heavy: the stage-7 selection rerun, the development runs of stage 9 (validation only), the final runs of stage 10 (frozen code), the test evaluation, SHAP, attention, ablation |

**Changed on 2026-11-03 (user):** the work itself moves to the college machine, so the code is
written and the smoke runs are made there too; the laptop row above applies only if work returns
to the laptop, and Q-18 is moot. The transfer is the package of `scripts/handoff_pack.py` (no
GitHub remote exists yet), described in `docs/HANDOFF.md`.

## 3. The runner

### 3.1 One entry point

`python -m solarpv.pipeline <command>`

| Command | Purpose |
|---|---|
| `list` | The steps, in order, with what each needs and produces |
| `check` | Environment (Python, packages, CUDA, GPU, CPU, memory, disk), data (every file against the bundle's SHA-256 manifest), gates (mode, git commit, tag, clean tree). Nothing is trained |
| `run` | Runs the selected steps in order (§3.3) |
| `status` | What has completed, with which code, configuration and data |
| `bundle` | On the laptop: packs the processed data and configuration the runner needs, with a manifest |
| `collect` | On the college machine: packs a run's logs, predictions, metrics and figures to bring back (checkpoints optional) |

### 3.2 The steps

| Step | Stage | What it does | Device |
|---|---|---|---|
| `env` | — | Environment report | — |
| `data` | — | Verifies the bundle | — |
| `select` | 7 | Feature selection (after the open decision); writes the 05e sets | CPU |
| `baselines` | 8 | Persistence, climatology, clear sky × MEAN_DAYTIME_K | CPU |
| `xgb` | 9/10 | XGB-Opt | CPU |
| `bilstm` | 9/10 | BiLSTM-Opt | GPU |
| `tft96`, `tft168` | 9/10 | TFT with encoders of 96 and 168 h | GPU |
| `tftbase` | 9/10 | TFT with fixed hyper-parameters | GPU |
| `patchtst` | 9/10 | PatchTST, univariate | GPU or CPU |
| `evaluate` | 8/11 | The evaluation layer on every prediction file of the run | CPU |
| `explain` | 11 | SHAP (XGB), attention (TFT), hyper-parameter importance (Optuna) | CPU/GPU |
| `report` | 11 | One summary of the run: metrics, training diagnostics, budgets, times, warnings | CPU |

### 3.3 Choosing what runs

- `--only xgb,tft96`: these steps only.
- `--from xgb --to patchtst`: a contiguous section.
- `--targets indirect`, `--horizons 1,2`, `--seeds 42`: a slice of one model.
- `--resume`: skips a step whose completed manifest has the same configuration, code commit and
  input hashes; `--force` reruns it.
- `--keep-going`: a failed model does not stop the sequence; the report marks it.
- `--dry-run`: prints the plan, the budgets and an estimate of the time, and runs nothing.

### 3.4 Modes

| Mode | Purpose | Budgets | Data | Gate |
|---|---|---|---|---|
| `smoke` | Does the code run? | 1 trial, 1–2 epochs, a few weeks of data | train/val | none |
| `dev` | Stage 9 | D-17: XGB 20 trials, one seed; the neural models with reduced budgets (§4.3) | train/val | none; the test split is refused |
| `final` | Stage 10 | D-08: the budgets of v6 | train/val for training; test only in `evaluate`, with `allow_test` | refuses to start unless `HEAD` carries the tag `g1-frozen` and the tree is clean |

### 3.5 Logs and the quality of training

Every step writes:
- **The run log** (`RunLog`): command, commit, versions, inputs and outputs with hashes, steps
  and times.
- **The resolved configuration** (`config.yaml`): mode, budgets, search spaces, seeds, features
  per target and horizon, data hashes.
- **Training logs:**
  - XGBoost: train and validation RMSE per boosting round, the best iteration;
  - neural models: train and validation loss, learning rate and time per epoch, the
    early-stopping epoch, the best checkpoint, peak GPU memory;
  - Optuna: every trial (parameters, value, state, duration), the study in SQLite, the
    optimisation history and the parameter importances.
- **Figures:** learning curves and optimisation histories.
- **Predictions** in the contract of §6.10 and the metrics of the evaluation layer.
- **Automatic warnings** in the report:
  - best epoch at the first or the last epoch;
  - NaN losses;
  - a large gap between train and validation loss;
  - the best trial on the edge of the search space;
  - a seed spread above 0.15 pp (the acceptance rule of stage 10).

Layout: `outputs/runs/<run_id>/` with `run.json`, `env.json`, `config.yaml` and one folder per
step; the prediction files also go to the paths of §6.10 for the evaluation. Checkpoints in
`models/` (not in git).

### 3.6 Safety rules enforced by the code

- In `smoke` and `dev` the loaders refuse test rows (as the stage-7 loader does now).
- `final` refuses to start off the frozen tag; every test evaluation is written to
  `test_access_log.csv`.
- Atomic writes: a failed step never leaves a file that looks complete.
- Memory is released between models; seeds are fixed and recorded.

## 4. The models (from §6.11, with what this plan must still fix)

### 4.1 Specified

- **XGB-Opt:** Optuna TPE, 60 trials (dev: 20), objective daily nRMSE on validation; the search
  space of §6.11; early stopping 50 rounds; `hist`; three seeds; two targets × three horizons;
  features: the 05e sets (D-13); indirect trained on valid k only; no refit (D-10).
- **BiLSTM-Opt:** encoder-decoder, quantile loss (0.1, 0.5, 0.9); Optuna 60 trials × 25 epochs;
  final 60 epochs, patience 10, three seeds.
- **TFT-96, TFT-168:** pytorch-forecasting 1.7.0; decoder 72 h; 20 trials per target; attention
  and study kept. **TFT baseline:** encoder 168, hidden 64, dropout 0.1, up to 150 epochs.
- **PatchTST:** neuralforecast 3.1.7; patch length 16, stride 8; target only; daily origins.

### 4.2 To fix in this plan before the code (the input contract)

- **Encoder:** measured series that ended by T0 (power, k, irradiance, temperatures), with the
  DST guard of stage 6.
- **Decoder:** only what is known at T0: calendar, clear sky and GFS (admitted, D-12). The list
  of the 18 known reals of the TFT is fixed here.
- **Mapping of decoder steps to horizons** by the local day of the valid time, not by the step
  number (D-16, T-02): a 23- or 25-hour day must not shift D+2 and D+3.
- **Which features** the sequence models see (L-12: per model family).
- **D-15 (monotone constraints)** for XGB on k: decided on validation in the dev run.

### 4.3 Development budgets of the neural models (proposal)

Smoke: 1 trial, 2 epochs. Dev: 5 trials × 10 epochs, one seed. Final: the budgets of D-08.

## 5. The environment on the college machine

- `environment-gpu.yml`: the CPU environment plus PyTorch built for the machine's CUDA,
  pytorch-forecasting 1.7.0, neuralforecast 3.1.7 and lightning, pinned; `requirements-gpu.txt`
  corrected.
- `check` prints every version and fails on a mismatch.
- XGBoost stays on the CPU with `hist`, as in v6 (§6.11).

## 6. Moving code, data and results

- **Code:** a private GitHub repository (an open item of stage 0) and a `git clone` of the commit
  to run; otherwise a zip of one commit.
- **Data:** `python -m solarpv.pipeline bundle` packs only what the runner reads (the stage-6 and
  stage-6b tables, the configs, the stage-7 sets) with a SHA-256 manifest; the raw data stay on
  the laptop. The true site coordinates (`site.private.yaml`) go only with the user's approval
  (Q-16).
- **Results:** `collect` packs the run; it is unpacked on the laptop into `outputs/runs/`, where
  the notebooks and the record are updated.

## 7. Order of the work

| # | Work | Verified by |
|---|---|---|
| W1 | Runner: steps, selection of steps, modes, logs, `env`, `data`, `bundle`, `collect`, `status`; the `baselines` and `evaluate` steps (no training) | unit tests; `check` and `run --only baselines,evaluate` locally |
| W2 | `xgb` with its training logs | unit tests; smoke |
| W3 | Datasets of the sequence models: windows, the decoder mapping, the known reals | T-02 and unit tests on synthetic series, including DST days |
| W4 | `bilstm`, `tft96`, `tft168`, `tftbase`, `patchtst` | smoke on the laptop (CPU) |
| W5 | `select`, `explain`, `report`, the final gate, the runbook, notebook `06_models.ipynb` | unit tests; smoke of the whole sequence |
| W6 | Hand-off: a `dev` run on the college machine, review (`forecasting-validity-review`), then gate G-1, the tag, and the `final` run | the run's report |

## 8. Questions for the user

- **Q-15, the college machine:** operating system; GPU model, memory and CUDA version; CPU cores
  and RAM; free disk; internet access (pip, conda, GitHub); a scheduler such as SLURM or a
  direct session; the longest allowed job; how you connect to it.
- **Q-16, transfer:** a private GitHub repository? May the true site coordinates be copied there?
- **Q-17, stage 7:** the fix of the direct-route selection (open since R7-03). It becomes the
  `select` step on the college machine.
- **Q-18:** smoke runs on the laptop (minutes, a few weeks of data, one trial), only to prove the
  code runs before it is shipped?

## 9. Tests (IDs continue PROJECT_MASTER.md §8.2)

To be numbered when the code is written: step selection (`--only`, `--from`/`--to`), `--resume`
and `--force`, the final gate refusing an untagged or dirty tree, `dev` never reading test rows,
atomic outputs, the prediction contract of every model, T-02 for every sequence model.

**Numbered for W1 (2026-11-05):**

| ID | Test | File |
|---|---|---|
| T-51 | Step selection (`--only`, `--from`/`--to`, unknown and unimplemented steps), the three modes and their budgets, the smoke window, `--dry-run` runs nothing | `tests/unit/test_pipeline_runner.py` |
| T-52 | Safety: smoke and dev load no test row; a prediction row in the test split fails a dev run; `final` refuses an untagged or dirty tree; `--allow-test` is refused outside `final` | `tests/unit/test_pipeline_runner.py` |
| T-53 | Records: one folder per run, never overwritten; a step is complete only with its `step.json`; a failed step keeps its log and no completion marker; `--resume`, `--force`, `--keep-going`; `evaluate` reads only the run's own files, checked by hash; `check` compares every input with its stage manifest; nothing is written outside the run folder | `tests/unit/test_pipeline_runner.py` |
| T-54 | Fixes to earlier code (X1–X6 of §10): duplicates across prediction files, no stale manifest after a failed evaluation, no glob in `evaluate.main`, run-log names, untracked code in `git_commit()` | `tests/unit/test_evaluation_8b.py`, `test_runlog.py`, `test_provenance.py` |
| T-55 | Real data: a dev run of `baselines,evaluate` equals the committed stage-8B validation tables without loading a test row; the smoke window is fully covered and contains the spring DST day | `tests/contracts/test_pipeline_output.py` |

## 10. Self-review before the code of W1 (2026-11-05)

Written before the runner code, at the start of the work on the college machine. It covers the
original project's orchestration, modelling and evaluation scripts that the runner replaces
(`submission_evidence/12_scripts/`), and my own earlier code that the runner builds on. The
evaluation scripts themselves were reviewed in stage 8B (E1–E11, A1–A6 of
`docs/stage8b_plan.md`); those findings are not repeated. Every finding below has an errata row,
a test and a run-journal note (R9-01).

### 10.1 The original project (M1–M6)

| ID | Mistake | Where | Fix in the runner | Test | Errata |
|---|---|---|---|---|---|
| M1 | A failed training step is collected in a list and the sequence goes on: the evaluation then scores whatever prediction files are in `outputs/`, including those of earlier runs, and only the exit code shows the failure. The evaluation is not tied to the predictions of the run | `reproduce_paper_results.py` (phases 1–2), `align_and_evaluate.py` (a missing file is skipped) | `evaluate` reads only the prediction files written by the completed steps of the same run, each checked against the hash in its step record. A failed step stops the run; with `--keep-going` the model is reported as missing and never replaced by an older file | T-53 | C-36 |
| M2 | After a run without failures the script prints "Expected paper results" (29.09 %, 12.06 %, 30.48 %, 12.60 %, "indirect outperforms direct by 1.7–2.1 pp") from constants in the code, whatever the run produced | `reproduce_paper_results.py`, end of `main()` | The run summary is read from the run's own output tables; no result number lives in the runner code | T-53 | C-37 |
| M3 | No history and no provenance: each step log is opened with `"w"` in the fixed `outputs/reproduction_logs/`, so every reproduction overwrites the previous logs; no commit, versions, configuration or input hashes are recorded; the outputs go to fixed folders and are overwritten too (`align_and_evaluate.py` copies over the canonical file) | `reproduce_paper_results.py` (`run_step`), `align_and_evaluate.py` | One folder per run (`outputs/runs/<run_id>/`) with `run.json`, `env.json`, `config.yaml` and one folder per step; a step rerun with `--force` moves the earlier one aside instead of deleting it | T-53 | C-38 |
| M4 | Silent fallbacks: the ablation uses default hyper-parameters when the trained model file is missing, the seed table drops a missing seed file with a printed note, and the prerequisite check only tests that the feature parquet exists, not which version it is | `run_ablation_study.py` (`load_saved_hparams`), `compile_seed_variability.py`, `reproduce_paper_results.py` | `check` compares every input with the hash recorded by the stage that produced it (or with git for tracked files); a missing or changed input is an error, never a default | T-53 | C-39 |
| M5 | No separation of development and final runs: the reproduction evaluates the test split every time, and the `--quick` "smoke test" of the ablation trains on train and scores on the test split | `reproduce_paper_results.py`, `run_ablation_study.py --quick` | Modes `smoke`, `dev` and `final`. In smoke and dev the loaders exclude test rows before they reach pandas, and a prediction row in the test split is an error. The test split is evaluated only in `final`, on the commit tagged `g1-frozen` with a clean tree, with `--allow-test`, and logged | T-52 | C-40 |
| M6 | Duplicated forecasts were removed silently: `drop_duplicates(keep="last")` after a sort by timestamp only (not stable), so which of several forecasts of the same hour was scored is arbitrary, and the alignment bug behind the duplicates was hidden | `align_and_evaluate.py` (`align_lstm`) | Duplicates are an error, within a file (8B) and now across the files of a run (X2) | T-54 | C-41 |

### 10.2 My own earlier code (X1–X8)

| ID | Problem | Where | Fix | Test |
|---|---|---|---|---|
| X1 | `evaluate.main()` picks the prediction files by a glob, `outputs/06_*/predictions_*.csv`: a stale or smoke file would be scored next to the current ones (M1 in my own code) | `solarpv/evaluation/evaluate.py` | The files are named explicitly (`--predictions`); without them only the references are scored. The runner passes the files of the run | T-54 |
| X2 | Duplicates are checked inside one prediction file only. Two files of the same model and seed are both scored: on synthetic data the benchmark n went from 54 to 108 (the intervals shrink), and the run failed later in the DM table with an unrelated shape error. The references computed by `evaluate.run` would also be counted twice if the runner passes them as files | `evaluate.run` | Duplicates across files are an error that names the files; `with_references=False` when the references come as files | T-54 |
| X3 | `load_truth()` reads every row, test included, in every mode; the split is chosen afterwards. "Dev mode never reads test rows" was not true at the reader | `solarpv/evaluation/truth.py` | `load_truth(exclude_splits=("test",))` filters the rows inside the parquet reader; the runner uses it in smoke and dev and asserts that no test row is left | T-52, T-55 |
| X4 | `evaluate.run` writes its tables one by one into an existing folder and the manifest last: a failure in the middle leaves new tables next to the previous run's manifest, a folder that looks complete and consistent | `evaluate.run` | The old manifest is removed before the first table is written; the runner writes every step into a temporary folder, renamed only on success, and a step is complete only with its `step.json` | T-53, T-54 |
| X5 | `RunLog` names its file by the UTC second and opens it with `"w"`: two runs that start in the same second in the same folder share one file, and the second erases the first (reproduced with a fixed clock: one file, the first run's note lost). A step that fails writes no end line | `solarpv/config/runlog.py` | Exclusive creation with a numbered suffix, an optional explicit path, and a "failed after" line for a failing step | T-54 |
| X6 | `git_commit()` ignores untracked files (`--untracked-files=no`): a new module in `src/` that is not committed runs while the manifest records a clean commit | `solarpv/config/provenance.py` | Untracked, not ignored files under the code paths make the commit `+dirty` | T-54 |
| X7 | `evaluate.run` keeps only the rows labelled with the evaluated split: a dev prediction file with test rows is filtered silently, although a model that predicted test hours has read test features | `evaluate.run` | In smoke and dev the runner refuses any prediction row whose timestamp is in the test split, whatever its label | T-52 |
| X8 | This plan (§3.5) said the prediction files "also go to the paths of §6.10": a smoke or dev run would overwrite the canonical `outputs/06_<model>/` files | §3.5 above | Smoke and dev write only inside their run folder. Copying a final run to the §6.10 paths is a separate, explicit step (W5/W6) | T-53 |

### 10.3 Design decisions of W1 that follow from the review

- **Steps and preflight.** `env` and `data` run first in every `run` (they are what `check`
  prints); the work steps are `select`, `baselines`, the six model steps, `evaluate`, `explain`
  and `report`, in that order. W1 implements `baselines` and `evaluate`; naming another step is an
  error that says which work item brings it.
- **The evaluation window.** The coverage rule of stage 8B (every model on every daytime hour of
  the split) holds inside a window: dev and final use the whole validation split; smoke uses the
  28 local days 2024-03-15 to 2024-04-11, which contain the spring DST change of 2024-03-29.
  Rows of the split before the window stay as context for the persistence reference; rows after
  it are dropped. Every model and every reference is scored on the same window.
- **References as prediction files.** The `baselines` step writes the three references in the
  file contract (§6.10), so they pass the same alignment and coverage checks as a model, and
  `evaluate` scores them from those files (`with_references=False`).
- **Completion and resume.** A step's fingerprint is the hash of its mode, resolved
  configuration, code commit and input hashes. `--resume <run_id>` skips a completed step with
  the same fingerprint, refuses a different one unless `--force`, and never trusts a `+dirty`
  commit.
- **Final gate.** `final` starts only when `HEAD` carries the tag `g1-frozen` and
  `git status --porcelain` is empty; the site file must be the private one.
- **What goes into git.** Run folders are committed for their logs, records and tables; the
  prediction CSVs of `outputs/runs/` are ignored (about 10 MB per dev run of references alone).

## 11. Changes after the move to the college machine (proposed 2026-11-05, waiting for the user)

The plan was written on the laptop and assumed that the training would be shipped to the college
machine. The laptop was only a bootstrap and all work now happens here (`CLAUDE.md`,
"Machines"). Nothing below has been dropped yet; W1 implements none of the items marked "drop".

| Where | Text now | Proposal | Why |
|---|---|---|---|
| §1, "Feature selection" | one decision is open (R7-03) | Update: stage 7 closed (R7-04, R7-05); the input is `model_inputs.json` | Fact changed on 2026-11-03 |
| §2 | the laptop row (code, smoke runs only, Q-18) | Drop the laptop row; one line: everything runs on the college machine | The laptop is no longer used |
| §3.1 `bundle` | packs the processed data on the laptop with a manifest | Drop. `check` verifies the inputs against the hashes of the stage manifests that produced them (implemented in W1) | The data are already here; the stage manifests are the better reference |
| §3.1 `collect` | packs a run to bring back to the laptop | Drop as a transfer. If you want a copy of the runs outside the VDI (its persistence after logoff is unknown, §1.1), an `archive` command could pack a run with its hashes (W5) | Nothing is carried back, but a backup may still be needed |
| §3.2 `data` | verifies the bundle | Verifies the inputs against the stage manifests | Follows from `bundle` |
| §6 | code by GitHub or zip, data by `bundle`, results by `collect` | Replace with: code in the local git repository (no push without your request); data and results stay on this machine; backups per §5.3 | Nothing moves between machines |
| §7, W1 | `bundle`, `collect`; verified "locally" | W1 without `bundle` and `collect`; verified on the college machine | As above |
| §7, W4 | smoke "on the laptop (CPU)" | Smoke on the college machine, on the GPU environment of §5 once you approve it | The laptop is not used |
| §7, W6 | "Hand-off: a dev run on the college machine" | "Dev run, review, G-1, tag, final run", all here | No hand-off left |
| §8, Q-16 | may `site.private.yaml` be copied to the college machine? | Close: you copied it (RH-02), and every run here records `site_private: true` | Answered by your action |
| §8, Q-18 | smoke runs on the laptop | Already moot (3.11); remove from the open list | — |
| `docs/HANDOFF.md` §7 | bringing the work back to the laptop | Keep as history, marked "not planned" | The laptop is not used |

## 12. Literature ideas for stage 9 (read on 2026-11-05)

From `PROJECT_MASTER.md` §16b and `docs/literature_review.md` ("Stages 9–10").

- **L-09, bias correction (MOS) of the final forecast.** Model level, so it cannot be tested
  before W2 (the references are not models). Registered now, before any model result: in W2 a
  linear correction of the final kW forecast of XGB-dev is fitted on out-of-fold train
  predictions (the date folds of stage 7), applied to validation, and adopted only if it beats the
  D-12 margin, max(0.05 pp, 2 × pooled seed std), with three seeds. Otherwise it is reported as
  tested and not adopted.
- **L-17 and L-18** (more light models, blending): outside the approved scope (25.10); not
  tested.
- **L-19:** decided (PatchTST univariate, as in v6); the smoke test of W4 records whether
  neuralforecast's PatchTST uses future exogenous inputs, for the report only.
- **L-13 and L-16** (CLIPER skill, nRMSE by the mean): already in the evaluation layer, which
  the `evaluate` step reuses unchanged; no new decision.
- **L-12:** the input of the sequence models is fixed in W3.
- No idea is adopted in W1, so nothing is tested on validation in this work item.
