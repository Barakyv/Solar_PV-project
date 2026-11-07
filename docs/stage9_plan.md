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
| Feature selection | Stage 7 closed on 2026-11-03 (R7-04, R7-05). The input of every model is `outputs/07_feature_selection/model_inputs.json`: the 05e sets for the indirect route, the full pool for the direct route (F-29). *Updated 2026-11-05 (§11, approved).* At writing (2026-11-01) one decision was open: the direct-route sets were too small because of the weak selection model (R7-03) |
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

Everything runs on the college machine: the code, the unit tests, the documents and notebooks,
the smoke, dev and final runs, the test evaluation, SHAP, attention and the ablation.

*History (§11, approved 2026-11-05).* The plan of 2026-11-01 split the work: the laptop wrote the
code and made smoke runs only (Q-18), the college machine ran everything heavy. On 2026-11-03 the
user moved the work to the college machine; the laptop was only a bootstrap and is not used, and
the one transfer was the package of `scripts/handoff_pack.py` (`docs/HANDOFF.md`).

## 3. The runner

### 3.1 One entry point

`python -m solarpv.pipeline <command>`

| Command | Purpose |
|---|---|
| `list` | The steps, in order, with what each needs and produces |
| `check` | Environment (Python, packages, CUDA, GPU, CPU, memory, disk), data (every input against the SHA-256 recorded by the stage manifest that produced it, and git-tracked inputs unmodified), gates (mode, git commit, tag, clean tree). Nothing is trained |
| `run` | Runs the selected steps in order (§3.3) |
| `status` | What has completed, with which code, configuration and data |

*Dropped (§11, approved 2026-11-05):* `bundle` (packed the processed data on the laptop) and
`collect` (packed a run to bring back to the laptop): nothing moves between machines, and the
stage manifests are a better reference than a bundle manifest. An optional `archive` command, to
pack a run with its hashes as a backup off the VDI (whose persistence after logoff is unknown,
§1.1), stays an idea for W5.

### 3.2 The steps

| Step | Stage | What it does | Device |
|---|---|---|---|
| `env` | — | Environment report | — |
| `data` | — | Verifies the inputs against the stage manifests (and git) | — |
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
step. Smoke and dev runs write only inside their run folder (X8, §10.2); copying a final run to
the paths of §6.10 is a separate, explicit step (W5/W6). Checkpoints in `models/` (not in git).

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

*Fixed in §17 (W3, 2026-11-07), with the open choices proposed as D-25–D-28.*

### 4.3 Development budgets of the neural models (proposal)

Smoke: 1 trial, 2 epochs. Dev: 5 trials × 10 epochs, one seed. Final: the budgets of D-08.

**D-17 approved on 2026-11-05 (user) for XGB only:** the dev budget is 20 Optuna trials with
seed 42; 60 trials and the three seeds only in the final run (stage 10). The neural dev budgets
above stay a proposal, to be decided in W4.

## 5. The environment on the college machine

- `environment-gpu.yml`: the CPU environment plus PyTorch built for the machine's CUDA,
  pytorch-forecasting 1.7.0, neuralforecast 3.1.7 and lightning, pinned; `requirements-gpu.txt`
  corrected.
- `check` prints every version and fails on a mismatch.
- XGBoost stays on the CPU with `hist`, as in v6 (§6.11).

## 6. Code, data and results

- **Code:** the local git repository on the college machine; no push without the user's request.
- **Data and results:** stay on this machine. Backups per `PROJECT_MASTER.md` §5.3.
- *Replaced (§11, approved 2026-11-05):* code by a GitHub clone or a zip, data by `bundle`,
  results by `collect` back to the laptop, and the site file only with approval (Q-16, closed:
  the user copied it, RH-02).

## 7. Order of the work

| # | Work | Verified by |
|---|---|---|
| W1 | Runner: steps, selection of steps, modes, logs, `env`, `data`, `status`; the `baselines` and `evaluate` steps (no training) | unit tests; `check` and `run --only baselines,evaluate` on the college machine |
| W2 | `xgb` with its training logs | unit tests; smoke |
| W3 | Datasets of the sequence models: windows, the decoder mapping, the known reals | T-02 and unit tests on synthetic series, including DST days |
| W4 | `bilstm`, `tft96`, `tft168`, `tftbase`, `patchtst` | smoke on the college machine, on the GPU environment of §5 once the user approves it |
| W5 | `select`, `explain`, `report`, the final gate, the runbook, notebook `06_models.ipynb`; optionally `archive` (§3.1) | unit tests; smoke of the whole sequence |
| W6 | Dev run, review (`forecasting-validity-review`), gate G-1, the tag, and the `final` run, all on the college machine | the run's report |

*Wording of W1, W4 and W6 changed by §11 (approved 2026-11-05): W1 was "with `bundle`, `collect`,
verified locally", W4 "smoke on the laptop (CPU)", W6 "hand-off: a dev run on the college
machine".*

**Status (2026-11-05):** W1 done (`solarpv.pipeline`, commit `c70af25` and the records commit
after it; run journal R9-01, R9-02). `bundle` and `collect` were dropped (§11, approved).
`check`, `run --only baselines,evaluate` (smoke and dev), `--resume` and `status` work on the
college machine; the dev run equals the stage-8B validation tables.

**Status (2026-11-07):** W2 done except one follow-up. The `xgb` step, `pipeline.data`, the
stage-7 provenance check and `python -m solarpv.models.xgb_checks` (commit `a3b8467`; T-56–T-63;
run journal R9-03). D-15 decided by §14.3: the v6 constraint stays for k (R9-04). Smoke passes
(R9-05). The dev run (20 trials, seed 42) passes the acceptance of §14.7, with D+3 below D+1
explained as the validation sample (R9-06). L-09 by §14.4: adopted for the indirect route, not
for the direct route; R-14 confirmed for the direct target at D+1 and D+3; A-14 ✅ for XGB
(R9-07). The adopted L-09 correction is part of the `xgb` step since commit `9c5d1b9` (§15,
OP-53, R9-08); the dev run with it reproduces the check exactly (R9-09). The correction made
the k-unit DM and skill of the indirect route worse; by the user's decision on Q-19 the
evaluation also computes DM and skill in kW for both routes, and the kW DM is the table of the
stage-10 acceptance (§16, OP-54, commit `f529913`, R9-10, R9-11).

## 8. Questions for the user

Status on 2026-11-05 (§11, approved):

- **Q-15, the college machine:** answered by the inspection of 2026-11-03 (§1.1). Still unknown:
  the longest allowed job and whether the desktop survives logoff (to ask the college IT before
  any long run).
- **Q-16, transfer and coordinates:** closed. The user copied `configs/site.private.yaml` to the
  college machine (RH-02), and every run here records `site_private: true`. No GitHub repository
  is needed: nothing moves between machines (§6).
- **Q-17, stage 7:** answered on 2026-11-03 (R7-04): the fixed XGBoost of D-11 in both routes.
- **Q-18:** moot since 2026-11-03 (the laptop is not used); removed from the open list.

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

**Numbered for W2 (2026-11-07),** all in `tests/unit/test_xgb.py` on synthetic tables in the
stage-6 and stage-6b layout (and `test_pipeline_runner.py` for the refusal of unimplemented
steps):

| ID | Test |
|---|---|
| T-56 | Inputs from the allow-list (a lag not available at the horizon, a duplicate and a leaky name are refused; the order is kept); monotone constraints by exact name (`fcst_gfs_ghi`, `irradiance_lag_3d`, `irradiance_clear_index_prevday` never constrained); the map reaches XGBoost (`(1,0,0)`) and the prediction is monotone in the constrained feature; the D-15 decision file is required and checked (M7, M12a) |
| T-57 | Fit, early-stopping and scoring rows (warm-up, op-limited, `k_valid`, theo > 50 kW, AC present, the window, op-limited hours kept in the objective, nights in the direct fit rows, the 56 smoke days); no imputation: NaN reaches XGBoost and is counted, and the prediction differs from a zero fill (M8, M11) |
| T-58 | The k → kW copies agree (negative k, theo 0, above capacity); `nwp.check.fit_score` equals the same model scored with `evaluate.to_kw`; in a smoke run of `baselines,xgb,evaluate` the recorded objective equals the evaluation's `nRMSE_kW_pct` (and `nRMSE_pct` in k) of every target × horizon, and the refit equals its trial (M9, X12) |
| T-59 | The record of every fit: full parameter set (seed, threads, `hist`, constraints), feature order, best iteration, rounds used = best + 1; a reloaded booster reproduces the written predictions exactly, with no clip; the same seed gives the same study, another seed another one; the warnings at the cap and on the edge of the search space (M10, M12c, M12d) |
| T-60 | `model_table` filters inside the parquet reader (by split, and the GFS table by timestamp) and refuses the test split without `allow_test`; a smoke run reads no test row; the stage-7 provenance rows of `check` (match, changed table, missing log, no run log); a pending decision is reported without blocking; the `xgb` step refuses without the D-15 decision (X10, X11) |
| T-61 | A-14: permuted validation labels change only the stopping round, never the trees; a fixed-round fit equals the early-stopping fit truncated at the same round; the L-09 coefficients do not change when the validation labels are permuted |
| T-62 | X13: `decide_k.XGB_PARAMS` is unchanged after the D-15 check, which writes a decision file the `xgb` step can read and no prediction file |
| T-63 | The checks of §14: the margin and the registered rule (D-12 form), the D-15 decision on constructed scores, the output guard (no `predictions_*`, nothing into a run folder), R-14's bias and interval on a constructed bias, and L-09 and R-14 on a synthetic dev run (seed 42 with the dev rounds equals the dev forecast; the run folder is unchanged) |
| T-64 | The adopted L-09 correction in the step (§15, X14–X16): the step's coefficients and corrected value equal the check's for the same base model, and the evaluation scores the corrected value; the check and the step share one implementation and the same chronological folds (smoke: its 56 days); `evaluate.to_kw(k')` equals the corrected kW, nights and the direct target unchanged, a daytime hour with theo ≤ 0 refused; the uncorrected file holds the booster output, and the written file is the booster plus the recorded coefficients; without the decision file a smoke run writes the raw forecast with a warning and a final run refuses; the decision file is checked |
| T-65 | Q-19 (§16, X17–X18), in `tests/unit/test_evaluation_8b.py`: the kW DM and skill of the direct target equal the target-unit ones exactly; for the indirect target the kW DM and skill equal a hand computation with `to_kw` and actual AC, and differ from the k-unit ones; the fixture mode writes no kW table; `main` on validation reads the truth without the test split |

**Numbered for W3 (2026-11-07),** in `tests/unit/test_sequence_data.py` on synthetic tables in
the stage-5, stage-6 and stage-6b layout (2023-09-01 to 2024-10-07 local, with the fall-back day
2023-10-29 in train, the spring-forward day 2024-03-29 in validation and one week of test rows),
and `tests/contracts/test_sequence_alignment.py` on the real tables (§17):

| ID | Test |
|---|---|
| T-66 | The mapping (M13, M14, X20): T0 at local midnight (22:00Z in winter, 21:00Z in summer); decoders of 71 / 72 / 73 steps for the origins whose three days hold 2024-03-29 or 2024-10-27; every D+h block equals the hours of its local day (23 / 24 / 25), checked against the time zone itself; `issuance_utc(valid, h) == T0` on every step from March to November 2024; a mapping by step number is wrong on both DST days and right on an ordinary one |
| T-67 | Origins and coverage (M15, X21, X23): the counts of §17.5 on the real calendar (872 training, 246 validation, 30 smoke, 245 test origins); in train and validation, both targets, every hour of the scope exactly once per horizon, with a 25-hour block in train and a 23-hour block in validation; development reads every file with `timestamp_utc < TEST_START` inside the reader (eight reads) and carries no time at or after it; training decoders end before VAL_START and the loss holds train hours after the warm-up only; 58 smoke training origins; test windows only with `allow_test`; the loss mask equals `fit_mask` row for row and count for count |
| T-68 | The encoder (M16, X22): its last hour is T0 − 1 h for every model; changing the measured series, the targets and the minute flags at or after an origin's T0 leaves every encoder and decoder input of that and every earlier origin unchanged; an hour stopped by the minute flags is NaN before the fill, filled from older days and flagged; the masked series equals the stage-6 `lag_1d` (0 mismatches), while the unmasked AC does not; the fill uses older hours only |
| T-69 | A-14 (M17): the scaler recomputed from the train hours and unchanged when every value outside train is replaced; with validation values replaced the training windows are identical; k is not transformed and AC / 3,449 inverts exactly |
| T-70 | The writer (M18): a round trip in the contract of §6.10 passes `load_predictions` and the strict alignment and coverage checks of `evaluate`; the 23-hour day keeps 23 rows at D+1; a wrong shape, a missing hour, a duplicated (horizon, timestamp) and a missing forecast are refused |
| T-71 | The inputs (D-26–D-28): the union of the known part of `model_inputs.json` in its order, lags dropped, a leaky or unknown name refused; on the real file 10 + 20 (indirect) and 27 + 22 (direct); a deterministic input that differs between tables is refused; each decoder step's GFS value is that of its horizon's table and each encoder hour's that of D+1; a step of another issuance or a run not public by T0 is refused; PatchTST holds the target only and needs its input size; the command passes all its checks on the synthetic inputs and refuses `final` |
| T-02 | Real tables (`tests/contracts/test_sequence_alignment.py`): train and validation, both targets, BiLSTM, TFT-96, TFT-168 and PatchTST: every check of `check_dataset`, 872 / 246 origins, 23- (and in train 25-) hour blocks, unique valid timestamps per horizon, training decoders before VAL_START, nothing at or after TEST_START |

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

### 10.2 My own earlier code (X1–X8), and one finding of the first runs (X9)

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
| X9 | Found by the first real runs (R9-02), after the code: the clean-tree check counted the untracked folders of earlier runs, so the final gate would refuse after any smoke run | `pipeline/checks.py` | A clean tree: no modified tracked file and no untracked code; untracked outputs are reported only | T-52 |

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

## 11. Changes after the move to the college machine (proposed 2026-11-05; approved and applied 2026-11-05)

**Approved by the user on 2026-11-05: all the proposals below. Applied the same day** in §1,
§2, §3.1, §3.2, §3.5 (the X8 fix), §6, §7 (W1, W4, W6) and §8 (Q-15 to Q-18), and in
`docs/HANDOFF.md` (§7 and the Q-16 item marked not planned / closed). `bundle` and `collect` are
dropped; an optional `archive` for backups off the VDI stays an idea for W5. The table is kept as
the record of what changed.

The plan was written on the laptop and assumed that the training would be shipped to the college
machine. The laptop was only a bootstrap and all work now happens here (`CLAUDE.md`,
"Machines").

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

## 13. W2: blocking decisions and self-review before the code (2026-11-05)

Written before any code of the `xgb` step and before any XGB-Opt result.

### 13.1 Blocking decisions (`PROJECT_MASTER.md` §3)

None is open for W2. D-08 (scope), D-10 (no refit), D-11 (k), D-12 (GFS), D-13 with R7-04
(the inputs, `model_inputs.json`), D-22 (op-limited hours out of the loss) and D-23/D-24 are
decided; D-17 was approved on 2026-11-05 (20 trials, seed 42). D-15 is not a blocker: it is the
question W2 decides, by the rule of §14 written now. D-14 (DM) and D-16 (T0) are "proposed" but
already implemented in stages 6 and 8 and not changed by W2; D-06 and D-07 concern the
environment and the repository location.

### 13.2 The original project (M7–M12)

Sources: `submission_evidence/12_scripts/run_ablation_study.py` (the only surviving XGB training
code), and v6 §3.6, §4.1 and §5.1 for the lost `xgb_sota.py`.

| ID | Mistake | Where | Fix in W2 | Test | Errata |
|---|---|---|---|---|---|
| M7 | Monotone constraints by substring: +1 on every column whose name contains `irradiance`, `ghi`, `fcst_ghi`, `fcst_dni`, `fcst_dhi`, `poa_global`, `ghi_clearsky`, `sun_elevation` …, for both targets. The report names three features; the code also constrains lags and rolling means of the measured irradiance, and in this rebuild's names it would hit `fcst_gfs_ghi`, `irradiance_lag_3d` and `irradiance_clear_index_prevday`. On the k target the marginal sign in train is negative (C-33) | `build_monotone` | The explicit list of D-15 (`ghi_clearsky`, `poa_global_clearsky`, `sun_elevation`), matched by exact name and passed to XGBoost as a name → sign map; the map of every target × horizon is recorded; for k it follows the registered D-15 rule (§14.3). In the inputs of stage 7 only `ghi_clearsky` is in the indirect sets (all three horizons), and all three are in the direct full pool | T-56 | C-20, C-33 |
| M8 | `fillna(0)` on the features of train and test: a missing GFS value or lag becomes a physical zero (0 W/m², 0 °C, 0 kW), the model cannot tell missing from zero, and XGBoost's own handling of missing values is bypassed | `X_train = ….fillna(0)` | No imputation in the model step: NaN goes to XGBoost, which learns a default branch per split. The NaN count of every feature is logged on the fit, early-stopping and predicted rows. On the real tables NaN is rare: `irradiance_clear_index_prevday` on 5–9 train scoring hours, and GFS ratios at night in the direct fit rows (about 9,600 of 20,393) | T-57 | C-42 |
| M9 | Post-processing scattered over scripts and not in the prediction files: the ablation clips every prediction at 0 before scoring (both targets), `compare_routes.py` clips kW at [0, capacity], FINDINGS describes a clip at [0, capacity]; with the training target clipped at 1.2 (F-06), the score of a forecast depended on the script that scored it | `np.clip(model.predict(X_test), 0, None)` | The prediction file holds the raw model output (k is never clipped, F-06). The only post-processing is the evaluation layer's `evaluate.to_kw` (indirect: clip(k × theo, 0, capacity); direct: as is), the same for every model. The Optuna objective calls that same function, so the tuning score equals the evaluation's `nRMSE_kW_pct` | T-58 | C-43 |
| M10 | Saved hyper-parameters were not the model: `load_saved_hparams` set `n_estimators` to `num_boosted_rounds()` of the saved booster, which after early stopping holds the best iteration plus the 50 rounds of patience; the constraints and the feature order were rebuilt by substring instead of read; a missing file gave defaults (C-39). The ablation "baseline" was therefore not the XGB-Opt it was compared with | `load_saved_hparams` | Per target × horizon × seed the step records the full parameter set (constraints, feature order, seed, threads), the best iteration, the rounds used (best + 1) and the best validation RMSE, and saves the booster. Predictions use `iteration_range=(0, best + 1)` explicitly; a reloaded booster reproduces the step's predictions exactly | T-59 | C-44 |
| M11 | Training rows: both targets were trained on every train hour, including night hours (where the old k was 0), invalid k, and stopped or curtailed hours, so the indirect model learned k = 0 at night and at stops | `train_mask = df.index <= TRAIN_END` | Fit masks: indirect, not warm-up, not op-limited (D-22), k valid and theo > 50 kW (L-04); direct, not warm-up, not op-limited, AC present (stage 7, D-12). The early-stopping rows are the validation rows with the same mask. Counts recorded | T-57 | C-45 |
| M12 | The lost `xgb_sota.py` as v6 describes it: (a) the features were the 05e set minus a block-list (v6 §3.6, "EXCL, lines 51–73"), while the SHAP file shows 76 inputs (C-31, F-05); (b) the same validation split chose the hyper-parameters ("daytime nRMSE on the validation set", §4.1) and the stopping round ("50 rounds on validation RMSE", §5.1) and was then reported, so validation numbers are optimistic by construction; the unit of the indirect objective (k or kW) is not stated; (c) "three independent runs (seeds 42, 123, 777)" does not say whether the seed drives the Optuna sampler, the XGBoost subsampling, or both; (d) `n_estimators` [200, 2000] is searched together with early stopping, so a trial can stop at its cap before converging, and no best iteration is reported | v6 §3.6, §4.1, §5.1 | (a) Allow-list from `model_inputs.json`, each feature checked against the stage-7 pool and `assert_no_leakage`, the order recorded. (b) D-10 keeps the design (no refit); every dev number is labelled as a tuning-split number, the test stays the only unbiased estimate; the objective unit is fixed in §14.1 and both units are logged per trial. (c) The seed drives both, `TPESampler(seed=S)` and `random_state=S`, recorded; the dev run has one seed (D-17); the D-15 and L-09 margins use fit-only seeds with fixed hyper-parameters. (d) The best iteration is logged and a warning is raised when it reaches the cap | T-56, T-59 | C-46 |

### 13.3 My own code that W2 builds on (X10–X13)

| ID | Problem | Where | Fix | Test |
|---|---|---|---|---|
| X10 | The stage-6b and stage-7 loaders read the whole parquet, test rows included, into pandas and filter afterwards. A model step built on them would break "smoke and dev never load a test row" (T-52, X3) | `nwp.check.load`, `selection.common.train_table` | A loader of the runner, `solarpv.pipeline.data.model_table(h, splits)`, filters inside the parquet reader: the feature table by `split`, the GFS table (no split column) by `timestamp_utc < TEST_START`; it asserts that no test row is left. The test rows of a final run are read only to be predicted, with `allow_test`. The stage loaders stay as they are (they are the record of their stages) | T-60 |
| X11 | `model_inputs.json` is checked only as an unmodified git file. Nothing ties it to the tables it was selected on: after a rerun of stage 6 or 6b, `check` would pass (new tables, new manifests) while the sets were chosen on the old tables | `pipeline/checks.py` | `check` adds one row per stage-7 input: the SHA-256 prefix recorded in the stage-7 run log named by `selection_manifest.json` must match the current table; a mismatch is an error | T-60 |
| X12 | The k → kW conversion exists in four copies (`evaluate.to_kw`, `selection.common.to_kw`, `nwp.check.fit_score`, `features.target.to_kw` used by `decide_k`). They agree today, but a fifth copy in the model step could drift from the evaluation | the modules named | The `xgb` step and the D-15 and L-09 checks call `evaluate.to_kw` only; a test asserts that the copies agree on random inputs (negative k, theo 0, values above capacity) | T-58 |
| X13 | The fixed XGBoost of D-11 (`decide_k.XGB_PARAMS`) is one mutable dict imported as the decision model of stages 6, 6b and 7. The D-15 check adds `monotone_constraints` to it: an in-place update would silently change every later decision model in the same process | `features/decide_k.py`, `selection/common.py` | Every use builds a new dict (`{**XGB_PARAMS, …}`); a test asserts the dict is unchanged after the D-15 check | T-62 |

### 13.4 Design of the `xgb` step that follows

- **Data:** `model_table(h)` per horizon, the inputs of `model_inputs.json` per target × horizon.
  Training rows: dev and final, all of train; smoke, the last 56 local days of train. Tuning and
  early-stopping rows: the evaluation window of the mode (smoke: the 28-day window of W1; dev and
  final: the whole validation split).
- **Search:** Optuna TPE, `TPESampler(seed=S)`, the space of §6.11, `tree_method="hist"`,
  `n_jobs = XGB_THREADS` (8), `eval_metric="rmse"`, early stopping 50 rounds on the validation
  rows; budgets per mode (smoke 1 trial, dev 20 trials, final 60 trials and three seeds). One
  study per target × horizon × seed, in `xgb/optuna.db` (SQLite, not in git).
- **Final fit:** the best trial's parameters, refitted with the same rows and early stopping (the
  same model the best trial trained; no refit on train+val, D-10).
- **Outputs:** `predictions_xgb_sota_seed<S>.csv` (every hour of the window, the contract of
  §6.10); `trials.csv` (every trial: parameters, value in kW, value in k for the indirect route,
  state, best iteration, duration); `learning_curves.csv` (train and validation RMSE per round of
  every final fit); `best_params.json`; `xgb_config.json` (search space, constraint maps, feature
  lists, row counts, NaN counts, seeds); the boosters in `models/` (not in git); two figures
  (learning curves, optimisation histories); the automatic warnings of §3.5.
- **The checks of §14** run as their own commands (`python -m solarpv.models.xgb_checks`), like
  `decide_k` and `nwp.check`: they write decision files in `configs/` and never write
  `predictions_*` into a run folder (the `evaluate` step scores every such file of a model step).

## 14. W2: decision rules, registered before any XGB result (2026-11-05)

Written and committed before the D-15 check, the smoke run and the dev run. Every decision is
taken on validation only; the test split is not read.

### 14.1 The tuning objective

Daytime nRMSE in kW (% of 3,449 kW) on the daytime hours of the evaluation window that have AC
truth, op-limited hours included (the primary definition of D-22); the indirect forecast is
converted with `evaluate.to_kw`. Why kW for both targets: one score drives every decision of the
rebuild (D-11, D-12, D-23, D-24); the route comparison and the use of the forecast are in kW; the
k-unit nRMSE gives full weight to low-sun hours where k is unstable and the power small (L-04).
v6 says "daytime nRMSE on the validation set" without a unit, so this is a choice within the
original plan. The k-unit nRMSE (RMSE of k / MEAN_DAYTIME_K) is logged for every trial, so the
other choice can be read from `trials.csv`.

### 14.2 Rows and early stopping

Fit rows as in M11; early stopping on the RMSE in the training unit (k or kW) over the
validation rows with the same mask, 50 rounds, the last evaluation set decides; no refit (D-10).

### 14.3 D-15, monotone constraints for k

- **Settings compared:** A, +1 on the D-15 features present in the indirect input (`ghi_clearsky`
  at D+1, D+2, D+3), as v6 §5.1 describes; B, no constraint.
- **Model:** the fixed XGBoost of D-11 (400 trees, depth 6, learning rate 0.05), seeds 42, 123 and
  777, trained on the indirect fit rows of train with the stage-7 inputs, scored by §14.1 on the
  whole validation split. A fixed model is used, as for every earlier decision, so the result does
  not depend on hyper-parameters tuned under one of the two settings.
- **Rule:** gain_h = mean_A − mean_B (positive: dropping helps), margin_h = max(0.05 pp, 2 × pooled
  seed std). B is adopted for k at every horizon only if gain_1 > margin_1 and gain_h > −margin_h
  at D+2 and D+3 (the D-12 form). Otherwise A, the v6 setting, stays: D-15 asks whether to keep
  the constraint, so dropping it is the change that needs evidence. Either way the result and the
  marginal sign (C-33) are reported.
- **Direct target:** the D-15 list with +1, as in v6 (physical: more irradiance, more power); not
  part of the question.
- **Output:** `configs/monotone_decision.yaml`, read by the `xgb` step.

### 14.4 L-09, bias correction (MOS) of the final kW forecast

- **Base model:** per target × horizon, the best parameters of the dev study, with a fixed number
  of rounds equal to the dev final fit's best iteration + 1 (no early stopping inside train);
  seeds 42, 123 and 777.
- **Correction:** out-of-fold predictions on train from the five expanding date folds of stage 7
  (`selection.common.date_folds`, 3-day gap), in kW (`evaluate.to_kw`); ordinary least squares
  y = a + b × ŷ on the out-of-fold daytime hours with AC truth, not op-limited and not warm-up
  (D-22: those hours are outside every loss); per seed.
- **Applied** to the validation forecast of the same base model trained on all of train:
  clip(a + b × ŷ, 0, capacity) on daytime hours; scored by §14.1.
- **Rule, per target:** gain_h = mean(uncorrected) − mean(corrected), margin_h as in §14.3;
  adopted for the target only if gain_1 > margin_1 and gain_h > −margin_h at D+2 and D+3.
  Otherwise reported as tested and not adopted.
- **Output:** `configs/mos_decision.yaml`. If adopted, the correction becomes part of the `xgb`
  step in a later commit, before G-1.
- *Result (2026-11-07, R9-07):* adopted for the indirect route (gain 0.11 / 0.22 / 0.09 pp
  against margins 0.05 / 0.18 / 0.09), not adopted for the direct route (worse at every
  horizon). In the step since `9c5d1b9` (§15, R9-09).

### 14.5 R-14, mean bias of the direct target on validation

Reported, not decided. The mean bias error (forecast − actual) of the dev XGB forecast on the
validation daytime hours with truth, in kW and in % of capacity, with a 95 % circular block
bootstrap interval (block 24 h, 1,000 resamples, seed 42, as §6.9), overall and per month; the
indirect route in kW alongside. R-14 predicts a positive bias for the direct target (output per
unit irradiance falls about 1 % a year, so a model trained to January 2024 forecasts too high). If
the interval excludes 0 with a positive sign, R-14 is confirmed and stays a declared limitation;
no trend feature is added (it would be a new idea with its own rule), and a level correction is
the business of L-09.

### 14.6 A-14, no validation information in a fitted transform

The `xgb` step fits no scaler and no imputer (M8); the boosters see train rows only, and
validation enters only as the early-stopping set and the Optuna objective (D-10, the v6 design).
Verified by tests: with a fixed number of rounds the predictions do not depend on the validation
labels, and the MOS coefficients do not depend on them either (T-61). Then A-14 is ✅ for XGB.

### 14.7 Acceptance of the dev run (`PROJECT_MASTER.md` §7, stage 9)

- XGB-dev has a lower `nRMSE_kW_pct` than persistence and climatology on validation for every
  target × horizon.
- The error is expected to rise with the horizon. If it is flat or falls, the inside-train order
  of R6b-07 and R7-05 is checked first (the validation sample showed D+3 below D+1 before), then
  T-09.
- The automatic warnings of §3.5 are read; the `forecasting-validity-review` skill is applied
  before any number is accepted.

## 15. W2 follow-up: the adopted L-09 correction in the `xgb` step (2026-11-07)

Written before the code. L-09 was decided by §14.4 (adopted for the indirect route, not for the
direct route; R9-07, `configs/mos_decision.yaml`). This section fixes how the step applies the
adopted correction. It changes no rule of §14: the correction is the registered one (out-of-fold
OLS on train, applied as clip(a + b × ŷ, 0, capacity) on daytime hours), and the base model is
tuned and stopped exactly as before.

### 15.1 Self-review before the code (X14–X16)

| ID | Problem | Where | Fix | Test |
|---|---|---|---|---|
| X14 | The correction would exist twice: the L-09 check fits it in `xgb_checks.mos_fit`, and the step would need its own copy of the folds, the rows, the OLS and the clip. Two copies can drift, as the four k → kW copies could (X12) | `models/xgb_checks.py` | One implementation in `models/xgb_sota.py` (the train scope, the rows, the folds, the fit and the application), used by the step and by the check; a test asserts that the step's coefficients and corrected score equal the check's for the same base model | T-64 |
| X15 | The indirect prediction file holds k, and the correction is in kW. Writing kW into the k file, or clipping k, would break the contract of §6.10 and F-06; mapping the corrected kW back to k needs theo > 0 on every corrected hour | the design | On daytime hours k' = clip(a + b × to_kw(k̂), 0, 3,449) / theo; night hours keep k̂ (they are not scored, and §14.4 corrects daytime only). The step refuses a daytime hour with theo ≤ 0 (none in train or validation: the smallest daytime theo with AC truth is about 62 kW). A test asserts that `evaluate.to_kw(k')` equals the corrected kW to 1e-9 | T-64 |
| X16 | Two readers assume that the written forecast is the raw booster output: the consistency check of `run_l09` (seed 42 with the dev rounds equals the dev forecast) and the reproduction test of a reloaded booster (T-59). After the change both would compare with the corrected file | `xgb_checks.run_l09`, `tests/unit/test_xgb.py` | The step also writes `uncorrected_xgb_sota_seed<S>.csv`, the raw booster output of every corrected target (not a `predictions_*` file, so `evaluate` does not score it); the check reads its base forecast from there; T-59 reproduces the uncorrected file from the booster, and the written file from the booster and the recorded coefficients | T-59, T-64 |

### 15.2 Design

- **Decision input:** `configs/mos_decision.yaml` becomes an input of the step (in its
  fingerprint) and a tracked input of `check`. Per target, `adopted: true` gives the corrected
  forecast, `false` the raw one. L-09 is decided on a dev run, so a smoke or dev run without the
  file is allowed: it writes the uncorrected forecast and a warning, and `check` reports the
  decision as pending. A final run refuses without it.
- **Refitted in every run:** the coefficients are fitted in each run for every target × horizon
  × seed whose target has the correction adopted, from that run's final fit: its searched
  parameters, seed, constraints and rounds (best iteration + 1), as fixed-round fits on the five
  expanding date folds of the run's training range (dev and final: all of train; smoke: its 56
  days); OLS on the held-out daytime hours with AC truth, not op-limited, not warm-up (§14.4).
  The base model of the final run (60 trials, three seeds) is another model than the dev one,
  and a correction fitted on another model's errors is not the registered method. With the same
  code as the check, a dev run that reproduces the dev study gives the coefficients of
  `outputs/09_xgb_checks/l09_check.csv` for seed 42.
- **Applied** to the final fit's forecast of every window of the run (validation; the test
  window of a final run), on daytime hours, written in the target's unit (X15).
- **Name:** the model stays `xgb_sota`: the adopted correction is part of XGB-Opt for the
  indirect route, and the evaluation, the route comparison and the DM tests see the forecast as
  adopted.
- **Records:** in `best_params.json` every entry gets `mos` (applied or not; a, b, the number of
  out-of-fold hours, the fit and held-out rows of every fold; the uncorrected and corrected value
  on the tuning window, in kW and for k in k units). `value_kw` keeps its meaning (the base
  model's objective, the value the refit must equal), and `value_kw_forecast` is the value of the
  written forecast, equal to the evaluation's `nRMSE_kW_pct` (T-58). Also `mos_coefficients.csv`,
  the decision and its hash in `xgb_config.json`, and both values in the run summary.
- **Not changed:** the search, the objective (the base model is tuned without the correction, as
  in §14.4), early stopping and the saved boosters.
- **A-14:** the coefficients come from out-of-fold train predictions only; no validation label
  enters (T-61 runs on the shared code). The hyper-parameters and the rounds came from
  validation, as for the base model itself (D-10).
- **Later steps:** `explain` (W5) explains the booster, the uncorrected forecast; the correction
  is a monotone linear map in kW and is reported with the SHAP values. A rerun of `r14` on a run
  with the correction reports the indirect route as written (corrected).
- **Cost:** five fixed-round fits per corrected target × horizon × seed (dev 3, final 9 of them,
  with the correction adopted for the indirect route only), measured in the dev run.

### 15.3 Verification

- **T-64 (synthetic):** the step's coefficients and corrected score equal the L-09 check's for
  the same base model; `evaluate.to_kw(k')` equals the corrected kW; night hours, the direct
  target and the uncorrected file are the raw booster output; without the decision file a dev
  run writes the raw forecast with a warning and a final run refuses; a daytime hour with
  theo ≤ 0 is refused; the decision file is checked (L-09, a boolean per target).
- **Real data:** smoke passes; a dev run (20 trials, seed 42) reproduces the dev study of R9-06,
  and its seed-42 coefficients and corrected validation values equal the seed-42 rows of
  `l09_check.csv`; the evaluation's `nRMSE_kW_pct` of `xgb_sota` indirect equals the corrected
  value.
- *Result (2026-11-07, R9-09):* all of the above holds exactly. The review found one side
  effect: the k-unit `nRMSE_pct`, `skill_score` and DM tests of the indirect route get worse
  (the kW intercept divided by a small theo at low sun), and the DM against climatology at D+3
  is no longer significant. Open as Q-19 in `PROJECT_MASTER.md`; nothing was changed.

## 16. Q-19: DM and skill in kW for both routes (2026-11-07)

**Decision (user, 2026-11-07, Q-19, option B):** the evaluation computes the DM tests and the
skill scores also in kW for both routes, against actual AC, the indirect forecast through
`evaluate.to_kw`; the tables in the target's unit stay as secondary statistics. The kW table is
the one the stage-10 acceptance reads ("XGB clearly better than the baselines, DM with HLN").
Recorded as a dated amendment of D-14 and §6.9 in `PROJECT_MASTER.md`. It was decided after the
k-unit side effect of L-09 was found (R9-09) and before any kW DM was computed. Written before
the code.

### 16.1 Self-review before the code (X17–X18)

| ID | Problem | Where | Fix | Test |
|---|---|---|---|---|
| X17 | The stage-8B command `python -m solarpv.evaluation.evaluate` (validation) still reads the test rows of the truth table: X3 was fixed only in the runner. The test rows are not scored, but they reach pandas | `evaluation/evaluate.main` | For the validation split the truth is read with `exclude_splits=("test",)`; the validation tables do not change (T-55 already shows that they do not need the test rows) | T-65 |
| X18 | A kW version of the DM and skill code could become a fifth copy of the k → kW conversion (X12), or a second DM loop that drifts from the first | the design | One conversion of the scored rows to kW (`evaluate.in_kw`, through `to_kw`) and the same `dm_table` on the converted rows; for the direct target the kW DM and skill must equal the existing ones exactly | T-65 |

### 16.2 Design

- `dm_tests_kw.csv`: the columns of `dm_tests.csv` and a `unit` column (`kW`); the same pairs,
  rows, reference (the runner: `xgb_sota`; stage 8B: persistence), Bonferroni families per
  target × horizon, HLN, the CI of the mean loss difference and the power analysis, on the
  squared kW errors against actual AC.
- `benchmark_summary.csv` gains `skill_score_kw` (1 − RMSE in kW of the model / that of
  persistence of h days, on the common hours) and `skill_cliper_kw` (1 − RMSE in kW / the CLIPER
  RMSE of the AC series). For the direct target they equal `skill_score` and `skill_cliper`.
- Only with a truth table (stage 8B and the runner). The fixture mode of stage 8A has no theo and
  stays as it is (T-15).
- No existing column or table changes. The committed stage-8B validation tables
  (`outputs/07_evaluation/val/`) are regenerated so that T-55 keeps comparing like with like,
  and notebook `07_evaluation.ipynb` is rebuilt with a note on the kW DM.
- Not in scope: the seasonal and common-mask tables stay in the target's unit (the views table
  already has the kW nRMSE).

### 16.3 Verification

- **T-65 (synthetic):** the kW DM and skill of the direct target equal the existing ones
  exactly; for the indirect target they equal a hand computation with `to_kw` and actual AC; the
  fixture mode writes no kW table; `main` on validation reads no test row.
- **Real data:** the regenerated stage-8B tables equal the committed ones in every existing
  column; a dev run gives `dm_tests_kw.csv`, read with the forecasting-validity review.
- *Result (2026-11-07, R9-11):* both hold. In kW, XGB-dev is significantly better than every
  reference in both routes and at every horizon; against climatology the effect is about the
  minimum detectable effect.

## 17. W3: the datasets of the sequence models (self-review and input contract, 2026-11-07)

Written before any W3 code. W3 builds numpy/pandas datasets only (no torch, no model); W4 trains
on them. Nothing in W3 is tuned or decided on validation: the datasets are checked against theory
(§17.8), and the choices that v6 and the evidence leave open are proposed in §17.6 for the user.

### 17.1 Blocking decisions and sources

- **Decided, not changed here:** D-16 (T0 = 00:00 local; D+h is the h-th local day), D-12 (GFS
  admitted), D-13 with R7-04 (`model_inputs.json`), D-22 (op-limited hours masked in past inputs
  and out of the loss), D-08 (the six models of v6), §6.8 (the splits, cut at UTC midnights),
  §6.11 (the time rules of the sequence models), L-19 (PatchTST univariate).
- **Open (§17.6, proposed D-25–D-28):** the decoder length on DST weeks, the 18 known reals of
  the TFT, the GFS inputs of the sequence decoders, the BiLSTM input. Where v6 is silent the
  proposal follows the submitted evidence (user, 2026-11-07: rely on the submission folder and on
  what was done there).
- **Sources (read-only):** v6 §3.4, §3.6.8, §3.7, §4.1, §5.2–§5.6 and the discussion;
  `submission_evidence/08_lstm_alignment/` (the surviving LSTM forecasts and the alignment log);
  `12_scripts/align_and_evaluate.py`; `01_benchmark/benchmark_summary.csv` (n per model);
  `11_audit/` (MISSING_PROJECT_WORK_AUDIT row 17, FEATURE_SELECTION_INTEGRATION_PLAN §5,
  PROJECT_ARTIFACT_INVENTORY); `old files/07_LSTM_Attention v2.ipynb` (an intermediate version:
  the final BiLSTM and TFT code did not survive, R-16) and `04_feature_engineering_v4.ipynb`
  (feature names). Literature: L-12 and L-19 (§12; `docs/literature_review.md`, C1: sequence
  models need temporal feature engineering less than tabular models do).

### 17.2 What the original sequence models did (reconstructed)

| ID | Fact | Evidence |
|---|---|---|
| E1 | BiLSTM: an origin every hour, an encoder of 72 h, a 72-step decoder, and one decoder step kept per horizon, steps 23, 47 and 71 (0-based): fixed leads of 24, 48 and 72 h. The encoder windows started at the test start | The surviving file has 5,689 rows per target × horizon (5,832 − 143), hourly, no duplicate; D+1, D+2, D+3 start at test start + 95, 119, 143 h and end at test end − 48, 24, 0 h. With a 72-step decoder only an encoder of 72 h and steps 23/47/71 fit these numbers. The script's comment ("target_step = h_end-1") and the audit's formula `valid_ts = window_end − (DECODER_LEN − 1 − target_step) h` say the same |
| E2 | TFT (all three variants): one origin per day from the test start, whole 72-step decoders inside the test, steps 1–24 / 25–48 / 49–72 = D+1 / D+2 / D+3. The encoder could read validation hours (TFT-168 has the n of TFT-96) | n = 2,554 / 2,556 / 2,558 is exactly 2,580 minus the daytime hours of May 30–31 (13 + 13), of Oct 1 and May 31 (11 + 13), and of Oct 1–2 (11 + 11), counted on the original labels with the F-02 mask |
| E3 | PatchTST: 243 daily origins from the test start; the decoder could run past the test end | n = 2,580 / 2,569 / 2,558 is 2,580 minus nothing, Oct 1, and Oct 1–2 |
| E4 | The BiLSTM decoder had no known future input ("the decoder must predict 72 steps ahead from a single context vector", v6 on the BiLSTM bias). Its input was the 05e set minus EXCL, like XGB | v6 §3.4 ("a feature matrix … for the tree and recurrent models") and §3.6.8; `11_audit/FEATURE_SELECTION_INTEGRATION_PLAN.md` §5.2–§5.3, `PROJECT_ARTIFACT_INVENTORY.md` |
| E5 | TFT known reals: "18 calendar and irradiance features". The v6 §3.4 table has the groups clear-sky irradiance (8) and time encodings (10). Notebook 04 v4 builds the 8 clear-sky components and 13 temporal features | v6 §3.4, §5.3; `04_feature_engineering_v4.ipynb` |
| E6 | No sequence model saw a weather forecast: every `fcst_*` column was excluded as leaky reanalysis | v6 §3.6.8, §3.7 |
| E7 | Every window ran on the local clock read as UTC (F-01): 24 hourly bins on every day, the fall-back hour merged into one bin of 120 minutes, the spring-forward bin empty | F-01: the LSTM `y_true` at 06:00Z is the local 06:00 hour |

### 17.3 The original project (M13–M18)

| ID | Mistake | Where | Fix in W3 | Test | Errata |
|---|---|---|---|---|---|
| M13 | The windows of every sequence model ran on the naive local clock (E7): the solar geometry was off by 2–3 h (F-01), and every day had 24 steps, so "steps 1–24 = D+1" looked right only because the clock was wrong | E7, F-01 | Windows on the true-UTC index of the stage-6 tables, where a local day has 23, 24 or 25 hours (8 such days in the data, 2021-10-31 to 2025-03-28) | T-66 | C-47 |
| M14 | Horizons by step number, and not the issuance of D-16. (a) The BiLSTM forecast every hour with a fixed lead (E1): the D+1 forecast of each hour came from a different origin 24 h earlier, a "day" was a mosaic of 24 origins, and the tabular and sequence models answered different questions. (b) The TFT took steps 1–24 / 25–48 / 49–72 by number (E2, v6 §5.2): on the true UTC axis a 23- or 25-hour day moves D+2 and D+3 by one hour (T-02). (c) v6 §5.2 describes "D+1: steps 1–24"; the surviving file keeps one step per window | v6 §5.2; E1, E2 | One origin per local day, T0 = 00:00 local (D-16), for every sequence model. A decoder step's horizon is the local day of its valid hour, counted from T0's day. The decoder covers exactly the three local days (proposed D-25) | T-66, T-02 | C-48 |
| M15 | Each sequence model was scored on its own hours: XGB 2,580 daytime hours, PatchTST 2,580 / 2,569 / 2,558, TFT 2,554–2,558, BiLSTM 2,510–2,514. The BiLSTM encoders started inside the test and no origin preceded the test start, so the first one to six days and the last two had no forecast, and the DM pairs had different n (`REPORT_CLAIM_EVIDENCE_MATRIX.csv` row 50). PatchTST's input size ("2 × INPUT_SIZE") is not reported | E1–E3; `benchmark_summary.csv` | The coverage rule of stage 8B, for every split: origins from the local day of the split's first hour minus two days to the local day of its last hour; encoders read the hours before T0 wherever they lie (they are known at T0, also in the previous split); decoders stop at the end of the split. Every hour of the split gets exactly one forecast per horizon. PatchTST's input size is recorded in W4 | T-67 | C-49 |
| M16 | Silent fallbacks and leaky defaults in the intermediate LSTM (v2): without the feature JSON it takes "intelligent default features", that is every `fcst*` column (same-timestamp reanalysis), the POA forecasts and `theoretical_power` (D-03); a missing feature is dropped with a print; every row with a NaN in any input or target is dropped before windowing, so 168 rows are not 168 hours and a window can jump over a gap | `07_LSTM_Attention v2`, cells 5 and 10 | Inputs from explicit lists, checked against the tables and `assert_no_leakage`; a missing column is an error. No row is dropped: the time axis is the complete hourly index, and a gap in an encoder series is filled from older hours only (the stage-6 rule) and flagged | T-68, T-71 | C-50 |
| M17 | Splits and scaling in v2: a chronological split by proportions (test 15 %, validation 15 % of the rest) instead of the dates of §6.8; the k target scaled by a MinMax fitted on train, which maps a later k above the train maximum outside [0, 1] (harmless for a linear head, a cap for a bounded one; k is no longer clipped, F-06). The feature scalers were fitted on train only, which was correct | v2, cells 3, 10 and 13 | The date splits of §6.8; input scalers fitted on train hours only, the same in every mode (A-14); no fitted transform of the targets (k as is, AC / 3,449) | T-69 | C-51 |
| M18 | The valid time was attached after training. Before the fix every step carried the window-end label (D+1 nRMSE about 17 % instead of about 30 %, v6 discussion); the fix was a formula on step numbers in a separate script, followed by `drop_duplicates(keep="last")` (M6, C-41) | `align_and_evaluate.py`; v6 discussion | Every decoder step carries its valid timestamp and horizon from the dataset; the writer of the predictions refuses a duplicate and a missing hour; `y_true` comes from the feature table (§6.9) | T-70 | C-52 |

E4 (no future input in the BiLSTM decoder) is not counted as a mistake: it was the architecture,
and v6 itself names it as the cause of the BiLSTM bias. The rebuild gives every decoder the inputs
known at T0 (§6.11, decided earlier).

### 17.4 My own code that W3 builds on (X19–X23)

| ID | Problem | Where | Fix | Test |
|---|---|---|---|---|
| X19 | The docstring of `lstm_sota.py` prescribes the original step-number formula `valid_ts = window_end − (DECODER_LEN − 1 − target_step) h` (M14) and "final training on Colab" (outdated, §2); `patchtst_model.py` says "243 rolling origins", the test-only count of M15 (with the coverage rule there are 245 test origins) | `models/lstm_sota.py`, `models/patchtst_model.py` | Docstrings rewritten; the mapping and the origins live in the W3 module only | T-66 |
| X20 | `features.build.issuance_utc(hours, h)` maps a target hour to its T0. W3 needs the inverse (an origin to its decoder hours and their horizons), and a second local-day computation could drift from the first, as the k → kW copies could (X12) | `features/build.py` | The W3 mapping is tested against `issuance_utc`: for every step of every origin, `issuance_utc(valid, h) == T0`, on synthetic series with both DST days and on the real index | T-66, T-02 |
| X21 | `pipeline.data.model_table(h, splits)` filters by the split label, so a validation run reads no train row. The windows need the hours before the split (encoders, and the decoders of the two origins before the split start) and must stop at TEST_START in smoke and dev | `pipeline/data.py` | A time-range loader of the same kind (filters inside the parquet reader, X10): every row before TEST_START in smoke and dev, test rows only with `allow_test`; the split label decides which steps are kept | T-67 |
| X22 | The D-22-masked hourly series (per-inverter masks from the minute flags) exist only inside `features.build.masked_sources`; the stage-6 tables keep them only as lags. Reading the unmasked hourly AC would feed stops and curtailment (human decisions, D-22) into the encoder, and rebuilding the masks in W3 would be a second copy | `features/build.py` | W3 calls `masked_sources` on the stage-5 hourly table and the minute flags, both read with `timestamp_utc < TEST_START` in smoke and dev; a test checks that the encoder series equals the stage-6 lag it feeds (`<source>_lag_1d` at s + 24 h) on every hour that was not imputed | T-68 |
| X23 | The loss rows of XGB (`xgb_sota.fit_mask`: warm-up, op-limited, k valid, theo > 50 kW, AC present) would need a copy for the sequence models | `models/xgb_sota.py` | W3 calls `fit_mask` on the decoder hours | T-67 |

### 17.5 The input contract (fixes §4.2)

**Origins.** One per local day: T0 = 00:00 Asia/Jerusalem of that day, in UTC (21:00Z in summer
time, 22:00Z in winter; local midnight is never ambiguous, the clock changes at 02:00). For a
split, or the part of it inside the run's window, the origins run from the local day of its first
hour minus two days to the local day of its last hour, and only the steps whose valid hour lies in
it are kept. The splits are cut at UTC midnights (§6.8), so the local days at their edges are
shared: 2024-02-01 local starts with two train hours, 2024-10-01 local with three validation
hours.

| Scope | Origins (local days) | Count |
|---|---|---|
| Training, dev and final | 2021-09-13 to 2024-02-01 (the first origin whose D+3 day is the first day after the warm-up; warm-up hours are out of the loss) | 872 |
| Training, smoke | the last 56 local days of train (as XGB, §13.4) and the two days before | 58 |
| Validation, dev | 2024-01-30 to 2024-10-01 | 246 |
| Validation, smoke window (2024-03-15 to 04-11, W1) | 2024-03-13 to 2024-04-11 | 30 |
| Test, final run only | 2024-09-29 to 2025-05-31 | 245 |

**Encoder.** The L hours [T0 − L, T0) in UTC. Every encoder hour ends by T0 (s + 1 h ≤ T0, the
guard of stage 6) by construction, and no value computed from a later hour enters (the fill of a
gap uses older hours only). L is 72 h for the BiLSTM (E1, proposed in D-28), 96 h for TFT-96 and
168 h for TFT-168 and the TFT baseline (§6.11); PatchTST uses the input size of W4, target only.

**Decoder.** The hours of the three local days from T0: 72, or 71 / 73 for the three origins
whose window holds a 23- / 25-hour day (12 + 12 origins in the data; in validation 2024-03-27 to
03-29). The arrays are padded to 73 steps with a step mask (proposed D-25). A step's horizon is
the index of its local day (steps 0–22, 0–23 or 0–24 are D+1, depending on the length of the
first day, and so on), never its step number. A decoder stops at the end of its split (training
windows at VAL_START, validation windows at TEST_START in every mode, test windows at the end of
the data); the steps beyond are padding, and no row beyond the limit is read.

**Encoder series** (§4.2; the same for the BiLSTM and the TFT). The seven measured series that
the stage-6 lags are built from: `s01_inv1_AC_kW`, `s01_inv2_AC_kW` and `site_total_AC_kW` with
the D-22 masks of `masked_sources`, `irradiance`, `mod_temp1`, `mod_temp2`, `ambient_temp`; for
the indirect route also `target_k` (0 where theo ≤ 50 kW by §6.5, NaN in op-limited hours). Gaps
are filled once on the whole loaded series with `impute_same_hour` (the same UTC hour over the
previous 30 days, older rows only); a value still missing after it (no older value at that hour)
takes the train mean, 0 after scaling. One 0/1 channel per series marks the filled hours. The
model's known inputs (below) are given on the encoder hours too: calendar and clear sky of those
hours, and the GFS value of the D+1 table, the day-ahead forecast issued at the T0 of the hour's
own day, which is at or before the origin's T0.

**Decoder inputs** (only what is known at T0, §6.11). Calendar, clear sky and geometry are
deterministic in the valid time and identical in the three horizon tables (asserted). The GFS
columns and `irradiance_clear_index_prevday` (L-11a, measured before T0) come from the D+h GFS
table for a step of day D+h; its `t0_utc` is the origin's T0, so the three days of one origin
come from the same 12Z run, and init + 6 h ≤ T0 holds on every step (the rule of T-17, asserted).
No lag, no rolling statistic and no measurement of a decoder hour. A missing known input takes the
train mean (counted). Per model: the TFT gets the known reals of D-26 and the GFS of D-27, the
TFT baseline the same inputs as TFT-96 (v6 §5.5), the BiLSTM the inputs of D-28, PatchTST none
(L-19). *By the decisions of §17.6 the BiLSTM and the three TFT models get the same known inputs
per target (indirect 10 + 20, direct 27 + 22).*

**Targets and loss.** Per step: `target_k` (indirect, never clipped, F-06) or
`site_total_AC_kW` (direct) at the valid hour, with `theo_kw`, `daytime` and `op_limited_flag`
for the writer. The loss mask is `fit_mask` (X23) and the step mask; a training window holds only
steps of the training split. Nights stay in the direct loss and out of the indirect loss, as for
XGB (M11).

**Scalers (A-14).** Inputs: a z-score per channel, with the mean and standard deviation over the
train hours that are not warm-up, NaN ignored, computed once before any window and the same in
every mode; the flags and the step mask are not scaled; the statistics are written to the dataset
report. Targets: no fitted transform (k as is, AC / 3,449, the fixed capacity), so the inverse is
exact. A normaliser inside the libraries of W4 (pytorch-forecasting's target normaliser,
neuralforecast's `scaler_type`) must act on each window's encoder or be fitted on the training
windows only; W4 records which.

**Into the prediction contract (§6.10).** One function turns the window forecasts
`y_pred[origin, step]` into the rows of §6.10: every kept step gives timestamp (its valid hour,
UTC), `y_true` (feature table), `y_pred`, target, horizon (its local day), model, seed and split.
A duplicate (horizon, timestamp), or an hour of the window without a forecast at some horizon, is
an error. The evaluation layer then applies its own alignment and coverage checks unchanged; on
validation every target has 5,832 rows per horizon.

**Module.** `solarpv.models.sequence_data`: the origins, the step → horizon mapping, a time-range
loader (X21), the encoder series (X22), the known inputs, the scaler, `build(...)` returning
arrays and metadata per model × target × scope, and the writer. `python -m
solarpv.models.sequence_data` builds the train and validation datasets of every model × target
and writes the checks of §17.8 to `outputs/09_sequence_data/` (reports only; no arrays in git).
W4 adapts the arrays to pytorch-forecasting (one group per origin) and neuralforecast, and
builds the GPU environment once the user approves it.

### 17.6 Proposed decisions (D-25–D-28), for the user

**D-25, the decoder length on DST weeks.** v6 says decoder 72; §6.11 and T-02 say the horizon is
the local day. On the true UTC axis three local days last 71, 72 or 73 hours.
- **(A, recommended)** The decoder covers exactly the three local days, padded to 73 with a mask:
  every D+h block is its local day (23/24/25 steps) and every hour of a split gets its three
  forecasts. "Decoder 72" in §6.11 becomes "the three local days (72 h, except 71 or 73 h on six
  origins a year)", as a dated amendment.
- **(B)** 72 steps from T0. After a 25-hour day the last hour of D+3 (23:00 local, always night)
  gets no forecast from that origin, so three night hours per fall-back day have no D+3
  forecast; after a 23-hour day step 72 falls on D+4 and is dropped.

**D-26, the 18 known reals of the TFT** (E5).
- **(A, recommended)** The two groups of the v6 table, with the v4 names. Clear-sky irradiance
  (8): `ghi_clearsky`, `dni_clearsky`, `dhi_clearsky`, `poa_global_clearsky`,
  `poa_direct_clearsky`, `poa_diffuse_clearsky`, `poa_sky_diffuse_clearsky`,
  `poa_ground_diffuse_clearsky`. Time encodings (10): the seven named in v6 (`hour_sin`,
  `hour_cos`, `doy_sin`, `doy_cos`, `month_sin`, `month_cos`, `season`) and the three temporal
  features of v4 that v6 places in no other group: `month`, `day_of_week`, `daylight_fraction`
  (v6 lists `is_weekend` and `hour_local` as "derived", and v4 counts `time_to_solar_noon` among
  its solar features). These three names are a reconstruction, not a certainty.
- **(B)** The same 8 + 7, with three geometry features that stage 7 kept in all three indirect
  sets instead of the uncertain three: `aoi`, `time_to_solar_noon`, `day_length`.
- In both, the definitions are the corrected ones of the rebuild (UTC-hour cyclicals, the F-25
  calendar, pvlib solar times).

**D-27, the GFS inputs of the sequence decoders (TFT and BiLSTM).** D-12 admitted GFS, §6.11
puts it in the decoder, and v6's sequence models had none (E6).
- **(A, recommended)** Per target, the GFS columns that stage 7 selected for it, as the union
  over the three horizons (one decoder spans all three): 20 of the 22 candidates for the indirect
  route (without `fcst_gfs_ghi` and `fcst_gfs_pv_kw`), all 22 for the direct route (its input is
  the full pool, F-29). One selection (D-13) stands behind every model, and the families are
  compared on the same weather information.
- **(B)** All 22 candidates of stage 6b for both targets, and the TFT's variable selection
  network does the selection.

**D-28, the BiLSTM input** (R-16: the final code is lost; E4: it used the 05e sets and no future
input).
- **(A, recommended)** As in the submission, the stage-7 input of its target, adapted to an
  encoder–decoder: the part of `model_inputs.json` known at T0 (union over the horizons: indirect
  10 deterministic + 20 GFS = 30, direct 27 + 22 = 49) on the encoder and decoder steps; the lag
  and rolling features are replaced by the measured series they are built from, in the encoder
  (the series of §17.5, as for the TFT). Encoder 72 h, as the surviving forecasts show (E1).
  Bidirectional encoder; the decoder runs forward over the known inputs (W4), so a decoder cut at
  the end of a split gives the same early steps.
- **(B)** The inputs of the TFT (the 18 known reals and the GFS of D-27), so that the BiLSTM and
  the TFT differ only in architecture. Encoder 72 h.

**Decisions (user, 2026-11-07), before any W3 code:**
- **D-25:** (A), the decoder covers exactly the three local days (71 / 72 / 73 steps, padded to
  73). A dated amendment of "decoder 72" in `PROJECT_MASTER.md` §6.11.
- **D-26:** neither (A) nor (B). The user pointed out that the project had a feature-selection
  stage that found the most relevant features, and that the known reals should come from it.
  Checked in the submission: the calendar, clear-sky and geometry features of the six submitted
  05e sets add up to 17 names, one of them a duplicate (`sun_azimuth` = `solar_azimuth`), close to
  v6's "18 calendar and irradiance features", which supports this reading. Offered as a new
  option and chosen: **per target, the calendar, clear-sky and geometry features of the stage-7
  selection, the union over the three horizons**: 10 for the indirect route, 27 for the direct
  route (its input is the full pool, F-29). The count differs from v6's 18 because the selection
  was redone on the corrected data (D-13).
- **D-27:** (A), the GFS columns of the stage-7 selection per target (20 / 22).
- **D-28:** (A), as in the submission: the part of `model_inputs.json` known at T0, the measured
  series in the encoder, encoder 72 h. With D-26 the BiLSTM and the three TFT models have the same
  known inputs per target; they differ in architecture and encoder length (72 / 96 / 168 h).

### 17.7 Tests (T-66–T-71 on synthetic series, T-02 on the real tables)

| ID | Test |
|---|---|
| T-66 | The mapping: T0 at local midnight (21:00Z / 22:00Z); decoders of 71 / 72 / 73 steps around 2024-03-29 and 2024-10-27; every D+h block is the set of hours of its local day (23 / 24 / 25); `issuance_utc(valid, h) == T0` on every step; a mapping by step number fails on these days |
| T-67 | Origins and coverage per split and mode (the counts of §17.5 on the real calendar): every hour of the split exactly once per horizon; no step at or after TEST_START in smoke or dev; training decoders end before VAL_START; the loader reads no row at or after TEST_START (filters inside the reader) and refuses test rows without `allow_test`; the loss mask is `fit_mask` |
| T-68 | The encoder: its last hour is T0 − 1 h; changing any measured value at or after T0 changes no input of the window (only targets); the fill uses older hours only; the masked hours (D-22) are NaN before the fill and flagged; the encoder series equals the stage-6 lag where nothing was imputed (X22) |
| T-69 | The scalers: fitted on train hours only; changing or permuting validation values leaves the statistics and every training window unchanged; k is not transformed and kW / 3,449 inverts exactly |
| T-70 | The writer: one row per (horizon, valid hour); a duplicate or a gap raises; `y_true` from the table; a round trip with `y_pred = y_true` passes the strict alignment and coverage checks of `evaluate`; horizons by local day on the DST days |
| T-71 | The input lists: the 18 known reals (D-26) exist in all three tables, are equal across them and are not leaky; the GFS (D-27) and BiLSTM (D-28) lists come from `model_inputs.json`, hold no lag or rolling name and pass `assert_no_leakage`; per GFS step `t0_utc == T0` and init + 6 h ≤ T0; PatchTST windows hold the target only |
| T-02 | Real data (`tests/contracts/`), train and validation, every model × target: unique valid timestamps per horizon, the encoder ends at T0, the decoder mapped by local day, no row at or after TEST_START |

### 17.8 Checks on real data (train and validation only) and review

The dev scope of `python -m solarpv.models.sequence_data` builds and checks: origins per split
(872 / 246) with their first and last T0; no row at or after TEST_START read or carried; every
encoder ends at T0; every D+h block covers exactly its local day (23 steps on 2024-03-29 in
validation; the DST days of train: 2021-10-31, 2022-03-25, 2022-10-30, 2023-03-24, 2023-10-29);
every validation hour has one step per horizon; the scaler statistics come from train hours only
(recomputed independently, and unchanged when validation values are perturbed); the GFS run of
every step; the filled encoder values per series and split. Then the `forecasting-validity-review`
skill. Run journal R9-12 on.

### 17.9 Records

With the code, in one commit: OP-55 (the sequence datasets), errata C-47–C-52 and X19–X23, the
§9 rows T-66–T-71 and T-02, the docstrings of X19. After the checks on real data: R9-12 on,
`PROJECT_MASTER.md` §1, D-25–D-28 as decided, A-14 (the scalers of the sequence datasets), L-12
(the inputs per model family) and §17.
