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
| College machine | Unknown: operating system, GPU and CUDA, memory, disk, internet access, scheduler, time limits (Q-15) |

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
