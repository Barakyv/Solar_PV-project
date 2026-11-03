# Hand-off to the college machine (2026-11-03)

**User request (2026-11-03):** the work moves to the college computer, which has the resources
for training. Claude Code is opened there and must know what was done on the laptop and what
comes next. Read this file after `CLAUDE.md` and `PROJECT_MASTER.md` §1, in the first session on
the new machine, and do §3 before anything else.

## 1. What changed with the move

- **Up to 2026-11-03** everything ran on the laptop (Windows 11, AMD CPU): stages
  0–6b, 8, the review of the old backup, and two runs of stage 7.
- **From now on** the college machine is used for both the code and the runs. The rule "no
  model training on the laptop" (user, 2026-11-01) stays. On the college machine, training is
  allowed within the project rules:
  - development runs (stage 9) read train and validation only, and the test split is refused;
  - final training (stage 10) happens only after gate G-1, on the commit tagged `g1-frozen`
    (D-08);
  - the test split never decides anything.
- The plan for the automated runner (one command for every model, step selection, modes,
  training logs) is `docs/stage9_plan.md`. No runner code exists yet: `xgb_sota.py`,
  `lstm_sota.py` and `patchtst_model.py` are docstrings only, and there is nothing for the TFT
  models.

## 2. The package

Made on the laptop with `python scripts/handoff_pack.py` (the file `HANDOFF_INFO.json` in the
package names the commit).

| In the package | Size on the laptop |
|---|---|
| The working copy with its full git history (`.git`, tags `stage-0x-passed`) | 17 MB of git objects |
| Untracked files, as they are (the stage-7 outputs, §4) | small |
| `Raw data/` (read-only) | 0.8 GB |
| `data/processed/` (stages 1–6b) | 0.7 GB |
| `data/external/gfs_1p00/` (the GFS downloads of stage 6b) | 48 MB |
| `HANDOFF_MANIFEST.sha256`, `HANDOFF_INFO.json` (git-ignored) | small |

Left out:
- `old files/solar_ds_capstone/` (2 GB): the old backup. It is audited in
  `00_data_audit/original_backup_2026-11-01/`, and the findings are F-26 to F-28.
- `data/external/original_backup_recovered/` (1.3 GB): the blobs recovered from it.
- Caches, `.claude/worktrees/` and `src/solarpv.egg-info/`.
- `configs/site.private.yaml`: the true site coordinates. They are copied by hand, only if the
  user approves (Q-16). Without the file, the code falls back to the anonymised placeholder
  site (`configs/site.example.yaml`). The processed tables already hold the geometry, so the
  models can run. But every run records `site_private: false`, and a result from a run that
  recomputes geometry is not valid for the study. The tests that need the file are skipped.

## 3. First session on the college machine, in order

1. Unpack the zip; the project is the folder `solar/`. Open Claude Code in it.
2. `git worktree prune`. This drops a stale laptop worktree. `git status` must show only the
   untracked stage-7 outputs listed in §4.
3. `python scripts/handoff_pack.py --verify`. It needs any Python 3 and the standard library
   only. Every file must match. Then `git fsck` checks the git objects.
4. Environment:
   - `conda env create -f environment.yml` creates the CPU environment `solar`, which installs
     the package with `pip install -e .`;
   - `environment.lock.yml` is the college machine's lock since 2026-11-03 (Windows; the
     laptop's lock is in git history before that date). On another OS, compare versions and
     record any difference;
   - the GPU environment does not exist yet. It is built here after the GPU is known, per
     `docs/stage9_plan.md` §5. `requirements-gpu.txt` lacks pytorch-forecasting 1.7.0.
5. `pytest -q`. The baselines are in §8. Without the private file, the tests that need the true
   site are skipped. Nothing may fail.
6. Inspect the machine and record it in `docs/stage9_plan.md` §1 (this answers Q-15):
   - the OS;
   - the GPU, its memory, and the driver and CUDA versions (`nvidia-smi`);
   - CPU cores, RAM and free disk;
   - internet access to pip, conda and GitHub;
   - whether there is a scheduler such as SLURM, and the longest allowed job;
   - how the user connects to it.
7. Seed Claude's auto-memory from `docs/claude_memory/`: copy the four notes and the
   `MEMORY.md` index into the memory directory named in the system prompt. The same rules are
   summarised in `CLAUDE.md` under "Working with the user".
8. Ask the user the open decisions in §5. Ask before rerunning stage 7 or writing the runner.

## 4. Where the work stands

- **Done:** stages 0–6b and 8 (`PROJECT_MASTER.md` §1). Stage 0 lacks only the external backup
  of `Raw data/`, which is the user's task.
- **Stage 7 (feature selection): run, diagnosed, one decision open.**
  - The protocol is D-23; the plan is `docs/stage7_plan.md`. The code is
    `src/solarpv/selection/` (`common.py`, `steps.py`, `run.py`), tested by T-47 to T-50.
  - Journal entries:
    - R7-01: the first run's duplicate step removed real information. It now reports
      correlated pairs only.
    - R7-02: the sign of the S8 red flag was inverted. Fixed, with a test.
    - R7-03: the diagnosis of the red flags (`00_data_audit/selection_2026-11-01/`).
  - k history was not admitted (D-24, `configs/k_history_decision.yaml`): the D+1 gain was
    0.026 pp, against a margin of 0.054 pp.
  - 05e sets:
    - indirect: 25 / 26 / 26 features (D+1 / D+2 / D+3);
    - direct: 19 / 16 / 12 features.
  - Validation look S8 (05e minus the full pool):
    - indirect: −0.015 / +0.005 / +0.088 pp;
    - direct: −0.005 / +0.333 / +0.425 pp;
    - red flags: indirect D+3, direct D+2 and D+3.
  - R7-03 found two different causes:
    - **direct, a method problem.** The gap appears inside train too. The selection model
      (100 trees, depth 5) is too weak: at D+3, one feature came within 1 % of all 42;
    - **indirect, a sample effect.** There is no gap inside train.
  - **Untracked on purpose:** `outputs/07_feature_selection/` (all but `runs/`) and
    `configs/k_history_decision.yaml`. The rule in `CLAUDE.md` puts the OP entries in the same
    commit as the outputs. Those entries wait for the direct-route decision, because it
    changes the outputs.
- **Stage 9:** `docs/stage9_plan.md` (the runner, steps, modes, logs, models, environment,
  work order W1–W6, questions Q-15 to Q-18). No code yet.

## 5. Open decisions for the user (ask first)

1. **Q-17, the direct-route selection.**
   - Recommendation: rerun the selection of the direct route with the fixed D-11 XGBoost
     (400 trees, depth 6, learning rate 0.05) as the selection model. First check inside
     train (the date folds and the inner hold-out of `diagnose_gap.py`). Then make one more
     validation look (S8), documented as the third.
   - Alternatives:
     - keep the sets as they are and report the gap;
     - use the full pool for the direct route.
   - This rerun becomes the `select` step, the first heavy job on the college machine.
2. **The MI vote.**
   - It endorsed all 42 starting features in every run, so the consensus equals the union of
     the elbow and stability sets.
   - Recommendation: keep it as registered (D-23) and report it as a limitation.
3. **Q-16, the coordinates:** may `configs/site.private.yaml` be copied to the college machine?
4. **Answered or moot:**
   - Q-15 is answered by the inspection in §3.6;
   - Q-18 (smoke runs on the laptop) is moot, since the development runs move to the college
     machine.
5. **Older items still with the user:**
   - Q-02 (does the supervisor know the numbers will change);
   - Q-11 to Q-14;
   - a backup of `Raw data/` and of the old backup folder;
   - a private GitHub repository (optional; see §7).

## 6. Work order on the college machine

1. **Finish stage 7** after Q-17:
   - the rerun (direct route; indirect unchanged unless the user decides otherwise);
   - `docs/data_processing_record.md`: a section "Feature selection (stage 7)" with OP-46,
     OP-47 and OP-48;
   - renumber the sections that follow it from 14–17 to 15–18, and change the cross-reference
     "§15" (line 269) to "§16";
   - section N15 in `scripts/build_data_record.py`, then run it;
   - errata statuses:
     - F-26 and F-27: fixed;
     - C-09;
     - C-31 and C-32: a text correction, since the k history was not admitted;
   - the D-24 row in `PROJECT_MASTER.md` §3, the status of L-12, and `docs/scope_register.md`
     S-06 to S-10;
   - `docs/methodology_changes.md`;
   - the notebook builder `scripts/make_feature_selection_notebook.py`:
     - fill the placeholders (`PLACEHOLDER_K`, `_SETS`, `_VAL`, `_OLD`);
     - drop the removed "after duplicates" count column;
     - build `notebooks/05_feature_selection.ipynb`;
     - add `"05_feature_selection"` to `tests/contracts/test_stage_notebooks.py`;
   - commit the outputs with their OP entries; update `PROJECT_MASTER.md` §1 and §17.
2. **Stage 9 and the runner**, per `docs/stage9_plan.md` W1–W6. Development and smoke runs now
   happen on the college machine (`smoke` and `dev` modes, validation only). Before writing the
   sequence models, fix the input contract of §4.2 (encoder, decoder, the 18 known reals, the
   decoder-to-horizon mapping by local day, T-02).
3. **Stage notebooks:** `05_feature_selection` (stage 7) and `06_models` (stage 9), each built
   from the outputs.
4. **Then G-1, the tag `g1-frozen`, and the `final` run** (stage 10). Every evaluation or
   selection result goes through the `forecasting-validity-review` skill before it is accepted.

## 7. Bringing the work back to the laptop

- **Code, documents and tracked outputs:** run `git bundle create solar_<date>.bundle main` on
  the college machine. On the laptop, run `git pull <path>/solar_<date>.bundle main`. Never
  unpack a package over the laptop folder: the laptop keeps files that are not in the package
  (§2).
- **Run outputs that are not in git** (checkpoints, Optuna databases, Lightning logs): the
  `collect` command of the runner (`docs/stage9_plan.md` §3.1, W1).
- **Alternative:** a private GitHub repository for the code and the documents, if the user
  opens one. The data still move only by package (`.gitignore` keeps `data/` and `Raw data/`
  out of git).

## 8. Facts recorded at hand-off

- Laptop environment: Python 3.11.17, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.2,
  shap 0.51.0 (`environment.lock.yml`).
- Test baselines at the hand-off commit:
  - on the laptop, with `site.private.yaml`: 357 passed, none skipped (about 2.5 minutes);
  - on an unpacked copy without it: 319 passed, 38 skipped, none failed (RH-01 in
    `docs/run_journal.md`).
- Git author: the local `.git/config` (it comes with the package).
