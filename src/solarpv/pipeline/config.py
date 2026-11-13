"""Modes, budgets, evaluation windows and the step order of the runner (docs/stage9_plan.md §3,
§10.3).

Three modes:
  smoke  does the code run? Minimal budgets; evaluated on a 28-day validation window that
         contains the spring DST change (2024-03-29). Test rows are never loaded.
  dev    stage 9: the development budgets (D-17); evaluated on the whole validation split. Test
         rows are never loaded.
  final  stage 10: the budgets of v6 (D-08); only on the commit tagged `g1-frozen` with a clean
         tree, checked before every step (X38); no slice of targets, horizons or seeds (X39).
         Two phases (D-32, docs/stage9_plan.md §19.7): the model steps forecast the validation
         and the test window, and `evaluate` scores validation only (the stage-10 acceptance);
         the test split is scored only by the step `test`, in a later invocation of the same
         run with --allow-test (stage 11). Every read of a test row is logged (X40).
"""

from dataclasses import asdict, dataclass

import pandas as pd

from solarpv.config.constants import (
    DEV_OPTUNA_TRIALS,
    DEV_SEED,
    FINAL_OPTUNA_TRIALS,
    HORIZONS,
    RANDOM_SEEDS,
    TEST_END,
    TEST_START,
    TIMEZONE_LOCAL,
    VAL_END,
    VAL_START,
    XGB_THREADS,
)

MODES = ("smoke", "dev", "final")
TARGETS = ("indirect", "direct")
FROZEN_TAG = "g1-frozen"
# 28 local days inside validation, with the spring-forward day 2024-03-29 (a 23-hour day)
SMOKE_WINDOW_LOCAL = ("2024-03-15", "2024-04-11")

# Budgets per mode and model step, recorded in every run's config.yaml. dev: D-17 (XGB, user
# 2026-11-05; the neural models, user 2026-11-11, docs/stage9_plan.md §18.6: one third of the v6
# trials, the full v6 epochs, one seed). final: D-08 and v6 §5.2-§5.6. Every neural budget has
# every field (X27). PatchTST counts neuralforecast training steps and validation checks.
_PATIENCE = 10


def _neural(bilstm: tuple, tft: tuple, tftbase_epochs: int, patchtst: tuple,
            bilstm_seeds: list) -> dict:
    trials, epochs_trial, epochs_final = bilstm
    tft_trials, tft_epochs = tft
    max_steps, val_check_steps = patchtst
    one = [DEV_SEED]
    tft_budget = {"trials": tft_trials, "epochs": tft_epochs, "patience": _PATIENCE,
                  "seeds": one}
    return {
        "bilstm": {"trials": trials, "epochs_trial": epochs_trial, "epochs_final": epochs_final,
                   "patience": _PATIENCE, "seeds": list(bilstm_seeds)},
        "tft96": dict(tft_budget), "tft168": dict(tft_budget),
        "tftbase": {"trials": 0, "epochs": tftbase_epochs, "patience": _PATIENCE, "seeds": one},
        "patchtst": {"max_steps": max_steps, "val_check_steps": val_check_steps,
                     "patience": _PATIENCE, "seeds": one},
    }


BUDGETS = {
    "smoke": {"xgb": {"trials": 1, "seeds": [DEV_SEED]},
              **_neural((1, 2, 2), (1, 2), 2, (20, 10), [DEV_SEED])},
    "dev": {"xgb": {"trials": DEV_OPTUNA_TRIALS, "seeds": [DEV_SEED]},
            **_neural((20, 25, 60), (7, 100), 150, (5000, 100), [DEV_SEED])},
    "final": {"xgb": {"trials": FINAL_OPTUNA_TRIALS, "seeds": list(RANDOM_SEEDS)},
              **_neural((60, 25, 60), (20, 100), 150, (5000, 100), list(RANDOM_SEEDS))},
}


@dataclass(frozen=True)
class StepSpec:
    """One step of the sequence; `work_item` names the item of the plan (§7) that brings it."""

    name: str
    stage: str
    purpose: str
    device: str
    needs: str
    produces: str
    work_item: str


# The work steps, in the order they run. `env` and `data` are the preflight of every run.
STEPS = (
    StepSpec("select", "7", "verifies the stage-7 selection and records its input; reruns "
             "nothing (D-30)", "CPU", "stage-7 outputs, the stage-6 and 6b table schemas",
             "select.json, copies of the sets", "W5"),
    StepSpec("baselines", "8", "persistence, climatology, clear sky x MEAN_DAYTIME_K", "CPU",
             "truth table, MEAN_DAYTIME_K", "predictions_<reference>.csv", "W1"),
    StepSpec("xgb", "9/10", "XGB-Opt", "CPU", "tables, model_inputs.json",
             "predictions, training logs, study", "W2"),
    StepSpec("cv", "11", "S-38: the xgb models on five expanding date folds of the train rows "
             "(v6 §12.2, D-36)", "CPU", "the xgb step's records, the train tables",
             "cv_folds.csv, cv_summary.csv", "W6"),
    StepSpec("ablation", "11", "S-41: the xgb models refitted without one feature group at a "
             "time (v6 §12.5, D-36)", "CPU", "the xgb step's records, the tables",
             "ablation_results.csv, ablation_summary.csv, forecasts", "W6"),
    StepSpec("bilstm", "9/10", "BiLSTM-Opt", "GPU", "sequence datasets (W3)",
             "predictions, training logs, study", "W4"),
    StepSpec("tft96", "9/10", "TFT with a 96-hour encoder", "GPU", "sequence datasets (W3)",
             "predictions, attention, study", "W4"),
    StepSpec("tft168", "9/10", "TFT with a 168-hour encoder", "GPU", "sequence datasets (W3)",
             "predictions, attention, study", "W4"),
    StepSpec("tftbase", "9/10", "TFT with fixed hyper-parameters", "GPU",
             "sequence datasets (W3)", "predictions, attention", "W4"),
    StepSpec("patchtst", "9/10", "PatchTST, univariate", "GPU or CPU", "target series",
             "predictions", "W4"),
    StepSpec("evaluate", "8/11", "the evaluation layer on every prediction file of the run",
             "CPU", "the run's prediction files, truth table", "the stage-8 tables", "W1"),
    StepSpec("explain", "11", "SHAP (XGB), TFT attention and variable selection, "
             "hyper-parameter importance (D-31)", "CPU", "the run's models and records",
             "explanation tables", "W5"),
    StepSpec("report", "11", "one summary of the run with automatic warnings", "CPU",
             "the run's records", "report", "W5"),
    StepSpec("test", "11", "the test split scored, and the XGB SHAP on test (final only, "
             "second phase, --allow-test; D-32)", "CPU", "the run's test forecasts, truth",
             "the stage-8 tables on test", "W5"),
)
STEP_ORDER = tuple(s.name for s in STEPS)
SPEC = {s.name: s for s in STEPS}
MODEL_STEPS = ("baselines", "xgb", "bilstm", "tft96", "tft168", "tftbase", "patchtst")
TEST_STEP = "test"
REPORT_STEP = "report"


@dataclass(frozen=True)
class Window:
    """Rows of `split` with start <= timestamp < end (UTC) are scored."""

    split: str
    start: str  # ISO, UTC
    end: str    # ISO, UTC, exclusive
    label: str

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)

    def as_dict(self) -> dict:
        return asdict(self)


def _utc(text: str) -> str:
    return pd.Timestamp(text, tz="UTC").isoformat()


def local_days_window(first: str, last: str, split: str = "val") -> Window:
    """Whole local days `first` .. `last` (inclusive), as a UTC window."""
    start = pd.Timestamp(first).tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")
    end = (pd.Timestamp(last) + pd.Timedelta(days=1)).tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")
    return Window(split, start.isoformat(), end.isoformat(), f"local days {first} to {last}")


def split_window(split: str) -> Window:
    """The whole split (the boundaries of §6.8, inclusive minutes -> exclusive hour)."""
    bounds = {"val": (VAL_START, VAL_END), "test": (TEST_START, TEST_END)}
    if split not in bounds:
        raise ValueError(f"no evaluation window for split {split!r}")
    start, end = bounds[split]
    end_excl = (pd.Timestamp(end, tz="UTC") + pd.Timedelta(minutes=1)).isoformat()
    return Window(split, _utc(start), end_excl, f"the whole {split} split")


def evaluation_windows(mode: str, allow_test: bool = False) -> list[Window]:
    """The windows a run forecasts: the validation window of the mode; in final mode also the
    whole test split, which the model steps forecast and only the step `test` scores (D-32).
    `allow_test` is the invocation flag of that step; outside final it is refused."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {MODES}")
    if allow_test and mode != "final":
        raise PermissionError("the test split is evaluated only in final mode (D-08, G-1)")
    windows = [local_days_window(*SMOKE_WINDOW_LOCAL) if mode == "smoke" else split_window("val")]
    if mode == "final":
        windows.append(split_window("test"))
    return windows


def check_final_scope(mode: str, targets=None, horizons=None, seeds=None) -> None:
    """X39: a final run is the whole v6 scope; it is built step by step with --only and
    --resume, never by a slice of targets, horizons or seeds."""
    if mode != "final":
        return
    sliced = []
    if targets is not None and list(targets) != list(TARGETS):
        sliced.append(f"targets {list(targets)}")
    if horizons is not None and [int(h) for h in horizons] != list(HORIZONS):
        sliced.append(f"horizons {list(horizons)}")
    if seeds is not None:
        sliced.append(f"seeds {list(seeds)}")
    if sliced:
        raise ValueError("a final run takes the whole v6 scope (D-08, X39); refused: "
                         + ", ".join(sliced))


def resolve(mode: str, targets=None, horizons=None, seeds=None, allow_test: bool = False
            ) -> dict:
    """The resolved configuration of a run, written to its config.yaml. `allow_test` records
    whether the run forecasts the test window (final runs, D-32); the invocation flag
    --allow-test only checks that the mode permits it, and does not change the configuration,
    so the second phase of a final run resumes the same run."""
    check_final_scope(mode, targets, horizons, seeds)
    evaluation_windows(mode, allow_test)  # refuses --allow-test outside final
    targets = list(targets or TARGETS)
    horizons = [int(h) for h in (horizons or HORIZONS)]
    bad = [t for t in targets if t not in TARGETS] + [h for h in horizons if h not in HORIZONS]
    if bad:
        raise ValueError(f"unknown target or horizon: {bad}")
    return {
        "mode": mode,
        "windows": [w.as_dict() for w in evaluation_windows(mode)],
        "allow_test": mode == "final",
        "targets": targets,
        "horizons": horizons,
        "seeds": [int(s) for s in seeds] if seeds else None,
        "budgets": BUDGETS[mode],
        "xgb_threads": XGB_THREADS,
    }
