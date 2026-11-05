"""Modes, budgets, evaluation windows and the step order of the runner (docs/stage9_plan.md §3,
§10.3).

Three modes:
  smoke  does the code run? Minimal budgets; evaluated on a 28-day validation window that
         contains the spring DST change (2024-03-29). Test rows are never loaded.
  dev    stage 9: the development budgets (D-17); evaluated on the whole validation split. Test
         rows are never loaded.
  final  stage 10: the budgets of v6 (D-08); only on the commit tagged `g1-frozen` with a clean
         tree. Validation always; the test split only with --allow-test (stage 11).
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

# Budgets per mode. W1 runs no model; the values are recorded in every run's config.yaml and
# used by the model steps of W2 and W4. dev = D-17 (and §4.3 of the plan), final = D-08.
BUDGETS = {
    "smoke": {
        "xgb": {"trials": 1, "seeds": [DEV_SEED]},
        "bilstm": {"trials": 1, "epochs_trial": 2, "epochs_final": 2, "seeds": [DEV_SEED]},
        "tft": {"trials": 1, "epochs": 2, "seeds": [DEV_SEED]},
        "patchtst": {"epochs": 2, "seeds": [DEV_SEED]},
    },
    "dev": {
        "xgb": {"trials": DEV_OPTUNA_TRIALS, "seeds": [DEV_SEED]},
        "bilstm": {"trials": 5, "epochs_trial": 10, "epochs_final": 10, "seeds": [DEV_SEED]},
        "tft": {"trials": 5, "epochs": 10, "seeds": [DEV_SEED]},
        "patchtst": {"epochs": 10, "seeds": [DEV_SEED]},
    },
    "final": {
        "xgb": {"trials": FINAL_OPTUNA_TRIALS, "seeds": list(RANDOM_SEEDS)},
        "bilstm": {"trials": 60, "epochs_trial": 25, "epochs_final": 60, "patience": 10,
                   "seeds": list(RANDOM_SEEDS)},
        "tft": {"trials": 20, "seeds": [DEV_SEED]},
        "patchtst": {"seeds": [DEV_SEED]},
    },
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
    StepSpec("select", "7", "feature selection (rerun of stage 7)", "CPU",
             "stage-6 and 6b tables", "model_inputs.json", "W5"),
    StepSpec("baselines", "8", "persistence, climatology, clear sky x MEAN_DAYTIME_K", "CPU",
             "truth table, MEAN_DAYTIME_K", "predictions_<reference>.csv", "W1"),
    StepSpec("xgb", "9/10", "XGB-Opt", "CPU", "tables, model_inputs.json",
             "predictions, training logs, study", "W2"),
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
    StepSpec("explain", "11", "SHAP, attention, hyper-parameter importance", "CPU/GPU",
             "trained models", "explanation tables", "W5"),
    StepSpec("report", "11", "one summary of the run with automatic warnings", "CPU",
             "the run's records", "report", "W5"),
)
STEP_ORDER = tuple(s.name for s in STEPS)
SPEC = {s.name: s for s in STEPS}
MODEL_STEPS = ("baselines", "xgb", "bilstm", "tft96", "tft168", "tftbase", "patchtst")


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
    """Validation window of the mode, plus the whole test split in final mode with allow_test."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {MODES}")
    if allow_test and mode != "final":
        raise PermissionError("the test split is evaluated only in final mode (D-08, G-1)")
    windows = [local_days_window(*SMOKE_WINDOW_LOCAL) if mode == "smoke" else split_window("val")]
    if allow_test:
        windows.append(split_window("test"))
    return windows


def resolve(mode: str, targets=None, horizons=None, seeds=None, allow_test: bool = False
            ) -> dict:
    """The resolved configuration of a run, written to its config.yaml."""
    targets = list(targets or TARGETS)
    horizons = [int(h) for h in (horizons or HORIZONS)]
    bad = [t for t in targets if t not in TARGETS] + [h for h in horizons if h not in HORIZONS]
    if bad:
        raise ValueError(f"unknown target or horizon: {bad}")
    return {
        "mode": mode,
        "windows": [w.as_dict() for w in evaluation_windows(mode, allow_test)],
        "allow_test": allow_test,
        "targets": targets,
        "horizons": horizons,
        "seeds": [int(s) for s in seeds] if seeds else None,
        "budgets": BUDGETS[mode],
        "xgb_threads": XGB_THREADS,
    }
