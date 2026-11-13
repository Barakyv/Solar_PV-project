"""XGB-Opt (PROJECT_MASTER.md §6.11; docs/stage9_plan.md §13.4, §14).

The model code of the runner's `xgb` step (`python -m solarpv.pipeline run --only xgb`) and of
the checks of §14 (`python -m solarpv.models.xgb_checks`). Rules from the W2 self-review:

- Inputs (M12a): the allow-list of `model_inputs.json` per target x horizon, each feature checked
  against the stage-7 pool and the leakage rules; the feature order is recorded.
- Monotone constraints (M7): the explicit D-15 list (`ghi_clearsky`, `poa_global_clearsky`,
  `sun_elevation`), matched by exact name and passed to XGBoost as a name -> sign map. The
  setting per target comes from `configs/monotone_decision.yaml` (§14.3): A, +1 on the D-15
  features present; B, none.
- No imputation (M8): NaN goes to XGBoost, which learns a default branch per split; the NaN
  count of every feature is logged on the fit, early-stopping and predicted rows.
- Rows (M11, §14.2): indirect, train rows not warm-up, not op-limited, k valid and theo > 50 kW;
  direct, not warm-up, not op-limited, AC present. Early stopping: 50 rounds on the RMSE in the
  training unit over the tuning window's rows with the same mask; the last evaluation set
  decides; no refit on train + validation (D-10).
- Objective (§14.1): daytime nRMSE in kW (% of 3,449 kW) on the daytime hours of the tuning
  window with AC truth, op-limited hours included; the indirect forecast is converted with
  `evaluate.to_kw` only (X12). The k-unit nRMSE is logged for every trial.
- Predictions (M9, M10): the raw model output (k is never clipped); the booster is saved and
  every prediction uses an explicit `iteration_range=(0, best_iteration + 1)`.
- Seeds (M12c): the seed drives both the TPE sampler and XGBoost's `random_state`.
- L-09 (§14.4, §15): where `configs/mos_decision.yaml` adopts it, the kW forecast of the final
  fit is corrected on daytime hours with clip(a + b * y_hat, 0, capacity); a and b come from
  out-of-fold train predictions of the same model, refitted in every run. One implementation for
  the step and the check (X14); for k the corrected kW is divided by theo again (X15).
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
import yaml  # noqa: E402

from solarpv.config.constants import (  # noqa: E402
    AC_CAPACITY_KW,
    MONOTONE_DECISION_FILE,
    MOS_DECISION_FILE,
    THEO_MIN_KW,
    TIMEZONE_LOCAL,
    XGB_THREADS,
)
from solarpv.config.paths import OUTPUTS_DIR  # noqa: E402
from solarpv.evaluation.evaluate import to_kw  # noqa: E402
from solarpv.evaluation.units import nrmse_k, nrmse_kw  # noqa: E402, F401  (X26)
from solarpv.features.leakage import assert_no_leakage  # noqa: E402
from solarpv.selection.common import date_folds  # noqa: E402
from solarpv.selection.common import pool as stage7_pool  # noqa: E402

MODEL = "xgb_sota"
TARGETS = ("indirect", "direct")
TARGET_COLUMN = {"indirect": "target_k", "direct": "site_total_AC_kW"}
TRUTH_KW = "site_total_AC_kW"
UNIT = {"indirect": "k", "direct": "kW"}
D15_FEATURES = ("ghi_clearsky", "poa_global_clearsky", "sun_elevation")
D15_SIGN = 1
SETTINGS = {"A": "+1 on the D-15 features present (v6 §5.1)", "B": "no constraint"}
EARLY_STOPPING_ROUNDS = 50
SMOKE_TRAIN_DAYS = 56  # smoke mode trains on the last 56 local days of train (§13.4)
MODEL_INPUTS_FILE = OUTPUTS_DIR / "07_feature_selection" / "model_inputs.json"
FIXED_PARAMS = {"tree_method": "hist", "n_jobs": XGB_THREADS, "objective": "reg:squarederror",
                "eval_metric": "rmse", "verbosity": 0}
# §6.11; min_child_weight is searched as an integer, like the other bracketed integer ranges
SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 200, "high": 2000, "log": False},
    "max_depth": {"type": "int", "low": 3, "high": 10, "log": False},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0, "log": False},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
    "min_child_weight": {"type": "int", "low": 1, "high": 20, "log": False},
    "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
}
# automatic warnings (docs/stage9_plan.md §3.5)
EDGE_FRACTION = 0.02  # a best parameter within 2 % of a bound (in its sampling scale)
GAP_RATIO_WARN = 2.0  # validation RMSE more than twice the train RMSE at the best iteration
SEED_SPREAD_WARN_PP = 0.15  # the acceptance rule of stage 10
REFIT_TOL_PP = 1e-9


# ── inputs and constraints ──────────────────────────────────────────────────


def load_model_inputs(path: Path = MODEL_INPUTS_FILE) -> dict[str, list[str]]:
    """`<target>_D<h>` -> the feature list of stage 7 (D-13, R7-04), in its recorded order."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: list(entry["features"]) for key, entry in data.items()}


def input_features(table: pd.DataFrame, target: str, h: int,
                   inputs: dict[str, list[str]]) -> list[str]:
    """The allow-list of the target x horizon (M12a): every feature in the stage-7 pool of the
    horizon (contract + admitted GFS candidates), none leaky, no duplicate."""
    key = f"{target}_D{h}"
    if key not in inputs:
        raise KeyError(f"model_inputs.json has no entry {key}")
    feats = list(inputs[key])
    if len(set(feats)) != len(feats) or not feats:
        raise ValueError(f"{key}: empty or duplicated feature list")
    allowed = set(stage7_pool(table, h))  # raises on a leaky name or a missing column
    outside = [f for f in feats if f not in allowed]
    if outside:
        raise ValueError(f"{key}: not in the stage-7 pool of D+{h}: {outside}")
    assert_no_leakage(feats)
    return feats


def load_monotone_decision(path: Path = MONOTONE_DECISION_FILE) -> dict:
    """The D-15 decision (§14.3). A missing file is an error: the xgb step never guesses."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing: D-15 is decided by "
                                "python -m solarpv.models.xgb_checks d15 (§14.3)")
    decision = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = decision.get("settings", {})
    for target in TARGETS:
        if settings.get(target) not in SETTINGS:
            raise ValueError(f"{path.name}: setting of {target} must be one of {list(SETTINGS)}")
    if tuple(decision.get("features", ())) != D15_FEATURES:
        raise ValueError(f"{path.name}: the D-15 list differs from {D15_FEATURES}")
    return decision


def constraint_map(feats: list[str], setting: str) -> dict[str, int]:
    """Name -> sign, by exact name (M7): `fcst_gfs_ghi`, `irradiance_lag_3d` or
    `irradiance_clear_index_prevday` are never constrained."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown monotone setting {setting!r}")
    if setting == "B":
        return {}
    return {f: D15_SIGN for f in feats if f in D15_FEATURES}


# ── rows ────────────────────────────────────────────────────────────────────


def fit_mask(table: pd.DataFrame, target: str) -> pd.Series:
    """Rows a model may learn from (M11, D-22, L-04), in any split."""
    base = ~table["warmup"].astype(bool) & ~table["op_limited_flag"].astype(bool)
    if target == "indirect":
        return base & table["k_valid"].astype(bool) & (table["theo_kw"] > THEO_MIN_KW)
    if target == "direct":
        return base & table[TRUTH_KW].notna()
    raise ValueError(f"unknown target {target!r}")


def local_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()


def train_scope(table: pd.DataFrame, smoke: bool = False) -> pd.Series:
    """The train rows a run learns from, before the fit mask: all of train; in smoke mode only
    the last SMOKE_TRAIN_DAYS local days of train."""
    rows = table["split"] == "train"
    if smoke:
        days = pd.Series(local_day(table.index), index=table.index)
        last = days[rows].max()
        rows = rows & (days >= last - pd.Timedelta(days=SMOKE_TRAIN_DAYS - 1))
    return rows


def training_rows(table: pd.DataFrame, target: str, smoke: bool = False) -> pd.Series:
    """Fit rows of train; in smoke mode only the last SMOKE_TRAIN_DAYS local days of train."""
    return train_scope(table, smoke) & fit_mask(table, target)


def window_rows(table: pd.DataFrame, window) -> pd.Series:
    """Rows of `window.split` with start <= timestamp < end (a pipeline.config.Window)."""
    inside = (table.index >= window.start_ts) & (table.index < window.end_ts)
    return (table["split"] == window.split) & inside


def stopping_rows(table: pd.DataFrame, target: str, window) -> pd.Series:
    return window_rows(table, window) & fit_mask(table, target)


def score_rows(table: pd.DataFrame, window) -> pd.Series:
    """§14.1: daytime hours of the window with AC truth, op-limited hours included."""
    return window_rows(table, window) & table["daytime"].astype(bool) & table[TRUTH_KW].notna()


def nan_counts(X: pd.DataFrame) -> dict[str, int]:
    counts = X.isna().sum()
    return {k: int(v) for k, v in counts[counts > 0].items()}


# ── one problem: target x horizon ───────────────────────────────────────────


@dataclass
class Problem:
    target: str
    horizon: int
    features: list[str]
    constraints: dict[str, int]
    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_stop: pd.DataFrame
    y_stop: pd.Series
    X_score: pd.DataFrame
    theo_score: np.ndarray
    truth_kw: np.ndarray
    k_score: np.ndarray
    mean_k: float
    rows: dict = field(default_factory=dict)
    nan: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.target}_D{self.horizon}"


def build_problem(table: pd.DataFrame, target: str, h: int, feats: list[str],
                  constraints: dict[str, int], window, mean_k: float,
                  smoke: bool = False) -> Problem:
    y = table[TARGET_COLUMN[target]]
    fit = training_rows(table, target, smoke)
    stop = stopping_rows(table, target, window)
    score = score_rows(table, window)
    for name, rows in (("fit", fit), ("early-stopping", stop), ("scoring", score)):
        if not rows.any():
            raise ValueError(f"{target} D+{h}: no {name} rows")
    if y[fit].isna().any() or y[stop].isna().any():
        raise ValueError(f"{target} D+{h}: missing target on fit or early-stopping rows")
    X_fit, X_stop, X_score = (table.loc[r, feats] for r in (fit, stop, score))
    truth = table.loc[score, TRUTH_KW].to_numpy(float)
    k_true = table.loc[score, "target_k"].to_numpy(float)
    if not (np.isfinite(truth).all() and np.isfinite(k_true).all()):
        raise ValueError(f"{target} D+{h}: missing truth on scoring rows")
    return Problem(
        target, h, list(feats), dict(constraints), X_fit, y[fit], X_stop, y[stop], X_score,
        table.loc[score, "theo_kw"].to_numpy(float), truth, k_true, float(mean_k),
        rows={"fit": int(fit.sum()), "early_stopping": int(stop.sum()),
              "scoring": int(score.sum()),
              "scoring_op_limited": int((score & table["op_limited_flag"].astype(bool)).sum())},
        nan={"fit": nan_counts(X_fit), "early_stopping": nan_counts(X_stop),
             "scoring": nan_counts(X_score)})


# ── training and prediction ─────────────────────────────────────────────────


def xgb_params(params: dict, seed: int, constraints: dict[str, int],
               n_estimators: int | None = None) -> dict:
    """The full parameter set passed to XGBoost: searched + fixed + seed + constraints. A new
    dict every time (X13)."""
    out = {**params, **FIXED_PARAMS, "random_state": int(seed)}
    if n_estimators is not None:
        out["n_estimators"] = int(n_estimators)
    if constraints:
        out["monotone_constraints"] = dict(constraints)
    return out


@dataclass
class Fit:
    booster: xgb.Booster
    best_iteration: int
    n_rounds: int  # best_iteration + 1: the rounds every prediction uses
    rounds_trained: int  # best + the patience rounds that did not improve
    n_estimators: int
    curve: pd.DataFrame  # round, train_rmse, val_rmse (training unit)

    @property
    def best_val_rmse(self) -> float:
        return float(self.curve["val_rmse"].iloc[self.best_iteration])

    @property
    def best_train_rmse(self) -> float:
        return float(self.curve["train_rmse"].iloc[self.best_iteration])

    @property
    def cap_hit(self) -> bool:
        """The best iteration is the last allowed round: the trial may not have converged."""
        return self.best_iteration >= self.n_estimators - 1


def fit_early_stopping(problem: Problem, params: dict, seed: int) -> Fit:
    """One fit with early stopping on the early-stopping rows (the last evaluation set)."""
    full = xgb_params(params, seed, problem.constraints)
    model = xgb.XGBRegressor(**full, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(problem.X_fit, problem.y_fit,
              eval_set=[(problem.X_fit, problem.y_fit), (problem.X_stop, problem.y_stop)],
              verbose=False)
    ev = model.evals_result()
    curve = pd.DataFrame({"round": np.arange(len(ev["validation_1"]["rmse"])),
                          "train_rmse": ev["validation_0"]["rmse"],
                          "val_rmse": ev["validation_1"]["rmse"]})
    best = int(model.best_iteration)
    booster = model.get_booster()
    return Fit(booster, best, best + 1, int(booster.num_boosted_rounds()),
               int(full["n_estimators"]), curve)


def fit_fixed(X: pd.DataFrame, y: pd.Series, params: dict, seed: int,
              constraints: dict[str, int], n_rounds: int) -> xgb.Booster:
    """A fit with a fixed number of rounds and no evaluation set (L-09, T-61)."""
    model = xgb.XGBRegressor(**xgb_params(params, seed, constraints, n_estimators=n_rounds))
    model.fit(X, y, verbose=False)
    return model.get_booster()


def predict(booster: xgb.Booster, X: pd.DataFrame, n_rounds: int) -> np.ndarray:
    """Raw model output with an explicit iteration range (M10); NaN stays NaN in the input."""
    rounds = int(booster.num_boosted_rounds())
    if not 0 < n_rounds <= rounds:
        raise ValueError(f"n_rounds {n_rounds} outside 1..{rounds}")
    dm = xgb.DMatrix(X, missing=np.nan)
    return booster.predict(dm, iteration_range=(0, int(n_rounds))).astype(float)


def booster_constraints(booster: xgb.Booster) -> str:
    """The monotone constraints as XGBoost applied them, in the feature order, e.g. '(1,0,0)'."""
    cfg = json.loads(booster.save_config())
    return str(cfg["learner"]["gradient_booster"].get("tree_train_param", {})
               .get("monotone_constraints", "()"))


def score_values(problem: Problem, pred) -> dict:
    """§14.1 of a forecast on the problem's scoring rows (target unit), and the k-unit value."""
    out = {"value_kw": nrmse_kw(problem.target, pred, problem.theo_score, problem.truth_kw)}
    if problem.target == "indirect":
        out["value_k"] = nrmse_k(pred, problem.k_score, problem.mean_k)
    return out


def score(problem: Problem, fit: Fit) -> dict:
    return score_values(problem, predict(fit.booster, problem.X_score, fit.n_rounds))


# ── the search ──────────────────────────────────────────────────────────────


def suggest(trial) -> dict:
    out = {}
    for name, s in SEARCH_SPACE.items():
        if s["type"] == "int":
            out[name] = trial.suggest_int(name, s["low"], s["high"], log=s["log"])
        else:
            out[name] = trial.suggest_float(name, s["low"], s["high"], log=s["log"])
    return out


def edge_params(params: dict, fraction: float = EDGE_FRACTION) -> list[str]:
    """Searched parameters within `fraction` of a bound, in their sampling scale (§3.5)."""
    edges = []
    for name, s in SEARCH_SPACE.items():
        v, lo, hi = float(params[name]), float(s["low"]), float(s["high"])
        pos = (np.log(v / lo) / np.log(hi / lo)) if s["log"] else (v - lo) / (hi - lo)
        if pos <= fraction or pos >= 1 - fraction:
            edges.append(name)
    return edges


def run_study(problem: Problem, seed: int, n_trials: int, storage, study_name: str,
              on_trial=None, state: dict | None = None, callbacks=None):
    """Optuna TPE with TPESampler(seed=seed); every trial uses random_state=seed (M12c). The
    value is the nRMSE in kW of §14.1; the k-unit value, the best iteration and the rounds are
    trial attributes. An exception in a trial stops the study (no silent failure).

    `state` (§23) is the finished trials and the sampler of an interrupted study: they are
    replayed into `storage` and the study runs the trials that are left. `callbacks` run after
    every finished trial, after `on_trial`."""
    import optuna

    from solarpv.models import checkpoint

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = checkpoint.replayed_study(storage, study_name, seed, state)

    def objective(trial) -> float:
        params = suggest(trial)
        t0 = time.perf_counter()
        fit = fit_early_stopping(problem, params, seed)
        values = score(problem, fit)
        attrs = {**values, "best_iteration": fit.best_iteration, "n_rounds": fit.n_rounds,
                 "rounds_trained": fit.rounds_trained, "cap_hit": fit.cap_hit,
                 "best_val_rmse": fit.best_val_rmse, "best_train_rmse": fit.best_train_rmse,
                 "duration_s": round(time.perf_counter() - t0, 2)}
        for k, v in attrs.items():
            trial.set_user_attr(k, v)
        return values["value_kw"]

    chain = ([on_trial] if on_trial is not None else []) + list(callbacks or [])
    left = int(n_trials) - (len(state["trials"]) if state else 0)
    if left > 0:
        study.optimize(objective, n_trials=left, callbacks=chain or None)
    return study


def trials_frame(study, target: str, h: int, seed: int) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        rows.append({"target": target, "horizon": h, "seed": seed, "trial": t.number,
                     "state": t.state.name, "value_kw": t.value,
                     "value_k": t.user_attrs.get("value_k", np.nan),
                     **{k: v for k, v in t.user_attrs.items() if k not in ("value_kw",
                                                                           "value_k")},
                     **{f"param_{k}": v for k, v in t.params.items()}})
    return pd.DataFrame(rows)


# ── L-09: bias correction (MOS) of the kW forecast (§14.4, §15) ─────────────


def uncorrected_name(seed: int) -> str:
    """The raw booster output of the corrected targets (X16). Not a `predictions_*` file: the
    `evaluate` step scores only those."""
    return f"uncorrected_{MODEL}_seed{int(seed)}.csv"


def load_mos_decision(path: Path = MOS_DECISION_FILE) -> dict | None:
    """The L-09 decision (§14.4): `adopted` true or false per target. None when the file does
    not exist yet (L-09 is decided on a dev run; the step allows that outside final mode)."""
    path = Path(path)
    if not path.exists():
        return None
    decision = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if decision.get("decision") != "L-09":
        raise ValueError(f"{path.name}: not an L-09 decision")
    adopted = decision.get("adopted") or {}
    for target in TARGETS:
        if not isinstance(adopted.get(target), bool):
            raise ValueError(f"{path.name}: adopted.{target} must be true or false")
    return decision


def mos_rows(table: pd.DataFrame) -> pd.Series:
    """Rows the correction is fitted on: daytime, AC truth, not op-limited, not warm-up (D-22)."""
    return (table["daytime"].astype(bool) & table[TRUTH_KW].notna()
            & ~table["op_limited_flag"].astype(bool) & ~table["warmup"].astype(bool))


def mos_folds(table: pd.DataFrame, scope: pd.Series) -> list[tuple[pd.Series, pd.Series]]:
    """The five expanding date folds of stage 7 (3-day gap) over the local days of `scope`."""
    days = pd.Series(local_day(table.index[scope]), index=table.index[scope])
    return [(fit.reindex(table.index, fill_value=False),
             held.reindex(table.index, fill_value=False)) for fit, held in date_folds(days)]


def ols(pred_kw: np.ndarray, truth_kw: np.ndarray) -> tuple[float, float]:
    """y = a + b * y_hat by ordinary least squares."""
    A = np.column_stack([np.ones_like(pred_kw), pred_kw])
    (a, b), *_ = np.linalg.lstsq(A, truth_kw, rcond=None)
    return float(a), float(b)


@dataclass
class Mos:
    a: float  # kW
    b: float
    n_oof: int  # out-of-fold hours the OLS was fitted on
    folds: list[dict]  # fit and held-out rows of every fold

    def as_dict(self) -> dict:
        return {"a_kw": self.a, "b": self.b, "n_oof": self.n_oof, "folds": self.folds}


def oof_folds(table: pd.DataFrame, target: str, feats: list[str], params: dict, seed: int,
              constraints: dict[str, int], n_rounds: int, scope: pd.Series,
              usable: pd.Series, what: str = "the out-of-fold fits"):
    """The out-of-fold loop, one implementation for the L-09 correction and the CV step (X68):
    for each of the five expanding date folds of `scope`, the model (its parameters, seed,
    constraints and rounds, fixed; no early stopping) is fitted on the fold's fit rows and
    predicts the fold's held rows (the `usable` rows of the held block). Yields
    (fold, fit rows, held rows, raw prediction on the held rows, in the training unit)."""
    y = table[TARGET_COLUMN[target]]
    fit_all = scope & fit_mask(table, target)
    for i, (fold_fit, fold_held) in enumerate(mos_folds(table, scope)):
        rows, held = fold_fit & fit_all, fold_held & usable
        if not rows.any() or not held.any():
            raise ValueError(f"{target}: fold {i} of {what} has no fit or held rows")
        booster = fit_fixed(table.loc[rows, feats], y[rows], params, seed, constraints, n_rounds)
        yield i, rows, held, predict(booster, table.loc[held, feats], n_rounds)


def fit_mos(table: pd.DataFrame, target: str, feats: list[str], params: dict, seed: int,
            constraints: dict[str, int], n_rounds: int, scope: pd.Series) -> Mos:
    """Out-of-fold kW predictions of the base model (its parameters, seed, constraints and
    rounds, fixed; no early stopping) on the date folds of `scope`, then OLS (§14.4). Only train
    rows enter: neither the features nor the labels of validation are used (A-14)."""
    preds, truth, folds = [], [], []
    for i, rows, held, pred in oof_folds(table, target, feats, params, seed, constraints,
                                         n_rounds, scope, mos_rows(table),
                                         what="the L-09 correction"):
        preds.append(to_kw(target, pred, table.loc[held, "theo_kw"].to_numpy()))
        truth.append(table.loc[held, TRUTH_KW].to_numpy(float))
        folds.append({"fold": i, "fit_rows": int(rows.sum()), "held_rows": int(held.sum())})
    preds, truth = np.concatenate(preds), np.concatenate(truth)
    a, b = ols(preds, truth)
    return Mos(a, b, int(preds.size), folds)


def corrected_kw(mos: Mos, kw) -> np.ndarray:
    return np.clip(mos.a + mos.b * np.asarray(kw, dtype=float), 0.0, AC_CAPACITY_KW)


def apply_mos(target: str, y_pred, theo, daytime, mos: Mos) -> np.ndarray:
    """The corrected forecast in the target's unit (X15): on daytime hours
    clip(a + b * to_kw(y_pred), 0, capacity), for k divided by theo again, so that
    `evaluate.to_kw` returns the corrected kW; other hours unchanged."""
    out = np.asarray(y_pred, dtype=float).copy()
    theo = np.asarray(theo, dtype=float)
    day = np.asarray(daytime, dtype=bool)
    kw = corrected_kw(mos, to_kw(target, out[day], theo[day]))
    if target == "indirect":
        bad = ~(theo[day] > 0)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} daytime hours with theo <= 0: the corrected kW "
                             "cannot be written as k")
        out[day] = kw / theo[day]
    else:
        out[day] = kw
    return out


# ── figures ─────────────────────────────────────────────────────────────────


def _grid(keys):
    import matplotlib.pyplot as plt

    targets = [t for t in TARGETS if any(k[0] == t for k in keys)]
    horizons = sorted({k[1] for k in keys})
    fig, axes = plt.subplots(len(targets), len(horizons), squeeze=False,
                             figsize=(4.2 * len(horizons), 3.2 * len(targets)))
    return plt, fig, {(t, h): axes[i][j] for i, t in enumerate(targets)
                      for j, h in enumerate(horizons)}


def plot_learning_curves(curves: pd.DataFrame, best: dict, path: Path) -> None:
    """Train and validation RMSE per round of every final fit, with the best iteration."""
    keys = sorted({(t, int(h)) for t, h in zip(curves["target"], curves["horizon"],
                                                strict=True)})
    plt, fig, ax = _grid(keys)
    for (target, h, seed), g in curves.groupby(["target", "horizon", "seed"]):
        a = ax[(target, int(h))]
        a.plot(g["round"], g["train_rmse"], lw=1, label=f"train, seed {seed}")
        a.plot(g["round"], g["val_rmse"], lw=1.4, label=f"validation, seed {seed}")
        b = best[f"{target}_D{h}_seed{seed}"]["best_iteration"]
        a.axvline(b, color="0.4", ls=":", lw=1)
        a.set_title(f"{target} D+{h}", fontsize=9)
        a.set_xlabel("boosting round")
        a.set_ylabel(f"RMSE [{UNIT[target]}]")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_histories(trials: pd.DataFrame, path: Path) -> None:
    """The value of every trial and the best value so far (optimisation history)."""
    keys = sorted({(t, int(h)) for t, h in zip(trials["target"], trials["horizon"],
                                                strict=True)})
    plt, fig, ax = _grid(keys)
    for (target, h, seed), g in trials.groupby(["target", "horizon", "seed"]):
        a = ax[(target, int(h))]
        g = g.sort_values("trial")
        a.plot(g["trial"], g["value_kw"], "o", ms=3, label=f"trial, seed {seed}")
        a.plot(g["trial"], g["value_kw"].cummin(), "-", lw=1.2, label=f"best so far, seed {seed}")
        a.set_title(f"{target} D+{h}", fontsize=9)
        a.set_xlabel("trial")
        a.set_ylabel("validation nRMSE [% of 3,449 kW]")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
