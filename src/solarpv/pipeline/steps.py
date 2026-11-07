"""The work steps implemented so far: `baselines` and `evaluate` (W1), `xgb` (W2)
(docs/stage9_plan.md §3.2, §10.3, §13.4). Every step gets a `Context` and writes only into
`ctx.out`, a temporary folder that the runner renames when the step succeeds.

Rules (the self-reviews of §10 and §13):
- the truth of smoke and dev runs is read without the test split (X3); the test split is read
  only to evaluate it, in final mode with allow_test;
- the evaluation window: rows of the split inside the window are scored; earlier rows of the split
  are context for the persistence reference; later rows are dropped (§10.3);
- `evaluate` reads only the prediction files of the run's completed model steps, each checked
  against the hash in its step record (M1, X1), and in smoke and dev refuses any row in the test
  split (X7);
- `xgb` reads its tables with `pipeline.data.model_table` (X10), its inputs from
  `model_inputs.json` and its constraints from the D-15 decision; it refuses to run without that
  decision. It tunes on the validation window of the mode and predicts every hour of every
  window of the run; the test window (final, allow_test) is only predicted. Where the L-09
  decision adopts it, the forecast of the final fit is corrected (§15): the coefficients are
  refitted in the run, out of fold on its train rows, and the raw forecast is kept in
  `uncorrected_*` files. Without the L-09 decision a smoke or dev run writes the raw forecast
  with a warning; a final run refuses.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.constants import (
    DERIVED_CONSTANTS_FILE,
    MONOTONE_DECISION_FILE,
    MOS_DECISION_FILE,
    TEST_START,
    mean_daytime_k,
)
from solarpv.config.provenance import sha256
from solarpv.config.runlog import RunLog
from solarpv.config.site import load_site
from solarpv.evaluation import evaluate as E
from solarpv.evaluation.truth import FEATURES_FILE, load_truth
from solarpv.models.baselines import reference_predictions
from solarpv.pipeline import data
from solarpv.pipeline.config import MODEL_STEPS, Window


class RunnerError(RuntimeError):
    """A rule of the runner is broken (missing input, test access, modified file)."""


@dataclass
class Context:
    run_id: str
    run_dir: Path
    mode: str
    config: dict
    windows: list[Window]
    out: Path
    log: RunLog
    completed: dict[str, dict] = field(default_factory=dict)  # step -> record (step.json)
    failed: list[str] = field(default_factory=list)  # steps of the run whose last try failed
    command: str = ""


@dataclass
class Result:
    outputs: list[Path]
    values: dict = field(default_factory=dict)
    summary: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


# ── truth and windows ───────────────────────────────────────────────────────


def truth_for(window: Window, path: Path | None = None) -> pd.DataFrame:
    """The truth table for one evaluation window: without test rows unless the window is the
    test split, restricted to the window."""
    exclude = () if window.split == "test" else ("test",)
    truth = load_truth(path or FEATURES_FILE, exclude_splits=exclude)
    if exclude and (truth["split"] == "test").any():  # load_truth checks this too
        raise RunnerError("test rows were loaded outside a test evaluation")
    return restrict_to_window(truth, window)


def restrict_to_window(truth: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Rows of `window.split` inside [start, end) keep their label; earlier rows of that split
    become '<split>_context' (they feed persistence only); rows from `end` on are dropped."""
    start, end = window.start_ts, window.end_ts
    in_split = truth["split"] == window.split
    inside = (truth.index >= start) & (truth.index < end)
    if not (in_split & inside).any():
        raise RunnerError(f"the window {window.label} has no row of the split {window.split}")
    if (inside & ~in_split.to_numpy()).any():
        raise RunnerError(f"the window {window.label} crosses the boundary of {window.split}")
    out = truth[truth.index < end].copy()
    before = (out["split"] == window.split) & (out.index < start)
    out.loc[before, "split"] = f"{window.split}_context"
    return out


def assert_no_test_rows(paths: list[Path]) -> None:
    """Smoke and dev: a prediction for a test hour means the model read test features (X7)."""
    first_test = pd.Timestamp(TEST_START, tz="UTC")
    for path in paths:
        ts = pd.to_datetime(pd.read_csv(path, usecols=["timestamp"])["timestamp"], utc=True)
        n = int((ts >= first_test).sum())
        if n:
            raise RunnerError(f"{path.name}: {n} rows in the test split in a development run")


# ── baselines ───────────────────────────────────────────────────────────────


def baselines_inputs(ctx: Context) -> list[Path]:
    return [FEATURES_FILE, DERIVED_CONSTANTS_FILE]


def run_baselines(ctx: Context) -> Result:
    """The three references (OP-40) for every row of each evaluation window, written as
    prediction files in the contract of §6.10, so they pass the checks every model passes."""
    mean_k = mean_daytime_k()
    targets, horizons = ctx.config["targets"], ctx.config["horizons"]
    frames = []
    for window in ctx.windows:
        with ctx.log.step(f"references on {window.split}, {window.label}"):
            truth = truth_for(window)
            refs = reference_predictions(truth, window.split, mean_k, horizons=horizons)
            refs = refs[refs["target"].isin(targets)]
            day = truth["daytime"] & (truth["split"] == window.split)
            ctx.log.note(f"{window.split}: {int((truth['split'] == window.split).sum())} hours "
                         f"in the window, {int(day.sum())} of them daytime; context rows "
                         f"{int((truth['split'] == window.split + '_context').sum())}")
            frames.append(refs)
    refs = pd.concat(frames, ignore_index=True)
    outputs = []
    for model, g in refs.groupby("model", sort=True):
        path = ctx.out / f"predictions_{model}.csv"
        g.sort_values(["target", "horizon", "timestamp"]).to_csv(
            path, index=False, lineterminator="\n")
        outputs.append(path)
        ctx.log.value(f"rows_{model}", len(g))
    return Result(outputs, values={"mean_daytime_k": mean_k, "rows": len(refs)},
                  summary=[f"{len(outputs)} reference files, {len(refs):,} rows"])


# ── xgb ─────────────────────────────────────────────────────────────────────


def _xgb():
    from solarpv.models import xgb_sota  # imported when the step runs: `check` stays fast

    return xgb_sota


def xgb_inputs(ctx: Context) -> list[Path]:
    X = _xgb()
    if not MONOTONE_DECISION_FILE.exists():
        raise RunnerError(f"{MONOTONE_DECISION_FILE.name} is missing: D-15 is decided first, by "
                          "python -m solarpv.models.xgb_checks d15 (docs/stage9_plan.md §14.3)")
    if ctx.mode == "final" and not MOS_DECISION_FILE.exists():
        raise RunnerError(f"{MOS_DECISION_FILE.name} is missing: a final run needs the L-09 "
                          "decision (python -m solarpv.models.xgb_checks l09 --run <dev run>, "
                          "docs/stage9_plan.md §14.4)")
    files = []
    for h in ctx.config["horizons"]:
        files += [data.feature_file(h), data.gfs_file(h)]
    mos = [MOS_DECISION_FILE] if MOS_DECISION_FILE.exists() else []
    return files + [X.MODEL_INPUTS_FILE, MONOTONE_DECISION_FILE, *mos, DERIVED_CONSTANTS_FILE]


def _tuning_window(ctx: Context) -> Window:
    tuning = [w for w in ctx.windows if w.split == "val"]
    if len(tuning) != 1:
        raise RunnerError("the xgb step needs exactly one validation window to tune on")
    return tuning[0]


MOS_COLUMNS = ["target", "horizon", "seed", "a_kw", "b", "n_oof", "value_kw_uncorrected",
               "value_kw_corrected", "value_k_uncorrected", "value_k_corrected"]


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _write_json(data_: dict, path: Path) -> Path:
    path.write_text(json.dumps(data_, indent=2, default=str) + "\n", encoding="utf-8",
                    newline="\n")
    return path


def run_xgb(ctx: Context) -> Result:
    """XGB-Opt per target x horizon x seed (docs/stage9_plan.md §13.4, §14.1-§14.2)."""
    import optuna

    X = _xgb()
    budget = ctx.config["budgets"]["xgb"]
    seeds = [int(s) for s in (ctx.config["seeds"] or budget["seeds"])]
    n_trials = int(budget["trials"])
    tuning = _tuning_window(ctx)
    with_test = any(w.split == "test" for w in ctx.windows)
    splits = ("train", "val", "test") if with_test else ("train", "val")
    mean_k = mean_daytime_k()
    inputs = X.load_model_inputs(X.MODEL_INPUTS_FILE)
    decision = X.load_monotone_decision(MONOTONE_DECISION_FILE)
    mos_decision = X.load_mos_decision(MOS_DECISION_FILE)
    if mos_decision is None:
        if ctx.mode == "final":
            raise RunnerError(f"{MOS_DECISION_FILE.name} is missing: a final run needs the L-09 "
                              "decision")
        ctx.log.warn(f"{MOS_DECISION_FILE.name} is missing: L-09 is not decided yet, so the "
                     "forecast is uncorrected (docs/stage9_plan.md §15)")
        adopted = {t: False for t in X.TARGETS}
    else:
        adopted = {t: bool(v) for t, v in mos_decision["adopted"].items()}
    models_dir = ctx.out / "models"
    models_dir.mkdir()
    db = ctx.out / "optuna.db"
    storage = optuna.storages.RDBStorage(f"sqlite:///{db.resolve().as_posix()}")
    train_text = (f"the last {X.SMOKE_TRAIN_DAYS} local days of train" if ctx.mode == "smoke"
                  else "all of train")
    ctx.log.note(f"budget {n_trials} trials, seeds {seeds}, tuning window {tuning.label}, "
                 f"training rows {train_text}; MEAN_DAYTIME_K {mean_k}; D-15 settings "
                 f"{decision['settings']}; L-09 adopted {adopted}")
    predictions = {s: [] for s in seeds}
    uncorrected = {s: [] for s in seeds}  # raw booster output of the corrected targets (X16)
    trial_frames, curves, best, problems, mos_rows = [], [], {}, {}, []
    try:
        for h in ctx.config["horizons"]:
            table = data.model_table(h, splits, allow_test=with_test)
            if not with_test and (table.index >= pd.Timestamp(TEST_START, tz="UTC")).any():
                raise RunnerError("test rows reached the xgb step in a development run")
            for target in ctx.config["targets"]:
                feats = X.input_features(table, target, h, inputs)
                setting = decision["settings"][target]
                constraints = X.constraint_map(feats, setting)
                problem = X.build_problem(table, target, h, feats, constraints, tuning, mean_k,
                                          smoke=ctx.mode == "smoke")
                key = problem.key
                ctx.log.note(f"{key}: {len(feats)} features; D-15 setting {setting}, "
                             f"constraints {constraints or 'none'}; rows {problem.rows}")
                for part, counts in problem.nan.items():
                    if counts:
                        ctx.log.note(f"{key}: NaN on the {part} rows (passed to XGBoost, M8): "
                                     f"{counts}")
                problems[key] = {"features": feats, "n_features": len(feats),
                                 "monotone_setting": setting,
                                 "monotone_constraints": constraints, "rows": problem.rows,
                                 "nan_counts": dict(problem.nan), "nan_counts_predicted": {}}
                for seed in seeds:
                    name = f"{key}_seed{seed}"
                    with ctx.log.step(f"{name}: {n_trials} trial(s)"):
                        def note(study, trial, name=name):
                            ctx.log.note(
                                f"{name} trial {trial.number}: {trial.value:.4f} % kW"
                                + (f", {trial.user_attrs['value_k']:.4f} % k"
                                   if "value_k" in trial.user_attrs else "")
                                + f", best iteration {trial.user_attrs['best_iteration']} of "
                                f"{trial.params['n_estimators']}, "
                                f"{trial.user_attrs['duration_s']} s")
                        study = X.run_study(problem, seed, n_trials, storage,
                                            f"{X.MODEL}_{name}", on_trial=note)
                        trials = X.trials_frame(study, target, h, seed)
                        trial_frames.append(trials)
                        done = int((trials["state"] == "COMPLETE").sum())
                        if done < n_trials:
                            ctx.log.warn(f"{name}: {done} of {n_trials} trials completed")
                        bt = study.best_trial
                        fit = X.fit_early_stopping(problem, bt.params, seed)  # the final fit
                        values = X.score(problem, fit)
                        diff = abs(values["value_kw"] - bt.value)
                        if diff > X.REFIT_TOL_PP:
                            ctx.log.warn(f"{name}: the final fit differs from its best trial "
                                         f"by {diff:.3g} pp (non-deterministic training?)")
                        booster_path = models_dir / f"{X.MODEL}_{name}.ubj"
                        fit.booster.save_model(booster_path)
                        curves.append(fit.curve.assign(target=target, horizon=h, seed=seed,
                                                       unit=X.UNIT[target]))
                        mos, forecast = None, dict(values)
                        if adopted[target]:
                            with ctx.log.step(f"{name}: L-09 correction, out-of-fold fits"):
                                mos = X.fit_mos(table, target, feats, bt.params, seed,
                                                constraints, fit.n_rounds,
                                                X.train_scope(table, ctx.mode == "smoke"))
                            raw_score = X.predict(fit.booster, problem.X_score, fit.n_rounds)
                            # every scoring row is a daytime hour (§14.1)
                            forecast = X.score_values(problem, X.apply_mos(
                                target, raw_score, problem.theo_score,
                                np.ones(raw_score.size, dtype=bool), mos))
                            ctx.log.note(f"{name}: L-09 a {mos.a:+.3f} kW, b {mos.b:.5f} on "
                                         f"{mos.n_oof} out-of-fold hours; tuning window "
                                         f"{values['value_kw']:.4f} -> "
                                         f"{forecast['value_kw']:.4f} % of capacity")
                            mos_rows.append({
                                "target": target, "horizon": h, "seed": seed, "a_kw": mos.a,
                                "b": mos.b, "n_oof": mos.n_oof,
                                "value_kw_uncorrected": values["value_kw"],
                                "value_kw_corrected": forecast["value_kw"],
                                "value_k_uncorrected": values.get("value_k", np.nan),
                                "value_k_corrected": forecast.get("value_k", np.nan)})
                        mos_record = (
                            {"applied": True, **mos.as_dict(),
                             "value_kw_uncorrected": values["value_kw"],
                             "value_kw_corrected": forecast["value_kw"]}
                            if mos is not None else
                            {"applied": False,
                             "reason": "not adopted by L-09" if mos_decision is not None
                             else "L-09 not decided yet"})
                        best[name] = {
                            "target": target, "horizon": h, "seed": seed,
                            "trial": bt.number, "searched": dict(bt.params),
                            "xgb_params": X.xgb_params(bt.params, seed, constraints),
                            "early_stopping_rounds": X.EARLY_STOPPING_ROUNDS,
                            "sampler": {"name": "TPESampler", "seed": seed},
                            "feature_order": feats, "monotone_setting": setting,
                            "monotone_constraints": constraints,
                            "monotone_constraints_xgb": X.booster_constraints(fit.booster),
                            "best_iteration": fit.best_iteration, "n_rounds": fit.n_rounds,
                            "rounds_trained": fit.rounds_trained,
                            "best_val_rmse": fit.best_val_rmse,
                            "best_train_rmse": fit.best_train_rmse,
                            "unit": X.UNIT[target], **values,
                            "best_trial_value_kw": bt.value,
                            "mos": mos_record,
                            **{f"{k}_forecast": v for k, v in forecast.items()},
                            "booster": booster_path.relative_to(ctx.out).as_posix()}
                        _xgb_warnings(ctx, X, name, fit, bt.params)
                        for w in ctx.windows:
                            rows = X.window_rows(table, w)
                            Xw = table.loc[rows, feats]
                            counts = X.nan_counts(Xw)
                            if counts:
                                problems[key]["nan_counts_predicted"][w.split] = counts
                            raw = X.predict(fit.booster, Xw, fit.n_rounds)
                            frame = pd.DataFrame({
                                "timestamp": table.index[rows],
                                "y_true": table.loc[rows, X.TARGET_COLUMN[target]].to_numpy(),
                                "y_pred": raw,
                                "target": target, "horizon": h, "model": X.MODEL,
                                "seed": seed, "split": w.split})
                            if mos is not None:
                                uncorrected[seed].append(frame)
                                frame = frame.assign(y_pred=X.apply_mos(
                                    target, raw, table.loc[rows, "theo_kw"].to_numpy(float),
                                    table.loc[rows, "daytime"].astype(bool).to_numpy(), mos))
                            predictions[seed].append(frame)
                        ctx.log.note(f"{name}: best trial {bt.number}, validation "
                                     f"{forecast['value_kw']:.4f} % of capacity"
                                     + (f" (uncorrected {values['value_kw']:.4f} %)"
                                        if mos is not None else "")
                                     + f", best iteration {fit.best_iteration} of "
                                     f"{fit.n_estimators}")
    finally:
        storage.remove_session()
        storage.engine.dispose()  # release the SQLite file before the folder is renamed
    outputs = []
    for seed, frames in predictions.items():
        frame = pd.concat(frames, ignore_index=True).sort_values(
            ["target", "horizon", "timestamp"])
        outputs.append(_write_csv(frame, ctx.out / f"predictions_{X.MODEL}_seed{seed}.csv"))
    for seed, frames in uncorrected.items():
        if frames:
            frame = pd.concat(frames, ignore_index=True).sort_values(
                ["target", "horizon", "timestamp"])
            outputs.append(_write_csv(frame, ctx.out / X.uncorrected_name(seed)))
    outputs.append(_write_csv(pd.DataFrame(mos_rows, columns=MOS_COLUMNS),
                              ctx.out / "mos_coefficients.csv"))
    trials = pd.concat(trial_frames, ignore_index=True)
    curves = pd.concat(curves, ignore_index=True)
    outputs.append(_write_csv(trials, ctx.out / "trials.csv"))
    outputs.append(_write_csv(curves, ctx.out / "learning_curves.csv"))
    outputs.append(_write_json(best, ctx.out / "best_params.json"))
    spread = _seed_spread(ctx, X, best)
    outputs.append(_write_json({
        "model": X.MODEL, "mode": ctx.mode, "budget": budget, "seeds": seeds,
        "tuning_window": tuning.as_dict(), "windows": [w.as_dict() for w in ctx.windows],
        "objective": "daytime nRMSE in kW (% of 3,449 kW) on the daytime hours of the tuning "
                     "window with AC truth, op-limited hours included; indirect converted "
                     "with evaluate.to_kw (docs/stage9_plan.md §14.1)",
        "early_stopping": f"{X.EARLY_STOPPING_ROUNDS} rounds on the RMSE in the training unit "
                          "over the tuning window's rows with the fit mask (§14.2)",
        "fit_rows": "indirect: train, not warm-up, not op-limited, k valid, theo > 50 kW; "
                    "direct: train, not warm-up, not op-limited, AC present (M11)"
                    + ("; smoke: the last 56 local days of train" if ctx.mode == "smoke"
                       else ""),
        "imputation": "none: NaN is passed to XGBoost (M8)",
        "fixed_params": X.FIXED_PARAMS, "search_space": X.SEARCH_SPACE,
        "mean_daytime_k": mean_k, "monotone_decision": decision,
        "monotone_decision_sha256": sha256(MONOTONE_DECISION_FILE),
        "mos": "L-09 where adopted: clip(a + b * kW forecast, 0, 3449) on daytime hours, a and b "
               "by OLS on out-of-fold train predictions of the final fit's model (five expanding "
               "date folds, 3-day gap, fixed rounds), refitted in this run; for k divided by "
               "theo again (docs/stage9_plan.md §14.4, §15)",
        "mos_adopted": adopted, "mos_decision": mos_decision,
        "mos_decision_sha256": sha256(MOS_DECISION_FILE) if mos_decision is not None else None,
        "seed_spread_pp": spread, "problems": problems,
        "optuna": optuna.__version__}, ctx.out / "xgb_config.json"))
    figures = ctx.out / "figures"
    figures.mkdir()
    X.plot_learning_curves(curves, best, figures / "learning_curves.png")
    X.plot_histories(trials, figures / "optimization_history.png")
    outputs += [figures / "learning_curves.png", figures / "optimization_history.png", db]
    outputs += sorted(models_dir.glob("*.ubj"))
    summary = [f"xgb ({ctx.mode}, {n_trials} trial(s), seeds {seeds}): nRMSE in kW on the "
               f"tuning window, {tuning.label}"]
    for name, b in best.items():
        summary.append(f"  {name:<22} {b['value_kw_forecast']:6.3f} %  best iteration "
                       f"{b['best_iteration']} of {b['xgb_params']['n_estimators']}"
                       + (f"  (L-09; uncorrected {b['value_kw']:.3f} %)"
                          if b["mos"]["applied"] else ""))
    return Result(outputs, values={"best": {k: v["value_kw_forecast"] for k, v in best.items()},
                                   "best_uncorrected": {k: v["value_kw"]
                                                        for k, v in best.items()},
                                   "trials": int(len(trials))},
                  summary=summary, notes={"seed_spread_pp": spread})


def _xgb_warnings(ctx: Context, X, name: str, fit, params: dict) -> None:
    """The automatic warnings of docs/stage9_plan.md §3.5 for one final fit."""
    if fit.best_iteration == 0:
        ctx.log.warn(f"{name}: best iteration at the first round")
    if fit.cap_hit:
        ctx.log.warn(f"{name}: best iteration {fit.best_iteration} at the cap of "
                     f"{fit.n_estimators} rounds: it may not have converged (M12d)")
    if not np.isfinite(fit.curve[["train_rmse", "val_rmse"]].to_numpy()).all():
        ctx.log.warn(f"{name}: NaN or infinite loss in the learning curve")
    ratio = fit.best_val_rmse / fit.best_train_rmse if fit.best_train_rmse > 0 else np.inf
    if ratio > X.GAP_RATIO_WARN:
        ctx.log.warn(f"{name}: validation RMSE {fit.best_val_rmse:.4g} is {ratio:.1f} x the "
                     f"train RMSE {fit.best_train_rmse:.4g} at the best iteration")
    edges = X.edge_params(params)
    if edges:
        ctx.log.warn(f"{name}: best trial on the edge of the search space for "
                     + ", ".join(f"{p} = {params[p]:.4g}" for p in edges))


def _seed_spread(ctx: Context, X, best: dict) -> dict:
    """Std over seeds of the validation nRMSE in kW of the written forecast per target x horizon
    (stage-10 rule)."""
    by_key: dict[str, list[float]] = {}
    for b in best.values():
        by_key.setdefault(f"{b['target']}_D{b['horizon']}", []).append(b["value_kw_forecast"])
    spread = {}
    for key, values in by_key.items():
        if len(values) < 2:
            continue
        spread[key] = float(np.std(values, ddof=1))
        if spread[key] > X.SEED_SPREAD_WARN_PP:
            ctx.log.warn(f"{key}: seed spread {spread[key]:.3f} pp above "
                         f"{X.SEED_SPREAD_WARN_PP} pp")
    if not spread:
        ctx.log.note("seed spread: one seed per target x horizon, not computed")
    return spread


# ── evaluate ────────────────────────────────────────────────────────────────


def prediction_files(ctx: Context) -> tuple[list[Path], list[str]]:
    """The prediction files of the completed model steps of this run, checked against the hashes
    in their records; and the model steps of the run that are not complete."""
    files = []
    for step in MODEL_STEPS:
        record = ctx.completed.get(step)
        if record is None:
            continue
        folder = ctx.run_dir / record["folder"]
        for name, digest in sorted(record["outputs"].items()):
            if not name.startswith("predictions_"):
                continue
            path = folder / name
            if not path.exists() or sha256(path) != digest:
                raise RunnerError(f"{record['folder']}/{name} is missing or was modified after "
                                  "its step completed; rerun the step with --force")
            files.append(path)
    missing = [s for s in MODEL_STEPS if s in ctx.failed and s not in ctx.completed]
    return files, missing


def evaluate_inputs(ctx: Context) -> list[Path]:
    return [FEATURES_FILE, DERIVED_CONSTANTS_FILE, *prediction_files(ctx)[0]]


def run_evaluate(ctx: Context) -> Result:
    if "baselines" not in ctx.completed:
        raise RunnerError("evaluate needs the references of this run: run the baselines step "
                          "first (they are the skill reference and the DM reference)")
    files, missing = prediction_files(ctx)
    if ctx.mode != "final":
        assert_no_test_rows(files)
    mean_k = mean_daytime_k()
    site = load_site()
    outputs, summary, notes = [], [], {"models_missing": missing}
    if missing:
        ctx.log.warn(f"model steps of this run that did not complete: {missing}")
    for window in ctx.windows:
        out = ctx.out / window.split
        with ctx.log.step(f"evaluate {window.split}, {window.label}"):
            truth = truth_for(window)
            tables = E.run(files, out, mean_k, site=site, truth=truth, split=window.split,
                           allow_test=ctx.config["allow_test"] and window.split == "test",
                           strict=True, with_references=False, ci=True, log=ctx.log,
                           command=ctx.command,
                           record_dir=ctx.run_dir / "evaluate" / window.split)
        outputs += sorted(out.glob("*.csv")) + [out / "manifest.json"]
        summary += benchmark_lines(out / "benchmark_summary.csv", window)
        ctx.log.value(f"rows_benchmark_{window.split}", len(tables["benchmark_summary"]))
    return Result(outputs, values={"files_scored": [f.name for f in files]}, summary=summary,
                  notes=notes)


def benchmark_lines(path: Path, window: Window) -> list[str]:
    """The run summary, read from the written table (M2: no result number in the code)."""
    bench = pd.read_csv(path)
    lines = [f"{window.split}, {window.label}: nRMSE in kW (% of capacity), n daytime hours"]
    for row in bench.sort_values(["target", "horizon", "model", "seed"]).itertuples():
        lines.append(f"  {row.target:<8} D+{row.horizon} {row.model:<12} seed {row.seed:>3}: "
                     f"{row.nRMSE_kW_pct:6.2f} %  (n = {row.n})")
    return lines


# ── registry ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Implementation:
    inputs: Callable[[Context], list[Path]]
    run: Callable[[Context], Result]


IMPLEMENTED = {
    "baselines": Implementation(baselines_inputs, run_baselines),
    "xgb": Implementation(xgb_inputs, run_xgb),
    "evaluate": Implementation(evaluate_inputs, run_evaluate),
}
