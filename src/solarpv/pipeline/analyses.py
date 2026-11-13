"""The two analyses of v6 that train models, as runner steps (docs/stage9_plan.md §21, D-36):

  `cv`        S-38, v6 §12.2: five expanding date folds of the train rows, per target x horizon x
              seed, with the model of the `xgb` step (M43, M44);
  `ablation`  S-41, v6 §12.5: the model refitted without one feature group at a time (M37-M42).

Both read the `xgb` step of the same run and refuse without it (M38): its searched parameters,
seed, rounds and feature order are the model they repeat. Neither decides anything: their values
are reported, and the comparison with validation and test is a description (D-34, D-36).

Rules (the self-review of §21):
- `cv` reads train rows only (`model_table(h, ("train",))`), scores the daytime hours with AC
  truth, op-limited hours included and warm-up excluded (X69), with the one kW metric of the
  objective and the evaluation (M42); the folds and the fold loop are those of stage 7 and of
  L-09 (X68). The L-09 correction is not refitted per fold: the CV scores the raw forecast.
- `ablation` repeats the final fit of the `xgb` step on the features that a mode keeps, with the
  best trial's searched parameters, the seed, early stopping on the validation rows (D-10) and
  constraints rebuilt by exact name for the features left (X70). The mode `full` must reproduce
  the `xgb` step's validation value to REFIT_TOL_PP, or the step fails. Scores are the raw
  forecast in kW on the scoring hours of the validation window. In a final run each fit also
  forecasts the test window (logged, X71); nothing scores it in phase 1 (M37, D-32). The step
  `test` scores those forecasts in phase 2 (`phase2_tables`).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.constants import TEST_START
from solarpv.config.provenance import sha256
from solarpv.features.decide_k import MIN_GAIN_PP
from solarpv.pipeline import data, steps
from solarpv.pipeline.config import Window
from solarpv.pipeline.steps import (
    Context,
    Result,
    RunnerError,
    check_selected_inputs,
    log_test_access,
    record_file,
)

MARGIN_FLOOR_PP = MIN_GAIN_PP  # the D-12 rule: max(0.05 pp, 2 x the pooled seed std)
CV_MIN_HELD = 500  # a fold with fewer held hours is a warning
GROUP_ORDER = ("nwp_gfs", "clearsky", "other", "time_encodings", "lag_power", "rolling_stats",
               "history_other", "k_history")
HISTORY = ("lag_power", "rolling_stats", "history_other", "k_history")  # measured past days
MINIMAL_GROUP = "time_encodings"
BEST_FILE = "best_params.json"
FORECAST_COLUMNS = ["timestamp", "target", "horizon", "mode", "seed", "split", "y_pred"]


def _x():
    from solarpv.models import xgb_sota as X

    return X


# ── what both steps read ────────────────────────────────────────────────────


def _need_xgb(ctx: Context) -> None:
    if "xgb" not in ctx.completed:
        raise RunnerError(f"{ctx.step} repeats the model of the xgb step: run xgb first in this "
                          "run (M38)")


def analysis_inputs(ctx: Context) -> list[Path]:
    X = _x()
    check_selected_inputs(ctx)
    _need_xgb(ctx)
    files = [record_file(ctx, "xgb", BEST_FILE), record_file(ctx, "xgb", "xgb_config.json")]
    for h in ctx.config["horizons"]:
        files += [data.feature_file(h), data.gfs_file(h)]
    return files + [X.MODEL_INPUTS_FILE, steps.DERIVED_CONSTANTS_FILE]


def xgb_best(ctx: Context) -> dict:
    """The `xgb` step's records of its final fits, one per target x horizon x seed."""
    return json.loads(record_file(ctx, "xgb", BEST_FILE).read_text(encoding="utf-8"))


def seeds_of(ctx: Context) -> list[int]:
    budget = ctx.config["budgets"]["xgb"]
    return [int(s) for s in (ctx.config["seeds"] or budget["seeds"])]


def record_of(best: dict, name: str) -> dict:
    """M38: the model of a fit is its record; a missing one is an error, never a default."""
    if name not in best:
        raise RunnerError(f"best_params.json of the xgb step has no record {name}")
    rec = best[name]
    for key in ("searched", "n_rounds", "feature_order", "monotone_setting", "value_kw"):
        if key not in rec:
            raise RunnerError(f"the xgb record {name} has no {key}")
    return rec


def features_of(table: pd.DataFrame, target: str, h: int, inputs: dict, rec: dict) -> list[str]:
    """The record's feature order, which must still be the allow-list of the input file."""
    X = _x()
    feats = X.input_features(table, target, h, inputs)
    if list(rec["feature_order"]) != feats:
        raise RunnerError(f"{target}_D{h}: the feature order of the xgb record differs from "
                          "model_inputs.json")
    return feats


def smoke_rows(ctx: Context) -> bool:
    """The training rows are those of the xgb step, whose recorded mode says whether it used the
    smoke subset of train (the last 56 local days). In a real run this equals the step's own
    mode; reading it from the record keeps a repeated fit on the rows of the fit it repeats."""
    cfg = json.loads(record_file(ctx, "xgb", "xgb_config.json").read_text(encoding="utf-8"))
    return cfg["mode"] == "smoke"


def cv_rows(table: pd.DataFrame) -> pd.Series:
    """X69: the scoring rule without a window: daytime hours with AC truth, op-limited hours
    included, warm-up excluded (its features are not defined)."""
    return (table["daytime"].astype(bool) & table["site_total_AC_kW"].notna()
            & ~table["warmup"].astype(bool))


# ── cv ──────────────────────────────────────────────────────────────────────

CV_FOLD_COLUMNS = ["target", "horizon", "seed", "fold", "fit_rows", "held_rows",
                   "held_rows_l09_rule", "first_day", "last_day", "months", "mean_truth_kw",
                   "nrmse_kw_pct"]
CV_SUMMARY_COLUMNS = ["target", "horizon", "seed", "n_folds", "cv_mean", "cv_std", "cv_min",
                      "cv_max", "validation_uncorrected", "validation_forecast",
                      "validation_minus_cv_pp"]


def run_cv(ctx: Context) -> Result:
    """S-38: the CV of the xgb models on the train rows (docs/stage9_plan.md §21.4)."""
    X = _x()
    best = xgb_best(ctx)
    seeds = seeds_of(ctx)
    smoke = smoke_rows(ctx)
    inputs = X.load_model_inputs(X.MODEL_INPUTS_FILE)
    folds, summary, short = [], [], []
    ctx.log.note(f"CV of the xgb models on the train rows, seeds {seeds}, five expanding date "
                 "folds with a 3-day gap, the raw forecast, nRMSE in kW (% of 3,449 kW)")
    for h in ctx.config["horizons"]:
        table = data.model_table(h, ("train",))
        if (table["split"] != "train").any():
            raise RunnerError("rows outside train reached the cv step")
        usable, scope = cv_rows(table), X.train_scope(table, smoke)
        legacy = X.mos_rows(table)
        for target in ctx.config["targets"]:
            for seed in seeds:
                name = f"{target}_D{h}_seed{seed}"
                rec = record_of(best, name)
                feats = features_of(table, target, h, inputs, rec)
                constraints = X.constraint_map(feats, rec["monotone_setting"])
                values = []
                with ctx.log.step(f"{name}: five folds"):
                    for i, rows, held, pred in X.oof_folds(
                            table, target, feats, rec["searched"], seed, constraints,
                            int(rec["n_rounds"]), scope, usable, what="the CV"):
                        theo = table.loc[held, "theo_kw"].to_numpy(float)
                        truth = table.loc[held, X.TRUTH_KW].to_numpy(float)
                        value = X.nrmse_kw(target, pred, theo, truth)
                        days = X.local_day(table.index[held])
                        folds.append({
                            "target": target, "horizon": h, "seed": seed, "fold": i,
                            "fit_rows": int(rows.sum()), "held_rows": int(held.sum()),
                            "held_rows_l09_rule": int((held & legacy).sum()),
                            "first_day": str(days.min().date()), "last_day": str(days.max().date()),
                            "months": ",".join(sorted({str(d)[:7] for d in days})),
                            "mean_truth_kw": float(truth.mean()), "nrmse_kw_pct": value})
                        values.append(value)
                        if held.sum() < CV_MIN_HELD:
                            short.append(f"{name} fold {i}: {int(held.sum())} held hours")
                mean = float(np.mean(values))
                summary.append({
                    "target": target, "horizon": h, "seed": seed, "n_folds": len(values),
                    "cv_mean": mean, "cv_std": float(np.std(values, ddof=1)),
                    "cv_min": float(np.min(values)), "cv_max": float(np.max(values)),
                    "validation_uncorrected": float(rec["value_kw"]),
                    "validation_forecast": float(rec.get("value_kw_forecast", rec["value_kw"])),
                    "validation_minus_cv_pp": float(rec["value_kw"]) - mean})
                ctx.log.note(f"{name}: CV {mean:.3f} +- {np.std(values, ddof=1):.3f} % kW over "
                             f"{len(values)} folds; validation (raw) {rec['value_kw']:.3f} %")
    for text in short:
        ctx.log.warn(f"cv: a fold with fewer than {CV_MIN_HELD} held hours: {text}")
    out = ctx.out
    folds_f = pd.DataFrame(folds, columns=CV_FOLD_COLUMNS)
    summary_f = pd.DataFrame(summary, columns=CV_SUMMARY_COLUMNS)
    outputs = [steps._write_csv(folds_f, out / "cv_folds.csv"),
               steps._write_csv(summary_f, out / "cv_summary.csv"),
               steps._write_json({
                   "step": "cv", "mode": ctx.mode, "seeds": seeds,
                   "folds": "the five expanding date folds of stage 7 (selection.common."
                            "date_folds, 3-day gap) over the local days of the train rows "
                            "(smoke: the last 56 days), the loop of the L-09 correction (X68)",
                   "model": "the xgb record of each fit: searched parameters, seed, constraints "
                            "by exact name, rounds n_rounds, fixed; no early stopping",
                   "held_rows": "daytime hours with AC truth, op-limited included, warm-up "
                                "excluded (X69)",
                   "metric": "nRMSE in kW, % of 3,449 kW (units.nrmse_kw); the k forecast "
                             "through to_kw",
                   "forecast": "raw, not corrected by L-09",
                   "comparison": "validation_uncorrected is the xgb step's value of the same "
                                 "fit on the validation window, which was its early-stopping "
                                 "set (D-10); the CV folds were not",
                   "xgb_best_params_sha256": sha256(record_file(ctx, "xgb", BEST_FILE))},
                   out / "cv_config.json")]
    lines = ["cv: nRMSE in kW (% of capacity), mean +- std over the five folds, against the "
             "validation value of the same model"]
    for r in summary_f.sort_values(["target", "horizon", "seed"]).itertuples():
        lines.append(f"  {r.target:<8} D+{r.horizon} seed {r.seed:>3}: CV {r.cv_mean:6.2f} +- "
                     f"{r.cv_std:5.2f}   validation {r.validation_uncorrected:6.2f}")
    return Result(outputs, values={"cv_mean": {f"{r.target}_D{r.horizon}_seed{r.seed}": r.cv_mean
                                               for r in summary_f.itertuples()},
                                   "test_access": [], "short_folds": short},
                  summary=lines)


# ── ablation ────────────────────────────────────────────────────────────────


def ablation_modes(features: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """The modes of one input (M39, M40): `full`; `no_<group>` for every group of the feature
    contract that the input has; `no_history` (the measured past-day groups together, when two or
    more are present); `minimal` (the time encodings only). Groups are the one function of the
    feature contract (`selection.common.group`, the SHAP groups of `explain`). A mode that leaves
    no feature, or repeats another, is skipped with its reason."""
    from solarpv.selection.common import group

    of = {f: group(f) for f in features}
    present = [g for g in GROUP_ORDER if g in set(of.values())]
    unknown = sorted(set(of.values()) - set(GROUP_ORDER))
    if unknown:
        raise RunnerError(f"feature groups without a place in the ablation: {unknown}")
    modes: dict[str, list[str]] = {"full": list(features)}
    skipped: dict[str, str] = {}
    candidates = {f"no_{g}": [f for f in features if of[f] != g] for g in present}
    past = [g for g in present if g in HISTORY]
    if len(past) >= 2:
        candidates["no_history"] = [f for f in features if of[f] not in HISTORY]
    else:
        skipped["no_history"] = "fewer than two past-day groups: it would repeat a no_<group> mode"
    candidates["minimal"] = [f for f in features if of[f] == MINIMAL_GROUP]
    for mode, feats in candidates.items():
        if not feats:
            skipped[mode] = "no feature left"
        elif any(feats == kept for kept in modes.values()):
            skipped[mode] = "the same features as another mode"
        else:
            modes[mode] = feats
    return modes, skipped


def ablation_summary(results: pd.DataFrame) -> pd.DataFrame:
    """M41: per target x horizon x mode, the mean and the std over seeds of the delta against the
    `full` fit of the same seed, the margin max(0.05 pp, 2 x the pooled seed std) and the
    verdict. The pooled std is the root of the mean variance over the modes of the target and
    horizon; with one seed the margin is the floor and the row says so."""
    rows = []
    for (target, h), g in results.groupby(["target", "horizon"]):
        deltas = g[g["mode"] != "full"].pivot(index="seed", columns="mode", values="delta_pp")
        n_seeds = deltas.shape[0]
        if n_seeds > 1:
            std = deltas.std(ddof=1)
            pooled = float(np.sqrt(np.nanmean(std.to_numpy() ** 2)))
            margin = max(MARGIN_FLOOR_PP, 2.0 * pooled)
        else:
            std, pooled, margin = pd.Series(np.nan, index=deltas.columns), np.nan, MARGIN_FLOOR_PP
        for mode in deltas.columns:
            mean = float(deltas[mode].mean())
            verdict = "helpful" if mean > margin else "harmful" if mean < -margin else "neutral"
            rows.append({"target": target, "horizon": int(h), "mode": mode, "n_seeds": n_seeds,
                         "mean_delta_pp": mean, "std_delta_pp": float(std[mode]),
                         "pooled_seed_std_pp": pooled, "margin_pp": margin,
                         "single_seed": n_seeds == 1, "verdict": verdict})
    return pd.DataFrame(rows, columns=["target", "horizon", "mode", "n_seeds", "mean_delta_pp",
                                       "std_delta_pp", "pooled_seed_std_pp", "margin_pp",
                                       "single_seed", "verdict"])


def add_deltas(results: pd.DataFrame, value: str = "value_kw") -> pd.DataFrame:
    """The delta of every mode against the `full` fit of the same target, horizon and seed."""
    full = (results[results["mode"] == "full"].set_index(["target", "horizon", "seed"])[value]
            .rename("full_value"))
    out = results.join(full, on=["target", "horizon", "seed"])
    out["delta_pp"] = out[value] - out["full_value"]
    return out.drop(columns="full_value")


RESULT_COLUMNS = ["target", "horizon", "seed", "mode", "n_features", "n_removed", "value_kw",
                  "value_k", "best_iteration", "n_rounds", "delta_pp"]


def run_ablation(ctx: Context) -> Result:
    """S-41: the feature-group ablation of the xgb models (docs/stage9_plan.md §21.4)."""
    from solarpv.selection.common import group

    X = _x()
    best = xgb_best(ctx)
    seeds = seeds_of(ctx)
    tuning = steps._tuning_window(ctx)
    smoke = smoke_rows(ctx)
    with_test = any(w.split == "test" for w in ctx.windows)
    splits = ("train", "val", "test") if with_test else ("train", "val")
    mean_k = steps.mean_daytime_k()
    inputs = X.load_model_inputs(X.MODEL_INPUTS_FILE)
    results, mode_rows, access = [], [], []
    forecasts = {s: [] for s in seeds}
    ctx.log.note(f"ablation of the xgb models, seeds {seeds}, tuning window {tuning.label}; "
                 f"modes by feature group; raw forecast in kW; test window forecast: {with_test}")
    for h in ctx.config["horizons"]:
        table = data.model_table(h, splits, allow_test=with_test)
        if not with_test and (table.index >= pd.Timestamp(TEST_START, tz="UTC")).any():
            raise RunnerError("test rows reached the ablation step in a development run")
        if with_test:  # final: the test window is forecast, never scored here (D-32, X71)
            access.append(log_test_access(
                ctx, f"forecast the test window of D+{h} for the ablation (features)",
                [data.feature_file(h), data.gfs_file(h)], int((table["split"] == "test").sum())))
        for target in ctx.config["targets"]:
            for seed in seeds:
                name = f"{target}_D{h}_seed{seed}"
                rec = record_of(best, name)
                feats_full = features_of(table, target, h, inputs, rec)
                modes, skipped = ablation_modes(feats_full)
                if seed == seeds[0]:
                    for mode, feats in modes.items():
                        mode_rows.append({"target": target, "horizon": h, "mode": mode,
                                          "skipped": False, "reason": "",
                                          "n_features": len(feats),
                                          "groups": ",".join(sorted({group(f) for f in feats})),
                                          "features": ";".join(feats)})
                    for mode, reason in skipped.items():
                        mode_rows.append({"target": target, "horizon": h, "mode": mode,
                                          "skipped": True, "reason": reason, "n_features": 0,
                                          "groups": "", "features": ""})
                with ctx.log.step(f"{name}: {len(modes)} mode(s)"):
                    for mode, feats in modes.items():
                        constraints = X.constraint_map(feats, rec["monotone_setting"])
                        problem = X.build_problem(table, target, h, feats, constraints, tuning,
                                                  mean_k, smoke=smoke)
                        fit = X.fit_early_stopping(problem, rec["searched"], seed)
                        values = X.score(problem, fit)
                        if mode == "full":  # M38: the baseline is the xgb step's own fit
                            diff = abs(values["value_kw"] - float(rec["value_kw"]))
                            if diff > X.REFIT_TOL_PP or fit.n_rounds != int(rec["n_rounds"]):
                                raise RunnerError(
                                    f"{name}: the mode full does not reproduce the xgb step "
                                    f"({values['value_kw']:.12f} against {rec['value_kw']:.12f}, "
                                    f"{fit.n_rounds} against {rec['n_rounds']} rounds)")
                        results.append({
                            "target": target, "horizon": h, "seed": seed, "mode": mode,
                            "n_features": len(feats), "n_removed": len(feats_full) - len(feats),
                            "value_kw": values["value_kw"],
                            "value_k": values.get("value_k", np.nan),
                            "best_iteration": fit.best_iteration, "n_rounds": fit.n_rounds})
                        for w in ctx.windows:
                            rows = X.window_rows(table, w) & table["daytime"].astype(bool)
                            raw = X.predict(fit.booster, table.loc[rows, feats], fit.n_rounds)
                            forecasts[seed].append(pd.DataFrame({
                                "timestamp": table.index[rows], "target": target, "horizon": h,
                                "mode": mode, "seed": seed, "split": w.split, "y_pred": raw}))
                    ctx.log.note(f"{name}: modes {list(modes)}; skipped {skipped or 'none'}")
    res = add_deltas(pd.DataFrame(results))
    res = res[RESULT_COLUMNS]
    summary = ablation_summary(res)
    outputs = [steps._write_csv(res, ctx.out / "ablation_results.csv"),
               steps._write_csv(pd.DataFrame(mode_rows), ctx.out / "ablation_modes.csv"),
               steps._write_csv(summary, ctx.out / "ablation_summary.csv")]
    for seed, frames in forecasts.items():
        frame = pd.concat(frames, ignore_index=True).sort_values(
            ["target", "horizon", "mode", "timestamp"])
        outputs.append(steps._write_csv(frame[FORECAST_COLUMNS],
                                        ctx.out / f"ablation_forecasts_seed{seed}.csv"))
    outputs.append(steps._write_json({
        "step": "ablation", "mode": ctx.mode, "seeds": seeds, "tuning_window": tuning.as_dict(),
        "windows": [w.as_dict() for w in ctx.windows],
        "modes": "full, no_<group> for each feature group of the input, no_history (the measured "
                 "past-day groups) and minimal (the time encodings), by selection.common.group",
        "fit": "the xgb step's final fit on the features a mode keeps: the searched "
               "parameters of the record, the seed, early stopping on the validation rows (D-10), "
               "constraints by exact name for the features left",
        "score": "nRMSE in kW (% of 3,449 kW) of the raw forecast on the scoring hours of the "
                 "validation window; L-09 is not applied",
        "delta": "mode minus full of the same target, horizon and seed (pp)",
        "verdict": f"helpful above, harmful below minus max({MARGIN_FLOOR_PP} pp, 2 x the pooled "
                   "seed std of the deltas); neutral otherwise",
        "forecast_rows": "daytime hours of each window; the test window only in a final run "
                         "(forecast, logged, not scored in phase 1)",
        "xgb_best_params_sha256": sha256(record_file(ctx, "xgb", BEST_FILE))},
        ctx.out / "ablation_config.json"))
    lines = [f"ablation ({len(res)} fit(s)): delta of nRMSE in kW against the full fit (pp), mean "
             "over seeds, with the verdict by the registered margin"]
    for r in summary.sort_values(["target", "horizon", "mode"]).itertuples():
        lines.append(f"  {r.target:<8} D+{r.horizon} {r.mode:<18} {r.mean_delta_pp:+7.3f} pp "
                     f"(margin {r.margin_pp:.3f}) {r.verdict}")
    return Result(outputs, values={"fits": int(len(res)), "test_access": access,
                                   "verdicts": summary["verdict"].value_counts().to_dict()},
                  summary=lines)


# ── the second phase of a final run (the step `test`, D-32) ─────────────────


def _raw_forecasts(ctx: Context, seed: int) -> pd.DataFrame:
    """The raw xgb forecast of a seed: the uncorrected file where L-09 corrected a target,
    the predictions file for the rest."""
    rec = ctx.completed["xgb"]
    X = _x()
    pred = pd.read_csv(record_file(ctx, "xgb", f"predictions_{X.MODEL}_seed{seed}.csv"))
    name = X.uncorrected_name(seed)
    if name in rec["outputs"]:
        unc = pd.read_csv(record_file(ctx, "xgb", name))
        pred = pd.concat([pred[~pred["target"].isin(set(unc["target"]))], unc], ignore_index=True)
    return pred


def _score_hours(frame: pd.DataFrame, truth: pd.DataFrame, target: str) -> tuple[float, int]:
    """nRMSE in kW of a forecast on the test hours of `truth` that are daytime with AC truth
    (the scoring rule); `frame` has `timestamp` and `y_pred` of one target and horizon."""
    X = _x()
    ts = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    t = truth.reindex(ts)
    ok = ((t["split"].to_numpy() == "test") & t["daytime"].fillna(False).to_numpy(bool)
          & t["direct"].notna().to_numpy())
    if not ok.any():
        raise RunnerError("no scoring hour on the test window")
    pred = frame["y_pred"].to_numpy(float)[ok]
    return (X.nrmse_kw(target, pred, t["theo_kw"].to_numpy(float)[ok],
                       t["direct"].to_numpy(float)[ok]), int(ok.sum()))


def phase2_inputs(ctx: Context) -> list[Path]:
    """The records the analyses of phase 2 read, for the step's fingerprint."""
    files = []
    if "cv" in ctx.completed:
        files.append(record_file(ctx, "cv", "cv_summary.csv"))
        files += [record_file(ctx, "xgb", n) for n in sorted(ctx.completed["xgb"]["outputs"])
                  if n.startswith(("predictions_", "uncorrected_"))]
    if "ablation" in ctx.completed:
        files += [record_file(ctx, "ablation", n)
                  for n in sorted(ctx.completed["ablation"]["outputs"])
                  if n.startswith("ablation_forecasts_")]
    return list(dict.fromkeys(files))


def phase2_tables(ctx: Context, window: Window) -> tuple[list[Path], list[dict]]:
    """Phase 2 of a final run: the test value of the CV comparison and the ablation on the test
    window, from the forecasts of phase 1 (the test rows are read here, logged; X71)."""
    if ctx.mode != "final" or not ctx.allow_test:
        raise RunnerError("the analyses are scored on the test window only in the step test of "
                          "a final run (D-32)")
    files = phase2_inputs(ctx)
    truth = steps.truth_for(window)
    out = ctx.out / "analyses"
    out.mkdir()
    outputs: list[Path] = []
    n_test = int(((truth["split"] == "test") & truth["daytime"]
                  & truth["direct"].notna()).sum())
    access = [log_test_access(ctx, "score the CV comparison and the ablation forecasts on the "
                                   "test window", files, n_test)]
    if "cv" in ctx.completed:
        summary = pd.read_csv(record_file(ctx, "cv", "cv_summary.csv"))
        rows = []
        for seed in sorted(summary["seed"].unique()):
            raw = _raw_forecasts(ctx, int(seed))
            for (target, h), g in raw.groupby(["target", "horizon"]):
                value, n = _score_hours(g, truth, target)
                rows.append({"target": target, "horizon": int(h), "seed": int(seed),
                             "test_uncorrected": value, "n_test_hours": n})
        merged = summary.merge(pd.DataFrame(rows), on=["target", "horizon", "seed"], how="left")
        merged["test_minus_cv_pp"] = merged["test_uncorrected"] - merged["cv_mean"]
        outputs.append(steps._write_csv(merged, out / "cv_vs_test.csv"))
    if "ablation" in ctx.completed:
        frames = []
        for name in sorted(ctx.completed["ablation"]["outputs"]):
            if name.startswith("ablation_forecasts_"):
                f = pd.read_csv(record_file(ctx, "ablation", name))
                frames.append(f[f["split"] == "test"])
        all_test = pd.concat(frames, ignore_index=True)
        rows = []
        for (target, h, seed, mode), g in all_test.groupby(["target", "horizon", "seed", "mode"]):
            value, n = _score_hours(g, truth, target)
            rows.append({"target": target, "horizon": int(h), "seed": int(seed), "mode": mode,
                         "value_kw": value, "n_test_hours": n})
        res = add_deltas(pd.DataFrame(rows))
        outputs.append(steps._write_csv(res, out / "ablation_test.csv"))
        outputs.append(steps._write_csv(ablation_summary(res), out / "ablation_test_summary.csv"))
    return outputs, access
