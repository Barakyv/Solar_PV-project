"""The `explain` step (docs/stage9_plan.md §19.5, D-31). CPU only, in `solar`; it trains
nothing and reloads no neural network.

XGB (M26-M28, X46, X47)
  Exact TreeSHAP (`shap.TreeExplainer`, tree_path_dependent: no background; the base value is
  the booster's mean output over its training cover) of every booster of the run, sliced to the
  rounds the forecast uses (best + 1; the saved booster holds 50 more, X46). Rows: the daytime
  scoring hours of the validation window with AC truth (§14.1), never fit rows. The output is the
  booster's raw output in the training unit (k for the indirect route, kW for the direct route).
  Checks: additivity against the booster forecast (the uncorrected file where L-09 is applied)
  within SHAP_TOL, and agreement with XGBoost's own `pred_contribs` within CONTRIBS_TOL; the
  unsliced booster's additivity error is recorded to show why the slice is needed. A kW view for
  the indirect route (X47): phi_kW = b * theo * phi_k with the base a + b * theo * base (b = 1,
  a = 0 without L-09), exact on rows the clip at 0 and 3,449 kW does not touch, and checked
  against the written forecast through `to_kw`. Groups: the mean over rows of |sum of phi over
  the group| (exact by additivity), groups from `selection.common.group`, with their member
  counts (M27). Rank stability over the features common to both rankings (M28).
TFT (M29, X48)
  From the stored W4 outputs: the validation attention [origin, step, encoder position] and the
  variable-selection weights. Attention by lag (overall and per horizon block), the share of the
  encoder attention on positions a whole number of days before each step's valid hour against
  the share of such positions (about 1/24), the edge of the window; weights per channel and per
  variable (a series and its fill flag added).
Optuna (M30, X45)
  fANOVA importance (Optuna's `FanovaImportanceEvaluator`) with seeds 0-9, mean and std, for
  every study of the run, on a copy of the study database (opening the original could write to
  it and change its recorded hash); the number of complete trials and a flag when every trial
  is inside TPE's random start-up (10 trials).

SHAP describes how the fitted booster maps its inputs to its output on these rows under the
tree-path-dependent masking; it is not a causal effect of a feature on the plant.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from solarpv.config.constants import AC_CAPACITY_KW
from solarpv.pipeline import data
from solarpv.pipeline.config import Window
from solarpv.pipeline.steps import Context, Result, RunnerError, log_test_access, record_file

# Tolerances relative to the scale of the output: TreeSHAP sums in float32, about 1e-6 of the
# output's magnitude (k is about 1, kW up to 3,449), so a fixed absolute tolerance would fail
# the kW boosters on precision alone. SCALE is the unit's scale.
SHAP_TOL = 1e-4  # additivity, relative to SCALE
CONTRIBS_TOL = 1e-5  # SHAP against XGBoost's pred_contribs, relative to SCALE
SCALE = {"k": 1.0, "kW": AC_CAPACITY_KW}
KW_TOL = SHAP_TOL * AC_CAPACITY_KW  # the kW view, kW
FANOVA_SEEDS = tuple(range(10))
TPE_STARTUP = 10  # optuna.samplers.TPESampler's default n_startup_trials
DOMINANCE_FACTOR = 10.0  # the stage-7 red flag: first feature above ten times the third
TFT_STEPS = ("tft96", "tft168", "tftbase")
STUDY_STEPS = ("bilstm", "tft96", "tft168")
FILLED = "_filled"
NON_CAUSAL = ("SHAP describes how the fitted booster maps its inputs to its output on these rows "
              "(tree-path-dependent masking); it is not a causal effect of a feature on the "
              "plant.")


def _x():
    from solarpv.models import xgb_sota as X

    return X


def _j():
    from solarpv.models import neural_job as J

    return J


# ── inputs ──────────────────────────────────────────────────────────────────


def _xgb_files(ctx: Context) -> list[Path]:
    rec = ctx.completed["xgb"]
    keep = [n for n in rec["outputs"] if n.startswith(("models/", "predictions_", "uncorrected_"))
            or n in ("best_params.json", "optuna.db", "xgb_config.json")]
    return [record_file(ctx, "xgb", n) for n in sorted(keep)]


def _tft_files(ctx: Context, step: str) -> list[Path]:
    rec = ctx.completed[step]
    keep = [n for n in rec["outputs"] if n.startswith(("worker/forecast_", "job/data_"))
            or n in ("best_params.json", "job/job.json")]
    return [record_file(ctx, step, n) for n in sorted(keep)]


def explain_inputs(ctx: Context) -> list[Path]:
    X = _x()
    files: list[Path] = []
    if "xgb" in ctx.completed:
        files += _xgb_files(ctx)
        for h in ctx.config["horizons"]:
            files += [data.feature_file(h), data.gfs_file(h)]
        files.append(X.MODEL_INPUTS_FILE)
    for step in TFT_STEPS:
        if step in ctx.completed:
            files += _tft_files(ctx, step)
    for step in STUDY_STEPS:
        if step in ctx.completed:
            files.append(record_file(ctx, step, "worker/optuna.db"))
    if not files:
        raise RunnerError("explain needs a completed xgb, BiLSTM or TFT step in this run")
    return list(dict.fromkeys(files))


# ── XGB: SHAP ───────────────────────────────────────────────────────────────


def feature_group(feature: str) -> str:
    from solarpv.selection.common import group

    return group(feature)  # one group per feature, by exact name; an unknown name raises


def shap_values(booster, X_rows: pd.DataFrame, n_rounds: int) -> dict:
    """TreeSHAP of the booster's first `n_rounds` rounds on `X_rows`, with its checks."""
    import shap
    import xgboost as xgb

    if list(booster.feature_names or []) != list(X_rows.columns):
        raise RunnerError("the booster's feature names differ from the explained columns (M26)")
    total = int(booster.num_boosted_rounds())
    if not 0 < n_rounds <= total:
        raise RunnerError(f"n_rounds {n_rounds} outside 1..{total}")
    sliced = booster[:n_rounds]  # X46: the model of the forecast, not the saved booster
    dm = xgb.DMatrix(X_rows, missing=np.nan)
    forecast = booster.predict(dm, iteration_range=(0, n_rounds)).astype(float)
    explainer = shap.TreeExplainer(sliced, feature_perturbation="tree_path_dependent")
    exp = explainer(X_rows)
    values = np.asarray(exp.values, dtype=float)
    base = np.broadcast_to(np.asarray(exp.base_values, dtype=float), (len(X_rows),)).copy()
    contribs = sliced.predict(dm, pred_contribs=True).astype(float)
    full = booster.predict(dm, pred_contribs=True).astype(float)  # every saved round
    return {
        "values": values, "base": base, "forecast": forecast,
        "additivity": float(np.max(np.abs(base + values.sum(axis=1) - forecast))),
        "contribs_diff": float(max(np.max(np.abs(contribs[:, :-1] - values)),
                                   np.max(np.abs(contribs[:, -1] - base)))),
        "unsliced_additivity": float(np.max(np.abs(full.sum(axis=1) - forecast))),
        "rounds_in_booster": total, "explainer": type(explainer).__name__,
    }


def kw_view(target: str, sv: dict, theo: np.ndarray, mos: dict | None) -> dict:
    """X47: the attribution of the written kW forecast. For k, phi_kW = b * theo * phi_k and the
    base a + b * theo * base; exact where the clip of L-09 does not bind. Direct: kW already."""
    if target == "direct":
        return {"values": sv["values"], "base": sv["base"], "unclipped": np.ones(
            len(theo), dtype=bool), "forecast_kw": sv["forecast"]}
    a = float(mos["a_kw"]) if mos else 0.0
    b = float(mos["b"]) if mos else 1.0
    scale = b * theo
    raw_kw = a + scale * sv["forecast"]
    unclipped = ((raw_kw > 0) & (raw_kw < AC_CAPACITY_KW)) if mos else np.ones(len(theo), bool)
    forecast_kw = np.clip(raw_kw, 0.0, AC_CAPACITY_KW) if mos else raw_kw
    return {"values": sv["values"] * scale[:, None], "base": a + scale * sv["base"],
            "unclipped": unclipped, "forecast_kw": forecast_kw}


def feature_table(features: list[str], phi_kw: np.ndarray, phi: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"feature": features, "group": [feature_group(f) for f in features],
                        "mean_abs_kw": np.abs(phi_kw).mean(axis=0),
                        "mean_kw": phi_kw.mean(axis=0),
                        "mean_abs_native": np.abs(phi).mean(axis=0)})
    out = out.sort_values("mean_abs_kw", ascending=False, kind="mergesort")
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def group_table(features: list[str], phi_kw: np.ndarray) -> pd.DataFrame:
    """M27: |sum over the group| per row, then the mean; and, for comparison only, the v6
    measure (the sum of the members' mean |phi|)."""
    groups = pd.Series([feature_group(f) for f in features])
    rows = []
    for g, idx in groups.groupby(groups).groups.items():
        cols = np.asarray(list(idx))
        rows.append({"group": g, "n_members": len(cols),
                     "mean_abs_group_kw": float(np.abs(phi_kw[:, cols].sum(axis=1)).mean()),
                     "sum_member_mean_abs_kw": float(np.abs(phi_kw[:, cols]).mean(axis=0)
                                                     .sum())})
    out = pd.DataFrame(rows)
    out["share_pct"] = out["mean_abs_group_kw"] / out["mean_abs_group_kw"].sum() * 100
    out["share_pct_v6_measure"] = (out["sum_member_mean_abs_kw"]
                                   / out["sum_member_mean_abs_kw"].sum() * 100)
    return out.sort_values("mean_abs_group_kw", ascending=False, kind="mergesort")


def rank_stability(features: pd.DataFrame) -> pd.DataFrame:
    """M28: Spearman of mean |phi_kW| over the features common to both rankings: across horizons
    (same target and seed) and across seeds (same target and horizon)."""
    from scipy.stats import spearmanr

    rows = []

    def compare(a: pd.DataFrame, b: pd.DataFrame, what: dict) -> None:
        common = sorted(set(a["feature"]) & set(b["feature"]))
        if len(common) < 3:
            return
        x = a.set_index("feature").loc[common, "mean_abs_kw"]
        y = b.set_index("feature").loc[common, "mean_abs_kw"]
        rho = spearmanr(x, y).statistic
        rows.append({**what, "n_common": len(common), "n_a": len(a), "n_b": len(b),
                     "spearman": float(rho)})

    for (t, s), g in features.groupby(["target", "seed"]):
        hs = sorted(g["horizon"].unique())
        for i, h1 in enumerate(hs):
            for h2 in hs[i + 1:]:
                compare(g[g["horizon"] == h1], g[g["horizon"] == h2],
                        {"target": t, "compare": f"D+{h1} vs D+{h2}", "seed": s, "horizon": ""})
    for (t, h), g in features.groupby(["target", "horizon"]):
        seeds = sorted(g["seed"].unique())
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1:]:
                compare(g[g["seed"] == s1], g[g["seed"] == s2],
                        {"target": t, "compare": f"seed {s1} vs seed {s2}", "seed": "",
                         "horizon": h})
    return pd.DataFrame(rows, columns=["target", "compare", "seed", "horizon", "n_common",
                                       "n_a", "n_b", "spearman"])


def _forecast_frame(path: Path, target: str, h: int) -> pd.Series:
    f = pd.read_csv(path, usecols=["timestamp", "y_pred", "target", "horizon"])
    f = f[(f["target"] == target) & (f["horizon"] == h)]
    return pd.Series(f["y_pred"].to_numpy(float), index=pd.to_datetime(f["timestamp"], utc=True))


def xgb_shap(ctx: Context, window: Window, out: Path, table_for, tag: str = ""
             ) -> tuple[list[Path], dict[str, pd.DataFrame], list[str]]:
    """SHAP of every XGB booster of the run on the scoring rows of `window`."""
    import xgboost as xgb

    from solarpv.evaluation.units import to_kw

    X = _x()
    folder = ctx.run_dir / ctx.completed["xgb"]["folder"]
    best = json.loads(record_file(ctx, "xgb", "best_params.json").read_text(encoding="utf-8"))
    inputs = X.load_model_inputs(X.MODEL_INPUTS_FILE)
    out.mkdir(parents=True, exist_ok=True)
    features, groups, checks, warnings, outputs = [], [], [], [], []
    by_h: dict[int, list[tuple[str, dict]]] = {}
    for name, b in best.items():
        by_h.setdefault(int(b["horizon"]), []).append((name, b))
    for h in sorted(by_h):
        table = table_for(h)
        rows = X.score_rows(table, window)
        for name, b in by_h[h]:
            target, seed = b["target"], int(b["seed"])
            feats = list(b["feature_order"])
            if feats != X.input_features(table, target, h, inputs):
                raise RunnerError(f"{name}: the booster's features are not the run's allow-list "
                                  "of model_inputs.json (M26)")
            booster = xgb.Booster()
            booster.load_model(record_file(ctx, "xgb", b["booster"]))
            Xr = table.loc[rows, feats]
            sv = shap_values(booster, Xr, int(b["n_rounds"]))
            mos = b["mos"] if b.get("mos", {}).get("applied") else None
            raw_file = (X.uncorrected_name(seed) if mos else f"predictions_{X.MODEL}_seed{seed}"
                        ".csv")
            written_raw = _forecast_frame(folder / raw_file, target, h).reindex(Xr.index)
            raw_diff = float(np.max(np.abs(written_raw.to_numpy() - sv["forecast"])))
            theo = table.loc[rows, "theo_kw"].to_numpy(float)
            kv = kw_view(target, sv, theo, mos)
            written = _forecast_frame(folder / f"predictions_{X.MODEL}_seed{seed}.csv",
                                      target, h).reindex(Xr.index).to_numpy()
            written_kw = to_kw(target, written, theo)
            kw_add = np.abs(kv["base"] + kv["values"].sum(axis=1) - written_kw)
            kw_add = float(kw_add[kv["unclipped"]].max()) if kv["unclipped"].any() else 0.0
            ft = feature_table(feats, kv["values"], sv["values"]).assign(
                target=target, horizon=h, seed=seed, split=window.split)
            gt = group_table(feats, kv["values"]).assign(target=target, horizon=h, seed=seed,
                                                         split=window.split)
            top = ft["mean_abs_kw"].to_numpy()
            dominance = float(top[0] / top[2]) if len(top) >= 3 and top[2] > 0 else np.nan
            check = {"target": target, "horizon": h, "seed": seed, "split": window.split,
                     "n_rows": int(len(Xr)), "n_features": len(feats),
                     "unit": X.UNIT[target], "n_rounds": int(b["n_rounds"]),
                     "rounds_in_booster": sv["rounds_in_booster"],
                     "base_value": float(sv["base"][0]), "additivity_max": sv["additivity"],
                     "contribs_max_diff": sv["contribs_diff"],
                     "unsliced_additivity_max": sv["unsliced_additivity"],
                     "written_forecast_max_diff": raw_diff, "kw_additivity_max": kw_add,
                     "n_clipped": int((~kv["unclipped"]).sum()), "mos_applied": bool(mos),
                     "dominance_first_over_third": dominance,
                     "nan_cells": int(Xr.isna().to_numpy().sum())}
            scale = SCALE[X.UNIT[target]]
            check["additivity_relative"] = sv["additivity"] / scale
            for key, value, tol in (("additivity_max", sv["additivity"], SHAP_TOL * scale),
                                    ("contribs_max_diff", sv["contribs_diff"],
                                     CONTRIBS_TOL * scale),
                                    ("written_forecast_max_diff", raw_diff, SHAP_TOL * scale),
                                    ("kw_additivity_max", kw_add, KW_TOL)):
                if not value <= tol:
                    raise RunnerError(f"{name}: {key} {value:.3g} above {tol:g} on "
                                      f"{window.split} (shap skill: never silence an "
                                      "additivity failure)")
            if dominance > DOMINANCE_FACTOR:
                warnings.append(f"{name}: SHAP dominance, the first feature "
                                f"({ft['feature'].iloc[0]}) is {dominance:.1f} x the third "
                                f"(stage-7 red flag above {DOMINANCE_FACTOR:g})")
            npz = out / f"shap_values_{target}_D{h}_seed{seed}_{window.split}.npz"
            np.savez_compressed(npz, values=sv["values"].astype(np.float32),
                                values_kw=kv["values"].astype(np.float32),
                                base=sv["base"], base_kw=kv["base"],
                                timestamp_ns=Xr.index.asi8, features=np.array(feats))
            outputs.append(npz)
            features.append(ft)
            groups.append(gt)
            checks.append(check)
            ctx.log.note(f"{name} {window.split}: SHAP on {len(Xr)} rows, additivity "
                         f"{sv['additivity']:.2e} ({X.UNIT[target]}), pred_contribs "
                         f"{sv['contribs_diff']:.2e}, unsliced booster "
                         f"{sv['unsliced_additivity']:.3g}; kW view {kw_add:.2e} kW on "
                         f"{int(kv['unclipped'].sum())} unclipped rows; top "
                         f"{ft['feature'].iloc[0]}")
    tables = {"xgb_shap_features": pd.concat(features, ignore_index=True),
              "xgb_shap_groups": pd.concat(groups, ignore_index=True),
              "xgb_shap_checks": pd.DataFrame(checks)}
    tables["xgb_shap_rank_stability"] = rank_stability(tables["xgb_shap_features"])
    for key, frame in tables.items():
        path = out / f"{key}{tag}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        outputs.append(path)
    return outputs, tables, warnings


def xgb_shap_test(ctx: Context, window: Window) -> tuple[list[Path], list[dict]]:
    """The SHAP files of §6.10 on the test window, in the step `test` (D-31, D-32)."""
    access = []

    def table_for(h: int) -> pd.DataFrame:
        table = data.model_table(h, ("train", "val", "test"), allow_test=True)
        access.append(log_test_access(ctx, f"XGB SHAP on the test window of D+{h} (features)",
                                      [data.feature_file(h), data.gfs_file(h)],
                                      int((table["split"] == "test").sum())))
        return table

    outputs, _, warnings = xgb_shap(ctx, window, ctx.out / "shap_test", table_for, "_test")
    for w in warnings:
        ctx.log.warn(w)
    return outputs, access


# ── TFT: attention and variable selection ───────────────────────────────────


def same_hour_mask(steps: int, length: int) -> np.ndarray:
    """[step, position]: True where encoder position p (lag -(length - p) h before T0) is a whole
    number of days before the valid hour of decoder step t (T0 + t h)."""
    t = np.arange(steps)[:, None]
    lag = length - np.arange(length)[None, :]  # position 0 is lag -length
    return (t + lag) % 24 == 0


def attention_tables(att: np.ndarray, keep: np.ndarray, horizon: np.ndarray, what: dict
                     ) -> tuple[pd.DataFrame, list[dict]]:
    """Mean attention by lag for all kept steps and per horizon block; the same-hour share."""
    att = att.astype(np.float64)
    n, steps, length = att.shape
    mass = att.sum(axis=2)  # the encoder part of each step's attention
    norm = att / np.where(mass > 0, mass, 1.0)[..., None]
    same = same_hour_mask(steps, length)
    lags, summary = [], []
    for block in ("all", 1, 2, 3):
        sel = keep if block == "all" else keep & (horizon == block)
        if not sel.any():
            continue
        mean = att[sel].mean(axis=0)
        lags.append(pd.DataFrame({**what, "block": str(block), "lag_h": np.arange(-length, 0),
                                  "attention": mean}))
        share = (norm * same[None]).sum(axis=2)[sel]
        reference = np.broadcast_to(same.sum(axis=1) / length, (n, steps))[sel]
        first = mean[0]
        later = [mean[j] for j in range(24, length, 24)]
        summary.append({**what, "block": str(block), "encoder_positions": length,
                        "n_steps": int(sel.sum()), "encoder_mass_mean": float(mass[sel].mean()),
                        "same_hour_share": float(share.mean()),
                        "same_hour_reference": float(reference.mean()),
                        "same_hour_ratio": float(share.mean() / reference.mean()),
                        "argmax_lag_h": int(np.argmax(mean) - length),
                        "first_position_ratio": float(first / np.mean(later)) if later else np.nan,
                        "earliest_lag_h": -length})
    return pd.concat(lags, ignore_index=True), summary


def variable_table(record: dict, what: dict) -> pd.DataFrame:
    """X48: the variable-selection weights per channel and per variable."""
    rows = []
    for part in ("encoder", "decoder"):
        weights = record.get(f"{part}_importance") or {}
        total = sum(weights.values())
        for channel, w in weights.items():
            rows.append({**what, "part": part, "channel": channel,
                         "variable": channel.removesuffix(FILLED), "weight": float(w),
                         "part_total": float(total)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["variable_weight"] = frame.groupby(["part", "variable"])["weight"].transform("sum")
    return frame


def tft_explain(ctx: Context) -> dict[str, pd.DataFrame]:
    J = _j()
    lags, summary, variables = [], [], []
    for step in TFT_STEPS:
        if step not in ctx.completed:
            continue
        best = json.loads(record_file(ctx, step, "best_params.json").read_text(encoding="utf-8"))
        job = json.loads(record_file(ctx, step, "job/job.json").read_text(encoding="utf-8"))
        for rec in best.values():
            target, seed = rec["target"], int(rec["seed"])
            what = {"step": step, "model": J.STEP_MODEL[step], "target": target, "seed": seed}
            fc = J.read_arrays(record_file(ctx, step, f"worker/{J.forecast_name(target, seed)}"))
            arrays = J.read_arrays(record_file(ctx, step, f"job/{job['files'][target]['path']}"))
            att = fc["val_attention"]
            if att.shape[-1] != job["meta"][target]["encoder_hours"]:
                raise RunnerError(f"{step} {target}: attention over {att.shape[-1]} positions, "
                                  f"not the {job['meta'][target]['encoder_hours']} of its "
                                  "encoder (M22)")
            lag, summ = attention_tables(att, arrays["val_dec_keep"].astype(bool),
                                         arrays["val_dec_horizon"], what)
            lags.append(lag)
            summary += summ
            variables.append(variable_table(rec, what))
    if not lags:
        return {}
    return {"tft_attention_lag": pd.concat(lags, ignore_index=True),
            "tft_attention_summary": pd.DataFrame(summary),
            "tft_variable_importance": pd.concat(variables, ignore_index=True)}


# ── Optuna: hyper-parameter importance ──────────────────────────────────────


def importance_rows(study, what: dict, seeds=FANOVA_SEEDS) -> list[dict]:
    """fANOVA importance with seeded evaluators (M30): mean and std over the seeds."""
    import optuna

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    base = {**what, "n_complete": len(complete), "n_trials": len(study.trials),
            "all_in_startup": len(complete) <= TPE_STARTUP}
    if len(complete) < 2:
        return [{**base, "param": "", "importance_mean": np.nan, "importance_std": np.nan,
                 "note": "fewer than 2 complete trials: not computed"}]
    runs = []
    for s in seeds:
        ev = optuna.importance.FanovaImportanceEvaluator(seed=int(s))
        runs.append(optuna.importance.get_param_importances(study, evaluator=ev))
    params = sorted({p for r in runs for p in r})
    rows = []
    for p in params:
        v = np.array([r.get(p, 0.0) for r in runs], dtype=float)
        note = "inside TPE's random start-up" if base["all_in_startup"] else ""
        if p == "hidden_continuous_size":
            note = (note + "; " if note else "") + ("sampled value; the network uses "
                                                    "min(value, hidden_size) (X45)")
        rows.append({**base, "param": p, "importance_mean": float(v.mean()),
                     "importance_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                     "note": note})
    return rows


def load_study_copy(db: Path, name: str, workdir: Path):
    """The study from a copy of its database: the run's file is never opened (X: its hash)."""
    import optuna

    copy = workdir / f"{name}.db"
    shutil.copyfile(db, copy)
    storage = optuna.storages.RDBStorage(f"sqlite:///{copy.resolve().as_posix()}")
    try:
        study = optuna.load_study(study_name=name, storage=storage)
        trials = study.trials  # read now; the storage is closed below
        frozen = optuna.create_study(direction="minimize")
        frozen.add_trials(trials)
        return frozen
    finally:
        storage.remove_session()
        storage.engine.dispose()


def study_list(ctx: Context) -> list[tuple[dict, Path, str]]:
    X, J = _x(), _j()
    out = []
    if "xgb" in ctx.completed:
        best = json.loads(record_file(ctx, "xgb", "best_params.json").read_text(encoding="utf-8"))
        db = record_file(ctx, "xgb", "optuna.db")
        for name, b in best.items():
            out.append(({"step": "xgb", "model": X.MODEL, "target": b["target"],
                         "horizon": int(b["horizon"]), "seed": int(b["seed"])}, db,
                        f"{X.MODEL}_{name}"))
    for step in STUDY_STEPS:
        if step not in ctx.completed:
            continue
        best = json.loads(record_file(ctx, step, "best_params.json").read_text(encoding="utf-8"))
        db = record_file(ctx, step, "worker/optuna.db")
        for rec in best.values():
            if not rec.get("study"):
                continue
            out.append(({"step": step, "model": J.STEP_MODEL[step], "target": rec["target"],
                         "horizon": "all", "seed": int(rec["seed"])}, db,
                        f"{J.STEP_MODEL[step]}_{rec['target']}_seed{rec['seed']}"))
    return out


def hparam_importance(ctx: Context) -> pd.DataFrame:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for what, db, name in study_list(ctx):
            study = load_study_copy(db, name, Path(tmp))
            rows += importance_rows(study, what)
    return pd.DataFrame(rows)


# ── the step ────────────────────────────────────────────────────────────────


def run_explain(ctx: Context) -> Result:
    import optuna
    import shap
    import xgboost

    outputs, values, summary = [], {}, [f"explain ({ctx.mode}, D-31): CPU, nothing trained"]
    windows = [w for w in ctx.windows if w.split == "val"]  # D-32: test only in the step test
    if len(windows) != 1:
        raise RunnerError("explain needs exactly one validation window")
    window = windows[0]
    tables: dict[str, pd.DataFrame] = {}
    if "xgb" in ctx.completed:
        def table_for(h: int) -> pd.DataFrame:
            return data.model_table(h, ("train", "val"))

        with ctx.log.step(f"XGB SHAP on {window.label}"):
            out, xt, warnings = xgb_shap(ctx, window, ctx.out / "xgb", table_for)
        outputs += out
        tables.update(xt)
        for w in warnings:
            ctx.log.warn(w)
        c = xt["xgb_shap_checks"]
        values["xgb"] = {"explained": len(c), "rows": int(c["n_rows"].sum()),
                         "additivity_relative_max": float(c["additivity_relative"].max()),
                         "additivity_max": float(c["additivity_max"].max()),
                         "contribs_max_diff": float(c["contribs_max_diff"].max()),
                         "unsliced_additivity_max": float(c["unsliced_additivity_max"].max()),
                         "kw_additivity_max": float(c["kw_additivity_max"].max())}
        summary.append(f"  XGB SHAP: {len(c)} boosters on {int(c['n_rows'].sum())} rows; "
                       f"additivity <= {c['additivity_relative'].max():.1e} of the output scale; "
                       f"the unsliced booster would be off by up to "
                       f"{c['unsliced_additivity_max'].max():.3g} (training unit, X46)")
    with ctx.log.step("TFT attention and variable selection (stored W4 outputs)"):
        tt = tft_explain(ctx)
    for key, frame in tt.items():
        path = ctx.out / "tft" / f"{key}.csv"
        path.parent.mkdir(exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
        outputs.append(path)
    if tt:
        s = tt["tft_attention_summary"]
        s_all = s[s["block"] == "all"]
        values["tft"] = {"models": int(len(s_all)),
                         "same_hour_ratio": {f"{r.step}_{r.target}": r.same_hour_ratio
                                             for r in s_all.itertuples()}}
        summary.append(f"  TFT: attention of {len(s_all)} model-target pairs; same-hour ratio "
                       f"{s_all['same_hour_ratio'].min():.2f} to "
                       f"{s_all['same_hour_ratio'].max():.2f}")
    with ctx.log.step("Optuna hyper-parameter importance (fANOVA, ten seeds)"):
        hp = hparam_importance(ctx)
    if not hp.empty:
        path = ctx.out / "hparam_importance.csv"
        hp.to_csv(path, index=False, lineterminator="\n")
        outputs.append(path)
        n_studies = hp.groupby(["step", "target", "horizon", "seed"]).ngroups
        values["hparam"] = {"studies": int(n_studies),
                            "in_startup": int(hp.drop_duplicates(
                                ["step", "target", "horizon", "seed"])["all_in_startup"].sum())}
        summary.append(f"  Optuna: {n_studies} studies, "
                       f"{values['hparam']['in_startup']} inside TPE's random start-up")
    config = {"decision": "D-31 (docs/stage9_plan.md §19.5)", "window": window.as_dict(),
              "rows": "the daytime scoring hours of the window with AC truth (§14.1), "
                      "op-limited hours included; never fit rows",
              "explainer": "shap.TreeExplainer, feature_perturbation='tree_path_dependent', no "
                           "background; the booster sliced to n_rounds = best iteration + 1",
              "output": "the booster's raw output in the training unit (k or kW); the kW view "
                        "phi_kW = b * theo * phi_k for the indirect route (X47)",
              "groups": "selection.common.group, one group per feature by exact name (M27)",
              "tolerances": {"additivity_relative": SHAP_TOL,
                             "pred_contribs_relative": CONTRIBS_TOL, "scale": SCALE,
                             "kw_view_kw": KW_TOL,
                             "why": "TreeSHAP sums in float32: about 1e-6 of the output scale"},
              "fanova_seeds": list(FANOVA_SEEDS), "tpe_startup_trials": TPE_STARTUP,
              "versions": {"shap": shap.__version__, "xgboost": xgboost.__version__,
                           "optuna": optuna.__version__},
              "non_causal": NON_CAUSAL,
              "not_explained": "BiLSTM and PatchTST (no attribution in v6; scope rule of §16b)"}
    path = ctx.out / "explain_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    outputs.append(path)
    return Result(outputs, values=values, summary=summary)
