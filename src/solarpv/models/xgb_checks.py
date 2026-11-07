"""The registered checks of docs/stage9_plan.md §14 for XGB. Validation only: the tables are read
with `pipeline.data.model_table(h, ("train", "val"))`, and no test row is read.

    python -m solarpv.models.xgb_checks d15              # §14.3: configs/monotone_decision.yaml
    python -m solarpv.models.xgb_checks l09 --run <id>   # §14.4: configs/mos_decision.yaml
    python -m solarpv.models.xgb_checks r14 --run <id>   # §14.5: reported, not decided

Like `decide_k` and `nwp.check`, the checks are their own commands. They write their evidence to
`outputs/09_xgb_checks/` and their decisions to `configs/`, and never write a `predictions_*`
file into a run folder (the runner's `evaluate` step scores every such file of a model step). A
dev run is only read, and its files are checked against the hashes of its step record.

The margin of every rule is max(0.05 pp, 2 x pooled seed std), the D-12 form; the pooled std is
the root mean square of the two settings' seed stds, as in `nwp.check.decide`.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from solarpv.config.constants import (
    AC_CAPACITY_KW,
    BOOTSTRAP_REPS,
    DEV_SEED,
    HORIZONS,
    MONOTONE_DECISION_FILE,
    MOS_DECISION_FILE,
    RANDOM_SEEDS,
    TEST_START,
    TIMEZONE_LOCAL,
)
from solarpv.config.paths import OUTPUTS_DIR
from solarpv.config.provenance import git_commit, sha256
from solarpv.config.runlog import RunLog
from solarpv.evaluation.evaluate import to_kw
from solarpv.evaluation.metrics import bootstrap_mean_ci
from solarpv.evaluation.truth import FEATURES_FILE, load_truth
from solarpv.features.decide_k import XGB_PARAMS
from solarpv.models import xgb_sota as X
from solarpv.pipeline import data

OUT_DIR = OUTPUTS_DIR / "09_xgb_checks"
RUNS_DIR = OUTPUTS_DIR / "runs"
MIN_GAIN_PP = 0.05
CHECK_SPLITS = ("train", "val")


class CheckError(RuntimeError):
    """A rule of the checks is broken (wrong run, modified file, test rows)."""


def guard_output(path: Path, runs_dir: Path = RUNS_DIR) -> Path:
    """No check writes a prediction file, and nothing goes into a run folder."""
    path = Path(path)
    if path.name.startswith("predictions_"):
        raise CheckError(f"the checks never write prediction files: {path.name}")
    if path.resolve().is_relative_to(Path(runs_dir).resolve()):
        raise CheckError(f"the checks never write into a run folder: {path}")
    return path


def load_tables(horizons=HORIZONS) -> dict[int, pd.DataFrame]:
    tables = {h: data.model_table(h, CHECK_SPLITS) for h in horizons}
    first_test = pd.Timestamp(TEST_START, tz="UTC")
    for h, t in tables.items():
        if (t.index >= first_test).any() or (t["split"] == "test").any():
            raise CheckError(f"test rows in the D+{h} table of a check")
    return tables


def margin(stds) -> float:
    stds = np.asarray(list(stds), dtype=float)
    return max(MIN_GAIN_PP, 2.0 * float(np.sqrt(np.mean(np.square(stds)))))


def registered_rule(gains: dict[int, float], margins: dict[int, float]) -> dict:
    """The D-12 form: gain_1 > margin_1, and gain_h > -margin_h at the other horizons."""
    passes = {h: (gains[h] > margins[h]) if h == 1 else (gains[h] > -margins[h])
              for h in sorted(gains)}
    return {"adopt": bool(all(passes.values())), "passes": passes}


def val_score_rows(table: pd.DataFrame) -> pd.Series:
    """§14.1 on the whole validation split: daytime hours with AC truth, op-limited included."""
    return ((table["split"] == "val") & table["daytime"].astype(bool)
            & table[X.TRUTH_KW].notna())


def _write(frame_or_text, path: Path) -> Path:
    guard_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame_or_text, pd.DataFrame):
        frame_or_text.to_csv(path, index=False, lineterminator="\n")
    else:
        path.write_text(frame_or_text, encoding="utf-8", newline="\n")
    return path


def _yaml(header: str, decision: dict) -> str:
    return header + yaml.safe_dump(decision, sort_keys=False, allow_unicode=True)


# ── D-15 (§14.3) ────────────────────────────────────────────────────────────


def d15_scores(tables: dict[int, pd.DataFrame], inputs: dict[str, list[str]],
               seeds=RANDOM_SEEDS, params: dict = XGB_PARAMS, log=None) -> pd.DataFrame:
    """The fixed XGBoost of D-11 on the indirect fit rows of train, setting A (+1 on the D-15
    features present) against B (none), scored by §14.1 on the whole validation split."""
    rows = []
    for h, table in sorted(tables.items()):
        feats = X.input_features(table, "indirect", h, inputs)
        fit = X.training_rows(table, "indirect")
        val = val_score_rows(table)
        Xf, yf, Xv = table.loc[fit, feats], table.loc[fit, "target_k"], table.loc[val, feats]
        theo, truth = table.loc[val, "theo_kw"].to_numpy(), table.loc[val, X.TRUTH_KW].to_numpy()
        for setting in X.SETTINGS:
            constraints = X.constraint_map(feats, setting)
            for seed in seeds:
                full = {**params, "random_state": int(seed)}  # a new dict (X13)
                if constraints:
                    full["monotone_constraints"] = dict(constraints)
                model = xgb.XGBRegressor(**full)
                model.fit(Xf, yf, verbose=False)
                booster = model.get_booster()
                pred = X.predict(booster, Xv, booster.num_boosted_rounds())
                value = X.nrmse_kw("indirect", pred, theo, truth)
                rows.append({"horizon": h, "setting": setting,
                             "constraints": json.dumps(constraints), "seed": int(seed),
                             "nrmse_kw_pct": value, "fit_rows": int(fit.sum()),
                             "val_rows": int(val.sum())})
                if log:
                    log.note(f"D-15 D+{h} setting {setting} seed {seed}: {value:.4f} % "
                             f"(constraints {constraints or 'none'})")
    return pd.DataFrame(rows)


def d15_decide(scores: pd.DataFrame) -> dict:
    """gain_h = mean_A - mean_B (positive: dropping the constraint helps); B only if the rule
    holds, otherwise A, the v6 setting, stays (§14.3)."""
    by_h, gains, margins = {}, {}, {}
    for h, g in scores.groupby("horizon"):
        s = g.groupby("setting")["nrmse_kw_pct"]
        mean, std = s.mean(), s.std(ddof=1)
        h = int(h)
        gains[h] = float(mean["A"] - mean["B"])
        margins[h] = margin([std["A"], std["B"]])
        by_h[f"D{h}"] = {"mean_A": round(float(mean["A"]), 4),
                         "std_A": round(float(std["A"]), 4),
                         "mean_B": round(float(mean["B"]), 4),
                         "std_B": round(float(std["B"]), 4),
                         "gain_pp": round(gains[h], 4), "margin_pp": round(margins[h], 4)}
    rule = registered_rule(gains, margins)
    for h, ok in rule["passes"].items():
        by_h[f"D{h}"]["passes"] = bool(ok)
    return {"indirect_setting": "B" if rule["adopt"] else "A", "by_horizon": by_h}


def marginal_signs(table: pd.DataFrame) -> pd.DataFrame:
    """C-33: Spearman correlation of k and of AC with each D-15 feature on the indirect fit rows
    of train (the rows the k model learns from)."""
    fit = X.training_rows(table, "indirect")
    rows = []
    for f in X.D15_FEATURES:
        rows.append({"feature": f, "rows": int(fit.sum()),
                     "spearman_k": float(table.loc[fit, "target_k"].corr(
                         table.loc[fit, f], method="spearman")),
                     "spearman_ac_kw": float(table.loc[fit, X.TRUTH_KW].corr(
                         table.loc[fit, f], method="spearman"))})
    return pd.DataFrame(rows)


def run_d15(out_dir: Path = OUT_DIR, decision_file: Path = MONOTONE_DECISION_FILE,
            tables: dict | None = None, inputs: dict | None = None, seeds=RANDOM_SEEDS,
            command: str = "python -m solarpv.models.xgb_checks d15") -> dict:
    out_dir = Path(out_dir)
    params_before = dict(XGB_PARAMS)
    with RunLog("09_xgb_checks", command, out_dir=out_dir) as log:
        if tables is None:
            log.inputs([data.feature_file(h) for h in HORIZONS]
                       + [data.gfs_file(h) for h in HORIZONS] + [X.MODEL_INPUTS_FILE])
            tables = load_tables()
        inputs = X.load_model_inputs(X.MODEL_INPUTS_FILE) if inputs is None else inputs
        with log.step("D-15: fixed XGBoost of D-11, settings A and B, three seeds"):
            scores = d15_scores(tables, inputs, seeds, log=log)
        result = d15_decide(scores)
        signs = marginal_signs(tables[min(tables)])
        if XGB_PARAMS != params_before:
            raise CheckError("decide_k.XGB_PARAMS changed during the D-15 check (X13)")
        for key, r in result["by_horizon"].items():
            log.note(f"D-15 {key}: A {r['mean_A']:.4f} +/- {r['std_A']:.4f}, B "
                     f"{r['mean_B']:.4f} +/- {r['std_B']:.4f}; gain {r['gain_pp']:+.4f} pp, "
                     f"margin {r['margin_pp']:.4f} pp, passes {r['passes']}")
        for row in signs.itertuples():
            log.note(f"C-33 marginal sign on the indirect fit rows of train: {row.feature} "
                     f"Spearman k {row.spearman_k:+.3f}, AC {row.spearman_ac_kw:+.3f}")
        log.value("d15_indirect_setting", result["indirect_setting"])
        constraints_a = {f"D{h}": X.constraint_map(X.input_features(t, "indirect", h, inputs),
                                                   "A") for h, t in sorted(tables.items())}
        decision = {
            "decision": "D-15",
            "decided_on": datetime.now(UTC).date().isoformat(),
            "features": list(X.D15_FEATURES), "sign": X.D15_SIGN,
            "settings": {"indirect": result["indirect_setting"], "direct": "A"},
            "setting_meaning": dict(X.SETTINGS),
            "rule": "B for k only if gain_1 > margin_1 and gain_h > -margin_h at D+2 and D+3; "
                    "gain = mean_A - mean_B; margin = max(0.05 pp, 2 x pooled seed std); "
                    "otherwise A, the v6 setting (docs/stage9_plan.md §14.3). The direct target "
                    "keeps the D-15 list with +1, not part of the question",
            "model": "the fixed XGBoost of D-11 (decide_k.XGB_PARAMS), indirect fit rows of "
                     "train, scored on the daytime validation hours with AC truth (§14.1)",
            "seeds": [int(s) for s in seeds],
            "results": result["by_horizon"],
            "constraints_A_indirect": constraints_a,
            "marginal_sign_train": {r.feature: {"spearman_k": round(r.spearman_k, 4),
                                                "spearman_ac_kw": round(r.spearman_ac_kw, 4)}
                                    for r in signs.itertuples()},
            "git_commit": git_commit(),
        }
        outputs = [_write(scores, out_dir / "d15_check.csv"),
                   _write(signs, out_dir / "d15_marginal_sign.csv"),
                   _write(_yaml("# D-15: monotone constraints of XGB. Written by\n"
                                "# python -m solarpv.models.xgb_checks d15 (stage 9, "
                                "docs/stage9_plan.md §14.3).\n# Evidence: "
                                "outputs/09_xgb_checks/d15_check.csv; read by the xgb step.\n",
                                decision), Path(decision_file))]
        log.outputs(outputs)
    return decision


# ── the dev run (read only) ─────────────────────────────────────────────────


def load_xgb_run(run_id: str, runs_dir: Path = RUNS_DIR) -> tuple[Path, dict, dict]:
    """The completed xgb step of a dev run: its folder, step record and best_params.json, every
    output checked against the hash in the step record."""
    run_dir = Path(runs_dir) / run_id
    state_path = run_dir / "run.json"
    if not state_path.exists():
        raise CheckError(f"no run {run_id!r} in {runs_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["mode"] != "dev":
        raise CheckError(f"run {run_id} is a {state['mode']} run; the checks use a dev run")
    entry = state["steps"].get("xgb", {})
    if entry.get("status") != "ok":
        raise CheckError(f"run {run_id} has no completed xgb step")
    folder = run_dir / entry["folder"]
    record = json.loads((folder / "step.json").read_text(encoding="utf-8"))
    for name, digest in record["outputs"].items():
        if sha256(folder / name) != digest:
            raise CheckError(f"{folder.name}/{name} was modified after its step completed")
    best = json.loads((folder / "best_params.json").read_text(encoding="utf-8"))
    return folder, record, best


def tuned_entry(best: dict, target: str, h: int, seed: int = DEV_SEED) -> dict:
    key = f"{target}_D{h}_seed{seed}"
    if key not in best:
        raise CheckError(f"best_params.json has no {key}")
    return best[key]


# ── L-09 (§14.4) ────────────────────────────────────────────────────────────


# the correction itself lives in the model module, shared with the xgb step (X14)
mos_rows = X.mos_rows
ols = X.ols


def train_folds(table: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    """The five expanding date folds of stage 7 (3-day gap) over the train rows."""
    return X.mos_folds(table, X.train_scope(table))


def mos_fit(table: pd.DataFrame, target: str, entry: dict, seed: int) -> dict:
    """Out-of-fold kW predictions on train -> (a, b); the same base model trained on all of train
    -> the uncorrected and the corrected validation scores (§14.4)."""
    feats, cons = entry["feature_order"], entry["monotone_constraints"]
    params, n_rounds = entry["searched"], int(entry["n_rounds"])
    y = table[X.TARGET_COLUMN[target]]
    mos = X.fit_mos(table, target, feats, params, seed, cons, n_rounds, X.train_scope(table))
    fit_all = X.training_rows(table, target)
    booster = X.fit_fixed(table.loc[fit_all, feats], y[fit_all], params, seed, cons, n_rounds)
    val = val_score_rows(table)
    pred_val = X.predict(booster, table.loc[val, feats], n_rounds)
    raw_kw = to_kw(target, pred_val, table.loc[val, "theo_kw"].to_numpy())
    corrected = X.corrected_kw(mos, raw_kw)
    truth = table.loc[val, X.TRUTH_KW].to_numpy(float)

    def nrmse(p):
        return float(np.sqrt(np.mean((p - truth) ** 2)) / AC_CAPACITY_KW * 100.0)

    return {"a": mos.a, "b": mos.b, "n_oof": mos.n_oof, "n_val": int(val.sum()),
            "uncorrected_pct": nrmse(raw_kw), "corrected_pct": nrmse(corrected),
            "val_pred": pd.Series(pred_val, index=table.index[val])}


def base_forecast(folder: Path, seed: int) -> pd.DataFrame:
    """The base model's forecast of a run's xgb step: its prediction file, with the rows of every
    corrected target taken from the uncorrected file (§15, X16)."""
    preds = pd.read_csv(folder / f"predictions_{X.MODEL}_seed{seed}.csv")
    raw_path = folder / X.uncorrected_name(seed)
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        preds = pd.concat([preds[~preds["target"].isin(set(raw["target"]))], raw],
                          ignore_index=True)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
    return preds


def l09_scores(tables: dict[int, pd.DataFrame], best: dict, targets=X.TARGETS,
               seeds=RANDOM_SEEDS, log=None) -> tuple[pd.DataFrame, dict]:
    rows, val_preds = [], {}
    for h, table in sorted(tables.items()):
        for target in targets:
            entry = tuned_entry(best, target, h)
            for seed in seeds:
                r = mos_fit(table, target, entry, int(seed))
                val_preds[(target, h, int(seed))] = r.pop("val_pred")
                rows.append({"target": target, "horizon": h, "seed": int(seed), **r})
                if log:
                    log.note(f"L-09 {target} D+{h} seed {seed}: a {r['a']:+.2f} kW, b "
                             f"{r['b']:.4f} on {r['n_oof']} OOF hours; validation "
                             f"{r['uncorrected_pct']:.4f} -> {r['corrected_pct']:.4f} %")
    return pd.DataFrame(rows), val_preds


def l09_decide(scores: pd.DataFrame) -> dict:
    out = {}
    for target, g in scores.groupby("target", sort=False):
        gains, margins, by_h = {}, {}, {}
        for h, gh in g.groupby("horizon"):
            h = int(h)
            gains[h] = float(gh["uncorrected_pct"].mean() - gh["corrected_pct"].mean())
            margins[h] = margin([gh["uncorrected_pct"].std(ddof=1),
                                 gh["corrected_pct"].std(ddof=1)])
            by_h[f"D{h}"] = {"uncorrected_mean": round(float(gh["uncorrected_pct"].mean()), 4),
                             "corrected_mean": round(float(gh["corrected_pct"].mean()), 4),
                             "gain_pp": round(gains[h], 4), "margin_pp": round(margins[h], 4),
                             "a_mean_kw": round(float(gh["a"].mean()), 3),
                             "b_mean": round(float(gh["b"].mean()), 5)}
        rule = registered_rule(gains, margins)
        for h, ok in rule["passes"].items():
            by_h[f"D{h}"]["passes"] = bool(ok)
        out[target] = {"adopted": rule["adopt"], "by_horizon": by_h}
    return out


def run_l09(run_id: str, runs_dir: Path = RUNS_DIR, out_dir: Path = OUT_DIR,
            decision_file: Path = MOS_DECISION_FILE, tables: dict | None = None,
            seeds=RANDOM_SEEDS, command: str | None = None) -> dict:
    out_dir = Path(out_dir)
    command = command or f"python -m solarpv.models.xgb_checks l09 --run {run_id}"
    folder, record, best = load_xgb_run(run_id, runs_dir)
    with RunLog("09_xgb_checks", command, out_dir=out_dir) as log:
        log.inputs([folder / "best_params.json", folder / "step.json"])
        if tables is None:
            tables = load_tables(record["config"]["horizons"])
        targets = record["config"]["targets"]
        with log.step("L-09: out-of-fold linear correction of the kW forecast, three seeds"):
            scores, val_preds = l09_scores(tables, best, targets, seeds, log=log)
        # consistency: seed 42 on all of train with the dev rounds is the dev final fit
        dev = base_forecast(folder, DEV_SEED)
        worst = 0.0
        for (target, h, seed), pred in val_preds.items():
            if seed != DEV_SEED:
                continue
            d = dev[(dev["target"] == target) & (dev["horizon"] == h)].set_index("timestamp")
            worst = max(worst, float(np.max(np.abs(d.loc[pred.index, "y_pred"].to_numpy()
                                                   - pred.to_numpy()))))
        log.value("l09_max_abs_diff_seed42_vs_dev_predictions", worst)
        if worst > 1e-5:
            log.warn(f"seed {DEV_SEED} with the dev rounds differs from the dev predictions by "
                     f"{worst:.3g}: the base model is not the dev final fit")
        result = l09_decide(scores)
        for target, r in result.items():
            log.value(f"l09_adopted_{target}", r["adopted"])
            for key, v in r["by_horizon"].items():
                log.note(f"L-09 {target} {key}: {v['uncorrected_mean']:.4f} -> "
                         f"{v['corrected_mean']:.4f} %, gain {v['gain_pp']:+.4f} pp, margin "
                         f"{v['margin_pp']:.4f} pp, passes {v['passes']}")
        decision = {
            "decision": "L-09",
            "decided_on": datetime.now(UTC).date().isoformat(),
            "adopted": {t: r["adopted"] for t, r in result.items()},
            "rule": "per target: adopted only if gain_1 > margin_1 and gain_h > -margin_h at "
                    "D+2 and D+3; gain = mean(uncorrected) - mean(corrected); margin = "
                    "max(0.05 pp, 2 x pooled seed std) (docs/stage9_plan.md §14.4)",
            "base_model": "best parameters of the dev study, fixed rounds = best iteration + 1, "
                          "no early stopping; seeds 42, 123, 777",
            "correction": "OLS y = a + b * y_hat on out-of-fold kW predictions of the five "
                          "expanding date folds of train (3-day gap), daytime, AC truth, not "
                          "op-limited, not warm-up; applied as clip(a + b * y_hat, 0, 3449)",
            "source_run": run_id,
            "best_params_sha256": sha256(folder / "best_params.json"),
            "results": result,
            "coefficients": {f"{r.target}_D{r.horizon}_seed{r.seed}":
                             {"a_kw": round(r.a, 4), "b": round(r.b, 6), "n_oof": r.n_oof}
                             for r in scores.itertuples()},
            "max_abs_diff_seed42_vs_dev": worst,
            "git_commit": git_commit(),
        }
        outputs = [_write(scores, out_dir / "l09_check.csv"),
                   _write(_yaml("# L-09: bias correction (MOS) of the XGB kW forecast. Written "
                                "by\n# python -m solarpv.models.xgb_checks l09 (stage 9, "
                                "docs/stage9_plan.md §14.4).\n# Evidence: "
                                "outputs/09_xgb_checks/l09_check.csv. Read by the xgb step: a "
                                "target with\n# adopted: true gets the corrected forecast "
                                "(docs/stage9_plan.md §15).\n", decision),
                          Path(decision_file))]
        log.outputs(outputs)
    return decision


# ── R-14 (§14.5) ────────────────────────────────────────────────────────────


def bias_rows(err: pd.Series, label: dict) -> dict:
    lo, hi = bootstrap_mean_ci(err.to_numpy(), reps=BOOTSTRAP_REPS)
    mbe = float(err.mean())
    return {**label, "n": int(err.size), "mbe_kw": mbe, "ci_lo_kw": lo, "ci_hi_kw": hi,
            "mbe_pct": mbe / AC_CAPACITY_KW * 100, "ci_lo_pct": lo / AC_CAPACITY_KW * 100,
            "ci_hi_pct": hi / AC_CAPACITY_KW * 100,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "sign": "positive" if lo > 0 else "negative" if hi < 0 else "includes 0"}


def r14_bias(preds: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Mean bias error (forecast - actual, kW) on the validation daytime hours with AC truth,
    overall and per local month, with the 95 % circular block bootstrap interval of §6.9. The
    forecast is the one the run wrote: with L-09 adopted, the indirect route is the corrected
    one (§15)."""
    rows = []
    for (target, h), g in preds.groupby(["target", "horizon"], sort=True):
        g = g.set_index("timestamp").sort_index()
        use = truth.index[(truth["split"] == "val") & truth["daytime"].astype(bool)
                          & truth["direct"].notna()]
        g = g.loc[g.index.intersection(use).sort_values()]
        fc = to_kw(target, g["y_pred"].to_numpy(), truth.loc[g.index, "theo_kw"].to_numpy())
        err = pd.Series(fc - truth.loc[g.index, "direct"].to_numpy(), index=g.index)
        rows.append(bias_rows(err, {"target": target, "horizon": int(h), "period": "all"}))
        months = err.index.tz_convert(TIMEZONE_LOCAL).strftime("%Y-%m")
        for month, e in err.groupby(months):
            rows.append(bias_rows(e, {"target": target, "horizon": int(h), "period": month}))
    return pd.DataFrame(rows)


def run_r14(run_id: str, runs_dir: Path = RUNS_DIR, out_dir: Path = OUT_DIR,
            truth: pd.DataFrame | None = None, command: str | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir)
    command = command or f"python -m solarpv.models.xgb_checks r14 --run {run_id}"
    folder, record, _ = load_xgb_run(run_id, runs_dir)
    path = folder / f"predictions_{X.MODEL}_seed{DEV_SEED}.csv"
    with RunLog("09_xgb_checks", command, out_dir=out_dir) as log:
        log.inputs([path, FEATURES_FILE] if truth is None else [path])
        truth = load_truth(exclude_splits=("test",)) if truth is None else truth
        if (truth["split"] == "test").any():
            raise CheckError("test rows in the truth of R-14")
        preds = pd.read_csv(path)
        preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
        preds = preds[preds["split"] == "val"]
        table = r14_bias(preds, truth)
        for r in table[table["period"] == "all"].itertuples():
            log.note(f"R-14 {r.target} D+{r.horizon}: MBE {r.mbe_kw:+.2f} kW "
                     f"({r.mbe_pct:+.3f} % of capacity), 95 % CI [{r.ci_lo_pct:+.3f}, "
                     f"{r.ci_hi_pct:+.3f}] %, {r.sign} (n = {r.n})")
        direct = table[(table["target"] == "direct") & (table["period"] == "all")]
        confirmed = bool((direct["sign"] == "positive").all()) if len(direct) else False
        log.value("r14_direct_positive_at_every_horizon", confirmed)
        log.outputs([_write(table, out_dir / "r14_bias.csv")])
    return table


# ── command line ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m solarpv.models.xgb_checks",
                                     description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="check", required=True)
    sub.add_parser("d15", help="§14.3: monotone constraints for k")
    for name, text in (("l09", "§14.4: bias correction (MOS) of the kW forecast"),
                       ("r14", "§14.5: mean bias of the dev forecast on validation")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--run", required=True, help="the run id of a dev run with an xgb step")
    args = parser.parse_args(argv)
    if args.check == "d15":
        print(json.dumps(run_d15(), indent=2, default=str))
    elif args.check == "l09":
        print(json.dumps(run_l09(args.run), indent=2, default=str))
    else:
        table = run_r14(args.run)
        print(table[table["period"] == "all"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
