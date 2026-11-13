"""S-38 and S-41 (docs/stage9_plan.md §21; D-36): the steps `cv` and `ablation`, on the synthetic
tables of the xgb tests.

T-100  cv (M43, M44, X68, X69): the folds are the stage-7 folds over the train rows and every fold
       value equals an independent fit and score; op-limited hours are held rows; `fit_mos` and
       the CV share one loop; the step refuses without the xgb record or with a changed one
T-101  ablation (M37-M42, X70, X71): modes by the groups of the feature contract (the rolling std
       features go with the rolling mean ones), `full` reproduces the xgb step and a changed
       baseline fails, constraints by exact name for a reduced set, the delta against the same
       seed, the verdict by the registered margin, the test window forecast and logged in a
       final run and scored only in phase 2
T-102  integration (X72-X74): the order of the steps, the final estimate by the seeds ratio, the
       report's summaries with their hashes, the runbook's segments
"""

import contextlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from test_xgb import DIRECT, INDIRECT, SEED, fake_report, tables, wire  # noqa: F401

from solarpv.config.constants import TEST_START
from solarpv.config.paths import DOCS_DIR
from solarpv.config.provenance import sha256
from solarpv.evaluation import evaluate as E
from solarpv.features.contract import features_for_horizon
from solarpv.models import xgb_sota as X
from solarpv.pipeline import analyses, config, data, runner, steps
from solarpv.pipeline.config import Window
from solarpv.pipeline.runner import Runner, scale_dev_to_final
from solarpv.selection import common as selection_common


class StubLog:
    """What a step uses of the run log."""

    def __init__(self):
        self.notes, self.warnings = [], []

    def note(self, text):
        self.notes.append(text)

    def warn(self, text):
        self.warnings.append(text)

    @contextlib.contextmanager
    def step(self, name):
        yield


@pytest.fixture(scope="module")
def analysis_run(tmp_path_factory, tables):  # noqa: F811
    """One smoke run of baselines, xgb, evaluate, cv and ablation on the synthetic tables, D+1."""
    tmp = tmp_path_factory.mktemp("analyses")
    with pytest.MonkeyPatch.context() as mp:
        w = wire(tmp, mp, tables)
        mp.setattr(E, "TEST_ACCESS_LOG", tmp / "test_access_log.csv")
        r = Runner(runs_dir=tmp / "runs", preflight=fake_report, echo=False)
        code = r.run("smoke", ["baselines", "xgb", "evaluate", "cv", "ablation"], horizons=[1])
        (run_dir,) = (tmp / "runs").iterdir()
        yield SimpleNamespace(code=code, dir=run_dir, runner=r,
                              access_log=tmp / "test_access_log.csv", **vars(w))


@pytest.fixture(autouse=True)
def access_log(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "TEST_ACCESS_LOG", tmp_path / "test_access_log.csv")
    return tmp_path / "test_access_log.csv"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_ctx(run, step, tmp_path, mode="smoke", windows=None, allow_test=False, completed=None,
             run_dir=None):
    """A step context over the files of the module's smoke run."""
    cfg = yaml.safe_load((run.dir / "config.yaml").read_text(encoding="utf-8"))
    if completed is None:
        completed = {s: read(run.dir / s / "step.json") for s in ("xgb", "cv", "ablation")
                     if (run.dir / s / "step.json").exists() and s != step}
    out = tmp_path / f"{step}_out"
    out.mkdir(parents=True, exist_ok=True)
    return steps.Context(
        run_id=run.dir.name, run_dir=run_dir or run.dir, mode=mode, config=cfg,
        windows=windows or [Window(**w) for w in cfg["windows"]], out=out, log=StubLog(),
        completed=completed, step=step, command="test", allow_test=allow_test)


# ── the runner and the run ──────────────────────────────────────────────────


def test_the_steps_follow_xgb_and_the_run_completes(analysis_run):
    assert analysis_run.code == 0
    order = list(config.STEP_ORDER)
    assert order.index("xgb") < order.index("cv") < order.index("ablation") < order.index("bilstm")
    assert {"cv", "ablation"} <= set(steps.IMPLEMENTED)
    assert "cv" in runner.select_steps(None, None, "report")
    for step in ("cv", "ablation"):
        assert read(analysis_run.dir / step / "step.json")["values"]["test_access"] == []


# ── T-100 cv ────────────────────────────────────────────────────────────────


def test_the_cv_folds_are_the_stage7_folds_and_match_an_independent_fit(analysis_run):
    folds = pd.read_csv(analysis_run.dir / "cv" / "cv_folds.csv")
    summary = pd.read_csv(analysis_run.dir / "cv" / "cv_summary.csv")
    best = read(analysis_run.dir / "xgb" / "best_params.json")
    assert set(folds["fold"]) == set(range(5))
    assert (pd.to_datetime(folds["last_day"]) < pd.Timestamp("2024-02-01")).all()  # train only
    assert folds.groupby(["target", "horizon", "seed"]).size().eq(5).all()
    table = data.model_table(1, ("train",))
    usable = analyses.cv_rows(table)
    scope = X.train_scope(table, smoke=True)
    days = pd.Series(X.local_day(table.index), index=table.index)
    blocks = selection_common.date_folds(days[scope])  # the stage-7 folds, not oof_folds
    for target in ("indirect", "direct"):
        rec = best[f"{target}_D1_seed{SEED}"]
        feats = rec["feature_order"]
        cons = X.constraint_map(feats, rec["monotone_setting"])
        fit_all = scope & X.fit_mask(table, target)
        for i, (f_fit, f_held) in enumerate(blocks):
            rows = f_fit.reindex(table.index, fill_value=False) & fit_all
            held = f_held.reindex(table.index, fill_value=False) & usable
            booster = X.fit_fixed(table.loc[rows, feats], table[X.TARGET_COLUMN[target]][rows],
                                  rec["searched"], SEED, cons, int(rec["n_rounds"]))
            pred = X.predict(booster, table.loc[held, feats], int(rec["n_rounds"]))
            value = X.nrmse_kw(target, pred, table.loc[held, "theo_kw"].to_numpy(float),
                               table.loc[held, X.TRUTH_KW].to_numpy(float))
            row = folds[(folds["target"] == target) & (folds["fold"] == i)].iloc[0]
            assert row["nrmse_kw_pct"] == pytest.approx(value, abs=1e-12), (target, i)
            assert row["held_rows"] == int(held.sum()) and row["fit_rows"] == int(rows.sum())
        s = summary[summary["target"] == target].iloc[0]
        vals = folds[folds["target"] == target]["nrmse_kw_pct"].to_numpy()
        assert s["cv_mean"] == pytest.approx(vals.mean()) and s["cv_std"] == pytest.approx(
            vals.std(ddof=1))
        assert s["validation_uncorrected"] == pytest.approx(rec["value_kw"])


def test_cv_keeps_the_op_limited_hours_that_the_l09_rule_drops(analysis_run):
    """X69: the held rows follow the scoring rule, which includes op-limited hours."""
    folds = pd.read_csv(analysis_run.dir / "cv" / "cv_folds.csv")
    assert (folds["held_rows"] >= folds["held_rows_l09_rule"]).all()
    assert (folds["held_rows"] > folds["held_rows_l09_rule"]).any()  # a fold has such an hour
    table = data.model_table(1, ("train",))
    held = analyses.cv_rows(table)
    assert (held & table["op_limited_flag"].astype(bool)).any()
    assert not (held & table["warmup"].astype(bool)).any()


def test_fit_mos_and_the_cv_share_one_loop(analysis_run):
    """X68: the L-09 coefficients equal the OLS over `oof_folds`, the generator the CV uses."""
    table = data.model_table(1, ("train",))
    best = read(analysis_run.dir / "xgb" / "best_params.json")
    rec = best[f"indirect_D1_seed{SEED}"]
    feats = rec["feature_order"]
    cons = X.constraint_map(feats, rec["monotone_setting"])
    scope = X.train_scope(table, smoke=True)
    mos = X.fit_mos(table, "indirect", feats, rec["searched"], SEED, cons, int(rec["n_rounds"]),
                    scope)
    preds, truth = [], []
    for _i, _rows, held, pred in X.oof_folds(table, "indirect", feats, rec["searched"], SEED,
                                             cons, int(rec["n_rounds"]), scope,
                                             X.mos_rows(table)):
        preds.append(X.to_kw("indirect", pred, table.loc[held, "theo_kw"].to_numpy()))
        truth.append(table.loc[held, X.TRUTH_KW].to_numpy(float))
    a, b = X.ols(np.concatenate(preds), np.concatenate(truth))
    assert (mos.a, mos.b) == (a, b)


def test_the_steps_refuse_without_the_xgb_step_or_with_a_changed_record(analysis_run, tmp_path):
    ctx = make_ctx(analysis_run, "cv", tmp_path, completed={})
    with pytest.raises(steps.RunnerError, match="run xgb first"):
        analyses.analysis_inputs(ctx)
    record = read(analysis_run.dir / "xgb" / "step.json")
    record["outputs"]["best_params.json"] = "0" * 64  # as if the file had been edited
    ctx = make_ctx(analysis_run, "ablation", tmp_path, completed={"xgb": record})
    with pytest.raises(steps.RunnerError, match="modified"):
        analyses.analysis_inputs(ctx)
    with pytest.raises(steps.RunnerError, match="no record"):
        analyses.record_of({}, "indirect_D1_seed42")  # never a default (M38)


def test_the_cv_refuses_rows_outside_train(analysis_run, tmp_path, monkeypatch):
    real = data.model_table
    monkeypatch.setattr(data, "model_table", lambda h, splits=("train",), **kw: real(
        h, ("train", "val"), **kw))
    ctx = make_ctx(analysis_run, "cv", tmp_path)
    with pytest.raises(steps.RunnerError, match="outside train"):
        analyses.run_cv(ctx)


# ── T-101 ablation ──────────────────────────────────────────────────────────


def group_of(f):
    return selection_common.group(f)


def test_the_modes_follow_the_contract_groups_and_rolling_std_goes_with_the_mean():
    feats = list(features_for_horizon(1)) + ["fcst_gfs_ghi", "irradiance_clear_index_prevday"]
    modes, skipped = analyses.ablation_modes(feats)
    for g in ("nwp_gfs", "clearsky", "other", "time_encodings", "lag_power", "rolling_stats",
              "history_other"):
        assert modes[f"no_{g}"] == [f for f in feats if group_of(f) != g], g
    rolling = [f for f in feats if "rolling_" in f]
    assert len(rolling) == 12 and any("rolling_std" in f for f in rolling)  # M39
    assert not any("rolling_" in f for f in modes["no_rolling_stats"])
    assert not any("rolling_" in f or "_lag_" in f or f == "irradiance_clear_index_prevday"
                   for f in modes["no_history"])
    assert modes["minimal"] == [f for f in feats if group_of(f) == "time_encodings"]  # M40
    assert modes["full"] == feats and "no_k_history" not in modes and not skipped


def test_a_mode_that_removes_nothing_or_everything_is_skipped_with_its_reason():
    modes, skipped = analyses.ablation_modes(["hour_sin", "hour_cos"])
    assert list(modes) == ["full"]
    assert skipped["no_time_encodings"] == "no feature left"
    assert "fewer than two" in skipped["no_history"]
    assert skipped["minimal"] == "the same features as another mode"
    with pytest.raises(KeyError):  # an unknown name is an error (one group per feature)
        analyses.ablation_modes(["hour_sin", "not_a_feature"])


def test_constraints_of_a_reduced_set_are_by_exact_name():
    """X70: rebuilt for the features left, never copied from the full list."""
    modes, _ = analyses.ablation_modes(DIRECT)
    full = X.constraint_map(modes["full"], "A")
    assert set(full) == {"ghi_clearsky", "poa_global_clearsky", "sun_elevation"}
    assert X.constraint_map(modes["no_clearsky"], "A") == {}
    assert X.constraint_map(modes["minimal"], "A") == {}
    assert X.constraint_map(modes["no_time_encodings"], "A") == full


def test_full_reproduces_the_xgb_step_and_the_deltas_are_against_the_same_seed(analysis_run):
    res = pd.read_csv(analysis_run.dir / "ablation" / "ablation_results.csv")
    best = read(analysis_run.dir / "xgb" / "best_params.json")
    for (target, h, seed), g in res.groupby(["target", "horizon", "seed"]):
        rec = best[f"{target}_D{h}_seed{seed}"]
        full = g[g["mode"] == "full"].iloc[0]
        assert full["value_kw"] == pytest.approx(rec["value_kw"], abs=1e-9)  # M38
        assert full["n_rounds"] == rec["n_rounds"] and full["delta_pp"] == 0
        assert full["n_removed"] == 0 and full["n_features"] == len(rec["feature_order"])
        for r in g[g["mode"] != "full"].itertuples():
            assert r.delta_pp == pytest.approx(r.value_kw - full["value_kw"])
            assert r.n_features + r.n_removed == full["n_features"]
    modes = pd.read_csv(analysis_run.dir / "ablation" / "ablation_modes.csv")
    assert {"full", "no_clearsky", "no_history", "minimal"} <= set(modes["mode"])
    assert modes[~modes["skipped"].astype(bool)].groupby("target")["mode"].nunique().ge(5).all()


def test_a_changed_baseline_fails_the_full_check(analysis_run, tmp_path, monkeypatch):
    real = analyses.record_of

    def tampered(best, name):
        rec = dict(real(best, name))
        rec["value_kw"] = float(rec["value_kw"]) + 1e-6
        return rec
    monkeypatch.setattr(analyses, "record_of", tampered)
    ctx = make_ctx(analysis_run, "ablation", tmp_path)
    with pytest.raises(steps.RunnerError, match="does not reproduce"):
        analyses.run_ablation(ctx)


def constructed(deltas: dict[str, list[float]], target="indirect", h=1):
    rows = []
    for mode, values in deltas.items():
        for seed, d in enumerate(values):
            rows.append({"target": target, "horizon": h, "seed": seed, "mode": mode,
                         "delta_pp": d})
    return pd.DataFrame(rows)


def test_the_verdict_follows_the_registered_margin():
    """M41: helpful above, harmful below minus max(0.05 pp, 2 x the pooled seed std)."""
    quiet = constructed({"a": [0.50, 0.52, 0.48], "b": [-0.50, -0.48, -0.52],
                         "c": [0.01, -0.01, 0.0]})
    s = analyses.ablation_summary(quiet).set_index("mode")
    assert s["margin_pp"].iloc[0] == pytest.approx(0.05)  # 2 x 0.02 < the floor
    assert list(s["verdict"]) == ["helpful", "harmful", "neutral"]
    assert not s["single_seed"].any() and (s["n_seeds"] == 3).all()
    noisy = constructed({"a": [0.35, 0.15, -0.05], "b": [0.0, 0.2, -0.2]})  # stds 0.2
    s = analyses.ablation_summary(noisy).set_index("mode")
    assert s["pooled_seed_std_pp"].iloc[0] == pytest.approx(0.2)
    assert s["margin_pp"].iloc[0] == pytest.approx(0.4)  # 2 x the pooled std
    assert s.loc["a", "mean_delta_pp"] == pytest.approx(0.15) and s.loc["a", "verdict"] == "neutral"
    one = analyses.ablation_summary(constructed({"a": [0.06], "b": [-0.06], "c": [0.04]}))
    assert one["single_seed"].all() and (one["margin_pp"] == 0.05).all()
    assert list(one["verdict"]) == ["helpful", "harmful", "neutral"]
    two = constructed({"a": [0.3, 0.3]}, "direct", 2)
    both = analyses.ablation_summary(pd.concat([quiet, two], ignore_index=True))
    assert set(zip(both["target"], both["horizon"], strict=True)) == {("indirect", 1),
                                                                       ("direct", 2)}


def test_the_summary_of_the_run_has_a_verdict_for_every_mode(analysis_run):
    summary = pd.read_csv(analysis_run.dir / "ablation" / "ablation_summary.csv")
    res = pd.read_csv(analysis_run.dir / "ablation" / "ablation_results.csv")
    assert set(summary["verdict"]) <= {"helpful", "neutral", "harmful"}
    assert summary["single_seed"].all()  # one development seed: the floor margin, said so
    assert len(summary) == len(res[res["mode"] != "full"])


def synthetic_test_window():
    return Window("test", "2024-10-01T00:00:00+00:00", "2024-10-15T00:00:00+00:00",
                  "synthetic test window")


def test_the_test_window_is_forecast_logged_and_scored_only_in_phase_2(analysis_run, tmp_path,
                                                                       access_log):
    """M37, X71: phase 1 forecasts the test hours of every fit (logged) and scores validation
    only; the step test scores those forecasts, logged again."""
    val = [Window(**w) for w in yaml.safe_load(
        (analysis_run.dir / "config.yaml").read_text(encoding="utf-8"))["windows"]]
    windows = [val[0], synthetic_test_window()]
    ctx = make_ctx(analysis_run, "ablation", tmp_path / "p1", mode="final", windows=windows)
    result = analyses.run_ablation(ctx)
    assert len(result.values["test_access"]) == 1  # one line per horizon (D+1 only here)
    log = pd.read_csv(access_log)
    assert len(log) == 1 and "ablation" in log["command"].iloc[0]
    res = pd.read_csv(ctx.out / "ablation_results.csv")
    assert not [c for c in res.columns if "test" in c]  # nothing is scored on test in phase 1
    fc = pd.read_csv(ctx.out / f"ablation_forecasts_seed{SEED}.csv")
    assert set(fc["split"]) == {"val", "test"}
    assert (pd.to_datetime(fc.loc[fc["split"] == "test", "timestamp"]) >= pd.Timestamp(
        TEST_START, tz="UTC")).all()
    # phase 2: a run folder that holds the ablation step as completed
    run2 = tmp_path / "run2"
    shutil.copytree(ctx.out, run2 / "ablation")
    rec = {"status": "ok", "folder": "ablation",
           "outputs": {p.name: sha256(p) for p in (run2 / "ablation").iterdir() if p.is_file()}}
    ctx2 = make_ctx(analysis_run, "test", tmp_path / "p2", mode="final", windows=[windows[1]],
                    allow_test=True, completed={"ablation": rec}, run_dir=run2)
    outputs, access = analyses.phase2_tables(ctx2, windows[1])
    assert len(access) == 1 and len(pd.read_csv(access_log)) == 2
    assert "score the CV comparison" in pd.read_csv(access_log)["command"].iloc[1]
    test = pd.read_csv(ctx2.out / "analyses" / "ablation_test.csv")
    # one value, recomputed independently from the file and the truth table
    row = test[(test["target"] == "direct") & (test["mode"] == "no_clearsky")].iloc[0]
    f = fc[(fc["target"] == "direct") & (fc["mode"] == "no_clearsky") & (fc["split"] == "test")]
    truth = steps.truth_for(windows[1]).reindex(pd.DatetimeIndex(pd.to_datetime(f["timestamp"],
                                                                                utc=True)))
    ok = truth["daytime"].astype(bool).to_numpy() & truth["direct"].notna().to_numpy()
    err = f["y_pred"].to_numpy()[ok] - truth["direct"].to_numpy()[ok]
    assert row["value_kw"] == pytest.approx(np.sqrt(np.mean(err ** 2)) / 3449.0 * 100.0)
    assert (test["horizon"] == 1).all() and (test["n_test_hours"] > 0).all()
    assert {p.name for p in outputs} == {"ablation_test.csv", "ablation_test_summary.csv"}
    summary = pd.read_csv(ctx2.out / "analyses" / "ablation_test_summary.csv")
    # R-G1-11: on the test split the label describes and decides nothing, so it is no `verdict`
    assert "test_label" in summary.columns and "verdict" not in summary.columns
    with pytest.raises(steps.RunnerError, match="only in the step test of a final run"):
        analyses.phase2_tables(make_ctx(analysis_run, "test", tmp_path / "p3"), windows[1])


def test_the_cv_comparison_on_the_test_window_uses_the_raw_forecast(analysis_run, tmp_path,
                                                                    access_log):
    """The step test compares the CV with the test value of the uncorrected forecast: the
    uncorrected file where L-09 corrected a target, the predictions file for the rest."""
    window = synthetic_test_window()
    base = analysis_run.base
    test_rows = base[(base.index >= window.start_ts) & (base.index < window.end_ts)
                     & base["daytime"].astype(bool)]
    run2 = tmp_path / "run2"
    (run2 / "xgb").mkdir(parents=True)
    shutil.copytree(analysis_run.dir / "cv", run2 / "cv")
    rng = np.random.default_rng(3)
    frames = {}
    for target in ("indirect", "direct"):
        raw = (rng.uniform(0.3, 0.7, len(test_rows)) if target == "indirect"
               else rng.uniform(300, 2500, len(test_rows)))
        frames[target] = pd.DataFrame({"timestamp": test_rows.index, "y_true": 0.0, "y_pred": raw,
                                       "target": target, "horizon": 1, "model": X.MODEL,
                                       "seed": SEED, "split": "test"})
    unc = frames["indirect"]  # the raw indirect forecast; the predictions file holds 2 x it
    corrected = unc.assign(y_pred=unc["y_pred"] * 2)
    pd.concat([corrected, frames["direct"]]).to_csv(
        run2 / "xgb" / f"predictions_{X.MODEL}_seed{SEED}.csv", index=False)
    unc.to_csv(run2 / "xgb" / X.uncorrected_name(SEED), index=False)
    rec_xgb = {"status": "ok", "folder": "xgb", "outputs": {
        p.name: sha256(p) for p in (run2 / "xgb").iterdir()}}
    rec_cv = {"status": "ok", "folder": "cv", "outputs": {
        p.name: sha256(p) for p in (run2 / "cv").iterdir() if p.is_file()}}
    ctx = make_ctx(analysis_run, "test", tmp_path / "p", mode="final", windows=[window],
                   allow_test=True, completed={"xgb": rec_xgb, "cv": rec_cv}, run_dir=run2)
    analyses.phase2_tables(ctx, window)
    cmp = pd.read_csv(ctx.out / "analyses" / "cv_vs_test.csv").set_index("target")
    t = test_rows[test_rows["site_total_AC_kW"].notna()]
    e = np.clip(unc.set_index("timestamp").loc[t.index, "y_pred"].to_numpy() * t["theo_kw"]
                .to_numpy(), 0, 3449.0) - t["site_total_AC_kW"].to_numpy()
    assert cmp.loc["indirect", "test_uncorrected"] == pytest.approx(
        np.sqrt(np.mean(e ** 2)) / 3449.0 * 100.0)
    assert cmp.loc["indirect", "test_minus_cv_pp"] == pytest.approx(
        cmp.loc["indirect", "test_uncorrected"] - cmp.loc["indirect", "cv_mean"])
    assert cmp.loc["direct", "n_test_hours"] == len(t)


# ── T-102 integration ───────────────────────────────────────────────────────


def test_the_final_estimate_of_the_new_steps_uses_the_seeds_ratio(tmp_path):
    """X72: three seeds in a final run against one in development."""
    for step in ("cv", "ablation"):
        folder = tmp_path / "runs" / "20261112T000000Z_dev" / step
        folder.mkdir(parents=True)
        (folder / "step.json").write_text(json.dumps({
            "mode": "dev", "status": "ok", "duration_s": 100.0,
            "config": {"budget": config.BUDGETS["dev"]["xgb"]}}), encoding="utf-8")
        assert scale_dev_to_final(step, folder) == (3.0, 3.0)
        text = Runner(runs_dir=tmp_path / "runs", echo=False).estimate(step, "final")
        assert "300 s" in text
    assert runner.BUDGET_FAMILY["cv"] == runner.BUDGET_FAMILY["ablation"] == "xgb"


def test_the_report_lists_the_summaries_with_their_hashes(analysis_run):
    r = analysis_run.runner
    assert r.run("smoke", ["report"], resume=analysis_run.dir.name, horizons=[1]) == 0
    rep = analysis_run.dir / "report"
    meta = read(rep / "report.json")
    for key in ("cv", "ablation"):
        assert key in meta["tables"] and (rep / f"report_{key}.csv").exists()
    assert "cv/cv_summary.csv" in meta["sources"]
    assert "ablation/ablation_summary.csv" in meta["sources"]
    assert not [k for k in meta["sources"] if "ablation_forecasts_" in k]  # bulky, not a source
    text = (rep / "report.md").read_text(encoding="utf-8")
    assert "Cross-validation of the xgb models" in text and "Feature-group ablation" in text


def test_the_runbook_segments_name_the_new_steps():
    """X74: a final run in segments would otherwise skip them."""
    text = (DOCS_DIR / "runbook_final.md").read_text(encoding="utf-8")
    assert "--only select,baselines,xgb,cv,ablation" in text
    assert "python -m solarpv.pipeline list" in text
