"""W5 of stage 9 (docs/stage9_plan.md §19): `select`, `explain`, `report`, the final gate, the
step `test`, `archive` and `promote`, on synthetic tables and constructed records.

T-82  select (M25, X44): verifies and reuses the stage-7 outputs; reads no table row; refuses a
      modified set; a model step refuses another model_inputs.json
T-83  X43: the final estimate from a dev step, scaled by the fits, with its upper bound
T-84  X38, X39: the gate before every final step; no slice in final; the v6 budgets
T-85  X40, X41, D-32: every read of a test row is logged, none in smoke or dev; evaluate scores
      validation only; the test split only by the step test, after the report, with --allow-test
T-86  X42, X49: the seed spread in kW from the benchmark; the warnings recomputed from the records
      equal the step texts
T-87  M26-M28, X46, X47: SHAP additivity of the sliced booster, the unsliced one fails, SHAP equals
      pred_contribs, the kW view of L-09, group attribution, rank stability on common features
T-88  M29, X48: attention summaries of constructed arrays, variable importance per variable
T-89  M30, X45: seeded fANOVA importance, the start-up flag, the effective hidden_continuous_size
T-90  M31: the report's numbers equal their files, every source is hashed, a changed source fails
T-91  archive and promote
"""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from test_xgb import DIRECT, INDIRECT, fake_report, tables, wired  # noqa: F401 (fixtures)

from solarpv.config.provenance import sha256
from solarpv.evaluation import evaluate as E
from solarpv.models import neural_job as J
from solarpv.models import xgb_sota as X
from solarpv.pipeline import (
    archive,
    checks,
    config,
    data,
    explain,
    neural,
    promote,
    report,
    runner,
    select_step,
    steps,
)
from solarpv.pipeline.runner import Runner, scale_dev_to_final


@pytest.fixture(autouse=True)
def access_log(tmp_path, monkeypatch):
    """The test-access log of every test goes to a temporary file."""
    path = tmp_path / "test_access_log.csv"
    monkeypatch.setattr(E, "TEST_ACCESS_LOG", path)
    return path


def make_runner(tmp_path, gate=None, impls=None):
    return Runner(runs_dir=tmp_path / "runs", preflight=fake_report, echo=False,
                  implemented=impls or steps.IMPLEMENTED,
                  step_gate=gate or (lambda commit: []))


def only_run(tmp_path):
    (run_dir,) = (tmp_path / "runs").iterdir()
    return run_dir


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── a synthetic XGB smoke run with explain and report ───────────────────────


@pytest.fixture
def xgb_run(wired, monkeypatch, access_log):  # noqa: F811
    r = make_runner(wired.tmp)
    code = r.run("smoke", ["baselines", "xgb", "evaluate", "explain", "report"], horizons=[1])
    run_dir = only_run(wired.tmp)
    return SimpleNamespace(code=code, dir=run_dir, runner=r, **vars(wired))


# ── T-82 select ─────────────────────────────────────────────────────────────


def stage7_folder(wired, direct_passes=False):  # noqa: F811
    """Committed-looking stage-7 outputs for D+1: the indirect 05e set, and the direct route
    with a failed inside-train check (the full pool, F-29)."""
    folder = wired.tmp / "07_feature_selection"
    folder.mkdir()
    cols = select_step.table_columns(1)
    from solarpv.selection.common import pool

    full = pool(cols, 1)
    for t, feats in (("indirect", INDIRECT), ("direct", DIRECT)):
        (folder / f"final_features_{t}_D1.json").write_text(json.dumps(
            {"recommended": {"features": feats, "n_features": len(feats)}}), encoding="utf-8")
        for h in (2, 3):
            (folder / f"final_features_{t}_D{h}.json").write_text(json.dumps(
                {"recommended": {"features": feats}}), encoding="utf-8")
    pd.DataFrame([{"target": "indirect", "horizon": 1, "passes": True},
                  {"target": "direct", "horizon": 1, "passes": direct_passes}]).to_csv(
        folder / "inner_train_check.csv", index=False)
    pd.DataFrame([{"target": "indirect", "horizon": 1, "red_flag": False},
                  {"target": "direct", "horizon": 1, "red_flag": True}]).to_csv(
        folder / "validation_check.csv", index=False)
    (folder / "selection_manifest.json").write_text(json.dumps(
        {"git_commit": "190e4df", "run_log": "x.log", "splits_read_for_selection": ["train"],
         "validation_read": "S8 only"}), encoding="utf-8")
    inputs = {"indirect_D1": {"source": "05e", "n_features": len(INDIRECT),
                              "features": INDIRECT},
              "direct_D1": {"source": "full pool", "n_features": len(full), "features": full}}
    path = folder / "model_inputs.json"
    path.write_text(json.dumps(inputs), encoding="utf-8")
    return folder, path, full


@pytest.fixture
def selection(wired, monkeypatch):  # noqa: F811
    folder, path, full = stage7_folder(wired)
    monkeypatch.setattr(X, "MODEL_INPUTS_FILE", path)
    monkeypatch.setattr(checks, "tracked_unmodified",
                        lambda p: {"path": str(p), "ok": True, "detail": "tracked, unmodified"})
    monkeypatch.setattr(checks, "stage7_provenance",
                        lambda m: [{"path": "t.parquet", "ok": True, "detail": "selected"}])
    monkeypatch.setattr(select_step, "code_changes", lambda c: [])
    return SimpleNamespace(folder=folder, path=path, full=full, **vars(wired))


def test_select_verifies_and_reuses_without_reading_a_row(selection, monkeypatch):
    calls = []
    real = pd.read_parquet

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)
    monkeypatch.setattr(data.pd, "read_parquet", spy)
    r = make_runner(selection.tmp)
    assert r.run("smoke", ["select"], horizons=[1]) == 0
    assert not calls  # the parquet schemas only (no row of any table)
    rec = read(only_run(selection.tmp) / "select" / "step.json")
    assert rec["values"]["model_inputs_sha256"] == sha256(selection.path)
    assert rec["values"]["inputs"] == {"indirect_D1": len(INDIRECT),
                                       "direct_D1": len(selection.full)}
    sel = read(only_run(selection.tmp) / "select" / "select.json")
    assert sel["rows_read"] == 0 and all(r["matches"] for r in sel["inputs"])
    copy = only_run(selection.tmp) / "select" / "sets" / "model_inputs.json"
    assert sha256(copy) == sha256(selection.path)


def test_select_refuses_inputs_that_differ_from_the_registered_rule(selection):
    data_ = read(selection.path)
    data_["direct_D1"]["features"] = data_["direct_D1"]["features"][:-1]
    data_["direct_D1"]["n_features"] -= 1
    selection.path.write_text(json.dumps(data_), encoding="utf-8")
    r = make_runner(selection.tmp)
    assert r.run("smoke", ["select"], horizons=[1]) == 1
    state = read(only_run(selection.tmp) / "run.json")
    assert "differs from the registered rule" in state["steps"]["select"]["error"]


def test_select_refuses_a_manifest_without_the_single_look(selection):
    m = read(selection.folder / "selection_manifest.json")
    m["validation_read"] = "every step"
    (selection.folder / "selection_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert make_runner(selection.tmp).run("smoke", ["select"], horizons=[1]) == 1


def test_a_model_step_refuses_another_model_inputs_file(selection):
    r = make_runner(selection.tmp)
    assert r.run("smoke", ["select"], horizons=[1]) == 0
    run_dir = only_run(selection.tmp)
    ctx = SimpleNamespace(mode="smoke", completed={"select": read(run_dir / "select" /
                                                                  "step.json")})
    steps.check_selected_inputs(ctx)  # the verified file: accepted
    selection.path.write_text(selection.path.read_text(encoding="utf-8") + " ",
                              encoding="utf-8")
    with pytest.raises(steps.RunnerError, match="verified by the select step"):
        steps.check_selected_inputs(ctx)
    with pytest.raises(steps.RunnerError, match="run the select step"):
        steps.check_selected_inputs(SimpleNamespace(mode="final", completed={}))
    assert "reruns nothing" in config.SPEC["select"].purpose  # X44


# ── T-83 the final estimate ─────────────────────────────────────────────────


def test_the_final_estimate_scales_the_dev_duration(tmp_path):
    runs = tmp_path / "runs"
    folder = runs / "20261111T000000Z_dev" / "tft96"
    folder.mkdir(parents=True)
    budget = config.BUDGETS["dev"]["tft96"]
    (folder / "step.json").write_text(json.dumps({"mode": "dev", "status": "ok",
                                                  "duration_s": 100.0,
                                                  "config": {"budget": budget}}),
                                      encoding="utf-8")
    # 2 targets x 8 fits of 25 epochs each in dev
    pd.DataFrame([{"target": t, "seed": 42, "fit": f"trial{i}", "epoch": e}
                  for t in ("indirect", "direct") for i in range(8) for e in range(25)]
                 ).to_csv(folder / "learning_curves.csv", index=False)
    ratio, upper = scale_dev_to_final("tft96", folder)
    assert ratio == pytest.approx(21 / 8)  # 20 trials + the retrain, against 7 + 1
    assert upper == pytest.approx(2 * 21 * 100 / (2 * 8 * 25))
    text = Runner(runs_dir=runs, echo=False).estimate("tft96", "final")
    assert f"{100 * 21 / 8:.0f} s" in text and f"{100 * upper:.0f} s" in text
    assert runner.fits("xgb", config.BUDGETS["final"]["xgb"]) / runner.fits(
        "xgb", config.BUDGETS["dev"]["xgb"]) == 9
    assert Runner(runs_dir=runs, echo=False).estimate("bilstm", "final").startswith("unknown")


# ── T-84 the final gate ─────────────────────────────────────────────────────


def ok_step(name):
    def inputs(ctx):
        return []

    def run(ctx):
        p = ctx.out / f"{name}.txt"
        p.write_text("x", encoding="utf-8")
        return steps.Result([p])
    return steps.Implementation(inputs, run)


def test_the_gate_is_checked_before_every_final_step(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    seen = []

    def gate(commit):
        seen.append(commit)
        return [] if len(seen) == 1 else ["HEAD is def5678, not abc1234 of the preflight"]
    r = make_runner(tmp_path, gate=gate, impls={"baselines": ok_step("b"), "xgb": ok_step("x")})
    assert r.run("final", ["baselines", "xgb"]) == 1
    state = read(only_run(tmp_path) / "run.json")
    assert state["steps"]["baselines"]["status"] == "ok"
    assert "final gate" in state["invocations"][-1]["results"]["xgb"]
    assert "xgb" not in state["steps"] and seen == ["abc1234", "abc1234"]
    r = make_runner(tmp_path, gate=lambda c: ["never"], impls={"baselines": ok_step("b")})
    assert r.run("smoke", ["baselines"]) == 0  # smoke and dev have no step gate


def test_final_step_gate_sees_a_moved_head(monkeypatch):
    monkeypatch.setattr(checks, "git_state", lambda: {"commit": "def5678", "tags_on_head": [
        "g1-frozen"], "tree_clean": True})
    monkeypatch.setattr(checks, "load_site", lambda: SimpleNamespace(is_private=True))
    assert checks.final_step_gate("def5678") == []
    assert any("nothing may be committed" in e for e in checks.final_step_gate("abc1234"))


def test_final_refuses_slices_and_keeps_the_v6_budgets():
    for kwargs in ({"horizons": [1]}, {"targets": ["indirect"]}, {"seeds": [42]}):
        with pytest.raises(ValueError, match="whole v6 scope"):
            config.resolve("final", **kwargs)
    config.resolve("final", targets=["indirect", "direct"], horizons=[1, 2, 3])  # the scope
    config.resolve("dev", horizons=[1], seeds=[7])  # slices stay allowed in development
    b = config.BUDGETS["final"]
    assert b["xgb"] == {"trials": 60, "seeds": [42, 123, 777]}
    assert (b["bilstm"]["trials"], b["bilstm"]["epochs_trial"], b["bilstm"]["epochs_final"],
            b["bilstm"]["seeds"]) == (60, 25, 60, [42, 123, 777])
    assert b["tft96"]["trials"] == b["tft168"]["trials"] == 20
    assert b["tftbase"]["epochs"] == 150 and b["patchtst"]["max_steps"] == 5000


# ── T-85 the test split ─────────────────────────────────────────────────────


def test_development_runs_read_no_test_row_and_log_none(xgb_run, access_log):
    assert xgb_run.code == 0
    assert not access_log.exists()
    for step in ("baselines", "xgb"):
        assert read(xgb_run.dir / step / "step.json")["values"].get("test_access") == []
    ctx = SimpleNamespace(mode="dev", step="xgb")
    with pytest.raises(steps.RunnerError, match="only in a final run"):
        steps.log_test_access(ctx, "x", [], 1)


def test_the_test_flag_belongs_to_the_step_test(tmp_path):
    r = make_runner(tmp_path, impls={"baselines": ok_step("b"), "test": ok_step("t")})
    with pytest.raises(ValueError, match="flag of the step test"):
        r.run("final", ["baselines"], allow_test=True)
    with pytest.raises(ValueError, match="only in a final run"):
        r.run("final", ["test"])
    with pytest.raises(PermissionError):
        r.run("dev", ["test"], allow_test=True)


def test_final_two_phases_score_test_only_in_the_step_test(tmp_path, monkeypatch, access_log):
    from test_pipeline_runner import synthetic_tables

    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    synthetic_tables(tmp_path, monkeypatch)
    r = make_runner(tmp_path)
    assert r.run("final", ["baselines", "evaluate", "report"]) == 0
    run_dir = only_run(tmp_path)
    log = pd.read_csv(access_log)
    assert len(log) == 1 and "step baselines" in log["command"].iloc[0]
    assert (run_dir / "evaluate" / "val" / "benchmark_summary.csv").exists()
    assert not (run_dir / "evaluate" / "test").exists()  # D-32: validation only
    refs = pd.read_csv(run_dir / "baselines" / "predictions_persistence.csv")
    assert set(refs["split"]) == {"val", "test"}  # forecast, not scored
    # R-G1-04: phase 1 holds no test truth, only the forecast of the test hours
    assert refs.loc[refs["split"] == "test", "y_true"].isna().all()
    assert refs.loc[refs["split"] == "test", "y_pred"].notna().any()
    assert refs.loc[refs["split"] == "val", "y_true"].notna().any()
    r2 = make_runner(tmp_path)
    assert r2.run("final", ["test"], resume=run_dir.name, allow_test=True) == 1  # X51
    assert "acceptance" in read(run_dir / "run.json")["steps"]["test"]["error"]
    (run_dir / steps.ACCEPTANCE_FILE).write_text("stage 10 accepted on validation\n",
                                                 encoding="utf-8")
    assert r2.run("final", ["test"], resume=run_dir.name, allow_test=True) == 0
    bench = pd.read_csv(run_dir / "test" / "test" / "benchmark_summary.csv")
    assert set(bench["model"]) == {"persistence", "climatology", "clearsky_k"}
    log = pd.read_csv(access_log)
    assert len(log) == 2 and "score the test split" in log["command"].iloc[1]


def test_the_step_test_needs_the_report_first(tmp_path, monkeypatch):
    from test_pipeline_runner import synthetic_tables

    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    synthetic_tables(tmp_path, monkeypatch)
    r = make_runner(tmp_path)
    assert r.run("final", ["baselines", "evaluate"]) == 0
    run_dir = only_run(tmp_path)
    assert r.run("final", ["test"], resume=run_dir.name, allow_test=True) == 1
    assert "stage-10 acceptance" in read(run_dir / "run.json")["steps"]["test"]["error"]


# ── T-86 seed spread and warnings ───────────────────────────────────────────


def test_the_seed_spread_comes_from_the_benchmark_in_kw():
    bench = pd.DataFrame({"model": ["xgb_sota"] * 3 + ["persistence"], "target": "indirect",
                          "horizon": 1, "seed": [42, 123, 777, -1],
                          "nRMSE_kW_pct": [6.0, 6.2, 6.5, 9.0],
                          "nRMSE_pct": [12.0, 12.4, 13.0, 18.0]})
    s = report.seed_spread(bench)
    assert len(s) == 1 and s["n_seeds"].iloc[0] == 3
    assert s["std_kw_pp"].iloc[0] == pytest.approx(np.std([6.0, 6.2, 6.5], ddof=1))
    assert s["std_target_unit_pp"].iloc[0] == pytest.approx(np.std([12, 12.4, 13], ddof=1))
    assert bool(s["above_rule"].iloc[0])  # 0.25 pp > 0.15 pp
    ctx = SimpleNamespace(completed={"xgb": {"notes": {"seed_spread_pp": {
        "indirect_D1": float(s["std_kw_pp"].iloc[0])}}}})
    assert report.spread_check(ctx, s) == []
    ctx.completed["xgb"]["notes"]["seed_spread_pp"]["indirect_D1"] += 0.01
    assert report.spread_check(ctx, s)


def test_the_warnings_are_recomputed_from_the_records():
    edge = {k: v["low"] for k, v in X.SEARCH_SPACE.items()}
    w = steps.xgb_warning_list("indirect_D1_seed42", 0, 1, False, 3.0, 1.0, edge)
    assert {x["kind"] for x in w} == {"first_round", "cap", "nan", "gap", "edge_space"}
    rec = {"params": {"hidden": 256, "n_layers": 2, "dropout": 0.2, "lr": 1e-3,
                      "batch_size": 64},
           "study": {"best_trial": 3, "n_trials": 20, "best_value_kw": 6.8},
           "final": {"nan_loss": True, "best_epoch": 1, "epochs_run": 1, "stopped_early": False,
                     "best_val_loss": 0.3, "train_loss_at_best": 0.1, "value_kw": 7.0},
           "reproduce_diff_pp": 0.1}
    w = neural.warning_list("bilstm", {"indirect": {"42": rec}}, config.BUDGETS["dev"]["bilstm"])
    assert {x["kind"] for x in w} == {"nan", "first_epoch", "last_epoch", "gap", "edge_space",
                                      "reproduce"}
    texts = [x["text"] for x in w]
    ctx = SimpleNamespace(step="report", completed={
        "bilstm": {"warnings": texts + ["bilstm: something else"]},
        "report": {"warnings": ["report: an earlier report"]}})  # a rerun skips its own (R-G1-02)
    m = report.match_warnings(ctx, [{**x, "step": "bilstm"} for x in w],
                              pd.DataFrame(columns=["above_rule"]))
    assert (m["status"] == "matched").sum() == len(w)
    assert m.loc[m["status"] == "step log only", "kind"].tolist() == ["other"]


def test_the_real_xgb_warnings_match_the_records(xgb_run):
    w = pd.read_csv(xgb_run.dir / "report" / "report_warnings.csv")
    xgb = w[w["step"] == "xgb"]
    assert not (xgb["status"] == "recomputed only").any()
    recomputable = {"first_round", "cap", "nan", "gap", "edge_space"}
    assert not ((xgb["status"] == "step log only") & xgb["kind"].isin(recomputable)).any()


# ── T-87 SHAP ───────────────────────────────────────────────────────────────


def test_shap_of_the_sliced_booster_adds_up_and_the_unsliced_does_not():
    rng = np.random.default_rng(1)
    Xd = pd.DataFrame(rng.normal(size=(1500, 4)), columns=["a", "b", "c", "d"])
    Xd.loc[rng.random(1500) < 0.1, "b"] = np.nan
    y = 2 * Xd["a"] + np.nan_to_num(Xd["b"]) ** 2 + rng.normal(0, 0.1, 1500)
    booster = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1,
                               tree_method="hist").fit(Xd, y).get_booster()
    sv = explain.shap_values(booster, Xd.iloc[:400], 70)
    assert sv["additivity"] <= explain.SHAP_TOL
    assert sv["contribs_diff"] <= explain.CONTRIBS_TOL
    assert sv["unsliced_additivity"] > 100 * explain.SHAP_TOL  # X46
    with pytest.raises(steps.RunnerError, match="feature names"):
        explain.shap_values(booster, Xd.iloc[:10][["b", "a", "c", "d"]], 70)


def test_shap_on_the_synthetic_run(xgb_run):
    c = pd.read_csv(xgb_run.dir / "explain" / "xgb" / "xgb_shap_checks.csv")
    scale = c["unit"].map(explain.SCALE)
    assert len(c) == 2 and (c["additivity_max"] <= explain.SHAP_TOL * scale).all()
    assert (c["additivity_relative"] <= explain.SHAP_TOL).all()
    assert (c["contribs_max_diff"] <= explain.CONTRIBS_TOL * scale).all()
    assert (c["kw_additivity_max"] <= explain.KW_TOL).all()
    ind = c[c["target"] == "indirect"].iloc[0]
    assert bool(ind["mos_applied"]) and ind["unit"] == "k"  # the L-09 kW view (X47)
    assert (c["rounds_in_booster"] >= c["n_rounds"]).all()
    rows = X.score_rows(data.model_table(1), config.evaluation_windows("smoke")[0])
    assert (c["n_rows"] == int(rows.sum())).all()  # the scoring hours, never fit rows
    f = pd.read_csv(xgb_run.dir / "explain" / "xgb" / "xgb_shap_features.csv")
    best = read(xgb_run.dir / "xgb" / "best_params.json")
    for b in best.values():
        got = f[(f["target"] == b["target"]) & (f["seed"] == b["seed"])]
        assert set(got["feature"]) == set(b["feature_order"])  # M26


def test_group_attribution_and_rank_stability():
    feats = ["ghi_clearsky", "sun_elevation", "fcst_gfs_kt"]
    phi = np.array([[1.0, -1.0, 2.0], [2.0, -1.0, 0.0]])
    g = explain.group_table(feats, phi).set_index("group")
    assert g.loc["clearsky", "n_members"] == 2
    assert g.loc["clearsky", "mean_abs_group_kw"] == pytest.approx((0 + 1) / 2)  # |sum|
    assert g.loc["clearsky", "sum_member_mean_abs_kw"] == pytest.approx(1.5 + 1.0)  # v6
    assert g["share_pct"].sum() == pytest.approx(100)
    with pytest.raises(KeyError):
        explain.feature_group("not_a_feature")
    a = pd.DataFrame({"feature": ["x", "y", "z", "w"], "mean_abs_kw": [4, 3, 2, 1.0],
                      "target": "direct", "horizon": 1, "seed": 42})
    b = pd.DataFrame({"feature": ["x", "y", "z", "v"], "mean_abs_kw": [4, 3, 2, 9.0],
                      "target": "direct", "horizon": 2, "seed": 42})
    r = explain.rank_stability(pd.concat([a, b]))
    assert r["n_common"].tolist() == [3] and r["spearman"].iloc[0] == pytest.approx(1.0)


def test_the_kw_view_of_the_l09_map():
    sv = {"values": np.array([[0.1, 0.2], [0.3, -0.1]]), "base": np.array([0.4, 0.4]),
          "forecast": np.array([0.7, 0.6])}
    theo = np.array([1000.0, 3000.0])
    kv = explain.kw_view("indirect", sv, theo, {"a_kw": 20.0, "b": 0.9})
    kw = 20 + 0.9 * theo * sv["forecast"]
    assert np.allclose(kv["base"] + kv["values"].sum(axis=1), kw)
    assert kv["unclipped"].tolist() == [True, True]
    kv = explain.kw_view("indirect", {**sv, "forecast": np.array([0.7, 1.5])}, theo,
                         {"a_kw": 20.0, "b": 0.9})
    assert kv["unclipped"].tolist() == [True, False]  # 4,070 kW is clipped at 3,449


# ── T-88 attention and variable selection ───────────────────────────────────


def test_attention_summaries_of_constructed_arrays():
    n, steps_, length = 4, 73, 96
    keep = np.ones((n, steps_), dtype=bool)
    horizon = np.repeat(np.array([1] * 24 + [2] * 24 + [3] * 25)[None, :], n, axis=0)
    uniform = np.full((n, steps_, length), 1 / length)
    _, summary = explain.attention_tables(uniform, keep, horizon, {"step": "tft96"})
    allb = next(s for s in summary if s["block"] == "all")
    assert allb["same_hour_ratio"] == pytest.approx(1.0)
    assert allb["earliest_lag_h"] == -96  # a 96-position encoder has nothing before -96 h
    same = explain.same_hour_mask(steps_, length).astype(float)
    peaked = np.repeat((same / same.sum(axis=1, keepdims=True))[None], n, axis=0)
    lag, summary = explain.attention_tables(peaked, keep, horizon, {"step": "tft96"})
    allb = next(s for s in summary if s["block"] == "all")
    assert allb["same_hour_share"] == pytest.approx(1.0)
    assert allb["same_hour_ratio"] == pytest.approx(1 / allb["same_hour_reference"])
    one = np.zeros((n, steps_, length))
    one[..., 72] = 1.0  # lag -24 h for every step
    _, summary = explain.attention_tables(one, keep, horizon, {"step": "tft96"})
    assert next(s for s in summary if s["block"] == "all")["argmax_lag_h"] == -24
    assert set(lag["block"]) == {"all", "1", "2", "3"}


def test_variable_importance_adds_a_series_and_its_flag():
    rec = {"encoder_importance": {"irradiance": 0.1, "irradiance_filled": 0.05, "aoi": 0.85},
           "decoder_importance": {"aoi": 1.0}}
    t = explain.variable_table(rec, {"step": "tft96"})
    enc = t[t["part"] == "encoder"].set_index("channel")
    assert enc.loc["irradiance_filled", "variable"] == "irradiance"
    assert enc.loc["irradiance", "variable_weight"] == pytest.approx(0.15)
    assert enc["part_total"].iloc[0] == pytest.approx(1.0)


# ── T-89 hyper-parameter importance ─────────────────────────────────────────


def _study(n_trials):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    s.optimize(lambda t: (t.suggest_float("x", 0, 1) - 0.3) ** 2 + 0 * t.suggest_float("y", 0, 1),
               n_trials=n_trials)
    return s


def test_seeded_importance_and_the_startup_flag():
    rows = pd.DataFrame(explain.importance_rows(_study(30), {"step": "xgb"}))
    imp = rows.set_index("param")["importance_mean"]
    assert imp["x"] > 0.9 > imp["y"] and not rows["all_in_startup"].any()
    assert (rows["importance_std"] >= 0).all() and rows["n_complete"].iloc[0] == 30
    again = pd.DataFrame(explain.importance_rows(_study(30), {"step": "xgb"}))
    assert np.allclose(again["importance_mean"], rows["importance_mean"])  # seeded
    assert pd.DataFrame(explain.importance_rows(_study(7), {}))["all_in_startup"].all()
    assert "fewer than 2" in explain.importance_rows(_study(1), {})[0]["note"]


def test_the_edge_check_uses_the_effective_continuous_size():
    space = J.SEARCH_SPACES["tft"]
    params = {"hidden_size": 32, "attention_head_size": 2, "hidden_continuous_size": 50,
              "dropout": 0.2, "learning_rate": 1e-3, "lstm_layers": 2}
    assert "hidden_continuous_size" not in J.edge_params(params, space)  # sampled 50 of 8-64
    eff, eff_space = J.effective(params, space)
    assert eff["hidden_continuous_size"] == 32 and eff_space["hidden_continuous_size"]["high"] == 32
    assert "hidden_continuous_size" in J.edge_params(eff, eff_space)  # X45: used 32, the cap


# ── T-90 the report ─────────────────────────────────────────────────────────


def test_the_report_reads_its_numbers_from_hashed_files(xgb_run):
    out = xgb_run.dir / "report"
    rec = read(out / "report.json")
    for rel, digest in rec["sources"].items():
        assert sha256(xgb_run.dir / rel) == digest
    mine = pd.read_csv(out / "report_benchmark_val.csv")
    theirs = pd.read_csv(xgb_run.dir / "evaluate" / "val" / "benchmark_summary.csv")
    assert np.allclose(mine["nRMSE_kW_pct"], theirs["nRMSE_kW_pct"])
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "## Sources" in md and "evaluate/val/benchmark_summary.csv" in md
    path = xgb_run.dir / "evaluate" / "val" / "benchmark_summary.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert xgb_run.runner.run("smoke", ["report"], resume=xgb_run.dir.name, force=True,
                              horizons=[1]) == 1


# ── T-91 archive and promote ────────────────────────────────────────────────


def fake_run(root, mode="final"):
    run = root / "runs" / f"20261112T000000Z_{mode}"
    (run / ".xgb.tmp").mkdir(parents=True)
    (run / ".xgb.tmp" / "half.csv").write_text("x", encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"run_id": run.name, "mode": mode, "steps": {}}),
                                  encoding="utf-8")
    (run / "a.csv").write_text("1,2\n", encoding="utf-8")
    return run


def test_archive_packs_hashes_and_verifies(tmp_path):
    run = fake_run(tmp_path)
    path = archive.archive(run, tmp_path / "arch")
    assert archive.verify(path) == []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    assert f"{run.name}/a.csv" in names and not any(".tmp" in n for n in names)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(bad, "w") as dst:
        for n in src.namelist():
            dst.writestr(n, b"changed" if n.endswith("a.csv") else src.read(n))
    assert any("differs from the manifest" in p for p in archive.verify(bad))
    (path.with_name(path.name + ".sha256")).write_text("0" * 64 + "  x\n", encoding="utf-8")
    assert any(".sha256" in p for p in archive.verify(path))
    (run / "site.private.yaml").write_text("x", encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="private"):
        archive.archive(run, tmp_path / "arch")
    with pytest.raises(archive.ArchiveError, match="inside the run"):
        archive.archive(run, run / "zips")


def complete_final_run(root):
    """A final run folder whose every required step is complete (constructed records)."""
    run = root / "runs" / "20261112T000000Z_final"
    run.mkdir(parents=True)
    states, scored = {}, {}  # `scored`: the prediction files, as the inputs of the step test
    for step in promote.REQUIRED:
        folder = run / step
        folder.mkdir()
        outputs = {}
        if step in config.MODEL_STEPS:
            model = {"baselines": "persistence", "xgb": "xgb_sota"}.get(step,
                                                                      neural_model(step))
            name = (f"predictions_{model}.csv" if step == "baselines"
                    else f"predictions_{model}_seed42.csv")
            # phase 1 of a final run: the test rows have no y_true (R-G1-04, R-G1-16)
            (folder / name).write_text(
                "timestamp,y_true,y_pred,target,horizon,model,seed,split\n"
                "2024-09-30 23:00:00+00:00,5.0,6.0,direct,1,m,42,val\n"
                "2024-10-01 00:00:00+00:00,,7.0,direct,1,m,42,test\n"
                "2024-10-01 01:00:00+00:00,,0.4,indirect,1,m,42,test\n", encoding="utf-8")
            outputs[name] = sha256(folder / name)
            scored[f"outputs/runs/{run.name}/{step}/{name}"] = outputs[name]
        if step == "xgb":
            (folder / "models").mkdir()
            (folder / "models" / "xgb_sota_indirect_D1_seed42.ubj").write_bytes(b"b")
            outputs["models/xgb_sota_indirect_D1_seed42.ubj"] = sha256(
                folder / "models" / "xgb_sota_indirect_D1_seed42.ubj")
        for name in promote.ANALYSIS_FILES.get(step, ()):  # S-38, S-41 (D-36, X73)
            (folder / name).write_text("a\n1\n", encoding="utf-8")
            outputs[name] = sha256(folder / name)
        if step == "test":
            pd.DataFrame({"timestamp": ["2024-10-01 00:00:00+00:00", "2024-10-01 01:00:00+00:00"],
                          "indirect": [0.5, 0.6], "direct": [1000.0, 1200.0]}).to_csv(
                folder / steps.TEST_TRUTH_FILE, index=False)
            outputs[steps.TEST_TRUTH_FILE] = sha256(folder / steps.TEST_TRUTH_FILE)
            (folder / "analyses").mkdir()
            for name in promote.ANALYSIS_TEST:
                (folder / name).write_text("a\n1\n", encoding="utf-8")
            (folder / "test").mkdir()
            pd.DataFrame({"model": ["xgb_sota", "persistence"], "seed": [42, -1],
                          "target": "direct", "horizon": 1, "nRMSE_kW_pct": [6.0, 9.0]}
                         ).to_csv(folder / "test" / "benchmark_summary.csv", index=False)
            (folder / "shap_test").mkdir()
            pd.DataFrame({"feature": ["a", "b"], "mean_abs_native": [2.0, 1.0],
                          "mean_abs_kw": [2.0, 1.0], "seed": 42, "target": "direct",
                          "horizon": 1}).to_csv(folder / "shap_test" /
                                                "xgb_shap_features_test.csv", index=False)
        record = {"step": step, "status": "ok", "folder": step, "commit": "abc1234",
                  "fingerprint": "f" * 64, "outputs": outputs, "duration_s": 1.0}
        if step == "test":
            record["inputs"] = dict(scored)
        (folder / "step.json").write_text(json.dumps(record), encoding="utf-8")
        states[step] = {"status": "ok", "folder": step}
    (run / "run.json").write_text(json.dumps({"run_id": run.name, "mode": "final",
                                              "steps": states}), encoding="utf-8")
    (run / "config.yaml").write_text("targets: [indirect, direct]\nhorizons: [1, 2, 3]\n"
                                     "seeds: null\n", encoding="utf-8")
    return run


def neural_model(step):
    return J.STEP_MODEL.get(step, step)


def test_promote_copies_a_complete_final_run_only(tmp_path):
    run = complete_final_run(tmp_path)
    out = tmp_path / "outputs"
    manifest = promote.promote(run, out, check_tag=False)
    files = read(manifest)["files"]
    assert "06_xgb_sota/predictions_xgb_sota_seed42.csv" in files
    assert "06_xgb_sota/metrics_xgb_sota_seed42.csv" in files
    assert "06_xgb_sota/models/xgb_sota_indirect_D1_seed42.ubj" in files
    # R-G1-16: the promoted file of §6.10 has the y_true of its test rows, from the step test
    promoted = pd.read_csv(out / "06_xgb_sota" / "predictions_xgb_sota_seed42.csv")
    assert promoted["y_true"].tolist() == [5.0, 1000.0, 0.6]
    assert promoted["y_pred"].tolist() == [6.0, 7.0, 0.4]
    assert "step test" in files["06_xgb_sota/predictions_xgb_sota_seed42.csv"]["source"]
    for name in ("cv_summary.csv", "ablation_summary.csv", "cv_vs_test.csv",
                 "ablation_test_summary.csv"):  # X73: the tables of S-38 and S-41
        assert f"06_xgb_sota/analysis/{name}" in files, name
    shap = pd.read_csv(out / "06_xgb_sota" / "shap" / "shap_direct_D1.csv")
    assert list(shap.columns[:2]) == ["feature", "shap_importance"]
    metrics = pd.read_csv(out / "06_persistence" / "metrics_persistence.csv")
    assert metrics["nRMSE_kW_pct"].tolist() == [9.0]
    with pytest.raises(promote.PromoteError, match="exist"):
        promote.promote(run, out, check_tag=False)
    promote.promote(run, out, replace=True, check_tag=False)
    assert list((out / "06_xgb_sota").glob("*.superseded-*"))
    (run / "config.yaml").write_text("targets: [indirect]\nhorizons: [1, 2, 3]\nseeds: null\n",
                                     encoding="utf-8")
    with pytest.raises(promote.PromoteError, match="v6 scope"):
        promote.plan(run, out, check_tag=False)
    smoke = fake_run(tmp_path / "s", mode="smoke")
    with pytest.raises(promote.PromoteError, match="only a final run"):
        promote.plan(smoke, out, check_tag=False)
    run2 = complete_final_run(tmp_path / "x")
    state = read(run2 / "run.json")
    del state["steps"]["test"]
    (run2 / "run.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(promote.PromoteError, match="not complete"):
        promote.plan(run2, out, check_tag=False)
    # R-G1-02: the report of the finished run must come after the step test
    for first, second, ok in (("test", "report", True), ("report", "test", False)):
        run4 = complete_final_run(tmp_path / f"order_{first}")
        for n, step in enumerate((first, second)):
            record = read(run4 / step / "step.json")
            record["finished_utc"] = f"2026-11-14T10:0{n}:00+00:00"
            (run4 / step / "step.json").write_text(json.dumps(record), encoding="utf-8")
        if ok:
            promote.plan(run4, out, check_tag=False)
        else:
            with pytest.raises(promote.PromoteError, match="report was written before"):
                promote.plan(run4, out, check_tag=False)
    # R-G1-19: what is promoted is what the step test scored, and no step ran after it
    run5 = complete_final_run(tmp_path / "rerun")  # xgb rerun after the test: other hash
    record = read(run5 / "test" / "step.json")
    record["inputs"][next(iter(record["inputs"]))] = "0" * 64
    (run5 / "test" / "step.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(promote.PromoteError, match="differ from those the step test scored") as e:
        promote.plan(run5, out, check_tag=False)
    assert "predictions_" in str(e.value)  # R-G1-24: the message names the file whose hash differs
    run6 = complete_final_run(tmp_path / "later")
    for step, stamp in (("test", "2026-11-14T10:00:00+00:00"),
                        ("xgb", "2026-11-14T10:05:00+00:00")):
        record = read(run6 / step / "step.json")
        record["finished_utc"] = stamp
        (run6 / step / "step.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(promote.PromoteError, match="finished after the step test"):
        promote.plan(run6, out, check_tag=False)
    for step in ("cv", "ablation"):  # D-36: a final run without them is not the v6 scope
        run3 = complete_final_run(tmp_path / f"no_{step}")
        state = read(run3 / "run.json")
        del state["steps"][step]
        (run3 / "run.json").write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(promote.PromoteError, match=f"not complete.*{step}"):
            promote.plan(run3, out, check_tag=False)
