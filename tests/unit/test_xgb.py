"""The `xgb` step, its model code and the checks of §14, on synthetic tables (docs/stage9_plan.md
§13-§14).

T-56  inputs from the allow-list and monotone constraints by exact name (M7, M12a)
T-57  no imputation, NaN passed to XGBoost and counted; the fit, early-stopping and scoring
      rows (M8, M11)
T-58  raw y_pred in the file; the objective is the evaluation's nRMSE in kW; the k -> kW copies
      agree (M9, X12)
T-59  the full parameter set, best iteration and rounds recorded; a reloaded booster reproduces
      the predictions; the seed drives the sampler and the model; the cap warning (M10, M12)
T-60  model_table filters inside the reader (X10); the stage-7 provenance row of check (X11);
      a pending decision does not block, and the xgb step refuses without it
T-61  A-14: validation labels do not change a fixed-round model or the MOS coefficients
T-62  X13: decide_k.XGB_PARAMS is unchanged after the D-15 check
T-63  the checks of §14: the registered rules, the outputs, never a predictions_* file nor a
      write into a run folder; L-09 and R-14 on a synthetic dev run
T-64  the adopted L-09 correction in the step (§15): one implementation with the check (X14),
      the corrected kW written as k and inverted exactly (X15), the uncorrected file and the
      base forecast of the check (X16), the decision file required in final mode only
"""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import yaml

from solarpv.config.constants import AC_CAPACITY_KW, TEST_START, TIMEZONE_LOCAL, VAL_START
from solarpv.config.provenance import sha256
from solarpv.config.site import load_site
from solarpv.evaluation import evaluate as E
from solarpv.evaluation.daytime import sun_elevation_mid_hour
from solarpv.evaluation.truth import load_truth
from solarpv.features import target as target_mod
from solarpv.features.contract import features_for_horizon
from solarpv.features.decide_k import XGB_PARAMS
from solarpv.models import xgb_checks as C
from solarpv.models import xgb_sota as X
from solarpv.nwp import check as nwp_check
from solarpv.pipeline import checks, config, data, runner, steps
from solarpv.pipeline.config import evaluation_windows
from solarpv.pipeline.runner import Runner
from solarpv.selection import common as selection_common

MEAN_K = 0.75
SEED = 42
INDIRECT = ["ghi_clearsky", "fcst_gfs_kt", "fcst_gfs_tcc", "hour_sin", "irradiance_lag_3d",
            "irradiance_clear_index_prevday", "fcst_gfs_ghi"]
DIRECT = INDIRECT + ["poa_global_clearsky", "sun_elevation", "aoi"]


# ── synthetic tables in the stage-6 and stage-6b layout ─────────────────────


def build_tables(seed: int = 0):
    idx = pd.date_range("2023-01-01", "2024-10-15", freq="h", tz="UTC", inclusive="left",
                        name="timestamp_utc")
    n = len(idx)
    rng = np.random.default_rng(seed)
    elev = sun_elevation_mid_hour(idx, load_site())
    day = elev > 5
    sin_el = np.clip(np.sin(np.radians(elev)), 0, None)
    ghi_cs, poa_cs = 1000 * sin_el ** 1.15, 1050 * sin_el ** 0.9
    theo = poa_cs * AC_CAPACITY_KW / 1000
    days = idx.tz_convert(TIMEZONE_LOCAL).tz_localize(None).normalize()
    codes, uniq = pd.factorize(days)
    cloud = np.clip(rng.beta(0.6, 1.5, len(uniq))[codes] + rng.normal(0, 0.05, n), 0, 1)
    k = np.where(theo > 50, np.clip(0.9 - 0.6 * cloud + rng.normal(0, 0.03, n), 0, None), 0.0)
    ac = k * theo
    gap = day & (rng.random(n) < 0.01)
    ac[gap], k[gap] = np.nan, np.nan
    split = np.where(idx < pd.Timestamp(VAL_START, tz="UTC"), "train",
                     np.where(idx < pd.Timestamp(TEST_START, tz="UTC"), "val", "test"))
    feats = {f: rng.normal(size=n) for f in features_for_horizon(1)}
    feats.update({"ghi_clearsky": ghi_cs, "poa_global_clearsky": poa_cs, "sun_elevation": elev,
                  "aoi": 90 - elev, "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
                  "irradiance_lag_3d": pd.Series(poa_cs * (1 - cloud)).shift(72).to_numpy()})
    base = pd.DataFrame({"t0_utc": idx, "site_total_AC_kW": ac, "theo_kw": theo,
                         "target_k": k, "k_valid": (theo > 50) & ~gap, "k_suspect": False,
                         "split": split, "daytime": day,
                         "op_limited_flag": rng.random(n) < 0.03, "hour_source": "minute",
                         "warmup": idx < idx[0] + pd.Timedelta(days=14), **feats}, index=idx)
    prev = rng.normal(0.6, 0.2, n)
    prev[rng.random(n) < 0.02] = np.nan
    gfs = pd.DataFrame({"t0_utc": idx,
                        "fcst_gfs_tcc": np.clip(cloud + rng.normal(0, 0.15, n), 0, 1),
                        "fcst_gfs_kt": np.where(day, 1 - cloud + rng.normal(0, 0.1, n), np.nan),
                        "fcst_gfs_ghi": ghi_cs * (1 - cloud) + rng.normal(0, 20, n),
                        "irradiance_clear_index_prevday": prev}, index=idx)
    return base, gfs


@pytest.fixture(scope="module")
def tables():
    return build_tables()


@pytest.fixture
def wired(tmp_path, monkeypatch, tables):
    """The synthetic tables, inputs and decision wired into the runner instead of the real
    files."""
    base, gfs = tables
    (tmp_path / "06_features").mkdir()
    (tmp_path / "06b_gfs").mkdir()
    for h in (1, 2, 3):
        base.to_parquet(tmp_path / "06_features" / f"features_1H_D{h}.parquet")
        gfs.to_parquet(tmp_path / "06b_gfs" / f"gfs_features_D{h}.parquet")
    monkeypatch.setattr(data, "FEATURES_DIR", tmp_path / "06_features")
    monkeypatch.setattr(data, "GFS_DIR", tmp_path / "06b_gfs")
    truth_file = tmp_path / "06_features" / "features_1H_D1.parquet"
    monkeypatch.setattr(steps, "FEATURES_FILE", truth_file)
    monkeypatch.setattr(E, "FEATURES_FILE", truth_file)
    constants = tmp_path / "derived_constants.yaml"
    constants.write_text(f"mean_daytime_k: {MEAN_K}\n", encoding="utf-8")
    monkeypatch.setattr(steps, "DERIVED_CONSTANTS_FILE", constants)
    monkeypatch.setattr(steps, "mean_daytime_k", lambda: MEAN_K)
    inputs = {f"{t}_D{h}": {"source": "test", "reason": "synthetic", "n_features": len(f),
                            "features": f}
              for h in (1, 2, 3) for t, f in (("indirect", INDIRECT), ("direct", DIRECT))}
    inputs_file = tmp_path / "model_inputs.json"
    inputs_file.write_text(json.dumps(inputs), encoding="utf-8")
    monkeypatch.setattr(X, "MODEL_INPUTS_FILE", inputs_file)
    decision = tmp_path / "monotone_decision.yaml"
    decision.write_text(yaml.safe_dump({"features": list(X.D15_FEATURES),
                                        "settings": {"indirect": "A", "direct": "A"}}),
                        encoding="utf-8")
    monkeypatch.setattr(steps, "MONOTONE_DECISION_FILE", decision)
    mos = tmp_path / "wired_mos_decision.yaml"  # the L-09 decision of R9-07
    mos.write_text(yaml.safe_dump({"decision": "L-09",
                                   "adopted": {"indirect": True, "direct": False}}),
                   encoding="utf-8")
    monkeypatch.setattr(steps, "MOS_DECISION_FILE", mos)
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    return SimpleNamespace(tmp=tmp_path, base=base, gfs=gfs, decision=decision, mos=mos,
                           inputs={k: v["features"] for k, v in inputs.items()},
                           truth_file=truth_file)


def fake_report(mode):
    return {"mode": mode, "data": [], "site_private": True,
            "environment": {"python": "3.11", "platform": "test", "packages": {},
                            "cpu_count": 8, "xgb_threads": 8, "memory": None,
                            "disk_free_gb": 1.0, "gpu": None},
            "git": {"commit": "abc1234", "tags_on_head": [], "tree_clean": True},
            "errors": [], "warnings": []}


def make_runner(tmp_path):
    return Runner(runs_dir=tmp_path / "runs", preflight=fake_report, echo=False)


@pytest.fixture
def smoke_run(wired, monkeypatch):
    """One smoke run of baselines,xgb,evaluate on the synthetic tables, D+1 only."""
    calls = []
    real = pd.read_parquet

    def spy(path, *args, **kwargs):
        out = real(path, *args, **kwargs)
        calls.append({"path": Path(path).name, "filters": kwargs.get("filters"),
                      "max": out.index.max() if len(out) else None})
        return out
    monkeypatch.setattr(data.pd, "read_parquet", spy)
    r = make_runner(wired.tmp)
    code = r.run("smoke", ["baselines", "xgb", "evaluate"], horizons=[1])
    (run_dir,) = (wired.tmp / "runs").iterdir()
    return SimpleNamespace(code=code, dir=run_dir, calls=calls, **vars(wired))


# ── T-56 inputs and constraints ─────────────────────────────────────────────


def test_constraints_by_exact_name():
    feats = ["fcst_gfs_ghi", "irradiance_lag_3d", "irradiance_clear_index_prevday",
             "ghi_clearsky", "poa_global_clearsky_lag", "sun_elevation", "poa_global_clearsky"]
    assert X.constraint_map(feats, "A") == {"ghi_clearsky": 1, "sun_elevation": 1,
                                            "poa_global_clearsky": 1}
    assert X.constraint_map(feats, "B") == {}
    with pytest.raises(ValueError):
        X.constraint_map(feats, "C")


def test_input_features_follow_the_allow_list(tables):
    base, gfs = tables
    table = base.join(gfs.drop(columns="t0_utc"))
    inputs = {"indirect_D1": INDIRECT, "indirect_D3": INDIRECT + ["irradiance_lag_1d"],
              "direct_D1": DIRECT + ["fcst_gfs_ghi"], "direct_D2": ["irradiance"]}
    assert X.input_features(table, "indirect", 1, inputs) == INDIRECT  # order kept
    for target, h in (("indirect", 3), ("direct", 1), ("direct", 2)):  # lag, duplicate, leak
        with pytest.raises(ValueError):
            X.input_features(table, target, h, inputs)
    with pytest.raises(KeyError):
        X.input_features(table, "indirect", 2, inputs)


def test_the_constraint_reaches_xgboost(tables):
    base, gfs = tables
    table = base.join(gfs.drop(columns="t0_utc"))
    rows = X.training_rows(table, "direct")
    feats = ["ghi_clearsky", "fcst_gfs_tcc", "hour_sin"]
    booster = X.fit_fixed(table.loc[rows, feats], table.loc[rows, "site_total_AC_kW"],
                          {"n_estimators": 60, "max_depth": 4, "learning_rate": 0.1}, SEED,
                          X.constraint_map(feats, "A"), 60)
    assert X.booster_constraints(booster) == "(1,0,0)"
    probe = pd.DataFrame({"ghi_clearsky": np.linspace(0, 1000, 50), "fcst_gfs_tcc": 0.4,
                          "hour_sin": 0.2})
    assert (np.diff(X.predict(booster, probe, 60)) >= 0).all()


def test_the_decision_file_is_required_and_checked(tmp_path):
    with pytest.raises(FileNotFoundError, match="xgb_checks d15"):
        X.load_monotone_decision(tmp_path / "none.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"features": ["ghi"], "settings": {"indirect": "A",
                                                                      "direct": "A"}}))
    with pytest.raises(ValueError, match="D-15 list"):
        X.load_monotone_decision(bad)


# ── T-57 no imputation; rows ────────────────────────────────────────────────


def _problem(tables, target="indirect", smoke=False, feats=None):
    base, gfs = tables
    table = base.join(gfs.drop(columns="t0_utc"))
    (window,) = evaluation_windows("smoke")
    feats = feats or (INDIRECT if target == "indirect" else DIRECT)
    return table, X.build_problem(table, target, 1, feats, {}, window, MEAN_K, smoke=smoke)


def test_rows_of_fit_early_stopping_and_scoring(tables):
    table, p = _problem(tables, "indirect")
    fit = table.loc[p.X_fit.index]
    assert (fit["split"] == "train").all() and not fit["warmup"].any()
    assert not fit["op_limited_flag"].any() and fit["k_valid"].all()
    assert (fit["theo_kw"] > 50).all()
    stop = table.loc[p.X_stop.index]
    (window,) = evaluation_windows("smoke")
    assert (stop.index >= window.start_ts).all() and (stop.index < window.end_ts).all()
    assert stop["k_valid"].all() and not stop["op_limited_flag"].any()
    score = table.loc[p.X_score.index]
    assert score["daytime"].all() and score["site_total_AC_kW"].notna().all()
    assert score["op_limited_flag"].any()  # §14.1: op-limited hours stay in the objective
    _, d = _problem(tables, "direct")
    direct_fit = table.loc[d.X_fit.index]
    assert (~direct_fit["daytime"]).any()  # nights are fit rows of the direct route
    assert direct_fit["site_total_AC_kW"].notna().all()
    _, s = _problem(tables, "direct", smoke=True)
    local = X.local_day(s.X_fit.index)
    assert (local.max() - local.min()).days == X.SMOKE_TRAIN_DAYS - 1


def test_no_imputation_nan_is_passed_and_counted(tables):
    table, p = _problem(tables, "direct")
    expected = int(table.loc[p.X_fit.index, "fcst_gfs_kt"].isna().sum())
    assert expected > 0 and p.nan["fit"]["fcst_gfs_kt"] == expected  # nights: GFS kt is NaN
    assert int(p.X_fit.isna().sum().sum()) == sum(p.nan["fit"].values())  # nothing filled
    fit = X.fit_early_stopping(p, {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.1},
                               SEED)
    with_nan = X.predict(fit.booster, p.X_fit, fit.n_rounds)
    zero = X.predict(fit.booster, p.X_fit.fillna(0), fit.n_rounds)
    assert np.isfinite(with_nan).all() and not np.allclose(with_nan, zero)


# ── T-58 raw output, one conversion ─────────────────────────────────────────


def test_the_k_to_kw_copies_agree():
    rng = np.random.default_rng(3)
    k = np.concatenate([rng.normal(0.5, 0.6, 500), [-0.2, 0.0, 3.0]])
    theo = np.concatenate([rng.uniform(0, 4000, 500), [0.0, 100.0, 3000.0]])
    ref = E.to_kw("indirect", k, theo)
    assert ref.min() >= 0 and ref.max() <= AC_CAPACITY_KW
    feature_copy = target_mod.to_kw(pd.Series(k), pd.Series(theo)).to_numpy()
    problem = SimpleNamespace(target="indirect", theo=pd.Series(theo))
    selection_copy = selection_common.to_kw(problem, k, pd.Series(True, index=range(len(k))))
    np.testing.assert_array_equal(ref, feature_copy)
    np.testing.assert_array_equal(ref, selection_copy)
    np.testing.assert_array_equal(E.to_kw("direct", k * 1000, theo), k * 1000)  # never clipped


def test_nwp_check_score_uses_the_same_conversion(tables):
    base, gfs = tables
    table = base.join(gfs.drop(columns="t0_utc"))
    table = table[table["split"].isin(["train", "val"])]
    feats = ["ghi_clearsky", "fcst_gfs_tcc"]
    value = nwp_check.fit_score(table, feats, "indirect", SEED)
    fit = X.training_rows(table, "indirect")
    model = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": SEED})
    model.fit(table.loc[fit, feats], table.loc[fit, "target_k"])
    val = C.val_score_rows(table)
    mine = X.nrmse_kw("indirect", model.predict(table.loc[val, feats]),
                      table.loc[val, "theo_kw"], table.loc[val, "site_total_AC_kW"])
    assert abs(value - mine) < 1e-9


def test_smoke_run_passes_and_the_objective_is_the_evaluation(smoke_run):
    assert smoke_run.code == 0
    xgb_dir = smoke_run.dir / "xgb"
    best = json.loads((xgb_dir / "best_params.json").read_text(encoding="utf-8"))
    bench = pd.read_csv(smoke_run.dir / "evaluate" / "val" / "benchmark_summary.csv")
    for name, b in best.items():
        row = bench[(bench["model"] == X.MODEL) & (bench["seed"] == b["seed"])
                    & (bench["target"] == b["target"]) & (bench["horizon"] == b["horizon"])]
        assert len(row) == 1, name
        # the evaluation scores the written forecast; equal at the table's precision (6 decimals)
        assert abs(float(row["nRMSE_kW_pct"].iloc[0]) - b["value_kw_forecast"]) < 1e-6, name
        assert abs(b["value_kw"] - b["best_trial_value_kw"]) < 1e-9  # the refit is the trial
        assert b["mos"]["applied"] is (b["target"] == "indirect")  # the wired L-09 decision
        if b["target"] == "indirect":
            assert abs(float(row["nRMSE_pct"].iloc[0]) - b["value_k_forecast"]) < 1e-6
            assert b["mos"]["value_kw_uncorrected"] == b["value_kw"]
            assert b["mos"]["value_kw_corrected"] == b["value_kw_forecast"]
        else:
            assert b["value_kw_forecast"] == b["value_kw"]
    for name in ("trials.csv", "learning_curves.csv", "xgb_config.json", "optuna.db",
                 "figures/learning_curves.png", "figures/optimization_history.png",
                 "predictions_xgb_sota_seed42.csv", "uncorrected_xgb_sota_seed42.csv",
                 "mos_coefficients.csv"):
        assert (xgb_dir / name).exists(), name
    record = json.loads((xgb_dir / "step.json").read_text(encoding="utf-8"))
    assert "optuna.db" in record["outputs"] and any(k.startswith("models/")
                                                    for k in record["outputs"])
    assert "uncorrected_xgb_sota_seed42.csv" in record["outputs"]
    assert not any(k.startswith("predictions_") and "uncorrected" in k for k in record["outputs"])
    cfg = json.loads((xgb_dir / "xgb_config.json").read_text(encoding="utf-8"))
    assert cfg["mos_adopted"] == {"indirect": True, "direct": False}
    assert cfg["mos_decision_sha256"] == sha256(smoke_run.mos)
    assert cfg["problems"]["indirect_D1"]["monotone_constraints"] == {"ghi_clearsky": 1}
    assert cfg["problems"]["direct_D1"]["monotone_constraints"] == {
        "ghi_clearsky": 1, "poa_global_clearsky": 1, "sun_elevation": 1}
    assert cfg["problems"]["direct_D1"]["nan_counts"]["fit"]["fcst_gfs_kt"] > 0


def test_development_runs_never_read_a_test_row(smoke_run):
    reads = [c for c in smoke_run.calls if c["path"].endswith("D1.parquet")]
    assert {c["path"] for c in reads} == {"features_1H_D1.parquet", "gfs_features_D1.parquet"}
    assert all(c["filters"] for c in reads)  # filtered inside the reader (X10)
    assert all(c["max"] < pd.Timestamp(TEST_START, tz="UTC") for c in reads)
    preds = pd.read_csv(smoke_run.dir / "xgb" / "predictions_xgb_sota_seed42.csv")
    (window,) = evaluation_windows("smoke")
    ts = pd.to_datetime(preds["timestamp"], utc=True)
    assert ts.min() >= window.start_ts and ts.max() < window.end_ts
    assert set(preds["split"]) == {"val"} and set(preds["model"]) == {X.MODEL}


# ── T-59 records and the saved booster ──────────────────────────────────────


def test_the_record_holds_the_model_and_the_booster_reproduces_it(smoke_run):
    xgb_dir = smoke_run.dir / "xgb"
    best = json.loads((xgb_dir / "best_params.json").read_text(encoding="utf-8"))
    written = pd.read_csv(xgb_dir / "predictions_xgb_sota_seed42.csv")
    written["timestamp"] = pd.to_datetime(written["timestamp"], utc=True)
    preds = C.base_forecast(xgb_dir, SEED)  # the booster output: uncorrected rows (X16)
    table = smoke_run.base.join(smoke_run.gfs.drop(columns="t0_utc"))
    for b in best.values():
        p = b["xgb_params"]
        assert p["random_state"] == b["seed"] == SEED and p["n_jobs"] == 8
        assert p["tree_method"] == "hist" and b["sampler"] == {"name": "TPESampler",
                                                              "seed": SEED}
        assert set(X.SEARCH_SPACE) <= set(p) and b["n_rounds"] == b["best_iteration"] + 1
        assert b["rounds_trained"] >= b["n_rounds"]
        assert p.get("monotone_constraints", {}) == b["monotone_constraints"]
        booster = xgb.Booster()
        booster.load_model(xgb_dir / b["booster"])
        assert booster.feature_names == b["feature_order"]
        mine = preds[(preds["target"] == b["target"]) & (preds["horizon"] == b["horizon"])]
        again = X.predict(booster, table.loc[mine["timestamp"], b["feature_order"]],
                          b["n_rounds"])
        np.testing.assert_array_equal(again, mine["y_pred"].to_numpy())  # exact (M10)
        raw = booster.predict(xgb.DMatrix(table.loc[mine["timestamp"], b["feature_order"]]),
                              iteration_range=(0, b["n_rounds"]))
        np.testing.assert_array_equal(raw.astype(float), mine["y_pred"].to_numpy())  # no clip
        out = written[(written["target"] == b["target"]) & (written["horizon"] == b["horizon"])]
        assert (out["timestamp"].to_numpy() == mine["timestamp"].to_numpy()).all()
        if b["mos"]["applied"]:  # the written forecast = booster + the recorded coefficients
            mos = X.Mos(b["mos"]["a_kw"], b["mos"]["b"], b["mos"]["n_oof"], b["mos"]["folds"])
            rows = table.loc[mine["timestamp"]]
            np.testing.assert_array_equal(
                X.apply_mos(b["target"], again, rows["theo_kw"].to_numpy(float),
                            rows["daytime"].to_numpy(bool), mos), out["y_pred"].to_numpy())
            assert not np.array_equal(again, out["y_pred"].to_numpy())
        else:
            np.testing.assert_array_equal(again, out["y_pred"].to_numpy())


def test_the_seed_drives_the_sampler_and_the_model(tables):
    _, p = _problem(tables, "indirect")

    def study(seed):
        s = X.run_study(p, seed, 3, None, f"t{seed}")
        return [(t.params, t.value) for t in s.trials]
    first, again, other = study(7), study(7), study(8)
    assert first == again
    assert [t[0] for t in first] != [t[0] for t in other]
    assert X.xgb_params({"max_depth": 3}, 8, {})["random_state"] == 8


def test_warnings_of_a_fit_at_the_cap_and_on_the_edge(tables):
    _, p = _problem(tables, "direct")
    fit = X.fit_early_stopping(p, {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.01},
                               SEED)
    assert fit.cap_hit and fit.best_iteration == 4
    warned = []
    ctx = SimpleNamespace(log=SimpleNamespace(warn=warned.append))
    params = {name: s["low"] for name, s in X.SEARCH_SPACE.items()}
    steps._xgb_warnings(ctx, X, "direct_D1_seed42", fit, params)
    assert any("at the cap" in w for w in warned)
    assert any("edge of the search space" in w for w in warned)
    mid = {"n_estimators": 1100, "max_depth": 6, "learning_rate": 0.0548, "subsample": 0.8,
           "colsample_bytree": 0.75, "min_child_weight": 10, "reg_alpha": 3e-4,
           "reg_lambda": 3e-4}
    assert X.edge_params(mid) == []


# ── T-60 the loader and the checks of the inputs ────────────────────────────


def test_model_table_filters_inside_the_reader(wired, monkeypatch):
    calls = []
    real = pd.read_parquet

    def spy(path, *args, **kwargs):
        calls.append(kwargs.get("filters"))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(data.pd, "read_parquet", spy)
    table = data.model_table(2)
    assert calls == [[("split", "in", ["train", "val"])],
                     [("timestamp_utc", "<", pd.Timestamp(TEST_START, tz="UTC"))]]
    assert set(table["split"]) == {"train", "val"}
    assert table.index.max() < pd.Timestamp(TEST_START, tz="UTC")
    assert "fcst_gfs_kt" in table and "t0_utc" in table  # the feature table's t0 is kept
    with pytest.raises(PermissionError):
        data.model_table(2, ("train", "val", "test"))
    full = data.model_table(2, ("train", "val", "test"), allow_test=True)
    assert (full["split"] == "test").any() and len(full) == len(wired.base)


def test_stage7_provenance_ties_the_sets_to_their_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "REPO_ROOT", tmp_path)
    table = tmp_path / "t.parquet"
    table.write_bytes(b"table v1")
    log = tmp_path / "sel.log"
    log.write_text(f"[ 0.3s] INPUT   t.parquet sha256={sha256(table)[:16]}\n", encoding="utf-8")
    manifest = tmp_path / "selection_manifest.json"
    manifest.write_text(json.dumps({"run_log": "sel.log", "validation_run_log": "none.log"}),
                        encoding="utf-8")
    rows = checks.stage7_provenance(manifest)
    assert [r["ok"] for r in rows] == [True, False]  # the table; the missing validation log
    log2 = tmp_path / "val.log"
    log2.write_text("[ 0.1s] NOTE    nothing read\n", encoding="utf-8")
    manifest.write_text(json.dumps({"run_log": "sel.log", "validation_run_log": "val.log"}),
                        encoding="utf-8")
    assert [r["ok"] for r in checks.stage7_provenance(manifest)] == [True]
    table.write_bytes(b"table v2, a rerun of stage 6")
    (row,) = checks.stage7_provenance(manifest)
    assert not row["ok"] and "selected on another table" in row["detail"]
    manifest.write_text(json.dumps({}), encoding="utf-8")
    assert not checks.stage7_provenance(manifest)[0]["ok"]


def test_a_pending_decision_is_reported_but_does_not_block(tmp_path):
    pending, other = tmp_path / "monotone_decision.yaml", tmp_path / "other.yaml"
    rows = checks.data_check((), (pending, other), selection=None,
                             pending={pending: "D-15 not decided yet"})
    by = {Path(r["path"]).name: r for r in rows}
    assert by["monotone_decision.yaml"]["blocking"] is False
    assert by["other.yaml"]["blocking"] is True
    for decision in (checks.MONOTONE_DECISION, checks.MOS_DECISION):
        assert decision in checks.TRACKED_INPUTS and decision in checks.PENDING_DECISIONS


def test_the_xgb_step_refuses_without_the_d15_decision(wired):
    wired.decision.unlink()
    r = make_runner(wired.tmp)
    assert r.run("smoke", ["baselines", "xgb"], horizons=[1]) == 1
    (run_dir,) = (wired.tmp / "runs").iterdir()
    state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    outcome = state["invocations"][-1]["results"]["xgb"]
    assert outcome.startswith("refused") and "xgb_checks d15" in outcome


# ── T-61 A-14 ───────────────────────────────────────────────────────────────


def test_validation_labels_do_not_change_the_trees(tables):
    _, p = _problem(tables, "indirect")
    params = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8}
    shuffled = dataclasses.replace(p, y_stop=pd.Series(
        np.random.default_rng(1).permutation(p.y_stop.to_numpy()), index=p.y_stop.index))
    a, b = X.fit_early_stopping(p, params, SEED), X.fit_early_stopping(shuffled, params, SEED)
    n = min(a.rounds_trained, b.rounds_trained)
    np.testing.assert_array_equal(X.predict(a.booster, p.X_score, n),
                                  X.predict(b.booster, p.X_score, n))
    fixed = X.fit_fixed(p.X_fit, p.y_fit, params, SEED, {}, a.n_rounds)  # no evaluation set
    np.testing.assert_array_equal(X.predict(fixed, p.X_score, a.n_rounds),
                                  X.predict(a.booster, p.X_score, a.n_rounds))


def test_mos_coefficients_do_not_depend_on_validation_labels(tables):
    base, gfs = tables
    table = base.join(gfs.drop(columns="t0_utc"))
    table = table[table["split"].isin(["train", "val"])].copy()
    entry = {"feature_order": INDIRECT, "monotone_constraints": {"ghi_clearsky": 1},
             "searched": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1},
             "n_rounds": 50}
    first = C.mos_fit(table, "indirect", entry, SEED)
    val = table["split"] == "val"
    rng = np.random.default_rng(2)
    table.loc[val, "site_total_AC_kW"] = rng.permutation(table.loc[val, "site_total_AC_kW"])
    table.loc[val, "target_k"] = rng.permutation(table.loc[val, "target_k"])
    second = C.mos_fit(table, "indirect", entry, SEED)
    assert (first["a"], first["b"], first["n_oof"]) == (second["a"], second["b"],
                                                        second["n_oof"])
    assert first["uncorrected_pct"] != second["uncorrected_pct"]  # validation is only scored


# ── T-62 and T-63 the checks ────────────────────────────────────────────────


def test_registered_rule_and_margin():
    assert C.margin([0.01, 0.02]) == 0.05
    assert abs(C.margin([0.1, 0.1]) - 0.2) < 1e-12
    assert C.registered_rule({1: 0.2, 2: -0.04, 3: 0.0}, {1: 0.1, 2: 0.05, 3: 0.05})["adopt"]
    assert not C.registered_rule({1: 0.08, 2: 0.3, 3: 0.3}, {1: 0.1, 2: 0.1, 3: 0.1})["adopt"]
    assert not C.registered_rule({1: 0.2, 2: -0.2, 3: 0.0}, {1: 0.1, 2: 0.1, 3: 0.1})["adopt"]
    scores = pd.DataFrame([{"horizon": h, "setting": s, "seed": seed,
                            "nrmse_kw_pct": 7.0 + (0.5 if s == "A" else 0) + 0.001 * seed}
                           for h in (1, 2, 3) for s in ("A", "B") for seed in (1, 2, 3)])
    assert C.d15_decide(scores)["indirect_setting"] == "B"  # A is 0.5 pp worse everywhere
    scores.loc[scores["setting"] == "A", "nrmse_kw_pct"] -= 0.48  # within the margin at D+1
    assert C.d15_decide(scores)["indirect_setting"] == "A"


def test_the_d15_check_writes_its_decision_and_leaves_xgb_params(tables, tmp_path):
    base, gfs = tables
    table = base.join(gfs.drop(columns="t0_utc"))
    table = table[table["split"].isin(["train", "val"])]
    before = dict(XGB_PARAMS)
    inputs = {f"indirect_D{h}": INDIRECT for h in (1, 2, 3)}
    decision_file = tmp_path / "configs" / "monotone_decision.yaml"
    decision_file.parent.mkdir()
    out = C.run_d15(tmp_path / "out", decision_file, tables={1: table}, inputs=inputs,
                    seeds=(42, 123), command="test")
    assert XGB_PARAMS == before  # X13
    assert out["settings"]["direct"] == "A" and out["settings"]["indirect"] in ("A", "B")
    assert out["constraints_A_indirect"] == {"D1": {"ghi_clearsky": 1}}
    loaded = X.load_monotone_decision(decision_file)  # readable by the xgb step
    assert loaded["settings"] == out["settings"]
    scores = pd.read_csv(tmp_path / "out" / "d15_check.csv")
    assert len(scores) == 2 * 2 and set(scores["setting"]) == {"A", "B"}
    assert not list((tmp_path / "out").rglob("predictions_*"))
    signs = pd.read_csv(tmp_path / "out" / "d15_marginal_sign.csv")
    assert list(signs["feature"]) == list(X.D15_FEATURES)


def test_guard_output(tmp_path):
    with pytest.raises(C.CheckError):
        C.guard_output(tmp_path / "predictions_x.csv", tmp_path / "runs")
    with pytest.raises(C.CheckError):
        C.guard_output(tmp_path / "runs" / "r1" / "xgb" / "l09.csv", tmp_path / "runs")
    assert C.guard_output(tmp_path / "out" / "l09.csv", tmp_path / "runs")


def test_r14_bias_and_interval():
    idx = pd.date_range("2024-02-01", periods=24 * 60, freq="h", tz="UTC")
    truth = pd.DataFrame({"split": "val", "daytime": idx.hour % 24 < 12,
                          "direct": 1000.0, "indirect": 0.5, "theo_kw": 2000.0}, index=idx)
    rng = np.random.default_rng(0)
    preds = pd.DataFrame({"timestamp": idx, "y_pred": 1010.0 + rng.normal(0, 5, len(idx)),
                          "target": "direct", "horizon": 1})
    table = C.r14_bias(preds, truth)
    overall = table[table["period"] == "all"].iloc[0]
    assert abs(overall["mbe_kw"] - 10) < 1 and overall["sign"] == "positive"
    assert overall["n"] == int(truth["daytime"].sum())
    assert set(table["period"]) == {"all", "2024-02", "2024-03"}


@pytest.mark.slow
def test_l09_and_r14_on_a_synthetic_dev_run(wired, monkeypatch):
    monkeypatch.setitem(config.BUDGETS["dev"], "xgb", {"trials": 2, "seeds": [SEED]})
    r = make_runner(wired.tmp)
    assert r.run("dev", ["baselines", "xgb", "evaluate"], horizons=[1]) == 0
    (run_dir,) = (wired.tmp / "runs").iterdir()
    before = {p: sha256(p) for p in run_dir.rglob("*") if p.is_file()}
    table = data.model_table(1)
    out = wired.tmp / "checks"
    mos_file = wired.tmp / "mos_decision.yaml"
    decision = C.run_l09(run_dir.name, wired.tmp / "runs", out, mos_file, tables={1: table},
                         seeds=(42, 123), command="test")
    assert decision["max_abs_diff_seed42_vs_dev"] < 1e-5  # the base model is the dev fit
    assert set(decision["adopted"]) == {"indirect", "direct"}
    scores = pd.read_csv(out / "l09_check.csv")
    bench = pd.read_csv(run_dir / "evaluate" / "val" / "benchmark_summary.csv")
    step_mos = pd.read_csv(run_dir / "xgb" / "mos_coefficients.csv")
    best = json.loads((run_dir / "xgb" / "best_params.json").read_text(encoding="utf-8"))
    assert list(step_mos["target"]) == ["indirect"]  # the wired decision: indirect only
    for target in ("indirect", "direct"):
        mine = scores[(scores["target"] == target) & (scores["seed"] == SEED)].iloc[0]
        dev = bench[(bench["model"] == X.MODEL) & (bench["target"] == target)]
        b = best[f"{target}_D1_seed{SEED}"]
        assert abs(mine["uncorrected_pct"] - b["value_kw"]) < 1e-9
        written = "corrected_pct" if target == "indirect" else "uncorrected_pct"
        # the evaluation scores the forecast as written (6 decimals in the table)
        assert abs(mine[written] - float(dev["nRMSE_kW_pct"].iloc[0])) < 1e-6
        if target == "indirect":  # X14: the step and the check fit the same correction
            row = step_mos.iloc[0]
            assert abs(row["a_kw"] - mine["a"]) < 1e-9 and abs(row["b"] - mine["b"]) < 1e-12
            assert row["n_oof"] == mine["n_oof"]
            assert abs(b["value_kw_forecast"] - mine["corrected_pct"]) < 1e-9
    bias = C.run_r14(run_dir.name, wired.tmp / "runs", out,
                     truth=load_truth(wired.truth_file, exclude_splits=("test",)),
                     command="test")
    assert set(bias["target"]) == {"indirect", "direct"}
    after = {p: sha256(p) for p in run_dir.rglob("*") if p.is_file()}
    assert before == after  # the checks only read the run
    assert not list(out.rglob("predictions_*"))
    with pytest.raises(C.CheckError, match="smoke"):
        smoke = make_runner(wired.tmp)
        smoke.run("smoke", ["baselines"], horizons=[1])
        (smoke_dir,) = [d for d in (wired.tmp / "runs").iterdir() if d.name.endswith("smoke")]
        C.load_xgb_run(smoke_dir.name, wired.tmp / "runs")


# ── T-64 the adopted L-09 correction in the step (§15) ──────────────────────


def test_the_corrected_kw_is_written_as_k_and_inverts_exactly():
    rng = np.random.default_rng(5)
    n = 400
    day = rng.random(n) < 0.6
    theo = np.where(day, rng.uniform(62, 3500, n), rng.uniform(0, 200, n))
    k = rng.normal(0.7, 0.4, n)  # negative and large values included, never clipped
    mos = X.Mos(a=20.0, b=0.97, n_oof=1, folds=[])
    out = X.apply_mos("indirect", k, theo, day, mos)
    expected = X.corrected_kw(mos, E.to_kw("indirect", k[day], theo[day]))
    np.testing.assert_allclose(E.to_kw("indirect", out[day], theo[day]), expected,
                               rtol=0, atol=1e-9)  # X15: the evaluation reads the corrected kW
    np.testing.assert_array_equal(out[~day], k[~day])  # nights unchanged
    assert expected.min() >= 0 and expected.max() <= AC_CAPACITY_KW
    kw = rng.uniform(-50, 3600, n)
    direct = X.apply_mos("direct", kw, theo, day, mos)
    np.testing.assert_array_equal(direct[day], X.corrected_kw(mos, kw[day]))
    np.testing.assert_array_equal(direct[~day], kw[~day])
    theo[np.flatnonzero(day)[0]] = 0.0
    with pytest.raises(ValueError, match="theo <= 0"):
        X.apply_mos("indirect", k, theo, day, mos)


def test_the_mos_decision_file_is_checked(tmp_path):
    assert X.load_mos_decision(tmp_path / "none.yaml") is None  # not decided yet
    path = tmp_path / "mos.yaml"
    path.write_text(yaml.safe_dump({"decision": "L-09",
                                    "adopted": {"indirect": True, "direct": False}}))
    assert X.load_mos_decision(path)["adopted"] == {"indirect": True, "direct": False}
    for bad in ({"decision": "D-15", "adopted": {"indirect": True, "direct": False}},
                {"decision": "L-09", "adopted": {"indirect": "yes", "direct": False}},
                {"decision": "L-09", "adopted": {"indirect": True}}):
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):
            X.load_mos_decision(path)


def test_the_check_and_the_step_share_one_correction(tables):
    base, gfs = tables
    table = base.join(gfs.drop(columns="t0_utc"))
    table = table[table["split"].isin(["train", "val"])]
    assert C.mos_rows is X.mos_rows and C.ols is X.ols  # X14: no second copy
    folds = C.train_folds(table)
    assert len(folds) == 5
    for fit, held in folds:  # expanding, chronological, a gap, train only
        assert table.index[fit].max() < table.index[held].min()
        assert (table.loc[held, "split"] == "train").all()
    params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1}
    entry = {"feature_order": INDIRECT, "monotone_constraints": {"ghi_clearsky": 1},
             "searched": params, "n_rounds": 40}
    check = C.mos_fit(table, "indirect", entry, SEED)
    step = X.fit_mos(table, "indirect", INDIRECT, params, SEED, {"ghi_clearsky": 1}, 40,
                     X.train_scope(table))
    assert (step.a, step.b, step.n_oof) == (check["a"], check["b"], check["n_oof"])
    assert [f["fold"] for f in step.folds] == list(range(5))
    smoke = X.train_scope(table, smoke=True)
    assert len(np.unique(X.local_day(table.index[smoke]))) == X.SMOKE_TRAIN_DAYS
    assert X.fit_mos(table, "indirect", INDIRECT, params, SEED, {}, 40, smoke).n_oof < step.n_oof


def test_without_the_l09_decision_dev_runs_are_raw_and_final_refuses(wired):
    wired.mos.unlink()
    r = make_runner(wired.tmp)
    assert r.run("smoke", ["baselines", "xgb"], horizons=[1]) == 0
    (run_dir,) = (wired.tmp / "runs").iterdir()
    xgb_dir = run_dir / "xgb"
    record = json.loads((xgb_dir / "step.json").read_text(encoding="utf-8"))
    assert any("L-09 is not decided yet" in w for w in record["warnings"])
    assert not (xgb_dir / "uncorrected_xgb_sota_seed42.csv").exists()
    assert pd.read_csv(xgb_dir / "mos_coefficients.csv").empty
    best = json.loads((xgb_dir / "best_params.json").read_text(encoding="utf-8"))
    for b in best.values():
        assert b["mos"] == {"applied": False, "reason": "L-09 not decided yet"}
        assert b["value_kw_forecast"] == b["value_kw"]
    verified = {"select": {"values": {"model_inputs_sha256": sha256(X.MODEL_INPUTS_FILE)}}}
    ctx = SimpleNamespace(mode="final", config={"horizons": [1]}, completed=verified)
    with pytest.raises(steps.RunnerError, match="L-09"):
        steps.xgb_inputs(ctx)
