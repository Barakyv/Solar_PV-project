"""The runner side of the neural steps (docs/stage9_plan.md §18), on the synthetic tables of the W3
tests and a stub of the training worker (the real worker is tested in tests/gpu, in solar-gpu).

T-73  M19: one prediction file per seed and model; the seed spread from the step's records
T-74  M20, X26: the recorded objective equals the evaluation's nRMSE in kW recomputed from the
      written prediction file and the truth; the edge-of-space warning (M23)
T-76  M24: the MBE of a forecast with a known bias, in kW and in % of capacity
T-77  M13-M15, M18: a stub whose forecast encodes (origin, step) reaches the valid hour of that
      origin and step, with the horizon of its local day, on every row; no hour is missing
T-79  X25, X30, X33: the worker gets no timestamp and no test row; changed or missing worker
      outputs are refused; a missing known input outside its allowed hours is an error; every
      dataset passes check_dataset
T-80  X27, X28: every neural budget has every field and follows the decisions; every model step
      writes a model name of the evaluation layer
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_sequence_data import INPUTS, build_inputs, write_inputs

from solarpv.config.constants import AC_CAPACITY_KW, RANDOM_SEEDS, TEST_START
from solarpv.evaluation import evaluate as E
from solarpv.evaluation.units import to_kw
from solarpv.features.build import issuance_utc
from solarpv.models import neural_job as J
from solarpv.models import sequence_data as SD
from solarpv.models import xgb_sota as X
from solarpv.pipeline import config, neural, runner, steps
from solarpv.pipeline.runner import Runner
from solarpv.selection.common import PREVDAY

MEAN_K = 0.788
STUB = {"kind": "truth", "bias": 0.0, "corrupt": None}


# ── a stub of the worker: the protocol of neural_worker, without torch ─────────


def stub_worker(ctx, job_dir, out_dir):
    job = J.read_json(job_dir / J.JOB_FILE)
    out_dir.mkdir()
    (out_dir / "models").mkdir()
    records = {}
    nq = len(job["quantiles"])
    for target in job["targets"]:
        f = job["files"][target]
        arrays = J.read_arrays(job_dir / f["path"], f["sha256"])
        scale = job["meta"][target]["target_scale"]
        records[target] = {}
        for seed in job["seeds"]:
            fc = {}
            for key in job["scopes"]:
                n = arrays[f"{key}_enc_x"].shape[0]
                if STUB["kind"] == "encode":
                    point = (np.arange(n)[:, None] * 1000.0
                             + np.arange(J.MAX_STEPS)[None, :]).astype(float)
                else:
                    point = np.nan_to_num(arrays[f"{key}_y"].astype(float)) + STUB["bias"] \
                        + 1e-3 * (seed % 5)
                fc[f"{key}_q"] = np.repeat(point[..., None], nq, axis=2).astype(np.float32)
            if job["family"] == "tft":  # the worker's attention of the validation origins (M22)
                n, length = arrays["val_enc_x"].shape[:2]
                fc["val_attention"] = np.full((n, J.MAX_STEPS, length), 1 / length, np.float16)
            J.write_arrays(out_dir / J.forecast_name(target, seed), fc)
            values = J.objective(target, fc["val_q"][..., nq // 2].astype(float), arrays, "val",
                                 scale, job["mean_k"])
            final = {**values, "best_epoch": 2, "epochs_run": 4, "stopped_early": True,
                     "nan_loss": False, "best_val_loss": 0.1, "train_loss_at_best": 0.09,
                     "best_step": 10, "steps_run": 20, "val_checks": 2}
            rec = {"target": target, "seed": seed, "params": {"hidden": 64},
                   "study": {"best_trial": 0, "n_trials": 1}, "final": final,
                   "reproduce_diff_pp": None, "exogenous_futr": False}
            J.write_json(out_dir / J.fit_name(target, seed), rec)
            pd.DataFrame([{"target": target, "seed": seed, "fit": "final", "epoch": e,
                           "step": e, "train_loss": 1 / e, "val_loss": 1.1 / e}
                          for e in (1, 2, 3, 4)]).to_csv(out_dir / J.curves_name(target, seed),
                                                         index=False)
            records[target][str(seed)] = rec
    outputs = {p.relative_to(out_dir).as_posix(): J.sha256_file(p)
               for p in sorted(out_dir.rglob("*")) if p.is_file()}
    if STUB["corrupt"] == "hash":
        outputs[next(iter(outputs))] = "0" * 64
    if STUB["corrupt"] == "missing":
        outputs = {k: v for k, v in outputs.items() if not k.startswith("forecast_")}
    J.write_json(out_dir / J.WORKER_FILE, {
        "step": job["step"], "environment": {"stub": True}, "determinism": {"stub": True},
        "duration_s": 0.1, "records": records, "outputs": outputs})


def fake_report(mode):
    return {"mode": mode, "data": [], "site_private": True,
            "environment": {"python": "3.11", "platform": "test", "packages": {},
                            "cpu_count": 8, "xgb_threads": 8, "memory": None,
                            "disk_free_gb": 1.0, "gpu": None, "gpu_env": None},
            "git": {"commit": "abc1234", "tags_on_head": [], "tree_clean": True},
            "errors": [], "warnings": []}


@pytest.fixture(scope="module")
def data():
    d = build_inputs()
    for g in d["gfs"].values():  # X30: the prev-day index may be missing in train hours only
        g.loc[g.index >= pd.Timestamp("2024-02-01", tz="UTC"), PREVDAY] = \
            g[PREVDAY].fillna(0.5)
    return d


@pytest.fixture
def wired(data, tmp_path, monkeypatch):
    paths = write_inputs(data, tmp_path / "inputs")
    monkeypatch.setattr(SD, "default_paths", lambda: paths)
    inputs_file = tmp_path / "model_inputs.json"
    inputs_file.write_text(json.dumps({k: {"features": v} for k, v in INPUTS.items()}),
                           encoding="utf-8")
    monkeypatch.setattr(X, "MODEL_INPUTS_FILE", inputs_file)
    constants = tmp_path / "derived_constants.yaml"
    constants.write_text(f"mean_daytime_k: {MEAN_K}\n", encoding="utf-8")
    monkeypatch.setattr(steps, "DERIVED_CONSTANTS_FILE", constants)
    monkeypatch.setattr(steps, "mean_daytime_k", lambda: MEAN_K)
    monkeypatch.setattr(neural, "gpu_report", lambda: {"errors": [], "versions": {},
                                                       "gpu": {"name": "stub"}})
    monkeypatch.setattr(neural, "call_worker", stub_worker)
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    STUB.update({"kind": "truth", "bias": 0.0, "corrupt": None})
    return SimpleNamespace(tmp=tmp_path, paths=paths)


def run_step(wired, step, **kw):
    r = Runner(runs_dir=wired.tmp / "runs", preflight=fake_report, echo=False)
    code = r.run("smoke", [step], **kw)
    (run_dir,) = (wired.tmp / "runs").iterdir()
    return code, run_dir


def frame_and_truth(paths):
    sources = SD.load_sources(paths=paths)
    return sources, sources.frame


# ── T-80 budgets and names (X27, X28) ────────────────────────────────────────


def test_budgets_have_every_field_and_follow_the_decisions():
    fields = {"bilstm": {"trials", "epochs_trial", "epochs_final", "patience", "seeds"},
              "tft96": {"trials", "epochs", "patience", "seeds"},
              "tft168": {"trials", "epochs", "patience", "seeds"},
              "tftbase": {"trials", "epochs", "patience", "seeds"},
              "patchtst": {"max_steps", "val_check_steps", "patience", "seeds"}}
    for mode, budgets in config.BUDGETS.items():
        for step, need in fields.items():
            assert set(budgets[step]) == need, (mode, step)
            assert runner.BUDGET_FAMILY[step] == step
    dev, final = config.BUDGETS["dev"], config.BUDGETS["final"]
    assert dev["bilstm"] == {"trials": 20, "epochs_trial": 25, "epochs_final": 60, "patience": 10,
                             "seeds": [42]}  # §18.6 (A), user 2026-11-11
    assert dev["tft96"] == dev["tft168"] == {"trials": 7, "epochs": 100, "patience": 10,
                                             "seeds": [42]}
    assert dev["tftbase"]["epochs"] == 150 and dev["tftbase"]["trials"] == 0
    assert dev["patchtst"] == {"max_steps": 5000, "val_check_steps": 100, "patience": 10,
                               "seeds": [42]}
    assert final["bilstm"]["trials"] == 60 and final["bilstm"]["seeds"] == list(RANDOM_SEEDS)
    assert final["tft96"]["trials"] == 20 and final["tft96"]["seeds"] == [42]  # D-08, v6
    assert final["tftbase"]["epochs"] == 150


def test_every_model_step_has_a_model_name_of_the_evaluation():
    assert set(J.STEP_MODEL) == {"bilstm", "tft96", "tft168", "tftbase", "patchtst"}
    assert set(J.STEP_MODEL.values()) <= set(E.MODEL_LABELS)
    assert all(s in steps.IMPLEMENTED for s in J.STEP_MODEL)


# ── T-77 the mapping back (M13-M15, M18) ───────────────────────────────────


@pytest.mark.parametrize("step", ["bilstm", "tft96", "patchtst"])
def test_forecasts_reach_the_valid_hour_of_their_origin_and_step(wired, step):
    STUB["kind"] = "encode"
    code, run_dir = run_step(wired, step)
    assert code == 0
    pred = pd.read_csv(run_dir / step / f"predictions_{J.STEP_MODEL[step]}_seed42.csv")
    assert set(pred["model"]) == {J.STEP_MODEL[step]} and set(pred["split"]) == {"val"}
    sources = SD.load_sources(paths=wired.paths)
    scaler = SD.Scaler.fit(sources)
    window = config.evaluation_windows("smoke")[0]
    for target in ("indirect", "direct"):
        ds = SD.build(sources, scaler, step, target, SD.evaluation_scope(sources, window),
                      inputs=INPUTS)
        p = pred[pred["target"] == target]
        code_value = np.rint(p["y_pred"].to_numpy() / ds.target_scale).astype(np.int64)
        origin, stepno = code_value // 1000, code_value % 1000
        ts = pd.DatetimeIndex(pd.to_datetime(p["timestamp"], utc=True))
        t0 = ds.origins[origin]
        assert (ts == t0 + pd.to_timedelta(stepno, unit="h")).all()
        for h in (1, 2, 3):
            sel = (p["horizon"] == h).to_numpy()
            assert (issuance_utc(ts[sel], h) == t0[sel]).all()  # the horizon is the local day
            assert len(ts[sel]) == len(ds.scope.hours)  # every hour of the window, once
    job = J.read_json(run_dir / step / "job" / J.JOB_FILE)
    assert job["model"] == J.STEP_MODEL[step] and job["budget"] == config.BUDGETS["smoke"][step]


# ── T-79 the job (X25, X30, X33) ───────────────────────────────────────────


def test_the_job_has_no_timestamp_and_no_test_row(wired):
    code, run_dir = run_step(wired, "bilstm")
    assert code == 0
    job_dir = run_dir / "bilstm" / "job"
    job = J.read_json(job_dir / J.JOB_FILE)
    for f in job["files"].values():
        arrays = J.read_arrays(job_dir / f["path"], f["sha256"])
        assert not any(a.dtype.kind == "M" for a in arrays.values())
        assert not any("time" in k for k in arrays)
        assert {"train_enc_x", "val_enc_x", "val_score_mask"} <= set(arrays)
        assert "test_enc_x" not in arrays and job["scopes"] == ["train", "val"]
    cfg = J.read_json(run_dir / "bilstm" / "lstm_sota_config.json")
    for key, rec in cfg["datasets"].items():  # X33: every dataset passed check_dataset
        assert all(v for v in rec["checks"].values() if isinstance(v, bool)), key
        assert pd.Timestamp(rec["last_t0"]) < pd.Timestamp(TEST_START, tz="UTC")


def test_patchtst_job_holds_its_training_series(wired):
    code, run_dir = run_step(wired, "patchtst")
    assert code == 0
    job_dir = run_dir / "patchtst" / "job"
    job = J.read_json(job_dir / J.JOB_FILE)
    window = config.evaluation_windows("smoke")[0]
    hours = pd.date_range(window.start_ts, window.end_ts, freq="h", inclusive="left")
    for f in job["files"].values():
        a = J.read_arrays(job_dir / f["path"], f["sha256"])
        assert int(a["series_val_size"]) == len(hours) == 671
        assert a["val_enc_x"].shape[1:] == (J.FIXED["patchtst"]["input_size"], 1)
        assert a["series_mask"][-len(hours):].any() and np.isfinite(a["series_y"]).all()


@pytest.mark.parametrize("corrupt", ["hash", "missing"])
def test_changed_or_missing_worker_outputs_are_refused(wired, corrupt):
    STUB["corrupt"] = corrupt
    code, run_dir = run_step(wired, "bilstm")
    assert code == 1
    assert not (run_dir / "bilstm").exists()  # the try moved aside as failed
    assert list(run_dir.glob("bilstm.failed-*"))


def test_missing_known_inputs_outside_their_hours_are_an_error(data, tmp_path):
    inputs = {k: list(v) for k, v in INPUTS.items()}
    inputs["direct_D1"] = inputs["direct_D1"] + ["airmass"]
    d = {k: v for k, v in data.items()}
    tables = {h: t.copy() for h, t in data["tables"].items()}
    night = tables[1]["sun_elevation"] <= 0
    for t in tables.values():
        t.loc[night.to_numpy(), "airmass"] = np.nan
    d["tables"] = tables
    paths = write_inputs(d, tmp_path / "a")
    sources = SD.load_sources(paths=paths)
    counts = neural.check_missing_known(sources, "direct", inputs)  # night only: allowed
    assert counts["airmass"]["train"] > 0
    day = tables[1].index[(~night).to_numpy() & (tables[1]["split"] == "val").to_numpy()][0]
    for t in tables.values():
        t.loc[day, "airmass"] = np.nan
    sources = SD.load_sources(paths=write_inputs(d, tmp_path / "b"))
    with pytest.raises(steps.RunnerError, match="airmass"):
        neural.check_missing_known(sources, "direct", inputs)
    gfs = {h: g.copy() for h, g in data["gfs"].items()}
    gfs[2].loc[day, PREVDAY] = np.nan  # the prev-day index in a validation hour
    d2 = {**data, "gfs": gfs}
    sources = SD.load_sources(paths=write_inputs(d2, tmp_path / "c"))
    with pytest.raises(steps.RunnerError, match=PREVDAY):
        neural.check_missing_known(sources, "direct", INPUTS)


def test_a_failing_gpu_environment_stops_the_step(wired, monkeypatch):
    monkeypatch.setattr(neural, "gpu_report", lambda: {"errors": ["no sm_61"]})
    code, run_dir = run_step(wired, "bilstm")
    assert code == 1 and not (run_dir / "bilstm").exists()


# ── T-73, T-74, T-76: seeds, the objective, the bias ────────────────────────


def test_two_seeds_write_two_files_and_a_spread(wired):
    code, run_dir = run_step(wired, "bilstm", seeds=[42, 7])
    assert code == 0
    folder = run_dir / "bilstm"
    assert (folder / "predictions_lstm_sota_seed42.csv").exists()
    assert (folder / "predictions_lstm_sota_seed7.csv").exists()
    cfg = J.read_json(folder / "lstm_sota_config.json")
    best = J.read_json(folder / "best_params.json")
    for target in ("indirect", "direct"):
        values = [best[f"{target}_seed{s}"]["final"]["value_kw_D1"] for s in (42, 7)]
        assert abs(cfg["seed_spread_pp"][f"{target}_D1"] - np.std(values, ddof=1)) < 1e-12


def test_the_objective_is_the_evaluation_on_the_written_file(wired):
    STUB["bias"] = 0.01
    code, run_dir = run_step(wired, "tft96")
    assert code == 0
    folder = run_dir / "tft96"
    pred = pd.read_csv(folder / "predictions_tft_96_seed42.csv")
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    _, f = frame_and_truth(wired.paths)
    best = J.read_json(folder / "best_params.json")
    for target in ("indirect", "direct"):
        for h in (1, 2, 3):
            p = pred[(pred["target"] == target) & (pred["horizon"] == h)]
            row = f.loc[pd.DatetimeIndex(p["timestamp"])]
            ok = row["daytime"].astype(bool).to_numpy() & row["site_total_AC_kW"].notna().to_numpy()
            kw = to_kw(target, p["y_pred"].to_numpy()[ok], row["theo_kw"].to_numpy()[ok])
            err = kw - row["site_total_AC_kW"].to_numpy()[ok]
            hand = np.sqrt(np.mean(err ** 2)) / AC_CAPACITY_KW * 100
            assert abs(best[f"{target}_seed42"]["final"][f"value_kw_D{h}"] - hand) < 1e-9


def test_the_mbe_of_a_known_bias(wired):
    STUB["bias"] = 0.02  # model units: 0.02 x 3,449 kW on the direct route
    code, run_dir = run_step(wired, "bilstm", targets=["direct"])
    assert code == 0
    mbe = pd.read_csv(run_dir / "bilstm" / "mbe.csv")
    row = mbe[(mbe["scope"] == "val") & (mbe["horizon"] == "all")].iloc[0]
    expected = (0.02 + 1e-3 * (42 % 5)) * AC_CAPACITY_KW
    # the forecasts travel as float32: equal to about 1e-6 of capacity
    assert abs(row["mbe_kw"] - expected) < 0.01
    assert abs(row["mbe_pct"] - expected / AC_CAPACITY_KW * 100) < 1e-4
    assert set(mbe["scope"]) == {"train", "val"}


def test_the_edge_of_the_search_space_is_reported(wired):
    worker = {"records": {"indirect": {"42": {
        "params": {"hidden": 256, "n_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 64},
        "study": {"best_trial": 3}, "reproduce_diff_pp": None,
        "final": {"nan_loss": False, "best_epoch": 5, "epochs_run": 15, "stopped_early": True,
                  "best_val_loss": 0.1, "train_loss_at_best": 0.09}}}}}
    ctx = SimpleNamespace(log=SimpleNamespace(warn=lambda m: None))
    out = neural._warnings(ctx, "bilstm", worker, {"budget": {}})
    assert any("edge" in w and "hidden = 256" in w for w in out)
