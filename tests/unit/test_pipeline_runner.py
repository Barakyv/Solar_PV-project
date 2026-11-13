"""The stage-9 runner on synthetic tables and fake steps (docs/stage9_plan.md §9, §10).

T-51  step selection, modes and budgets, the smoke window, --dry-run runs nothing
T-52  safety: no test row loaded in smoke or dev, a test-hour prediction fails a dev run, the
      final gate, --allow-test only in final
T-53  records: one folder per run, completion only with step.json, failed tries kept, --resume,
      --force, --keep-going, evaluate reads only the run's own files (by hash), check compares
      inputs with the stage manifests, nothing written outside the step folder
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solarpv.config.constants import (
    DEV_OPTUNA_TRIALS,
    FINAL_OPTUNA_TRIALS,
    RANDOM_SEEDS,
    TEST_START,
    TIMEZONE_LOCAL,
    VAL_START,
)
from solarpv.config.provenance import sha256
from solarpv.config.site import load_site
from solarpv.evaluation import evaluate as E
from solarpv.evaluation.daytime import daytime_mask
from solarpv.pipeline import checks, runner, steps
from solarpv.pipeline.config import STEP_ORDER, evaluation_windows, resolve
from solarpv.pipeline.runner import Runner, select_steps
from solarpv.pipeline.steps import Implementation, Result

MEAN_K = 0.8


def fake_report(mode="smoke", errors=(), warnings=()):
    return {"mode": mode, "data": [], "site_private": True,
            "environment": {"python": "3.11", "platform": "test", "packages": {},
                            "cpu_count": 8, "xgb_threads": 8, "memory": None,
                            "disk_free_gb": 1.0, "gpu": None},
            "git": {"commit": "abc1234", "tags_on_head": [], "tree_clean": True},
            "errors": list(errors), "warnings": list(warnings)}


@pytest.fixture(autouse=True)
def fixed_commit(monkeypatch):
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")


def fake(name, source, fail=False, outside=None):
    """A step that reads `source` and writes predictions_<name>.csv (or fails)."""
    def inputs(ctx):
        return [source]

    def run(ctx):
        ctx.log.note(f"fake {name}")
        if fail:
            raise RuntimeError(f"{name} broke")
        path = outside or ctx.out / f"predictions_{name}.csv"
        path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return Result([path], summary=[f"{name} done"])
    return Implementation(inputs, run)


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("v1", encoding="utf-8")
    return p


def make_runner(tmp_path, impls, report=None):
    return Runner(runs_dir=tmp_path / "runs", implemented=impls,
                  preflight=lambda mode: report or fake_report(mode), echo=False)


def only_run(tmp_path):
    runs = sorted((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    return runs[0]


# ── T-51 selection, modes, dry run ──────────────────────────────────────────

def test_select_steps():
    assert select_steps(["evaluate", "baselines"]) == ["baselines", "evaluate"]  # canonical order
    assert select_steps(None, "baselines", "xgb") == ["baselines", "xgb"]
    assert select_steps(None, "explain") == ["explain", "report"]
    # the second phase of a final run (D-32) runs only when it is named
    assert select_steps() == [s for s in STEP_ORDER if s != "test"]
    # R-G1-02: named together, `test` runs first, so that the report written by the invocation
    # holds the test tables
    assert select_steps(["test", "report"]) == ["test", "report"]
    assert select_steps(None, "report", "test") == ["test", "report"]
    assert select_steps(["test"]) == ["test"] and select_steps(["report"]) == ["report"]
    for bad in ({"only": ["nope"]}, {"only": ["xgb"], "from_": "xgb"},
                {"from_": "evaluate", "to": "baselines"}):
        with pytest.raises(ValueError):
            select_steps(**bad)


def test_unimplemented_steps_are_refused_with_their_work_item(tmp_path, source):
    r = Runner(runs_dir=tmp_path / "runs", implemented={"baselines": fake("b", source)},
               preflight=lambda m: fake_report(m), echo=False)
    with pytest.raises(ValueError, match=r"explain \(W5\)"):
        r.run("smoke", select_steps(None, "baselines", "explain"))
    assert not (tmp_path / "runs").exists()


def test_modes_budgets_and_windows():
    dev, final = resolve("dev"), resolve("final")
    assert dev["budgets"]["xgb"] == {"trials": DEV_OPTUNA_TRIALS, "seeds": [42]}  # D-17
    assert final["budgets"]["xgb"] == {"trials": FINAL_OPTUNA_TRIALS,
                                       "seeds": list(RANDOM_SEEDS)}  # D-08
    assert final["budgets"]["bilstm"]["trials"] == 60 and final["budgets"]["tft96"]["trials"] == 20
    (smoke,) = evaluation_windows("smoke")
    hours = pd.date_range(smoke.start_ts, smoke.end_ts, freq="h", inclusive="left")
    local = hours.tz_convert(TIMEZONE_LOCAL)
    assert local[0].hour == 0 and (local[-1] + pd.Timedelta(hours=1)).hour == 0  # whole days
    assert len(hours) == 28 * 24 - 1  # contains the 23-hour spring-forward day
    assert pd.Timestamp("2024-03-29").date() in set(local.date)
    (val,) = evaluation_windows("dev")
    assert val.start_ts == pd.Timestamp(VAL_START, tz="UTC")
    assert val.end_ts == pd.Timestamp(TEST_START, tz="UTC")
    # D-32: a final run always forecasts the test window; only the step test scores it
    assert [w.split for w in evaluation_windows("final")] == ["val", "test"]
    assert [w.split for w in evaluation_windows("final", allow_test=True)] == ["val", "test"]
    assert resolve("final")["allow_test"] is True and resolve("dev")["allow_test"] is False
    for mode in ("smoke", "dev"):
        with pytest.raises(PermissionError):
            evaluation_windows(mode, allow_test=True)


def test_dry_run_writes_nothing(tmp_path, source, capsys):
    called = []
    r = Runner(runs_dir=tmp_path / "runs", implemented={"baselines": fake("b", source)},
               preflight=lambda m: called.append(m), echo=True)
    assert r.run("smoke", ["baselines"], dry_run=True) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists() and not called  # not even the preflight


# ── T-52 safety ─────────────────────────────────────────────────────────────

def test_final_gate():
    clean = {"commit": "abc1234", "tags_on_head": ["g1-frozen"], "tree_clean": True}
    assert checks.gate_problems("final", clean, True) == ([], [])
    for state, private, text in (({**clean, "tags_on_head": []}, True, "g1-frozen"),
                                 ({**clean, "tree_clean": False}, True, "not clean"),
                                 (clean, False, "true site")):
        errors, _ = checks.gate_problems("final", state, private)
        assert any(text in e for e in errors), errors
    errors, warnings = checks.gate_problems(
        "dev", {"commit": "abc1234+dirty", "tags_on_head": [], "tree_clean": False}, True)
    assert not errors and any("uncommitted" in w for w in warnings)


def test_untracked_run_outputs_do_not_make_the_tree_dirty(monkeypatch):
    """X9: after a smoke run its untracked folder blocked the final gate."""
    class Done:
        def __init__(self, out):
            self.returncode, self.stdout = 0, out

    def fake_git(*args):
        if args[0] == "tag":
            return Done("g1-frozen\n")
        if "--untracked-files=no" in args:
            return Done(tracked)
        return Done(tracked + "?? outputs/runs/20261105T000000Z_smoke/\n")
    monkeypatch.setattr(checks, "_git", fake_git)
    monkeypatch.setattr(checks, "git_commit", lambda: "abc1234")
    tracked = ""
    state = checks.git_state()
    assert state["tree_clean"] is True and state["untracked_outputs"] == 1
    assert checks.gate_problems("final", state, True) == ([], [])
    tracked = " M outputs/07_evaluation/val/benchmark_summary.csv\n"
    assert checks.git_state()["tree_clean"] is False
    tracked = ""
    monkeypatch.setattr(checks, "git_commit", lambda: "abc1234+dirty")  # untracked code
    assert checks.git_state()["tree_clean"] is False


def test_a_failed_preflight_refuses_the_run(tmp_path, source):
    r = make_runner(tmp_path, {"baselines": fake("b", source)},
                    report=fake_report("final", errors=["HEAD does not carry the tag"]))
    assert r.run("final", ["baselines"]) == 2
    assert not (tmp_path / "runs").exists()


def test_allow_test_only_in_final(tmp_path, source):
    r = make_runner(tmp_path, {"baselines": fake("b", source)})
    with pytest.raises(PermissionError):
        r.run("dev", ["baselines"], allow_test=True)


# ── synthetic stage-6 table for the real steps ──────────────────────────────

def synthetic_tables(tmp_path, monkeypatch, seed=0):
    """A truth table in the stage-6 layout over 2023-01 .. 2024-10 (train, val, test by the real
    boundaries), wired into the steps instead of the real files."""
    idx = pd.date_range("2023-01-01", "2024-10-20", freq="h", tz="UTC", inclusive="left",
                        name="timestamp_utc")
    rng = np.random.default_rng(seed)
    day = daytime_mask(idx, load_site())
    theo = np.where(day, rng.uniform(100, 3000, len(idx)), 10.0)
    k = np.where(theo > 50, rng.uniform(0.2, 1.1, len(idx)), 0.0)
    split = np.where(idx < pd.Timestamp(VAL_START, tz="UTC"), "train",
                     np.where(idx < pd.Timestamp(TEST_START, tz="UTC"), "val", "test"))
    table = pd.DataFrame({"target_k": k, "site_total_AC_kW": k * theo, "theo_kw": theo,
                          "k_valid": theo > 50, "daytime": day, "split": split,
                          "op_limited_flag": rng.random(len(idx)) < 0.05,
                          "hour_source": "minute"}, index=idx)
    path = tmp_path / "features_1H_D1.parquet"
    table.to_parquet(path)
    constants = tmp_path / "derived_constants.yaml"
    constants.write_text(f"mean_daytime_k: {MEAN_K}\n", encoding="utf-8")
    monkeypatch.setattr(steps, "FEATURES_FILE", path)
    monkeypatch.setattr(E, "FEATURES_FILE", path)
    monkeypatch.setattr(steps, "DERIVED_CONSTANTS_FILE", constants)
    monkeypatch.setattr(steps, "mean_daytime_k", lambda: MEAN_K)
    seen = []
    real = steps.load_truth

    def spy(path, exclude_splits=()):
        out = real(path, exclude_splits=exclude_splits)
        seen.append(set(out["split"]))
        return out
    monkeypatch.setattr(steps, "load_truth", spy)
    return table, seen


def test_smoke_run_on_synthetic_tables(tmp_path, monkeypatch):
    table, seen = synthetic_tables(tmp_path, monkeypatch)
    r = make_runner(tmp_path, steps.IMPLEMENTED)
    assert r.run("smoke", ["baselines", "evaluate"]) == 0
    assert seen and all("test" not in s for s in seen)  # X3: no test row reached pandas
    run_dir = only_run(tmp_path)
    (window,) = evaluation_windows("smoke")
    inside = (table.index >= window.start_ts) & (table.index < window.end_ts)
    for model in ("persistence", "climatology", "clearsky_k"):
        p = pd.read_csv(run_dir / "baselines" / f"predictions_{model}.csv")
        ts = pd.DatetimeIndex(pd.to_datetime(p["timestamp"], utc=True))
        assert ts.min() >= window.start_ts and ts.max() < window.end_ts
        assert set(p["split"]) == {"val"}
        assert len(p) == 2 * 3 * int(inside.sum())
    bench = pd.read_csv(run_dir / "evaluate" / "val" / "benchmark_summary.csv")
    day_hours = int((inside & table["daytime"].to_numpy()).sum())
    assert (bench["n"] == day_hours).all()  # every reference on every daytime hour
    record = json.loads((run_dir / "evaluate" / "step.json").read_text(encoding="utf-8"))
    # M2: the summary is the table, not numbers from the code
    for row in bench.itertuples():
        line = next(s for s in record["summary"] if f"D+{row.horizon} {row.model:<12}" in s
                    and s.strip().startswith(row.target))
        assert f"{row.nRMSE_kW_pct:6.2f} %" in line


def test_a_prediction_in_the_test_split_fails_a_development_run(tmp_path, monkeypatch):
    table, _ = synthetic_tables(tmp_path, monkeypatch)
    hour = table.index[table["split"] == "test"][0]
    leak = tmp_path / "leak.csv"
    pd.DataFrame({"timestamp": [hour], "y_true": [0.5], "y_pred": [0.5], "target": "indirect",
                  "horizon": 1, "model": "xgb_sota", "seed": 42, "split": "val"}
                 ).to_csv(leak, index=False)  # labelled val, but a test hour (X7)
    impls = {**steps.IMPLEMENTED, "xgb": fake("xgb_sota", leak)}
    r = make_runner(tmp_path, impls)
    assert r.run("smoke", ["baselines", "xgb", "evaluate"]) == 1
    state = json.loads((only_run(tmp_path) / "run.json").read_text(encoding="utf-8"))
    assert state["steps"]["evaluate"]["status"] == "failed"
    assert "test split" in state["steps"]["evaluate"]["error"]


def test_evaluate_needs_the_references(tmp_path, monkeypatch):
    synthetic_tables(tmp_path, monkeypatch)
    r = make_runner(tmp_path, steps.IMPLEMENTED)
    assert r.run("smoke", ["evaluate"]) == 1
    state = json.loads((only_run(tmp_path) / "run.json").read_text(encoding="utf-8"))
    assert "baselines step first" in state["steps"]["evaluate"]["error"]


# ── T-53 records ────────────────────────────────────────────────────────────

def test_a_run_leaves_a_complete_record(tmp_path, source):
    r = make_runner(tmp_path, {"baselines": fake("b", source), "xgb": fake("x", source)})
    assert r.run("smoke", ["baselines", "xgb"], command="test") == 0
    run_dir = only_run(tmp_path)
    assert run_dir.name.endswith("_smoke")
    for name in ("run.json", "config.yaml", "baselines/step.json", "xgb/step.json"):
        assert (run_dir / name).exists(), name
    assert not list(run_dir.glob(".*.tmp"))
    assert len(list((run_dir / "invocations").glob("*.json"))) == 1
    record = json.loads((run_dir / "xgb" / "step.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok" and record["commit"] == "abc1234"
    assert record["outputs"] == {"predictions_x.csv": sha256(run_dir / "xgb" /
                                                             "predictions_x.csv")}
    assert record["inputs"][next(iter(record["inputs"]))] == sha256(source)
    assert Path(record["log"]).parent == run_dir / "logs" and Path(record["log"]).exists()
    r.run("smoke", ["baselines"])
    assert len(list((tmp_path / "runs").iterdir())) == 2  # a new run never reuses a folder


def test_a_failed_step_is_kept_but_never_complete(tmp_path, source):
    impls = {"baselines": fake("b", source, fail=True), "xgb": fake("x", source)}
    r = make_runner(tmp_path, impls)
    assert r.run("smoke", ["baselines", "xgb"]) == 1
    run_dir = only_run(tmp_path)
    assert not (run_dir / "baselines").exists() and not (run_dir / "xgb").exists()  # stopped
    (failed,) = run_dir.glob("baselines.failed-*")
    record = json.loads((failed / "step.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed" and "broke" in record["error"]
    (log,) = (run_dir / "logs").glob("baselines_*.log")
    text = log.read_text(encoding="utf-8")
    assert "b broke" in text and "status: failed (RuntimeError)" in text


def test_keep_going_runs_the_next_steps(tmp_path, source):
    impls = {"baselines": fake("b", source, fail=True), "xgb": fake("x", source)}
    r = make_runner(tmp_path, impls)
    assert r.run("smoke", ["baselines", "xgb"], keep_going=True) == 1
    assert (only_run(tmp_path) / "xgb" / "step.json").exists()


def test_resume_skips_refuses_and_force_reruns(tmp_path, source):
    r = make_runner(tmp_path, {"baselines": fake("b", source)})
    assert r.run("smoke", ["baselines"]) == 0
    run_id = only_run(tmp_path).name
    assert r.run("smoke", ["baselines"], resume=run_id) == 0  # same fingerprint: skipped
    state = json.loads((tmp_path / "runs" / run_id / "run.json").read_text(encoding="utf-8"))
    assert state["invocations"][-1]["results"]["baselines"] == "skipped (up to date)"
    source.write_text("v2", encoding="utf-8")  # an input changed
    assert r.run("smoke", ["baselines"], resume=run_id) == 1
    state = json.loads((tmp_path / "runs" / run_id / "run.json").read_text(encoding="utf-8"))
    assert state["invocations"][-1]["results"]["baselines"].startswith("refused")
    assert state["steps"]["baselines"]["status"] == "ok"  # the completed step is untouched
    assert r.run("smoke", ["baselines"], resume=run_id, force=True) == 0
    run_dir = tmp_path / "runs" / run_id
    assert len(list(run_dir.glob("baselines.superseded-*"))) == 1  # kept, not deleted
    assert (run_dir / "baselines" / "predictions_b.csv").read_text(encoding="utf-8") == "v2"
    with pytest.raises(ValueError, match="dev run|smoke run"):
        r.run("dev", ["baselines"], resume=run_id)
    with pytest.raises(ValueError, match="another configuration"):
        r.run("smoke", ["baselines"], resume=run_id, horizons=[1])


def test_resume_does_not_trust_uncommitted_code(tmp_path, source, monkeypatch):
    r = make_runner(tmp_path, {"baselines": fake("b", source)})
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234+dirty")
    assert r.run("smoke", ["baselines"]) == 0
    run_id = only_run(tmp_path).name
    assert r.run("smoke", ["baselines"], resume=run_id) == 1


def test_modified_outputs_are_detected(tmp_path, source):
    r = make_runner(tmp_path, {"baselines": fake("b", source)})
    r.run("smoke", ["baselines"])
    run_dir = only_run(tmp_path)
    assert r.status(run_dir.name) == 0
    (run_dir / "baselines" / "predictions_b.csv").write_text("edited", encoding="utf-8")
    assert r.status(run_dir.name) == 1
    assert r.run("smoke", ["baselines"], resume=run_dir.name) == 1


def test_a_step_writing_outside_its_folder_fails(tmp_path, source):
    r = make_runner(tmp_path, {"baselines": fake("b", source, outside=tmp_path / "x.csv")})
    assert r.run("smoke", ["baselines"]) == 1
    assert not (only_run(tmp_path) / "baselines").exists()


def test_evaluate_reads_only_the_runs_own_files(tmp_path, monkeypatch):
    synthetic_tables(tmp_path, monkeypatch)
    stale = tmp_path / "outputs" / "06_xgb_sota" / "predictions_xgb_sota.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("timestamp,y_true,y_pred,target,horizon,model\n", encoding="utf-8")
    monkeypatch.setattr(E, "OUTPUTS_DIR", tmp_path / "outputs")
    r = make_runner(tmp_path, steps.IMPLEMENTED)
    assert r.run("smoke", ["baselines", "evaluate"]) == 0
    run_dir = only_run(tmp_path)
    manifest = json.loads((run_dir / "evaluate" / "val" / "manifest.json").read_text(
        encoding="utf-8"))
    assert sorted(manifest["inputs_sha256"]) == ["predictions_clearsky_k.csv",
                                                 "predictions_climatology.csv",
                                                 "predictions_persistence.csv"]
    assert manifest["references"] == "from the files"
    with (run_dir / "baselines" / "predictions_persistence.csv").open("a") as f:
        f.write("\n")  # modified after its step completed
    assert r.run("smoke", ["evaluate"], resume=run_dir.name, force=True) == 1
    state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert "modified after its step completed" in state["invocations"][-1]["results"]["evaluate"]


def test_data_check_compares_inputs_with_the_stage_manifests(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"outputs": {a.as_posix(): sha256(a),
                                                b.as_posix(): sha256(b),
                                                (tmp_path / "c").as_posix(): "0" * 64}}),
                        encoding="utf-8")
    b.write_bytes(b"changed")
    rows = {r["path"]: r for r in checks.data_check([manifest, tmp_path / "none.json"], ())}
    assert rows[a.as_posix()]["ok"]
    assert not rows[b.as_posix()]["ok"] and "differs" in rows[b.as_posix()]["detail"]
    assert rows[(tmp_path / "c").as_posix()]["detail"] == "missing"
    assert any("manifest missing" in r["detail"] for r in rows.values())


def test_the_preflight_report_formats(tmp_path):
    lines = checks.format_preflight(fake_report(errors=["x"]))
    assert lines[-1] == "check FAILED" and "ERROR: x" in lines
