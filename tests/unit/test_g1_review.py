"""The answers to the external review of gate G-1 (docs/g1_review_log.md, R-G1-01 to R-G1-11).

T-104  R-G1-01, R-G1-04, D-32: phase 1 of a final run holds no test truth and scores no test row:
       a neural step writes no `mbe` rows for the test scope, its prediction file has the forecast
       but no truth on the test rows, its job arrays hold no test truth; the same for the xgb
       step and the references
T-105  R-G1-02: named together, `test` runs before `report`, and the report written by phase 2
       holds the test tables
T-106  R-G1-03: `--allow-test` runs `test` and `report` only; the step test refuses prediction files
       that the run's `evaluate` did not score, and a step that finished after the acceptance
T-107  R-G1-05: the evaluation CLI cannot score the test split
"""

import numpy as np
import pandas as pd
import pytest
from test_analyses import StubLog, analysis_run, make_ctx, synthetic_test_window  # noqa: F401
from test_neural_runner import data, wired  # noqa: F401 (fixtures)
from test_pipeline_runner import synthetic_tables
from test_w5_steps import make_runner, only_run, read
from test_xgb import tables  # noqa: F401 (fixture)

from solarpv.config.constants import TEST_START
from solarpv.evaluation import evaluate as E
from solarpv.models import neural_job as J
from solarpv.pipeline import config, neural, runner, steps
from solarpv.pipeline.runner import check_test_flag, select_steps
from solarpv.pipeline.steps import Context

TEST_WEEK = ("2024-10-01T00:00:00+00:00", "2024-10-05T00:00:00+00:00")


@pytest.fixture(autouse=True)
def access_log(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "TEST_ACCESS_LOG", tmp_path / "test_access_log.csv")
    return tmp_path / "test_access_log.csv"


# ── T-104 phase 1 holds no test truth ───────────────────────────────────────


def test_withhold_test_truth_blanks_the_test_rows_only():
    frame = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "y_pred": [4.0, 5.0, 6.0],
                          "split": ["val", "test", "test"]})
    out = E.withhold_test_truth(frame)
    assert list(out["y_true"].isna()) == [False, True, True]
    assert list(out["y_pred"]) == [4.0, 5.0, 6.0]
    assert list(frame["y_true"]) == [1.0, 2.0, 3.0]  # the input is not changed


def phase_1_context(tmp_path, step):
    """A final-mode context of one model step on the synthetic tables: the validation window of
    the smoke run and four days of test rows (the table has a week of them)."""
    cfg = config.resolve("smoke")
    val = config.evaluation_windows("smoke")[0]
    test = config.Window("test", *TEST_WEEK, "four synthetic test days")
    out = tmp_path / f"{step}_out"
    out.mkdir()
    return Context(run_id="r", run_dir=tmp_path, mode="final", config=cfg, windows=[val, test],
                   out=out, log=StubLog(), completed={}, command="test", step=step)


@pytest.mark.usefixtures("wired")
def test_a_neural_step_of_phase_1_scores_and_holds_no_test_row(tmp_path, monkeypatch):
    """R-G1-01 (blocking): `mbe.csv` had rows `scope = test`, the test split scored in phase 1.
    R-G1-04: the prediction file and the job arrays held the test truth."""
    monkeypatch.setattr(steps, "check_selected_inputs", lambda ctx: None)
    ctx = phase_1_context(tmp_path, "bilstm")
    neural.run_neural(ctx, "bilstm")
    mbe = pd.read_csv(ctx.out / "mbe.csv")
    assert set(mbe["scope"]) == {"train", "val"}
    pred = pd.read_csv(ctx.out / "predictions_lstm_sota_seed42.csv")
    test = pred["split"] == "test"
    assert test.any() and set(pred["split"]) == {"val", "test"}
    assert pred.loc[test, "y_pred"].notna().all() and pred.loc[test, "y_true"].isna().all()
    assert pred.loc[~test, "y_true"].notna().any()
    job = J.read_json(ctx.out / "job" / J.JOB_FILE)
    assert "test" in job["scopes"]
    for f in job["files"].values():
        arrays = J.read_arrays(ctx.out / "job" / f["path"], f["sha256"])
        assert np.isnan(arrays["test_truth_kw"]).all() and np.isnan(arrays["test_k_true"]).all()
        assert not arrays["test_score_mask"].any() and not arrays["test_loss_mask"].any()
        assert not arrays["test_y"].any()
        assert np.isfinite(arrays["val_truth_kw"]).any() and arrays["val_score_mask"].any()


def test_the_xgb_step_of_phase_1_writes_no_test_truth(analysis_run, tmp_path):  # noqa: F811
    """R-G1-04 for the xgb step (the same writer as the validation rows)."""
    import yaml

    val = [config.Window(**w) for w in yaml.safe_load(
        (analysis_run.dir / "config.yaml").read_text(encoding="utf-8"))["windows"]]
    ctx = make_ctx(analysis_run, "xgb", tmp_path / "x", mode="final",
                   windows=[val[0], synthetic_test_window()], completed={})
    steps.run_xgb(ctx)
    files = sorted(ctx.out.glob("predictions_*.csv")) + sorted(ctx.out.glob("uncorrected_*.csv"))
    assert files
    for f in files:
        d = pd.read_csv(f)
        test = d["split"] == "test"
        assert test.any(), f.name
        assert d.loc[test, "y_true"].isna().all() and d.loc[test, "y_pred"].notna().all(), f.name
        assert d.loc[~test, "y_true"].notna().any(), f.name


# ── T-105 and T-106 phase 2 ─────────────────────────────────────────────────


def phase_1(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    synthetic_tables(tmp_path, monkeypatch)
    r = make_runner(tmp_path)
    assert r.run("final", ["baselines", "evaluate", "report"]) == 0
    return r, only_run(tmp_path)


def test_named_together_the_test_runs_before_the_report_and_the_report_holds_it(
        tmp_path, monkeypatch):
    """R-G1-02: the report that `promote` keeps after phase 2 had no test table, because `report`
    ran first, before `test` was complete."""
    assert select_steps(["report", "test"]) == ["test", "report"]
    r, run_dir = phase_1(tmp_path, monkeypatch)
    before = (run_dir / "report" / "report.md").read_text(encoding="utf-8")
    assert "Benchmark on test" not in before
    (run_dir / steps.ACCEPTANCE_FILE).write_text("stage 10 accepted on validation\n",
                                                 encoding="utf-8")
    r2 = make_runner(tmp_path)
    code = r2.run("final", select_steps(["report", "test"]), resume=run_dir.name,
                  allow_test=True, force=True)
    state = read(run_dir / "run.json")["steps"]
    assert code == 0, {k: (v["status"], v.get("error")) for k, v in state.items()}
    after = (run_dir / "report" / "report.md").read_text(encoding="utf-8")
    assert "Benchmark on test" in after
    state = read(run_dir / "run.json")["steps"]
    assert state["test"]["status"] == state["report"]["status"] == "ok"


def test_the_allow_test_flag_runs_test_and_report_only():
    check_test_flag("final", ["test"], True)
    check_test_flag("final", ["test", "report"], True)
    for steps_named in (["xgb", "test"], ["baselines", "test", "report"], ["evaluate", "test"]):
        with pytest.raises(ValueError, match="only the steps test and report"):
            check_test_flag("final", steps_named, True)


def test_the_step_test_refuses_files_that_evaluate_did_not_score(tmp_path, monkeypatch):
    """R-G1-03: a model step rerun after `evaluate` leaves `evaluate` ok in run.json; `test`
    would score forecasts that were never evaluated on validation or read in the report."""
    r, run_dir = phase_1(tmp_path, monkeypatch)
    monkeypatch.setattr(steps, "mean_daytime_k", lambda: 0.5)  # other references, other bytes
    assert r.run("final", ["baselines"], resume=run_dir.name, force=True) == 0
    (run_dir / steps.ACCEPTANCE_FILE).write_text("accepted\n", encoding="utf-8")  # after it
    r2 = make_runner(tmp_path)
    assert r2.run("final", ["test"], resume=run_dir.name, allow_test=True) == 1
    error = read(run_dir / "run.json")["steps"]["test"]["error"]
    assert "did not score on validation" in error and "predictions_clearsky_k.csv" in error


def test_the_step_test_refuses_a_step_that_finished_after_the_acceptance(tmp_path, monkeypatch):
    """R-G1-03: the acceptance judged an earlier state, although the files are equal."""
    r, run_dir = phase_1(tmp_path, monkeypatch)
    (run_dir / steps.ACCEPTANCE_FILE).write_text("accepted\n", encoding="utf-8")
    assert r.run("final", ["baselines"], resume=run_dir.name, force=True) == 0  # same bytes
    r2 = make_runner(tmp_path)
    assert r2.run("final", ["test"], resume=run_dir.name, allow_test=True) == 1
    error = read(run_dir / "run.json")["steps"]["test"]["error"]
    assert "baselines" in error and "finished after the stage-10 acceptance" in error
    (run_dir / steps.ACCEPTANCE_FILE).write_text("accepted again\n", encoding="utf-8")
    assert r2.run("final", ["test"], resume=run_dir.name, allow_test=True) == 0


# ── T-108 the copies of the k to kW conversion ──────────────────────────────


def test_the_copies_of_the_conversion_agree_with_the_one_of_the_evaluation():
    """R-G1-07: closed-stage modules carry their own copy of `units.to_kw`; they agree because
    the theoretical power is never negative, including for a forecast k below zero or above the
    capacity."""
    from types import SimpleNamespace

    from solarpv.config.constants import AC_CAPACITY_KW
    from solarpv.evaluation.units import to_kw
    from solarpv.features import target as feature_target
    from solarpv.selection import common as selection_common

    k = np.array([-0.5, -1e-9, 0.0, 0.4, 0.9, 1.3, 2.5])
    theo = np.array([0.0, 10.0, 49.0, 800.0, 3000.0, 3449.0, 4000.0])
    grid_k, grid_theo = np.meshgrid(k, theo)
    k, theo = grid_k.ravel(), grid_theo.ravel()
    expected = to_kw("indirect", k, theo)
    assert expected.min() >= 0.0 and expected.max() <= AC_CAPACITY_KW
    assert np.array_equal(feature_target.to_kw(pd.Series(k), pd.Series(theo)).to_numpy(), expected)
    rows = pd.RangeIndex(len(k))
    problem = SimpleNamespace(target="indirect", theo=pd.Series(theo, index=rows))
    assert np.array_equal(selection_common.to_kw(problem, k, rows), expected)
    clipped = np.clip(np.clip(k, 0, None) * theo, 0, AC_CAPACITY_KW)  # the inline copies
    assert np.array_equal(clipped, expected)


# ── T-107 the evaluation CLI ────────────────────────────────────────────────


def test_the_evaluation_cli_cannot_score_the_test_split(capsys):
    """R-G1-05: only the step `test` of a final run scores the test split."""
    for argv in (["--split", "test"], ["--allow-test"], ["--split", "test", "--allow-test"]):
        with pytest.raises(SystemExit):
            E.main(argv)
    assert "--split" in capsys.readouterr().err
    assert TEST_START  # the split exists; only the CLI no longer reaches it
