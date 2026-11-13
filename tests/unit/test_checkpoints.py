"""Checkpoints and resume inside the long steps (docs/stage9_plan.md §23, D-38, X94).

T-113  the store: atomic writes, an identity that must match, a tree with uncommitted code that
       never reads one, finished units with their hashes, damaged files dropped, the data digest
T-114  a study replayed from its finished trials and its pickled sampler draws the parameters an
       uninterrupted study draws, also when a trial was running at the interruption
T-115  the runner gives a step its checkpoint folder and the fingerprint, keeps the folder after
       a failure, shows it in `status`, deletes it after a success, moves aside one that belongs
       to another fingerprint; `archive` does not pack it
T-116  the xgb step interrupted in a study and after a unit continues to the files of an
       uninterrupted step
T-119  the neural step passes the checkpoint folder and the identity to the worker and records what
       the worker restored (the worker itself: tests/gpu/test_checkpoint_gpu.py, T-117, T-118)
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest
from test_analyses import analysis_run, make_ctx  # noqa: F401 (a fixture)
from test_neural_runner import data, run_step, stub_worker, wired  # noqa: F401 (fixtures)
from test_w5_steps import make_runner, only_run, read
from test_xgb import tables  # noqa: F401 (fixture)

from solarpv.models import checkpoint as CK
from solarpv.models import neural_job as J
from solarpv.pipeline import archive, neural, runner, steps

IDENTITY = {"run_id": "r1", "step": "bilstm", "mode": "final", "fingerprint": "abc"}


def say_into(messages):
    return messages.append


# ── T-113 the store ─────────────────────────────────────────────────────────


def test_an_atomic_write_leaves_no_part_file_and_a_cut_off_one_is_removed(tmp_path):
    CK.write_atomic(tmp_path / "a.bin", lambda f: f.write(b"abc"))
    assert (tmp_path / "a.bin").read_bytes() == b"abc"
    assert not list(tmp_path.glob("*.part"))
    root = tmp_path / "ck"
    root.mkdir()
    (root / "fit.pt.part").write_bytes(b"half")  # a write that was cut off by a kill
    store = CK.open_store(root, IDENTITY, say=lambda m: None)
    assert store.enabled and not list(root.glob("*.part"))


def test_a_store_with_the_same_identity_is_resumed_and_another_is_moved_aside(tmp_path):
    root, messages = tmp_path / "ck", []
    first = CK.open_store(root, IDENTITY, say=say_into(messages))
    assert not first.had_checkpoint
    first.save_pickle("study_x.pkl", {"trials": [1, 2]})
    again = CK.open_store(root, IDENTITY, say=say_into(messages))
    assert again.had_checkpoint and again.load_pickle("study_x.pkl") == {"trials": [1, 2]}
    other = CK.open_store(root, {**IDENTITY, "fingerprint": "def"}, say=say_into(messages))
    assert not other.had_checkpoint and other.load_pickle("study_x.pkl") is None
    (stale,) = tmp_path.glob("ck.stale-*")
    assert (stale / "study_x.pkl").exists()
    assert any("another step fingerprint" in m for m in messages)


def test_a_tree_with_uncommitted_code_never_reads_a_checkpoint(tmp_path):
    root, messages = tmp_path / "ck", []
    CK.open_store(root, IDENTITY, say=say_into(messages)).save_pickle("p.pkl", 1)
    store = CK.open_store(root, IDENTITY, trust=False, say=say_into(messages))
    assert store.load_pickle("p.pkl") is None and list(tmp_path.glob("ck.stale-*"))
    assert any("uncommitted code" in m for m in messages)


def test_a_disabled_store_does_nothing():
    store = CK.open_store(None, None)
    assert not store.enabled
    store.save_pickle("x", 1)
    store.save_unit("u", {}, {})
    assert store.load_pickle("x") is None and store.restore_unit("u", ".") is None
    store.bind("indirect", "d")
    store.drop("x")


def test_a_unit_is_restored_with_its_hashes_and_a_changed_file_discards_it(tmp_path):
    src = tmp_path / "src"
    (src / "models").mkdir(parents=True)
    (src / "forecast.npz").write_bytes(b"forecast")
    (src / "models" / "m.pt").write_bytes(b"model")
    store = CK.open_store(tmp_path / "ck", IDENTITY, say=lambda m: None)
    store.save_unit("indirect_seed42", {"forecast.npz": src / "forecast.npz",
                                        "models/m.pt": src / "models" / "m.pt"}, {"seed": 42})
    out = tmp_path / "out"
    assert store.restore_unit("indirect_seed42", out) == {"seed": 42}
    assert (out / "forecast.npz").read_bytes() == b"forecast"
    assert (out / "models" / "m.pt").read_bytes() == b"model"
    messages = []
    store.warn = say_into(messages)
    (tmp_path / "ck" / "unit_indirect_seed42" / "forecast.npz").write_bytes(b"changed")
    assert store.restore_unit("indirect_seed42", tmp_path / "out2") is None
    assert not (tmp_path / "ck" / "unit_indirect_seed42").exists()
    assert any("damaged" in m for m in messages)
    # a unit counts only once its record exists (the record is written last)
    (tmp_path / "ck" / "unit_half").mkdir()
    (tmp_path / "ck" / "unit_half" / "forecast.npz").write_bytes(b"x")
    assert store.restore_unit("half", tmp_path / "out3") is None


def test_describe_counts_both_kinds_of_finished_unit(tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(b"x")
    store = CK.open_store(tmp_path / "ck", IDENTITY, say=lambda m: None)
    store.save_unit("indirect_seed42", {"f.bin": src}, {})  # a neural unit: a folder
    store.save_pickle("unit_xgb_indirect_D1_seed42.pkl", {})  # an xgb unit: a pickle
    store.save_pickle("study_xgb_direct_D1_seed42.pkl", {})
    lines = CK.describe(tmp_path / "ck")
    assert "2 finished unit(s), 1 study checkpoint(s)" in lines[0]
    assert "indirect_seed42" in lines[1] and "xgb_indirect_D1_seed42" in lines[1]


def test_a_write_is_retried_when_the_target_is_held_for_a_moment(tmp_path, monkeypatch):
    """R-G1-28: on Windows an antivirus or an indexer may hold the target of the rename."""
    monkeypatch.setattr(CK, "REPLACE_WAIT_S", 0.0)
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) <= 2:
            raise PermissionError("held by another process")
        return real(src, dst)
    monkeypatch.setattr(CK.os, "replace", flaky)
    CK.write_atomic(tmp_path / "a.bin", lambda f: f.write(b"abc"))
    assert (tmp_path / "a.bin").read_bytes() == b"abc" and len(calls) == 3
    monkeypatch.setattr(CK.os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(PermissionError):
        CK.write_atomic(tmp_path / "b.bin", lambda f: f.write(b"abc"))


def test_an_unreadable_manifest_is_another_identity(tmp_path):
    """R-G1-30: the folder is moved aside and the step starts again, at every resume."""
    root, messages = tmp_path / "ck", []
    root.mkdir()
    (root / CK.MANIFEST).write_text("{ not json", encoding="utf-8")
    (root / "study_x.pkl").write_bytes(b"x")
    store = CK.open_store(root, IDENTITY, say=say_into(messages))
    assert not store.had_checkpoint and list(tmp_path.glob("ck.stale-*"))
    assert any("manifest cannot be read" in m for m in messages)


def test_a_resume_on_another_environment_is_recorded_and_warned(tmp_path):
    """R-G1-31: valid, and not claimed to equal an uninterrupted run (A-20)."""
    notes, warnings = [], []
    first = CK.open_store(tmp_path / "ck", IDENTITY, say=notes.append, warn=warnings.append)
    first.environment({"gpu": "P40", "torch": "2.5.1"})
    same_stack = CK.open_store(tmp_path / "ck", IDENTITY, say=notes.append, warn=warnings.append)
    same_stack.environment({"gpu": "P40", "torch": "2.5.1"})
    assert not same_stack.events and not warnings
    other = CK.open_store(tmp_path / "ck", IDENTITY, say=notes.append, warn=warnings.append)
    other.environment({"gpu": "A100", "torch": "2.5.1"})
    assert [e["kind"] for e in other.events] == ["environment_differs"]
    assert other.events[0]["made_on"][0]["gpu"] == "P40" and other.events[0]["now"]["gpu"] == "A100"
    assert warnings and "not claimed to equal" in warnings[0]
    # R-G1-36: a third try on the new stack still says that part of the folder is from the old one
    third = CK.open_store(tmp_path / "ck", IDENTITY, say=notes.append, warn=warnings.append)
    third.environment({"gpu": "A100", "torch": "2.5.1"})
    assert [e["kind"] for e in third.events] == ["environment_differs"]
    assert third.events[0]["made_on"] == [{"gpu": "P40", "torch": "2.5.1"}]


def test_the_warnings_of_a_store_go_to_the_warn_callback(tmp_path):
    notes, warnings = [], []
    CK.open_store(tmp_path / "ck", IDENTITY, say=notes.append, warn=warnings.append)
    CK.open_store(tmp_path / "ck", {**IDENTITY, "fingerprint": "x"}, say=notes.append,
                  warn=warnings.append)
    assert len(warnings) == 1 and "not used" in warnings[0] and not notes


def test_a_finished_unit_without_its_study_is_computed_again(tmp_path):
    """R-G1-30: it was a failure at every resume; now the unit is dropped."""
    from solarpv.models import neural_worker

    src = tmp_path / "f.bin"
    src.write_bytes(b"x")
    messages = []
    store = CK.open_store(tmp_path / "ck", IDENTITY, say=lambda m: None, warn=messages.append)
    store.save_unit("indirect_seed42", {"f.bin": src}, {"seed": 42})
    out = tmp_path / "out"
    out.mkdir()
    assert neural_worker.restore_unit(store, out, "indirect_seed42", None, "bilstm", 42) is None
    assert not store.has("unit_indirect_seed42") and not (out / "f.bin").exists()
    assert any("no readable study checkpoint" in m for m in messages)
    # a unit of a step without a study (the TFT baseline, PatchTST) needs none
    store.save_unit("direct_seed42", {"f.bin": src}, {"seed": 42})
    assert neural_worker.restore_unit(store, out, "direct_seed42") == {"seed": 42}


def test_a_damaged_pickle_is_dropped(tmp_path):
    messages = []
    store = CK.open_store(tmp_path / "ck", IDENTITY, say=say_into(messages))
    (tmp_path / "ck" / "study_x.pkl").write_bytes(b"not a pickle")
    assert store.load_pickle("study_x.pkl") is None
    assert not (tmp_path / "ck" / "study_x.pkl").exists()
    assert any("cannot be read" in m for m in messages)


def test_changed_data_drops_the_checkpoints_of_that_target_only(tmp_path):
    store = CK.open_store(tmp_path / "ck", IDENTITY, say=lambda m: None)
    store.bind("indirect", "d1")
    store.bind("direct", "d2")
    for name in ("study_indirect_seed42.pkl", "fit_indirect_seed42.pt", "study_direct_seed42.pkl"):
        store.save_pickle(name, 1)
    again = CK.open_store(tmp_path / "ck", IDENTITY, say=lambda m: None)
    again.bind("indirect", "d1")  # the same data: nothing is dropped
    assert again.has("study_indirect_seed42.pkl")
    again.bind("indirect", "other")
    assert not again.has("study_indirect_seed42.pkl") and not again.has("fit_indirect_seed42.pt")
    assert again.has("study_direct_seed42.pkl")


def test_the_fault_point_ends_the_process_only_when_the_variable_names_it(tmp_path):
    code = "from solarpv.models import checkpoint as C; C.fault('epoch:a:1'); C.fault('epoch:a:2')"
    env = {**os.environ}
    env.pop(CK.FAULT_VARIABLE, None)
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
    env[CK.FAULT_VARIABLE] = "epoch:a:2"
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == CK.FAULT_EXIT_CODE and "[fault] epoch:a:2" in proc.stdout


def test_the_arrays_digest_ignores_the_zip_and_sees_every_value(tmp_path):
    import numpy as np

    a = {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "m": np.array([True, False])}
    p1, p2 = tmp_path / "1.npz", tmp_path / "2.npz"
    J.write_arrays(p1, a)
    J.write_arrays(p2, a)
    assert J.arrays_digest(J.read_arrays(p1)) == J.arrays_digest(J.read_arrays(p2)) \
        == J.arrays_digest(a)
    b = {**a, "x": a["x"].copy()}
    b["x"][1, 2] += 1
    assert J.arrays_digest(b) != J.arrays_digest(a)
    assert J.arrays_digest({**a, "x": a["x"].astype(np.float64)}) != J.arrays_digest(a)


# ── T-114 a replayed study ──────────────────────────────────────────────────


class Boom(Exception):
    pass


def objective(trial):
    x = trial.suggest_float("x", 1e-3, 1.0, log=True)
    n = trial.suggest_int("n", 1, 10)
    c = trial.suggest_categorical("c", [16, 32, 64])
    y = trial.suggest_float("y", 0.0, 1.0)
    return (x - 0.2) ** 2 + 0.01 * n + 0.001 * c + (y - 0.5) ** 2


def drawn(study):
    return [(t.number, t.params, t.value) for t in study.trials]


def sqlite(path):
    import optuna

    return optuna.storages.RDBStorage(f"sqlite:///{path.as_posix()}")


@pytest.mark.parametrize("where", ["after_trial_11", "during_trial_14"])
def test_a_replayed_study_draws_the_parameters_of_an_uninterrupted_study(tmp_path, monkeypatch,
                                                                         where):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    plain = CK.replayed_study(sqlite(tmp_path / "plain.db"), "s", 7)
    plain.optimize(objective, n_trials=25)

    store = CK.open_store(tmp_path / "ck", IDENTITY, say=lambda m: None)
    if where == "after_trial_11":
        def fault(point):
            if point == "t:11":
                raise Boom()
        monkeypatch.setattr(CK, "fault", fault)
        run = objective
    else:
        monkeypatch.setattr(CK, "fault", lambda p: None)

        def run(trial):  # the process dies inside trial 14, before it finishes
            if trial.number == 14:
                raise Boom()
            return objective(trial)
    first = CK.replayed_study(sqlite(tmp_path / "a.db"), "s", 7)
    with pytest.raises(Boom):
        first.optimize(run, n_trials=25, callbacks=[CK.study_callback(store, "study.pkl", "t")])
    state = store.load_pickle("study.pkl")
    assert len(state["trials"]) == (12 if where == "after_trial_11" else 14)
    resumed = CK.replayed_study(sqlite(tmp_path / "b.db"), "s", 7, state)
    assert [t.number for t in resumed.trials] == list(range(len(state["trials"])))
    resumed.optimize(objective, n_trials=25 - len(state["trials"]))
    assert drawn(resumed) == drawn(plain)
    assert resumed.best_params == plain.best_params


# ── T-115 the runner ────────────────────────────────────────────────────────


def checkpointing_step(seen, fail_first=True):
    """A step that keeps something in its checkpoint folder, dies on its first try, and on the
    second finds it."""
    def inputs(ctx):
        return []

    def run(ctx):
        store = CK.open_store(ctx.checkpoints, steps.checkpoint_identity(ctx),
                              ctx.trust_checkpoints, say=ctx.log.note)
        seen.append({"had": store.had_checkpoint, "fp": ctx.fingerprint,
                     "trust": ctx.trust_checkpoints, "folder": ctx.checkpoints})
        if not store.has("marker.pkl"):
            store.save_pickle("marker.pkl", {"epoch": 3})
            if fail_first:
                raise RuntimeError("the process died")
        else:
            store.note("marker", epoch=store.load_pickle("marker.pkl")["epoch"])
        out = ctx.out / "x.txt"
        out.write_text("x", encoding="utf-8")
        return steps.Result([out], notes={"resumed": store.events} if store.events else {})
    return steps.Implementation(inputs, run)


def test_a_failed_step_keeps_its_checkpoint_and_a_resume_uses_and_deletes_it(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    seen = []
    r = make_runner(tmp_path, impls={"xgb": checkpointing_step(seen)})
    assert r.run("smoke", ["xgb"]) == 1
    run_dir = only_run(tmp_path)
    folder = run_dir / "checkpoints" / "xgb"
    assert folder.is_dir() and (folder / "marker.pkl").exists()
    assert seen[0]["folder"] == folder and not seen[0]["had"] and seen[0]["trust"]
    r.echo = True
    r.status(run_dir.name)
    assert "checkpoint xgb: 0 finished unit(s)" in capsys.readouterr().out
    r.echo = False
    assert r.run("smoke", ["xgb"], resume=run_dir.name) == 0
    assert seen[1]["had"] and seen[1]["fp"] == seen[0]["fp"]
    assert not (run_dir / "checkpoints").exists()  # deleted after the success
    notes = read(run_dir / "xgb" / "step.json")["notes"]
    assert notes["resumed"] == [{"kind": "marker", "epoch": 3}]


def test_a_checkpoint_of_another_fingerprint_is_moved_aside(tmp_path, monkeypatch):
    seen = []
    watched = tmp_path / "input.txt"
    watched.write_text("one", encoding="utf-8")
    impl = checkpointing_step(seen)
    impl = steps.Implementation(lambda ctx: [watched], impl.run)
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    r = make_runner(tmp_path, impls={"xgb": impl})
    assert r.run("smoke", ["xgb"]) == 1
    watched.write_text("two", encoding="utf-8")  # another input, another fingerprint
    run_dir = only_run(tmp_path)
    assert r.run("smoke", ["xgb"], resume=run_dir.name) == 1  # it finds no marker, dies again
    assert not seen[1]["had"] and seen[1]["fp"] != seen[0]["fp"]
    assert list((run_dir / "checkpoints").glob("xgb.stale-*"))
    assert r.run("smoke", ["xgb"], resume=run_dir.name) == 0  # now the marker is its own
    assert seen[2]["had"] and not list((run_dir / "checkpoints").glob("*"))


def test_a_tree_with_uncommitted_code_does_not_trust_a_checkpoint(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234+dirty")
    r = make_runner(tmp_path, impls={"xgb": checkpointing_step(seen)})
    assert r.run("smoke", ["xgb"]) == 1
    run_dir = only_run(tmp_path)
    assert r.run("smoke", ["xgb"], resume=run_dir.name) == 1
    assert not seen[0]["trust"] and not seen[1]["trust"] and not seen[1]["had"]
    assert list((run_dir / "checkpoints").glob("xgb.stale-*"))


def test_an_unknown_commit_is_not_trusted_either(tmp_path, monkeypatch):
    """R-G1-31: when git fails the fingerprints of different code are equal."""
    seen = []
    monkeypatch.setattr(runner, "git_commit", lambda: "unknown")
    r = make_runner(tmp_path, impls={"xgb": checkpointing_step(seen)})
    assert r.run("smoke", ["xgb"]) == 1
    assert r.run("smoke", ["xgb"], resume=only_run(tmp_path).name) == 1
    assert not seen[0]["trust"] and not seen[1]["trust"] and not seen[1]["had"]


def test_a_complete_step_leaves_no_checkpoint_when_it_is_adopted_or_skipped(
        tmp_path, monkeypatch):
    """R-G1-29: the folder was removed only on the success path of the try that wrote it."""
    monkeypatch.setattr(runner, "git_commit", lambda: "abc1234")
    seen = []
    r = make_runner(tmp_path, impls={"xgb": checkpointing_step(seen, fail_first=False)})
    assert r.run("smoke", ["xgb"]) == 0
    run_dir = only_run(tmp_path)
    left = run_dir / "checkpoints" / "xgb"
    left.mkdir(parents=True)
    (left / "study_x.pkl").write_bytes(b"x")
    assert r.run("smoke", ["xgb"], resume=run_dir.name) == 0  # up to date: skipped
    assert not (run_dir / "checkpoints").exists()
    # an adopted folder: run.json lost the entry of the completed step
    state = read(run_dir / "run.json")
    state["steps"].pop("xgb")
    (run_dir / "run.json").write_text(json.dumps(state), encoding="utf-8")
    left.mkdir(parents=True)
    (left / "study_x.pkl").write_bytes(b"x")
    assert r.run("smoke", ["xgb"], resume=run_dir.name) == 0
    assert not (run_dir / "checkpoints").exists()


def test_git_ignores_the_checkpoints_of_a_run():
    from solarpv.config.paths import REPO_ROOT
    from solarpv.config.provenance import find_git

    for name in ("study_a.pkl", "fit_a.pt", "unit_a/unit.json", "manifest.json"):
        out = subprocess.run([find_git(), "check-ignore", "-q",
                              f"outputs/runs/20261113T000000Z_final/checkpoints/bilstm/{name}"],
                             cwd=REPO_ROOT)
        assert out.returncode == 0, name


def test_the_report_counts_the_restores_of_a_step():
    from types import SimpleNamespace

    from solarpv.pipeline import report

    rec = {"status": "ok", "commit": "abc", "fingerprint": "f" * 16, "duration_s": 1.0,
           "outputs": {}, "warnings": [], "config": {}}
    ctx = SimpleNamespace(step="report", failed=[], completed={
        "xgb": {**rec, "notes": {}},
        "bilstm": {**rec, "notes": {"resumed": [{"kind": "unit"}, {"kind": "fit"}]}}})
    table = report.steps_table(ctx).set_index("step")
    assert table.loc["bilstm", "n_resumed"] == 2 and table.loc["xgb", "n_resumed"] == 0


def test_archive_does_not_pack_the_checkpoints(tmp_path):
    run = tmp_path / "run"
    (run / "xgb").mkdir(parents=True)
    (run / "checkpoints" / "xgb").mkdir(parents=True)
    (run / "run.json").write_text("{}", encoding="utf-8")
    (run / "xgb" / "predictions.csv").write_text("x", encoding="utf-8")
    (run / "checkpoints" / "xgb" / "study_a.pkl").write_bytes(b"x")
    names = [p.relative_to(run).as_posix() for p in archive._files(run)]
    assert "xgb/predictions.csv" in names and not any(n.startswith("checkpoints") for n in names)


# ── T-116 the xgb step ──────────────────────────────────────────────────────


def xgb_ctx(run, tmp_path, name, checkpoints=None, trust=True):
    ctx = make_ctx(run, "xgb", tmp_path / name, completed={})
    ctx.config["budgets"]["xgb"] = {**ctx.config["budgets"]["xgb"], "trials": 3}
    ctx.checkpoints, ctx.fingerprint, ctx.trust_checkpoints = checkpoints, "fp", trust
    return ctx


def same_outputs(a, b):
    """Every file that carries a result is equal; the durations of the trials are not."""
    names = sorted(p.name for p in a.glob("*.csv")) + sorted(p.name for p in
                                                             (a / "models").glob("*.ubj"))
    assert names
    for name in names:
        pa, pb = (a / name, b / name) if name.endswith(".csv") else (a / "models" / name,
                                                                     b / "models" / name)
        if name == "trials.csv":
            da, db = pd.read_csv(pa).drop(columns="duration_s"), pd.read_csv(pb).drop(
                columns="duration_s")
            assert da.equals(db)
        else:
            assert pa.read_bytes() == pb.read_bytes(), name
    ba, bb = read(a / "best_params.json"), read(b / "best_params.json")
    assert ba == bb


@pytest.mark.parametrize("point, kinds", [
    ("trial:xgb_indirect_D1_seed42:1", ["study"]),  # inside the first study, two trials done
    ("unit:xgb_indirect_D1_seed42", ["unit"]),  # the first unit finished
    ("trial:xgb_direct_D1_seed42:0", ["unit", "study"])])  # the second study, one trial done
def test_the_xgb_step_interrupted_continues_to_the_files_of_an_uninterrupted_step(
        analysis_run, tmp_path, monkeypatch, point, kinds):  # noqa: F811
    plain = xgb_ctx(analysis_run, tmp_path, "plain")
    steps.run_xgb(plain)
    ck = tmp_path / "ck"
    real = CK.fault

    def fault(p):
        if p == point:
            raise Boom()
        real(p)
    monkeypatch.setattr(CK, "fault", fault)
    with pytest.raises(Boom):
        steps.run_xgb(xgb_ctx(analysis_run, tmp_path, "killed", ck))
    monkeypatch.setattr(CK, "fault", real)
    resumed = xgb_ctx(analysis_run, tmp_path, "resumed", ck)
    result = steps.run_xgb(resumed)
    same_outputs(plain.out, resumed.out)
    assert [e["kind"] for e in result.notes["resumed"]] == kinds
    assert any("continued from a checkpoint" in w for w in resumed.log.warnings)  # R-G1-32
    # the studies of the step's database hold every trial, as in the uninterrupted step
    import optuna

    for ctx in (plain, resumed):
        db = optuna.storages.RDBStorage(f"sqlite:///{(ctx.out / 'optuna.db').as_posix()}")
        counts = {s.study_name: len(optuna.load_study(study_name=s.study_name,
                                                      storage=db).trials)
                  for s in optuna.get_all_study_summaries(db)}
        db.remove_session()
        db.engine.dispose()
        assert counts == {"xgb_sota_indirect_D1_seed42": 3, "xgb_sota_direct_D1_seed42": 3}, counts


def test_a_tree_with_uncommitted_code_runs_the_xgb_step_from_its_first_line(
        analysis_run, tmp_path, monkeypatch):  # noqa: F811
    ck = tmp_path / "ck"
    real = CK.fault
    monkeypatch.setattr(CK, "fault", lambda p: (_ for _ in ()).throw(Boom())
                        if p == "unit:xgb_indirect_D1_seed42" else None)
    with pytest.raises(Boom):
        steps.run_xgb(xgb_ctx(analysis_run, tmp_path, "killed", ck))
    monkeypatch.setattr(CK, "fault", real)
    ctx = xgb_ctx(analysis_run, tmp_path, "again", ck, trust=False)
    result = steps.run_xgb(ctx)
    assert "resumed" not in result.notes and list(tmp_path.glob("ck.stale-*"))


# ── T-119 the neural step, on the stub worker ───────────────────────────────


def test_the_neural_step_passes_the_checkpoint_folder_and_records_what_was_restored(
        wired, monkeypatch):  # noqa: F811
    def worker_that_restored(ctx, job_dir, out_dir):
        stub_worker(ctx, job_dir, out_dir)
        path = out_dir / J.WORKER_FILE
        record = J.read_json(path)
        record["resumed"] = [{"kind": "unit", "unit": "indirect_seed42"},
                             {"kind": "fit", "unit": "direct_seed42", "fit": "trial3", "epoch": 17}]
        J.write_json(path, record)
        # the stub worker lists outputs before this write; worker.json itself is not an output

    monkeypatch.setattr(neural, "call_worker", worker_that_restored)
    monkeypatch.setattr(steps, "check_selected_inputs", lambda ctx: None)
    code, run_dir = run_step(wired, "bilstm")
    assert code == 0
    job = J.read_json(run_dir / "bilstm" / "job" / J.JOB_FILE)
    assert job["resume"]["dir"] == str(run_dir / "checkpoints" / "bilstm")
    assert job["resume"]["trust"] is True
    assert set(job["resume"]["identity"]) == {"run_id", "step", "mode", "fingerprint"}
    assert job["resume"]["fit_interval_s"] == CK.FIT_INTERVAL_S  # R-G1-33
    assert job["resume"]["identity"]["step"] == "bilstm"
    record = read(run_dir / "bilstm" / "step.json")
    assert [e["kind"] for e in record["notes"]["resumed"]] == ["unit", "fit"]
    assert any("continued from a checkpoint" in w for w in record["warnings"])


def test_a_neural_step_without_a_restore_has_no_resumed_note(wired, monkeypatch):  # noqa: F811
    monkeypatch.setattr(steps, "check_selected_inputs", lambda ctx: None)
    code, run_dir = run_step(wired, "bilstm")
    assert code == 0
    record = read(run_dir / "bilstm" / "step.json")
    assert "resumed" not in record["notes"] and not any(
        "checkpoint" in w for w in record["warnings"])
    assert json.loads((run_dir / "bilstm" / "job" / J.JOB_FILE).read_text("utf-8"))["resume"]
