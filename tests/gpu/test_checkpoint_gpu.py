"""Checkpoints and resume of the neural worker, on the GPU (docs/stage9_plan.md §23, D-38).

T-117  `fit` stopped after an epoch and continued from the saved state gives the weights, the curve
       and the best epoch of an uninterrupted fit, for the BiLSTM and the TFT
T-118  a worker killed at an epoch, at a trial and at the end of a target x seed (exit code 86 at a
       fault point), and started again on the same checkpoint folder, writes forecasts, trials and
       records equal to an uninterrupted worker's, and says what it restored

A resume is a new process: the tests that matter run the worker as a subprocess, so that nothing
but the checkpoint carries over.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_neural_gpu import (
    BILSTM,
    DEVICE,
    META,
    TFT,
    data_of,
    objective_fn,
    scope_arrays,
)

from solarpv.models import checkpoint as CK
from solarpv.models import lstm_sota, neural_job, neural_torch, tft_model
from solarpv.models.neural_job import QUANTILES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


@pytest.fixture(scope="module")
def arrays():
    """R-G1-27: half of the training origins are short (71 or 72 steps), as the real windows are
    at the DST origins and the end of a split; the TFT cuts its decoder to the longest origin of a
    batch, so the shapes of a batch depend on which rows it holds."""
    return {**scope_arrays("train", 48, 0, short=24), **scope_arrays("val", 12, 1, short=True)}


# ── T-117: fit, stopped and continued ───────────────────────────────────────


def fit_epochs(module, params, arrays, epochs, stop_after=None, resume=None, patience=None):
    """(model, result, states): `states` are the checkpoints after every epoch."""
    neural_torch.seed_everything(7)
    if module is lstm_sota:
        model = module.build_model(params, META, 3).to(DEVICE)
        q, s = QUANTILES["bilstm"], module.train_settings(params, neural_job.FIXED["bilstm"])
    else:
        model = module.build_model(params, META).to(DEVICE)
        q, s = QUANTILES["tft"], module.train_settings(params, neural_job.FIXED["tft"])
    states = []

    def keep(state):
        states.append(CK_roundtrip(state))

    res = neural_torch.fit(model, module.forward, data_of(arrays, "train"), data_of(arrays, "val"),
                           epochs=epochs, patience=patience, lr=s["lr"],
                           batch_size=min(s["batch_size"], 16), clip=s["clip"], quantiles=q,
                           seed=7,
                           objective_fn=objective_fn(arrays), point_index=q.index(0.5),
                           resume=resume, on_checkpoint=keep)
    return model, res, states


def CK_roundtrip(state, tmp=None):
    """What a checkpoint file does to a state: serialised and read back on the CPU."""
    import io

    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


@pytest.mark.parametrize("module, params", [
    (lstm_sota, BILSTM), (lstm_sota, {**BILSTM, "n_layers": 1}),
    (lstm_sota, {**BILSTM, "dropout": 0.0}), (tft_model, TFT),
    (tft_model, {**TFT, "lstm_layers": 2}), (tft_model, {**TFT, "lstm_layers": 3, "dropout": 0.3})],
    ids=["bilstm-2-layers-dropout", "bilstm-1-layer", "bilstm-no-dropout", "tft-1-layer",
         "tft-2-layers-dropout", "tft-3-layers-dropout"])
def test_a_fit_continued_from_an_epoch_equals_an_uninterrupted_fit(arrays, module, params):
    _, full, states = fit_epochs(module, params, arrays, epochs=5)
    assert [s["epoch"] for s in states] == [1, 2, 3, 4, 5]
    model, cont, more = fit_epochs(module, params, arrays, epochs=5, resume=states[1])
    assert [s["epoch"] for s in more] == [3, 4, 5]
    assert same(cont.state, full.state)
    assert cont.best_epoch == full.best_epoch and cont.epochs_run == full.epochs_run
    assert [(r["epoch"], r["train_loss"], r["val_loss"]) for r in cont.curve] == \
           [(r["epoch"], r["train_loss"], r["val_loss"]) for r in full.curve]


def test_the_replay_runs_the_batches_of_the_real_epochs(arrays):
    """R-G1-27: the replay of cuDNN's dropout state draws the permutations of the fit's own
    generator, so it runs exactly the batches (rows, order and sizes) of the finished epochs."""
    seen = []

    def spy(model, data, idx):
        if model.training:
            seen.append(idx.tolist())
        return tft_model.forward(model, data, idx)
    params = {**TFT, "lstm_layers": 2}
    neural_torch.seed_everything(7)
    model = tft_model.build_model(params, META).to(DEVICE)
    s = tft_model.train_settings(params, neural_job.FIXED["tft"])
    neural_torch.fit(model, spy, data_of(arrays, "train"), data_of(arrays, "val"), epochs=3,
                     patience=None, lr=s["lr"], batch_size=16, clip=s["clip"],
                     quantiles=QUANTILES["tft"], seed=7, objective_fn=objective_fn(arrays),
                     point_index=3)
    real = list(seen)
    seen.clear()
    neural_torch.seed_everything(7)
    fresh = tft_model.build_model(params, META).to(DEVICE)
    neural_torch.advance_cudnn_dropout(fresh, spy, data_of(arrays, "train"), 3, 16, 7)
    assert seen == real and len(real) == 3 * 3  # three epochs of three batches, the same rows


def test_the_replay_of_the_rows_in_their_order_would_not_do_for_the_tft(arrays, monkeypatch):
    """R-G1-27 (should fix, the reviewer's finding): the first replay ran the finished epochs on
    the rows in their order. With short rows in the windows and several batches per epoch, the
    TFT (which cuts its decoder to the longest row of a batch) then ended with other weights than
    the uninterrupted fit; the replay of the real batches is equal (the tests above). The BiLSTM,
    whose forward has fixed shapes, was equal either way. (The first tests had one batch of all
    the rows for the TFT, so they could not see it.)"""
    def in_order(model, forward, train, epochs_done, batch_size, seed):
        model.train()
        idx_all = torch.arange(train.n, device=train.y.device)
        for _ in range(int(epochs_done)):
            for start in range(0, train.n, int(batch_size)):
                forward(model, train, idx_all[start:start + int(batch_size)])
    params = {**TFT, "lstm_layers": 2}
    _, full, states = fit_epochs(tft_model, params, arrays, epochs=5)
    monkeypatch.setattr(neural_torch, "advance_cudnn_dropout", in_order)
    _, wrong, _ = fit_epochs(tft_model, params, arrays, epochs=5, resume=states[1])
    assert not same(wrong.state, full.state)


def test_the_interval_of_the_checkpoints_of_a_fit(arrays):
    """R-G1-33: a checkpoint is written at the end of an epoch only if `checkpoint_every_s`
    seconds passed since the last one; 0 writes after every epoch."""
    for every, expected in ((0.0, [1, 2, 3, 4]), (1e9, [])):
        neural_torch.seed_everything(7)
        model = lstm_sota.build_model(BILSTM, META, 3).to(DEVICE)
        s = lstm_sota.train_settings(BILSTM, neural_job.FIXED["bilstm"])
        got = []
        neural_torch.fit(model, lstm_sota.forward, data_of(arrays, "train"), data_of(arrays, "val"),
                         epochs=4, patience=None, lr=s["lr"], batch_size=s["batch_size"],
                         clip=s["clip"], quantiles=QUANTILES["bilstm"], seed=7,
                         objective_fn=objective_fn(arrays), point_index=1,
                         on_checkpoint=lambda st, got=got: got.append(st["epoch"]),
                         checkpoint_every_s=every)
        assert got == expected


def test_a_state_can_be_resumed_from_twice_with_the_same_result(arrays):
    """X96: `fit` updated Adam's step counters of the state it was given, in place, so that a second
    resume from the same object trained with other bias corrections (found by the resume check
    script, which reuses one state; a checkpoint file is read afresh, so a run never met it)."""
    _, full, states = fit_epochs(lstm_sota, BILSTM, arrays, epochs=5)
    snapshot = CK_roundtrip(states[1])
    _, first, _ = fit_epochs(lstm_sota, BILSTM, arrays, epochs=5, resume=states[1])
    _, second, _ = fit_epochs(lstm_sota, BILSTM, arrays, epochs=5, resume=states[1])
    assert same(first.state, full.state) and same(second.state, full.state)
    for k, v in snapshot["opt"]["state"].items():  # the state it was given is unchanged
        assert torch.equal(states[1]["opt"]["state"][k]["step"], v["step"])


def test_a_checkpoint_after_the_last_epoch_or_a_stop_finishes_the_fit(arrays):
    _, full, states = fit_epochs(lstm_sota, BILSTM, arrays, epochs=3)
    _, again, more = fit_epochs(lstm_sota, BILSTM, arrays, epochs=3, resume=states[-1])
    assert more == [] and same(again.state, full.state) and again.epochs_run == 3
    # early stopping: the stop is itself checkpointed, and a resume ends there
    _, es, st = fit_epochs(tft_model, TFT, arrays, epochs=20, patience=1)
    assert st[-1]["done"] is (es.stopped_early or es.nan_loss)
    _, es2, more = fit_epochs(tft_model, TFT, arrays, epochs=20, patience=1, resume=st[-1])
    assert more == [] and es2.epochs_run == es.epochs_run and same(es2.state, es.state)


# ── T-118: the worker, killed and started again ─────────────────────────────


def job_files(folder: Path, arrays: dict, step: str, budget: dict, ckpt: Path | None) -> Path:
    folder.mkdir()
    sha = neural_job.write_arrays(folder / "data_indirect.npz", arrays)
    family = neural_job.STEP_FAMILY[step]
    job = {"version": neural_job.JOB_VERSION, "step": step, "model": neural_job.STEP_MODEL[step],
           "family": family, "mode": "smoke", "run_id": "test", "targets": ["indirect"],
           "seeds": budget["seeds"], "budget": budget, "scopes": ["train", "val"],
           "tune_scope": "val", "mean_k": 0.788, "meta": {"indirect": META},
           "files": {"indirect": {"path": "data_indirect.npz", "sha256": sha}}}
    if ckpt is not None:
        job["resume"] = {"dir": str(ckpt), "trust": True, "fit_interval_s": 0,
                         "identity": {"run_id": "test", "step": step, "mode": "smoke",
                                      "fingerprint": "abc"}}
    neural_job.write_json(folder / neural_job.JOB_FILE, job)
    return folder


def worker(job_dir: Path, out: Path, fault: str | None = None):
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONIOENCODING": "utf-8"}
    env.pop(CK.FAULT_VARIABLE, None)
    if fault:
        env[CK.FAULT_VARIABLE] = fault
    return subprocess.run([sys.executable, "-m", "solarpv.models.neural_worker", str(job_dir),
                           str(out)], capture_output=True, text=True, env=env, timeout=1500)


def compare(a: Path, b: Path, step: str, seeds):
    """The outputs that carry the result are equal: forecasts (arrays), the fit records, the
    trials' values and parameters, the curves' losses."""
    for seed in seeds:
        fa = neural_job.read_arrays(a / neural_job.forecast_name("indirect", seed))
        fb = neural_job.read_arrays(b / neural_job.forecast_name("indirect", seed))
        assert fa.keys() == fb.keys()
        for k in fa:
            assert np.array_equal(fa[k], fb[k]), f"{k} differs"
        ra = json.loads((a / neural_job.fit_name("indirect", seed)).read_text("utf-8"))
        rb = json.loads((b / neural_job.fit_name("indirect", seed)).read_text("utf-8"))
        for k in ("params", "study", "reproduce_diff_pp"):
            assert ra[k] == rb[k], k
        for k in ("best_epoch", "epochs_run", "best_val_loss", "train_loss_at_best", "value_kw"):
            assert ra["final"][k] == rb["final"][k], k
        import pandas as pd

        ca = pd.read_csv(a / neural_job.curves_name("indirect", seed))
        cb = pd.read_csv(b / neural_job.curves_name("indirect", seed))
        assert ca[["fit", "epoch", "train_loss", "val_loss"]].equals(
            cb[["fit", "epoch", "train_loss", "val_loss"]])
        ta, tb = a / neural_job.trials_name("indirect", seed), b / neural_job.trials_name(
            "indirect", seed)
        assert ta.exists() == tb.exists()
        if ta.exists():
            da, db = pd.read_csv(ta), pd.read_csv(tb)
            cols = [c for c in da.columns if c.startswith("param_") or c in ("trial", "value_kw")]
            assert da[cols].equals(db[cols])
    return json.loads((b / neural_job.WORKER_FILE).read_text("utf-8"))  # the resumed worker's


CASES = [
    # (step, budget, fault point, what the second worker must report as restored)
    ("bilstm", {"trials": 3, "epochs_trial": 3, "epochs_final": 3, "patience": 10, "seeds": [42]},
     "epoch:indirect_seed42:trial1:2", {"fit", "study"}),
    ("bilstm", {"trials": 3, "epochs_trial": 3, "epochs_final": 3, "patience": 10, "seeds": [42]},
     "trial:indirect_seed42:1", {"study"}),
    ("bilstm", {"trials": 2, "epochs_trial": 2, "epochs_final": 3, "patience": 10, "seeds": [42]},
     "epoch:indirect_seed42:final:2", {"fit", "study"}),
    ("bilstm", {"trials": 1, "epochs_trial": 2, "epochs_final": 2, "patience": 10,
                "seeds": [42, 123]}, "unit:indirect_seed42", {"unit"}),
    ("tft96", {"trials": 3, "epochs": 3, "patience": 10, "seeds": [42]},
     "epoch:indirect_seed42:trial2:2", {"fit", "study"}),
    ("tft96", {"trials": 2, "epochs": 3, "patience": 10, "seeds": [42]},
     "epoch:indirect_seed42:final:3", {"fit", "study"}),
    ("tftbase", {"trials": 0, "epochs": 4, "patience": 10, "seeds": [42]},
     "epoch:indirect_seed42:final:2", {"fit"}),
]


@pytest.mark.parametrize("step, budget, fault, kinds", CASES,
                         ids=[f"{c[0]}-{c[2]}" for c in CASES])
def test_a_killed_worker_continues_to_the_result_of_an_uninterrupted_worker(
        arrays, tmp_path, step, budget, fault, kinds):
    plain = job_files(tmp_path / "job_plain", arrays, step, budget, None)
    proc = worker(plain, tmp_path / "out_plain")
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    ckpt = tmp_path / "ckpt"
    first = job_files(tmp_path / "job_1", arrays, step, budget, ckpt)
    killed = worker(first, tmp_path / "out_1", fault=fault)
    assert killed.returncode == CK.FAULT_EXIT_CODE, killed.stdout[-1500:] + killed.stderr[-1500:]
    assert not (tmp_path / "out_1" / neural_job.WORKER_FILE).exists()
    second = job_files(tmp_path / "job_2", arrays, step, budget, ckpt)
    resumed = worker(second, tmp_path / "out_2")
    assert resumed.returncode == 0, resumed.stdout[-1500:] + resumed.stderr[-1500:]
    record = compare(tmp_path / "out_plain", tmp_path / "out_2", step, budget["seeds"])
    assert {e["kind"] for e in record["resumed"]} >= kinds
    assert (tmp_path / "out_plain" / neural_job.WORKER_FILE).exists()


def test_a_checkpoint_of_another_identity_is_not_used(arrays, tmp_path):
    budget = {"trials": 1, "epochs_trial": 2, "epochs_final": 2, "patience": 10, "seeds": [42]}
    ckpt = tmp_path / "ckpt"
    first = job_files(tmp_path / "job_1", arrays, "bilstm", budget, ckpt)
    assert worker(first, tmp_path / "out_1", fault="unit:indirect_seed42").returncode == 86
    other = job_files(tmp_path / "job_2", arrays, "bilstm", budget, ckpt)
    job = json.loads((other / neural_job.JOB_FILE).read_text("utf-8"))
    job["resume"]["identity"]["fingerprint"] = "another"
    neural_job.write_json(other / neural_job.JOB_FILE, job)
    proc = worker(other, tmp_path / "out_2")
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    record = json.loads((tmp_path / "out_2" / neural_job.WORKER_FILE).read_text("utf-8"))
    assert record["resumed"] == []
    assert list(tmp_path.glob("ckpt.stale-*"))
