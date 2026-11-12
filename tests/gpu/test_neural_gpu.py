"""The neural worker on synthetic windows, on the GPU (docs/stage9_plan.md §18.7).

T-77  the adapters keep the origin and step of every forecast: the TFT forecast of an origin does
      not depend on the batch it is in, also with 71- and 72-step decoders; PatchTST's forecasts
      are matched back to their origins (M14, M15)
T-78  X29, X31, M16, M17: the TFT has no static or categorical input, identity output transformer,
      and never reads the unknown reals on decoder steps; the masked loss equals a hand
      computation; changed targets on masked steps leave the trained network identical;
      PatchTST's trained weights do not depend on the validation values
T-79  X25: the worker refuses a job whose arrays changed; a worker run writes every output with
      its hash
T-81  X32: two fits with the same seed give identical networks and forecasts
M22   the TFT attention has the L encoder positions of its own encoder
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from solarpv.models import lstm_sota, neural_job, neural_torch, tft_model
from solarpv.models.neural_job import MAX_STEPS, QUANTILES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
DEVICE = torch.device("cuda")
L, N_UNKNOWN, N_KNOWN = 24, 4, 3
ENC = [f"u{i}" for i in range(N_UNKNOWN)] + [f"k{i}" for i in range(N_KNOWN)]
DEC = ENC[N_UNKNOWN:]
META = {"enc_channels": ENC, "dec_channels": DEC, "n_unknown": N_UNKNOWN, "encoder_hours": L,
        "target_scale": 1.0}
BILSTM = {"hidden": 16, "n_layers": 2, "dropout": 0.1, "lr": 3e-3, "batch_size": 16}
TFT = {"hidden_size": 16, "attention_head_size": 2, "hidden_continuous_size": 8, "dropout": 0.1,
       "learning_rate": 3e-3, "lstm_layers": 1}

neural_torch.setup_determinism()


def scope_arrays(key: str, n: int, seed: int, short: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    known = rng.normal(size=(n, L + MAX_STEPS, N_KNOWN)).astype(np.float32)
    unknown = rng.normal(size=(n, L, N_UNKNOWN)).astype(np.float32)
    y = (0.5 + 0.3 * known[:, L:, 0] - 0.2 * known[:, L:, 1]).astype(np.float32)
    lengths = np.full(n, MAX_STEPS)
    if short:
        lengths[-2:] = (71, 72)
    exists = np.arange(MAX_STEPS)[None, :] < lengths[:, None]
    mask = exists & (rng.random((n, MAX_STEPS)) < 0.6)
    y = np.where(exists, y, np.nan).astype(np.float32)
    dec_x = np.where(exists[..., None], known[:, L:], 0.0).astype(np.float32)
    horizon = np.where(exists, 1 + np.arange(MAX_STEPS)[None, :] // 24, 0).clip(0, 3)
    day = exists & (rng.random((n, MAX_STEPS)) < 0.5)
    return {f"{key}_enc_x": np.concatenate([unknown, known[:, :L]], axis=2),
            f"{key}_dec_x": dec_x, f"{key}_y": y, f"{key}_loss_mask": mask,
            f"{key}_dec_exists": exists, f"{key}_dec_keep": exists,
            f"{key}_dec_horizon": horizon.astype(np.int8), f"{key}_score_mask": day,
            f"{key}_theo_kw": np.where(exists, 2000.0, 0.0), f"{key}_truth_kw":
            np.where(exists, np.nan_to_num(y) * 2000.0, np.nan),
            f"{key}_k_true": np.where(exists, y, np.nan)}


@pytest.fixture(scope="module")
def arrays():
    return {**scope_arrays("train", 48, 0), **scope_arrays("val", 12, 1, short=True)}


def data_of(arrays, key):
    return neural_torch.Data.from_arrays(arrays, key, N_UNKNOWN, DEVICE)


def objective_fn(arrays):
    return lambda p: neural_job.objective("indirect", p, arrays, "val", 1.0, 0.788)


def train(module, params, arrays, seed=7, epochs=3, patience=None):
    neural_torch.seed_everything(seed)
    if module is lstm_sota:
        model = module.build_model(params, META, 3).to(DEVICE)
        q, s = QUANTILES["bilstm"], module.train_settings(params, neural_job.FIXED["bilstm"])
    else:
        model = module.build_model(params, META).to(DEVICE)
        q, s = QUANTILES["tft"], module.train_settings(params, neural_job.FIXED["tft"])
    res = neural_torch.fit(model, module.forward, data_of(arrays, "train"), data_of(arrays, "val"),
                           epochs=epochs, patience=patience, lr=s["lr"],
                           batch_size=s["batch_size"], clip=s["clip"], quantiles=q, seed=seed,
                           objective_fn=objective_fn(arrays), point_index=q.index(0.5))
    return model, res


def same_state(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


# ── the loss (X31) ──────────────────────────────────────────────────────────


def test_pinball_is_the_mean_over_the_masked_steps():
    rng = np.random.default_rng(3)
    pred = rng.normal(size=(5, MAX_STEPS, 3))
    y = rng.normal(size=(5, MAX_STEPS))
    mask = rng.random((5, MAX_STEPS)) < 0.4
    q = np.array([0.1, 0.5, 0.9])
    err = y[..., None] - pred
    hand = np.maximum(q * err, (q - 1) * err)[mask].mean()
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    loss, count = neural_torch.pinball(t(pred), t(y), torch.as_tensor(mask), t(q))
    assert count == mask.sum() and abs(float(loss) - hand) < 1e-5


def test_data_refuses_broken_windows(arrays):
    outside = copy.deepcopy(arrays)
    outside["val_loss_mask"][-2, -1] = True  # step 73 of a 71-step decoder
    with pytest.raises(neural_job.JobError, match="outside the decoder"):
        data_of(outside, "val")
    gap = copy.deepcopy(arrays)
    gap["val_dec_exists"][0, 5] = False
    gap["val_loss_mask"][0, 5] = False
    with pytest.raises(neural_job.JobError, match="prefix"):
        data_of(gap, "val")
    nan = copy.deepcopy(arrays)
    i, j = np.argwhere(nan["train_loss_mask"])[0]
    nan["train_y"][i, j] = np.nan
    with pytest.raises(neural_job.JobError, match="finite target"):
        data_of(nan, "train")


@pytest.mark.parametrize("module,params", [(lstm_sota, BILSTM), (tft_model, TFT)])
def test_masked_targets_do_not_change_the_network(arrays, module, params):
    other = copy.deepcopy(arrays)
    off = ~other["train_loss_mask"] & other["train_dec_exists"]
    other["train_y"][off] = 1e6
    other["train_y"][~other["train_dec_exists"]] = np.nan
    _, a = train(module, params, arrays)
    _, b = train(module, params, other)
    assert same_state(a.state, b.state)


# ── determinism (X32, T-81) ────────────────────────────────────────────────


@pytest.mark.parametrize("module,params", [(lstm_sota, BILSTM), (tft_model, TFT)])
def test_the_same_seed_gives_the_same_network(arrays, module, params):
    m1, a = train(module, params, arrays, seed=11)
    m2, b = train(module, params, arrays, seed=11)
    assert same_state(a.state, b.state)
    assert [r["val_loss"] for r in a.curve] == [r["val_loss"] for r in b.curve]
    assert [r["train_loss"] for r in a.curve] == [r["train_loss"] for r in b.curve]
    v = data_of(arrays, "val")
    p1 = neural_torch.predict(m1, module.forward, v)
    p2 = neural_torch.predict(m2, module.forward, v)
    assert np.array_equal(p1, p2)
    _, c = train(module, params, arrays, seed=12)
    assert not same_state(a.state, c.state)


def test_early_stopping_restores_the_best_epoch(arrays):
    model, res = train(lstm_sota, BILSTM, arrays, epochs=6, patience=2)
    best = min(r["val_loss"] for r in res.curve)
    assert res.best_val_loss == best
    assert res.curve[res.best_epoch - 1]["val_loss"] == best
    assert all(torch.equal(model.state_dict()[k].cpu(), res.state[k]) for k in res.state)


# ── the TFT adapter (X29, X34, M16, M17, M22, T-77) ─────────────────────────


@pytest.mark.parametrize("n_in,n_out", [(1, 8), (1, 16), (3, 16), (16, 64), (7, 5), (8, 8)])
def test_the_interpolation_matrix_equals_the_library_interpolation(n_in, n_out):
    """X34: the same values and gradients as F.interpolate(linear, align_corners=True)."""
    import torch.nn.functional as F

    x = torch.randn(50, n_in, dtype=torch.float64, requires_grad=True)
    ref = F.interpolate(x.unsqueeze(1), n_out, mode="linear", align_corners=True).squeeze(1)
    ours = x @ tft_model.linear_map(n_in, n_out, x.device, x.dtype)
    assert torch.allclose(ref, ours, atol=1e-12)
    g = torch.randn_like(ref)
    (gr,) = torch.autograd.grad(ref, x, g)
    (go,) = torch.autograd.grad(ours, x, g)
    assert torch.allclose(gr, go, atol=1e-12)


def test_tft_inputs_and_output_transformer():
    model = tft_model.build_model(TFT, META)
    hp = model.hparams
    assert hp.static_reals == [] and hp.static_categoricals == [] and hp.x_categoricals == []
    assert hp.x_reals == ENC and hp.time_varying_reals_encoder == ENC
    assert hp.time_varying_reals_decoder == DEC
    assert model.output_transformer.method == "identity"
    assert model.output_transformer.center is False
    assert model.decoder_variables == DEC
    big = tft_model.build_model({**TFT, "hidden_continuous_size": 60}, META)
    assert big.hparams.hidden_continuous_size == 16  # min(value, hidden_size), §18.4


def test_tft_never_reads_unknown_reals_on_decoder_steps(arrays):
    model, _ = train(tft_model, TFT, arrays, epochs=1)
    model.eval()
    v = data_of(arrays, "val")
    idx = torch.arange(v.n, device=DEVICE)
    x = tft_model.network_input(v, idx)
    with torch.no_grad():
        base = model(x)["prediction"]
        x2 = dict(x)
        dec = x["decoder_cont"].clone()
        dec[..., :N_UNKNOWN] = torch.randn_like(dec[..., :N_UNKNOWN]) * 100
        x2["decoder_cont"] = dec
        assert torch.equal(base, model(x2)["prediction"])


def test_tft_forecast_of_an_origin_does_not_depend_on_its_batch(arrays):
    model, _ = train(tft_model, TFT, arrays, epochs=1)
    v = data_of(arrays, "val")
    together = neural_torch.predict(model, tft_model.forward, v, batch=64)
    lengths = v.lengths.cpu().numpy()
    for i in (0, v.n - 2, v.n - 1):  # 73, 71 and 72 steps
        alone = neural_torch.predict(model, tft_model.forward, v, batch=1)[i]
        n = lengths[i]
        assert np.allclose(alone[:n], together[i, :n], atol=1e-5)
        assert (alone[n:] == 0).all()


def test_tft_attention_covers_its_own_encoder(arrays):
    model, _ = train(tft_model, TFT, arrays, epochs=1)
    out = tft_model.interpretation(model, data_of(arrays, "val"))
    assert out["attention"].shape == (12, MAX_STEPS, L)
    assert len(out["encoder_importance"]) == len(ENC)
    assert len(out["decoder_importance"]) == len(DEC)


def test_bilstm_decoder_is_causal(arrays):
    model, _ = train(lstm_sota, BILSTM, arrays, epochs=1)
    model.eval()
    v = data_of(arrays, "val")
    with torch.no_grad():
        a = model(v.enc_x, v.dec_x)
        dec = v.dec_x.clone()
        dec[:, 40:] += 5.0
        b = model(v.enc_x, dec)
    assert torch.equal(a[:, :40], b[:, :40]) and not torch.equal(a[:, 40:], b[:, 40:])


# ── PatchTST (T-77, T-78) ──────────────────────────────────────────────────


def patchtst_arrays(seed=0, n_hours=1200, val=240):
    from solarpv.models.neural_job import FIXED

    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    y = (np.clip(np.sin(2 * np.pi * (t % 24) / 24), 0, None)
         * (0.8 + 0.1 * rng.normal(size=n_hours)))
    mask = rng.random(n_hours) < 0.8
    length = FIXED["patchtst"]["input_size"]
    starts = n_hours - val - length + 24 * np.arange(5)  # five origins, a day apart
    enc = np.stack([y[s:s + length] for s in starts])[..., None].astype(np.float32)
    return {"series_y": y.astype(np.float32), "series_mask": mask,
            "series_val_size": np.int64(val), "val_enc_x": enc}, starts


def test_patchtst_maps_every_forecast_to_its_origin():
    from solarpv.models import patchtst_model

    arrays, _ = patchtst_arrays()
    out = patchtst_model.fit_predict(arrays, ["val"], {"max_steps": 10, "val_check_steps": 5,
                                                       "patience": 10}, seed=3)
    q = out["forecasts"]["val_q"]
    assert q.shape == (5, MAX_STEPS, 1) and np.isfinite(q).all()
    assert out["exogenous_futr"] is False  # L-19
    assert out["best_step"] is not None and out["state"] is not None
    # the origins in reverse order give the forecasts in reverse order; one origin alone its row
    enc = arrays["val_enc_x"]
    rev = patchtst_model.predict_origins(out["nf"], enc[::-1].copy())
    assert np.allclose(rev[::-1], q, atol=1e-5)
    alone = patchtst_model.predict_origins(out["nf"], enc[2:3].copy())
    assert np.allclose(alone[0], q[2], atol=1e-5)


def test_patchtst_weights_do_not_depend_on_validation_values():
    from neuralforecast import NeuralForecast

    from solarpv.models import patchtst_model

    arrays, _ = patchtst_arrays()
    states = []
    for scale in (1.0, 5.0):
        a = dict(arrays)
        y = a["series_y"].copy()
        y[-int(a["series_val_size"]):] *= scale
        a["series_y"] = y
        df, val_size = patchtst_model.training_frame(a)
        cb = patchtst_model.best_weights_callback()
        model = patchtst_model.build({"max_steps": 8, "val_check_steps": 4, "patience": -1},
                                     3, cb)
        nf = NeuralForecast(models=[model], freq=1)
        nf.fit(df=df, val_size=val_size)
        states.append({k: v.detach().cpu() for k, v in nf.models[0].state_dict().items()})
        ran = patchtst_model.fitted_callback(nf.models[0], cb)
        assert len(ran.history) == 2  # validation checks at steps 4 and 8
    assert same_state(*states)


# ── the worker (X25, T-79) ─────────────────────────────────────────────────


def write_job(folder: Path, arrays: dict, step="bilstm") -> dict:
    folder.mkdir()
    sha = neural_job.write_arrays(folder / "data_indirect.npz", arrays)
    budget = {"trials": 1, "epochs_trial": 2, "epochs_final": 2, "patience": 10, "seeds": [42]}
    job = {"version": neural_job.JOB_VERSION, "step": step, "model": "lstm_sota",
           "family": "bilstm", "mode": "smoke", "run_id": "test", "targets": ["indirect"],
           "seeds": [42], "budget": budget, "scopes": ["train", "val"], "tune_scope": "val",
           "mean_k": 0.788, "meta": {"indirect": META},
           "files": {"indirect": {"path": "data_indirect.npz", "sha256": sha}}}
    neural_job.write_json(folder / neural_job.JOB_FILE, job)
    return job


def run_worker(job_dir: Path, out: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "solarpv.models.neural_worker", str(job_dir),
                           str(out)], capture_output=True, text=True, env=env, timeout=900)


def test_the_worker_writes_every_output_with_its_hash(arrays, tmp_path):
    write_job(tmp_path / "job", arrays)
    proc = run_worker(tmp_path / "job", tmp_path / "worker")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    worker = json.loads((tmp_path / "worker" / neural_job.WORKER_FILE).read_text("utf-8"))
    for name, digest in worker["outputs"].items():
        assert neural_job.sha256_file(tmp_path / "worker" / name) == digest
    fc = neural_job.read_arrays(tmp_path / "worker" / neural_job.forecast_name("indirect", 42))
    assert fc["train_q"].shape == (48, MAX_STEPS, 3) and fc["val_q"].shape == (12, MAX_STEPS, 3)
    assert worker["determinism"]["use_deterministic_algorithms"] is True
    # X55: the worker records the code it imported (this checkout's src)
    src = Path(__file__).resolve().parents[2] / "src"
    assert Path(worker["solarpv_file"]).is_relative_to(src)
    rec = worker["records"]["indirect"]["42"]
    assert rec["study"]["n_trials"] == 1 and rec["final"]["epochs_run"] == 2


def test_the_worker_refuses_changed_arrays(arrays, tmp_path):
    write_job(tmp_path / "job", arrays)
    changed = dict(arrays)
    changed["val_y"] = arrays["val_y"] + 1
    np.savez_compressed(tmp_path / "job" / "data_indirect.npz", **changed)
    proc = run_worker(tmp_path / "job", tmp_path / "worker")
    assert proc.returncode != 0 and "SHA-256" in proc.stderr
    assert not (tmp_path / "worker" / neural_job.WORKER_FILE).exists()
