"""The job of a neural step: what the runner (env `solar`) hands to the training worker (env
`solar-gpu`), and the settings both sides share (docs/stage9_plan.md §18.3-§18.5).

NumPy only: imported on both sides. The runner builds the windows (W3), writes them with this
module as compressed arrays with their SHA-256 and a `job.json`, and starts
`python -m solarpv.models.neural_worker <job folder>` in `solar-gpu` (X25). The worker checks every
hash, trains, and writes its outputs with their hashes in `worker.json`; the runner checks them.
No timestamp crosses: the worker sees integer positions only (M13).

The settings of §18.4 (registered before any result) live here, in one place.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from solarpv.evaluation.units import nrmse_k, nrmse_kw

JOB_VERSION = 1
JOB_FILE = "job.json"
WORKER_FILE = "worker.json"
MAX_STEPS = 73  # the decoder of three local days, padded (D-25); also PatchTST's h

# step -> the model name of the evaluation layer (X28: evaluate.MODEL_LABELS)
STEP_MODEL = {"bilstm": "lstm_sota", "tft96": "tft_96", "tft168": "tft_168", "tftbase": "tft",
              "patchtst": "patchtst"}
STEP_FAMILY = {"bilstm": "bilstm", "tft96": "tft", "tft168": "tft", "tftbase": "tft",
               "patchtst": "patchtst"}
NEURAL_STEPS = tuple(STEP_MODEL)

# §18.4: the training losses. BiLSTM: v6 §5.2. TFT: pytorch-forecasting's default QuantileLoss.
# PatchTST: neuralforecast's default loss (MAE): its output is the point forecast.
QUANTILES = {"bilstm": (0.1, 0.5, 0.9),
             "tft": (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98),
             "patchtst": (0.5,)}
POINT_QUANTILE = 0.5

# §18.4: the search spaces. BiLSTM: v6 §5.2. TFT: hidden_size of v6; heads, continuous size and
# dropout from pytorch-forecasting's own tuning function; learning rate and LSTM layers the BiLSTM
# ranges of v6. hidden_continuous_size is sampled in 8..64 and used as min(value, hidden_size).
SEARCH_SPACES = {
    "bilstm": {
        "hidden": {"type": "categorical", "choices": [64, 128, 256]},
        "n_layers": {"type": "int", "low": 1, "high": 3, "log": False},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4, "log": False},
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    },
    "tft": {
        "hidden_size": {"type": "categorical", "choices": [16, 32, 64]},
        "attention_head_size": {"type": "int", "low": 1, "high": 4, "log": False},
        "hidden_continuous_size": {"type": "int", "low": 8, "high": 64, "log": False},
        "dropout": {"type": "float", "low": 0.1, "high": 0.3, "log": False},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "lstm_layers": {"type": "int", "low": 1, "high": 3, "log": False},
    },
}
# §18.4: settings that are not searched
FIXED = {
    "bilstm": {"grad_clip_norm": 1.0, "optimizer": "Adam"},
    "tft": {"batch_size": 64, "grad_clip_norm": 0.1, "optimizer": "Adam"},
    # M21: v6's three values; the rest the defaults of pytorch-forecasting 1.7.0's TFT
    "tftbase": {"hidden_size": 64, "dropout": 0.1, "attention_head_size": 4,
                "hidden_continuous_size": 8, "lstm_layers": 1, "learning_rate": 1e-3},
    "patchtst": {"input_size": 168, "h": MAX_STEPS, "patch_len": 16, "stride": 8,
                 "scaler_type": "identity", "revin": True},
}
EDGE_FRACTION = 0.02  # as XGB (§3.5)
PREDICT_BATCH = 256


class JobError(RuntimeError):
    """A job file or a worker output breaks the protocol (hash, shape, missing array)."""


# the worker's outputs, in `<step>/worker/` (the runner reads them by these names)
def forecast_name(target: str, seed: int) -> str:
    return f"forecast_{target}_seed{int(seed)}.npz"


def fit_name(target: str, seed: int) -> str:
    return f"fit_{target}_seed{int(seed)}.json"


def trials_name(target: str, seed: int) -> str:
    return f"trials_{target}_seed{int(seed)}.csv"


def curves_name(target: str, seed: int) -> str:
    return f"curves_{target}_seed{int(seed)}.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Compressed arrays; returns their SHA-256. No object arrays (nothing to unpickle)."""
    for name, a in arrays.items():
        if np.asarray(a).dtype == object:
            raise JobError(f"{name}: object arrays are not allowed")
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return sha256_file(path)


def arrays_digest(arrays: dict[str, np.ndarray]) -> str:
    """A digest of the contents of the arrays (names, dtypes, shapes and values), not of the bytes
    of the zip file that holds them: the file carries a time stamp, the contents do not (§23)."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype}|{a.shape}|".encode())
        h.update(a.reshape(-1).view(np.uint8) if a.size else b"")
    return h.hexdigest()


def read_arrays(path: Path, sha256: str | None = None) -> dict[str, np.ndarray]:
    """The arrays of `path`, after its hash is checked (X25: a changed file is refused)."""
    path = Path(path)
    if sha256 is not None and sha256_file(path) != sha256:
        raise JobError(f"{path.name}: SHA-256 differs from the job record")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def write_json(path: Path, data: dict) -> str:
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8",
                          newline="\n")
    return sha256_file(path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


HCS = "hidden_continuous_size"


def effective(params: dict, space: dict) -> tuple[dict, dict]:
    """X45: the TFT samples `hidden_continuous_size` in 8..64 and uses min(value, hidden_size).
    The parameters with the effective value, and the space with its effective bounds
    [low, hidden_size], so that an edge is judged on what the network used."""
    if HCS not in params or HCS not in space or "hidden_size" not in params:
        return dict(params), dict(space)
    hidden = int(params["hidden_size"])
    out = {**params, HCS: min(int(params[HCS]), hidden)}
    eff = {**space, HCS: {**space[HCS], "high": hidden}}
    if eff[HCS]["high"] <= eff[HCS]["low"]:  # one possible value: never an edge result
        eff = {k: v for k, v in eff.items() if k != HCS}
    return out, eff


def edge_params(params: dict, space: dict, fraction: float = EDGE_FRACTION) -> list[str]:
    """Searched parameters on the edge of their space (§3.5, M23): within `fraction` of a bound
    in the sampling scale, or the first or last choice of a categorical."""
    edges = []
    for name, s in space.items():
        if name not in params:
            continue
        v = params[name]
        if s["type"] == "categorical":
            if v in (s["choices"][0], s["choices"][-1]):
                edges.append(name)
            continue
        v, lo, hi = float(v), float(s["low"]), float(s["high"])
        if s.get("log"):
            pos = np.log(v / lo) / np.log(hi / lo)
        else:
            pos = (v - lo) / (hi - lo)
        if pos <= fraction or pos >= 1 - fraction:
            edges.append(name)
    return edges


def objective(target: str, pred: np.ndarray, arrays: dict, scope: str, target_scale: float,
              mean_k: float) -> dict:
    """§14.1 on the scoring steps of `scope` (daytime hours of the window with AC truth): the
    nRMSE in kW of the point forecast `pred` [origins, steps] in model units, pooled over the
    three horizons (one model per target), and per horizon; for k also the k-unit value."""
    sel = arrays[f"{scope}_score_mask"].astype(bool)
    if not sel.any():
        raise JobError(f"{scope}: no scoring step")
    y = np.asarray(pred, dtype=float) * target_scale
    if not np.isfinite(y[sel]).all():
        raise JobError(f"{scope}: a scoring step has no finite forecast")
    theo, truth = arrays[f"{scope}_theo_kw"], arrays[f"{scope}_truth_kw"]
    horizon = arrays[f"{scope}_dec_horizon"]
    out = {"value_kw": nrmse_kw(target, y[sel], theo[sel], truth[sel]), "n_scored": int(sel.sum())}
    for h in (1, 2, 3):
        s = sel & (horizon == h)
        out[f"value_kw_D{h}"] = nrmse_kw(target, y[s], theo[s], truth[s])
    if target == "indirect":
        out["value_k"] = nrmse_k(y[sel], arrays[f"{scope}_k_true"][sel], mean_k)
    return out
