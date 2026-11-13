"""Training on the GPU, shared by BiLSTM-Opt and the TFT models (docs/stage9_plan.md §18.3-§18.4).
Runs in `solar-gpu` only (it imports torch).

- Determinism (X32): seeds for Python, NumPy and torch, `torch.use_deterministic_algorithms`,
  cuDNN deterministic without benchmark, FP32, no data-loader workers. `CUBLAS_WORKSPACE_CONFIG`
  is set by the runner before the process starts.
- The loss (X31): the quantile (pinball) loss as a mean over the steps of the W3 loss mask only;
  a masked target is set to 0 and never reaches the loss.
- One fit: Adam, the step's gradient clipping, an epoch over the training windows in a seeded
  random order; after every epoch the loss on the tuning window's loss-mask steps (the
  early-stopping criterion, §18.4) and the objective of §14.1; the weights of the best epoch are
  restored at the end.
"""

import copy
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from solarpv.models.neural_job import MAX_STEPS, PREDICT_BATCH, JobError


def setup_determinism(seed: int | None = None) -> dict:
    """Global settings of the worker process (X32); returns what was set."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise JobError("CUBLAS_WORKSPACE_CONFIG must be set before the worker starts (X32)")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if seed is not None:
        seed_everything(seed)
    return {"use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "tf32": False,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "precision": "float32", "data_loader_workers": 0}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Data:
    """The windows of one scope on the device. `y` has 0 where the loss mask is off (X31)."""

    enc_x: torch.Tensor  # [n, L, C_enc]
    dec_x: torch.Tensor  # [n, MAX_STEPS, C_dec]
    y: torch.Tensor  # [n, MAX_STEPS], model units
    mask: torch.Tensor  # [n, MAX_STEPS] bool, the loss mask
    lengths: torch.Tensor  # [n] long: decoder steps that exist (a prefix, D-25)
    n_unknown: int  # encoder-only channels (the measured series and their flags)

    @property
    def n(self) -> int:
        return int(self.enc_x.shape[0])

    @classmethod
    def from_arrays(cls, arrays: dict, scope: str, n_unknown: int, device) -> "Data":
        enc_x = arrays[f"{scope}_enc_x"].astype(np.float32)
        dec_x = arrays[f"{scope}_dec_x"].astype(np.float32)
        y = arrays[f"{scope}_y"].astype(np.float32)
        mask = arrays[f"{scope}_loss_mask"].astype(bool)
        exists = arrays[f"{scope}_dec_exists"].astype(bool)
        if y.shape != (enc_x.shape[0], MAX_STEPS) or dec_x.shape[1] != MAX_STEPS:
            raise JobError(f"{scope}: unexpected shapes {enc_x.shape} {dec_x.shape} {y.shape}")
        if (mask & ~exists).any():
            raise JobError(f"{scope}: a loss step outside the decoder")
        if not np.isfinite(y[mask]).all():
            raise JobError(f"{scope}: a loss step without a finite target")
        if not (np.isfinite(enc_x).all() and np.isfinite(dec_x).all()):
            raise JobError(f"{scope}: non-finite inputs")
        lengths = exists.sum(axis=1)
        prefix = np.arange(MAX_STEPS)[None, :] < lengths[:, None]
        if not np.array_equal(prefix, exists) or (lengths < 1).any():
            raise JobError(f"{scope}: the existing decoder steps are not a non-empty prefix")
        y = np.where(mask, y, 0.0).astype(np.float32)
        t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
        return cls(t(enc_x), t(dec_x), t(y), t(mask), t(lengths.astype(np.int64)), int(n_unknown))


def pinball(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, quantiles: torch.Tensor
            ) -> tuple[torch.Tensor, int]:
    """The mean quantile loss over the masked steps and the quantiles (X31), and the number of
    masked steps. pred [B, T, Q], y and mask [B, T]."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    err = y.unsqueeze(-1) - pred
    loss = torch.maximum(quantiles * err, (quantiles - 1.0) * err) * m
    count = int(mask.sum())
    return loss.sum() / max(count * len(quantiles), 1), count


@torch.no_grad()
def predict(model, forward, data: Data, batch: int = PREDICT_BATCH) -> np.ndarray:
    """[n, MAX_STEPS, Q] in model units, in the order of the dataset's origins."""
    model.eval()
    parts = []
    for start in range(0, data.n, batch):
        idx = torch.arange(start, min(start + batch, data.n), device=data.y.device)
        parts.append(forward(model, data, idx).float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def masked_loss_np(pred: np.ndarray, data: Data, quantiles) -> float:
    q = np.asarray(quantiles, dtype=np.float64)
    y = data.y.cpu().numpy().astype(np.float64)
    m = data.mask.cpu().numpy()
    err = y[..., None] - pred.astype(np.float64)
    loss = np.maximum(q * err, (q - 1.0) * err)[m]
    return float(loss.mean()) if loss.size else float("nan")


@dataclass
class FitResult:
    state: dict
    best_epoch: int
    epochs_run: int
    best_val_loss: float
    train_loss_at_best: float
    objective_at_best: dict
    stopped_early: bool
    nan_loss: bool
    seconds: float
    gpu_peak_mb: float
    n_parameters: int
    curve: list = field(default_factory=list)


def rng_state() -> dict:
    """Every random state a fit draws from: Python, NumPy and torch (CPU and every CUDA device)."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state: dict, cuda: bool = True) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if cuda and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def has_interlayer_dropout(model) -> bool:
    """A multi-layer LSTM with dropout between its layers: cuDNN keeps a dropout state that torch's
    random state does not hold (§23.2)."""
    return any(isinstance(m, torch.nn.RNNBase) and m.num_layers > 1 and m.dropout > 0
               for m in model.modules())


def advance_cudnn_dropout(model, forward, train: Data, epochs_done: int, batch_size: int,
                          seed: int) -> None:
    """Bring a freshly built model to where an uninterrupted fit is after `epochs_done` epochs, in
    the one thing that torch's random state does not carry (measured, T-117): the dropout state
    that cuDNN keeps between the layers of a multi-layer LSTM. It is seeded from the CUDA
    generator at the first training pass after the generator is set, and it advances with every
    training pass by an amount that depends on the shapes only. The shapes depend on the rows of
    a batch (the TFT cuts its decoder to the longest of them, R-G1-27), so the passes are those of
    the real epochs: the same permutations, drawn from the generator of the fit, the same batches,
    on the initial weights. The CUDA generator advances exactly as it did (the head's dropout) and
    is NOT restored afterwards, because setting it would reseed cuDNN's state. The weights and the
    optimiser are loaded after this."""
    model.train()
    gen = torch.Generator().manual_seed(int(seed))
    for _ in range(int(epochs_done)):
        order = torch.randperm(train.n, generator=gen).to(train.y.device)
        for start in range(0, train.n, int(batch_size)):
            forward(model, train, order[start:start + int(batch_size)])


def fit(model, forward, train: Data, tune: Data, *, epochs: int, patience: int | None, lr: float,
        batch_size: int, clip: float | None, quantiles, seed: int, objective_fn,
        point_index: int, on_epoch=None, resume: dict | None = None,
        on_checkpoint=None, checkpoint_every_s: float = 0.0) -> FitResult:
    """One training run (§18.4). The model is built by the caller after `seed_everything(seed)`.

    `on_checkpoint(state)` is called at the end of an epoch, when at least `checkpoint_every_s`
    seconds passed since the last call (0: after every epoch), with everything the next epoch
    needs (§23): the model, the optimiser, the order generator, every random state, the best
    epoch's weights, the curve and the seconds so far. `resume` is such a state: the fit continues
    with the epoch after it, in the same random streams, and ends as an uninterrupted fit would."""
    device = train.y.device
    q = torch.as_tensor(np.asarray(quantiles, dtype=np.float32), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_val, best_epoch, best_state, best_train, best_obj = math.inf, 0, None, math.nan, {}
    curve, stopped, nan_loss, done = [], False, False, False
    elapsed_before = 0.0
    epoch = 0
    if resume is not None:
        left = not resume["done"] and int(resume["epoch"]) < int(epochs)  # epochs still to train
        interlayer = left and has_interlayer_dropout(model)
        if interlayer:
            advance_cudnn_dropout(model, forward, train, int(resume["epoch"]), batch_size, seed)
        model.load_state_dict(resume["model"])
        opt.load_state_dict(copy.deepcopy(resume["opt"]))  # Adam updates its counters in place
        gen.set_state(resume["gen"])
        set_rng_state(resume["rng"], cuda=not interlayer)
        best_val, best_epoch = resume["best_val"], resume["best_epoch"]
        best_state, best_train, best_obj = (copy.deepcopy(resume["best_state"]),
                                            resume["best_train"], copy.deepcopy(resume["best_obj"]))
        curve = copy.deepcopy(resume["curve"])
        stopped, nan_loss = resume["stopped"], resume["nan_loss"]
        done, epoch, elapsed_before = resume["done"], int(resume["epoch"]), resume["elapsed_s"]
    t_fit = time.perf_counter() - elapsed_before
    last_checkpoint = time.perf_counter()
    first = epoch + 1
    for epoch in range(first, int(epochs) + 1) if not done else range(0):
        t0 = time.perf_counter()
        model.train()
        order = torch.randperm(train.n, generator=gen).to(device)
        total, count = 0.0, 0
        for start in range(0, train.n, int(batch_size)):
            idx = order[start:start + int(batch_size)]
            pred = forward(model, train, idx)
            loss, m = pinball(pred, train.y[idx], train.mask[idx], q)
            if m == 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            opt.step()
            total += float(loss.detach()) * m
            count += m
        train_loss = total / max(count, 1)
        tune_pred = predict(model, forward, tune)
        val_loss = masked_loss_np(tune_pred, tune, quantiles)
        obj = objective_fn(tune_pred[..., point_index])
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_nrmse_kw": obj["value_kw"], "lr": lr,
               "seconds": round(time.perf_counter() - t0, 3),
               "gpu_peak_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                               if device.type == "cuda" else 0.0)}
        curve.append(row)
        if on_epoch is not None:
            on_epoch(row)
        stop = False
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            nan_loss = True
            stop = True
        elif val_loss < best_val:
            best_val, best_epoch, best_train, best_obj = val_loss, epoch, train_loss, obj
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        elif patience and epoch - best_epoch >= int(patience):
            stopped = True
            stop = True
        if on_checkpoint is not None and (time.perf_counter() - last_checkpoint
                                          >= checkpoint_every_s):  # §23, R-G1-33
            last_checkpoint = time.perf_counter()
            on_checkpoint({
                "epoch": epoch, "done": stop, "model": model.state_dict(),
                "opt": opt.state_dict(), "gen": gen.get_state(), "rng": rng_state(),
                "best_val": best_val, "best_epoch": best_epoch, "best_state": best_state,
                "best_train": best_train, "best_obj": best_obj, "curve": curve,
                "stopped": stopped, "nan_loss": nan_loss,
                "elapsed_s": time.perf_counter() - t_fit})
        if stop:
            break
    if best_state is None:
        raise JobError("no finite validation loss in any epoch")
    model.load_state_dict(best_state)
    return FitResult(
        state=best_state, best_epoch=best_epoch, epochs_run=epoch, best_val_loss=best_val,
        train_loss_at_best=best_train, objective_at_best=best_obj, stopped_early=stopped,
        nan_loss=nan_loss, seconds=round(time.perf_counter() - t_fit, 2),
        gpu_peak_mb=curve[-1]["gpu_peak_mb"] if curve else 0.0,
        n_parameters=int(sum(p.numel() for p in model.parameters())), curve=curve)


def suggest(trial, space: dict) -> dict:
    out = {}
    for name, s in space.items():
        if s["type"] == "categorical":
            out[name] = trial.suggest_categorical(name, list(s["choices"]))
        elif s["type"] == "int":
            out[name] = trial.suggest_int(name, int(s["low"]), int(s["high"]), log=s["log"])
        else:
            out[name] = trial.suggest_float(name, float(s["low"]), float(s["high"]),
                                            log=s["log"])
    return out


def run_study(space: dict, n_trials: int, seed: int, storage, name: str, trial_fn,
              state: dict | None = None, callbacks=None):
    """Optuna TPE with TPESampler(seed) (§18.4: no pruning). `trial_fn(params, number)` returns
    (value, attributes); an exception in a trial stops the study (no silent failure).

    `state` (§23) is the finished trials and the sampler of an interrupted study: they are replayed
    into `storage` and the study runs the trials that are left, with the parameters an
    uninterrupted study would have drawn. `callbacks` run after every finished trial."""
    import optuna

    from solarpv.models import checkpoint

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = checkpoint.replayed_study(storage, name, seed, state)

    def objective(trial):
        value, attrs = trial_fn(suggest(trial, space), trial.number)
        for k, v in attrs.items():
            trial.set_user_attr(k, v)
        return value

    left = int(n_trials) - (len(state["trials"]) if state else 0)
    if left > 0:
        study.optimize(objective, n_trials=left, callbacks=callbacks)
    return study
