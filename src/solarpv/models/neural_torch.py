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


def fit(model, forward, train: Data, tune: Data, *, epochs: int, patience: int | None, lr: float,
        batch_size: int, clip: float | None, quantiles, seed: int, objective_fn,
        point_index: int, on_epoch=None) -> FitResult:
    """One training run (§18.4). The model is built by the caller after `seed_everything(seed)`."""
    device = train.y.device
    q = torch.as_tensor(np.asarray(quantiles, dtype=np.float32), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_val, best_epoch, best_state, best_train, best_obj = math.inf, 0, None, math.nan, {}
    curve, stopped, nan_loss = [], False, False
    t_fit = time.perf_counter()
    epoch = 0
    for epoch in range(1, int(epochs) + 1):
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
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            nan_loss = True
            break
        if val_loss < best_val:
            best_val, best_epoch, best_train, best_obj = val_loss, epoch, train_loss, obj
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        elif patience and epoch - best_epoch >= int(patience):
            stopped = True
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


def run_study(space: dict, n_trials: int, seed: int, storage, name: str, trial_fn):
    """Optuna TPE with TPESampler(seed) (§18.4: no pruning). `trial_fn(params, number)` returns
    (value, attributes); an exception in a trial stops the study (no silent failure)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(storage=storage, study_name=name, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=int(seed)))

    def objective(trial):
        value, attrs = trial_fn(suggest(trial, space), trial.number)
        for k, v in attrs.items():
            trial.set_user_attr(k, v)
        return value

    study.optimize(objective, n_trials=int(n_trials))
    return study
