"""PatchTST via neuralforecast 3.1.7: patch_len 16, stride 8, the target series only (L-19;
PROJECT_MASTER.md §6.11; docs/stage9_plan.md §18.4). Runs in the training worker of `solar-gpu`.

- Input size 168 h (X33); horizon 73 steps (D-25), the steps after the third local day dropped by
  the runner. Every other setting is the neuralforecast default, recorded from the model.
- Training: neuralforecast's own windows of one series, the target in model units (k; AC /
  3,449) from the data start to the end of the tuning window. `available_mask` is the W3 loss mask
  (the fit rows of the training scope and of the tuning window), so the loss never sees another
  hour (X31). The last `val_size` hours are the tuning window: neuralforecast's validation set
  for early stopping (the training windows end before it). In smoke mode the validation hours
  between the end of train and the tuning window carry no mask: they are in no loss.
- Normalisation (X29): `scaler_type="identity"`, no `local_scaler_type`, RevIN on (per window, the
  v6 architecture).
- Early stopping: neuralforecast stops after `patience` validation checks without improvement but
  keeps the last weights; `BestWeights` restores the weights of the best check (§18.4).
- Prediction: one series per origin, its 168 encoder hours (the W3 PatchTST windows); the output
  is matched back to the origins by series id.
- L-19, for the report: this PatchTST has `EXOGENOUS_FUTR = False`, it cannot take future inputs.
"""

import numpy as np
import pandas as pd

from solarpv.models.neural_job import FIXED, MAX_STEPS, JobError

ALIAS = "PatchTST"


def best_weights_callback(monitor: str = "ptl/val_loss"):
    import pytorch_lightning as pl

    class BestWeights(pl.Callback):
        """Keeps the weights of the best validation check (sanity checks excluded)."""

        def __init__(self):
            self.best, self.step, self.state, self.history = float("inf"), None, None, []

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            value = trainer.callback_metrics.get(monitor)
            if value is None:
                return
            value = float(value)
            self.history.append((int(trainer.global_step), value))
            if value < self.best:
                self.best, self.step = value, int(trainer.global_step)
                self.state = {k: v.detach().cpu().clone()
                              for k, v in pl_module.state_dict().items()}

    return BestWeights()


def fitted_callback(fitted, original):
    """The BestWeights instance that ran: NeuralForecast trains a copy of the model, with its own
    copy of the callback."""
    found = [c for c in fitted.trainer_kwargs.get("callbacks", [])
             if type(c).__name__ == "BestWeights"]
    if len(found) != 1:
        raise JobError(f"PatchTST: {len(found)} BestWeights callbacks on the fitted model")
    return found[0] if found[0].history or original.state is None else original


def build(budget: dict, seed: int, callback):
    from neuralforecast.models import PatchTST

    f = FIXED["patchtst"]
    return PatchTST(
        h=f["h"], input_size=f["input_size"], patch_len=f["patch_len"], stride=f["stride"],
        scaler_type=f["scaler_type"], revin=f["revin"], max_steps=int(budget["max_steps"]),
        val_check_steps=int(budget["val_check_steps"]),
        early_stop_patience_steps=int(budget["patience"]), random_seed=int(seed), alias=ALIAS,
        accelerator="gpu", devices=1, deterministic=True, enable_progress_bar=False,
        logger=False, enable_model_summary=False, num_sanity_val_steps=0,
        callbacks=[callback])


def training_frame(arrays: dict) -> tuple[pd.DataFrame, int]:
    y = arrays["series_y"].astype(np.float64)
    mask = arrays["series_mask"].astype(np.float64)
    val_size = int(arrays["series_val_size"])
    if not np.isfinite(y).all():
        raise JobError("PatchTST: the training series has a non-finite value")
    if val_size < MAX_STEPS or val_size >= len(y) - FIXED["patchtst"]["input_size"]:
        raise JobError(f"PatchTST: validation size {val_size} does not fit the series")
    if not mask[-val_size:].any() or not mask[:-val_size].any():
        raise JobError("PatchTST: no loss step in the training or the validation part")
    df = pd.DataFrame({"unique_id": "series", "ds": np.arange(len(y), dtype=np.int64), "y": y,
                       "available_mask": mask})
    return df, val_size


def history_frame(enc_x: np.ndarray) -> pd.DataFrame:
    n, length = enc_x.shape[0], enc_x.shape[1]
    return pd.DataFrame({
        "unique_id": np.repeat([f"o{i:05d}" for i in range(n)], length),
        "ds": np.tile(np.arange(length, dtype=np.int64), n),
        "y": enc_x[..., 0].astype(np.float64).ravel(),
        "available_mask": 1.0})


def predict_origins(nf, enc_x: np.ndarray) -> np.ndarray:
    """[n, MAX_STEPS, 1]: the forecast of every origin from its encoder hours."""
    n, length = enc_x.shape[0], enc_x.shape[1]
    if length != FIXED["patchtst"]["input_size"]:
        raise JobError(f"PatchTST windows of {length} hours, not the input size")
    out = nf.predict(df=history_frame(enc_x)).reset_index()
    out["origin"] = out["unique_id"].str[1:].astype(int)
    out["step"] = out["ds"].astype(int) - length
    if len(out) != n * MAX_STEPS or out["step"].min() != 0 or out["step"].max() != MAX_STEPS - 1:
        raise JobError(f"PatchTST returned {len(out)} rows, not {n} x {MAX_STEPS}")
    grid = np.full((n, MAX_STEPS), np.nan)
    grid[out["origin"].to_numpy(), out["step"].to_numpy()] = out[ALIAS].to_numpy(dtype=float)
    if not np.isfinite(grid).all():
        raise JobError("PatchTST: an origin or a step without a forecast")
    return grid[..., None].astype(np.float32)


def fit_predict(arrays: dict, scopes: list[str], budget: dict, seed: int) -> dict:
    """Train on the series, restore the best validation check, predict every scope's origins."""
    from neuralforecast import NeuralForecast

    df, val_size = training_frame(arrays)
    cb = best_weights_callback()
    model = build(budget, seed, cb)
    if model.EXOGENOUS_FUTR:
        raise JobError("PatchTST must stay univariate (L-19)")
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df=df, val_size=val_size)
    fitted = nf.models[0]
    cb = fitted_callback(fitted, cb)
    if cb.state is None:
        raise JobError("PatchTST: no validation check was recorded")
    fitted.load_state_dict(cb.state)
    forecasts = {f"{s}_q": predict_origins(nf, arrays[f"{s}_enc_x"]) for s in scopes}
    hp = {k: v for k, v in dict(fitted.hparams).items()
          if isinstance(v, int | float | str | bool | type(None))}
    train_traj = [(int(s), float(v)) for s, v in getattr(fitted, "train_trajectories", [])]
    return {"forecasts": forecasts, "best_step": cb.step, "best_val_loss": cb.best,
            "val_checks": cb.history, "train_trajectory": train_traj,
            "steps_run": int(train_traj[-1][0]) if train_traj else None,
            "hparams": hp, "exogenous_futr": bool(model.EXOGENOUS_FUTR),
            "n_parameters": int(sum(p.numel() for p in fitted.parameters())),
            "state": cb.state, "training_rows": int(len(df)), "val_size": val_size, "nf": nf}
