"""The k -> kW conversion of a forecast and the tuning objective (docs/stage9_plan.md §14.1).

One implementation (X12), in a module without pvlib or pandas (X26), so that the neural training
worker of `solar-gpu` scores its trials with the same code as the evaluation layer and the `xgb`
step. `evaluate.to_kw` and `xgb_sota.nrmse_kw` / `nrmse_k` are these functions.
"""

import numpy as np

from solarpv.config.constants import AC_CAPACITY_KW


def to_kw(target: str, y_pred, theo_kw) -> np.ndarray:
    """kW forecast: the direct forecast as is; the indirect one as clip(k * theo, 0, cap)."""
    y_pred = np.asarray(y_pred, dtype=float)
    if target == "direct":
        return y_pred
    if target != "indirect":
        raise ValueError(f"unknown target {target!r}")
    return np.clip(y_pred * np.asarray(theo_kw, dtype=float), 0.0, AC_CAPACITY_KW)


def nrmse_kw(target: str, y_pred, theo, truth_kw) -> float:
    """§14.1: the evaluation's nRMSE in kW (% of capacity), through `to_kw`."""
    err = to_kw(target, y_pred, theo) - np.asarray(truth_kw, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)) / AC_CAPACITY_KW * 100.0)


def nrmse_k(y_pred, k_true, mean_k: float) -> float:
    """RMSE of k / MEAN_DAYTIME_K x 100: the evaluation's nRMSE_pct of the indirect route."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(k_true, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)) / mean_k * 100.0)
