"""Point metrics and bootstrap confidence intervals on daytime rows (PROJECT_MASTER.md §6.9).

The bootstrap resamples circular blocks of consecutive rows of the time-ordered daytime series
(block = 24 rows, about two days of daytime hours), as in the original detectable_effect.py.
Stage 8B adds the CLIPER reference (L-13) and the CI of a mean loss differential (the original
dm_tests_extended), docs/stage8b_plan.md §3.5.
"""

import numpy as np
import pandas as pd

from solarpv.config.constants import BOOTSTRAP_BLOCK, BOOTSTRAP_REPS, BOOTSTRAP_SEED

DM_CI_REPS = 2000  # as in the original detectable_effect.py


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("non-finite values in y_true or y_pred")
    return y, p


def rmse(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def r2(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def nrmse_pct(y_true, y_pred, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return rmse(y_true, y_pred) / denominator * 100.0


def point_metrics(y_true, y_pred, denominator: float) -> dict[str, float]:
    """n, nRMSE_pct, MAE, R2, RMSE on the given rows (already daytime-filtered)."""
    y, p = _arrays(y_true, y_pred)
    return {
        "n": int(y.size),
        "nRMSE_pct": nrmse_pct(y, p, denominator),
        "MAE": mae(y, p),
        "R2": r2(y, p),
        "RMSE": rmse(y, p),
    }


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """1 - RMSE_model / RMSE_reference (reference: horizon-matched persistence)."""
    return 1.0 - rmse_model / rmse_reference


def circular_block_indices(
    n: int, block: int, reps: int, rng: np.random.Generator
) -> np.ndarray:
    """(reps, n) resampling indices built from circular blocks of `block` consecutive rows."""
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(reps, -1)[:, :n]


def bootstrap_nrmse_ci(
    y_true,
    y_pred,
    denominator: float,
    block: int = BOOTSTRAP_BLOCK,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of nRMSE from a circular block bootstrap of squared errors."""
    y, p = _arrays(y_true, y_pred)
    sq = (y - p) ** 2
    idx = circular_block_indices(sq.size, block, reps, np.random.default_rng(seed))
    stats = np.sqrt(sq[idx].mean(axis=1)) / denominator * 100.0
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_mean_ci(d, block: int = BOOTSTRAP_BLOCK, reps: int = DM_CI_REPS,
                      seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of the mean of d (a loss differential) from a circular block bootstrap.
    The original resampled non-circular block starts; circular blocks avoid the edge bias."""
    d = np.asarray(d, dtype=float)
    if d.size == 0 or not np.isfinite(d).all():
        raise ValueError("d must be a non-empty finite array")
    idx = circular_block_indices(d.size, block, reps, np.random.default_rng(seed))
    lo, hi = np.percentile(d[idx].mean(axis=1), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cliper_rmse(y: pd.Series, use: pd.Series, lag_hours: int) -> float:
    """RMSE of the optimal climatology-persistence blend, sigma * sqrt(1 - gamma^2) (L-13; Mayer
    and Yang 2024). sigma is the standard deviation of y on the `use` rows; gamma is the
    correlation of y(t) and y(t - lag) over the rows where both hours are `use` rows.

    The caller passes the evaluated daytime rows: only on the same sample is the blend optimal,
    so that it is never worse than persistence (MSE 2 sigma^2 (1 - gamma) >= sigma^2 (1 -
    gamma^2)). Only observations enter, no forecast. Train statistics were tried first and
    let plain persistence beat the reference on validation (docs/stage8b_plan.md §7)."""
    use = use.reindex(y.index, fill_value=False) & y.notna()
    lagged = y.shift(lag_hours, freq="h").reindex(y.index)
    both = use & use.shift(lag_hours, freq="h").reindex(y.index, fill_value=False) \
        & lagged.notna()
    if use.sum() < 2 or both.sum() < 2:
        raise ValueError("too few rows for the CLIPER reference")
    sigma = float(y[use].std(ddof=1))
    gamma = float(np.corrcoef(y[both], lagged[both])[0, 1])
    return sigma * float(np.sqrt(max(1.0 - gamma ** 2, 0.0)))
